#!/usr/bin/env python3
"""
Error Handling & Retry System for AgentAlina
Implements comprehensive error handling, retry strategies, and fallback mechanisms
for robust agent operations and prompt processing.
"""

import asyncio
import logging
import time
import random
import traceback
from typing import Dict, Any, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    JITTERED_BACKOFF = "jittered_backoff"
    CUSTOM = "custom"

class FallbackType(Enum):
    """Types of fallback mechanisms."""
    ALTERNATIVE_PROMPT = "alternative_prompt"
    SIMPLIFIED_PROMPT = "simplified_prompt"
    CACHED_RESPONSE = "cached_response"
    DEFAULT_RESPONSE = "default_response"
    HUMAN_ESCALATION = "human_escalation"
    ALTERNATIVE_MODEL = "alternative_model"

@dataclass
class ErrorContext:
    """Context information for errors."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    stack_trace: str
    operation: str
    input_data: Dict[str, Any]
    attempt_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp.isoformat(),
            'error_type': self.error_type,
            'error_message': self.error_message,
            'severity': self.severity.value,
            'stack_trace': self.stack_trace,
            'operation': self.operation,
            'input_data': self.input_data,
            'attempt_number': self.attempt_number,
            'metadata': self.metadata
        }

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_on_exceptions: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    stop_on_exceptions: List[Type[Exception]] = field(default_factory=list)
    custom_delay_func: Optional[Callable[[int], float]] = None

@dataclass
class FallbackConfig:
    """Configuration for fallback mechanisms."""
    enabled: bool = True
    fallback_type: FallbackType = FallbackType.SIMPLIFIED_PROMPT
    fallback_prompts: List[str] = field(default_factory=list)
    default_response: str = "I apologize, but I'm experiencing technical difficulties. Please try again later."
    cache_fallback_responses: bool = True
    escalation_threshold: int = 3
    alternative_models: List[str] = field(default_factory=list)

class ErrorTracker:
    """Tracks and analyzes error patterns."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.error_history: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = {}
        self.error_patterns: Dict[str, List[datetime]] = {}
        self.metrics = {
            'total_errors': 0,
            'errors_by_severity': {severity.value: 0 for severity in ErrorSeverity},
            'errors_by_type': {},
            'average_resolution_time': 0.0,
            'recurring_errors': 0
        }
    
    def record_error(self, error_context: ErrorContext):
        """Record an error occurrence."""
        self.error_history.append(error_context)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Update counts and patterns
        error_key = f"{error_context.error_type}:{error_context.operation}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        if error_key not in self.error_patterns:
            self.error_patterns[error_key] = []
        self.error_patterns[error_key].append(error_context.timestamp)
        
        # Update metrics
        self.metrics['total_errors'] += 1
        self.metrics['errors_by_severity'][error_context.severity.value] += 1
        self.metrics['errors_by_type'][error_context.error_type] = \
            self.metrics['errors_by_type'].get(error_context.error_type, 0) + 1
        
        # Check for recurring patterns
        if self.error_counts[error_key] > 1:
            self.metrics['recurring_errors'] += 1
        
        logger.warning(f"Error recorded: {error_context.error_id} - {error_context.error_message}")
    
    def get_error_frequency(self, error_type: str, operation: str, 
                           time_window: timedelta = timedelta(hours=1)) -> int:
        """Get error frequency for a specific error type and operation."""
        error_key = f"{error_type}:{operation}"
        if error_key not in self.error_patterns:
            return 0
        
        cutoff_time = datetime.now() - time_window
        recent_errors = [ts for ts in self.error_patterns[error_key] if ts > cutoff_time]
        return len(recent_errors)
    
    def is_recurring_error(self, error_type: str, operation: str, 
                          threshold: int = 3, time_window: timedelta = timedelta(minutes=30)) -> bool:
        """Check if an error is recurring within a time window."""
        frequency = self.get_error_frequency(error_type, operation, time_window)
        return frequency >= threshold
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get error tracking metrics."""
        return self.metrics.copy()
    
    def get_recent_errors(self, count: int = 10) -> List[ErrorContext]:
        """Get most recent errors."""
        return self.error_history[-count:] if self.error_history else []

class RetryManager:
    """Manages retry logic with different strategies."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.metrics = {
            'total_retries': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'average_retry_delay': 0.0
        }
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        if self.config.custom_delay_func:
            return self.config.custom_delay_func(attempt)
        
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
        
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        
        elif self.config.strategy == RetryStrategy.JITTERED_BACKOFF:
            base_delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
            jitter = random.uniform(0.1, 0.9) if self.config.jitter else 1.0
            delay = base_delay * jitter
        
        else:
            delay = self.config.base_delay
        
        # Apply max delay limit
        delay = min(delay, self.config.max_delay)
        
        return delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.config.max_attempts:
            return False
        
        # Check stop conditions
        for stop_exception in self.config.stop_on_exceptions:
            if isinstance(exception, stop_exception):
                return False
        
        # Check retry conditions
        for retry_exception in self.config.retry_on_exceptions:
            if isinstance(exception, retry_exception):
                return True
        
        return False
    
    async def execute_with_retry(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                if attempt > 1:
                    self.metrics['successful_retries'] += 1
                    logger.info(f"Operation succeeded on attempt {attempt}")
                
                return result
            
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt):
                    self.metrics['failed_retries'] += 1
                    logger.error(f"Operation failed permanently after {attempt} attempts: {e}")
                    raise e
                
                if attempt < self.config.max_attempts:
                    delay = self.calculate_delay(attempt)
                    self.metrics['total_retries'] += 1
                    
                    # Update average delay metric
                    total_retries = self.metrics['total_retries']
                    current_avg = self.metrics['average_retry_delay']
                    self.metrics['average_retry_delay'] = \
                        ((current_avg * (total_retries - 1)) + delay) / total_retries
                    
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
        
        # If we get here, all attempts failed
        self.metrics['failed_retries'] += 1
        raise last_exception
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retry manager metrics."""
        return self.metrics.copy()

class FallbackManager:
    """Manages fallback mechanisms when operations fail."""
    
    def __init__(self, config: FallbackConfig):
        self.config = config
        self.fallback_cache: Dict[str, Any] = {}
        self.escalation_count: Dict[str, int] = {}
        self.metrics = {
            'fallbacks_triggered': 0,
            'fallbacks_by_type': {fb_type.value: 0 for fb_type in FallbackType},
            'cache_hits': 0,
            'escalations': 0
        }
    
    async def execute_fallback(self, operation: str, input_data: Dict[str, Any], 
                              error_context: ErrorContext) -> Dict[str, Any]:
        """Execute appropriate fallback mechanism."""
        if not self.config.enabled:
            raise RuntimeError("Fallback disabled and primary operation failed")
        
        self.metrics['fallbacks_triggered'] += 1
        fallback_type = self._determine_fallback_type(error_context)
        self.metrics['fallbacks_by_type'][fallback_type.value] += 1
        
        logger.info(f"Executing fallback: {fallback_type.value} for operation: {operation}")
        
        if fallback_type == FallbackType.CACHED_RESPONSE:
            return await self._get_cached_response(operation, input_data)
        
        elif fallback_type == FallbackType.ALTERNATIVE_PROMPT:
            return await self._use_alternative_prompt(operation, input_data)
        
        elif fallback_type == FallbackType.SIMPLIFIED_PROMPT:
            return await self._use_simplified_prompt(operation, input_data)
        
        elif fallback_type == FallbackType.DEFAULT_RESPONSE:
            return await self._get_default_response(operation, input_data)
        
        elif fallback_type == FallbackType.HUMAN_ESCALATION:
            return await self._escalate_to_human(operation, input_data, error_context)
        
        elif fallback_type == FallbackType.ALTERNATIVE_MODEL:
            return await self._use_alternative_model(operation, input_data)
        
        else:
            return await self._get_default_response(operation, input_data)
    
    def _determine_fallback_type(self, error_context: ErrorContext) -> FallbackType:
        """Determine the most appropriate fallback type."""
        # Check escalation threshold
        operation_key = f"{error_context.operation}:{error_context.error_type}"
        escalation_count = self.escalation_count.get(operation_key, 0)
        
        if escalation_count >= self.config.escalation_threshold:
            return FallbackType.HUMAN_ESCALATION
        
        # Check error severity
        if error_context.severity == ErrorSeverity.CRITICAL:
            return FallbackType.HUMAN_ESCALATION
        
        # Check for specific error types
        if "timeout" in error_context.error_message.lower():
            return FallbackType.CACHED_RESPONSE
        
        if "model" in error_context.error_message.lower() and self.config.alternative_models:
            return FallbackType.ALTERNATIVE_MODEL
        
        # Default to configured fallback type
        return self.config.fallback_type
    
    async def _get_cached_response(self, operation: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get cached response if available."""
        cache_key = self._generate_cache_key(operation, input_data)
        
        if cache_key in self.fallback_cache:
            self.metrics['cache_hits'] += 1
            logger.info(f"Using cached fallback response for {operation}")
            return {
                'status': 'success',
                'result': self.fallback_cache[cache_key],
                'source': 'cache',
                'fallback_used': True
            }
        
        # No cache available, use default response
        return await self._get_default_response(operation, input_data)
    
    async def _use_alternative_prompt(self, operation: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use alternative prompt from configuration."""
        if not self.config.fallback_prompts:
            return await self._get_default_response(operation, input_data)
        
        # Select random alternative prompt
        alternative_prompt = random.choice(self.config.fallback_prompts)
        
        return {
            'status': 'success',
            'result': f"Using alternative approach: {alternative_prompt}",
            'source': 'alternative_prompt',
            'fallback_used': True,
            'prompt_used': alternative_prompt
        }
    
    async def _use_simplified_prompt(self, operation: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use simplified version of the prompt."""
        simplified_response = "I'll provide a simplified response based on the available information."
        
        # Try to extract key information from input
        if 'query' in input_data:
            simplified_response += f" Regarding: {input_data['query'][:100]}..."
        
        return {
            'status': 'success',
            'result': simplified_response,
            'source': 'simplified_prompt',
            'fallback_used': True
        }
    
    async def _get_default_response(self, operation: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get default fallback response."""
        return {
            'status': 'success',
            'result': self.config.default_response,
            'source': 'default_response',
            'fallback_used': True
        }
    
    async def _escalate_to_human(self, operation: str, input_data: Dict[str, Any], 
                                error_context: ErrorContext) -> Dict[str, Any]:
        """Escalate to human intervention."""
        operation_key = f"{operation}:{error_context.error_type}"
        self.escalation_count[operation_key] = self.escalation_count.get(operation_key, 0) + 1
        self.metrics['escalations'] += 1
        
        escalation_data = {
            'operation': operation,
            'input_data': input_data,
            'error_context': error_context.to_dict(),
            'escalation_time': datetime.now().isoformat(),
            'escalation_id': f"esc_{int(time.time())}"
        }
        
        logger.critical(f"Escalating to human: {escalation_data['escalation_id']}")
        
        return {
            'status': 'escalated',
            'result': 'This request has been escalated to human support.',
            'source': 'human_escalation',
            'fallback_used': True,
            'escalation_data': escalation_data
        }
    
    async def _use_alternative_model(self, operation: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use alternative model if available."""
        if not self.config.alternative_models:
            return await self._get_default_response(operation, input_data)
        
        alternative_model = random.choice(self.config.alternative_models)
        
        return {
            'status': 'success',
            'result': f"Response generated using alternative model: {alternative_model}",
            'source': 'alternative_model',
            'fallback_used': True,
            'model_used': alternative_model
        }
    
    def _generate_cache_key(self, operation: str, input_data: Dict[str, Any]) -> str:
        """Generate cache key for input data."""
        # Create a hash of the operation and input data
        import hashlib
        data_str = json.dumps({'operation': operation, 'input': input_data}, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def cache_response(self, operation: str, input_data: Dict[str, Any], response: Any):
        """Cache a successful response for future fallback use."""
        if not self.config.cache_fallback_responses:
            return
        
        cache_key = self._generate_cache_key(operation, input_data)
        self.fallback_cache[cache_key] = response
        logger.debug(f"Cached response for operation: {operation}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get fallback manager metrics."""
        return self.metrics.copy()

class ErrorHandlingRetrySystem:
    """Main error handling and retry system."""
    
    def __init__(self, retry_config: RetryConfig = None, fallback_config: FallbackConfig = None):
        self.error_tracker = ErrorTracker()
        self.retry_manager = RetryManager(retry_config or RetryConfig())
        self.fallback_manager = FallbackManager(fallback_config or FallbackConfig())
        self.metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'operations_with_fallback': 0
        }
    
    def resilient_operation(self, operation_name: str = None):
        """Decorator for making operations resilient with retry and fallback."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                op_name = operation_name or func.__name__
                return await self.execute_resilient_operation(op_name, func, *args, **kwargs)
            return wrapper
        return decorator
    
    async def execute_resilient_operation(self, operation_name: str, operation: Callable, 
                                        *args, **kwargs) -> Dict[str, Any]:
        """Execute operation with full error handling, retry, and fallback support."""
        self.metrics['total_operations'] += 1
        start_time = time.time()
        
        try:
            # Try operation with retry logic
            result = await self.retry_manager.execute_with_retry(operation, *args, **kwargs)
            
            # Cache successful result for potential fallback use
            input_data = {'args': args, 'kwargs': kwargs}
            self.fallback_manager.cache_response(operation_name, input_data, result)
            
            self.metrics['successful_operations'] += 1
            
            return {
                'status': 'success',
                'result': result,
                'operation': operation_name,
                'execution_time': time.time() - start_time,
                'fallback_used': False
            }
        
        except Exception as e:
            # Create error context
            error_context = ErrorContext(
                error_id=f"err_{int(time.time())}_{random.randint(1000, 9999)}",
                timestamp=datetime.now(),
                error_type=type(e).__name__,
                error_message=str(e),
                severity=self._determine_error_severity(e),
                stack_trace=traceback.format_exc(),
                operation=operation_name,
                input_data={'args': str(args), 'kwargs': str(kwargs)},
                attempt_number=self.retry_manager.config.max_attempts
            )
            
            # Record error
            self.error_tracker.record_error(error_context)
            
            # Try fallback
            try:
                input_data = {'args': args, 'kwargs': kwargs}
                fallback_result = await self.fallback_manager.execute_fallback(
                    operation_name, input_data, error_context
                )
                
                self.metrics['operations_with_fallback'] += 1
                
                return {
                    'status': 'success_with_fallback',
                    'result': fallback_result['result'],
                    'operation': operation_name,
                    'execution_time': time.time() - start_time,
                    'fallback_used': True,
                    'fallback_type': fallback_result['source'],
                    'original_error': error_context.to_dict()
                }
            
            except Exception as fallback_error:
                # Both operation and fallback failed
                self.metrics['failed_operations'] += 1
                
                logger.critical(f"Operation {operation_name} failed completely: {e}, Fallback also failed: {fallback_error}")
                
                return {
                    'status': 'failed',
                    'error': str(e),
                    'fallback_error': str(fallback_error),
                    'operation': operation_name,
                    'execution_time': time.time() - start_time,
                    'fallback_used': False,
                    'error_context': error_context.to_dict()
                }
    
    def _determine_error_severity(self, exception: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type."""
        if isinstance(exception, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        elif isinstance(exception, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        elif isinstance(exception, (ValueError, TypeError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all components."""
        return {
            'system': self.metrics.copy(),
            'error_tracker': self.error_tracker.get_metrics(),
            'retry_manager': self.retry_manager.get_metrics(),
            'fallback_manager': self.fallback_manager.get_metrics()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        total_ops = self.metrics['total_operations']
        if total_ops == 0:
            return {'status': 'unknown', 'reason': 'no operations recorded'}
        
        success_rate = (self.metrics['successful_operations'] + 
                       self.metrics['operations_with_fallback']) / total_ops
        
        if success_rate >= 0.95:
            status = 'healthy'
        elif success_rate >= 0.80:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return {
            'status': status,
            'success_rate': success_rate,
            'total_operations': total_ops,
            'recent_errors': len(self.error_tracker.get_recent_errors(10))
        }

if __name__ == "__main__":
    # Example usage
    async def example_usage():
        # Configure retry and fallback
        retry_config = RetryConfig(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=1.0,
            max_delay=10.0
        )
        
        fallback_config = FallbackConfig(
            fallback_type=FallbackType.ALTERNATIVE_PROMPT,
            fallback_prompts=[
                "Let me try a different approach to help you.",
                "I'll provide a simplified response based on available information."
            ],
            alternative_models=["gpt-3.5-turbo", "claude-instant"]
        )
        
        # Initialize system
        error_system = ErrorHandlingRetrySystem(retry_config, fallback_config)
        
        # Example resilient operation
        @error_system.resilient_operation("test_operation")
        async def test_operation(should_fail: bool = False):
            if should_fail:
                raise ValueError("Simulated failure")
            return "Operation successful"
        
        # Test successful operation
        result1 = await test_operation(False)
        print("Success result:", json.dumps(result1, indent=2))
        
        # Test failed operation with fallback
        result2 = await test_operation(True)
        print("Fallback result:", json.dumps(result2, indent=2))
        
        # Get system metrics
        metrics = error_system.get_comprehensive_metrics()
        print("System metrics:", json.dumps(metrics, indent=2))
        
        # Get health status
        health = error_system.get_health_status()
        print("Health status:", json.dumps(health, indent=2))
    
    # Run example
    asyncio.run(example_usage())