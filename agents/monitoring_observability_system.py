#!/usr/bin/env python3
"""
Monitoring & Observability System for AgentAlina
Implements comprehensive monitoring, metrics collection, alerting,
and observability for agent operations and system performance.
"""

import asyncio
import logging
import time
import json
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"

@dataclass
class MetricPoint:
    """Represents a single metric data point."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'metric_type': self.metric_type.value
        }

@dataclass
class Alert:
    """Represents a system alert."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data

class MetricsCollector:
    """Collects and stores metrics data."""
    
    def __init__(self, max_points_per_metric: int = 10000):
        self.max_points_per_metric = max_points_per_metric
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.Lock()
        
        # Start system metrics collection
        self._start_system_metrics_collection()
    
    def record_metric(self, name: str, value: Union[int, float], 
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Dict[str, str] = None) -> None:
        """Record a metric point."""
        with self.lock:
            timestamp = datetime.now()
            tags = tags or {}
            
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=timestamp,
                tags=tags,
                metric_type=metric_type
            )
            
            # Store in general metrics collection
            self.metrics[name].append(metric_point)
            
            # Store in type-specific collections
            if metric_type == MetricType.COUNTER:
                self.counters[name] += value
            elif metric_type == MetricType.GAUGE:
                self.gauges[name] = value
            elif metric_type == MetricType.HISTOGRAM:
                self.histograms[name].append(value)
                # Keep only recent values for histograms
                if len(self.histograms[name]) > 1000:
                    self.histograms[name] = self.histograms[name][-1000:]
            elif metric_type == MetricType.TIMER:
                self.timers[name].append(value)
                # Keep only recent values for timers
                if len(self.timers[name]) > 1000:
                    self.timers[name] = self.timers[name][-1000:]
            elif metric_type == MetricType.RATE:
                self.rates[name].append((timestamp, value))
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Set a gauge metric."""
        self.record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record a histogram value."""
        self.record_metric(name, value, MetricType.HISTOGRAM, tags)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None) -> None:
        """Record a timer value."""
        self.record_metric(name, duration, MetricType.TIMER, tags)
    
    def timer(self, name: str, tags: Dict[str, str] = None):
        """Context manager for timing operations."""
        class TimerContext:
            def __init__(self, collector, metric_name, metric_tags):
                self.collector = collector
                self.metric_name = metric_name
                self.metric_tags = metric_tags
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.collector.record_timer(self.metric_name, duration, self.metric_tags)
        
        return TimerContext(self, name, tags)
    
    def get_metric_summary(self, name: str, time_window: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        with self.lock:
            if name not in self.metrics:
                return {'error': f'Metric {name} not found'}
            
            cutoff_time = datetime.now() - time_window
            recent_points = [p for p in self.metrics[name] if p.timestamp > cutoff_time]
            
            if not recent_points:
                return {'error': f'No recent data for metric {name}'}
            
            values = [p.value for p in recent_points]
            metric_type = recent_points[0].metric_type
            
            summary = {
                'name': name,
                'metric_type': metric_type.value,
                'count': len(values),
                'time_window_minutes': time_window.total_seconds() / 60,
                'latest_value': values[-1],
                'latest_timestamp': recent_points[-1].timestamp.isoformat()
            }
            
            if metric_type in [MetricType.GAUGE, MetricType.HISTOGRAM, MetricType.TIMER]:
                summary.update({
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values)
                })
                
                if len(values) > 1:
                    summary['std_dev'] = statistics.stdev(values)
                
                # Percentiles for histograms and timers
                if metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                    sorted_values = sorted(values)
                    summary.update({
                        'p50': self._percentile(sorted_values, 50),
                        'p90': self._percentile(sorted_values, 90),
                        'p95': self._percentile(sorted_values, 95),
                        'p99': self._percentile(sorted_values, 99)
                    })
            
            elif metric_type == MetricType.COUNTER:
                summary['total'] = sum(values)
                summary['rate_per_minute'] = len(values) / (time_window.total_seconds() / 60)
            
            elif metric_type == MetricType.RATE:
                if len(values) > 1:
                    time_diffs = [(recent_points[i].timestamp - recent_points[i-1].timestamp).total_seconds() 
                                 for i in range(1, len(recent_points))]
                    if time_diffs:
                        summary['average_rate'] = statistics.mean(values)
                        summary['average_interval'] = statistics.mean(time_diffs)
            
            return summary
    
    def _percentile(self, sorted_values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not sorted_values:
            return 0.0
        
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        
        if lower_index == upper_index:
            return sorted_values[lower_index]
        
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight
    
    def _start_system_metrics_collection(self):
        """Start collecting system metrics."""
        def collect_system_metrics():
            while True:
                try:
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.set_gauge('system.cpu.usage_percent', cpu_percent, {'component': 'system'})
                    
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.set_gauge('system.memory.usage_percent', memory.percent, {'component': 'system'})
                    self.set_gauge('system.memory.available_bytes', memory.available, {'component': 'system'})
                    self.set_gauge('system.memory.used_bytes', memory.used, {'component': 'system'})
                    
                    # Disk metrics
                    disk = psutil.disk_usage('/')
                    self.set_gauge('system.disk.usage_percent', (disk.used / disk.total) * 100, {'component': 'system'})
                    self.set_gauge('system.disk.free_bytes', disk.free, {'component': 'system'})
                    
                    # Network metrics
                    network = psutil.net_io_counters()
                    self.record_metric('system.network.bytes_sent', network.bytes_sent, MetricType.RATE, {'component': 'system'})
                    self.record_metric('system.network.bytes_recv', network.bytes_recv, MetricType.RATE, {'component': 'system'})
                    
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                
                time.sleep(30)  # Collect every 30 seconds
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        with self.lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histogram_counts': {name: len(values) for name, values in self.histograms.items()},
                'timer_counts': {name: len(values) for name, values in self.timers.items()},
                'rate_counts': {name: len(values) for name, values in self.rates.items()}
            }

class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_handlers: List[Callable] = []
        self.lock = threading.Lock()
        
        # Default alert rules
        self._setup_default_alert_rules()
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        self.alert_rules = [
            {
                'name': 'high_cpu_usage',
                'condition': lambda metrics: metrics.get('gauges', {}).get('system.cpu.usage_percent', 0) > 80,
                'severity': AlertSeverity.WARNING,
                'description': 'CPU usage is above 80%'
            },
            {
                'name': 'high_memory_usage',
                'condition': lambda metrics: metrics.get('gauges', {}).get('system.memory.usage_percent', 0) > 85,
                'severity': AlertSeverity.WARNING,
                'description': 'Memory usage is above 85%'
            },
            {
                'name': 'critical_memory_usage',
                'condition': lambda metrics: metrics.get('gauges', {}).get('system.memory.usage_percent', 0) > 95,
                'severity': AlertSeverity.CRITICAL,
                'description': 'Memory usage is critically high (>95%)'
            },
            {
                'name': 'high_disk_usage',
                'condition': lambda metrics: metrics.get('gauges', {}).get('system.disk.usage_percent', 0) > 90,
                'severity': AlertSeverity.ERROR,
                'description': 'Disk usage is above 90%'
            }
        ]
    
    def add_alert_rule(self, name: str, condition: Callable, severity: AlertSeverity, description: str):
        """Add a custom alert rule."""
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'severity': severity,
            'description': description
        })
    
    def add_notification_handler(self, handler: Callable):
        """Add a notification handler for alerts."""
        self.notification_handlers.append(handler)
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics."""
        with self.lock:
            for rule in self.alert_rules:
                try:
                    if rule['condition'](metrics):
                        self._trigger_alert(rule)
                    else:
                        self._resolve_alert(rule['name'])
                except Exception as e:
                    logger.error(f"Error checking alert rule {rule['name']}: {e}")
    
    def _trigger_alert(self, rule: Dict[str, Any]):
        """Trigger an alert."""
        alert_name = rule['name']
        
        if alert_name in self.alerts and self.alerts[alert_name].status == AlertStatus.ACTIVE:
            # Alert already active, just update timestamp
            self.alerts[alert_name].updated_at = datetime.now()
            return
        
        # Create new alert
        alert = Alert(
            id=f"alert_{int(time.time())}_{alert_name}",
            name=alert_name,
            description=rule['description'],
            severity=rule['severity'],
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.alerts[alert_name] = alert
        
        # Send notifications
        self._send_notifications(alert)
        
        logger.warning(f"Alert triggered: {alert.name} - {alert.description}")
    
    def _resolve_alert(self, alert_name: str):
        """Resolve an alert."""
        if alert_name in self.alerts and self.alerts[alert_name].status == AlertStatus.ACTIVE:
            self.alerts[alert_name].status = AlertStatus.RESOLVED
            self.alerts[alert_name].resolved_at = datetime.now()
            self.alerts[alert_name].updated_at = datetime.now()
            
            logger.info(f"Alert resolved: {alert_name}")
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self.lock:
            return [alert for alert in self.alerts.values() if alert.status == AlertStatus.ACTIVE]
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        with self.lock:
            return [alert for alert in self.alerts.values() if alert.created_at > cutoff_time]
    
    def acknowledge_alert(self, alert_name: str) -> bool:
        """Acknowledge an alert."""
        with self.lock:
            if alert_name in self.alerts:
                self.alerts[alert_name].status = AlertStatus.ACKNOWLEDGED
                self.alerts[alert_name].updated_at = datetime.now()
                return True
            return False

class PerformanceTracker:
    """Tracks performance metrics for operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0.0,
            'min_duration': float('inf'),
            'max_duration': 0.0,
            'error_count': 0,
            'last_execution': None
        })
        self.lock = threading.Lock()
    
    def track_operation(self, operation_name: str):
        """Context manager for tracking operation performance."""
        class OperationTracker:
            def __init__(self, tracker, op_name):
                self.tracker = tracker
                self.op_name = op_name
                self.start_time = None
                self.success = True
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.success = exc_type is None
                self.tracker._record_operation(self.op_name, duration, self.success)
            
            def mark_error(self):
                self.success = False
        
        return OperationTracker(self, operation_name)
    
    def _record_operation(self, operation_name: str, duration: float, success: bool):
        """Record operation performance data."""
        with self.lock:
            stats = self.operation_stats[operation_name]
            
            stats['count'] += 1
            stats['total_duration'] += duration
            stats['min_duration'] = min(stats['min_duration'], duration)
            stats['max_duration'] = max(stats['max_duration'], duration)
            stats['last_execution'] = datetime.now()
            
            if not success:
                stats['error_count'] += 1
            
            # Record metrics
            self.metrics_collector.record_timer(f'operation.{operation_name}.duration', duration, 
                                               {'operation': operation_name})
            self.metrics_collector.increment_counter(f'operation.{operation_name}.count', 1, 
                                                    {'operation': operation_name, 'success': str(success)})
            
            if not success:
                self.metrics_collector.increment_counter(f'operation.{operation_name}.errors', 1, 
                                                        {'operation': operation_name})
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get performance statistics for an operation."""
        with self.lock:
            if operation_name not in self.operation_stats:
                return {'error': f'No stats for operation {operation_name}'}
            
            stats = self.operation_stats[operation_name].copy()
            
            if stats['count'] > 0:
                stats['average_duration'] = stats['total_duration'] / stats['count']
                stats['error_rate'] = stats['error_count'] / stats['count']
                stats['success_rate'] = 1 - stats['error_rate']
            
            if stats['last_execution']:
                stats['last_execution'] = stats['last_execution'].isoformat()
            
            return stats
    
    def get_all_operation_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all operations."""
        with self.lock:
            return {op_name: self.get_operation_stats(op_name) 
                   for op_name in self.operation_stats.keys()}

class MonitoringObservabilitySystem:
    """Main monitoring and observability system."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.performance_tracker = PerformanceTracker(self.metrics_collector)
        
        # Setup default notification handler
        self.alert_manager.add_notification_handler(self._default_alert_handler)
        
        # Start monitoring loop
        self._monitoring_task = None
        self._start_monitoring_loop()
    
    def _default_alert_handler(self, alert: Alert):
        """Default alert notification handler."""
        logger.warning(f"ALERT: {alert.name} ({alert.severity.value}) - {alert.description}")
    
    def _start_monitoring_loop(self):
        """Start the main monitoring loop."""
        async def monitoring_loop():
            while True:
                try:
                    # Check alerts based on current metrics
                    current_metrics = self.metrics_collector.get_all_metrics()
                    self.alert_manager.check_alerts(current_metrics)
                    
                    # Record monitoring system health
                    self.metrics_collector.set_gauge('monitoring.alerts.active', 
                                                    len(self.alert_manager.get_active_alerts()))
                    
                    await asyncio.sleep(60)  # Check every minute
                
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(60)
        
        self._monitoring_task = asyncio.create_task(monitoring_loop())
    
    def record_agent_operation(self, agent_name: str, operation: str, 
                              duration: float, success: bool, metadata: Dict[str, Any] = None):
        """Record agent operation metrics."""
        tags = {'agent': agent_name, 'operation': operation, 'success': str(success)}
        if metadata:
            tags.update(metadata)
        
        self.metrics_collector.record_timer(f'agent.{agent_name}.{operation}.duration', duration, tags)
        self.metrics_collector.increment_counter(f'agent.{agent_name}.{operation}.count', 1, tags)
        
        if not success:
            self.metrics_collector.increment_counter(f'agent.{agent_name}.{operation}.errors', 1, tags)
    
    def record_llm_operation(self, model: str, tokens_used: int, cost: float, 
                           latency: float, success: bool):
        """Record LLM operation metrics."""
        tags = {'model': model, 'success': str(success)}
        
        self.metrics_collector.record_histogram('llm.tokens_used', tokens_used, tags)
        self.metrics_collector.record_histogram('llm.cost', cost, tags)
        self.metrics_collector.record_timer('llm.latency', latency, tags)
        self.metrics_collector.increment_counter('llm.requests', 1, tags)
        
        if not success:
            self.metrics_collector.increment_counter('llm.errors', 1, tags)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {
                'cpu_usage': self.metrics_collector.get_metric_summary('system.cpu.usage_percent'),
                'memory_usage': self.metrics_collector.get_metric_summary('system.memory.usage_percent'),
                'disk_usage': self.metrics_collector.get_metric_summary('system.disk.usage_percent')
            },
            'alerts': {
                'active': [alert.to_dict() for alert in self.alert_manager.get_active_alerts()],
                'recent': [alert.to_dict() for alert in self.alert_manager.get_alert_history(24)]
            },
            'performance': self.performance_tracker.get_all_operation_stats(),
            'metrics_summary': self.metrics_collector.get_all_metrics()
        }
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get system health check status."""
        active_alerts = self.alert_manager.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        
        if critical_alerts:
            status = 'critical'
        elif active_alerts:
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'active_alerts': len(active_alerts),
            'critical_alerts': len(critical_alerts),
            'uptime_seconds': time.time() - self._start_time if hasattr(self, '_start_time') else 0
        }
    
    async def shutdown(self):
        """Gracefully shutdown the monitoring system."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring system shutdown complete")

if __name__ == "__main__":
    # Example usage
    async def example_usage():
        # Initialize monitoring system
        monitoring = MonitoringObservabilitySystem()
        monitoring._start_time = time.time()
        
        # Simulate some operations
        with monitoring.performance_tracker.track_operation('test_operation'):
            await asyncio.sleep(0.1)  # Simulate work
        
        # Record some custom metrics
        monitoring.metrics_collector.increment_counter('custom.events', 1, {'type': 'test'})
        monitoring.metrics_collector.set_gauge('custom.queue_size', 42)
        
        # Record LLM operation
        monitoring.record_llm_operation('gpt-4', 150, 0.003, 1.2, True)
        
        # Record agent operation
        monitoring.record_agent_operation('knowledge_agent', 'query', 0.5, True, {'query_type': 'search'})
        
        # Wait a bit for metrics to be collected
        await asyncio.sleep(2)
        
        # Get dashboard data
        dashboard = monitoring.get_dashboard_data()
        print("Dashboard data:", json.dumps(dashboard, indent=2, default=str))
        
        # Get health check
        health = monitoring.get_health_check()
        print("Health check:", json.dumps(health, indent=2))
        
        # Cleanup
        await monitoring.shutdown()
    
    # Run example
    asyncio.run(example_usage())