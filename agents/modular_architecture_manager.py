#!/usr/bin/env python3
"""
Modular Architecture Manager for AgentAlina
Implements separation of concerns with distinct modules for prompt templates,
chain logic, and memory adapters to enhance maintainability.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    """Data class for prompt templates."""
    name: str
    template: str
    variables: List[str]
    version: str
    created_at: datetime
    metadata: Dict[str, Any]

class PromptTemplateManager:
    """Manages prompt templates with versioning and validation."""
    
    def __init__(self, templates_dir: str = "config/prompt_templates"):
        self.templates_dir = templates_dir
        self.templates: Dict[str, PromptTemplate] = {}
        self._ensure_templates_dir()
        self._load_templates()
    
    def _ensure_templates_dir(self):
        """Ensure templates directory exists."""
        os.makedirs(self.templates_dir, exist_ok=True)
    
    def _load_templates(self):
        """Load all templates from the templates directory."""
        if not os.path.exists(self.templates_dir):
            return
        
        for filename in os.listdir(self.templates_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.templates_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)
                        template = PromptTemplate(
                            name=template_data['name'],
                            template=template_data['template'],
                            variables=template_data['variables'],
                            version=template_data['version'],
                            created_at=datetime.fromisoformat(template_data['created_at']),
                            metadata=template_data.get('metadata', {})
                        )
                        self.templates[template.name] = template
                        logger.info(f"Loaded template: {template.name} v{template.version}")
                except Exception as e:
                    logger.error(f"Error loading template {filename}: {e}")
    
    def create_template(self, name: str, template: str, variables: List[str], 
                       metadata: Dict[str, Any] = None) -> PromptTemplate:
        """Create a new prompt template."""
        version = "1.0.0"
        if name in self.templates:
            # Increment version
            current_version = self.templates[name].version
            major, minor, patch = map(int, current_version.split('.'))
            version = f"{major}.{minor}.{patch + 1}"
        
        prompt_template = PromptTemplate(
            name=name,
            template=template,
            variables=variables,
            version=version,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        self.templates[name] = prompt_template
        self._save_template(prompt_template)
        logger.info(f"Created template: {name} v{version}")
        return prompt_template
    
    def _save_template(self, template: PromptTemplate):
        """Save template to file."""
        filepath = os.path.join(self.templates_dir, f"{template.name}.json")
        template_data = {
            'name': template.name,
            'template': template.template,
            'variables': template.variables,
            'version': template.version,
            'created_at': template.created_at.isoformat(),
            'metadata': template.metadata
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2, ensure_ascii=False)
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def render_template(self, name: str, **kwargs) -> str:
        """Render a template with provided variables."""
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        
        # Validate required variables
        missing_vars = set(template.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        return template.template.format(**kwargs)

class ChainLogic(ABC):
    """Abstract base class for chain logic components."""
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the chain logic."""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data."""
        pass

class DataProcessingChain(ChainLogic):
    """Chain logic for data processing workflows."""
    
    def __init__(self, steps: List[str]):
        self.steps = steps
        self.metrics = {'executions': 0, 'errors': 0, 'avg_duration': 0.0}
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data processing chain."""
        start_time = datetime.now()
        
        try:
            result = input_data.copy()
            
            for step in self.steps:
                logger.info(f"Executing step: {step}")
                result = await self._execute_step(step, result)
            
            self.metrics['executions'] += 1
            duration = (datetime.now() - start_time).total_seconds()
            self._update_avg_duration(duration)
            
            return {
                'status': 'success',
                'result': result,
                'duration': duration,
                'steps_executed': len(self.steps)
            }
        
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error in data processing chain: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'duration': (datetime.now() - start_time).total_seconds()
            }
    
    async def _execute_step(self, step: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step in the chain."""
        # Placeholder for actual step execution logic
        # In a real implementation, this would route to specific processors
        if step == 'validate':
            return self._validate_data(data)
        elif step == 'transform':
            return self._transform_data(data)
        elif step == 'enrich':
            return self._enrich_data(data)
        else:
            return data
    
    def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data step."""
        data['validated'] = True
        data['validation_timestamp'] = datetime.now().isoformat()
        return data
    
    def _transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data step."""
        data['transformed'] = True
        data['transform_timestamp'] = datetime.now().isoformat()
        return data
    
    def _enrich_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich data step."""
        data['enriched'] = True
        data['enrich_timestamp'] = datetime.now().isoformat()
        return data
    
    def _update_avg_duration(self, duration: float):
        """Update average duration metric."""
        total_executions = self.metrics['executions']
        current_avg = self.metrics['avg_duration']
        self.metrics['avg_duration'] = ((current_avg * (total_executions - 1)) + duration) / total_executions
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for data processing chain."""
        required_fields = ['data', 'source']
        return all(field in input_data for field in required_fields)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get chain execution metrics."""
        return self.metrics.copy()

class MemoryAdapter(ABC):
    """Abstract base class for memory adapters."""
    
    @abstractmethod
    async def store(self, key: str, value: Any) -> bool:
        """Store data in memory."""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from memory."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data from memory."""
        pass

class RedisMemoryAdapter(MemoryAdapter):
    """Redis implementation of memory adapter."""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.metrics = {'stores': 0, 'retrievals': 0, 'deletions': 0, 'errors': 0}
    
    async def store(self, key: str, value: Any) -> bool:
        """Store data in Redis."""
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            self.redis_client.set(key, value)
            self.metrics['stores'] += 1
            logger.debug(f"Stored data with key: {key}")
            return True
        
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error storing data with key {key}: {e}")
            return False
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from Redis."""
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
            
            # Try to parse as JSON, fallback to string
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
            
            self.metrics['retrievals'] += 1
            logger.debug(f"Retrieved data with key: {key}")
            return value
        
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error retrieving data with key {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete data from Redis."""
        try:
            result = self.redis_client.delete(key)
            self.metrics['deletions'] += 1
            logger.debug(f"Deleted data with key: {key}")
            return result > 0
        
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error deleting data with key {key}: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get memory adapter metrics."""
        return self.metrics.copy()

class ModularArchitectureManager:
    """Main manager for modular architecture components."""
    
    def __init__(self, redis_client=None):
        self.prompt_manager = PromptTemplateManager()
        self.chains: Dict[str, ChainLogic] = {}
        self.memory_adapter = RedisMemoryAdapter(redis_client) if redis_client else None
        self.metrics = {'total_operations': 0, 'successful_operations': 0}
        
        # Initialize default chains
        self._initialize_default_chains()
        
        # Initialize default templates
        self._initialize_default_templates()
    
    def _initialize_default_chains(self):
        """Initialize default chain logic components."""
        self.chains['data_processing'] = DataProcessingChain(['validate', 'transform', 'enrich'])
        logger.info("Initialized default chains")
    
    def _initialize_default_templates(self):
        """Initialize default prompt templates."""
        default_templates = [
            {
                'name': 'data_analysis',
                'template': 'Analyze the following data: {data}\nFocus on: {focus_areas}\nProvide insights on: {insights_requested}',
                'variables': ['data', 'focus_areas', 'insights_requested']
            },
            {
                'name': 'knowledge_extraction',
                'template': 'Extract knowledge from: {source}\nTarget entities: {entities}\nRelationships to identify: {relationships}',
                'variables': ['source', 'entities', 'relationships']
            }
        ]
        
        for template_data in default_templates:
            if not self.prompt_manager.get_template(template_data['name']):
                self.prompt_manager.create_template(**template_data)
    
    def register_chain(self, name: str, chain: ChainLogic):
        """Register a new chain logic component."""
        self.chains[name] = chain
        logger.info(f"Registered chain: {name}")
    
    def get_chain(self, name: str) -> Optional[ChainLogic]:
        """Get a chain by name."""
        return self.chains.get(name)
    
    async def execute_workflow(self, chain_name: str, template_name: str, 
                              input_data: Dict[str, Any], template_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete workflow using chain logic and prompt templates."""
        try:
            self.metrics['total_operations'] += 1
            
            # Get chain
            chain = self.get_chain(chain_name)
            if not chain:
                raise ValueError(f"Chain '{chain_name}' not found")
            
            # Validate input
            if not chain.validate_input(input_data):
                raise ValueError("Invalid input data for chain")
            
            # Render prompt template
            prompt = self.prompt_manager.render_template(template_name, **template_vars)
            
            # Add prompt to input data
            input_data['prompt'] = prompt
            
            # Execute chain
            result = await chain.execute(input_data)
            
            # Store result in memory if adapter is available
            if self.memory_adapter:
                workflow_key = f"workflow:{chain_name}:{template_name}:{datetime.now().isoformat()}"
                await self.memory_adapter.store(workflow_key, result)
            
            self.metrics['successful_operations'] += 1
            return result
        
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'chain_name': chain_name,
                'template_name': template_name
            }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        metrics = {
            'architecture_manager': self.metrics.copy(),
            'chains': {},
            'memory_adapter': None
        }
        
        # Get chain metrics
        for name, chain in self.chains.items():
            if hasattr(chain, 'get_metrics'):
                metrics['chains'][name] = chain.get_metrics()
        
        # Get memory adapter metrics
        if self.memory_adapter and hasattr(self.memory_adapter, 'get_metrics'):
            metrics['memory_adapter'] = self.memory_adapter.get_metrics()
        
        return metrics

if __name__ == "__main__":
    # Example usage
    import redis
    
    # Initialize Redis client (optional)
    try:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()
    except:
        redis_client = None
        logger.warning("Redis not available, memory adapter disabled")
    
    # Initialize modular architecture manager
    manager = ModularArchitectureManager(redis_client)
    
    # Example workflow execution
    async def example_workflow():
        input_data = {
            'data': {'sample': 'data', 'values': [1, 2, 3]},
            'source': 'test_source'
        }
        
        template_vars = {
            'data': 'Sample dataset with numerical values',
            'focus_areas': 'statistical patterns, outliers',
            'insights_requested': 'trends, correlations, recommendations'
        }
        
        result = await manager.execute_workflow(
            'data_processing',
            'data_analysis',
            input_data,
            template_vars
        )
        
        print("Workflow result:", json.dumps(result, indent=2))
        print("System metrics:", json.dumps(manager.get_system_metrics(), indent=2))
    
    # Run example
    import asyncio
    asyncio.run(example_workflow())