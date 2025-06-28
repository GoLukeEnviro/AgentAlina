#!/usr/bin/env python3
"""
Agent Improvement Engine

Selbstverbessernde Engine für kontinuierliche Agent-Optimierung.
Analysiert Performance-Metriken und passt Agent-Parameter automatisch an.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from config.env_config import get_neo4j_config
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance-Metrik für einen Agent."""
    agent_id: str
    timestamp: datetime
    response_time: float
    accuracy: float
    memory_usage: float
    cpu_usage: float
    success_rate: float
    user_satisfaction: float
    error_count: int
    task_completion_rate: float

@dataclass
class AgentConfiguration:
    """Konfiguration eines Agents."""
    agent_id: str
    learning_rate: float
    batch_size: int
    temperature: float
    max_tokens: int
    context_window: int
    memory_threshold: float
    optimization_strategy: str
    model_parameters: Dict[str, Any]

# Temporäre Anpassung aufgrund von API-Änderungen in agentops
# TODO: Aktualisieren Sie dies basierend auf der aktuellen agentops-Dokumentation (Version 0.4.16)
import agentops

class AgentImprovementEngine:
    def __init__(self):
        # Temporäre Anpassung aufgrund von API-Änderungen in agentops
        # TODO: Initialisieren Sie agentops basierend auf der aktuellen Dokumentation (Version 0.4.16)
        # self.ops = AgentOps()
        # self.ops.record_workflow_start()
        self.neo4j_config = get_neo4j_config()
        self.driver = None
        self.performance_history: List[PerformanceMetric] = []
        self.agent_configs: Dict[str, AgentConfiguration] = {}
        self.improvement_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.optimization_running = False
        self.coordination_history: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialisiert die Improvement Engine."""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=(self.neo4j_config['username'], self.neo4j_config['password'])
            )
            await self._load_historical_data()
            await self._load_agent_configurations()
            logger.info("Agent Improvement Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Agent Improvement Engine: {e}")
            raise
    
    async def close(self):
        """Schließt Verbindungen."""
        if self.driver:
            self.driver.close()
    
    async def record_performance(self, metric: PerformanceMetric):
        """Zeichnet eine Performance-Metrik auf.
        
        Args:
            metric: Performance-Metrik des Agents
        """
        try:
            self.performance_history.append(metric)
            await self._store_performance_metric(metric)
            
            # Trigger Verbesserung wenn genügend Daten vorhanden
            if len(self.performance_history) % 10 == 0:
                await self._trigger_improvement_analysis(metric.agent_id)
                
        except Exception as e:
            logger.error(f"Failed to record performance metric: {e}")
    
    async def _store_performance_metric(self, metric: PerformanceMetric):
        """Speichert Performance-Metrik in Neo4j."""
        query = """
        MERGE (a:Agent {id: $agent_id})
        CREATE (m:PerformanceMetric {
            timestamp: datetime($timestamp),
            response_time: $response_time,
            accuracy: $accuracy,
            memory_usage: $memory_usage,
            cpu_usage: $cpu_usage,
            success_rate: $success_rate,
            user_satisfaction: $user_satisfaction,
            error_count: $error_count,
            task_completion_rate: $task_completion_rate
        })
        CREATE (a)-[:HAS_METRIC]->(m)
        """
        
        with self.driver.session() as session:
            session.run(query, {
                'agent_id': metric.agent_id,
                'timestamp': metric.timestamp.isoformat(),
                'response_time': metric.response_time,
                'accuracy': metric.accuracy,
                'memory_usage': metric.memory_usage,
                'cpu_usage': metric.cpu_usage,
                'success_rate': metric.success_rate,
                'user_satisfaction': metric.user_satisfaction,
                'error_count': metric.error_count,
                'task_completion_rate': metric.task_completion_rate
            })
    
    async def _load_historical_data(self):
        """Lädt historische Performance-Daten."""
        query = """
        MATCH (a:Agent)-[:HAS_METRIC]->(m:PerformanceMetric)
        RETURN a.id as agent_id, m
        ORDER BY m.timestamp DESC
        LIMIT 1000
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                metric_data = record['m']
                metric = PerformanceMetric(
                    agent_id=record['agent_id'],
                    timestamp=datetime.fromisoformat(metric_data['timestamp']),
                    response_time=metric_data['response_time'],
                    accuracy=metric_data['accuracy'],
                    memory_usage=metric_data['memory_usage'],
                    cpu_usage=metric_data['cpu_usage'],
                    success_rate=metric_data['success_rate'],
                    user_satisfaction=metric_data['user_satisfaction'],
                    error_count=metric_data['error_count'],
                    task_completion_rate=metric_data['task_completion_rate']
                )
                self.performance_history.append(metric)
        
        logger.info(f"Loaded {len(self.performance_history)} historical performance metrics")
    
    async def _load_agent_configurations(self):
        """Lädt Agent-Konfigurationen."""
        query = """
        MATCH (a:Agent)-[:HAS_CONFIG]->(c:AgentConfiguration)
        RETURN a.id as agent_id, c
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                config_data = record['c']
                config = AgentConfiguration(
                    agent_id=record['agent_id'],
                    learning_rate=config_data.get('learning_rate', 0.001),
                    batch_size=config_data.get('batch_size', 32),
                    temperature=config_data.get('temperature', 0.7),
                    max_tokens=config_data.get('max_tokens', 2048),
                    context_window=config_data.get('context_window', 4096),
                    memory_threshold=config_data.get('memory_threshold', 0.8),
                    optimization_strategy=config_data.get('optimization_strategy', 'adaptive'),
                    model_parameters=json.loads(config_data.get('model_parameters', '{}'))
                )
                self.agent_configs[record['agent_id']] = config
        
        logger.info(f"Loaded configurations for {len(self.agent_configs)} agents")
    
    async def _trigger_improvement_analysis(self, agent_id: str):
        """Triggert Verbesserungsanalyse für einen Agent.
        
        Args:
            agent_id: ID des zu analysierenden Agents
        """
        if self.optimization_running:
            return
            
        try:
            self.optimization_running = True
            
            # Analysiere Performance-Trends
            trends = await self._analyze_performance_trends(agent_id)
            
            # Generiere Verbesserungsvorschläge
            improvements = await self._generate_improvements(agent_id, trends)
            
            # Wende Verbesserungen an
            if improvements:
                await self._apply_improvements(agent_id, improvements)
                
        except Exception as e:
            logger.error(f"Failed to analyze improvements for agent {agent_id}: {e}")
        finally:
            self.optimization_running = False
    
    async def _analyze_performance_trends(self, agent_id: str) -> Dict[str, Any]:
        """Analysiert Performance-Trends für einen Agent.
        
        Args:
            agent_id: ID des Agents
            
        Returns:
            Dictionary mit Trend-Analysen
        """
        # Filtere Metriken für den spezifischen Agent
        agent_metrics = [m for m in self.performance_history if m.agent_id == agent_id]
        
        if len(agent_metrics) < 5:
            return {'insufficient_data': True}
        
        # Sortiere nach Zeitstempel
        agent_metrics.sort(key=lambda x: x.timestamp)
        
        # Berechne Trends
        recent_metrics = agent_metrics[-10:]  # Letzte 10 Metriken
        older_metrics = agent_metrics[-20:-10] if len(agent_metrics) >= 20 else agent_metrics[:-10]
        
        trends = {
            'response_time_trend': self._calculate_trend([m.response_time for m in recent_metrics]),
            'accuracy_trend': self._calculate_trend([m.accuracy for m in recent_metrics]),
            'memory_usage_trend': self._calculate_trend([m.memory_usage for m in recent_metrics]),
            'success_rate_trend': self._calculate_trend([m.success_rate for m in recent_metrics]),
            'user_satisfaction_trend': self._calculate_trend([m.user_satisfaction for m in recent_metrics]),
            'performance_degradation': self._detect_performance_degradation(recent_metrics, older_metrics),
            'bottlenecks': self._identify_bottlenecks(recent_metrics)
        }
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """Berechnet Trend für eine Werteliste.
        
        Args:
            values: Liste von Werten
            
        Returns:
            Dictionary mit Trend-Informationen
        """
        if len(values) < 2:
            return {'slope': 0.0, 'direction': 'stable'}
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Lineare Regression für Trend
        slope = np.polyfit(x, y, 1)[0]
        
        direction = 'improving' if slope > 0.01 else 'degrading' if slope < -0.01 else 'stable'
        
        return {
            'slope': float(slope),
            'direction': direction,
            'current_value': float(values[-1]),
            'average': float(np.mean(values)),
            'std': float(np.std(values))
        }
    
    def _detect_performance_degradation(self, recent: List[PerformanceMetric], 
                                      older: List[PerformanceMetric]) -> Dict[str, bool]:
        """Erkennt Performance-Verschlechterungen.
        
        Args:
            recent: Aktuelle Metriken
            older: Ältere Metriken
            
        Returns:
            Dictionary mit Degradation-Flags
        """
        if not older:
            return {'detected': False}
        
        recent_avg = {
            'response_time': np.mean([m.response_time for m in recent]),
            'accuracy': np.mean([m.accuracy for m in recent]),
            'success_rate': np.mean([m.success_rate for m in recent]),
            'user_satisfaction': np.mean([m.user_satisfaction for m in recent])
        }
        
        older_avg = {
            'response_time': np.mean([m.response_time for m in older]),
            'accuracy': np.mean([m.accuracy for m in older]),
            'success_rate': np.mean([m.success_rate for m in older]),
            'user_satisfaction': np.mean([m.user_satisfaction for m in older])
        }
        
        degradation = {
            'detected': False,
            'response_time_worse': recent_avg['response_time'] > older_avg['response_time'] * 1.2,
            'accuracy_worse': recent_avg['accuracy'] < older_avg['accuracy'] * 0.9,
            'success_rate_worse': recent_avg['success_rate'] < older_avg['success_rate'] * 0.9,
            'satisfaction_worse': recent_avg['user_satisfaction'] < older_avg['user_satisfaction'] * 0.9
        }
        
        degradation['detected'] = any([
            degradation['response_time_worse'],
            degradation['accuracy_worse'],
            degradation['success_rate_worse'],
            degradation['satisfaction_worse']
        ])
        
        return degradation
    
    def _identify_bottlenecks(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Identifiziert Performance-Bottlenecks.
        
        Args:
            metrics: Liste von Performance-Metriken
            
        Returns:
            Dictionary mit identifizierten Bottlenecks
        """
        if not metrics:
            return {}
        
        avg_response_time = np.mean([m.response_time for m in metrics])
        avg_memory_usage = np.mean([m.memory_usage for m in metrics])
        avg_cpu_usage = np.mean([m.cpu_usage for m in metrics])
        avg_error_count = np.mean([m.error_count for m in metrics])
        
        bottlenecks = {
            'high_response_time': avg_response_time > 2.0,  # > 2 Sekunden
            'high_memory_usage': avg_memory_usage > 0.8,   # > 80%
            'high_cpu_usage': avg_cpu_usage > 0.8,         # > 80%
            'high_error_rate': avg_error_count > 5,        # > 5 Fehler
            'primary_bottleneck': None
        }
        
        # Identifiziere primären Bottleneck
        if bottlenecks['high_response_time']:
            bottlenecks['primary_bottleneck'] = 'response_time'
        elif bottlenecks['high_memory_usage']:
            bottlenecks['primary_bottleneck'] = 'memory'
        elif bottlenecks['high_cpu_usage']:
            bottlenecks['primary_bottleneck'] = 'cpu'
        elif bottlenecks['high_error_rate']:
            bottlenecks['primary_bottleneck'] = 'errors'
        
        return bottlenecks
    
    async def _generate_improvements(self, agent_id: str, trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generiert Verbesserungsvorschläge basierend auf Trends.
        
        Args:
            agent_id: ID des Agents
            trends: Analysierte Trends
            
        Returns:
            Liste von Verbesserungsvorschlägen
        """
        improvements = []
        
        if trends.get('insufficient_data'):
            return improvements
        
        current_config = self.agent_configs.get(agent_id)
        if not current_config:
            return improvements
        
        # Response Time Verbesserungen
        if trends['response_time_trend']['direction'] == 'degrading':
            improvements.append({
                'type': 'reduce_max_tokens',
                'current_value': current_config.max_tokens,
                'suggested_value': max(512, int(current_config.max_tokens * 0.8)),
                'reason': 'Reduce response time by limiting token generation'
            })
        
        # Accuracy Verbesserungen
        if trends['accuracy_trend']['direction'] == 'degrading':
            improvements.append({
                'type': 'adjust_temperature',
                'current_value': current_config.temperature,
                'suggested_value': max(0.1, current_config.temperature - 0.1),
                'reason': 'Improve accuracy by reducing randomness'
            })
        
        # Memory Usage Verbesserungen
        if trends['memory_usage_trend']['current_value'] > 0.8:
            improvements.append({
                'type': 'reduce_context_window',
                'current_value': current_config.context_window,
                'suggested_value': max(1024, int(current_config.context_window * 0.7)),
                'reason': 'Reduce memory usage by limiting context window'
            })
        
        # Bottleneck-spezifische Verbesserungen
        bottlenecks = trends.get('bottlenecks', {})
        if bottlenecks.get('primary_bottleneck') == 'memory':
            improvements.append({
                'type': 'adjust_batch_size',
                'current_value': current_config.batch_size,
                'suggested_value': max(8, int(current_config.batch_size * 0.5)),
                'reason': 'Reduce memory pressure by decreasing batch size'
            })
        
        return improvements
    
    async def _apply_improvements(self, agent_id: str, improvements: List[Dict[str, Any]]):
        """Wendet Verbesserungen auf einen Agent an.
        
        Args:
            agent_id: ID des Agents
            improvements: Liste von Verbesserungen
        """
        current_config = self.agent_configs.get(agent_id)
        if not current_config:
            logger.warning(f"No configuration found for agent {agent_id}")
            return
        
        # Erstelle neue Konfiguration
        new_config = AgentConfiguration(
            agent_id=current_config.agent_id,
            learning_rate=current_config.learning_rate,
            batch_size=current_config.batch_size,
            temperature=current_config.temperature,
            max_tokens=current_config.max_tokens,
            context_window=current_config.context_window,
            memory_threshold=current_config.memory_threshold,
            optimization_strategy=current_config.optimization_strategy,
            model_parameters=current_config.model_parameters.copy()
        )
        
        # Wende Verbesserungen an
        for improvement in improvements:
            improvement_type = improvement['type']
            suggested_value = improvement['suggested_value']
            
            if improvement_type == 'reduce_max_tokens':
                new_config.max_tokens = suggested_value
            elif improvement_type == 'adjust_temperature':
                new_config.temperature = suggested_value
            elif improvement_type == 'reduce_context_window':
                new_config.context_window = suggested_value
            elif improvement_type == 'adjust_batch_size':
                new_config.batch_size = suggested_value
            
            logger.info(f"Applied improvement {improvement_type} for agent {agent_id}: "
                       f"{improvement['current_value']} -> {suggested_value}")
        
        # Speichere neue Konfiguration
        await self._store_agent_configuration(new_config)
        self.agent_configs[agent_id] = new_config
        
        # Protokolliere Verbesserung
        await self._log_improvement_event(agent_id, improvements)
    
    async def _store_agent_configuration(self, config: AgentConfiguration):
        """Speichert Agent-Konfiguration in Neo4j."""
        query = """
        MERGE (a:Agent {id: $agent_id})
        MERGE (a)-[:HAS_CONFIG]->(c:AgentConfiguration)
        SET c.learning_rate = $learning_rate,
            c.batch_size = $batch_size,
            c.temperature = $temperature,
            c.max_tokens = $max_tokens,
            c.context_window = $context_window,
            c.memory_threshold = $memory_threshold,
            c.optimization_strategy = $optimization_strategy,
            c.model_parameters = $model_parameters,
            c.updated_at = datetime()
        """
        
        with self.driver.session() as session:
            session.run(query, {
                'agent_id': config.agent_id,
                'learning_rate': config.learning_rate,
                'batch_size': config.batch_size,
                'temperature': config.temperature,
                'max_tokens': config.max_tokens,
                'context_window': config.context_window,
                'memory_threshold': config.memory_threshold,
                'optimization_strategy': config.optimization_strategy,
                'model_parameters': json.dumps(config.model_parameters)
            })
    
    async def _log_improvement_event(self, agent_id: str, improvements: List[Dict[str, Any]]):
        """Protokolliert Verbesserungs-Event."""
        query = """
        MATCH (a:Agent {id: $agent_id})
        CREATE (e:ImprovementEvent {
            timestamp: datetime(),
            improvements: $improvements,
            improvement_count: $improvement_count
        })
        CREATE (a)-[:HAD_IMPROVEMENT]->(e)
        """
        
        with self.driver.session() as session:
            session.run(query, {
                'agent_id': agent_id,
                'improvements': json.dumps(improvements),
                'improvement_count': len(improvements)
            })
            
    async def generate_mandatory_session_end_prompt(self, work_completed: str, current_status: str, 
                                                   next_agent_instructions: str, critical_context: str, 
                                                   files_modified: List[str]) -> str:
        """Generiert einen strukturierten Prompt für Session-Ende.
        
        Args:
            work_completed: Zusammenfassung der erledigten Arbeit
            current_status: Aktueller Status des Agents
            next_agent_instructions: Anweisungen für den nächsten Agent
            critical_context: Kritischer Kontext für die Übergabe
            files_modified: Liste der geänderten Dateien
            
        Returns:
            Strukturierter Prompt für Session-Ende
        """
        prompt = (
            f"# SESSION END REPORT\n"
            f"## Work Completed\n{work_completed}\n\n"
            f"## Current Status\n{current_status}\n\n"
            f"## Next Agent Instructions\n{next_agent_instructions}\n\n"
            f"## Critical Context\n{critical_context}\n\n"
            f"## Files Modified\n" + '\n'.join(f'- {f}' for f in files_modified)
        )
        
        # Log coordination event
        self.coordination_history.append({
            'type': 'session_end',
            'timestamp': datetime.now(),
            'agent_id': self.agent_id,
            'prompt': prompt
        })
        
        return prompt
        
    async def generate_handoff_prompt_only(self, next_agent_instructions: str, 
                                         critical_context: str) -> str:
        """Generiert einen reduzierten Handoff-Prompt.
        
        Args:
            next_agent_instructions: Anweisungen für den nächsten Agent
            critical_context: Kritischer Kontext für die Übergabe
            
        Returns:
            Strukturierter Handoff-Prompt
        """
        prompt = (
            f"# AGENT HANDOFF\n"
            f"## Next Agent Instructions\n{next_agent_instructions}\n\n"
            f"## Critical Context\n{critical_context}"
        )
        
        # Log coordination event
        self.coordination_history.append({
            'type': 'handoff',
            'timestamp': datetime.now(),
            'agent_id': self.agent_id,
            'prompt': prompt
        })
        
        return prompt
        
    async def generate_meta_feedback(self, feedback_type: str, feedback_content: str, 
                                    severity: str = 'medium') -> Dict[str, Any]:
        """Generiert Meta-Feedback für das System.
        
        Args:
            feedback_type: Typ des Feedbacks (z.B. 'performance', 'coordination')
            feedback_content: Inhalt des Feedbacks
            severity: Schweregrad ('low', 'medium', 'high')
            
        Returns:
            Dictionary mit Feedback-Details
        """
        feedback = {
            'type': feedback_type,
            'content': feedback_content,
            'severity': severity,
            'timestamp': datetime.now(),
            'agent_id': self.agent_id
        }
        
        # Store in coordination history
        self.coordination_history.append({
            'type': 'meta_feedback',
            'timestamp': datetime.now(),
            'feedback': feedback
        })
        
        # Also store in Neo4j
        query = """
        MATCH (a:Agent {id: $agent_id})
        CREATE (f:MetaFeedback {
            type: $type,
            content: $content,
            severity: $severity,
            timestamp: datetime()
        })
        CREATE (a)-[:PROVIDED_FEEDBACK]->(f)
        """
        
        with self.driver.session() as session:
            session.run(query, {
                'agent_id': self.agent_id,
                'type': feedback_type,
                'content': feedback_content,
                'severity': severity
            })
        
        return feedback
    
    async def get_agent_performance_summary(self, agent_id: str) -> Dict[str, Any]:
        """Gibt Performance-Zusammenfassung für einen Agent zurück.
        
        Args:
            agent_id: ID des Agents
            
        Returns:
            Dictionary mit Performance-Zusammenfassung
        """
        agent_metrics = [m for m in self.performance_history if m.agent_id == agent_id]
        
        if not agent_metrics:
            return {'error': 'No metrics found for agent'}
        
        recent_metrics = agent_metrics[-10:] if len(agent_metrics) >= 10 else agent_metrics
        
        summary = {
            'agent_id': agent_id,
            'total_metrics': len(agent_metrics),
            'recent_performance': {
                'avg_response_time': np.mean([m.response_time for m in recent_metrics]),
                'avg_accuracy': np.mean([m.accuracy for m in recent_metrics]),
                'avg_success_rate': np.mean([m.success_rate for m in recent_metrics]),
                'avg_user_satisfaction': np.mean([m.user_satisfaction for m in recent_metrics]),
                'total_errors': sum([m.error_count for m in recent_metrics])
            },
            'trends': await self._analyze_performance_trends(agent_id),
            'current_config': asdict(self.agent_configs.get(agent_id)) if agent_id in self.agent_configs else None
        }
        
        return summary
    
    async def start_continuous_optimization(self, interval_minutes: int = 30):
        """Startet kontinuierliche Optimierung.
        
        Args:
            interval_minutes: Intervall in Minuten zwischen Optimierungsläufen
        """
        logger.info(f"Starting continuous optimization with {interval_minutes} minute intervals")
        
        while True:
            try:
                # Optimiere alle Agents
                for agent_id in self.agent_configs.keys():
                    await self._trigger_improvement_analysis(agent_id)
                
                # Warte bis zum nächsten Intervall
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in continuous optimization: {e}")
                await asyncio.sleep(60)  # Kurze Pause bei Fehlern

# Beispiel-Nutzung
if __name__ == "__main__":
    async def main():
        engine = AgentImprovementEngine()
        await engine.initialize()
        
        # Beispiel-Metrik
        metric = PerformanceMetric(
            agent_id="test_agent",
            timestamp=datetime.now(),
            response_time=1.5,
            accuracy=0.85,
            memory_usage=0.6,
            cpu_usage=0.4,
            success_rate=0.9,
            user_satisfaction=0.8,
            error_count=2,
            task_completion_rate=0.95
        )
        
        await engine.record_performance(metric)
        summary = await engine.get_agent_performance_summary("test_agent")
        print(json.dumps(summary, indent=2, default=str))
        
        await engine.close()
    
    asyncio.run(main())
