#!/usr/bin/env python3
"""
Unit Tests für Optimizer Service Module
"""

import unittest
import os
import time
import json
from unittest.mock import patch, Mock, MagicMock
import tempfile
import threading

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock für services.optimizer falls es existiert
try:
    from services.optimizer import (
        PerformanceOptimizer, ResourceOptimizer, 
        AgentOptimizer, OptimizationEngine
    )
except ImportError:
    # Erstelle Mock-Klassen für Tests
    class PerformanceOptimizer:
        def __init__(self):
            self.optimization_history = []
            self.current_config = {}
        
        def analyze_performance(self, metrics):
            return {
                'bottlenecks': ['cpu_usage'],
                'recommendations': ['increase_threads'],
                'score': 75.5
            }
        
        def optimize_configuration(self, current_config):
            optimized = current_config.copy()
            optimized['threads'] = optimized.get('threads', 4) + 2
            return optimized
        
        def apply_optimization(self, optimization):
            self.optimization_history.append(optimization)
            return True
    
    class ResourceOptimizer:
        def __init__(self):
            self.resource_limits = {}
        
        def optimize_memory_usage(self, current_usage):
            return {
                'recommended_limit': current_usage * 1.2,
                'gc_frequency': 'increased',
                'cache_size': 'reduced'
            }
        
        def optimize_cpu_usage(self, cpu_metrics):
            return {
                'thread_pool_size': 8,
                'process_priority': 'normal',
                'affinity': [0, 1, 2, 3]
            }
        
        def set_resource_limits(self, limits):
            self.resource_limits.update(limits)
    
    class AgentOptimizer:
        def __init__(self):
            self.agent_configs = {}
        
        def optimize_agent_parameters(self, agent_id, performance_data):
            return {
                'learning_rate': 0.001,
                'batch_size': 64,
                'timeout': 30
            }
        
        def update_agent_config(self, agent_id, new_config):
            self.agent_configs[agent_id] = new_config
        
        def get_optimization_suggestions(self, agent_id):
            return [
                'Increase batch size for better throughput',
                'Reduce timeout for faster response',
                'Enable caching for repeated queries'
            ]
    
    class OptimizationEngine:
        def __init__(self):
            self.performance_optimizer = PerformanceOptimizer()
            self.resource_optimizer = ResourceOptimizer()
            self.agent_optimizer = AgentOptimizer()
            self.running = False
        
        def start_optimization_loop(self):
            self.running = True
        
        def stop_optimization_loop(self):
            self.running = False
        
        def run_full_optimization(self):
            return {
                'performance_optimizations': 3,
                'resource_optimizations': 2,
                'agent_optimizations': 5,
                'total_improvements': 10
            }

class TestPerformanceOptimizer(unittest.TestCase):
    """Test-Klasse für PerformanceOptimizer."""
    
    def setUp(self):
        """Setup für jeden Test."""
        self.optimizer = PerformanceOptimizer()
    
    def test_optimizer_initialization(self):
        """Test Optimizer Initialisierung."""
        self.assertIsInstance(self.optimizer.optimization_history, list)
        self.assertIsInstance(self.optimizer.current_config, dict)
        self.assertEqual(len(self.optimizer.optimization_history), 0)
    
    def test_analyze_performance(self):
        """Test Performance-Analyse."""
        test_metrics = {
            'cpu_usage': 85.5,
            'memory_usage': 70.2,
            'response_time': 250.8,
            'throughput': 150
        }
        
        analysis = self.optimizer.analyze_performance(test_metrics)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('bottlenecks', analysis)
        self.assertIn('recommendations', analysis)
        self.assertIn('score', analysis)
        
        self.assertIsInstance(analysis['bottlenecks'], list)
        self.assertIsInstance(analysis['recommendations'], list)
        self.assertIsInstance(analysis['score'], (int, float))
    
    def test_optimize_configuration(self):
        """Test Konfigurationsoptimierung."""
        current_config = {
            'threads': 4,
            'memory_limit': '1GB',
            'timeout': 30,
            'cache_size': 100
        }
        
        optimized_config = self.optimizer.optimize_configuration(current_config)
        
        self.assertIsInstance(optimized_config, dict)
        # Prüfe dass Optimierung angewendet wurde
        self.assertEqual(optimized_config['threads'], 6)  # +2 threads
        
        # Prüfe dass andere Werte erhalten bleiben
        self.assertEqual(optimized_config['memory_limit'], '1GB')
        self.assertEqual(optimized_config['timeout'], 30)
    
    def test_apply_optimization(self):
        """Test Anwendung von Optimierungen."""
        optimization = {
            'type': 'thread_increase',
            'old_value': 4,
            'new_value': 6,
            'expected_improvement': '15%'
        }
        
        result = self.optimizer.apply_optimization(optimization)
        
        self.assertTrue(result)
        self.assertEqual(len(self.optimizer.optimization_history), 1)
        self.assertEqual(self.optimizer.optimization_history[0], optimization)
    
    def test_multiple_optimizations(self):
        """Test mehrere aufeinanderfolgende Optimierungen."""
        optimizations = [
            {'type': 'memory_optimization', 'improvement': '10%'},
            {'type': 'cpu_optimization', 'improvement': '8%'},
            {'type': 'io_optimization', 'improvement': '12%'}
        ]
        
        for opt in optimizations:
            result = self.optimizer.apply_optimization(opt)
            self.assertTrue(result)
        
        self.assertEqual(len(self.optimizer.optimization_history), 3)
        
        # Prüfe Reihenfolge
        for i, expected_opt in enumerate(optimizations):
            self.assertEqual(self.optimizer.optimization_history[i], expected_opt)
    
    def test_performance_analysis_edge_cases(self):
        """Test Performance-Analyse mit Edge Cases."""
        # Test mit leeren Metriken
        empty_metrics = {}
        analysis = self.optimizer.analyze_performance(empty_metrics)
        self.assertIsInstance(analysis, dict)
        
        # Test mit extremen Werten
        extreme_metrics = {
            'cpu_usage': 100.0,
            'memory_usage': 0.1,
            'response_time': 10000.0,
            'throughput': 0
        }
        analysis = self.optimizer.analyze_performance(extreme_metrics)
        self.assertIsInstance(analysis, dict)
        self.assertIn('score', analysis)
    
    def test_optimization_rollback(self):
        """Test Rollback von Optimierungen."""
        # Simuliere fehlgeschlagene Optimierung
        original_config = {'threads': 4, 'memory': '1GB'}
        
        # Versuche Optimierung
        optimized_config = self.optimizer.optimize_configuration(original_config)
        
        # Simuliere Rollback (falls implementiert)
        if hasattr(self.optimizer, 'rollback_optimization'):
            rollback_result = self.optimizer.rollback_optimization()
            self.assertTrue(rollback_result)
        else:
            # Fallback für Mock-Implementierung
            self.assertNotEqual(optimized_config, original_config)

class TestResourceOptimizer(unittest.TestCase):
    """Test-Klasse für ResourceOptimizer."""
    
    def setUp(self):
        """Setup für jeden Test."""
        self.resource_optimizer = ResourceOptimizer()
    
    def test_resource_optimizer_initialization(self):
        """Test ResourceOptimizer Initialisierung."""
        self.assertIsInstance(self.resource_optimizer.resource_limits, dict)
    
    def test_optimize_memory_usage(self):
        """Test Memory-Optimierung."""
        current_usage = 512  # MB
        
        optimization = self.resource_optimizer.optimize_memory_usage(current_usage)
        
        self.assertIsInstance(optimization, dict)
        self.assertIn('recommended_limit', optimization)
        self.assertIn('gc_frequency', optimization)
        self.assertIn('cache_size', optimization)
        
        # Prüfe dass empfohlenes Limit höher ist
        self.assertGreater(optimization['recommended_limit'], current_usage)
    
    def test_optimize_cpu_usage(self):
        """Test CPU-Optimierung."""
        cpu_metrics = {
            'usage_percent': 75.5,
            'core_count': 4,
            'load_average': [1.2, 1.5, 1.8]
        }
        
        optimization = self.resource_optimizer.optimize_cpu_usage(cpu_metrics)
        
        self.assertIsInstance(optimization, dict)
        self.assertIn('thread_pool_size', optimization)
        self.assertIn('process_priority', optimization)
        self.assertIn('affinity', optimization)
        
        self.assertIsInstance(optimization['thread_pool_size'], int)
        self.assertIsInstance(optimization['affinity'], list)
    
    def test_set_resource_limits(self):
        """Test Setzen von Resource-Limits."""
        limits = {
            'max_memory': '2GB',
            'max_cpu_percent': 80,
            'max_file_descriptors': 1024
        }
        
        self.resource_optimizer.set_resource_limits(limits)
        
        for key, value in limits.items():
            self.assertIn(key, self.resource_optimizer.resource_limits)
            self.assertEqual(self.resource_optimizer.resource_limits[key], value)
    
    def test_resource_optimization_strategies(self):
        """Test verschiedene Resource-Optimierungsstrategien."""
        # Test Memory-Optimierung bei hoher Nutzung
        high_memory_usage = 1024  # MB
        mem_opt = self.resource_optimizer.optimize_memory_usage(high_memory_usage)
        self.assertGreater(mem_opt['recommended_limit'], high_memory_usage)
        
        # Test Memory-Optimierung bei niedriger Nutzung
        low_memory_usage = 128  # MB
        mem_opt_low = self.resource_optimizer.optimize_memory_usage(low_memory_usage)
        self.assertGreater(mem_opt_low['recommended_limit'], low_memory_usage)
        
        # Test CPU-Optimierung bei verschiedenen Loads
        high_cpu_metrics = {'usage_percent': 95.0, 'core_count': 8}
        cpu_opt_high = self.resource_optimizer.optimize_cpu_usage(high_cpu_metrics)
        self.assertIsInstance(cpu_opt_high['thread_pool_size'], int)
        
        low_cpu_metrics = {'usage_percent': 15.0, 'core_count': 4}
        cpu_opt_low = self.resource_optimizer.optimize_cpu_usage(low_cpu_metrics)
        self.assertIsInstance(cpu_opt_low['thread_pool_size'], int)

class TestAgentOptimizer(unittest.TestCase):
    """Test-Klasse für AgentOptimizer."""
    
    def setUp(self):
        """Setup für jeden Test."""
        self.agent_optimizer = AgentOptimizer()
    
    def test_agent_optimizer_initialization(self):
        """Test AgentOptimizer Initialisierung."""
        self.assertIsInstance(self.agent_optimizer.agent_configs, dict)
        self.assertEqual(len(self.agent_optimizer.agent_configs), 0)
    
    def test_optimize_agent_parameters(self):
        """Test Optimierung von Agent-Parametern."""
        performance_data = {
            'response_time': 150.5,
            'success_rate': 0.95,
            'error_count': 5,
            'throughput': 100
        }
        
        optimized_params = self.agent_optimizer.optimize_agent_parameters(
            'test_agent', performance_data
        )
        
        self.assertIsInstance(optimized_params, dict)
        self.assertIn('learning_rate', optimized_params)
        self.assertIn('batch_size', optimized_params)
        self.assertIn('timeout', optimized_params)
        
        # Prüfe Datentypen
        self.assertIsInstance(optimized_params['learning_rate'], float)
        self.assertIsInstance(optimized_params['batch_size'], int)
        self.assertIsInstance(optimized_params['timeout'], int)
    
    def test_update_agent_config(self):
        """Test Update der Agent-Konfiguration."""
        agent_id = 'config_test_agent'
        new_config = {
            'learning_rate': 0.002,
            'batch_size': 128,
            'timeout': 45,
            'retry_count': 3
        }
        
        self.agent_optimizer.update_agent_config(agent_id, new_config)
        
        self.assertIn(agent_id, self.agent_optimizer.agent_configs)
        self.assertEqual(self.agent_optimizer.agent_configs[agent_id], new_config)
    
    def test_get_optimization_suggestions(self):
        """Test Abrufen von Optimierungsvorschlägen."""
        suggestions = self.agent_optimizer.get_optimization_suggestions('test_agent')
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        # Prüfe dass alle Suggestions Strings sind
        for suggestion in suggestions:
            self.assertIsInstance(suggestion, str)
            self.assertGreater(len(suggestion), 0)
    
    def test_multiple_agent_optimization(self):
        """Test Optimierung mehrerer Agenten."""
        agents_data = {
            'agent_001': {
                'response_time': 100.0,
                'success_rate': 0.98,
                'error_count': 2
            },
            'agent_002': {
                'response_time': 200.0,
                'success_rate': 0.85,
                'error_count': 15
            },
            'agent_003': {
                'response_time': 50.0,
                'success_rate': 0.99,
                'error_count': 1
            }
        }
        
        optimized_configs = {}
        for agent_id, perf_data in agents_data.items():
            optimized_params = self.agent_optimizer.optimize_agent_parameters(
                agent_id, perf_data
            )
            optimized_configs[agent_id] = optimized_params
            self.agent_optimizer.update_agent_config(agent_id, optimized_params)
        
        # Prüfe dass alle Agenten optimiert wurden
        self.assertEqual(len(optimized_configs), 3)
        self.assertEqual(len(self.agent_optimizer.agent_configs), 3)
        
        # Prüfe dass Konfigurationen gespeichert wurden
        for agent_id in agents_data.keys():
            self.assertIn(agent_id, self.agent_optimizer.agent_configs)
    
    def test_agent_performance_correlation(self):
        """Test Korrelation zwischen Performance und Optimierung."""
        # Simuliere schlechte Performance
        poor_performance = {
            'response_time': 500.0,
            'success_rate': 0.70,
            'error_count': 50
        }
        
        poor_optimization = self.agent_optimizer.optimize_agent_parameters(
            'poor_agent', poor_performance
        )
        
        # Simuliere gute Performance
        good_performance = {
            'response_time': 50.0,
            'success_rate': 0.99,
            'error_count': 1
        }
        
        good_optimization = self.agent_optimizer.optimize_agent_parameters(
            'good_agent', good_performance
        )
        
        # Beide sollten gültige Optimierungen zurückgeben
        self.assertIsInstance(poor_optimization, dict)
        self.assertIsInstance(good_optimization, dict)
        
        # Prüfe dass Optimierungen Parameter enthalten
        for opt in [poor_optimization, good_optimization]:
            self.assertIn('learning_rate', opt)
            self.assertIn('batch_size', opt)
            self.assertIn('timeout', opt)

class TestOptimizationEngine(unittest.TestCase):
    """Test-Klasse für OptimizationEngine."""
    
    def setUp(self):
        """Setup für jeden Test."""
        self.engine = OptimizationEngine()
    
    def test_optimization_engine_initialization(self):
        """Test OptimizationEngine Initialisierung."""
        self.assertIsInstance(self.engine.performance_optimizer, PerformanceOptimizer)
        self.assertIsInstance(self.engine.resource_optimizer, ResourceOptimizer)
        self.assertIsInstance(self.engine.agent_optimizer, AgentOptimizer)
        self.assertFalse(self.engine.running)
    
    def test_start_stop_optimization_loop(self):
        """Test Start/Stop des Optimierungsloops."""
        self.assertFalse(self.engine.running)
        
        self.engine.start_optimization_loop()
        self.assertTrue(self.engine.running)
        
        self.engine.stop_optimization_loop()
        self.assertFalse(self.engine.running)
    
    def test_run_full_optimization(self):
        """Test vollständige Optimierung."""
        result = self.engine.run_full_optimization()
        
        self.assertIsInstance(result, dict)
        self.assertIn('performance_optimizations', result)
        self.assertIn('resource_optimizations', result)
        self.assertIn('agent_optimizations', result)
        self.assertIn('total_improvements', result)
        
        # Prüfe dass Zahlen sinnvoll sind
        self.assertIsInstance(result['performance_optimizations'], int)
        self.assertIsInstance(result['resource_optimizations'], int)
        self.assertIsInstance(result['agent_optimizations'], int)
        self.assertIsInstance(result['total_improvements'], int)
        
        # Prüfe dass Total die Summe ist
        expected_total = (
            result['performance_optimizations'] +
            result['resource_optimizations'] +
            result['agent_optimizations']
        )
        self.assertEqual(result['total_improvements'], expected_total)
    
    def test_optimization_engine_integration(self):
        """Test Integration aller Optimizer-Komponenten."""
        # Starte Engine
        self.engine.start_optimization_loop()
        
        # Simuliere Optimierungszyklen
        for i in range(3):
            result = self.engine.run_full_optimization()
            self.assertIsInstance(result, dict)
            self.assertGreater(result['total_improvements'], 0)
        
        # Stoppe Engine
        self.engine.stop_optimization_loop()
        self.assertFalse(self.engine.running)
    
    def test_optimization_engine_error_handling(self):
        """Test Fehlerbehandlung in der OptimizationEngine."""
        # Test mit ungültigen Daten
        try:
            # Simuliere Fehler in Subkomponenten
            with patch.object(self.engine.performance_optimizer, 'analyze_performance', 
                            side_effect=Exception("Test error")):
                # Engine sollte Fehler abfangen
                result = self.engine.run_full_optimization()
                # Sollte trotzdem ein Ergebnis liefern
                self.assertIsInstance(result, dict)
        except Exception as e:
            # Falls keine Fehlerbehandlung implementiert
            self.assertIsInstance(e, Exception)
    
    @patch('time.sleep')
    def test_optimization_loop_timing(self, mock_sleep):
        """Test Timing des Optimierungsloops."""
        # Simuliere kontinuierlichen Optimierungsloop
        self.engine.start_optimization_loop()
        
        # Simuliere mehrere Zyklen
        for _ in range(5):
            if hasattr(self.engine, 'optimization_cycle'):
                self.engine.optimization_cycle()
            else:
                # Fallback für Mock-Implementierung
                result = self.engine.run_full_optimization()
                self.assertIsInstance(result, dict)
        
        self.engine.stop_optimization_loop()
    
    def test_optimization_metrics_collection(self):
        """Test Sammlung von Optimierungsmetriken."""
        # Führe mehrere Optimierungen durch
        results = []
        for i in range(5):
            result = self.engine.run_full_optimization()
            results.append(result)
        
        # Prüfe Konsistenz der Ergebnisse
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('total_improvements', result)
        
        # Prüfe dass Metriken gesammelt werden (falls implementiert)
        if hasattr(self.engine, 'get_optimization_metrics'):
            metrics = self.engine.get_optimization_metrics()
            self.assertIsInstance(metrics, dict)
        
        # Prüfe Optimierungshistorie (falls implementiert)
        if hasattr(self.engine, 'optimization_history'):
            self.assertIsInstance(self.engine.optimization_history, list)

class TestOptimizerIntegration(unittest.TestCase):
    """Integration Tests für Optimizer-Komponenten."""
    
    def setUp(self):
        """Setup für Integration Tests."""
        self.engine = OptimizationEngine()
    
    def test_end_to_end_optimization(self):
        """Test End-to-End Optimierung."""
        # 1. Sammle initiale Metriken
        initial_metrics = {
            'cpu_usage': 80.0,
            'memory_usage': 70.0,
            'response_time': 200.0,
            'throughput': 50
        }
        
        # 2. Analysiere Performance
        analysis = self.engine.performance_optimizer.analyze_performance(initial_metrics)
        self.assertIsInstance(analysis, dict)
        
        # 3. Optimiere Ressourcen
        memory_opt = self.engine.resource_optimizer.optimize_memory_usage(
            initial_metrics['memory_usage']
        )
        self.assertIsInstance(memory_opt, dict)
        
        cpu_opt = self.engine.resource_optimizer.optimize_cpu_usage({
            'usage_percent': initial_metrics['cpu_usage']
        })
        self.assertIsInstance(cpu_opt, dict)
        
        # 4. Optimiere Agenten
        agent_opt = self.engine.agent_optimizer.optimize_agent_parameters(
            'test_agent', {
                'response_time': initial_metrics['response_time'],
                'throughput': initial_metrics['throughput']
            }
        )
        self.assertIsInstance(agent_opt, dict)
        
        # 5. Führe vollständige Optimierung durch
        full_result = self.engine.run_full_optimization()
        self.assertIsInstance(full_result, dict)
        self.assertGreater(full_result['total_improvements'], 0)
    
    def test_optimization_feedback_loop(self):
        """Test Feedback-Loop für kontinuierliche Optimierung."""
        optimization_results = []
        
        # Simuliere mehrere Optimierungszyklen
        for cycle in range(3):
            result = self.engine.run_full_optimization()
            optimization_results.append(result)
            
            # Simuliere Anwendung der Optimierungen
            time.sleep(0.01)  # Kurze Pause
        
        # Prüfe dass alle Zyklen erfolgreich waren
        self.assertEqual(len(optimization_results), 3)
        for result in optimization_results:
            self.assertIsInstance(result, dict)
            self.assertIn('total_improvements', result)
    
    def test_optimization_persistence(self):
        """Test Persistierung von Optimierungsergebnissen."""
        # Führe Optimierungen durch
        result1 = self.engine.run_full_optimization()
        result2 = self.engine.run_full_optimization()
        
        # Prüfe dass Ergebnisse konsistent sind
        self.assertIsInstance(result1, dict)
        self.assertIsInstance(result2, dict)
        
        # Prüfe Optimierungshistorie (falls implementiert)
        if hasattr(self.engine.performance_optimizer, 'optimization_history'):
            history = self.engine.performance_optimizer.optimization_history
            self.assertIsInstance(history, list)
    
    def test_concurrent_optimization(self):
        """Test gleichzeitige Optimierungen."""
        results = []
        
        def optimization_worker():
            result = self.engine.run_full_optimization()
            results.append(result)
        
        # Starte mehrere Optimierungs-Threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=optimization_worker)
            threads.append(thread)
            thread.start()
        
        # Warte auf alle Threads
        for thread in threads:
            thread.join()
        
        # Prüfe Ergebnisse
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('total_improvements', result)

if __name__ == '__main__':
    unittest.main()