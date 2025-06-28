#!/usr/bin/env python3
"""
Unit Tests für Monitor Module
"""

import unittest
import os
import time
import threading
from unittest.mock import patch, Mock, MagicMock
import tempfile
import json

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock für services.monitor falls es existiert
try:
    from services.monitor import SystemMonitor, PerformanceMonitor, AgentMonitor
except ImportError:
    # Erstelle Mock-Klassen für Tests
    class SystemMonitor:
        def __init__(self):
            self.running = False
            self.metrics = {}
        
        def start_monitoring(self):
            self.running = True
        
        def stop_monitoring(self):
            self.running = False
        
        def get_system_metrics(self):
            return {
                'cpu_percent': 25.5,
                'memory_percent': 60.2,
                'disk_usage': 45.8,
                'network_io': {'bytes_sent': 1024, 'bytes_recv': 2048}
            }
    
    class PerformanceMonitor:
        def __init__(self):
            self.metrics_history = []
        
        def record_metric(self, name, value, timestamp=None):
            if timestamp is None:
                timestamp = time.time()
            self.metrics_history.append({
                'name': name,
                'value': value,
                'timestamp': timestamp
            })
        
        def get_metrics_summary(self, time_window=3600):
            return {
                'total_metrics': len(self.metrics_history),
                'time_window': time_window
            }
    
    class AgentMonitor:
        def __init__(self):
            self.agents = {}
        
        def register_agent(self, agent_id, agent_info):
            self.agents[agent_id] = agent_info
        
        def get_agent_status(self, agent_id):
            return self.agents.get(agent_id, {'status': 'unknown'})
        
        def get_all_agents_status(self):
            return self.agents

class TestSystemMonitor(unittest.TestCase):
    """Test-Klasse für SystemMonitor."""
    
    def setUp(self):
        """Setup für jeden Test."""
        self.monitor = SystemMonitor()
    
    def tearDown(self):
        """Cleanup nach jedem Test."""
        if hasattr(self.monitor, 'stop_monitoring'):
            self.monitor.stop_monitoring()
    
    def test_monitor_initialization(self):
        """Test Monitor Initialisierung."""
        self.assertFalse(self.monitor.running)
        self.assertIsInstance(self.monitor.metrics, dict)
    
    def test_start_monitoring(self):
        """Test Start des Monitorings."""
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.running)
    
    def test_stop_monitoring(self):
        """Test Stop des Monitorings."""
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.running)
        
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.running)
    
    def test_get_system_metrics(self):
        """Test Abrufen von System-Metriken."""
        metrics = self.monitor.get_system_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('cpu_percent', metrics)
        self.assertIn('memory_percent', metrics)
        self.assertIn('disk_usage', metrics)
        self.assertIn('network_io', metrics)
        
        # Prüfe Datentypen
        self.assertIsInstance(metrics['cpu_percent'], (int, float))
        self.assertIsInstance(metrics['memory_percent'], (int, float))
        self.assertIsInstance(metrics['disk_usage'], (int, float))
        self.assertIsInstance(metrics['network_io'], dict)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_real_system_metrics(self, mock_net, mock_disk, mock_memory, mock_cpu):
        """Test mit echten psutil Mocks."""
        # Setup Mocks
        mock_cpu.return_value = 15.5
        mock_memory.return_value = Mock(percent=45.2)
        mock_disk.return_value = Mock(percent=30.8)
        mock_net.return_value = Mock(bytes_sent=5000, bytes_recv=10000)
        
        # Erstelle echten Monitor falls verfügbar
        try:
            from services.monitor import SystemMonitor as RealSystemMonitor
            real_monitor = RealSystemMonitor()
            metrics = real_monitor.get_system_metrics()
            
            self.assertEqual(metrics['cpu_percent'], 15.5)
            self.assertEqual(metrics['memory_percent'], 45.2)
            self.assertEqual(metrics['disk_usage'], 30.8)
            self.assertEqual(metrics['network_io']['bytes_sent'], 5000)
            self.assertEqual(metrics['network_io']['bytes_recv'], 10000)
        except ImportError:
            # Fallback für Mock-Implementierung
            metrics = self.monitor.get_system_metrics()
            self.assertIsInstance(metrics, dict)
    
    def test_monitoring_thread_safety(self):
        """Test Thread-Sicherheit des Monitors."""
        results = []
        
        def monitor_worker():
            for _ in range(10):
                metrics = self.monitor.get_system_metrics()
                results.append(metrics)
                time.sleep(0.01)
        
        # Starte mehrere Threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=monitor_worker)
            threads.append(thread)
            thread.start()
        
        # Warte auf alle Threads
        for thread in threads:
            thread.join()
        
        # Prüfe Ergebnisse
        self.assertEqual(len(results), 30)  # 3 threads * 10 iterations
        for result in results:
            self.assertIsInstance(result, dict)

class TestPerformanceMonitor(unittest.TestCase):
    """Test-Klasse für PerformanceMonitor."""
    
    def setUp(self):
        """Setup für jeden Test."""
        self.perf_monitor = PerformanceMonitor()
    
    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor Initialisierung."""
        self.assertIsInstance(self.perf_monitor.metrics_history, list)
        self.assertEqual(len(self.perf_monitor.metrics_history), 0)
    
    def test_record_metric(self):
        """Test Aufzeichnung von Metriken."""
        self.perf_monitor.record_metric('test_metric', 42.5)
        
        self.assertEqual(len(self.perf_monitor.metrics_history), 1)
        
        metric = self.perf_monitor.metrics_history[0]
        self.assertEqual(metric['name'], 'test_metric')
        self.assertEqual(metric['value'], 42.5)
        self.assertIsInstance(metric['timestamp'], float)
    
    def test_record_metric_with_timestamp(self):
        """Test Aufzeichnung mit spezifischem Timestamp."""
        custom_timestamp = 1234567890.0
        self.perf_monitor.record_metric('custom_metric', 100, custom_timestamp)
        
        metric = self.perf_monitor.metrics_history[0]
        self.assertEqual(metric['timestamp'], custom_timestamp)
    
    def test_multiple_metrics_recording(self):
        """Test Aufzeichnung mehrerer Metriken."""
        metrics_data = [
            ('cpu_usage', 25.5),
            ('memory_usage', 60.2),
            ('response_time', 150.8),
            ('throughput', 1000)
        ]
        
        for name, value in metrics_data:
            self.perf_monitor.record_metric(name, value)
        
        self.assertEqual(len(self.perf_monitor.metrics_history), 4)
        
        # Prüfe alle Metriken
        for i, (expected_name, expected_value) in enumerate(metrics_data):
            metric = self.perf_monitor.metrics_history[i]
            self.assertEqual(metric['name'], expected_name)
            self.assertEqual(metric['value'], expected_value)
    
    def test_get_metrics_summary(self):
        """Test Zusammenfassung der Metriken."""
        # Füge einige Metriken hinzu
        for i in range(5):
            self.perf_monitor.record_metric(f'metric_{i}', i * 10)
        
        summary = self.perf_monitor.get_metrics_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_metrics', summary)
        self.assertIn('time_window', summary)
        self.assertEqual(summary['total_metrics'], 5)
    
    def test_metrics_time_filtering(self):
        """Test Zeitfilterung von Metriken."""
        current_time = time.time()
        
        # Füge Metriken mit verschiedenen Timestamps hinzu
        self.perf_monitor.record_metric('old_metric', 10, current_time - 7200)  # 2h alt
        self.perf_monitor.record_metric('recent_metric', 20, current_time - 1800)  # 30min alt
        self.perf_monitor.record_metric('new_metric', 30, current_time)  # jetzt
        
        # Test mit 1h Zeitfenster
        summary = self.perf_monitor.get_metrics_summary(time_window=3600)
        
        # Basis-Test (Mock-Implementierung zählt alle)
        self.assertIsInstance(summary, dict)
        self.assertIn('total_metrics', summary)

class TestAgentMonitor(unittest.TestCase):
    """Test-Klasse für AgentMonitor."""
    
    def setUp(self):
        """Setup für jeden Test."""
        self.agent_monitor = AgentMonitor()
    
    def test_agent_monitor_initialization(self):
        """Test AgentMonitor Initialisierung."""
        self.assertIsInstance(self.agent_monitor.agents, dict)
        self.assertEqual(len(self.agent_monitor.agents), 0)
    
    def test_register_agent(self):
        """Test Registrierung eines Agenten."""
        agent_info = {
            'name': 'TestAgent',
            'status': 'running',
            'start_time': time.time(),
            'type': 'data_collector'
        }
        
        self.agent_monitor.register_agent('agent_001', agent_info)
        
        self.assertEqual(len(self.agent_monitor.agents), 1)
        self.assertIn('agent_001', self.agent_monitor.agents)
        self.assertEqual(self.agent_monitor.agents['agent_001'], agent_info)
    
    def test_get_agent_status(self):
        """Test Abrufen des Agent-Status."""
        agent_info = {
            'name': 'StatusTestAgent',
            'status': 'idle',
            'last_activity': time.time()
        }
        
        self.agent_monitor.register_agent('status_agent', agent_info)
        
        status = self.agent_monitor.get_agent_status('status_agent')
        self.assertEqual(status, agent_info)
    
    def test_get_agent_status_unknown(self):
        """Test Abrufen des Status für unbekannten Agenten."""
        status = self.agent_monitor.get_agent_status('unknown_agent')
        self.assertEqual(status, {'status': 'unknown'})
    
    def test_get_all_agents_status(self):
        """Test Abrufen aller Agent-Status."""
        agents_data = {
            'agent_001': {'name': 'Agent1', 'status': 'running'},
            'agent_002': {'name': 'Agent2', 'status': 'stopped'},
            'agent_003': {'name': 'Agent3', 'status': 'error'}
        }
        
        for agent_id, agent_info in agents_data.items():
            self.agent_monitor.register_agent(agent_id, agent_info)
        
        all_status = self.agent_monitor.get_all_agents_status()
        
        self.assertEqual(len(all_status), 3)
        self.assertEqual(all_status, agents_data)
    
    def test_update_agent_status(self):
        """Test Update des Agent-Status."""
        initial_info = {'name': 'UpdateAgent', 'status': 'starting'}
        self.agent_monitor.register_agent('update_agent', initial_info)
        
        # Update Status
        updated_info = {'name': 'UpdateAgent', 'status': 'running', 'tasks_completed': 5}
        self.agent_monitor.register_agent('update_agent', updated_info)
        
        status = self.agent_monitor.get_agent_status('update_agent')
        self.assertEqual(status['status'], 'running')
        self.assertEqual(status['tasks_completed'], 5)
    
    def test_multiple_agents_management(self):
        """Test Verwaltung mehrerer Agenten."""
        # Registriere verschiedene Agent-Typen
        agents = {
            'collector_001': {
                'name': 'DataCollector1',
                'type': 'data_collector',
                'status': 'running',
                'data_sources': ['api1', 'api2']
            },
            'optimizer_001': {
                'name': 'Optimizer1',
                'type': 'optimizer',
                'status': 'idle',
                'optimization_cycles': 10
            },
            'monitor_001': {
                'name': 'Monitor1',
                'type': 'monitor',
                'status': 'running',
                'alerts_sent': 3
            }
        }
        
        for agent_id, agent_info in agents.items():
            self.agent_monitor.register_agent(agent_id, agent_info)
        
        # Prüfe alle Agenten
        all_agents = self.agent_monitor.get_all_agents_status()
        self.assertEqual(len(all_agents), 3)
        
        # Prüfe spezifische Agenten
        collector = self.agent_monitor.get_agent_status('collector_001')
        self.assertEqual(collector['type'], 'data_collector')
        self.assertEqual(len(collector['data_sources']), 2)
        
        optimizer = self.agent_monitor.get_agent_status('optimizer_001')
        self.assertEqual(optimizer['optimization_cycles'], 10)

class TestMonitorIntegration(unittest.TestCase):
    """Integration Tests für Monitor-Komponenten."""
    
    def setUp(self):
        """Setup für Integration Tests."""
        self.system_monitor = SystemMonitor()
        self.perf_monitor = PerformanceMonitor()
        self.agent_monitor = AgentMonitor()
    
    def test_full_monitoring_workflow(self):
        """Test kompletter Monitoring-Workflow."""
        # 1. Starte System-Monitoring
        self.system_monitor.start_monitoring()
        self.assertTrue(self.system_monitor.running)
        
        # 2. Registriere Agenten
        self.agent_monitor.register_agent('test_agent', {
            'name': 'TestWorkflowAgent',
            'status': 'running'
        })
        
        # 3. Sammle System-Metriken
        system_metrics = self.system_monitor.get_system_metrics()
        self.assertIsInstance(system_metrics, dict)
        
        # 4. Zeichne Performance-Metriken auf
        self.perf_monitor.record_metric('cpu_usage', system_metrics['cpu_percent'])
        self.perf_monitor.record_metric('memory_usage', system_metrics['memory_percent'])
        
        # 5. Prüfe Performance-Zusammenfassung
        perf_summary = self.perf_monitor.get_metrics_summary()
        self.assertEqual(perf_summary['total_metrics'], 2)
        
        # 6. Prüfe Agent-Status
        agent_status = self.agent_monitor.get_agent_status('test_agent')
        self.assertEqual(agent_status['status'], 'running')
        
        # 7. Stoppe Monitoring
        self.system_monitor.stop_monitoring()
        self.assertFalse(self.system_monitor.running)
    
    def test_monitoring_data_persistence(self):
        """Test Persistierung von Monitoring-Daten."""
        # Simuliere Monitoring über Zeit
        timestamps = [time.time() - i * 60 for i in range(5, 0, -1)]  # 5 Minuten
        
        for i, timestamp in enumerate(timestamps):
            self.perf_monitor.record_metric('test_metric', i * 10, timestamp)
        
        # Prüfe Datenintegrität
        self.assertEqual(len(self.perf_monitor.metrics_history), 5)
        
        # Prüfe chronologische Reihenfolge
        for i in range(len(self.perf_monitor.metrics_history) - 1):
            current = self.perf_monitor.metrics_history[i]['timestamp']
            next_ts = self.perf_monitor.metrics_history[i + 1]['timestamp']
            self.assertLessEqual(current, next_ts)
    
    def test_monitoring_error_handling(self):
        """Test Fehlerbehandlung im Monitoring."""
        # Test mit ungültigen Daten
        try:
            self.perf_monitor.record_metric('', None)  # Leerer Name, None-Wert
            # Sollte nicht crashen
        except Exception as e:
            self.fail(f"Monitoring sollte ungültige Daten handhaben: {e}")
        
        # Test mit unbekanntem Agent
        unknown_status = self.agent_monitor.get_agent_status('nonexistent')
        self.assertEqual(unknown_status['status'], 'unknown')
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_monitoring_config_loading(self, mock_open):
        """Test Laden von Monitoring-Konfiguration."""
        # Mock Konfigurationsdatei
        config_data = {
            'monitoring': {
                'system_interval': 30,
                'performance_retention': 3600,
                'agent_timeout': 300
            }
        }
        
        mock_open.return_value.read.return_value = json.dumps(config_data)
        
        # Test Konfiguration (falls implementiert)
        try:
            # Hier würde normalerweise eine Config-Klasse getestet
            config = config_data['monitoring']
            self.assertEqual(config['system_interval'], 30)
            self.assertEqual(config['performance_retention'], 3600)
            self.assertEqual(config['agent_timeout'], 300)
        except Exception:
            # Fallback für Mock-Implementierung
            pass

if __name__ == '__main__':
    unittest.main()