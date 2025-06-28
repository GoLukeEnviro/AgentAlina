#!/usr/bin/env python3
"""
Unit Tests für Knowledge Graph Collector
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the actual class for testing
from agents.knowledge_graph_collector import KnowledgeGraphCollector
from neo4j import GraphDatabase

import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
import requests

class TestKnowledgeGraphCollector:
    """Test-Klasse für KnowledgeGraphCollector."""
    
    @pytest.fixture(autouse=True)
    def collector(self):
        """Setup für jeden Test."""
        collector = KnowledgeGraphCollector()
        # Mock the driver and session to avoid actual database connections
        collector.driver = MagicMock()
        collector.session = MagicMock()
        return collector
    
    def test_connect_to_neo4j_success(self, collector):
        """Test der erfolgreichen Verbindung zu Neo4j."""
        with patch('neo4j.GraphDatabase.driver', return_value=MagicMock()):
            collector._connect_to_neo4j()
            assert collector.driver is not None
            assert collector.session is not None
    
    def test_connect_to_neo4j_failure(self, collector):
        """Test der fehlgeschlagenen Verbindung zu Neo4j."""
        with patch('neo4j.GraphDatabase.driver', side_effect=Exception("Connection failed")):
            with pytest.raises(Exception):
                collector._connect_to_neo4j()
    
    def test_create_node_success(self, collector):
        """Test der erfolgreichen Erstellung eines Knotens."""
        node_properties = {'id': 'test_id', 'name': 'Test Node'}
        with patch.object(collector.session, 'run', return_value=MagicMock()):
            result = collector.create_node('TestLabel', node_properties)
            assert result is True
            collector.session.run.assert_called_once()
    
    def test_create_node_failure(self, collector):
        """Test der fehlgeschlagenen Erstellung eines Knotens."""
        node_properties = {'id': 'test_id', 'name': 'Test Node'}
        with patch.object(collector.session, 'run', side_effect=Exception("Query failed")):
            result = collector.create_node('TestLabel', node_properties)
            assert result is False
    
    def test_create_relationship_success(self, collector):
        """Test der erfolgreichen Erstellung einer Beziehung."""
        from_node = {'id': 'source_id'}
        to_node = {'id': 'target_id'}
        relationship_type = 'RELATED_TO'
        properties = {'strength': 0.8}
        with patch.object(collector.session, 'run', return_value=MagicMock()):
            result = collector.create_relationship(from_node, to_node, relationship_type, properties)
            assert result is True
            collector.session.run.assert_called_once()
    
    def test_create_relationship_failure(self, collector):
        """Test der fehlgeschlagenen Erstellung einer Beziehung."""
        from_node = {'id': 'source_id'}
        to_node = {'id': 'target_id'}
        relationship_type = 'RELATED_TO'
        properties = {'strength': 0.8}
        with patch.object(collector.session, 'run', side_effect=Exception("Query failed")):
            result = collector.create_relationship(from_node, to_node, relationship_type, properties)
            assert result is False
    
    def test_fetch_data_from_api_success(self, collector):
        """Test der erfolgreichen API-Datenabfrage."""
        api_url = 'https://api.example.com/data'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': [{'id': 1, 'name': 'Test Item'}]}
        with patch('requests.get', return_value=mock_response):
            result = collector.fetch_data_from_api(api_url)
            assert result is not None
            assert result['data'][0]['name'] == 'Test Item'
    
    def test_fetch_data_from_api_failure(self, collector):
        """Test der fehlgeschlagenen API-Datenabfrage."""
        api_url = 'https://api.example.com/notfound'
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("Not found")
        with patch('requests.get', return_value=mock_response):
            result = collector.fetch_data_from_api(api_url)
            assert result is None
    
    def test_scrape_webpage_success(self, collector):
        """Test des erfolgreichen Webpage-Scrapings."""
        url = 'https://example.com'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'<html><title>Test Page</title><p>Content</p></html>'
        with patch('requests.get', return_value=mock_response):
            mock_soup = MagicMock()
            mock_title = MagicMock()
            mock_title.get_text.return_value = 'Test Page'
            mock_soup.find.return_value = mock_title
            mock_soup.find_all.return_value = [MagicMock(get_text=lambda: 'Content')]
            with patch('bs4.BeautifulSoup', return_value=mock_soup):
                result = collector.scrape_webpage(url)
                assert result is not None
                assert result['title'] == 'Test Page'
    
    def test_scrape_webpage_failure(self, collector):
        """Test des fehlgeschlagenen Webpage-Scrapings."""
        url = 'https://forbidden.com'
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("Forbidden")
        with patch('requests.get', return_value=mock_response):
            result = collector.scrape_webpage(url)
            assert result is None
    
    def test_integrate_api_data_success(self, collector):
        """Test der erfolgreichen API-Datenintegration."""
        api_name = 'test_api'
        api_url = 'https://api.example.com/items'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'items': [{'id': 'item1', 'name': 'Item 1'}]}
        with patch('requests.get', return_value=mock_response):
            with patch.object(collector, 'create_node', return_value=True):
                with patch('agents.knowledge_graph_collector.has_api_key', return_value=False):
                    result = collector.integrate_api_data(api_name, api_url)
                    assert result is True
                    collector.create_node.assert_called_once()
    
    def test_integrate_api_data_failure(self, collector):
        """Test der fehlgeschlagenen API-Datenintegration."""
        api_name = 'test_api'
        api_url = 'https://api.example.com/notfound'
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("Not found")
        with patch('requests.get', return_value=mock_response):
            result = collector.integrate_api_data(api_name, api_url)
            assert result is False
    
    def test_integrate_web_data_success(self, collector):
        """Test der erfolgreichen Web-Datenintegration."""
        url = 'https://example.com'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'<html><title>Test Page</title><p>Content</p></html>'
        with patch('requests.get', return_value=mock_response):
            mock_soup = MagicMock()
            mock_title = MagicMock()
            mock_title.get_text.return_value = 'Test Page'
            mock_soup.find.return_value = mock_title
            mock_soup.find_all.return_value = [MagicMock(get_text=lambda: 'Content')]
            with patch('bs4.BeautifulSoup', return_value=mock_soup):
                with patch.object(collector, 'create_node', return_value=True):
                    result = collector.integrate_web_data(url)
                    assert result is True
                    collector.create_node.assert_called_once()
    
    def test_integrate_web_data_failure(self, collector):
        """Test der fehlgeschlagenen Web-Datenintegration."""
        url = 'https://forbidden.com'
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("Forbidden")
        with patch('requests.get', return_value=mock_response):
            result = collector.integrate_web_data(url)
            assert result is False
    
    def test_run_collection_cycle(self, collector):
        """Test des vollständigen Datensammlungszyklus."""
        with patch.object(collector, 'integrate_api_data', side_effect=[True, True, True]):
            with patch.object(collector, 'integrate_web_data', side_effect=[True, True]):
                results = collector.run_collection_cycle()
                assert len(results) == 5  # 3 API + 2 Web
                assert results['api_coindesk'] is True
                assert results['api_coingecko'] is True
                assert results['api_github'] is True
    
    def test_close_connection(self, collector):
        """Test des Verbindungsschließens."""
        mock_session = MagicMock()
        mock_driver = MagicMock()
        collector.session = mock_session
        collector.driver = mock_driver
        collector.close()
        mock_session.close.assert_called_once()
        mock_driver.close.assert_called_once()

if __name__ == '__main__':
    unittest.main()
