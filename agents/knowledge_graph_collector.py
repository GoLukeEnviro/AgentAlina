#!/usr/bin/env python3
"""
Knowledge Graph Collector Agent

Implementiert einen AI-Agenten für automatisierte Datenerfassung und Integration in Neo4j.
Verwendet BeautifulSoup für Web-Scraping und requests für API-Abfragen.
"""

import requests
import json
import logging
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.env_config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    FINANCE_API_URL, NEWS_API_URL, WEATHER_API_URL, CRYPTO_API_URL,
    get_api_key, has_api_key
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphCollector:
    """AI-Agent für automatisierte Datenerfassung und Integration in Neo4j."""
    
    def __init__(self):
        """Initialisiert den Knowledge Graph Collector."""
        self.driver = None
        self.session = None
        self._connect_to_neo4j()
        
    def _connect_to_neo4j(self):
        """Stellt Verbindung zu Neo4j her."""
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI, 
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
            )
            self.session = self.driver.session()
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def create_node(self, label: str, properties: Dict[str, Any]) -> bool:
        """Erstellt einen Knoten im Knowledge Graph."""
        try:
            query = f"CREATE (n:{label} $props) RETURN n"
            result = self.session.run(query, props=properties)
            logger.info(f"Created node with label {label}")
            return True
        except Exception as e:
            logger.error(f"Failed to create node: {e}")
            return False
    
    def create_relationship(self, from_node: Dict, to_node: Dict, 
                          relationship_type: str, properties: Dict = None) -> bool:
        """Erstellt eine Beziehung zwischen zwei Knoten."""
        try:
            query = """
            MATCH (a), (b)
            WHERE a.id = $from_id AND b.id = $to_id
            CREATE (a)-[r:%s $props]->(b)
            RETURN r
            """ % relationship_type
            
            props = properties or {}
            result = self.session.run(query, 
                                    from_id=from_node.get('id'),
                                    to_id=to_node.get('id'),
                                    props=props)
            logger.info(f"Created relationship {relationship_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False
    
    def fetch_data_from_api(self, api_url: str, headers: Dict = None) -> Optional[Dict]:
        """Holt Daten von einer API ab."""
        try:
            response = requests.get(api_url, headers=headers or {}, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {api_url}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {api_url}: {e}")
            return None
    
    def scrape_webpage(self, url: str) -> Optional[Dict[str, str]]:
        """Scrapt eine Webseite und extrahiert relevante Informationen."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extrahiere grundlegende Informationen
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
            # Extrahiere Meta-Beschreibung
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ''
            
            # Extrahiere Haupttext (vereinfacht)
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs[:5]])
            
            return {
                'url': url,
                'title': title_text,
                'description': description,
                'content': content[:1000]  # Begrenzt auf 1000 Zeichen
            }
        except Exception as e:
            logger.error(f"Web scraping failed for {url}: {e}")
            return None
    
    def integrate_api_data(self, api_name: str, api_url: str) -> bool:
        """Integriert API-Daten in den Knowledge Graph."""
        try:
            # Hole API-Key falls verfügbar
            headers = {}
            if has_api_key(api_name.upper()):
                api_key = get_api_key(api_name.upper())
                if api_key:
                    headers['Authorization'] = f'Bearer {api_key}'
            
            data = self.fetch_data_from_api(api_url, headers)
            if not data:
                return False
            
            # Erstelle API-Datenknoten
            node_properties = {
                'id': f"{api_name}_{hash(str(data))}",
                'source': api_name,
                'data': json.dumps(data)[:1000],  # Begrenzt die Datengröße
                'timestamp': str(requests.utils.default_headers())
            }
            
            return self.create_node('APIData', node_properties)
        except Exception as e:
            logger.error(f"API integration failed for {api_name}: {e}")
            return False
    
    def integrate_web_data(self, url: str) -> bool:
        """Integriert Web-Scraping-Daten in den Knowledge Graph."""
        try:
            scraped_data = self.scrape_webpage(url)
            if not scraped_data:
                return False
            
            # Erstelle Web-Datenknoten
            node_properties = {
                'id': f"web_{hash(url)}",
                'url': url,
                'title': scraped_data.get('title', ''),
                'description': scraped_data.get('description', ''),
                'content': scraped_data.get('content', '')
            }
            
            return self.create_node('WebData', node_properties)
        except Exception as e:
            logger.error(f"Web data integration failed for {url}: {e}")
            return False
    
    def run_collection_cycle(self) -> Dict[str, bool]:
        """Führt einen vollständigen Datensammlungszyklus aus."""
        results = {}
        
        # Teste verschiedene APIs
        api_tests = [
            ('coindesk', 'https://api.coindesk.com/v1/bpi/currentprice.json'),
            ('coingecko', 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd'),
            ('github', 'https://api.github.com/repos/microsoft/vscode')
        ]
        
        for api_name, api_url in api_tests:
            results[f'api_{api_name}'] = self.integrate_api_data(api_name, api_url)
        
        # Teste Web-Scraping
        web_tests = [
            'https://example.com',
            'https://httpbin.org/html'
        ]
        
        for url in web_tests:
            results[f'web_{hash(url)}'] = self.integrate_web_data(url)
        
        return results
    
    def close(self):
        """Schließt die Datenbankverbindung."""
        if self.session:
            self.session.close()
        if self.driver:
            self.driver.close()
        logger.info("Neo4j connection closed")

def main():
    """Hauptfunktion für Standalone-Ausführung."""
    collector = KnowledgeGraphCollector()
    
    try:
        logger.info("Starting knowledge graph collection cycle...")
        results = collector.run_collection_cycle()
        
        logger.info("Collection results:")
        for key, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"  {key}: {status}")
        
        successful_operations = sum(results.values())
        total_operations = len(results)
        logger.info(f"Completed {successful_operations}/{total_operations} operations successfully")
        
    except Exception as e:
        logger.error(f"Collection cycle failed: {e}")
    finally:
        collector.close()

if __name__ == "__main__":
    main()

def integrate_bedrock_knowledge(self, kb_id: str):
    """Verknüpft AWS Bedrock KB mit Graph"""
    self.driver.execute_query(
        "CREATE (k:KnowledgeBase {id: $id}) RETURN k",
        {"id": kb_id}
    )