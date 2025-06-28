#!/usr/bin/env python3
"""
Environment Configuration Module

Lädt und verwaltet Umgebungsvariablen für Neo4j, Feedback und API-URLs.
Stellt zentrale Konfigurationsschnittstellen für andere Module bereit.
"""

import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Lade .env Datei
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j Konfiguration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')

# API URLs
FINANCE_API_URL = os.getenv('FINANCE_API_URL', 'https://api.coindesk.com/v1/bpi/currentprice.json')
NEWS_API_URL = os.getenv('NEWS_API_URL', 'https://newsapi.org/v2/top-headlines')
WEATHER_API_URL = os.getenv('WEATHER_API_URL', 'https://api.openweathermap.org/data/2.5/weather')
CRYPTO_API_URL = os.getenv('CRYPTO_API_URL', 'https://api.coingecko.com/api/v3/simple/price')

# API Keys
API_KEYS = {
    'NEWS': os.getenv('NEWS_API_KEY'),
    'WEATHER': os.getenv('OPENWEATHER_API_KEY'),
    'GITHUB': os.getenv('GITHUB_API_KEY'),
    'OPENAI': os.getenv('OPENAI_API_KEY'),
    'DEEPSEEK': os.getenv('DEEPSEEK_API_KEY'),
    'ANTHROPIC': os.getenv('ANTHROPIC_API_KEY'),
    'TAVILY': os.getenv('TAVILY_API_KEY'),
    'BRAVE_SEARCH': os.getenv('BRAVE_SEARCH_API_KEY'),
    'HYPERBROWSER_1': os.getenv('HYPERBROWSER_API_KEY_1'),
    'HYPERBROWSER_2': os.getenv('HYPERBROWSER_API_KEY_2')
}

# Trading Configuration
BINANCE_WS_URL = os.getenv('BINANCE_WS_URL', 'wss://ws-api.binance.com:443/ws-api/v3')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

# Hyperbrowser Configuration
HYPERBROWSER_API_URL = os.getenv('HYPERBROWSER_API_URL', 'https://app.hyperbrowser.ai/')

def get_api_key(service: str) -> Optional[str]:
    """Holt einen API-Key für einen bestimmten Service.
    
    Args:
        service: Name des Services (z.B. 'NEWS', 'WEATHER', 'GITHUB')
        
    Returns:
        API-Key als String oder None falls nicht verfügbar
    """
    return API_KEYS.get(service.upper())

def has_api_key(service: str) -> bool:
    """Prüft ob ein API-Key für einen Service verfügbar ist.
    
    Args:
        service: Name des Services
        
    Returns:
        True falls API-Key verfügbar, False sonst
    """
    key = get_api_key(service)
    return key is not None and key.strip() != ''

def get_neo4j_config() -> Dict[str, str]:
    """Gibt die Neo4j Konfiguration zurück.
    
    Returns:
        Dictionary mit Neo4j Verbindungsparametern
    """
    return {
        'uri': NEO4J_URI,
        'username': NEO4J_USERNAME,
        'password': NEO4J_PASSWORD
    }

def get_api_urls() -> Dict[str, str]:
    """Gibt alle konfigurierten API URLs zurück.
    
    Returns:
        Dictionary mit API URLs
    """
    return {
        'finance': FINANCE_API_URL,
        'news': NEWS_API_URL,
        'weather': WEATHER_API_URL,
        'crypto': CRYPTO_API_URL,
        'binance_ws': BINANCE_WS_URL,
        'hyperbrowser': HYPERBROWSER_API_URL
    }

def get_trading_config() -> Dict[str, Optional[str]]:
    """Gibt die Trading-Konfiguration zurück.
    
    Returns:
        Dictionary mit Trading-Parametern
    """
    return {
        'binance_ws_url': BINANCE_WS_URL,
        'binance_api_key': BINANCE_API_KEY,
        'binance_secret_key': BINANCE_SECRET_KEY
    }

def validate_configuration() -> Dict[str, Any]:
    """Validiert die gesamte Konfiguration.
    
    Returns:
        Dictionary mit Validierungsergebnissen
    """
    validation_results = {
        'neo4j': {
            'uri_set': bool(NEO4J_URI and NEO4J_URI != 'bolt://localhost:7687'),
            'username_set': bool(NEO4J_USERNAME and NEO4J_USERNAME != 'neo4j'),
            'password_set': bool(NEO4J_PASSWORD and NEO4J_PASSWORD != 'password')
        },
        'api_keys': {},
        'api_urls': {
            'finance': bool(FINANCE_API_URL),
            'news': bool(NEWS_API_URL),
            'weather': bool(WEATHER_API_URL),
            'crypto': bool(CRYPTO_API_URL)
        },
        'trading': {
            'binance_ws_url': bool(BINANCE_WS_URL),
            'binance_api_key': bool(BINANCE_API_KEY),
            'binance_secret_key': bool(BINANCE_SECRET_KEY)
        }
    }
    
    # Validiere API Keys
    for service, key in API_KEYS.items():
        validation_results['api_keys'][service.lower()] = bool(key and key.strip())
    
    return validation_results

def print_configuration_status():
    """Gibt den aktuellen Konfigurationsstatus aus."""
    logger.info("=== Environment Configuration Status ===")
    
    # Neo4j Status
    logger.info(f"Neo4j URI: {'✓' if NEO4J_URI else '✗'} {NEO4J_URI}")
    logger.info(f"Neo4j Username: {'✓' if NEO4J_USERNAME else '✗'} {NEO4J_USERNAME}")
    logger.info(f"Neo4j Password: {'✓' if NEO4J_PASSWORD else '✗'} {'*' * len(NEO4J_PASSWORD) if NEO4J_PASSWORD else 'Not set'}")
    
    # API Keys Status
    logger.info("\nAPI Keys Status:")
    for service, key in API_KEYS.items():
        status = '✓' if key and key.strip() else '✗'
        masked_key = f"{key[:8]}...{key[-4:]}" if key and len(key) > 12 else 'Not set'
        logger.info(f"  {service}: {status} {masked_key}")
    
    # API URLs Status
    logger.info("\nAPI URLs:")
    urls = get_api_urls()
    for name, url in urls.items():
        logger.info(f"  {name}: {url}")
    
    # Validation Summary
    validation = validate_configuration()
    total_configs = sum([
        len(validation['neo4j']),
        len(validation['api_keys']),
        len(validation['api_urls']),
        len(validation['trading'])
    ])
    
    valid_configs = sum([
        sum(validation['neo4j'].values()),
        sum(validation['api_keys'].values()),
        sum(validation['api_urls'].values()),
        sum(validation['trading'].values())
    ])
    
    logger.info(f"\nConfiguration Summary: {valid_configs}/{total_configs} items configured")

# Initialisierung beim Import
if __name__ == "__main__":
    print_configuration_status()
else:
    logger.info("Environment configuration loaded successfully")
    logger.info(f"Neo4j: {NEO4J_URI}")
    logger.info(f"API Keys configured: {sum(1 for k in API_KEYS.values() if k)}")