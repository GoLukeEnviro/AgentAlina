#!/usr/bin/env python3
"""
Validierung der Umgebungskonfiguration für AgentAlina
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Füge das Projekt-Root-Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from config.env_config import (
        get_api_key, has_api_key, get_neo4j_config, 
        get_api_urls, get_trading_config, validate_configuration,
        print_configuration_status
    )
except ImportError:
    print("Warning: env_config module not found. Using fallback validation.")
    
    def get_api_key(key_name: str) -> Optional[str]:
        return os.getenv(key_name)
    
    def has_api_key(key_name: str) -> bool:
        return bool(os.getenv(key_name))
    
    def get_neo4j_config() -> Dict[str, Any]:
        return {
            'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
            'password': os.getenv('NEO4J_PASSWORD', '')
        }
    
    def get_api_urls() -> Dict[str, str]:
        return {
            'finance_api': os.getenv('FINANCE_API_URL', 'https://api.example.com/finance'),
            'news_api': os.getenv('NEWS_API_URL', 'https://api.example.com/news'),
            'weather_api': os.getenv('WEATHER_API_URL', 'https://api.example.com/weather')
        }
    
    def get_trading_config() -> Dict[str, Any]:
        return {
            'exchange': os.getenv('TRADING_EXCHANGE', 'binance'),
            'api_key': os.getenv('BINANCE_API_KEY', ''),
            'secret_key': os.getenv('BINANCE_SECRET_KEY', ''),
            'testnet': os.getenv('TRADING_TESTNET', 'true').lower() == 'true'
        }

class ConfigurationValidator:
    """Klasse zur Validierung der Umgebungskonfiguration."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        
        # Definiere erforderliche und optionale Konfigurationen
        self.required_configs = {
            'NEO4J_URI': 'Neo4j Database URI',
            'NEO4J_USERNAME': 'Neo4j Username',
            'NEO4J_PASSWORD': 'Neo4j Password'
        }
        
        self.optional_configs = {
            'NEWS_API_KEY': 'News API Key für Nachrichtenabruf',
            'WEATHER_API_KEY': 'Weather API Key für Wetterdaten',
            'GITHUB_TOKEN': 'GitHub Token für Repository-Zugriff',
            'OPENAI_API_KEY': 'OpenAI API Key für GPT-Modelle',
            'DEEPSEEK_API_KEY': 'DeepSeek API Key für alternative LLM',
            'ANTHROPIC_API_KEY': 'Anthropic API Key für Claude-Modelle',
            'TAVILY_API_KEY': 'Tavily API Key für Web-Suche',
            'BRAVE_SEARCH_API_KEY': 'Brave Search API Key',
            'HYPERBROWSER_API_KEY': 'Hyperbrowser API Key für Web-Scraping',
            'BINANCE_API_KEY': 'Binance API Key für Trading',
            'BINANCE_SECRET_KEY': 'Binance Secret Key für Trading'
        }
        
        self.api_urls = {
            'FINANCE_API_URL': 'Finance API Endpoint',
            'NEWS_API_URL': 'News API Endpoint',
            'WEATHER_API_URL': 'Weather API Endpoint',
            'CRYPTO_API_URL': 'Cryptocurrency API Endpoint',
            'BINANCE_API_URL': 'Binance API Endpoint',
            'HYPERBROWSER_API_URL': 'Hyperbrowser API Endpoint'
        }
        
        self.trading_configs = {
            'TRADING_EXCHANGE': 'Trading Exchange (default: binance)',
            'TRADING_TESTNET': 'Use Trading Testnet (default: true)',
            'TRADING_MAX_POSITION_SIZE': 'Maximum Position Size',
            'TRADING_RISK_LEVEL': 'Risk Level (low/medium/high)'
        }
    
    def validate_required_configs(self) -> bool:
        """Validiere erforderliche Konfigurationen."""
        all_valid = True
        
        for config_key, description in self.required_configs.items():
            value = os.getenv(config_key)
            if not value:
                self.errors.append(f"FEHLER: {config_key} ({description}) ist erforderlich aber nicht gesetzt")
                all_valid = False
            elif value.strip() == '':
                self.errors.append(f"FEHLER: {config_key} ({description}) ist leer")
                all_valid = False
            else:
                self.info.append(f"✓ {config_key} ist gesetzt")
        
        return all_valid
    
    def validate_optional_configs(self) -> None:
        """Validiere optionale Konfigurationen."""
        for config_key, description in self.optional_configs.items():
            value = os.getenv(config_key)
            if not value:
                self.warnings.append(f"WARNUNG: {config_key} ({description}) ist nicht gesetzt - einige Features könnten nicht verfügbar sein")
            elif value.strip() == '':
                self.warnings.append(f"WARNUNG: {config_key} ({description}) ist leer")
            else:
                self.info.append(f"✓ {config_key} ist gesetzt")
    
    def validate_neo4j_config(self) -> bool:
        """Validiere Neo4j-spezifische Konfiguration."""
        try:
            config = get_neo4j_config()
            
            # URI Validierung
            uri = config.get('uri', '')
            if not uri:
                self.errors.append("FEHLER: Neo4j URI ist nicht gesetzt")
                return False
            
            if not (uri.startswith('bolt://') or uri.startswith('neo4j://') or uri.startswith('neo4j+s://')):
                self.warnings.append(f"WARNUNG: Neo4j URI '{uri}' verwendet möglicherweise ein ungewöhnliches Protokoll")
            
            # Username/Password Validierung
            username = config.get('username', '')
            password = config.get('password', '')
            
            if not username:
                self.errors.append("FEHLER: Neo4j Username ist nicht gesetzt")
                return False
            
            if not password:
                self.errors.append("FEHLER: Neo4j Password ist nicht gesetzt")
                return False
            
            self.info.append(f"✓ Neo4j Konfiguration: {uri} (User: {username})")
            return True
            
        except Exception as e:
            self.errors.append(f"FEHLER: Fehler beim Validieren der Neo4j-Konfiguration: {e}")
            return False
    
    def validate_api_urls(self) -> None:
        """Validiere API URLs."""
        try:
            urls = get_api_urls()
            
            for url_key, description in self.api_urls.items():
                url = os.getenv(url_key)
                if url:
                    if not (url.startswith('http://') or url.startswith('https://')):
                        self.warnings.append(f"WARNUNG: {url_key} ({description}) ist keine gültige HTTP(S) URL: {url}")
                    else:
                        self.info.append(f"✓ {url_key} ist gesetzt: {url}")
                else:
                    self.warnings.append(f"WARNUNG: {url_key} ({description}) ist nicht gesetzt - Standard-URL wird verwendet")
                    
        except Exception as e:
            self.warnings.append(f"WARNUNG: Fehler beim Validieren der API URLs: {e}")
    
    def validate_trading_config(self) -> None:
        """Validiere Trading-Konfiguration."""
        try:
            config = get_trading_config()
            
            # Exchange Validierung
            exchange = config.get('exchange', 'binance')
            supported_exchanges = ['binance', 'coinbase', 'kraken', 'bitfinex']
            if exchange not in supported_exchanges:
                self.warnings.append(f"WARNUNG: Trading Exchange '{exchange}' ist möglicherweise nicht unterstützt")
            else:
                self.info.append(f"✓ Trading Exchange: {exchange}")
            
            # API Keys für Trading
            api_key = config.get('api_key', '')
            secret_key = config.get('secret_key', '')
            
            if api_key and secret_key:
                self.info.append("✓ Trading API Keys sind gesetzt")
                
                # Testnet Validierung
                testnet = config.get('testnet', True)
                if testnet:
                    self.info.append("✓ Trading im Testnet-Modus aktiviert (sicher für Tests)")
                else:
                    self.warnings.append("WARNUNG: Trading im Live-Modus aktiviert - Vorsicht bei echten Trades!")
            else:
                self.warnings.append("WARNUNG: Trading API Keys sind nicht vollständig gesetzt - Trading-Features nicht verfügbar")
            
            # Zusätzliche Trading-Parameter
            for config_key, description in self.trading_configs.items():
                value = os.getenv(config_key)
                if value:
                    self.info.append(f"✓ {config_key}: {value}")
                    
        except Exception as e:
            self.warnings.append(f"WARNUNG: Fehler beim Validieren der Trading-Konfiguration: {e}")
    
    def validate_file_permissions(self) -> None:
        """Validiere Dateiberechtigungen für wichtige Verzeichnisse."""
        important_dirs = [
            'logs',
            'data',
            'config',
            'agents',
            'services',
            'tests'
        ]
        
        project_root = Path(__file__).parent.parent
        
        for dir_name in important_dirs:
            dir_path = project_root / dir_name
            
            if dir_path.exists():
                if os.access(dir_path, os.R_OK):
                    self.info.append(f"✓ Verzeichnis {dir_name} ist lesbar")
                else:
                    self.errors.append(f"FEHLER: Verzeichnis {dir_name} ist nicht lesbar")
                
                if os.access(dir_path, os.W_OK):
                    self.info.append(f"✓ Verzeichnis {dir_name} ist schreibbar")
                else:
                    self.warnings.append(f"WARNUNG: Verzeichnis {dir_name} ist nicht schreibbar")
            else:
                self.warnings.append(f"WARNUNG: Verzeichnis {dir_name} existiert nicht")
    
    def validate_python_dependencies(self) -> None:
        """Validiere wichtige Python-Abhängigkeiten."""
        required_packages = [
            'neo4j',
            'requests',
            'beautifulsoup4',
            'pandas',
            'numpy',
            'scikit-learn',
            'aiohttp',
            'asyncio'
        ]
        
        optional_packages = [
            'ccxt',  # für Trading
            'openai',  # für OpenAI API
            'anthropic',  # für Claude API
            'tavily-python',  # für Web-Suche
            'pytest',  # für Tests
            'docker'  # für Container-Management
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                self.info.append(f"✓ Erforderliches Paket {package} ist installiert")
            except ImportError:
                self.errors.append(f"FEHLER: Erforderliches Paket {package} ist nicht installiert")
        
        for package in optional_packages:
            try:
                __import__(package)
                self.info.append(f"✓ Optionales Paket {package} ist installiert")
            except ImportError:
                self.warnings.append(f"WARNUNG: Optionales Paket {package} ist nicht installiert")
    
    def validate_environment_file(self) -> None:
        """Validiere .env Datei."""
        project_root = Path(__file__).parent.parent
        env_file = project_root / '.env'
        env_example_file = project_root / '.env.example'
        
        if env_file.exists():
            self.info.append("✓ .env Datei gefunden")
            
            # Prüfe ob .env Datei lesbar ist
            try:
                with open(env_file, 'r') as f:
                    content = f.read()
                    if len(content.strip()) == 0:
                        self.warnings.append("WARNUNG: .env Datei ist leer")
                    else:
                        lines = [line for line in content.split('\n') if line.strip() and not line.startswith('#')]
                        self.info.append(f"✓ .env Datei enthält {len(lines)} Konfigurationszeilen")
            except Exception as e:
                self.errors.append(f"FEHLER: Kann .env Datei nicht lesen: {e}")
        else:
            self.warnings.append("WARNUNG: .env Datei nicht gefunden")
        
        if env_example_file.exists():
            self.info.append("✓ .env.example Datei gefunden (Vorlage verfügbar)")
        else:
            self.warnings.append("WARNUNG: .env.example Datei nicht gefunden")
    
    def run_full_validation(self) -> Tuple[bool, Dict[str, List[str]]]:
        """Führe vollständige Validierung durch."""
        self.errors.clear()
        self.warnings.clear()
        self.info.clear()
        
        print("🔍 Starte Konfigurationsvalidierung...\n")
        
        # Validiere verschiedene Bereiche
        required_valid = self.validate_required_configs()
        self.validate_optional_configs()
        neo4j_valid = self.validate_neo4j_config()
        self.validate_api_urls()
        self.validate_trading_config()
        self.validate_file_permissions()
        self.validate_python_dependencies()
        self.validate_environment_file()
        
        # Gesamtergebnis
        overall_valid = required_valid and neo4j_valid and len(self.errors) == 0
        
        results = {
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info
        }
        
        return overall_valid, results
    
    def print_validation_results(self, overall_valid: bool, results: Dict[str, List[str]]) -> None:
        """Drucke Validierungsergebnisse."""
        print("\n" + "="*80)
        print("📋 KONFIGURATIONSVALIDIERUNG ERGEBNISSE")
        print("="*80)
        
        # Fehler
        if results['errors']:
            print("\n❌ FEHLER:")
            for error in results['errors']:
                print(f"   {error}")
        
        # Warnungen
        if results['warnings']:
            print("\n⚠️  WARNUNGEN:")
            for warning in results['warnings']:
                print(f"   {warning}")
        
        # Informationen
        if results['info']:
            print("\n✅ ERFOLGREICH KONFIGURIERT:")
            for info in results['info']:
                print(f"   {info}")
        
        # Gesamtstatus
        print("\n" + "="*80)
        if overall_valid:
            print("🎉 KONFIGURATION GÜLTIG - AgentAlina kann gestartet werden!")
        else:
            print("❌ KONFIGURATION UNGÜLTIG - Bitte behebe die Fehler vor dem Start")
        print("="*80)
        
        # Statistiken
        print(f"\n📊 Zusammenfassung:")
        print(f"   Fehler: {len(results['errors'])}")
        print(f"   Warnungen: {len(results['warnings'])}")
        print(f"   Erfolgreich: {len(results['info'])}")

def create_sample_env_file() -> None:
    """Erstelle eine Beispiel-.env Datei."""
    project_root = Path(__file__).parent.parent
    env_file = project_root / '.env'
    
    if env_file.exists():
        print(f"⚠️  .env Datei existiert bereits: {env_file}")
        return
    
    sample_content = """
# AgentAlina Umgebungskonfiguration
# Kopiere diese Datei nach .env und fülle die Werte aus

# === NEO4J DATENBANK (ERFORDERLICH) ===
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here

# === API KEYS (OPTIONAL) ===
NEWS_API_KEY=your_news_api_key_here
WEATHER_API_KEY=your_weather_api_key_here
GITHUB_TOKEN=your_github_token_here
OPENAI_API_KEY=your_openai_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
BRAVE_SEARCH_API_KEY=your_brave_search_api_key_here
HYPERBROWSER_API_KEY=your_hyperbrowser_api_key_here

# === TRADING KONFIGURATION (OPTIONAL) ===
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
TRADING_EXCHANGE=binance
TRADING_TESTNET=true
TRADING_MAX_POSITION_SIZE=1000
TRADING_RISK_LEVEL=low

# === API URLS (OPTIONAL - Standards werden verwendet) ===
FINANCE_API_URL=https://api.example.com/finance
NEWS_API_URL=https://api.example.com/news
WEATHER_API_URL=https://api.example.com/weather
CRYPTO_API_URL=https://api.example.com/crypto
BINANCE_API_URL=https://api.binance.com
HYPERBROWSER_API_URL=https://api.hyperbrowser.com
""".strip()
    
    try:
        with open(env_file, 'w') as f:
            f.write(sample_content)
        print(f"✅ Beispiel-.env Datei erstellt: {env_file}")
        print("📝 Bitte bearbeite die Datei und füge deine API-Keys hinzu")
    except Exception as e:
        print(f"❌ Fehler beim Erstellen der .env Datei: {e}")

def main():
    """Hauptfunktion für Konfigurationsvalidierung."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AgentAlina Konfigurationsvalidierung')
    parser.add_argument('--create-env', action='store_true', 
                       help='Erstelle eine Beispiel-.env Datei')
    parser.add_argument('--json', action='store_true',
                       help='Ausgabe im JSON-Format')
    parser.add_argument('--quiet', action='store_true',
                       help='Nur Fehler ausgeben')
    
    args = parser.parse_args()
    
    if args.create_env:
        create_sample_env_file()
        return
    
    validator = ConfigurationValidator()
    overall_valid, results = validator.run_full_validation()
    
    if args.json:
        # JSON-Ausgabe für programmatische Nutzung
        output = {
            'valid': overall_valid,
            'errors': results['errors'],
            'warnings': results['warnings'],
            'info': results['info']
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    elif args.quiet:
        # Nur Fehler ausgeben
        if results['errors']:
            for error in results['errors']:
                print(error)
        if not overall_valid:
            sys.exit(1)
    else:
        # Vollständige Ausgabe
        validator.print_validation_results(overall_valid, results)
    
    # Exit Code setzen
    if not overall_valid:
        sys.exit(1)

if __name__ == '__main__':
    main()