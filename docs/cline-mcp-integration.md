# Cline MCP Integration

## Übersicht

Die Cline IDE wurde erfolgreich mit dem AgentAlina MCP Server integriert. Diese Integration ermöglicht es Cline, direkt auf die Tools und Funktionen des AgentAlina-Systems zuzugreifen.

## Konfiguration

### Konfigurationsdatei
```json
{
  "mcpServers": {
    "alina-mcp-server": {
      "autoApprove": [],
      "disabled": false,
      "timeout": 60,
      "type": "stdio",
      "command": "python3",
      "args": [
        "/root/AgentAlina/services/alina-mcp/src/server.py"
      ],
      "env": {}
    }
  }
}
```

**Speicherort:** `/root/.trae-server/data/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`

### Kommunikation
- **Protokoll:** JSON-RPC 2.0
- **Transport:** Standard I/O (stdio)
- **Timeout:** 60 Sekunden

## Verfügbare Tools

Der AgentAlina MCP Server stellt folgende Tools zur Verfügung:

1. **toolloader** - Plugin-Management und -Ausführung
2. **memory** - Knowledge Graph Memory für Kontext und Logs
3. **trading** - CCXT Bot-Ausführung und P&L-Monitoring
4. **monitor** - Metriken-Export zu Prometheus/Grafana
5. **optimizer** - Output-Evaluierung und Hyperparameter-Anpassung

## Test der Integration

### Funktionstest
```bash
echo '{"jsonrpc":"2.0","method":"list_tools","id":1}' | python3 /root/AgentAlina/services/alina-mcp/src/server.py
```

### Erwartete Antwort
Der Server sollte eine JSON-Antwort mit allen verfügbaren Tools und deren Schemas zurückgeben.

## Nutzung in Cline

Nach der Konfiguration kann Cline automatisch auf die MCP-Tools zugreifen:
- Tools werden in der Cline-Oberfläche angezeigt
- Direkte Ausführung von AgentAlina-Funktionen möglich
- Nahtlose Integration in den Entwicklungsworkflow

## Troubleshooting

### Häufige Probleme
1. **Server startet nicht:** Überprüfen Sie den Python-Pfad und die Abhängigkeiten
2. **Timeout-Fehler:** Erhöhen Sie den Timeout-Wert in der Konfiguration
3. **Tool nicht verfügbar:** Stellen Sie sicher, dass der MCP Server läuft

### Logs überprüfen
```bash
# Server-Logs anzeigen
python3 /root/AgentAlina/services/alina-mcp/src/server.py 2>&1 | tee server.log
```

## Status

✅ **Integration erfolgreich konfiguriert**  
✅ **Server funktionsfähig**  
✅ **Tools verfügbar**  
✅ **Kommunikation getestet**  

---

*Erstellt am: 27. Juni 2025*  
*Letzte Aktualisierung: 27. Juni 2025*