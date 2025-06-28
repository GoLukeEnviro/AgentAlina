# Sicherheitshinweise für AgentElli

## Melden von Schwachstellen
Bitte melde Sicherheitslücken oder Schwachstellen verantwortungsvoll per E-Mail an das Projektteam oder über das GitHub-Issue-Tracking-System. Gib dabei möglichst viele Details an, damit wir das Problem schnell nachvollziehen und beheben können.

## Umgang mit Secrets
- Niemals echte Passwörter, API-Keys oder Tokens ins Repository pushen!
- Nutze `.env`-Dateien und halte diese aus dem Versionskontrollsystem heraus (siehe `.gitignore`).
- Beispiel-Umgebungsvariablen findest du in `.env.example`.

## Abhängigkeiten
- Halte alle Abhängigkeiten aktuell und prüfe regelmäßig auf CVEs.
- Nutze Tools wie `pip-audit` oder `npm audit` für Sicherheitsprüfungen.

## Weitere Hinweise
- Aktiviere 2FA für alle Projektmitglieder.
- Nutze SSH-Schlüssel für Serverzugriffe.
- Setze Token-Rotation und Monitoring ein.

Danke für deine Mithilfe zur Sicherheit von AgentElli!
