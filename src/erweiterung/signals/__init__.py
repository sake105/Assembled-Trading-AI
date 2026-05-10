"""Neuartige Signal-Generatoren (auf altdata + Preisen aufbauend).

Alle Signale sind PIT-sicher: keine Information aus t > current_date wird
verwendet. Wo eine Verzögerung notwendig ist (z. B. SEC-Filings ~T+1),
ist die Verzögerung **explizit im Signal codiert**, nicht als Annahme.
"""
