"""News-Impact-Modeling: wie News Maerkte bewegen und wie wir das modellieren.

Integration-Status
------------------
Diese Module sind eine **Research-Toolbox**, nicht der Live-Trading-Pfad.
Live-Tilt laeuft ueber das separate Modul ``erweiterung.signals.news_tilt_builder``,
das in ``LiveDecisionEngine`` via ``attach_news_tilt_scores()`` haengt.

Die folgenden Module sind getestet (siehe tests/erweiterung/test_news_impact_with_real_data.py)
aber NICHT in LiveDecisionEngine integriert — sie sind fuer Offline-Analyse + Backtest-Research:

- ``decay_model``           : Exponential-decay return-impact nach News-Shocks
- ``news_surprise``         : Erwartetes vs realisiertes Sentiment = Surprise
- ``reactivity_index``      : Asset-spezifische News-Elastizitaet
- ``cross_asset_spillover`` : Mention-basierte Asset-Kopplungs-Matrix
- ``topic_drift``           : Themen-Verteilungs-Change-Point-Detection
- ``time_of_day_impact``    : Market-Hours vs Pre/Aftermarket News-Wirkung
- ``event_clustering``      : Embedding-basierte News-Cluster
- ``sentiment_divergence``  : Traditional-News vs Social-Media-Sentiment-Gap

Hinweis fuer Live-Pfad
----------------------
Wenn du eines dieser Module in den Live-Pfad ziehen willst:
1. Output in pd.Series(symbol -> score) konvertieren
2. ``engine.attach_news_tilt_scores(scores)`` aufrufen
3. Mit ``LiveEngineConfig(enable_news_tilt=True, news_tilt_strength=...)`` aktivieren

Composite-Geo-Stress sollte zuerst dort sein — Edge ist statistisch nicht signifikant
auf 19y (siehe scripts/erweiterung/run_19y_master_composite.py). News-basierte Tilts
brauchen laengere News-History als die aktuell 5 Monate.
"""
