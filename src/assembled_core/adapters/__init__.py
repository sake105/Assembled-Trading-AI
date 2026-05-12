"""Adapters — concrete implementations of ports.

- ``adapters.inbound``  — driving adapters: CLI, REST, scheduler, notebook
- ``adapters.outbound`` — driven adapters: broker, factor store, audit logger

Adapters are the only modules permitted to import third-party
integration libraries (httpx, fastapi, alpaca-py, polygon, yfinance).
Domain code is barred from importing those directly.
"""
