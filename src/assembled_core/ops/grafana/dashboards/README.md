# Assembled Trading — Grafana Dashboards

These JSON files are Grafana dashboard definitions (schema version 36, Grafana ≥9.0).

## Dashboards

| File | Purpose |
|------|---------|
| `trading_overview.json` | Morning health check — kill-switch, orders, vol, drift |
| `slippage_analysis.json` | Execution quality — slippage histograms + P95/P50 |
| `order_flow.json` | Order/fill/rejection counters per strategy |
| `drift_monitor.json` | Feature drift PSI trend over time |

## How to Import

1. Open Grafana → Dashboards → Import
2. Upload the JSON file (or paste contents)
3. Select your Prometheus datasource
4. Save

Or use Grafana's provisioning (recommended for reproducibility):

```yaml
# grafana/provisioning/dashboards/assembled.yml
apiVersion: 1
providers:
  - name: assembled-trading
    type: file
    options:
      path: /path/to/src/assembled_core/ops/grafana/dashboards
```

## Metric Source

Metrics are written by `ops/metrics_exporter.py` to `output/metrics/assembled.prom`.
Configure Prometheus to scrape that file — see `../prometheus_config_example.yml`.

## Local Quick-Start (Docker)

```bash
# Start Prometheus + Grafana
docker run -d -p 9090:9090 \
  -v $(pwd)/src/assembled_core/ops/grafana/prometheus_config_example.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

docker run -d -p 3000:3000 grafana/grafana
```

Then import dashboards at http://localhost:3000 (admin/admin).
