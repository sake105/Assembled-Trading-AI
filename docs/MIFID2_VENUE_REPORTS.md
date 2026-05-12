# MiFID-II RTS-27 / RTS-28 Best-Execution Reports

> Audit C4-092 — when (and only when) the firm operates as an
> investment firm with discretion over client orders (i.e. *not*
> purely own-account), MiFID-II requires two annual reports:
>
> - **RTS-27** — quality-of-execution data published by *execution
>   venues* (we are not a venue; this section is informational).
> - **RTS-28** — top-five-venue summary published by *investment
>   firms* (we would be here if we route client flow).
>
> **Status today:** the firm trades own-account through a single
> broker (Alpaca paper today, Alpaca / IBKR / Lynx in the future
> live setup). RTS-27/28 do not apply. The skeleton below activates
> if we ever route flow for a third party.

## 1. When this becomes mandatory

- Investment-firm status under MiFID-II Art. 4(1)(1) — typically
  triggered by client-money handling, third-party flow, or
  discretionary management.
- Stays voluntary for pure Eigengeschäft.
- ESMA suspended quarterly RTS-27 publication for trading venues
  via Delegated Regulation 2021/1697, but RTS-28 (firm-side,
  annual, top-five venues) remains active.

## 2. RTS-28 annual report — structure

For each financial-instrument class (we only trade equities today,
class **Equities — Shares & Depository Receipts > tick-size liquidity
band 5-6**), the report MUST contain:

| Field | Source today | Pre-fill from |
|---|---|---|
| Top 5 execution venues by trading volume | Alpaca (single venue) | `output/ops/orders_audit.jsonl` aggregated annually |
| % of orders routed to each venue | 100% Alpaca | dito |
| % of passive vs aggressive orders | n/a — we don't tag this today | needs order-tagging in Order-Lifecycle |
| % of directed orders | 100% directed (we choose venue) | dito |
| Summary of analysis & conclusions on execution quality (free text) | weekly broker-fill-quality review | needed: a 1-page memo |
| Use of execution data published under RTS-27 | "We do not analyse RTS-27 data because we route to a single venue and rely on the broker's own best-ex confirmation." | static |

## 3. Template skeleton

```markdown
# Top-5 Execution Venue Report (RTS-28) — FY <YYYY>

**Firm:** <name>
**Reporting period:** YYYY-01-01 → YYYY-12-31
**Instrument class:** Equities — Shares & DRs (LIS band 5-6)

## Venues
| Rank | Venue (LEI / MIC) | % volume | % orders | % passive | % aggressive | % directed | Connected via |
|---|---|---|---|---|---|---|---|
| 1 | <MIC> (<LEI>) | 100.0 | 100.0 | XX | YY | 100 | <broker> |

## Conclusion
<one-paragraph rationale, e.g. "Single venue chosen on the basis of
fee structure, fill quality observed over Q1-Q4, and connectivity
reliability. No material change recommended for the upcoming year.">

## RTS-27 data
<one-line statement>

## Signed
<operator>, <date>
```

## 4. Order tagging required before this can be auto-generated

Today our order audit log captures `(symbol, side, qty, price,
source, route)`. To populate RTS-28 we additionally need:

- `liquidity_flag`: `passive | aggressive` — derived from whether
  the order sat in the book before filling.
- `directed_flag`: `directed | non-directed` — we are 100% directed
  (we pick the venue) for as long as we route to a single broker.
- `venue_lei` and `venue_mic` — the legal/MIC identifier of the
  execution venue (Alpaca's IEX route: MIC `IEXG`; Polygon-connected
  venues vary).

These are tagging tasks for `execution/order_lifecycle.py` —
*not implemented yet*. Tracked as future audit follow-up if we
ever cross into investment-firm status.

## 5. Distribution

When required: publish the RTS-28 report on the firm website by
30 April of the following year (Delegated Regulation 2017/576
Art. 3(3)). For internal-use-only own-account today, no
distribution requirement.

## 6. What this template is NOT

- Not RTS-27 (we are not a venue).
- Not a substitute for the broker's own best-execution policy
  acknowledgement.
- Not legal advice.
