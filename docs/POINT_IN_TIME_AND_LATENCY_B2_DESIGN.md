POINT-IN-TIME SAFETY AND LATENCY FOR ALT-DATA FACTORS (B2)
===========================================================

1. Overview and Motivation
--------------------------

This document defines the point-in-time (PIT) and latency rules for
alt-data driven factors in Phase B2 (news, macro, and related event
features). The goal is to prevent look-ahead bias by ensuring that
features for a given backtest date T only use information that is
actually known as of T.

We focus on:

- Alt-data domains such as:
  - Insider trades
  - Congress / politician trades
  - Earnings events (already partially handled in B1)
  - News sentiment and macro regimes (B2)
  - Shipping and other alternative flows (future)
- Explicit modelling of:
  - event_date: when something happens in the real world
  - disclosure_date: when the event becomes observable to us
    (for example, when a filing is published or when a news item
    hits the feed)
- A simple, consistent rule:
  - For backtest date T we may only use events with
    ``disclosure_date <= T`` when building factors/features.


2. Current State of Alt-Data Modules
------------------------------------

2.1 Earnings and Insider Factors (B1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module
``src/assembled_core/features/altdata_earnings_insider_factors.py``
implements earnings and insider factors. Key functions:

- ``build_earnings_surprise_factors(events_earnings, prices, ...)``
- ``build_insider_activity_factors(events_insider, prices, ...)``
  (see full module for details)

Characteristics:

- Inputs:
  - ``events_earnings`` and ``events_insider`` DataFrames with
    at least ``timestamp``, ``symbol`` and ``event_type``.
  - ``prices`` with ``timestamp``, ``symbol``, ``close``.
- Time handling:
  - Timestamps are converted to UTC-aware ``datetime``.
  - ``merge_asof`` is used to align events up to each price
    timestamp (for example "last earnings surprise" up to date T).
- Limitations for PIT:
  - The code implicitly assumes that the event timestamp can be
    used both as when the event happens and when it becomes known.
  - There is no explicit ``disclosure_date`` or reporting latency
    in the current contracts.

2.2 News and Macro Factors (B2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module
``src/assembled_core/features/altdata_news_macro_factors.py```
implements:

- ``build_news_sentiment_factors(news_sentiment_daily, prices, ...)``
- ``build_macro_regime_factors(macro_daily, prices, ...)`` (see
  full module)

Characteristics:

- Inputs:
  - ``news_sentiment_daily`` with ``timestamp``, ``sentiment_score``,
    ``sentiment_volume`` and optionally ``symbol``.
  - ``macro`` panels with daily macro regime indicators.
- Time handling:
  - Timestamps are normalised to UTC.
  - Rolling windows over ``sentiment_score`` and ``sentiment_volume``
    are used to compute:
    - rolling mean and trend
    - shock flags
    - rolling volume metrics
  - Merge is done via per-symbol or market-wide joins on daily
    timestamps.
- Limitations for PIT:
  - The daily timestamp is treated as if the sentiment is known
    for that day without explicit latency.
  - No explicit ``disclosure_date`` or delayed arrival of news.

2.3 Other Alt-Data Domains (Placeholder)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additional alt-data sources such as shipping congestion,
alternative credit indicators or high-frequency news are expected
in future phases. They will follow the same general pattern:

- Raw events with some event time (for example vessel arrival,
  congestion start, downgrade).
- A potentially different time when the information is observable
  to the strategy (for example report publication, data vendor
  delivery, batch ingestion).


3. Definitions: event_date vs. disclosure_date
----------------------------------------------

To reason about PIT correctness, we distinguish:

3.1 event_date
~~~~~~~~~~~~~~

- The time when the underlying real-world event happens.
- Examples:
  - The actual trade date of an insider purchase.
  - The date when a member of Congress executes a trade.
  - The time when an earnings call takes place.
  - The time when a ship departs or arrives.
  - The instant when a news item is first written.

This timestamp is often recorded in raw event feeds as
``event_time``, ``trade_date``, ``event_timestamp`` or similar.

3.2 disclosure_date
~~~~~~~~~~~~~~~~~~~

- The time when the event becomes observable to the model or
  strategy. This is when the information can first influence
  features, signals and decisions.
- Examples:
  - For insider trades:
    - The filing date when the Form 4 is published by the
      regulator (for example T+2 relative to trade date).
  - For Congress trades:
    - The date when a periodic transaction report (PTR) is
      published (often days or weeks after the actual trade).
  - For earnings:
    - The timestamp when the earnings announcement is published.
    - In many cases this is close to the event time, but still
      should be treated as disclosure.
  - For news:
    - The timestamp when the news article is available via the
      data vendor’s feed to the backend.
  - For shipping:
    - The timestamp when a congestion signal is computed and
      stored (for example daily snapshot after all port events
      for the day are ingested).

3.3 Latency and Effective Date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In many real-world feeds the disclosure date is:

- Explicitly provided (for example ``disclosure_date`` column).
- Or derived as:
  - ``event_date + reporting_lag``
  - or the ingestion time when the record first appears in our
    local snapshot.

For backtesting we define:

- **Effective date for features:** for any backtest date T, an
  event can only influence features if
  ``disclosure_date <= T``.

This rule ensures that we never use information that was not yet
known at that date, preventing look-ahead bias due to reporting
lags or ingestion delays.


4. Target PIT Rule for Alt-Data Features
----------------------------------------

The core design rule for B2 is:

> For backtest date T, only events with
> ``disclosure_date <= T`` may contribute to features or
> factors used on T.

Concretely:

- Earnings and insider:
  - Replace uses of raw ``event_time`` as join key by a PIT-safe
    ``disclosure_date`` wherever available.
  - When only the filing date is known, this becomes the
    ``disclosure_date``.
- Congress trades:
  - Use the date when the transaction is officially disclosed as
    ``disclosure_date``.
  - If only trade date is available, introduce a conservative
    lag to simulate publication delay.
- News sentiment:
  - For daily sentiment panels, interpret the daily timestamp as
    the date on which all news up to the end of the day has been
    processed.
  - Use ``disclosure_date = date`` (end-of-day snapshot) and
    apply rolling windows only over dates ``<= T``.
- Shipping:
  - Treat any shipping congestion / flows as of daily
    snapshot date when the data is consolidated.
  - Only allow shipping factors for dates on or after the
    snapshot’s disclosure date.


5. Planned API and Data Contract Adjustments
--------------------------------------------

5.1 Event Data Contracts
~~~~~~~~~~~~~~~~~~~~~~~~

For all alt-data event DataFrames, B2 introduces the following
contract extensions:

- Required or recommended columns:
  - ``event_date`` (or a domain-specific field such as
    ``trade_date`` or ``announcement_time``).
  - ``disclosure_date``:
    - If provided by upstream, it is taken as-is.
    - If not provided, we derive it in a documented way (for
      example event_date + lag).

Module-level expectations:

- Earnings:
  - If upstream provides only an event timestamp used for
    announcements, we will treat that as ``disclosure_date`` and
    store it as such in the processed frame.
- Insider:
  - Prefer separate ``trade_date`` and ``filing_date``.
  - Use ``filing_date`` as ``disclosure_date``.
- Congress:
  - Prefer separate ``transaction_date`` and
    ``disclosure_date`` (publication).
  - If only one date is available, introduce a conservative lag
    (configuration-driven) to simulate disclosure timing.
- News:
  - For intraday events, we may keep the raw timestamp but expose
    a daily aggregated panel ``news_sentiment_daily`` with an
    effective ``disclosure_date`` equal to the date.

5.2 Feature Builder APIs
~~~~~~~~~~~~~~~~~~~~~~~~

Existing feature builder functions will gain optional PIT-related
parameters:

- ``as_of``: a single timestamp or date up to which events should
  be considered.
- ``max_disclosure_date``: alias for ``as_of`` where appropriate.

Examples:

- Earnings:

  - Current:

    - ``build_earnings_surprise_factors(events_earnings, prices, ...)``

  - Planned:

    - ``build_earnings_surprise_factors(events_earnings, prices, ..., as_of=None)``

    - If ``as_of`` is not ``None``, the function filters
      ``events_earnings`` to
      ``disclosure_date <= as_of`` before any merge.

- Insider / Congress:

  - New functions or updated signatures:
    - ``build_insider_activity_factors(events_insider, prices, ..., as_of=None)``
    - ``build_congress_activity_factors(events_congress, prices, ..., as_of=None)``

  - For each price row at date T, only events with
    ``disclosure_date <= T`` are considered in rolling
    aggregations.

- News / macro:

  - ``build_news_sentiment_factors(news_sentiment_daily, prices, ..., as_of=None)``
  - ``build_macro_regime_factors(macro_daily, prices, ..., as_of=None)``

  - When ``as_of`` is provided:
    - Filter daily panels to ``timestamp <= as_of``.
    - Ensure that any rolling window computations operate only on
      data up to ``as_of``.

5.3 Internal Implementation Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For all feature builders the internal implementation pattern will
be:

- Step 1: Normalise timestamps to UTC.
- Step 2: Apply explicit filters on ``disclosure_date`` (or
  derived effective date) before any grouping, rolling or
  ``merge_asof``.
- Step 3: Perform rolling aggregations and joins only on the
  filtered subset.

This keeps PIT enforcement simple and auditable.


6. Planned Test Set
-------------------

To validate PIT behaviour we will add synthetic tests, for example:

6.1 Earnings / Insider Latency Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Construct a small price panel for a single symbol with dates
  T0, T1, T2, T3.
- Create an earnings or insider event with:
  - ``event_date = T1``
  - ``disclosure_date = T3``
- Build factors:
  - For dates T0, T1, T2 all earnings / insider factors that
    depend on this event must remain NaN or zero (no effect
    before disclosure).
  - Only from T3 onwards may the factor reflect the event.

6.2 Congress Reporting Delay Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Similar setup with a ``transaction_date`` and
  ``disclosure_date = transaction_date + 10 days``.
- Verify that any congress factor is zero / NaN before the
  disclosure date and non-zero only afterwards.

6.3 News Latency Test
~~~~~~~~~~~~~~~~~~~~~

- Build a synthetic news sentiment panel where a large sentiment
  shock happens on date T1 but is declared as disclosed only on
  T2 in a mock ``disclosure_date`` column.
- Build factors with and without PIT enforcement:
  - Without PIT enforcement (reference): factors show the shock
    on T1.
  - With PIT enforcement: the shock only affects factors on T2
    and later.

6.4 Backtest Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Construct a minimal strategy that goes long when a specific
  alt-data factor exceeds a threshold.
- Create synthetic events with delayed disclosure such that:
  - The price move associated with the event occurs on event_date.
  - The factor only becomes available on disclosure_date.
- Verify:
  - A naive, non-PIT implementation would trade before the
    information is disclosed (look-ahead).
  - The PIT-safe implementation trades only on or after
    disclosure_date.


7. Integration in Backtests and EOD
-----------------------------------

7.1 Backtest Integration
~~~~~~~~~~~~~~~~~~~~~~~~

In portfolio backtests, the engine loops over time (for example
daily or intraday timestamps). For each step with date T:

- Any feature-building function that depends on alt-data must
  enforce:
  - ``disclosure_date <= T`` at the data access boundary.
- Practical patterns:
  - Pre-build daily panels of alt-data factors that are already
    PIT-safe (for example via dedicated pre-processing jobs).
  - Alternatively, when building features on-the-fly inside the
    backtest, pass ``as_of=T`` to the feature builders.

7.2 EOD Integration
~~~~~~~~~~~~~~~~~~~

For EOD jobs (for example ``scripts/run_eod_pipeline.py``), the
notion of ``as_of`` is:

- EOD date ``T_eod`` (for example "today" in UTC).

Feature-building during EOD should:

- Use ``as_of = T_eod`` consistently for all alt-data domains.
- Ensure that any alt-data snapshots written to disk for later
  reuse (for example factor panels or stored alt-data features)
  are marked with the effective ``as_of`` date in metadata so
  that backtests can reason about availability windows.


8. Implementation Plan (B2.1–B2.3)
----------------------------------

B2.1 – Event Contracts and Normalisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Introduce or standardise column names for:
  - ``event_date``
  - ``disclosure_date``
  - domain-specific aliases (for example ``filing_date``).
- Update the alt-data ingestion and transformation scripts so
  that each event feed emits a PIT-ready event table with:
  - explicit ``disclosure_date`` (either provided or derived).
- Add small unit tests that validate the mapping logic from raw
  fields to ``disclosure_date`` for each domain.

B2.2 – Feature Builder API Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Extend feature builder signatures in:
  - ``altdata_earnings_insider_factors.py``
  - ``altdata_news_macro_factors.py``
  - future shipping / congress modules
  with optional ``as_of`` (or equivalent) parameters.
- Enforce PIT filtering at the top of these functions:
  - Filter event tables to ``disclosure_date <= as_of`` when
    provided.
- Add synthetic tests described in section 6 to confirm that:
  - events with later disclosure do not affect factors before
    disclosure_date.

B2.3 – Backtest and EOD Wiring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Identify the places where alt-data features are built in:
  - backtest workflows
  - EOD workflows
  - research playbooks
- Ensure that:
  - backtests pass the current backtest date as ``as_of`` (when
    building features on-the-fly), or
  - precomputed factor panels used in backtests are documented as
    PIT-safe and built with correct ``disclosure_date`` handling.
- Add E2E smoke tests:
  - Small backtest using synthetic alt-data with delayed
    disclosure.
  - Confirm that trades only occur after ``disclosure_date``.

This plan keeps existing factor semantics intact while tightening
PIT guarantees and making latency assumptions explicit across
alt-data domains.


