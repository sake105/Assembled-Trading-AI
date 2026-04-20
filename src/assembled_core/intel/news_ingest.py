"""GDELT 2.0 GKG batch ingest — 15-minute polling, no API key required."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import requests

from src.assembled_core.intel.models import NewsEvent, SourceTier

logger = logging.getLogger(__name__)

GDELT_LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
GDELT_SOURCE_ID = "gdelt"
GDELT_TIER = SourceTier.T2

# CAMEO/GDELT themes that are geopolitically relevant
RELEVANT_THEMES = {
    # Conflict / War
    "CRISISLEX_CRISISLEXREC", "TAX_MILITARY", "WB_1800_CONFLICT_AND_VIOLENCE",
    "CONFLICT", "WB_2512_WAR_CRIME", "PROTEST", "REBELLION",
    # Energy
    "ENV_OIL", "ENV_GAS", "ENERGY", "ENV_ENERGYCRISIS",
    # Sanctions
    "ECON_SANCTIONS", "TAX_WORLDBANK_POVERTY_ECONOMY_SANCTIONS",
    # Shipping / Chokepoints
    "TRANSPORT", "MARITIME", "MANMADE_DISASTER_MARITIME",
    # Cyber
    "CYBER_ATTACK", "TAX_CYBER",
    # Geopolitical crisis
    "CRISISLEX_T02_IMMINENT_DANGER_VIOLENCE", "WB_635_POLITICAL_INSTABILITY",
}

# GKG column indices (0-based, tab-separated)
_COL_RECORDID = 0
_COL_DATE = 1
_COL_SOURCE_COLLECTION = 2
_COL_SOURCE_NAME = 3
_COL_DOC_IDENTIFIER = 4
_COL_THEMES = 7
_COL_LOCATIONS = 9
_COL_PERSONS = 10
_COL_ORGANIZATIONS = 11
_COL_TONE = 12

_MAX_RECORDS_PER_BATCH = 500


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GdeltBatchRecord:
    record_id: str
    date_str: str
    source_name: str
    url: str
    themes: list[str] = field(default_factory=list)
    country_codes: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    persons: list[str] = field(default_factory=list)
    tone: float = 0.0  # first tone field, -100 to +100; negative = bad news
    batch_ts: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass
class GdeltFetchState:
    last_batch_url: str = ""
    last_fetch_ts: str = ""
    total_events_ingested: int = 0
    consecutive_failures: int = 0


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_lastupdate(text: str) -> str | None:
    """
    Parse lastupdate.txt and return the GKG CSV URL.

    The file has 3 lines (Events, Mentions, GKG). Each line is:
      <bytes> <hash> <url>
    We want line 3 (index 2), field 3 (index 2).
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        logger.warning("[WARN] lastupdate.txt has fewer than 3 lines: %d", len(lines))
        return None
    gkg_line = lines[2]
    parts = gkg_line.split()
    if len(parts) < 3:
        logger.warning("[WARN] GKG line in lastupdate.txt has fewer than 3 fields: %r", gkg_line)
        return None
    return parts[2]


def _parse_locations(raw: str) -> list[str]:
    """Extract unique 2-letter country codes from GDELT location field."""
    codes: list[str] = []
    if not raw:
        return codes
    for entry in raw.split(";"):
        parts = entry.split("#")
        # Format: Type#Name#CountryCode#Lat#Lon
        if len(parts) >= 3 and parts[2]:
            cc = parts[2].strip()
            if cc and cc not in codes:
                codes.append(cc)
    return codes


def _parse_tone(raw: str) -> float:
    """Parse the first (overall) tone value from GDELT tone field."""
    if not raw:
        return 0.0
    try:
        first = raw.split(",")[0]
        return float(first)
    except (ValueError, IndexError):
        return 0.0


def _parse_themes(raw: str) -> list[str]:
    """Parse semicolon-separated GDELT themes, stripping empty values."""
    if not raw:
        return []
    return [t.strip() for t in raw.split(";") if t.strip()]


def _parse_organizations(raw: str) -> list[str]:
    """Parse semicolon-separated organizations."""
    if not raw:
        return []
    return [o.strip() for o in raw.split(";") if o.strip()]


def _parse_persons(raw: str) -> list[str]:
    """Parse semicolon-separated V2Persons."""
    if not raw:
        return []
    return [p.strip() for p in raw.split(";") if p.strip()]


def _row_is_relevant(themes: list[str], tone: float) -> bool:
    """Return True if the row is geopolitically relevant or very negative."""
    if tone < -3.0:
        return True
    for theme in themes:
        if theme in RELEVANT_THEMES:
            return True
    return False


# ---------------------------------------------------------------------------
# Batch fetcher
# ---------------------------------------------------------------------------


def fetch_gkg_batch(
    gkg_url: str,
    timeout: int = 30,
    *,
    retries: int = 3,
    backoff_base: float = 2.0,
) -> list[GdeltBatchRecord]:
    """Download and parse a GDELT GKG zip batch (T5.2: retry + backoff).

    Returns up to _MAX_RECORDS_PER_BATCH relevant records.
    Returns [] on any network or parse failure after all retries.
    """
    import time as _time

    batch_ts = datetime.now(tz=timezone.utc)
    last_exc: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            resp = requests.get(gkg_url, timeout=timeout)
            if resp.status_code == 429:
                wait = backoff_base ** attempt
                logger.warning(
                    "[WARN] fetch_gkg_batch: 429 rate-limited (attempt %d/%d), waiting %.1fs",
                    attempt + 1, retries, wait,
                )
                _time.sleep(wait)
                continue
            resp.raise_for_status()
            last_exc = None
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                wait = backoff_base ** attempt
                logger.warning(
                    "[WARN] fetch_gkg_batch: attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt + 1, retries, exc, wait,
                )
                _time.sleep(wait)
    else:
        logger.warning("[WARN] fetch_gkg_batch: all %d attempts failed for %s: %s", retries, gkg_url, last_exc)
        return []

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            if not names:
                logger.warning("[WARN] fetch_gkg_batch: empty zip from %s", gkg_url)
                return []
            csv_name = names[0]
            with zf.open(csv_name) as raw_csv:
                content = raw_csv.read().decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, KeyError, Exception) as exc:
        logger.warning("[WARN] fetch_gkg_batch: zip parse error for %s: %s", gkg_url, exc)
        return []

    records: list[GdeltBatchRecord] = []
    reader = csv.reader(io.StringIO(content), delimiter="\t")

    for row in reader:
        if len(row) <= _COL_TONE:
            continue

        try:
            themes = _parse_themes(row[_COL_THEMES])
            tone = _parse_tone(row[_COL_TONE])

            if not _row_is_relevant(themes, tone):
                continue

            country_codes = _parse_locations(row[_COL_LOCATIONS])
            organizations = _parse_organizations(row[_COL_ORGANIZATIONS])
            persons = (
                _parse_persons(row[_COL_PERSONS])
                if len(row) > _COL_PERSONS
                else []
            )

            record = GdeltBatchRecord(
                record_id=row[_COL_RECORDID],
                date_str=row[_COL_DATE],
                source_name=row[_COL_SOURCE_NAME],
                url=row[_COL_DOC_IDENTIFIER],
                themes=themes,
                country_codes=country_codes,
                organizations=organizations,
                persons=persons,
                tone=tone,
                batch_ts=batch_ts,
            )
            records.append(record)

            if len(records) >= _MAX_RECORDS_PER_BATCH:
                break

        except Exception as exc:
            logger.debug("[SKIP] Row parse error: %s", exc)
            continue

    logger.info("[OK] fetch_gkg_batch: parsed %d relevant records from %s", len(records), gkg_url)
    return records


# ---------------------------------------------------------------------------
# Conversion: GdeltBatchRecord → NewsEvent
# ---------------------------------------------------------------------------


def _parse_gdelt_datetime(date_str: str) -> datetime:
    """Parse YYYYMMDDHHMMSS into UTC-aware datetime."""
    try:
        dt = datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(tz=timezone.utc)


def records_to_news_events(records: list[GdeltBatchRecord]) -> list[NewsEvent]:
    """Convert a list of GdeltBatchRecord to NewsEvent objects."""
    now = datetime.now(tz=timezone.utc)
    events: list[NewsEvent] = []

    for rec in records:
        url_bytes = (rec.url + rec.date_str).encode("utf-8")
        event_id = "ne_" + hashlib.sha256(url_bytes).hexdigest()[:16]

        # GDELT GKG batch files don't carry article titles — use source + url tail
        url_tail = rec.url[-60:] if len(rec.url) > 60 else rec.url
        title = f"{rec.source_name}: {url_tail}"

        content_hash = hashlib.sha256(rec.url.encode("utf-8")).hexdigest()[:16]
        published_at = _parse_gdelt_datetime(rec.date_str)

        # GDELT tone is -100..+100; clamp and normalise to sentiment [-1, +1].
        sentiment = max(-1.0, min(1.0, rec.tone / 10.0))

        # Merge orgs + persons into entities (deduped, order-preserving).
        entities: list[str] = []
        seen: set[str] = set()
        for ent in list(rec.organizations) + list(rec.persons):
            key = ent.lower()
            if key and key not in seen:
                seen.add(key)
                entities.append(ent)
                if len(entities) >= 10:
                    break

        event = NewsEvent(
            event_id=event_id,
            source_id=GDELT_SOURCE_ID,
            source_tier=GDELT_TIER,
            title=title,
            url=rec.url,
            published_at=published_at,
            ingested_at=now,
            geo_tags=rec.country_codes,
            entities=entities,
            keywords=[t.lower() for t in rec.themes[:10]],
            content_hash=content_hash,
            sentiment_score=sentiment,
        )
        events.append(event)

    return events


# ---------------------------------------------------------------------------
# Stateful fetcher
# ---------------------------------------------------------------------------


class GdeltFetcher:
    """
    Stateful GDELT batch fetcher.

    Tracks last-seen batch URL so the same batch is never processed twice.
    State is persisted to a JSON file at state_path.
    """

    def __init__(self, state_path: str | Path) -> None:
        self._state_path = Path(state_path)
        self._state = self.load_state()

    def load_state(self) -> GdeltFetchState:
        """Load state from JSON file. Returns fresh state if file doesn't exist."""
        if self._state_path.exists():
            try:
                with open(self._state_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                return GdeltFetchState(
                    last_batch_url=data.get("last_batch_url", ""),
                    last_fetch_ts=data.get("last_fetch_ts", ""),
                    total_events_ingested=data.get("total_events_ingested", 0),
                    consecutive_failures=data.get("consecutive_failures", 0),
                )
            except Exception as exc:
                logger.warning("[WARN] GdeltFetcher.load_state: failed to load %s: %s", self._state_path, exc)
        return GdeltFetchState()

    def save_state(self, state: GdeltFetchState) -> None:
        """Persist state to JSON file. Creates parent directories as needed."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._state_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "last_batch_url": state.last_batch_url,
                        "last_fetch_ts": state.last_fetch_ts,
                        "total_events_ingested": state.total_events_ingested,
                        "consecutive_failures": state.consecutive_failures,
                    },
                    fh,
                    indent=2,
                )
        except Exception as exc:
            logger.warning("[WARN] GdeltFetcher.save_state: failed to save %s: %s", self._state_path, exc)

    def fetch_new_events(self) -> tuple[list[NewsEvent], bool]:
        """
        Fetch the latest GDELT GKG batch.

        Returns:
            (events, is_new_batch) where is_new_batch=False means nothing new was found.
        """
        try:
            resp = requests.get(GDELT_LASTUPDATE_URL, timeout=15)
            resp.raise_for_status()
            gkg_url = parse_lastupdate(resp.text)
            if not gkg_url:
                logger.warning("[WARN] GdeltFetcher: could not extract GKG URL from lastupdate.txt")
                self._state.consecutive_failures += 1
                self.save_state(self._state)
                return [], False

            if gkg_url == self._state.last_batch_url:
                logger.debug("[SKIP] GdeltFetcher: batch already processed: %s", gkg_url)
                return [], False

            records = fetch_gkg_batch(gkg_url)
            events = records_to_news_events(records)

            self._state.last_batch_url = gkg_url
            self._state.last_fetch_ts = datetime.now(tz=timezone.utc).isoformat()
            self._state.total_events_ingested += len(events)
            self._state.consecutive_failures = 0
            self.save_state(self._state)

            logger.info("[OK] GdeltFetcher: %d events from new batch %s", len(events), gkg_url)
            return events, True

        except Exception as exc:
            logger.warning("[WARN] GdeltFetcher.fetch_new_events: unexpected error: %s", exc)
            self._state.consecutive_failures += 1
            self.save_state(self._state)
            return [], False
