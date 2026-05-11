"""Earnings-Call Q&A-Parser + Tone-Analyse mit Speaker-Separation.

Idee
----
Earnings-Calls haben strukturierte Sektionen:
- **Prepared Remarks** (CEO / CFO): scripted, oft positiv-biased.
- **Q&A**: spontane Antworten — hier zeigt sich echtes Tone.
- **Analysten-Fragen**: Sentiment der Analysten = Signal über zukünftige
  Erwartungsänderung.

Empirisch (Larcker/Zakolyukina 2012): Q&A-Tone prognostiziert Stock-Returns
besser als Press-Release.

Implementation
--------------
Regex-basiertes Parsing typischer Transcript-Formate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from erweiterung.transcripts.loughran_mcdonald import lm_sentiment_score


@dataclass
class TranscriptSegment:
    speaker: str
    speaker_role: str  # 'operator' | 'executive' | 'analyst' | 'unknown'
    section: str  # 'prepared_remarks' | 'qa' | 'opening'
    text: str


_SPEAKER_RE = re.compile(r"^([A-Z][\w\s,.\-']+):\s*", re.MULTILINE)
_OPERATOR_RE = re.compile(r"\boperator\b", re.IGNORECASE)
_ANALYST_RE = re.compile(
    r"\b(analyst|jpmorgan|morgan stanley|goldman|barclays|deutsche|credit suisse|ubs|wells fargo|bank of america|citi|wedbush|piper sandler|kbw)\b",
    re.IGNORECASE,
)
_EXEC_RE = re.compile(
    r"\b(ceo|cfo|coo|cto|president|chief|founder|chairman)\b", re.IGNORECASE
)
_QA_START_RE = re.compile(
    r"(question[- ]and[- ]answer|q\s*&\s*a|begin the question|first question)",
    re.IGNORECASE,
)


def classify_speaker_role(speaker_line: str) -> str:
    """Classify speaker role from speaker-introduction string."""
    lower = speaker_line.lower()
    if _OPERATOR_RE.search(lower):
        return "operator"
    if _ANALYST_RE.search(lower):
        return "analyst"
    if _EXEC_RE.search(lower):
        return "executive"
    return "unknown"


def parse_transcript(transcript_text: str) -> list[TranscriptSegment]:
    """Parse transcript text → segments with speaker + section labels.

    Args:
        transcript_text: full transcript string.

    Returns:
        List of TranscriptSegment.
    """
    if not isinstance(transcript_text, str) or len(transcript_text) < 50:
        return []

    # Find Q&A start
    qa_match = _QA_START_RE.search(transcript_text)
    qa_start_pos = qa_match.start() if qa_match else None

    # Split by speaker-labels. Allow parens für titles wie "Tim Cook (CEO)".
    pieces = re.split(
        r"^([A-Z][\w\s,.\-'()]{1,100}):\s*", transcript_text, flags=re.MULTILINE
    )
    # pieces alternates: pre-text, name1, text1, name2, text2, ...
    segments: list[TranscriptSegment] = []
    if len(pieces) < 3:
        return segments

    cursor = len(pieces[0])
    for i in range(1, len(pieces) - 1, 2):
        speaker = pieces[i].strip()
        text = pieces[i + 1].strip()
        role = classify_speaker_role(speaker)
        section = "prepared_remarks"
        if qa_start_pos is not None and cursor >= qa_start_pos:
            section = "qa"
        cursor += len(pieces[i]) + len(pieces[i + 1]) + 2
        segments.append(
            TranscriptSegment(
                speaker=speaker, speaker_role=role, section=section, text=text
            )
        )
    return segments


def score_segments(segments: list[TranscriptSegment]) -> pd.DataFrame:
    """LM-Sentiment-Score je Segment.

    Returns:
        DataFrame [speaker, role, section, sentiment, positive_pct, negative_pct, uncertainty_pct, n_tokens].
    """
    rows = []
    for seg in segments:
        sc = lm_sentiment_score(seg.text)
        rows.append(
            {
                "speaker": seg.speaker,
                "role": seg.speaker_role,
                "section": seg.section,
                **sc,
            }
        )
    return pd.DataFrame(rows)


def call_summary(segments: list[TranscriptSegment]) -> dict:
    """Aggregate metrics: exec vs analyst, prepared vs Q&A.

    Returns:
        dict with 4-quadrant tone-summary.
    """
    df = score_segments(segments)
    if df.empty:
        return {"error": "no segments"}

    def _avg(sub: pd.DataFrame, col: str) -> Optional[float]:
        if sub.empty or sub[col].isna().all():
            return None
        # token-weighted average
        w = sub["n_tokens"].clip(lower=1)
        return float((sub[col] * w).sum() / w.sum())

    out = {}
    for role in ("executive", "analyst"):
        for section in ("prepared_remarks", "qa"):
            sub = df[(df["role"] == role) & (df["section"] == section)]
            key = f"{role}_{section}"
            out[f"{key}_sentiment"] = _avg(sub, "sentiment")
            out[f"{key}_uncertainty"] = _avg(sub, "uncertainty_pct")
            out[f"{key}_n_tokens"] = int(sub["n_tokens"].sum()) if not sub.empty else 0
    # Headline metric: exec-Q&A-sentiment minus exec-Prepared = "spontaneity gap"
    if (
        out.get("executive_qa_sentiment") is not None
        and out.get("executive_prepared_remarks_sentiment") is not None
    ):
        out["spontaneity_gap"] = (
            out["executive_qa_sentiment"] - out["executive_prepared_remarks_sentiment"]
        )
    return out


__all__ = [
    "TranscriptSegment",
    "classify_speaker_role",
    "parse_transcript",
    "score_segments",
    "call_summary",
]
