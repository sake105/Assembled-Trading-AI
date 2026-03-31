"""News entity linking: map headlines to ticker symbols."""

from __future__ import annotations

import pandas as pd


def link_news_to_symbols(
    news: pd.DataFrame,
    symbols: list[str] | None = None,
    mapping_df: pd.DataFrame | None = None,
    security_master_df: pd.DataFrame | None = None,
    missing: str = "keep",
) -> pd.DataFrame:
    """Link news articles to symbols based on entity/ticker mentions.

    Resolution order:
      1. If 'symbol' column already exists, strip whitespace and keep it.
      2. Try resolving via 'ticker' column (stripped), then 'entity' column (stripped).
      3. Lookup order: mapping_df first (entity → symbol), then security_master_df
         (direct symbol match). mapping_df always takes priority.

    Args:
        news: News DataFrame.
        symbols: Optional allowlist (currently unused, reserved for future filtering).
        mapping_df: DataFrame with columns 'entity' and 'symbol' for name→ticker lookup.
        security_master_df: DataFrame with column 'symbol'; if ticker matches a symbol,
            use it directly.
        missing: Behaviour for rows that cannot be resolved when a mapping source is
            provided. One of:
            - "keep" (default): leave symbol as None/NA.
            - "drop": remove unresolved rows.
            - "keep_unknown": set symbol to "UNKNOWN".
            - "raise": raise ValueError("Cannot map ...").

    Returns:
        Copy of news with 'symbol' column populated where possible.
    """
    out = news.copy()

    if out.empty:
        if "symbol" not in out.columns:
            out["symbol"] = pd.NA
        return out

    # Strip existing symbol column
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )
    else:
        out["symbol"] = pd.NA

    # Build entity → symbol lookup dict (mapping_df takes priority over security_master)
    entity_map: dict[str, str] = {}

    if security_master_df is not None and not security_master_df.empty:
        if "symbol" in security_master_df.columns:
            for val in security_master_df["symbol"]:
                if val is not None and pd.notna(val):
                    key = str(val).strip()
                    if key:
                        entity_map[key] = key

    if mapping_df is not None and not mapping_df.empty:
        for _, row in mapping_df.iterrows():
            entity_val = row.get("entity")
            sym_val = row.get("symbol")
            if (
                entity_val is not None
                and pd.notna(entity_val)
                and sym_val is not None
                and pd.notna(sym_val)
            ):
                entity_map[str(entity_val).strip()] = str(sym_val).strip()

    # Resolve symbols for rows that still have no symbol
    has_mapping_source = mapping_df is not None or security_master_df is not None
    has_entity_col = "ticker" in out.columns or "entity" in out.columns

    if has_mapping_source and has_entity_col:
        for idx in out.index:
            if pd.notna(out.at[idx, "symbol"]):
                continue
            # Prefer ticker over entity column
            source_val: str | None = None
            if "ticker" in out.columns:
                val = out.at[idx, "ticker"]
                if val is not None and pd.notna(val):
                    stripped = str(val).strip()
                    if stripped:
                        source_val = stripped
            if source_val is None and "entity" in out.columns:
                val = out.at[idx, "entity"]
                if val is not None and pd.notna(val):
                    stripped = str(val).strip()
                    if stripped:
                        source_val = stripped

            if source_val is not None:
                resolved = entity_map.get(source_val)
                if resolved is not None:
                    out.at[idx, "symbol"] = resolved

    # Handle rows that remain unmapped when a mapping source was provided
    if has_mapping_source and has_entity_col and missing != "keep":
        still_missing = out["symbol"].apply(
            lambda x: x is None or (hasattr(x, "__class__") and pd.isna(x))
        )
        if still_missing.any():
            if missing == "raise":
                raise ValueError(f"Cannot map {still_missing.sum()} row(s) to a symbol")
            elif missing == "drop":
                out = out[~still_missing].reset_index(drop=True)
            elif missing == "keep_unknown":
                out.loc[still_missing, "symbol"] = "UNKNOWN"

    return out
