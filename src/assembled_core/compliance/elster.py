"""ELSTER Anlage-KAP XML export for German capital-gains tax reporting.

ELSTER (Elektronische Steuererklärung) is Germany's electronic tax-filing
system.  This module generates a simplified Anlage-KAP XML fragment that
covers the fields typically required for equity trading accounts:

  - KAP line 7/8:  Kapitalerträge aus Aktienveräußerungen (gains / losses)
  - KAP line 18:   Verluste aus Aktienveräußerungen (carry-forward eligible)
  - KAP line 48/49: Anrechenbare Steuern (withholding tax already paid)
  - Tax computation: Abgeltungsteuer 25 % + Soli 5.5 %

The generated XML follows the ERiC schema structure used by the ELSTER
Rich-Client (ERiC) API for ESt transfers.  It is intentionally simplified
(no church tax, no foreign-income lines) — extend as needed.

Usage::

    from assembled_core.compliance.tax_report import TaxReportSummary, summarize_closed_lots
    from assembled_core.compliance.elster import build_anlage_kap_xml, ElsterExportConfig

    summary = summarize_closed_lots(lots, year=2025)
    cfg = ElsterExportConfig(steuerpflichtiger_id="12345678901", tax_year=2025)
    xml_str = build_anlage_kap_xml(summary, cfg)
    Path("anlage_kap_2025.xml").write_text(xml_str, encoding="utf-8")
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assembled_core.compliance.tax_report import TaxReportSummary


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ElsterExportConfig:
    """Metadata required for the ELSTER submission envelope."""

    tax_year: int
    steuerpflichtiger_id: str  # Steueridentifikationsnummer (11 digits)
    finanzamt_nr: str = "9201"  # Finanzamt-Nummer (Bundesland + Amt)
    software_name: str = "AssembledTradingAI"
    software_version: str = "1.0"
    kist_kirchensteuer_satz: float = 0.0  # 0 = no church tax
    anrechenbare_quellensteuer_eur: float = 0.0  # already-paid withholding


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_anlage_kap_xml(
    summary: "TaxReportSummary",
    config: ElsterExportConfig,
) -> str:
    """Build ELSTER Anlage-KAP XML string from a TaxReportSummary.

    Returns UTF-8 encoded XML with ELSTER ESt envelope.
    """
    # -- root ----------------------------------------------------------
    root = ET.Element("Elster", xmlns="http://www.elster.de/elsterxml/schema/v12")
    transfer_header = ET.SubElement(root, "TransferHeader", version="11")
    ET.SubElement(transfer_header, "Verfahren").text = "ElsterAnmeldung"
    ET.SubElement(transfer_header, "DatenArt").text = "ESt"
    ET.SubElement(transfer_header, "Vorgang").text = "send-Auth"
    ET.SubElement(transfer_header, "Testmerker").text = "0"

    sw_info = ET.SubElement(transfer_header, "SoftwareInfo")
    ET.SubElement(sw_info, "SWName").text = config.software_name
    ET.SubElement(sw_info, "Version").text = config.software_version
    ET.SubElement(sw_info, "Herstellerinfo").text = "Assembled-Trading-AI"

    # -- DatenTeil ----------------------------------------------------
    daten_teil = ET.SubElement(root, "DatenTeil")
    nutzdaten_block = ET.SubElement(daten_teil, "Nutzdatenblock")

    nutzdaten_header = ET.SubElement(nutzdaten_block, "NutzdatenHeader", version="11")
    ET.SubElement(nutzdaten_header, "NutzdatenTicket").text = f"ESt{config.tax_year}001"
    empfaenger = ET.SubElement(nutzdaten_header, "Empfaenger", id="F")
    empfaenger.text = config.finanzamt_nr

    # -- Nutzdaten / ESt ----------------------------------------------
    nutzdaten = ET.SubElement(nutzdaten_block, "Nutzdaten")
    est = ET.SubElement(
        nutzdaten,
        "Anmeldungssteuern",
        art="ESt",
        zeitraum=str(config.tax_year),
    )

    stpfl = ET.SubElement(est, "Steuerpflichtiger")
    ET.SubElement(stpfl, "IdNr").text = config.steuerpflichtiger_id.replace(" ", "")

    # -- Anlage KAP ---------------------------------------------------
    kap = ET.SubElement(est, "AnlageKAP", veranlagungszeitraum=str(config.tax_year))

    # Gains from equity sales (Zeile 7)
    _add_field(
        kap, "Kap_Z7_Veraeusserungsgewinne", max(0.0, summary.total_wins_eur), "EUR"
    )

    # Losses from equity sales (Zeile 8)
    _add_field(
        kap,
        "Kap_Z8_Veraeusserungsverluste",
        abs(min(0.0, summary.total_losses_eur)),
        "EUR",
    )

    # Net taxable capital gains (Zeile 18 / carry-forward)
    net = summary.total_wins_eur + summary.total_losses_eur  # losses negative
    if net < 0:
        _add_field(kap, "Kap_Z18_VerlustuebertragAktien", abs(net), "EUR")
    else:
        _add_field(kap, "Kap_Z18_VerlustuebertragAktien", 0.0, "EUR")

    # Taxable amount after Sparer-Pauschbetrag (Zeile 19)
    _add_field(kap, "Kap_Z19_Besteuerungsgrundlage", summary.taxable_pnl_eur, "EUR")

    # Abgeltungsteuer (Zeile 36): flat 25% only (estimated_tax_eur = 25% * 1.055 combined)
    abgeltungsteuer = round(summary.estimated_tax_eur / 1.055, 2)
    _add_field(kap, "Kap_Z36_Abgeltungsteuer", abgeltungsteuer, "EUR")

    # Solidaritätszuschlag (Zeile 37): 5.5% of Abgeltungsteuer only
    soli = round(abgeltungsteuer * 0.055, 2)
    _add_field(kap, "Kap_Z37_Solidaritaetszuschlag", soli, "EUR")

    # Anrechenbare Quellensteuer (Zeile 48)
    if config.anrechenbare_quellensteuer_eur > 0:
        _add_field(
            kap,
            "Kap_Z48_AnrechenbareSteuer",
            config.anrechenbare_quellensteuer_eur,
            "EUR",
        )

    # Metadata
    ET.SubElement(kap, "AnzahlGeschaeftsvorfaelle").text = str(summary.trade_count)
    # date.today() sweep: explicit Europe/Berlin for German tax docs (CET).
    # Fallback UTC if zoneinfo unavailable (avoids cross-platform local-tz drift).
    try:
        from zoneinfo import ZoneInfo

        _erstellung = datetime.now(tz=ZoneInfo("Europe/Berlin")).date()
    except Exception:
        _erstellung = datetime.now(tz=timezone.utc).date()
    ET.SubElement(kap, "ErstellungsDatum").text = _erstellung.isoformat()

    return _pretty_xml(root)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_field(parent: ET.Element, tag: str, value: float, unit: str = "EUR") -> None:
    el = ET.SubElement(parent, tag)
    el.text = f"{value:.2f}"
    if unit:
        el.set("einheit", unit)


def _pretty_xml(root: ET.Element) -> str:
    """Return indented XML string."""
    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        root, encoding="unicode", xml_declaration=False
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def export_to_file(
    summary: "TaxReportSummary",
    config: ElsterExportConfig,
    path: str,
) -> None:
    """Write ELSTER XML to *path* (UTF-8)."""
    xml_str = build_anlage_kap_xml(summary, config)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(xml_str)


__all__ = [
    "ElsterExportConfig",
    "build_anlage_kap_xml",
    "export_to_file",
]
