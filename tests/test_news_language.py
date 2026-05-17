"""Tests for lightweight language detection."""

from __future__ import annotations

import pytest

from src.assembled_core.intel.news_language import detect_language, is_english


@pytest.mark.fast
class TestLanguageDetection:
    def test_english(self):
        assert detect_language("Russia launches missile attack on Ukraine") == "en"
        assert is_english("Fed raises rates by 25 bps")

    def test_german(self):
        assert (
            detect_language("Die Zentralbank erhöht die Zinsen um 25 Basispunkte")
            == "de"
        )

    def test_french(self):
        assert detect_language("La banque centrale augmente les taux d'intérêt") == "fr"

    def test_spanish(self):
        assert detect_language("El banco central aumenta las tasas de interés") == "es"

    def test_russian(self):
        assert detect_language("Россия запустила ракетный удар по Украине") == "ru"

    def test_ukrainian(self):
        assert detect_language("Росія запустила ракетний удар по Україні") == "uk"

    def test_chinese(self):
        assert detect_language("中国央行加息25个基点") == "zh"

    def test_japanese(self):
        assert detect_language("日本銀行は金利を引き上げた") == "ja"

    def test_arabic(self):
        assert detect_language("البنك المركزي يرفع أسعار الفائدة") == "ar"

    def test_empty_falls_back_to_english(self):
        assert detect_language("") == "en"
        assert detect_language("12345 !!! ???") == "en"

    def test_mixed_script_picks_dominant(self):
        # Mostly German stopwords with one English word
        assert detect_language("Die US-Notenbank und die Zentralbank") == "de"
