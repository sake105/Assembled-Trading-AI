# 21 — Paid Modelle und APIs (mit Kosten, <100 EUR/Monat)

**Zweck:** Modelle und APIs, die Geld kosten — aber nur die, die echten ML-Edge oder Latency-Vorteil bringen.

**Budget-Realität:** Das einzige Paid-Modell mit starkem ROI ist **Claude Haiku 4.5** für News-LLM-Zweitrunde. Alles andere ist optional.

---

## 21.1 Claude Haiku 4.5 API — DER Paid-Modell-Must

**Was:** Kleinstes Frontier-Modell von Anthropic, JSON-tauglich, <1s Latenz.

**Preis:** 
- Input: ~0.25 USD pro 1M Tokens (~0.23 EUR)
- Output: ~1.25 USD pro 1M Tokens (~1.15 EUR)

**Budget-Kalkulation:**
```
500 Nachrichten/Tag × 30 Tage = 15.000/Monat
Input:   15.000 × 400 Tokens = 6M Tokens × 0.25 USD = 1.50 USD/Monat
Output:  15.000 × 200 Tokens = 3M Tokens × 1.25 USD = 3.75 USD/Monat
Veto-Calls: ~100/Tag × 30 = 3.000 × 800 Tokens input, 400 output
Input:   3.000 × 800 = 2.4M × 0.25 USD = 0.60 USD
Output:  3.000 × 400 = 1.2M × 1.25 USD = 1.50 USD
Total: ~7.35 USD ≈ 7 EUR/Monat
```

**Mit Puffer realistisch <10 EUR/Monat.**

**Install:** `pip install anthropic`

### Use-Case 1: Target-Sentiment-Extraction

Headlines können mehrere Firmen betreffen mit unterschiedlichem Sentiment. FinBERT gibt nur ein Score; Haiku kann Multi-Target mit Rationale:

```python
import anthropic
import json

client = anthropic.Anthropic(api_key=settings.anthropic_key)

def extract_news_context(headline: str, body: str = "") -> dict:
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system="Du bist ein präziser Finanz-News-Analyst. Antworte nur mit JSON.",
        messages=[{
            "role": "user",
            "content": f"""Analysiere diese Finanz-Nachricht.

Headline: {headline}
Body: {body[:1500]}

Extrahiere als JSON:
{{
  "primary_tickers": [...],
  "mentioned_tickers": [...],
  "sentiment_primary": -1.0 bis +1.0,
  "sentiment_per_ticker": {{"AAPL": 0.5, "MSFT": -0.2}},
  "event_type": "earnings|m&a|management|regulatory|product|legal|analyst|macro",
  "novelty_claim": "neu|wiederholend|spekulativ",
  "rationale": "1 Satz Begründung"
}}"""
        }]
    )
    return json.loads(msg.content[0].text)
```

### Use-Case 2: Primary vs Mentioned Disambiguation

Nach spaCy-NER bleibt Mehrdeutigkeit. Haiku entscheidet:

```python
def classify_ticker_role(headline, body, tickers):
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"""In dieser Nachricht werden folgende Tickers erwähnt: {tickers}

Headline: {headline}
Body: {body[:1000]}

Klassifiziere jeden Ticker als JSON:
{{"TICKER": "primary" oder "mentioned" oder "reference"}}

primary = Nachricht handelt hauptsächlich von dieser Firma
mentioned = Firma wird beiläufig genannt, aber nicht Hauptthema
reference = nur Vergleichsreferenz (z.B. "wie Apple")"""
        }]
    )
    return json.loads(msg.content[0].text)
```

### Use-Case 3: Composite-Score-Veto bei Grenzfällen

Bei hoher Score-Magnitude vor Position Entry ein Veto-Call:

```python
def llm_veto_check(ticker, composite_score, news_context, ta_features):
    if abs(composite_score) < 0.8:
        return {"veto": False, "reason": "score_below_threshold"}
    
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"""Ticker {ticker} hat einen Composite-Score von {composite_score:.2f}.

Top-3-Signale: {ta_features['top_signals']}
News-Context: {news_context}

Würde ein erfahrener Portfolio-Manager DIESEM Trade zustimmen?
Gib als JSON:
{{
  "approve": true|false,
  "confidence": 0-1,
  "red_flags": [...],
  "reason": "1 Satz"
}}"""
        }]
    )
    return json.loads(msg.content[0].text)
```

**Regel:** LLM-Veto ist **Sicherheits-Layer**, nicht Primary-Entscheider. Wenn LLM "disapprove" bei positivem Composite → Position halbieren, nicht skippen.

### Kosten-Kontrolle

```python
# Simple Budget-Guard
class LLMBudgetGuard:
    def __init__(self, monthly_budget_usd=10):
        self.budget = monthly_budget_usd
        self.spent = 0.0
        self.reset_at = datetime.now().replace(day=1) + timedelta(days=32)
    
    def can_spend(self, estimated_cost):
        self._maybe_reset()
        return self.spent + estimated_cost < self.budget
    
    def record(self, actual_cost):
        self.spent += actual_cost
```

**Prometheus-Alert:** Wenn `spent > 0.8 * budget` → Slack-Notification.

---

## 21.2 Lokale LLM als Backup (Ollama) — Free, aber Paid-relevant erwähnt

**Was:** Lokale Inferenz ohne API-Kosten.

**Install:** https://ollama.com (Windows-tauglich).

**Modelle:**

| Modell | Größe | Quantisiert | RAM | Geschwindigkeit CPU |
|---|---|---|---|---|
| `llama3.1:8b-instruct-q4_K_M` | 4.7 GB | Q4 | 8 GB | 3-8 Tok/s |
| `mistral:7b-instruct-q4_K_M` | 4.1 GB | Q4 | 8 GB | 4-10 Tok/s |
| `qwen2.5:7b-instruct-q4_K_M` | 4.5 GB | Q4 | 8 GB | 3-8 Tok/s |

**Wann nutzen:**
- Budget-Überschreitung bei Anthropic
- Privacy-kritische Daten (Alpha-Signale)
- Bulk-Reprocessing historischer News

**Wann NICHT:**
- JSON-strikte Output-Qualität (Haiku deutlich besser)
- Sub-Sekunden-Latenz nötig (lokale 7B zu langsam CPU)

**Integration:**
```python
import ollama

def local_llm_sentiment(text):
    response = ollama.generate(
        model="llama3.1:8b-instruct-q4_K_M",
        prompt=f"Sentiment as JSON {{sentiment: -1..1, reason: '...'}}\n{text}",
        options={"temperature": 0.1}
    )
    return json.loads(response['response'])
```

**Verdict:** Netter Backup, aber **Haiku 4.5 API ist Primary**. Lokaler LLM nicht als Hauptlösung.

---

## 21.3 FinBERT-Fine-Tuning (Phase 3)

**Was:** Eigenes FinBERT-Modell auf 5-10k selbst-gelabelten Headlines.

**Kosten:** 
- Rechenleistung: einmalig ~20 EUR auf Lambda Labs / RunPod (H100 für 4 Stunden) **oder** Oracle Always-Free (mehrere Stunden, aber kostenfrei).
- Daten-Labelling: optional Amazon Mechanical Turk (~50-200 EUR) oder selbst labeln.

**Warum:**
- Distant Supervision: Label = Vorzeichen der Preisreaktion ±t Minuten nach News.
- Funktioniert besonders gut bei domain-spezifischen Tickern (Biotech, Semis).

**Pattern:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

base = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=3)

# distant supervision
# label = sign(price_change_5min_after_news)
trainer = Trainer(model=model, train_dataset=ds_train, eval_dataset=ds_val, ...)
trainer.train()
trainer.save_model("./finbert-ata-v1")
```

**Verdict:** Phase 3, nur wenn News-Alpha in Basis-Pipeline bereits validiert ist.

---

## 21.4 Grafana Cloud Pro (optional)

**Free-Tier reicht** für Solo. **Pro-Tier (~25 USD/Monat)** nur wenn du:
- >10k Metriken hast (gehst du nicht in Phase 1-3)
- Längere Retention (>14 Tage) willst

**Verdict:** Skip. Grafana Cloud Free reicht.

---

## 21.5 Voyage AI Embeddings (Optional)

**Was:** Kommerzielle Finance-spezifische Embeddings.

**Preis:** $0.02/1M Tokens, 200M free Tokens initial.

**Modell:** `voyage-finance-2` vs `voyage-3.5-lite` (generic).

**Wann:** Wenn BGE-small oder MiniLM die Retrieval-Qualität nicht liefert.

**Kosten-Kalkulation:**
```
500 Headlines/Tag × 50 Tokens = 25k/Tag = 750k/Monat
750k × 0.02/1M = 0.015 USD/Monat
```

Praktisch 0 EUR mit 200M Free-Tokens-Puffer. **Gratis bis zum SaaS-Launch.**

**Install:** `pip install voyageai`

**Verdict:** Phase 2 Upgrade, falls Novelty-Detection schlechter wird.

---

## Der empfohlene LLM-Stack

```
Primary Text-Understanding:    Claude Haiku 4.5 (~10 EUR/Monat)
Fallback wenn Budget voll:     Ollama llama3.1:8b lokal (0 EUR)
Embedding-Primary:             BAAI/bge-small-en-v1.5 lokal (0 EUR)  
Embedding-Alternative:         Voyage AI Finance-2 (~0 EUR mit Free-Tier)
FinBERT-Tone:                 lokal (0 EUR)
FOMC-RoBERTa:                 lokal (0 EUR)
Optional Fine-Tune:           Phase 3, einmalig ~20 EUR Compute
```

---

## Umsetzungs-Checkliste

**Phase 1:**
- [ ] Anthropic Console-Account
- [ ] API-Key in `.env.sops.yaml` (nie plain)
- [ ] Budget-Guard mit Monatlichem Reset
- [ ] Prometheus-Metrik für Anthropic-Ausgaben
- [ ] Target-Sentiment-Extraction produktiv

**Phase 2:**
- [ ] Primary-vs-Mentioned-Disambiguation
- [ ] LLM-Veto-Check bei Composite-Grenzfällen
- [ ] Ollama als Fallback installiert
- [ ] A/B-Test Haiku vs Ollama-lokal für Non-kritische Tasks

**Phase 3:**
- [ ] FinBERT-Fine-Tune-Pipeline (distant supervision)
- [ ] Optional Voyage-Finance-2 Embeddings A/B

---

## Ehrliche Einschätzung

Das einzige **Paid-Modell mit klarem ROI ist Claude Haiku 4.5**. Alles andere (FinBERT-Fine-Tune, Voyage-Embeddings, Ollama) ist optional oder free.

**10 EUR/Monat für Haiku ist der einzige Paid-Model-Posten, der sich lohnt.** Er gibt dir:
- Multi-Target-Sentiment (FinBERT kann nicht)
- Primary-vs-Mentioned-Disambiguation (NER kann nicht)
- LLM-Veto als zusätzliche Safety-Schicht

**Zusammen mit EODHD (18 EUR) sind das 28 EUR für die beiden einzigen wirklich impact-vollen Paid-Upgrades.**
