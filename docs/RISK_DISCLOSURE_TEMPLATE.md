# Risk Disclosure & Performance Disclaimer Template

> Audit C4-094 — if the system or any of its outputs is ever made
> available to a third party (a public website, a Substack, a paid
> service), the following risk-disclosure and performance-disclaimer
> language MUST appear, prominently, before any number is shown.
>
> **Status today:** internal use only — no third-party distribution.
> This template activates the moment we publish a single Sharpe number
> outside this git repository to a non-personal audience.

## 1. Generic risk-disclosure block

> **Risk disclosure.** Trading securities (including stocks, ETFs,
> options, futures, FX, and digital assets) involves substantial
> risk of loss and is not suitable for every investor. The
> valuation of securities may fluctuate, and as a result, clients
> may lose more than their original investment. The impact of
> seasonal and geopolitical events is already factored into
> market prices. The leveraged nature of FX, futures, and certain
> derivative products means that any market movement will have an
> equally proportional effect on your deposited funds. This may
> work against you as well as for you. Past performance is not
> indicative of future results.

Localised German version:

> **Risikohinweis.** Der Handel mit Wertpapieren (einschließlich
> Aktien, ETFs, Optionen, Futures, Devisen und digitalen Vermögens-
> werten) beinhaltet ein erhebliches Verlustrisiko und ist nicht
> für jeden Anleger geeignet. Die Bewertung von Wertpapieren kann
> schwanken, sodass Anleger mehr als ihren ursprünglichen Einsatz
> verlieren können. Frühere Wertentwicklungen sind kein verlässlicher
> Indikator für künftige Wertentwicklungen.

## 2. Backtest performance disclaimer (every backtest result MUST carry this)

> **Backtest disclaimer.** The performance results shown are
> **hypothetical** and reflect simulated trading on historical
> data. No actual capital was at risk. Hypothetical results have
> many inherent limitations, including (but not limited to): the
> ability to design the strategy with the benefit of hindsight,
> the lack of liquidity constraints, the inability to react in
> real time to market changes, and the omission of fees, taxes,
> and slippage that may apply to a live account.
>
> The strategy assumed: a commission of `<X>` basis points, a
> spread of `<Y>` basis points, a slippage of `<Z>` basis points,
> a starting capital of `<C>`, and a universe of `<N>` symbols
> sourced from `<source>` over `<from> → <to>`.
>
> Survivorship bias: `<acknowledged | corrected>`.
> Look-ahead bias: PIT-audit `<passed | not run>`.
> Statistical significance: Deflated Sharpe Ratio `<DSR>`,
> Probabilistic Sharpe Ratio `<PSR>`, PBO `<PBO>`,
> Permutation p-value `<p>`.

## 3. Live track-record disclaimer (if a live record is published)

> **Live results disclaimer.** Live results shown reflect actual
> trading in the operator's own account from `<start>` to `<end>`.
> Capital and instrument selection may differ from any future
> deployment. Custodian, tax treatment, and commission schedule of
> the operator's account differ from any client account.
> No representation is made that any client account will achieve
> profits or losses similar to those shown. Past performance is
> not indicative of future results.

## 4. Compliance & licensing disclaimer

> The operator of this system does **not** provide investment
> advice within the meaning of § 1 (1a) KWG. The output is
> personal research and is shared for educational and
> illustrative purposes only. No content here constitutes a
> recommendation to buy, sell, or hold any specific security.
> Readers should consult a qualified investment advisor before
> making any investment decision.

## 5. Where each block goes

| Surface | Block to include |
|---|---|
| Public-facing website / landing page | §1 (Generic) + §4 (Compliance) |
| Backtest report / blog post citing numbers | §1 + §2 |
| Paper-track tear sheet | §1 + §2 |
| Live track-record publication | §1 + §3 + §4 |
| Substack / newsletter | §1 + §4, plus §2 or §3 as applicable |

## 6. Translation

For DE-language audiences, both EN and DE versions MUST appear
side-by-side or DE-only with the EN version available on demand.

## 7. Versioning

Every published disclosure should carry a version tag
`disclosure-vYYYY.Q.N` so a reader of an archived post knows
which version was current at the time. Active version lives in
this file's header — bump on every material change.

Current version: **disclosure-v2026.2.1** (2026-05-12).
