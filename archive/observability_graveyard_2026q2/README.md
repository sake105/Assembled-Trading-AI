# Observability Graveyard 2026 Q2

Files moved out of `src/assembled_core/` because they were only wired
for observability (import + config instantiation + meta-dict entry)
without actually influencing trading decisions.

See `autonome weiterarbeit/AUDIT_TEIL2_Modul_Dekomposition.md` for full rationale.

If you need to reactivate any of these, move them back and ensure
they are actually used in a signal/sizing/risk/execution decision.
