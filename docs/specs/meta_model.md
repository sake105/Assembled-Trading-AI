# Spec: `signals/meta_model.py`

## Purpose

Multiplicative confidence filter on the multifactor composite. See
ADR-006 context: the meta-model is NOT a 30th additive factor. It
scales the already-computed composite by a learned confidence and
filters out low-confidence signals.

## Public API

- `MetaModel` — holds the trained sklearn estimator, feature list,
  and metadata (training date, CV metrics).
- `train_meta_model(X, y, feature_names, ...) -> MetaModel` — fit
  the estimator with purged CV.
- `save_meta_model(model, path)` / `load_meta_model(path) -> MetaModel`
  — persist as joblib with a metadata JSON sidecar.

## Inputs

- `X`: DataFrame of factor scores + regime label + vol regime,
  one row per signal.
- `y`: binary label (`1` if forward 5-day return > threshold).
- Feature list is persisted with the model; inference rejects any
  input that does not match the trained feature list.

## Invariants

- `predict_proba` is monotone in the trained feature space when the
  underlying estimator is — this is not enforced, it is a property
  of the fit.
- Inference with an unknown feature set must raise, not silently
  impute zero.
- The model is versioned. Loading a model with a mismatched
  `MODEL_VERSION` is a hard error, not a warning.

## Error handling

- Missing features at inference → raise.
- NaN in features → raise (impute upstream, not inside the model).
- Unpickle failure → raise with the expected version in the message.

## Test strategy

- Round-trip: save → load → predict matches.
- Unknown feature → raises.
- Version mismatch → raises.
- Synthetic fit smoke test: 100 rows, trivial y, ROC-AUC > 0.9.

## Known limits

- Single-model only; no ensembling layer yet.
- No online update path. The model is refit offline per retrain
  cadence (monthly target).
- Validation split is a follow-up (see KNOWN_ISSUES.md).
