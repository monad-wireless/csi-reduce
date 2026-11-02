# Changelog

## reducer version 2 (2026-10-31)

- `motion` gains four injector-scoped columns: the inter-antenna coherence
  across tones of chains 0 and 1, as the 2-s mean and SD
  (`inj_coh_amp_mean/_sd`: Pearson of the two chains' dB amplitudes;
  `inj_coh_cplx_mean/_sd`: |Σ a·conj(b)| / √(Σ|a|²·Σ|b|²)). Ports
  `ingest._coherence`. Symmetric in the chains and magnitude-only, so the
  orientation events do not move them.
- `motion` also gains each chain's RSSI spread and distinct whole-dB levels
  (`inj_rssi_a_sd`, `inj_rssi_b_sd`, `inj_lv_a`, `inj_lv_b`), frames whose
  chain reports -127 left out. `inj_rssi_sd` stays the stronger chain.
- No existing value changes; the bump re-reduces the archive because the
  reducer id is in the feature spec.
- Parity on a real injector segment (monad05, 30 Oct, 901 rows): the eight new
  columns agree with the Python reference to ≤ 3e-14 relative (the RSSI ones
  exactly). One 2-s window differs in pre-existing columns, at the documented
  orientation-tie limit (161 vs 159 swaps in that window).

## 0.1.0 — reducer version 1

- First port of `monad_knowledge.lake.ingest.decode_segment` at reduction
  schema 7: packets (column batches), blocks, motion, census, window index,
  the half-band tone-swap guard and the orientation correction.
- `.csiq`, `.csiq.zst` and `capture.raw` input; a cut container keeps its
  prefix and reports `TruncatedCapture: ...`.
- PyO3 module `csi_reduce` (abi3, Python ≥ 3.10); CI wheels for linux
  x86_64/aarch64 (manylinux_2_28) and macOS arm64.
