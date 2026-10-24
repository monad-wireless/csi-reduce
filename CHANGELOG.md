# Changelog

## 0.1.0 — reducer version 1

- First port of `monad_knowledge.lake.ingest.decode_segment` at reduction
  schema 7: packets (column batches), blocks, motion, census, window index,
  the half-band tone-swap guard and the orientation correction.
- `.csiq`, `.csiq.zst` and `capture.raw` input; a cut container keeps its
  prefix and reports `TruncatedCapture: ...`.
- PyO3 module `csi_reduce` (abi3, Python ≥ 3.10); CI wheels for linux
  x86_64/aarch64 (manylinux_2_28) and macOS arm64.
