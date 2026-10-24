# csi-reduce

The native reducer of the monad measurement lake. One CSIQ segment
(`capture.csiq`, `capture.csiq.zst` or `capture.raw`) in; the lake's
per-segment outputs out:

- `packets` rows, in column batches (one per `batch_size` rows),
- `blocks` (class × tone group × minute) and `motion` (class × 2 s) statistics,
- the record-class census and the IP-164 window index.

It ports `monad_knowledge.lake.ingest.decode_segment` at reduction schema 7.
The Python reducer stays the reference and the test oracle.

## Same, to what precision

The port repeats every order of floating-point operations numpy fixes
(`src/npcompat.rs`): pairwise sums seeded with the identity, means and
standard deviations, Smith's complex64 division, `linspace`, `unwrap`,
`trapezoid`, and the complex64 → complex128 division of the circular mean.
It cannot repeat which library computes `log10f`, `atan2f`, `sinf`/`cosf`,
or BLAS's dot product, so values agree to a few float32 ulps, not bit for bit.
On the first real segments every row key, NULL mask, integer column, the
census and the window index were identical, and statistics agreed within
rel 2e-5 (study `notebooks/studies/2026-10-24-native-reducer/` in
monad-knowledge). Near a tie the orientation detector can decide differently,
which on the documented known-limit link moves the rows it touches.

For that reason the lake treats this reducer as its **own feature
specification** (`reducer = "native-<REDUCER_VERSION>"`). Bump
`REDUCER_VERSION` in `src/lib.rs` for any change that can move an output value.

## Build and test

```bash
cargo test                                    # unit + synthetic end-to-end
cargo clippy --all-targets --features python -- -D warnings
maturin build --release                       # abi3 wheel, Python >= 3.10
```

The Python module is `csi_reduce`:

```python
import csi_reduce
csi_reduce.reduce_segment(path, "csiq", segment_index, params, packet_sink) -> dict
csi_reduce.version()            # the semantic reducer version
csi_reduce.reduction_schema()   # the REDUCTION_SCHEMA it ports
```

`monad_knowledge.lake.native` is the only intended caller.

## Release and deploy

CI (`.github/workflows/ci.yml`) runs format, clippy and tests, builds one abi3
wheel per platform (linux x86_64 and aarch64 manylinux_2_28, macOS arm64),
smoke-imports each, and uploads `csi-reduce-wheels-<commit>` with
`SHA256SUMS`, `COMMIT`, `VERSION` and `REDUCER_VERSION`. A `v*` tag also
publishes a GitHub Release.

monad-knowledge's `roles/csi_reduce` installs the bundle of the commit checked
out at `repos/csi-reduce` on the control host, at `/opt/csi-reduce/<commit>`
with `/opt/csi-reduce/current` pointing at it. Installing does not switch the
lake; `[lake].reducer = "native"` does.

## Licence

GPL-2.0-only, like the `csiq` reader it builds on.
