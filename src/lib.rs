//! # csi-reduce — the native reducer of the monad measurement lake
//!
//! One CSIQ segment in, the lake's per-segment outputs out: `packets` rows (in
//! column batches), `blocks`, `motion`, the record-class census and the IP-164
//! window index. It ports `monad_knowledge.lake.ingest.decode_segment`, which
//! stays the reference and the test oracle (IP-164 Phase 5).
//!
//! ## What "the same" means here
//!
//! Every order of floating-point operations that numpy fixes is repeated
//! ([npcompat]). What cannot be repeated is the library behind a
//! transcendental function (`log10`, `atan2`, `sin`, `cos`) and BLAS's dot
//! product in the tone-swap guard, so the outputs agree to a tolerance, not
//! bit for bit. The lake therefore treats this reducer as its own feature
//! specification (`reducer = "native-<REDUCER_VERSION>"` in the decode
//! parameters), and the parity gate in monad-knowledge
//! (`tests/lake/test_native_parity.py`) states the tolerances per column.
//!
//! Bump [REDUCER_VERSION] whenever a change here can move an output value:
//! it is part of the feature specification digest, so a bump requeues the
//! archive and a forgotten bump would mix two reducers under one id.

pub mod frame;
pub mod npcompat;
pub mod reduce;
pub mod source;

#[cfg(feature = "python")]
mod python;

/// The semantic version of the reduction this crate computes. See the module doc.
pub const REDUCER_VERSION: &str = "1";

/// The `monad_knowledge.lake.schema.REDUCTION_SCHEMA` this crate ports.
pub const REDUCTION_SCHEMA: i64 = 7;
