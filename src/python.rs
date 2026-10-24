//! The PyO3 module `csi_reduce`.
//!
//! One call per segment. Packet rows cross to Python per column batch as
//! little-endian buffers (`numpy.frombuffer` on the other side), never once per
//! frame; the few thousand `blocks`/`motion` rows come back as dicts of
//! numbers, and `monad_knowledge.lake.native` adds the keys and timestamps
//! with the same expressions the Python reducer uses.

use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::reduce::{self, PacketBatch, Params, Reduction};
use crate::source::Kind;

fn bytes<'py, T: Copy>(
    py: Python<'py>,
    v: &[T],
    to_le: impl Fn(T) -> Vec<u8>,
) -> Bound<'py, PyBytes> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(v));
    for &x in v {
        out.extend_from_slice(&to_le(x));
    }
    PyBytes::new(py, &out)
}

fn batch_dict<'py>(py: Python<'py>, b: &PacketBatch) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n", b.len())?;
    let i64s = |v: &[i64]| bytes(py, v, |x| x.to_le_bytes().to_vec());
    let i32s = |v: &[i32]| bytes(py, v, |x| x.to_le_bytes().to_vec());
    let i16s = |v: &[i16]| bytes(py, v, |x| x.to_le_bytes().to_vec());
    let u32s = |v: &[u32]| bytes(py, v, |x| x.to_le_bytes().to_vec());
    let f64s = |v: &[f64]| bytes(py, v, |x| x.to_le_bytes().to_vec());
    let u8s = |v: &[u8]| PyBytes::new(py, v);
    d.set_item("dt_days", i32s(&b.dt_days))?;
    d.set_item("t_ns", i64s(&b.t_ns))?;
    d.set_item("ticks", i64s(&b.ticks))?;
    d.set_item("mono_us", i64s(&b.mono_us))?;
    d.set_item("mono_valid", u8s(&b.mono_valid))?;
    d.set_item("seq", i64s(&b.seq))?;
    d.set_item("smac", u32s(&b.smac))?;
    d.set_item("ntone", i32s(&b.ntone))?;
    d.set_item("modulation", i32s(&b.modulation))?;
    d.set_item("bandwidth_code", i16s(&b.bandwidth_code))?;
    d.set_item("antenna_sel", i16s(&b.antenna_sel))?;
    d.set_item("bw_valid", u8s(&b.bw_valid))?;
    d.set_item("n_chains", i32s(&b.n_chains))?;
    d.set_item("rssi", i16s(&b.rssi))?;
    d.set_item("rssi_valid", u8s(&b.rssi_valid))?;
    d.set_item("rssi_a", i16s(&b.rssi_a))?;
    d.set_item("rssi_a_valid", u8s(&b.rssi_a_valid))?;
    d.set_item("rssi_b", i16s(&b.rssi_b))?;
    d.set_item("rssi_b_valid", u8s(&b.rssi_b_valid))?;
    d.set_item("chain_rms_offsets", i32s(&b.chain_rms_offsets))?;
    d.set_item("chain_rms_values", f64s(&b.chain_rms_values))?;
    d.set_item("chain_rms_valid", u8s(&b.chain_rms_valid))?;
    d.set_item("ratio_amp_db", f64s(&b.ratio_amp_db))?;
    d.set_item("ratio_amp_valid", u8s(&b.ratio_amp_valid))?;
    d.set_item("ratio_phase_rad", f64s(&b.ratio_phase_rad))?;
    d.set_item("ratio_phase_valid", u8s(&b.ratio_phase_valid))?;
    d.set_item("tone_swapped", u8s(&b.tone_swapped))?;
    d.set_item("tone_swapped_valid", u8s(&b.tone_swapped_valid))?;
    d.set_item("macs", PyList::new(py, &b.macs)?)?;
    d.set_item("modulations", PyList::new(py, &b.modulations)?)?;
    Ok(d)
}

fn class_dict<'py>(py: Python<'py>, r: &Reduction, cls: usize) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ntone", r.classes[cls].ntone)?;
    d.set_item("modulation", r.classes[cls].modulation.clone())?;
    Ok(d)
}

fn result_dict<'py>(py: Python<'py>, r: &Reduction) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let blocks = PyList::empty(py);
    for b in &r.blocks {
        let d = class_dict(py, r, b.class)?;
        d.set_item("start_ns", b.start_ns)?;
        d.set_item("tone_group", b.tone_group)?;
        d.set_item("tone_lo", b.tone_lo)?;
        d.set_item("tone_hi", b.tone_hi)?;
        d.set_item("n_packets", b.n_packets)?;
        d.set_item("ratio_amp_db_median", b.amp_median)?;
        d.set_item("ratio_phase_rad_circmedian", b.phase_circmedian)?;
        d.set_item("amp_s", b.amp_s)?;
        d.set_item("ratio_amp_sd_db", b.ratio_amp_sd_db)?;
        d.set_item("minutes_since_segment_start", b.minutes_since_segment_start)?;
        d.set_item("inj_n_packets", b.inj_n_packets)?;
        d.set_item("inj_ratio_amp_sd_db", b.inj_ratio_amp_sd_db)?;
        for (k, (suffix, _, _)) in reduce::DOPPLER_BANDS.iter().enumerate() {
            d.set_item(format!("doppler_pw_{suffix}"), b.doppler[k])?;
            d.set_item(format!("inj_doppler_pw_{suffix}"), b.inj_doppler[k])?;
        }
        blocks.append(d)?;
    }
    let motion = PyList::empty(py);
    for m in &r.motion {
        let d = class_dict(py, r, m.class)?;
        d.set_item("start_ns", m.start_ns)?;
        d.set_item("n_packets", m.n_packets)?;
        d.set_item("n_subwindows", m.n_subwindows)?;
        d.set_item("ratio_amp_sd_db", m.ratio_amp_sd_db)?;
        d.set_item("minutes_since_segment_start", m.minutes_since_segment_start)?;
        d.set_item("inj_n_packets", m.inj_n_packets)?;
        d.set_item("inj_n_subwindows", m.inj_n_subwindows)?;
        d.set_item("inj_ratio_amp_sd_db", m.inj_ratio_amp_sd_db)?;
        for (k, (suffix, _, _)) in reduce::DOPPLER_BANDS.iter().enumerate() {
            d.set_item(format!("doppler_pw_{suffix}_mean"), m.doppler_mean[k])?;
            d.set_item(format!("doppler_pw_{suffix}_max"), m.doppler_max[k])?;
            d.set_item(
                format!("inj_doppler_pw_{suffix}_mean"),
                m.inj_doppler_mean[k],
            )?;
            d.set_item(format!("inj_doppler_pw_{suffix}_max"), m.inj_doppler_max[k])?;
        }
        d.set_item("inj_n_steps", m.inj_n_steps)?;
        d.set_item("inj_ratio_amp_sd_db_corr", m.inj_ratio_amp_sd_db_corr)?;
        d.set_item("inj_n_swaps", m.inj_n_swaps)?;
        d.set_item("inj_n_offsets", m.inj_n_offsets)?;
        d.set_item("inj_dphi_mean", m.inj_dphi_mean)?;
        d.set_item("inj_damp_mean", m.inj_damp_mean)?;
        d.set_item("inj_c2_mean", m.inj_c2_mean)?;
        d.set_item("inj_rssi_mean", m.inj_rssi_mean)?;
        d.set_item("inj_rssi_sd", m.inj_rssi_sd)?;
        d.set_item("inj_dab_mean", m.inj_dab_mean)?;
        d.set_item("inj_dab_sd", m.inj_dab_sd)?;
        motion.append(d)?;
    }
    let census = PyList::empty(py);
    for c in &r.census {
        let d = class_dict(py, r, c.class)?;
        d.set_item("n_records", c.n_records)?;
        d.set_item("n_rx", c.n_rx)?;
        d.set_item("n_tx", c.n_tx)?;
        d.set_item("bandwidth_mhz", PyList::new(py, &c.bandwidth_mhz)?)?;
        census.append(d)?;
    }
    let windows = PyList::empty(py);
    for w in &r.windows {
        let d = class_dict(py, r, w.class)?;
        d.set_item("n_rx", w.n_rx)?;
        d.set_item("n_tx", w.n_tx)?;
        d.set_item("window_start_ns", w.window_start_ns)?;
        d.set_item("first_ordinal", w.first_ordinal)?;
        d.set_item("last_ordinal", w.last_ordinal)?;
        d.set_item("n_frames", w.n_frames)?;
        windows.append(d)?;
    }
    out.set_item("blocks", blocks)?;
    out.set_item("motion", motion)?;
    out.set_item("census", census)?;
    out.set_item("windows", windows)?;
    out.set_item("truncation", r.truncation.clone())?;
    out.set_item("records", r.records)?;
    out.set_item("packet_rows", r.packet_rows)?;
    out.set_item("warmup_seconds", r.warmup_seconds)?;
    Ok(out)
}

fn get<'py, T: FromPyObject<'py>>(params: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    params
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing reduction parameter {key:?}")))?
        .extract()
}

/// Reduce one segment. See `monad_knowledge.lake.native` for the contract.
///
/// `params` carries `block_seconds`, `n_groups`, `warmup_seconds`,
/// `min_packets`, `batch_size`, `motion_seconds`, `origin_policy`
/// ("segment" | "utc") and `window_index`. `packet_sink` is called with one
/// dict of column buffers per batch.
#[pyfunction]
#[pyo3(signature = (path, kind, segment_index, params, packet_sink))]
fn reduce_segment<'py>(
    py: Python<'py>,
    path: &str,
    kind: &str,
    segment_index: Option<i64>,
    params: &Bound<'py, PyDict>,
    packet_sink: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let origin: String = get(params, "origin_policy")?;
    let utc_origin = match origin.as_str() {
        "segment" => false,
        "utc" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "origin_policy must be 'segment' or 'utc', not {other:?}"
            )))
        }
    };
    let p = Params {
        block_seconds: get(params, "block_seconds")?,
        n_groups: get(params, "n_groups")?,
        warmup_seconds: get(params, "warmup_seconds")?,
        min_packets: get(params, "min_packets")?,
        batch_size: get(params, "batch_size")?,
        motion_seconds: get(params, "motion_seconds")?,
        utc_origin,
        window_index: get(params, "window_index")?,
        segment_index,
    };
    let kind = Kind::parse(kind).map_err(PyValueError::new_err)?;
    let mut sink_err: Option<PyErr> = None;
    let result = reduce::reduce_segment(Path::new(path), kind, p, |batch| {
        let call = batch_dict(py, &batch).and_then(|d| packet_sink.call1((d,)).map(|_| ()));
        call.map_err(|e| {
            let msg = e.to_string();
            sink_err = Some(e);
            msg
        })
    });
    match result {
        Ok(r) => result_dict(py, &r),
        Err(msg) => Err(sink_err
            .take()
            .unwrap_or_else(|| PyValueError::new_err(msg))),
    }
}

/// The reducer's semantic version (part of the lake's feature specification).
#[pyfunction]
fn version() -> &'static str {
    crate::REDUCER_VERSION
}

/// The `REDUCTION_SCHEMA` this build ports.
#[pyfunction]
fn reduction_schema() -> i64 {
    crate::REDUCTION_SCHEMA
}

#[pymodule]
fn csi_reduce(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(reduce_segment, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(reduction_schema, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
