//! One segment to packets, blocks, motion, census and window rows.
//!
//! A port of `monad_knowledge.lake.ingest.decode_segment` at reduction schema
//! 7, statement for statement where the order of operations can change a
//! value. The comments here say WHAT each step is; WHY it is so lives once, in
//! the Python reference, and is not repeated.

use std::f64::consts::PI;
use std::path::Path;

use indexmap::IndexMap;

use crate::frame::{self, Frame, FEATURE_BATCH};
use crate::npcompat as np;
use crate::source::{Kind, Records};

/// `ingest.SUBWINDOW_SECONDS`.
pub const SUBWINDOW_SECONDS: f64 = 0.25;
/// `ingest._MOTION_MIN_PACKETS`.
const MOTION_MIN_PACKETS: usize = 100;
/// `ingest._OPEN_BLOCKS`.
const OPEN_BLOCKS: i64 = 3;
/// `ingest._OPEN_SUBWINDOWS`.
const OPEN_SUBWINDOWS: i64 = 2;
/// `ingest._DOPPLER_BANDS`, in order: index 0 is "20_60", index 1 is "5_20".
pub const DOPPLER_BANDS: [(&str, f64, f64); 2] = [("20_60", 20.0, 60.0), ("5_20", 5.0, 20.0)];
/// `ingest._CHANGE_PENALTY`.
const CHANGE_PENALTY: f64 = 0.25;

/// The reduction parameters (`sync.decode_params`).
#[derive(Debug, Clone)]
pub struct Params {
    pub block_seconds: f64,
    pub n_groups: usize,
    pub warmup_seconds: f64,
    pub min_packets: usize,
    pub batch_size: usize,
    pub motion_seconds: f64,
    /// `true` for `origin_policy = "utc"`.
    pub utc_origin: bool,
    pub window_index: bool,
    /// The segment's index; `None` for an unsegmented session.
    pub segment_index: Option<i64>,
}

/// A record class: `(ntone, modulation)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Class {
    pub ntone: u16,
    pub modulation: Option<String>,
}

/// One column-batch of `packets` rows. Validity vectors are 1 for a value, 0 for NULL.
#[derive(Debug, Default)]
pub struct PacketBatch {
    pub dt_days: Vec<i32>,
    pub t_ns: Vec<i64>,
    pub ticks: Vec<i64>,
    pub mono_us: Vec<i64>,
    pub mono_valid: Vec<u8>,
    pub seq: Vec<i64>,
    /// Index into [Reduction::macs].
    pub smac: Vec<u32>,
    pub ntone: Vec<i32>,
    /// Index into [Reduction::modulations]; -1 for NULL.
    pub modulation: Vec<i32>,
    pub bandwidth_code: Vec<i16>,
    pub antenna_sel: Vec<i16>,
    pub bw_valid: Vec<u8>,
    pub n_chains: Vec<i32>,
    pub rssi: Vec<i16>,
    pub rssi_valid: Vec<u8>,
    pub rssi_a: Vec<i16>,
    pub rssi_a_valid: Vec<u8>,
    pub rssi_b: Vec<i16>,
    pub rssi_b_valid: Vec<u8>,
    /// List offsets (len = rows + 1) into `chain_rms_values`.
    pub chain_rms_offsets: Vec<i32>,
    pub chain_rms_values: Vec<f64>,
    pub chain_rms_valid: Vec<u8>,
    pub ratio_amp_db: Vec<f64>,
    pub ratio_amp_valid: Vec<u8>,
    pub ratio_phase_rad: Vec<f64>,
    pub ratio_phase_valid: Vec<u8>,
    pub tone_swapped: Vec<u8>,
    pub tone_swapped_valid: Vec<u8>,
    /// The `smac` and `modulation` dictionaries as they stand at this batch.
    /// They only grow, so they name every code above.
    pub macs: Vec<String>,
    pub modulations: Vec<String>,
}

impl PacketBatch {
    fn new() -> Self {
        PacketBatch {
            chain_rms_offsets: vec![0],
            ..Default::default()
        }
    }
    pub fn len(&self) -> usize {
        self.t_ns.len()
    }
    pub fn is_empty(&self) -> bool {
        self.t_ns.is_empty()
    }
}

/// One `blocks` row's values; the Python side adds the keys and timestamps.
#[derive(Debug, Clone)]
pub struct BlockOut {
    pub class: usize,
    pub start_ns: i64,
    pub tone_group: usize,
    pub tone_lo: usize,
    pub tone_hi: usize,
    pub n_packets: usize,
    pub amp_median: Option<f64>,
    pub phase_circmedian: f64,
    pub amp_s: f64,
    pub ratio_amp_sd_db: Option<f64>,
    pub doppler: [Option<f64>; 2],
    pub minutes_since_segment_start: Option<f64>,
    pub inj_n_packets: usize,
    pub inj_ratio_amp_sd_db: Option<f64>,
    pub inj_doppler: [Option<f64>; 2],
}

/// One `motion` row's values.
#[derive(Debug, Clone)]
pub struct MotionOut {
    pub class: usize,
    pub start_ns: i64,
    pub n_packets: usize,
    pub n_subwindows: usize,
    pub ratio_amp_sd_db: Option<f64>,
    pub minutes_since_segment_start: Option<f64>,
    pub doppler_mean: [Option<f64>; 2],
    pub doppler_max: [Option<f64>; 2],
    pub inj_n_packets: usize,
    pub inj_n_subwindows: usize,
    pub inj_ratio_amp_sd_db: Option<f64>,
    pub inj_doppler_mean: [Option<f64>; 2],
    pub inj_doppler_max: [Option<f64>; 2],
    pub inj_n_steps: usize,
    pub inj_ratio_amp_sd_db_corr: Option<f64>,
    pub inj_n_swaps: usize,
    pub inj_n_offsets: usize,
    pub inj_dphi_mean: Option<f64>,
    pub inj_damp_mean: Option<f64>,
    pub inj_c2_mean: Option<f64>,
    pub inj_rssi_mean: Option<f64>,
    pub inj_rssi_sd: Option<f64>,
    pub inj_dab_mean: Option<f64>,
    pub inj_dab_sd: Option<f64>,
}

/// One census entry, in first-seen class order.
#[derive(Debug, Clone)]
pub struct CensusOut {
    pub class: usize,
    pub n_records: usize,
    pub n_rx: u8,
    pub n_tx: u8,
    /// The set of bandwidths seen, sorted.
    pub bandwidth_mhz: Vec<u32>,
}

/// One IP-164 window-index row.
#[derive(Debug, Clone)]
pub struct WindowOut {
    pub class: usize,
    pub n_rx: u8,
    pub n_tx: u8,
    pub window_start_ns: i64,
    pub first_ordinal: i64,
    pub last_ordinal: i64,
    pub n_frames: i64,
}

/// Everything a segment reduces to, except the packets, which go to the sink.
#[derive(Debug, Default)]
pub struct Reduction {
    pub classes: Vec<Class>,
    pub macs: Vec<String>,
    pub modulations: Vec<String>,
    pub blocks: Vec<BlockOut>,
    pub motion: Vec<MotionOut>,
    pub census: Vec<CensusOut>,
    pub windows: Vec<WindowOut>,
    pub truncation: Option<String>,
    pub records: u64,
    pub packet_rows: u64,
    pub warmup_seconds: f64,
}

#[derive(Default)]
struct OpenBlock {
    amp_db: Vec<f64>,
    phase: Vec<f32>,
    per_packet: Vec<f64>,
    n_packets: usize,
}

#[derive(Default)]
struct SubWindow {
    t_ns: Vec<i64>,
    phase: Vec<f64>,
    inj_t_ns: Vec<i64>,
    inj_phase: Vec<f64>,
}

#[derive(Default)]
struct MotionAcc {
    amp: Vec<f64>,
    powers: [Vec<f64>; 2],
    n_packets: usize,
    inj_amp: Vec<f64>,
    inj_powers: [Vec<f64>; 2],
    inj_n_packets: usize,
    inj_amp_corr: Vec<f64>,
    inj_c2_sum: f64,
    inj_n_phase: usize,
    inj_rssi: Vec<f64>,
    inj_dab: Vec<f64>,
    inj_n_steps: usize,
    inj_n_swaps: usize,
    inj_n_offsets: usize,
    inj_dphi_sum: f64,
    inj_damp_sum: f64,
}

#[derive(Default)]
struct BlockScalars {
    amp: Vec<f64>,
    powers: [Vec<f64>; 2],
    inj_amp: Vec<f64>,
    inj_powers: [Vec<f64>; 2],
    inj_n_packets: usize,
}

/// `ingest._Orientation`.
#[derive(Default)]
struct Orientation {
    swapped_parity: bool,
    offset: bool,
    prev: Option<(f64, f64)>,
}

impl Orientation {
    fn sigma(&self) -> f64 {
        if self.swapped_parity {
            -1.0
        } else {
            1.0
        }
    }

    /// `ingest._orient`: advance by one frame as read; `(swapped, offset)` for this step.
    fn step(&mut self, phase: f64, amp: f64) -> (bool, bool) {
        let mut swap = false;
        let mut off = false;
        if let Some((p0, a0)) = self.prev {
            let cand = [
                np::wrap(phase - p0).abs() + (amp - a0).abs(),
                np::wrap(phase - p0 - PI).abs() + (amp - a0).abs() + CHANGE_PENALTY,
                np::wrap(phase + p0).abs() + (amp + a0).abs() + CHANGE_PENALTY,
                np::wrap(phase + p0 - PI).abs() + (amp + a0).abs() + CHANGE_PENALTY,
            ];
            // `min(range(4), key=...)`: the first index of the minimum.
            let mut best = 0;
            for k in 1..4 {
                if cand[k] < cand[best] {
                    best = k;
                }
            }
            swap = best >= 2;
            off = best == 1 || best == 3;
            if swap {
                self.swapped_parity = !self.swapped_parity;
            }
            if off {
                self.offset = !self.offset;
            }
        }
        self.prev = Some((phase, amp));
        (swap, off)
    }

    /// `ingest._corrected`.
    fn corrected(&self, phase: f64, amp: f64) -> (f64, f64) {
        let s = self.sigma();
        (
            np::wrap(s * (phase - if self.offset { PI } else { 0.0 })),
            s * amp,
        )
    }
}

/// `ingest._band_grid`.
fn band_grid(lo: f64, hi: f64, seconds: f64) -> (Vec<f64>, Vec<f64>) {
    let count = 12usize.max(((hi - lo) * (seconds / 2.0)) as usize);
    let freqs = np::linspace(lo, hi, count);
    let two_pi = 2.0 * PI;
    let omega = freqs.iter().map(|f| two_pi * f).collect();
    (freqs, omega)
}

/// `ingest._lombscargle`: the unnormalised Press & Rybicki periodogram.
fn lombscargle(t: &[f64], y: &[f64], omega: &[f64]) -> Vec<f64> {
    let n = t.len();
    let mut out = Vec::with_capacity(omega.len());
    let mut buf_a = vec![0.0; n];
    let mut buf_b = vec![0.0; n];
    for &w in omega {
        let wt: Vec<f64> = t.iter().map(|&ti| w * ti).collect();
        for s in 0..n {
            let two = 2.0 * wt[s];
            buf_a[s] = two.sin();
            buf_b[s] = two.cos();
        }
        let tau = 0.5 * np::sum(&buf_a).atan2(np::sum(&buf_b));
        let c: Vec<f64> = wt.iter().map(|&x| (x - tau).cos()).collect();
        let sn: Vec<f64> = wt.iter().map(|&x| (x - tau).sin()).collect();
        for s in 0..n {
            buf_a[s] = c[s] * c[s];
            buf_b[s] = sn[s] * sn[s];
        }
        let cc = np::sum(&buf_a);
        let ss = np::sum(&buf_b);
        for s in 0..n {
            buf_a[s] = y[s] * c[s];
            buf_b[s] = y[s] * sn[s];
        }
        let yc = np::sum(&buf_a);
        let ys = np::sum(&buf_b);
        let first = if cc > 0.0 { yc * yc / cc } else { 0.0 };
        let second = if ss > 0.0 { ys * ys / ss } else { 0.0 };
        out.push(0.5 * (first + second));
    }
    out
}

/// `ingest._band_power`: per-frame band power of a phase series, or `None`.
fn band_power(t_ns: &[i64], phase: &[f64]) -> Option<[f64; 2]> {
    let n = t_ns.len();
    if n < 30 {
        return None;
    }
    let t0 = t_ns[0] as f64;
    let t: Vec<f64> = t_ns.iter().map(|&v| (v as f64 - t0) / 1e9).collect();
    let y = np::unwrap(phase);
    let t_mean = np::mean(&t);
    let dt: Vec<f64> = t.iter().map(|v| v - t_mean).collect();
    let sq: Vec<f64> = dt.iter().map(|d| d * d).collect();
    let var = np::sum(&sq);
    if var <= 0.0 {
        return None;
    }
    let y_mean = np::mean(&y);
    let prod: Vec<f64> = dt.iter().zip(&y).map(|(d, v)| d * (v - y_mean)).collect();
    let slope = np::sum(&prod) / var;
    let y: Vec<f64> = y
        .iter()
        .zip(&dt)
        .map(|(v, d)| v - (y_mean + slope * d))
        .collect();
    if np::std(&y, 0) == 0.0 {
        return None;
    }
    let mut out = [0.0; 2];
    for (b, &(_, lo, hi)) in DOPPLER_BANDS.iter().enumerate() {
        let (freqs, omega) = band_grid(lo, hi, SUBWINDOW_SECONDS);
        out[b] = np::trapezoid(&lombscargle(&t, &y, &omega), &freqs) / n as f64;
    }
    Some(out)
}

/// `datetime.fromtimestamp(ts / 1e9, tz=utc).date()` as days since the epoch:
/// the float second, rounded to the microsecond half-even as Python does.
pub fn python_date_days(ts: i64) -> i32 {
    let x = ts as f64 / 1e9;
    let mut t = x.trunc();
    let us = ((x - t) * 1e6).round_ties_even();
    if us >= 1e6 {
        t += 1.0;
    } else if us < 0.0 {
        t -= 1.0;
    }
    (t / 86400.0).floor() as i32
}

/// Append one sub-window's two band powers to their per-band lists.
fn push_bands(dst: &mut [Vec<f64>; 2], powers: [f64; 2]) {
    for (list, p) in dst.iter_mut().zip(powers) {
        list.push(p);
    }
}

fn std1(v: &[f64]) -> Option<f64> {
    (v.len() > 1).then(|| np::std(v, 1))
}

/// `int(x)` of a Python float: truncation toward zero.
fn py_int(x: f64) -> i64 {
    x as i64
}

/// The reducer's state for one segment.
struct State {
    p: Params,
    sub_ns: i64,
    motion_ns: i64,
    block_ns: i64,
    block_div: i64,
    origin_ns: Option<i64>,
    first_ns: Option<i64>,
    newest_index: i64,
    newest_sub: i64,
    groups_for: IndexMap<usize, Vec<(usize, usize)>>,
    open_blocks: IndexMap<(usize, usize, i64), OpenBlock>,
    open_subs: IndexMap<(usize, i64), SubWindow>,
    open_motion: IndexMap<(usize, i64), MotionAcc>,
    open_scalars: IndexMap<(usize, i64), BlockScalars>,
    out: Reduction,
}

impl State {
    fn since_segment_start(&self, start_ns: i64) -> Option<f64> {
        self.first_ns.map(|f| (start_ns - f) as f64 / 6e10)
    }

    fn close_subs(&mut self, upto: Option<i64>) {
        let keys: Vec<(usize, i64)> = self
            .open_subs
            .keys()
            .filter(|k| upto.is_none_or(|u| k.1 < u))
            .copied()
            .collect();
        for key in keys {
            let (cls, index) = key;
            let buf = self.open_subs.shift_remove(&key).expect("key listed");
            let Some(powers) = band_power(&buf.t_ns, &buf.phase) else {
                continue;
            };
            let origin = self.origin_ns.expect("origin set before any sub-window");
            let start_ns = origin + index * self.sub_ns;
            let row = self
                .open_motion
                .entry((cls, (start_ns - origin).div_euclid(self.motion_ns)))
                .or_default();
            push_bands(&mut row.powers, powers);
            let inj = band_power(&buf.inj_t_ns, &buf.inj_phase);
            if let Some(inj) = inj {
                push_bands(&mut row.inj_powers, inj);
            }
            let scal = self
                .open_scalars
                .entry((cls, (start_ns - origin).div_euclid(self.block_ns)))
                .or_default();
            push_bands(&mut scal.powers, powers);
            if let Some(inj) = inj {
                push_bands(&mut scal.inj_powers, inj);
            }
        }
    }

    fn flush_motion(&mut self, upto: Option<i64>) {
        let keys: Vec<(usize, i64)> = self
            .open_motion
            .keys()
            .filter(|k| upto.is_none_or(|u| k.1 < u))
            .copied()
            .collect();
        for key in keys {
            let (cls, index) = key;
            let row = self.open_motion.shift_remove(&key).expect("key listed");
            let start_ns = self.origin_ns.expect("origin set") + index * self.motion_ns;
            let scored = row.powers[0].len();
            let enough = row.n_packets >= MOTION_MIN_PACKETS && scored > 0;
            let inj_scored = row.inj_powers[0].len();
            let inj_enough = row.inj_n_packets >= MOTION_MIN_PACKETS && inj_scored > 0;
            let pick =
                |ok: bool, v: &Vec<f64>, f: fn(&[f64]) -> f64| (ok && !v.is_empty()).then(|| f(v));
            let steps = inj_enough && row.inj_n_steps > 0;
            let mean_of = |ok: bool, v: &Vec<f64>| (ok && !v.is_empty()).then(|| np::mean(v));
            let sd_of = |ok: bool, v: &Vec<f64>| if ok { std1(v) } else { None };
            self.out.motion.push(MotionOut {
                class: cls,
                start_ns,
                n_packets: row.n_packets,
                n_subwindows: scored,
                ratio_amp_sd_db: sd_of(enough, &row.amp),
                minutes_since_segment_start: self.since_segment_start(start_ns),
                doppler_mean: [
                    pick(enough, &row.powers[0], np::mean),
                    pick(enough, &row.powers[1], np::mean),
                ],
                doppler_max: [
                    pick(enough, &row.powers[0], np::max),
                    pick(enough, &row.powers[1], np::max),
                ],
                inj_n_packets: row.inj_n_packets,
                inj_n_subwindows: inj_scored,
                inj_ratio_amp_sd_db: sd_of(inj_enough, &row.inj_amp),
                inj_doppler_mean: [
                    pick(inj_enough, &row.inj_powers[0], np::mean),
                    pick(inj_enough, &row.inj_powers[1], np::mean),
                ],
                inj_doppler_max: [
                    pick(inj_enough, &row.inj_powers[0], np::max),
                    pick(inj_enough, &row.inj_powers[1], np::max),
                ],
                inj_n_steps: row.inj_n_steps,
                inj_ratio_amp_sd_db_corr: sd_of(inj_enough, &row.inj_amp_corr),
                inj_n_swaps: row.inj_n_swaps,
                inj_n_offsets: row.inj_n_offsets,
                inj_dphi_mean: steps.then(|| row.inj_dphi_sum / row.inj_n_steps as f64),
                inj_damp_mean: steps.then(|| row.inj_damp_sum / row.inj_n_steps as f64),
                inj_c2_mean: (inj_enough && row.inj_n_phase > 0)
                    .then(|| row.inj_c2_sum / row.inj_n_phase as f64),
                inj_rssi_mean: mean_of(inj_enough, &row.inj_rssi),
                inj_rssi_sd: sd_of(inj_enough, &row.inj_rssi),
                inj_dab_mean: mean_of(inj_enough, &row.inj_dab),
                inj_dab_sd: sd_of(inj_enough, &row.inj_dab),
            });
        }
    }

    fn flush(&mut self, upto: Option<i64>) {
        self.close_subs(upto.map(|u| (u * self.block_ns).div_euclid(self.sub_ns)));
        let keys: Vec<(usize, usize, i64)> = self
            .open_blocks
            .keys()
            .filter(|k| upto.is_none_or(|u| k.2 < u))
            .copied()
            .collect();
        for key in keys {
            let (cls, group, index) = key;
            let buf = self.open_blocks.shift_remove(&key).expect("key listed");
            if buf.n_packets < self.p.min_packets {
                continue;
            }
            let origin = self.origin_ns.expect("origin set");
            let start_ns = origin + py_int(index as f64 * self.p.block_seconds * 1e9);
            let ntone = self.out.classes[cls].ntone as usize;
            let (lo, hi) = self.groups_for[&ntone][group];
            let scal = self.open_scalars.get(&(cls, index));
            let amp_s = if buf.per_packet.len() > 1 {
                np::std(&buf.per_packet, 1)
            } else {
                0.0
            };
            let ratio_sd = scal.and_then(|s| std1(&s.amp));
            let doppler = [0, 1].map(|b| {
                scal.and_then(|s| (!s.powers[b].is_empty()).then(|| np::mean(&s.powers[b])))
            });
            let inj_n = scal.map_or(0, |s| s.inj_n_packets);
            let inj_sd = scal.and_then(|s| std1(&s.inj_amp));
            let inj_doppler = [0, 1].map(|b| {
                scal.and_then(|s| (!s.inj_powers[b].is_empty()).then(|| np::mean(&s.inj_powers[b])))
            });
            let minutes = self.since_segment_start(start_ns);
            self.out.blocks.push(BlockOut {
                class: cls,
                start_ns,
                tone_group: group,
                tone_lo: lo,
                tone_hi: hi - 1,
                n_packets: buf.n_packets,
                amp_median: np::nanmedian(&buf.amp_db),
                phase_circmedian: np::circular_median32(&buf.phase),
                amp_s,
                ratio_amp_sd_db: ratio_sd,
                doppler,
                minutes_since_segment_start: minutes,
                inj_n_packets: inj_n,
                inj_ratio_amp_sd_db: inj_sd,
                inj_doppler,
            });
        }
        let gone: Vec<(usize, i64)> = self
            .open_scalars
            .keys()
            .filter(|k| upto.is_none_or(|u| k.1 < u))
            .copied()
            .collect();
        for key in gone {
            self.open_scalars.shift_remove(&key);
        }
        self.flush_motion(upto.map(|u| (u * self.block_ns).div_euclid(self.motion_ns)));
    }
}

fn class_id(classes: &mut Vec<Class>, cache: &mut IndexMap<Class, usize>, class: Class) -> usize {
    if let Some(&id) = cache.get(&class) {
        return id;
    }
    let id = classes.len();
    classes.push(class.clone());
    cache.insert(class, id);
    id
}

fn intern(list: &mut Vec<String>, cache: &mut IndexMap<String, u32>, value: String) -> u32 {
    if let Some(&id) = cache.get(&value) {
        return id;
    }
    let id = list.len() as u32;
    list.push(value.clone());
    cache.insert(value, id);
    id
}

/// Reduce one capture. Packet batches go to *sink* as they fill.
pub fn reduce_segment(
    path: &Path,
    kind: Kind,
    params: Params,
    mut sink: impl FnMut(PacketBatch) -> Result<(), String>,
) -> Result<Reduction, String> {
    let mut warmup_seconds = params.warmup_seconds;
    if !matches!(params.segment_index, None | Some(1)) {
        warmup_seconds = 0.0;
    }
    let sub_ns = 1i64.max(py_int(SUBWINDOW_SECONDS * 1e9));
    let motion_ns = 1i64.max(py_int(params.motion_seconds * 1e9));
    let block_ns = 1i64.max(py_int(params.block_seconds * 1e9));
    if params.utc_origin && (block_ns % motion_ns != 0 || motion_ns % sub_ns != 0) {
        return Err(format!(
            "origin_policy='utc' needs block_seconds ({}) to be a whole number of {} s rows and those a whole number of {} s sub-windows",
            params.block_seconds, params.motion_seconds, SUBWINDOW_SECONDS
        ));
    }
    let block_div = py_int(params.block_seconds * 1e9);
    let mut st = State {
        p: params.clone(),
        sub_ns,
        motion_ns,
        block_ns,
        block_div,
        origin_ns: None,
        first_ns: None,
        newest_index: -1,
        newest_sub: -1,
        groups_for: IndexMap::new(),
        open_blocks: IndexMap::new(),
        open_subs: IndexMap::new(),
        open_motion: IndexMap::new(),
        open_scalars: IndexMap::new(),
        out: Reduction {
            warmup_seconds,
            ..Reduction::default()
        },
    };
    let mut class_cache: IndexMap<Class, usize> = IndexMap::new();
    let mut mac_cache: IndexMap<String, u32> = IndexMap::new();
    let mut mod_cache: IndexMap<String, u32> = IndexMap::new();
    let mut census: IndexMap<usize, (usize, u8, u8, std::collections::BTreeSet<u32>)> =
        IndexMap::new();
    let mut windows: IndexMap<(usize, i64), [i64; 3]> = IndexMap::new();
    let mut window_meta: IndexMap<usize, (u8, u8)> = IndexMap::new();
    let mut inj_prev: IndexMap<usize, (f64, f64)> = IndexMap::new();
    let mut inj_orient: IndexMap<usize, Orientation> = IndexMap::new();
    let mut unwrap = csiq::FtmUnwrapper::new();
    let mut pending = PacketBatch::new();
    let mut ordinal: i64 = 0;

    let mut records = Records::open(path, kind)?;
    let mut batch = Vec::with_capacity(FEATURE_BATCH);
    loop {
        batch.clear();
        while batch.len() < FEATURE_BATCH {
            match records.next_record()? {
                Some(r) => batch.push(r),
                None => break,
            }
        }
        if batch.is_empty() {
            break;
        }
        let frames: Vec<Frame> = frame::frame_features(&batch, params.n_groups);
        for (rec, fr) in batch.iter().zip(frames) {
            let this_ordinal = ordinal;
            ordinal += 1;
            st.out.records += 1;
            let ticks = unwrap.push(rec.ftm);
            let ts = rec.unix_ts_ns as i64;
            let modulation = rec.phy.map(|p| frame::modulation_label(p.modulation));
            let cls = class_id(
                &mut st.out.classes,
                &mut class_cache,
                Class {
                    ntone: rec.ntone,
                    modulation: modulation.clone(),
                },
            );
            if params.window_index && ts != 0 {
                let key = (cls, ts - ts.rem_euclid(block_ns));
                match windows.get_mut(&key) {
                    None => {
                        windows.insert(key, [this_ordinal, this_ordinal, 1]);
                        window_meta.entry(cls).or_insert((rec.nrx, rec.ntx));
                    }
                    Some(span) => {
                        span[1] = this_ordinal;
                        span[2] += 1;
                    }
                }
            }
            let entry = census
                .entry(cls)
                .or_insert((0, rec.nrx, rec.ntx, Default::default()));
            entry.0 += 1;
            if let Some(bw) = rec.bw_antsel.and_then(|b| b.bandwidth.mhz()) {
                entry.3.insert(bw);
            }
            if ts == 0 {
                continue;
            }
            let first_ns = *st.first_ns.get_or_insert(ts);

            let amp_scalar = fr.amp_scalar;
            let phase_scalar = fr.phase_scalar;
            let mac = &rec.src_mac;
            let injector = frame::is_injector(mac);
            let nrx = rec.nrx as usize;
            let rssi: &[i16] = &rec.rssi[..rec.rssi.len().min(nrx)];
            let rssi_max = (!rssi.is_empty() && rssi.iter().any(|&v| v != 0))
                .then(|| *rssi.iter().max().unwrap());
            let rssi_a = rssi.first().copied().filter(|&v| v != 0);
            let rssi_b = rssi.get(1).copied().filter(|&v| v != 0);

            // -- the packet row ------------------------------------------------
            if pending.len() >= params.batch_size {
                st.out.packet_rows += pending.len() as u64;
                pending.macs = st.out.macs.clone();
                pending.modulations = st.out.modulations.clone();
                sink(std::mem::replace(&mut pending, PacketBatch::new()))?;
            }
            let mac_id = intern(&mut st.out.macs, &mut mac_cache, frame::mac_string(mac));
            let mod_id = match &modulation {
                Some(m) => intern(&mut st.out.modulations, &mut mod_cache, m.clone()) as i32,
                None => -1,
            };
            let pb = &mut pending;
            pb.dt_days.push(python_date_days(ts));
            pb.t_ns.push(ts);
            pb.ticks.push(ticks as i64);
            pb.mono_us.push(rec.mono_us.unwrap_or(0) as i64);
            pb.mono_valid.push(u8::from(rec.mono_us.is_some()));
            pb.seq.push(rec.seq as i64);
            pb.smac.push(mac_id);
            pb.ntone.push(rec.ntone as i32);
            pb.modulation.push(mod_id);
            match rec.bw_antsel {
                Some(b) => {
                    pb.bandwidth_code.push(b.bandwidth.to_code() as i16);
                    pb.antenna_sel.push(b.antenna_sel as i16);
                    pb.bw_valid.push(1);
                }
                None => {
                    pb.bandwidth_code.push(0);
                    pb.antenna_sel.push(0);
                    pb.bw_valid.push(0);
                }
            }
            pb.n_chains.push(rec.nrx as i32 * rec.ntx as i32);
            for (col, valid, v) in [
                (&mut pb.rssi, &mut pb.rssi_valid, rssi_max),
                (&mut pb.rssi_a, &mut pb.rssi_a_valid, rssi_a),
                (&mut pb.rssi_b, &mut pb.rssi_b_valid, rssi_b),
            ] {
                col.push(v.unwrap_or(0));
                valid.push(u8::from(v.is_some()));
            }
            match &fr.chain_amp_rms {
                Some(v) => {
                    pb.chain_rms_values.extend_from_slice(v);
                    pb.chain_rms_valid.push(1);
                }
                None => pb.chain_rms_valid.push(0),
            }
            pb.chain_rms_offsets.push(pb.chain_rms_values.len() as i32);
            pb.ratio_amp_db.push(amp_scalar.unwrap_or(0.0));
            pb.ratio_amp_valid.push(u8::from(amp_scalar.is_some()));
            pb.ratio_phase_rad.push(phase_scalar.unwrap_or(0.0));
            pb.ratio_phase_valid.push(u8::from(phase_scalar.is_some()));
            pb.tone_swapped
                .push(u8::from(fr.tone_swapped == Some(true)));
            pb.tone_swapped_valid
                .push(u8::from(fr.tone_swapped.is_some()));

            // -- orientation (reduction schema 7) ---------------------------------
            let (mut phase_c, mut amp_c) = (phase_scalar, amp_scalar);
            let (mut swapped, mut offset) = (false, false);
            if let (Some(ph), Some(am), true) = (phase_scalar, amp_scalar, injector) {
                let state = inj_orient.entry(cls).or_default();
                (swapped, offset) = state.step(ph, am);
                let (pc, ac) = state.corrected(ph, am);
                phase_c = Some(pc);
                amp_c = Some(ac);
            }

            let (Some(amp_db), Some(phase)) = (fr.amp_db.as_ref(), fr.phase.as_ref()) else {
                continue;
            };
            if warmup_seconds > 0.0 {
                // Python compares the int `ts` with a float exactly; the float
                // here is integral at epoch-nanosecond magnitudes.
                let thr = first_ns as f64 + warmup_seconds * 1e9;
                if (ts as i128) < (thr as i128)
                    || ((ts as i128) == (thr as i128) && thr.fract() > 0.0)
                {
                    continue;
                }
            }
            let origin = *st.origin_ns.get_or_insert(if params.utc_origin {
                ts - ts.rem_euclid(block_ns)
            } else {
                ts
            });
            let index = (ts - origin).div_euclid(st.block_div);
            if index > st.newest_index {
                st.newest_index = index;
                st.flush(Some(index - OPEN_BLOCKS));
            }
            // -- the whole-band scalar path ------------------------------------
            let motion_key = (cls, (ts - origin).div_euclid(motion_ns));
            let scalar_key = (cls, index);
            if let Some(a) = amp_scalar {
                let row = st.open_motion.entry(motion_key).or_default();
                row.amp.push(a);
                if injector {
                    row.inj_amp.push(a);
                    if let Some(ac) = amp_c {
                        row.inj_amp_corr.push(ac);
                    }
                }
                let scal = st.open_scalars.entry(scalar_key).or_default();
                scal.amp.push(a);
                if injector {
                    scal.inj_amp.push(a);
                }
            } else {
                st.open_motion.entry(motion_key).or_default();
            }
            if let Some(pc) = phase_c {
                let sub_index = (ts - origin).div_euclid(sub_ns);
                let buf = st.open_subs.entry((cls, sub_index)).or_default();
                buf.t_ns.push(ts);
                buf.phase.push(pc);
                if injector {
                    buf.inj_t_ns.push(ts);
                    buf.inj_phase.push(pc);
                }
                if sub_index > st.newest_sub {
                    st.newest_sub = sub_index;
                    st.close_subs(Some(sub_index - OPEN_SUBWINDOWS));
                }
            }
            {
                let row = st.open_motion.entry(motion_key).or_default();
                row.n_packets += 1;
                if injector {
                    row.inj_n_packets += 1;
                    // `ingest._inj_frame_features`.
                    if let Some(r) = rssi_max {
                        row.inj_rssi.push(r as f64);
                    }
                    if let (Some(a), Some(b)) = (rssi_a, rssi_b) {
                        row.inj_dab.push((a - b) as f64);
                    }
                    if let (Some(ph), Some(am)) = (phase_c, amp_c) {
                        row.inj_c2_sum += (2.0 * ph).cos();
                        row.inj_n_phase += 1;
                        let last = inj_prev.insert(cls, (ph, am));
                        if let Some((p0, a0)) = last {
                            row.inj_n_steps += 1;
                            row.inj_n_swaps += usize::from(swapped);
                            row.inj_n_offsets += usize::from(offset);
                            row.inj_dphi_sum += np::wrap(ph - p0).abs();
                            row.inj_damp_sum += (am - a0).abs();
                        }
                    }
                }
            }
            if injector {
                st.open_scalars.entry(scalar_key).or_default().inj_n_packets += 1;
            }

            let ntone = rec.ntone as usize;
            let groups = st
                .groups_for
                .entry(ntone)
                .or_insert_with(|| frame::tone_groups(ntone, params.n_groups))
                .clone();
            for (g, &(lo, hi)) in groups.iter().enumerate() {
                let buf = st.open_blocks.entry((cls, g, index)).or_default();
                buf.amp_db.extend_from_slice(&amp_db[lo..hi]);
                buf.phase.extend_from_slice(&phase[lo..hi]);
                if let Some(pp) = fr.group_medians.get(g).copied().flatten() {
                    buf.per_packet.push(pp);
                }
                buf.n_packets += 1;
            }
        }
    }
    st.out.truncation = records.truncation.take();
    st.flush(None);
    if !pending.is_empty() {
        st.out.packet_rows += pending.len() as u64;
        pending.macs = st.out.macs.clone();
        pending.modulations = st.out.modulations.clone();
        sink(pending)?;
    }

    for (cls, (n, nrx, ntx, bws)) in census {
        st.out.census.push(CensusOut {
            class: cls,
            n_records: n,
            n_rx: nrx,
            n_tx: ntx,
            bandwidth_mhz: bws.into_iter().collect(),
        });
    }
    let classes = st.out.classes.clone();
    st.out
        .motion
        .sort_by(|a, b| (&classes[a.class], a.start_ns).cmp(&(&classes[b.class], b.start_ns)));
    if params.window_index {
        let mut rows: Vec<WindowOut> = windows
            .into_iter()
            .map(|((cls, start), [first, last, count])| {
                let (n_rx, n_tx) = window_meta[&cls];
                WindowOut {
                    class: cls,
                    n_rx,
                    n_tx,
                    window_start_ns: start,
                    first_ordinal: first,
                    last_ordinal: last,
                    n_frames: count,
                }
            })
            .collect();
        rows.sort_by(|a, b| {
            (&classes[a.class], a.window_start_ns).cmp(&(&classes[b.class], b.window_start_ns))
        });
        st.out.windows = rows;
    }
    Ok(st.out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_dates_round_to_the_microsecond_first() {
        assert_eq!(python_date_days(0), 0);
        assert_eq!(python_date_days(86_400_000_000_000), 1);
        assert_eq!(
            python_date_days(86_399_999_999_999),
            1,
            "rounds up across midnight like fromtimestamp"
        );
    }

    #[test]
    fn orientation_names_a_swap_and_an_offset() {
        let mut o = Orientation::default();
        assert_eq!(o.step(1.2, 2.3), (false, false));
        assert_eq!(o.step(-1.2, -2.3), (true, false));
        let (p, a) = o.corrected(-1.2, -2.3);
        assert!((p - 1.2).abs() < 1e-12 && (a - 2.3).abs() < 1e-12);
        assert_eq!(o.step(-1.2 + PI, -2.3), (false, true));
    }

    #[test]
    fn band_power_needs_thirty_frames() {
        let t: Vec<i64> = (0..29).map(|i| i * 4_000_000).collect();
        assert!(band_power(&t, &vec![0.1; 29]).is_none());
    }
}
