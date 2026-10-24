//! Per-frame features of a batch of records: `ingest._frame_features`.
//!
//! The Python reference computes these per batch of [FEATURE_BATCH] records,
//! grouped by geometry, and the half-band tone-swap guard takes its reference
//! spectrum from the batch. So the batch boundaries are part of the meaning,
//! and this module keeps them.

use csiq::record::Modulation;
use csiq::CsiRecord;
use indexmap::IndexMap;

use crate::npcompat;

/// `ingest.FEATURE_BATCH`.
pub const FEATURE_BATCH: usize = 4096;

/// The half-band tone-swap guard (`ingest._SWAP_*`).
const SWAP_NTONE: usize = 52;
const SWAP_MARGIN: f64 = 0.3;
const SWAP_MIN_GROUP: usize = 50;

/// The per-frame values the reduction needs (`ingest._Frame`).
#[derive(Debug, Clone, Default)]
pub struct Frame {
    /// Per-tone ratio amplitude in dB; NaN where the denominator was zero.
    pub amp_db: Option<Vec<f64>>,
    /// Per-tone ratio phase, float32 as numpy computes it.
    pub phase: Option<Vec<f32>>,
    pub amp_scalar: Option<f64>,
    pub phase_scalar: Option<f64>,
    /// `_nanmedian` of `amp_db` per tone group, in `tone_groups` order.
    pub group_medians: Vec<Option<f64>>,
    pub chain_amp_rms: Option<Vec<f64>>,
    /// True: repaired; false: tested and in order; None: not tested.
    pub tone_swapped: Option<bool>,
}

/// The label the Python reader gives a modulation.
pub fn modulation_label(m: Modulation) -> String {
    match m {
        Modulation::Cck => "cck".into(),
        Modulation::LegacyOfdm => "legacy_ofdm".into(),
        Modulation::Ht => "ht".into(),
        Modulation::Vht => "vht".into(),
        Modulation::He => "he".into(),
        Modulation::Eht => "eht".into(),
        Modulation::Unknown(code) => format!("unknown({code})"),
    }
}

/// `CsiRecord.mac`: `aa:bb:cc:dd:ee:ff`.
pub fn mac_string(mac: &[u8; 6]) -> String {
    format!(
        "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]
    )
}

/// `ingest._is_injector`: the fleet injector prefix `02:6d:6f:6e:`.
pub fn is_injector(mac: &[u8; 6]) -> bool {
    mac[..4] == [0x02, 0x6d, 0x6f, 0x6e]
}

/// `drift.tone_groups`: `np.array_split(arange(ntone), n_groups)`, empty groups dropped,
/// as half-open tone ranges.
pub fn tone_groups(ntone: usize, n_groups: usize) -> Vec<(usize, usize)> {
    let base = ntone / n_groups;
    let extra = ntone % n_groups;
    let mut out = Vec::with_capacity(n_groups);
    let mut lo = 0;
    for g in 0..n_groups {
        let size = base + usize::from(g < extra);
        if size > 0 {
            out.push((lo, lo + size));
        }
        lo += size;
    }
    out
}

/// `ingest._rowcorr` for one row: Pearson correlation with *rc* (already
/// centred) and its sum of squares; NaN for a flat row.
fn rowcorr(x: &[f64], rc: &[f64], rc_ss: f64) -> f64 {
    let m = npcompat::mean(x);
    let xc: Vec<f64> = x.iter().map(|v| v - m).collect();
    let sq: Vec<f64> = xc.iter().map(|v| v * v).collect();
    let den = (npcompat::sum(&sq) * rc_ss).sqrt();
    // `xc @ rc` is a BLAS dot product in numpy; its summation order is the
    // library's, so this is the one sum here not repeated operation for operation.
    let dot: f64 = xc.iter().zip(rc).map(|(a, b)| a * b).sum();
    if den > 0.0 {
        dot / den
    } else {
        f64::NAN
    }
}

/// `ingest._guard_tone_swap`: flag and roll back half-band-swapped records.
///
/// *cplx* holds the group's records as `[member][chain][tone]` interleaved
/// `(re, im)` float32 pairs, and is rolled in place.
fn guard_tone_swap(
    records: &[CsiRecord],
    members: &[usize],
    chains: usize,
    ntone: usize,
    cplx: &mut [f32],
) -> Vec<Option<bool>> {
    let mut verdicts = vec![None; members.len()];
    if chains != 2 || ntone != SWAP_NTONE {
        return verdicts;
    }
    let stride = chains * ntone * 2;
    let mut by_mac: IndexMap<[u8; 6], Vec<usize>> = IndexMap::new();
    for (k, &i) in members.iter().enumerate() {
        let rec = &records[i];
        match rec.phy {
            Some(p) if p.modulation == Modulation::LegacyOfdm => {}
            _ => continue,
        }
        if is_injector(&rec.src_mac) {
            by_mac.entry(rec.src_mac).or_default().push(k);
        }
    }
    let half = SWAP_NTONE / 2;
    for ks in by_mac.values() {
        if ks.len() < SWAP_MIN_GROUP {
            continue;
        }
        // Chain-0 amplitude in dB, `max(|z|, 1)` floored, float64.
        let amp: Vec<Vec<f64>> = ks
            .iter()
            .map(|&k| {
                let base = k * stride;
                (0..ntone)
                    .map(|t| {
                        let re = cplx[base + 2 * t];
                        let im = cplx[base + 2 * t + 1];
                        let mag = re.hypot(im) as f64;
                        20.0 * mag.max(1.0).log10()
                    })
                    .collect()
            })
            .collect();
        let reference: Vec<f64> = (0..ntone)
            .map(|t| {
                let mut col: Vec<f64> = amp.iter().map(|row| row[t]).collect();
                npcompat::median(&mut col)
            })
            .collect();
        let rm = npcompat::mean(&reference);
        let rc: Vec<f64> = reference.iter().map(|v| v - rm).collect();
        let rc_sq: Vec<f64> = rc.iter().map(|v| v * v).collect();
        let rc_ss = npcompat::sum(&rc_sq);
        for (j, &k) in ks.iter().enumerate() {
            let row = &amp[j];
            // `np.roll(amp, half, axis=1)`: out[t] = row[(t - half) mod n].
            let rolled: Vec<f64> = (0..ntone)
                .map(|t| row[(t + ntone - half) % ntone])
                .collect();
            let margin = rowcorr(&rolled, &rc, rc_ss) - rowcorr(row, &rc, rc_ss);
            if margin.is_finite() {
                let swapped = margin > SWAP_MARGIN;
                verdicts[k] = Some(swapped);
                if swapped {
                    let base = k * stride;
                    for c in 0..chains {
                        let off = base + c * ntone * 2;
                        let block: Vec<f32> = cplx[off..off + ntone * 2].to_vec();
                        for t in 0..ntone {
                            let src = (t + ntone - half) % ntone;
                            cplx[off + 2 * t] = block[2 * src];
                            cplx[off + 2 * t + 1] = block[2 * src + 1];
                        }
                    }
                }
            }
        }
    }
    verdicts
}

/// Per-chain RMS of one record as float32 (`ingest._chain_amp_rms` and the batch path).
fn chain_rms(rec: &CsiRecord, chains: usize, ntone: usize) -> Vec<f64> {
    (0..chains)
        .map(|c| {
            let power: Vec<f32> = (0..ntone)
                .map(|t| {
                    let i = 2 * (c * ntone + t);
                    let a = rec.iq[i] as f32;
                    let b = rec.iq[i + 1] as f32;
                    a * a + b * b
                })
                .collect();
            // add.reduce in float32, then a float64 true division by the
            // `intp` count written back into float32, then sqrt in float32.
            let total = npcompat::sum(&power);
            let mean = ((total as f64) / (ntone as f64)) as f32;
            mean.sqrt() as f64
        })
        .collect()
}

/// `ingest._frame_features`: features of a batch, in record order.
pub fn frame_features(records: &[CsiRecord], n_groups: usize) -> Vec<Frame> {
    let mut frames: Vec<Option<Frame>> = vec![None; records.len()];
    let mut geometry: IndexMap<(usize, usize), Vec<usize>> = IndexMap::new();
    for (i, rec) in records.iter().enumerate() {
        let chains = rec.nrx as usize * rec.ntx as usize;
        if chains == 0 || rec.iq.len() != 2 * rec.coeff_count() {
            frames[i] = Some(Frame::default());
            continue;
        }
        if rec.ntone == 0 {
            // Python's per-record path: an empty tone axis divides 0 by 0.
            frames[i] = Some(Frame {
                chain_amp_rms: Some(vec![f64::NAN; chains]),
                ..Frame::default()
            });
            continue;
        }
        geometry
            .entry((rec.ntone as usize, chains))
            .or_default()
            .push(i);
    }

    for (&(ntone, chains), members) in &geometry {
        let rms: Vec<Vec<f64>> = members
            .iter()
            .map(|&i| chain_rms(&records[i], chains, ntone))
            .collect();
        if chains < 2 {
            for (k, &i) in members.iter().enumerate() {
                frames[i] = Some(Frame {
                    chain_amp_rms: Some(rms[k].clone()),
                    ..Frame::default()
                });
            }
            continue;
        }
        // `matrix()`: value = iq[2k+1] + 1j*iq[2k], chain-major blocks of ntone.
        let stride = chains * ntone * 2;
        let mut cplx = vec![0f32; members.len() * stride];
        for (k, &i) in members.iter().enumerate() {
            let iq = &records[i].iq;
            let base = k * stride;
            for c in 0..chains {
                for t in 0..ntone {
                    let src = 2 * (c * ntone + t);
                    let dst = base + 2 * (c * ntone + t);
                    cplx[dst] = iq[src + 1] as f32;
                    cplx[dst + 1] = iq[src] as f32;
                }
            }
        }
        let swapped = guard_tone_swap(records, members, chains, ntone, &mut cplx);
        let groups = tone_groups(ntone, n_groups);
        for (k, &i) in members.iter().enumerate() {
            let base = k * stride;
            let mut amp_db = vec![f64::NAN; ntone];
            let mut phase = vec![f32::NAN; ntone];
            for t in 0..ntone {
                let (nr, ni) = (cplx[base + 2 * t], cplx[base + 2 * t + 1]);
                let (dr, di) = (
                    cplx[base + 2 * (ntone + t)],
                    cplx[base + 2 * (ntone + t) + 1],
                );
                if dr.hypot(di) > 0.0 {
                    let (rr, ri) = npcompat::cdiv32(nr, ni, dr, di);
                    let mag = rr.hypot(ri);
                    if mag > 0.0 {
                        // log10 in float32 into a float64 output, then 20 * in float64.
                        amp_db[t] = 20.0 * (mag.log10() as f64);
                    }
                    phase[t] = ri.atan2(rr);
                }
                // A masked tone keeps the NaN of the `np.full(..., nan)` output.
            }
            let amp_scalar = npcompat::nanmedian(&amp_db);
            // The batch path (all-finite rows) and the scalar path (rows with a
            // NaN tone, finite angles only) are the same arithmetic.
            let pm = npcompat::circular_median32(&phase);
            let group_medians = groups
                .iter()
                .map(|&(lo, hi)| npcompat::nanmedian(&amp_db[lo..hi]))
                .collect();
            frames[i] = Some(Frame {
                amp_db: Some(amp_db),
                phase: Some(phase),
                amp_scalar,
                phase_scalar: if pm.is_finite() { Some(pm) } else { None },
                group_medians,
                chain_amp_rms: Some(rms[k].clone()),
                tone_swapped: swapped[k],
            });
        }
    }
    frames
        .into_iter()
        .map(|f| f.expect("every record gets a frame"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_groups_split_like_numpy() {
        assert_eq!(
            tone_groups(52, 4),
            vec![(0, 13), (13, 26), (26, 39), (39, 52)]
        );
        assert_eq!(tone_groups(10, 4), vec![(0, 3), (3, 6), (6, 8), (8, 10)]);
        assert_eq!(tone_groups(2, 4), vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn the_injector_prefix_is_two_mon() {
        assert!(is_injector(&[0x02, 0x6d, 0x6f, 0x6e, 0x01, 0x05]));
        assert!(!is_injector(&[0xef, 0xbe, 0xad, 0xde, 0xad, 0xde]));
        assert_eq!(
            mac_string(&[0x02, 0x6d, 0x6f, 0x6e, 0x01, 0x0a]),
            "02:6d:6f:6e:01:0a"
        );
    }
}
