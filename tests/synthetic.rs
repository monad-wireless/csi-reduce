//! End-to-end on synthetic containers written with `csiq::Writer`.
//!
//! Parity with the Python reference is tested in monad-knowledge, where the
//! reference lives (`tests/lake/test_native_reducer.py` and the study
//! `notebooks/studies/2026-10-24-native-reducer/`). These tests pin what this
//! crate can check alone: the envelope does not change a value, a cut
//! container keeps its prefix, and an injector stream reduces to motion rows.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use csi_reduce::reduce::{reduce_segment, PacketBatch, Params};
use csi_reduce::source::Kind;
use csiq::record::{Modulation, NodeState, PhyLabel, Width};
use csiq::{CsiRecord, Writer};

fn injector_record(i: u64) -> CsiRecord {
    let ntone = 52usize;
    let t = i as f64 * 0.004;
    let mut iq = Vec::with_capacity(2 * 2 * ntone);
    for chain in 0..2 {
        for tone in 0..ntone {
            // A walker-like phase wobble on chain 0, a steady chain 1.
            let phase = if chain == 0 {
                0.8 * (2.0 * std::f64::consts::PI * 30.0 * t).sin() + tone as f64 * 0.01
            } else {
                0.0
            };
            let amp = 300.0 + 40.0 * chain as f64 + tone as f64;
            iq.push((amp * phase.sin()) as i16); // imag first
            iq.push((amp * phase.cos()) as i16); // then real
        }
    }
    CsiRecord {
        ftm: (i as u32).wrapping_mul(1_280_000),
        us: i as u32,
        unix_ts_ns: 1_792_000_000_000_000_000 + i * 4_000_000,
        rnf: 0x4100,
        phy: Some(PhyLabel {
            modulation: Modulation::LegacyOfdm,
            mcs: 0,
            nss: 1,
        }),
        bw_antsel: csiq::raw::decode_bw_antsel(0x4100),
        mono_us: Some(i * 4000),
        vendor_hdr: None,
        node: NodeState::default(),
        seq: (i % 256) as u8,
        nrx: 2,
        ntx: 1,
        ntone: ntone as u16,
        rssi: vec![-48, -51],
        src_mac: [0x02, 0x6d, 0x6f, 0x6e, 0x01, 0x05],
        channel: 48,
        width: Width::Ht20,
        iq,
    }
}

fn write_container(path: &Path, n: u64, zst: bool) {
    let mut buf = Vec::new();
    {
        let mut w = Writer::new(&mut buf, None).unwrap();
        for i in 0..n {
            w.write_record(&injector_record(i)).unwrap();
        }
        w.finish().unwrap();
    }
    let bytes = if zst {
        zstd::encode_all(&buf[..], 3).unwrap()
    } else {
        buf
    };
    File::create(path).unwrap().write_all(&bytes).unwrap();
}

fn params() -> Params {
    Params {
        block_seconds: 60.0,
        n_groups: 4,
        warmup_seconds: 0.0,
        min_packets: 10,
        batch_size: 5000,
        motion_seconds: 2.0,
        utc_origin: false,
        window_index: true,
        segment_index: Some(2),
    }
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("csi-reduce-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn run(path: &Path) -> (csi_reduce::reduce::Reduction, Vec<PacketBatch>) {
    let mut batches = Vec::new();
    let out = reduce_segment(path, Kind::Csiq, params(), |b| {
        batches.push(b);
        Ok(())
    })
    .unwrap();
    (out, batches)
}

#[test]
fn the_envelope_changes_no_value() {
    let plain = tmp("a.csiq");
    let zst = tmp("a.csiq.zst");
    write_container(&plain, 15_000, false);
    write_container(&zst, 15_000, true);
    let (a, pa) = run(&plain);
    let (b, pb) = run(&zst);
    assert_eq!(a.records, 15_000);
    assert_eq!(a.packet_rows, b.packet_rows);
    assert_eq!(pa.len(), 3, "15,000 rows in batches of 5,000");
    for (x, y) in pa.iter().zip(&pb) {
        assert_eq!(x.t_ns, y.t_ns);
        assert_eq!(x.ratio_amp_db, y.ratio_amp_db);
        assert_eq!(x.ratio_phase_rad, y.ratio_phase_rad);
    }
    assert_eq!(a.motion.len(), b.motion.len());
    for (x, y) in a.motion.iter().zip(&b.motion) {
        assert_eq!(x.doppler_mean, y.doppler_mean);
        assert_eq!(x.inj_ratio_amp_sd_db, y.inj_ratio_amp_sd_db);
    }
    assert!(a.truncation.is_none());
}

#[test]
fn an_injector_stream_reduces_to_scored_motion_rows() {
    let path = tmp("b.csiq");
    write_container(&path, 15_000, false);
    let (out, _) = run(&path);
    // 60 s at 250 Hz: thirty 2-s rows, each over the 100-frame floor.
    assert_eq!(out.motion.len(), 30);
    let row = &out.motion[5];
    assert_eq!(row.inj_n_packets, row.n_packets);
    assert!(
        row.inj_doppler_mean[0].is_some(),
        "a 30 Hz wobble scores in the 20-60 Hz band"
    );
    assert_eq!(row.inj_n_swaps, 0, "a clean link has no orientation events");
    assert!(!out.blocks.is_empty());
    assert_eq!(out.census.len(), 1);
    assert_eq!(out.census[0].bandwidth_mhz, vec![20]);
    assert!(!out.windows.is_empty());
}

#[test]
fn a_cut_container_keeps_its_prefix_and_says_why() {
    let full = tmp("c.csiq");
    write_container(&full, 2_000, false);
    let bytes = std::fs::read(&full).unwrap();
    let cut = tmp("c-cut.csiq");
    File::create(&cut)
        .unwrap()
        .write_all(&bytes[..bytes.len() - 100])
        .unwrap();
    let (out, _) = run(&cut);
    assert_eq!(out.records, 1_999, "every whole record before the cut");
    assert!(out
        .truncation
        .as_deref()
        .unwrap_or("")
        .starts_with("TruncatedCapture"));
}

#[test]
fn a_file_that_is_not_a_container_fails() {
    let path = tmp("d.csiq");
    File::create(&path)
        .unwrap()
        .write_all(b"NOPE-not-a-capture")
        .unwrap();
    let got = reduce_segment(&path, Kind::Csiq, params(), |_| Ok(()));
    assert!(got.is_err());
}
