//! numpy's arithmetic, operation for operation.
//!
//! The Python reducer (`monad_knowledge.lake.ingest.decode_segment`) is the
//! reference. Where numpy's result depends on the ORDER of floating-point
//! operations (sums, means, standard deviations, `linspace`, `unwrap`,
//! complex division), this module repeats numpy's order, so the two reducers
//! differ only where a transcendental function (`log10`, `atan2`, `sin`, ...)
//! is computed by a different library. Measured, not assumed: numpy's
//! `add.reduce` is `0 + pairwise(a)` for float32 and float64, on 1,000+ random
//! arrays each and along axis 1 of a 2-D array (study script
//! `np_pairwise.py`, 2026-10-24).

use std::f64::consts::PI;

/// numpy's `PW_BLOCKSIZE`.
const PW_BLOCKSIZE: usize = 128;

/// A float type numpy sums pairwise.
pub trait Float: Copy + std::ops::Add<Output = Self> + PartialEq {
    const ZERO: Self;
}
impl Float for f64 {
    const ZERO: Self = 0.0;
}
impl Float for f32 {
    const ZERO: Self = 0.0;
}

/// numpy's `@TYPE@_pairwise_sum` over a contiguous slice.
fn pairwise<T: Float>(a: &[T]) -> T {
    let n = a.len();
    if n < 8 {
        let mut res = T::ZERO;
        for &x in a {
            res = res + x;
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] = r[j] + a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res = res + a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise(&a[..n2]) + pairwise(&a[n2..])
    }
}

/// `np.add.reduce(a)`: the identity, then the pairwise sum.
pub fn sum<T: Float>(a: &[T]) -> T {
    T::ZERO + pairwise(a)
}

/// `np.mean(a)` of a float64 vector: the sum, then a true division by the count.
pub fn mean(a: &[f64]) -> f64 {
    sum(a) / a.len() as f64
}

/// `np.std(a, ddof=ddof)` of a float64 vector.
pub fn std(a: &[f64], ddof: usize) -> f64 {
    let m = mean(a);
    let sq: Vec<f64> = a
        .iter()
        .map(|&x| {
            let d = x - m;
            d * d
        })
        .collect();
    (sum(&sq) / (a.len() - ddof) as f64).sqrt()
}

/// `max(a)` as numpy reduces it (NaN-propagating; inputs here are finite).
pub fn max(a: &[f64]) -> f64 {
    let mut m = a[0];
    for &x in &a[1..] {
        if x > m || x.is_nan() {
            m = x;
        }
    }
    m
}

/// Median of the finite values, `None` when there are none.
///
/// `ingest._nanmedian`: `np.partition` selects order statistics, which are
/// values, so any exact selection algorithm returns the same numbers. The
/// even-count midpoint is `(lo + hi) * 0.5`, as there.
pub fn nanmedian(values: &[f64]) -> Option<f64> {
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let n = finite.len();
    if n == 0 {
        return None;
    }
    let half = n / 2;
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap();
    if n % 2 == 1 {
        let (_, v, _) = finite.select_nth_unstable_by(half, cmp);
        Some(*v)
    } else {
        let (lo_part, hi, _) = finite.select_nth_unstable_by(half, cmp);
        let hi = *hi;
        let lo = lo_part.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some((lo + hi) * 0.5)
    }
}

/// `np.median` of a vector without NaN: the midpoint is `(lo + hi) / 2`.
pub fn median(values: &mut [f64]) -> f64 {
    let n = values.len();
    let half = n / 2;
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap();
    if n % 2 == 1 {
        let (_, v, _) = values.select_nth_unstable_by(half, cmp);
        *v
    } else {
        let (lo_part, hi, _) = values.select_nth_unstable_by(half, cmp);
        let hi = *hi;
        let lo = lo_part.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // numpy takes `mean` of the two middle values: their sum over 2.
        (lo + hi) / 2.0
    }
}

/// Python's (and numpy's) float modulo: the result takes the divisor's sign.
pub fn py_mod(x: f64, m: f64) -> f64 {
    let r = x % m; // fmod
    if r != 0.0 {
        if (m < 0.0) != (r < 0.0) {
            r + m
        } else {
            r
        }
    } else {
        0.0_f64.copysign(m)
    }
}

/// `ingest._wrap`: an angle into [-π, π).
pub fn wrap(x: f64) -> f64 {
    py_mod(x + PI, 2.0 * PI) - PI
}

/// `np.unwrap(p)` with the default discontinuity and period.
pub fn unwrap(p: &[f64]) -> Vec<f64> {
    let n = p.len();
    let mut out = p.to_vec();
    if n < 2 {
        return out;
    }
    let period = 2.0 * PI;
    let high = period / 2.0;
    let low = -high;
    let mut acc = 0.0;
    for i in 1..n {
        let dd = p[i] - p[i - 1];
        let mut ddmod = py_mod(dd - low, period) + low;
        if ddmod == low && dd > 0.0 {
            ddmod = high;
        }
        let mut corr = ddmod - dd;
        if dd.abs() < high {
            corr = 0.0;
        }
        // `np.cumsum` adds left to right.
        acc += corr;
        out[i] = p[i] + acc;
    }
    out
}

/// `np.linspace(lo, hi, num)` (endpoint included).
pub fn linspace(lo: f64, hi: f64, num: usize) -> Vec<f64> {
    if num == 1 {
        return vec![lo];
    }
    let div = (num - 1) as f64;
    let delta = hi - lo;
    let step = delta / div;
    let mut y: Vec<f64> = (0..num)
        .map(|i| {
            let k = i as f64;
            if step == 0.0 {
                (k / div) * delta
            } else {
                k * step
            }
        })
        .map(|v| v + lo)
        .collect();
    y[num - 1] = hi;
    y
}

/// `np.trapezoid(y, x)`.
pub fn trapezoid(y: &[f64], x: &[f64]) -> f64 {
    let terms: Vec<f64> = (1..y.len())
        .map(|i| (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0)
        .collect();
    sum(&terms)
}

/// numpy's complex64 division (`CFLOAT_divide`, Smith's method in float32).
#[inline]
pub fn cdiv32(ar: f32, ai: f32, br: f32, bi: f32) -> (f32, f32) {
    let br_abs = br.abs();
    let bi_abs = bi.abs();
    if br_abs >= bi_abs {
        if br_abs == 0.0 && bi_abs == 0.0 {
            (ar / br_abs, ai / br_abs)
        } else {
            let rat = bi / br;
            let scl = 1.0f32 / (br + bi * rat);
            ((ar + ai * rat) * scl, (ai - ar * rat) * scl)
        }
    } else {
        let rat = br / bi;
        let scl = 1.0f32 / (bi + br * rat);
        ((ar * rat + ai) * scl, (ai * rat - ar) * scl)
    }
}

/// numpy's `CFLOAT_pairwise_sum` over interleaved `(re, im)` pairs.
fn cpairwise(a: &[f32]) -> (f32, f32) {
    let n = a.len(); // floats, always even
    if n < 8 {
        let (mut rr, mut ri) = (0.0f32, 0.0f32);
        let mut i = 0;
        while i < n {
            rr += a[i];
            ri += a[i + 1];
            i += 2;
        }
        (rr, ri)
    } else if n <= PW_BLOCKSIZE {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut rr = (r[0] + r[2]) + (r[4] + r[6]);
        let mut ri = (r[1] + r[3]) + (r[5] + r[7]);
        while i < n {
            rr += a[i];
            ri += a[i + 1];
            i += 2;
        }
        (rr, ri)
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        let (r1, i1) = cpairwise(&a[..n2]);
        let (r2, i2) = cpairwise(&a[n2..]);
        (r1 + r2, i1 + i2)
    }
}

/// The circular median of `drift._circular_median` / `ingest._circular_median`
/// for float32 angles: `exp(1j*a)` in complex64, a pairwise complex sum, a
/// complex128 division by the integer count, a cast back to complex64, then
/// `atan2` in float32. NaN when no angle is finite.
pub fn circular_median32(angles: &[f32]) -> f64 {
    let mut z: Vec<f32> = Vec::with_capacity(angles.len() * 2);
    for &a in angles {
        if a.is_finite() {
            // `1j * a` in complex64 is (0*a - 1*0, 0*0 + 1*a); its exp is (cos a, sin a).
            z.push(a.cos());
            z.push(a.sin());
        }
    }
    if z.is_empty() {
        return f64::NAN;
    }
    let count = (z.len() / 2) as f64;
    let (rr, ri) = cpairwise(&z);
    let (rr, ri) = (0.0f32 + rr, 0.0f32 + ri);
    // complex128 / (count + 0j): Smith's method with a zero imaginary divisor.
    let scl = 1.0f64 / count;
    let re = ((rr as f64) + (ri as f64) * 0.0) * scl;
    let im = ((ri as f64) - (rr as f64) * 0.0) * scl;
    (im as f32).atan2(re as f32) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_modulo_takes_the_divisor_sign() {
        assert_eq!(py_mod(-1.0, 3.0), 2.0);
        assert_eq!(py_mod(1.0, 3.0), 1.0);
        assert!(py_mod(-3.0, 3.0).is_sign_positive());
    }

    #[test]
    fn unwrap_removes_a_two_pi_step() {
        let u = unwrap(&[3.0, -3.0, -2.9]);
        assert!((u[1] - (2.0 * PI - 3.0)).abs() < 1e-12);
        assert!((u[2] - u[1] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn linspace_ends_on_its_endpoints() {
        let f = linspace(20.0, 60.0, 12);
        assert_eq!(f[0], 20.0);
        assert_eq!(f[11], 60.0);
    }

    #[test]
    fn medians_match_the_definitions() {
        assert_eq!(nanmedian(&[3.0, f64::NAN, 1.0, 2.0]), Some(2.0));
        assert_eq!(nanmedian(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(nanmedian(&[f64::NAN]), None);
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), 2.5);
    }

    #[test]
    fn pairwise_matches_a_plain_sum_on_exact_values() {
        let a: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        assert_eq!(sum(&a), 499_500.0);
    }
}
