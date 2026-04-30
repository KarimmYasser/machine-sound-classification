use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::collections::HashMap;
use rayon::prelude::*;
use std::sync::Arc;

#[pyclass]
pub struct AudioFeatures {
    #[pyo3(get)]
    pub features: HashMap<String, f32>,
}

struct SparseRow {
    indices: Vec<usize>,
    values: Vec<f32>,
}

impl SparseRow {
    fn dot(&self, vector: &[f32]) -> f32 {
        let mut sum = 0.0;
        for i in 0..self.indices.len() {
            sum += self.values[i] * vector[self.indices[i]];
        }
        sum
    }
}

struct DctPlan {
    matrix: Vec<f32>,
    n_mfcc: usize,
    n_mels: usize,
}

impl DctPlan {
    fn new(n_mfcc: usize, n_mels: usize) -> Self {
        let mut matrix = Vec::with_capacity(n_mfcc * n_mels);
        let n = n_mels as f32;
        for k in 0..n_mfcc {
            let scale = if k == 0 { (1.0 / n).sqrt() } else { (2.0 / n).sqrt() };
            for i in 0..n_mels {
                matrix.push(scale * (PI * k as f32 * (i as f32 + 0.5) / n).cos());
            }
        }
        Self { matrix, n_mfcc, n_mels }
    }

    fn apply(&self, input: &[f32], output: &mut [f32]) {
        for k in 0..self.n_mfcc {
            let row_offset = k * self.n_mels;
            let mut sum = 0.0;
            for i in 0..self.n_mels {
                sum += self.matrix[row_offset + i] * input[i];
            }
            output[k] = sum;
        }
    }
}

fn get_stats(data: &[f32]) -> (f32, f32, f32, f32) {
    if data.is_empty() { return (0.0, 0.0, 0.0, 0.0); }
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &x in data {
        let x64 = x as f64;
        sum += x64;
        sum_sq += x64 * x64;
        if x < min { min = x; }
        if x > max { max = x; }
    }
    let n = data.len() as f64;
    let mean = (sum / n) as f32;
    let var = (sum_sq / n - (sum / n).powi(2)).max(0.0);
    (mean, var.sqrt() as f32, min, max)
}

struct FrameResult {
    rms: f32,
    zcr: f32,
    centroid: f32,
    bandwidth: f32,
    rolloff: f32,
    mel_power: Vec<f32>,
    contrast: Vec<f32>,
}

#[pyfunction]
#[pyo3(signature = (y, sr, mel_basis, contrast_octa_bins, n_fft=2048, hop_length=512, n_mfcc=13, n_mels=128, n_bands=6, quantile=0.02))]
fn extract_features(
    _py: Python<'_>,
    y: PyReadonlyArray1<f32>,
    sr: f32,
    mel_basis: PyReadonlyArray2<f32>,
    contrast_octa_bins: Vec<usize>,
    n_fft: usize,
    hop_length: usize,
    n_mfcc: usize,
    n_mels: usize,
    n_bands: usize,
    quantile: f32,
) -> PyResult<AudioFeatures> {
    let y_slice = y.as_slice()?;
    let mel_array = mel_basis.as_array();
    let n_freq = n_fft / 2 + 1;

    let sparse_mel: Vec<SparseRow> = (0..n_mels).map(|m| {
        let row = mel_array.row(m);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (i, &v) in row.indexed_iter() {
            if v != 0.0 {
                indices.push(i);
                values.push(v);
            }
        }
        SparseRow { indices, values }
    }).collect();

    let dct_plan = Arc::new(DctPlan::new(n_mfcc, n_mels));

    let pad_len = n_fft / 2;
    let mut padded = vec![0.0f32; y_slice.len() + 2 * pad_len];
    padded[pad_len..pad_len + y_slice.len()].copy_from_slice(y_slice);

    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n_fft - 1) as f32).cos()))
        .collect();

    let frame_indices: Vec<usize> = (0..=(padded.len() - n_fft)).step_by(hop_length).collect();
    let num_frames = frame_indices.len();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    let results: Vec<FrameResult> = frame_indices.par_iter().map_init(
        || {
            (
                vec![Complex::new(0.0, 0.0); n_fft],
                vec![0.0f32; n_freq],
                vec![0.0f32; n_freq],
                vec![0.0f32; n_freq],
            )
        },
        |(buf, magnitudes, power, sub_buf), &start_idx| {
            let frame = &padded[start_idx..start_idx + n_fft];
            
            for (j, (&x, &w)) in frame.iter().zip(window.iter()).enumerate() {
                buf[j] = Complex::new(x * w, 0.0);
            }
            fft.process(buf);
            
            for j in 0..n_freq {
                let p = buf[j].re * buf[j].re + buf[j].im * buf[j].im;
                power[j] = p;
                magnitudes[j] = p.sqrt();
            }

            let mut rms_sum_sq = 0.0f32;
            let mut zcr_count = 0usize;
            for j in 0..n_fft {
                rms_sum_sq += frame[j] * frame[j];
                if j > 0 && (frame[j] >= 0.0) != (frame[j-1] >= 0.0) {
                    zcr_count += 1;
                }
            }
            let rms = (rms_sum_sq / n_fft as f32).sqrt();
            let zcr = zcr_count as f32 / n_fft as f32;

            let mut centroid = 0.0;
            let mut rolloff = 0.0;
            let mut bandwidth = 0.0;
            let mag_sum: f32 = magnitudes.iter().sum();
            if mag_sum > 0.0 {
                let freq_scale = sr / n_fft as f32;
                for (j, &mag) in magnitudes.iter().enumerate() {
                    centroid += (mag as f64 / mag_sum as f64) * (j as f64 * freq_scale as f64);
                }
                let threshold = 0.85 * mag_sum;
                let mut cum = 0.0f32;
                for (j, &mag) in magnitudes.iter().enumerate() {
                    cum += mag;
                    if cum >= threshold {
                        rolloff = j as f32 * freq_scale;
                        break;
                    }
                }
                let mut bw_sum = 0.0f64;
                for (j, &mag) in magnitudes.iter().enumerate() {
                    let diff = (j as f64 * freq_scale as f64) - centroid;
                    bw_sum += (mag as f64 / mag_sum as f64) * diff * diff;
                }
                bandwidth = bw_sum.sqrt() as f32;
            }

            let mut mel_power = vec![0.0f32; n_mels];
            for m in 0..n_mels {
                mel_power[m] = sparse_mel[m].dot(power);
            }

            let mut contrast = vec![0.0f32; n_bands + 1];
            for b in 0..=n_bands {
                let s = contrast_octa_bins[b];
                let e = contrast_octa_bins[b + 1].min(n_freq);
                if s < e {
                    let len = e - s;
                    sub_buf[..len].copy_from_slice(&magnitudes[s..e]);
                    let slice = &mut sub_buf[..len];
                    let idx = ((quantile * len as f32).round() as usize).max(1).min(len);
                    slice.select_nth_unstable_by(idx - 1, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let valley = slice[..idx].iter().sum::<f32>() / idx as f32;
                    slice.select_nth_unstable_by(len - idx, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let peak = slice[len - idx..].iter().sum::<f32>() / idx as f32;
                    contrast[b] = 10.0 * peak.max(1e-10).log10() - 10.0 * valley.max(1e-10).log10();
                }
            }

            FrameResult { rms, zcr, centroid: centroid as f32, bandwidth, rolloff, mel_power, contrast }
        }
    ).collect();

    let amin = 1e-10f32;
    let top_db = 80.0f32;
    let global_max_db = results.par_iter()
        .map(|r| r.mel_power.iter().map(|&v| 10.0 * v.max(amin).log10()).fold(f32::NEG_INFINITY, f32::max))
        .reduce(|| f32::NEG_INFINITY, f32::max);
    let floor = global_max_db - top_db;

    let all_mfccs: Vec<Vec<f32>> = results.par_iter().map(|r| {
        let mut log_mel = vec![0.0f32; n_mels];
        for m in 0..n_mels {
            let val = 10.0 * r.mel_power[m].max(amin).log10();
            log_mel[m] = val.max(floor);
        }
        let mut mfcc = vec![0.0f32; n_mfcc];
        dct_plan.apply(&log_mel, &mut mfcc);
        mfcc
    }).collect();

    let mut f = HashMap::with_capacity(100);
    let mut tmp = Vec::with_capacity(num_frames);

    tmp.extend(results.iter().map(|r| r.rms));
    let (m, _, _, _) = get_stats(&tmp); f.insert("rms_mean".into(), m);

    tmp.clear(); tmp.extend(results.iter().map(|r| r.zcr));
    let (m, s, _, _) = get_stats(&tmp); f.insert("zcr_mean".into(), m); f.insert("zcr_std".into(), s);

    tmp.clear(); tmp.extend(results.iter().map(|r| r.centroid));
    let (m, s, _, _) = get_stats(&tmp); f.insert("centroid_mean".into(), m); f.insert("centroid_std".into(), s);

    tmp.clear(); tmp.extend(results.iter().map(|r| r.bandwidth));
    let (m, _, _, _) = get_stats(&tmp); f.insert("bandwidth_mean".into(), m);

    tmp.clear(); tmp.extend(results.iter().map(|r| r.rolloff));
    let (m, s, _, _) = get_stats(&tmp); f.insert("rolloff_mean".into(), m); f.insert("rolloff_std".into(), s);

    for m in 0..n_mfcc {
        tmp.clear(); tmp.extend(all_mfccs.iter().map(|frame| frame[m]));
        let (mean, std, min, max) = get_stats(&tmp);
        let idx = m + 1;
        f.insert(format!("mfcc_{idx}_mean"), mean);
        f.insert(format!("mfcc_{idx}_std"), std);
        f.insert(format!("mfcc_{idx}_min"), min);
        f.insert(format!("mfcc_{idx}_max"), max);
    }

    for b in 0..=n_bands {
        tmp.clear(); tmp.extend(results.iter().map(|r| r.contrast[b]));
        let (mean, std, min, _) = get_stats(&tmp);
        let idx = b + 1;
        f.insert(format!("contrast_{idx}_mean"), mean);
        f.insert(format!("contrast_{idx}_std"), std);
        f.insert(format!("contrast_{idx}_min"), min);
    }

    Ok(AudioFeatures { features: f })
}

#[pymodule]
fn fast_features(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_features, m)?)?;
    m.add_class::<AudioFeatures>()?;
    Ok(())
}
