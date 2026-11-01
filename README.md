# Formative 2: Human Activity Recognition with a Hidden Markov Model

Individual project. A Gaussian-emission Hidden Markov Model (Baum-Welch training, Viterbi decoding) implemented from scratch in NumPy, with no `hmmlearn`.

I recorded accelerometer and gyroscope signals with Sensor Logger for four activities (still, standing, walking, jumping), extracted time-domain and FFT-based features from 2 s windows, and trained the HMM to recover the hidden activity sequence. The model is evaluated on two unseen recordings with a confusion matrix, sensitivity, specificity and accuracy.

**Use case.** Passive activity monitoring of an older adult living alone, using a phone carried at the waist. Separating resting (still, standing) from walking and from sudden high-impact movement (jumping stands in for it here) is a first step towards detecting prolonged inactivity or fall-like events without cameras.

## Results at a glance

| State (Activity) | Number of Samples | Sensitivity | Specificity | Overall Accuracy |
|---|---|---|---|---|
| Still | 19 | 1.000 | 1.000 | 1.000 |
| Standing | 40 | 1.000 | 1.000 | 1.000 |
| Walking | 20 | 1.000 | 1.000 | 1.000 |
| Jumping | 20 | 1.000 | 1.000 | 1.000 |
| All | 99 | n/a | n/a | 1.000 |

Accuracy is 1.000 both before and after Baum-Welch. Treat this as optimistic: it is one person on one phone, and 51 of 150 test windows around transitions (pick-up, activity switches) were left unscored. Baum-Welch raised the training log-likelihood from about -645 to -632.6 and stopped after 5 evaluations (relative change below 1e-6).

## Repository structure

```
.
├── data/
│   ├── raw/
│   │   ├── takes/        Sensor Logger exports, one folder per take: <activity>_s<k> (14 takes)
│   │   └── test/         test_seq1, test_seq2 (exports of the two test sequences)
│   ├── train/            88 labelled 10 s CSVs: <activity>_s<k>_<nn>.csv (t, ax, ay, az, gx, gy, gz)
│   └── test/             test_seq1.csv, test_seq2.csv, test_labels.csv (file, start_s, end_s, activity)
├── src/
│   └── prepare_data.py   merges accelerometer + gyroscope, cuts takes into 10 s files
├── figures/              plots saved by the notebook
├── report/               report outline
├── notebook.ipynb        preprocessing, features, HMM, evaluation, figures (outputs saved)
├── requirements.txt
└── README.md
```

## Data

- **Device and app:** TECNO Camon 15 (model CD7), Android, Sensor Logger v1.68 (build 31459800).
- **Channels:** accelerometer = Sensor Logger `TotalAcceleration.csv` (includes gravity, so phone orientation is preserved; the plain `Accelerometer.csv` is gravity-removed and was not used), plus `Gyroscope.csv`.
- **Training:** 14 continuous takes recorded 31 Oct to 1 Nov 2026 in Nairobi, cut into 10 s files (first 3 s and last 2 s of each take dropped).

| Activity | Takes (nominal rate) | Files (10 s) | Seconds after trim |
|---|---|---|---|
| Still | 2 (1 x 100 Hz, 1 x 50 Hz) | 16 | 143.8 |
| Standing | 3 (1 x 100 Hz, 2 x 50 Hz) | 24 | 215.7 |
| Walking | 3 (1 x 100 Hz, 2 x 50 Hz) | 24 | 215.7 |
| Jumping | 6 (2 x 100 Hz, 4 x 50 Hz) | 24 | 215.8 |
| Total | 14 | 88 | 791.0 |

- **Test:** two separate continuous recordings (`test_seq1`, `test_seq2`, about 77 s each, native 50 Hz): still, standing, walking, jumping, standing in 15 s blocks. They were never used for training or for the scaler. The cue timing drifted, so segment boundaries in `test_labels.csv` were set from the signal energy, with a margin of about 2 to 5 s around each transition excluded from the metrics.
- **Data quality:** `prepare_data.py` discards any 10 s chunk containing a timestamp gap above 0.1 s instead of interpolating over it. My first still recording lost 28 s of samples to Android power-saving, so I re-recorded it; none of the final 88 files is affected.

## Sampling rates

Nine takes and both test files ran at a nominal 50 Hz, and five takes at 100 Hz, so every activity has training data from both settings. Everything is put on one 50 Hz grid (zero-phase 4th-order Butterworth low-pass at 20 Hz before downsampling files above 50 Hz, then linear interpolation).

| Device | Nominal setting | Recordings | Measured (Hz) | Jitter (ms) |
|---|---|---|---|---|
| TECNO Camon 15 | 50 Hz | 9 training takes, 2 test files | 50.0 | 0.01 to 0.24 |
| TECNO Camon 15 | 100 Hz | 5 training takes | 99.9 | 0.01 to 0.17 |

## Method in brief

- **Windows:** 2 s (100 samples at 50 Hz), 50% overlap, 661 training windows.
- **Features (15):** 11 time-domain (accelerometer magnitude mean, per-axis means, std, RMS, SMA, axis correlations, gyroscope std) and 4 FFT-derived (dominant frequency of accelerometer and gyroscope, spectral energy, spectral entropy), z-scored with training statistics only.
- **HMM:** 4 hidden states (the activities), diagonal-Gaussian emissions, log-space forward-backward, Baum-Welch with a log-likelihood convergence check, log-space Viterbi. Parameters are initialised from labels; variances are clipped at 1e-3 in z-scored units.

## Reproduce

Run from the project root. On recent Ubuntu or WSL, use a virtual environment.

```bash
python3 -m venv ~/hmm_venv && source ~/hmm_venv/bin/activate
pip install -r requirements.txt ipykernel

python src/prepare_data.py takes --raw data/raw/takes --out data/train
python src/prepare_data.py test  --raw data/raw/test  --out data/test

jupyter notebook notebook.ipynb    # Run All
```

`data/test/test_labels.csv` is included and must stay in `data/test/`.

## Videos (one per activity, face and full body visible)

- Standing: <paste link>
- Walking: <paste link>
- Jumping: <paste link>
- Still: <paste link>

## Limitations

One participant, one phone and one holding style. The gravity-direction features depend on how the phone is held, so a different device or orientation convention could break them. Windows near transitions were excluded from the test metrics, so the reported accuracy is higher than scoring every window would give.