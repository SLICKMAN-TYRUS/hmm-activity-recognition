# Formative 2: Human Activity Recognition with a Hidden Markov Model
Individual project. HMM (Baum-Welch + Viterbi) implemented from scratch in numpy; no hmmlearn.

This individual project models human activity from smartphone motion data. I recorded accelerometer and gyroscope signals with Sensor Logger for four activities (still, standing, walking, jumping), extracted time-domain and FFT-based features from 2 s windows, and trained a Gaussian-emission Hidden Markov Model written from scratch in NumPy. Baum-Welch learns the parameters and Viterbi decodes the activity sequence. The model is evaluated on unseen recordings with a confusion matrix, sensitivity, specificity and accuracy. The motivation is.........................my use case eg...................................

## Repository structure
- Note: accelerometer data = Sensor Logger `TotalAcceleration.csv` (includes gravity)
- `data/raw/takes/`  Sensor Logger exports, one folder per continuous take, named `<activity>_s<k>` (e.g. `walking_s1`)
- `data/raw/test/`   exports of the two test sequences, folders `test_seq1`, `test_seq2`
- `data/train/`      labelled 10 s CSVs `<activity>_s<k>_<nn>.csv` (t, ax, ay, az, gx, gy, gz)
- `data/test/`       `test_seq1.csv`, `test_seq2.csv`, `test_labels.csv` (file,start_s,end_s,activity)
- `src/prepare_data.py`  merges accelerometer + gyroscope and cuts takes into 10 s files
- `notebook.ipynb`   preprocessing, features, HMM, evaluation, figures
- `report/`          report outline

## Reproduce
```
pip install -r requirements.txt
python src/prepare_data.py takes --raw data/raw/takes --out data/train
python src/prepare_data.py test  --raw data/raw/test  --out data/test
jupyter notebook notebook.ipynb    # Run All
```

## Videos (one per activity, face and full body visible)
- Standing: 
- Walking: 
- Jumping: 
- Still: 

## Sampling rates
| Source | Nominal (Hz) | Measured (Hz) | Jitter (ms) |
|---|---|---|---|
| (copy from notebook output) | | | |
