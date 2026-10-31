"""Prepare CSVs from Sensor Logger exports (TotalAcceleration.csv, falling back to Accelerometer.csv, + Gyroscope.csv per folder).

  takes: python src/prepare_data.py takes --raw data/raw/takes --out data/train
         folder <activity>_s<k> -> files <activity>_s<k>_<nn>.csv, each `chunk` seconds long
         (first `head` s and last `tail` s of every take are dropped: button presses / settling)
  test : python src/prepare_data.py test --raw data/raw/test --out data/test
         folder test_seq<k> -> test_seq<k>.csv (whole recording, untouched)

Output columns: t (s from start), ax, ay, az, gx, gy, gz. Gyroscope is interpolated onto accelerometer timestamps.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

def _read(p):
    df = pd.read_csv(p); df.columns = [c.strip().lower() for c in df.columns]; return df

def _time(df):
    if 'seconds_elapsed' in df.columns:
        return df['seconds_elapsed'].to_numpy(float)
    t = df['time'].to_numpy(float)
    scale = 1e9 if t[0] > 1e17 else 1e6 if t[0] > 1e14 else 1e3 if t[0] > 1e11 else 1.0
    return (t - t[0]) / scale

def merge(folder):
    # Sensor Logger 'Accelerometer.csv' is gravity-removed; 'TotalAcceleration.csv' keeps gravity (needed for orientation features)
    acc = 'TotalAcceleration.csv' if (folder/'TotalAcceleration.csv').exists() else 'Accelerometer.csv'
    a, g = _read(folder/acc), _read(folder/'Gyroscope.csv')
    ta, tg = _time(a), _time(g)
    out = pd.DataFrame({'t': ta, 'ax': a['x'], 'ay': a['y'], 'az': a['z']})
    for c in 'xyz':
        out['g'+c] = np.interp(ta, tg, g[c].to_numpy(float))
    out = out.dropna().drop_duplicates('t').sort_values('t').reset_index(drop=True)
    out['t'] = out['t'] - out['t'].iloc[0]
    return out

def do_takes(raw, out, chunk, head, tail, maxgap=0.1):
    """Cut each take into `chunk`-second files, keeping only chunks with no timestamp gap above `maxgap` seconds
    (Android sometimes drops samples when the screen sleeps; interpolating over such gaps would invent data)."""
    out.mkdir(parents=True, exist_ok=True); totals = {}
    for f in sorted(p for p in raw.iterdir() if p.is_dir()):
        df = merge(f); act = f.name.split('_')[0]; t = df['t'].to_numpy(); end = t[-1] - tail; s, i = head, 1
        n_gaps = int((np.diff(t) > maxgap).sum())
        while s + chunk <= end:
            lo, hi = np.searchsorted(t, s), np.searchsorted(t, s + chunk); seg = df.iloc[lo:hi]
            if len(seg) > 1 and np.diff(seg.t).max() <= maxgap and seg.t.iloc[0] - s <= maxgap and (s + chunk) - seg.t.iloc[-1] <= maxgap:
                seg = seg.copy(); seg['t'] -= seg['t'].iloc[0]
                seg.to_csv(out/f'{f.name}_{i:02d}.csv', index=False); s += chunk; i += 1
            else:
                s += 0.5
        totals[act] = totals.get(act, 0) + i - 1
        print(f'{f.name}: take {t[-1]:.0f} s, {n_gaps} gap(s) > {maxgap} s -> {i-1} files' + ('   <-- WARNING: gaps, re-record or add takes' if n_gaps else ''))
    print('files per activity:', totals, '| total:', sum(totals.values()))

def do_test(raw, out):
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(p for p in raw.iterdir() if p.is_dir()):
        df = merge(f); ng = int((np.diff(df['t'].to_numpy()) > 0.1).sum()); df.to_csv(out/f'{f.name}.csv', index=False)
        print('test file written:', f.name, '| gaps > 0.1 s:', ng, '(re-record if > 0)' if ng else '')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['takes', 'test'])
    p.add_argument('--raw', required=True); p.add_argument('--out', required=True)
    p.add_argument('--chunk', type=float, default=10); p.add_argument('--head', type=float, default=3); p.add_argument('--tail', type=float, default=2)
    a = p.parse_args()
    do_takes(Path(a.raw), Path(a.out), a.chunk, a.head, a.tail) if a.mode == 'takes' else do_test(Path(a.raw), Path(a.out))
