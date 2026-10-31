# Report outline (4-5 pages, .docx, first person: "I", never "we/group")
## Background and motivation  (ONE paragraph: my use case, why HAR is needed)
## Data collection and pre-processing
- Files, takes, sessions, per-activity seconds (notebook table); takes cut into 10 s files (state this)
- 4 video links; sampling-rate table (nominal / measured / jitter); interpolation + low-pass logic
- Window size tied to fs: 2 s = 100 samples at 50 Hz, 0.5 Hz FFT resolution, Nyquist 25 Hz, 50% overlap
- Figure 1: raw signal + spectrum per activity
## HMM setup and implementation
- States, observations, A, B, pi; feature table with justification; z-score (train stats only) + why
- Algorithm choice tied to metrics (Baum-Welch epsilon stop, Viterbi)
## Results and interpretation
- How the unseen data was obtained; metrics table; confusion matrix; A and emission figures; decoded vs true; LL curve
- Accuracy before vs after Baum-Welch
## Discussion and conclusion
- Easiest/hardest; realism of A (it reflects my sequence construction); noise/sampling; single-subject limit; improvements
