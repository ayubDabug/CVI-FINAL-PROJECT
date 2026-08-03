# CVI620 Final Project — Self-Driving Car (Behavioral Cloning)

Trains an Nvidia-style CNN to predict steering angle from the Udacity
simulator's center-camera images, then drives the car autonomously in the
simulator.

**The core idea:** feed the model images from the car's front camera, and
have it output the correct steering angle to keep the car on the road.

## Project structure

```
CVI-FINAL-PROJECT/
├── IMG/                     # kaung's collected center/left/right camera frames
├── driving_log.csv          # kaung's driving log
├── IMG_ayub/                # ayub's collected camera frames
├── driving_log_ayub.csv     # ayub's driving log
├── data_mtp/driving_log.csv # (legacy duplicate — kept for history)
├── data_loader.py           # loads a driving_log.csv, reviews/balances steering histogram
├── preprocessing.py         # img_preprocess(): crop, YUV, blur, resize, normalize
├── augmentation.py          # pan/zoom/brightness/rotate/flip augmentation
├── batching.py              # batch_generator(): combines augmentation + preprocessing per batch
├── train.py                 # builds + trains the Nvidia model, saves model + graphs
├── TestSimulation.py        # drives the car in the simulator's Autonomous Mode (inference)
├── test.py                  # exploratory histogram script (Kaung)
└── model/                   # output: model.h5, histogram_*.png, loss.png
```

## 1. Environment setup

Create a conda environment (Python 3.8, TensorFlow-GPU 2.3.0, Flask-SocketIO,
OpenCV, matplotlib, scikit-learn, imgaug) using the provided package list:

```bash
conda create --name cvi620 --file package_list.txt
conda activate cvi620
pip install -r pip_pkgs.txt
```

## 2. Data collection

Already done by the team — `driving_log.csv` / `IMG/` and
`driving_log_ayub.csv` / `IMG_ayub/` (ayub) contain camera frames + steering
angles collected by driving the track in the simulator's Training Mode
(forward + reverse laps).

## 3. Train the model

```bash
python train.py --datadir . --epochs 10
```

(`--datadir` points at the folder containing `driving_log.csv` and `IMG/` —
use `.` for the repo root, or point it at another dataset such as one built
from `driving_log_ayub.csv`/`IMG_ayub/`.)

This will:

1. Load the driving log (center image + steering angle only).
2. **Balance** the steering histogram (angles are capped per bin, since most
   raw samples are near-zero) — saves `model/histogram_before.png` and
   `model/histogram_after.png`.
3. Split 80/20 into train/validation.
4. Train with a batch generator that applies **augmentation** (pan, zoom,
   brightness, rotation, horizontal flip — each applied to a random subset,
   flip negates the steering angle) only on the training split.
5. Every image is **preprocessed** identically for train/val/inference: crop
   road area (rows 60–135), convert to YUV, Gaussian blur, resize to
   200×66, normalize to [0,1].
6. Save `model/model.h5` and a training/validation loss plot `model/loss.png`.

Useful flags: `--epochs`, `--steps_per_epoch`, `--batch_size`,
`--samples_per_bin` (histogram cap), `--out` (output folder).

## 4. Test in the simulator

1. Launch the simulator with the same settings used for data collection and
   choose **Autonomous Mode**.
2. In the `cvi620` environment, run:

```bash
python TestSimulation.py model/model.h5
```

The script starts a SocketIO/Flask server on port 4567 that the simulator
connects to; it receives each camera frame, preprocesses it the same way as
training, predicts the steering angle, and sends back steering + throttle
(throttle backs off as speed approaches `MAX_SPEED`).

## Notes / troubleshooting

- On Windows, if `conda activate` fails with "Unable to create process" (a
  known conda bug triggered by spaces in the install path/username), skip
  activation and instead call the env's `python.exe` directly, with its
  `Library\bin` folder added to `PATH` for that shell session so native DLLs
  (numpy, TensorFlow, etc.) resolve correctly.
- The CNN architecture (`nvidia_model()` in `train.py`) mirrors the
  assignment's Nvidia end-to-end figure: 5 conv layers (24/36/48/64/64
  filters) → flatten (1152) → dense 1164/100/50/10 → 1 output neuron
  (steering angle). Verified output feature-map shapes (31×98, 14×47, 5×22,
  3×20, 1×18) match the figure.

## Approach

The pipeline follows the classic Nvidia end-to-end behavioral cloning
approach: collect (image, steering angle) pairs by driving manually, then
train a CNN to regress steering angle directly from pixels — no lane
detection or explicit feature engineering. The main design decisions:

- **Center camera only** for training/inference, per the assignment — left/
  right frames are collected but unused.
- **Histogram balancing** before training, since raw driving data is
  dominated by near-zero steering angles (straight-line driving); without
  capping the over-represented bins the model would learn to just drive
  straight.
- **Augmentation only on the training split, only on a random subset of
  samples per batch** — validation data stays "clean" so the loss curve
  reflects real generalization, not augmentation noise.
- **Identical preprocessing everywhere** (train/validation/live inference)
  so the model never sees a distribution shift between training and the
  simulator.

## Challenges encountered

- **`conda activate` failing on Windows** with `Unable to create process
using "...\anaconda3\python.exe" ...` — caused by a known conda bug
  triggered by the space in the Windows username/install path. Worked
  around by calling the environment's `python.exe` directly and adding its
  `Library\bin` (and related) folders to `PATH` for the session instead of
  relying on `conda activate`.
- **`numpy`/TensorFlow DLL import errors** (`DLL load failed while importing
_multiarray_umath`) when running the env's Python without those same
  `Library\bin` folders on `PATH` — same root cause/fix as above.
- **Steering angle imbalance** — roughly 80% of collected samples had a
  near-zero steering angle. Addressed with histogram-based balancing
  (`data_loader.balance_data`) before the train/validation split.
- **Merging into a shared team repo** that already had teammates' raw
  driving data (`IMG_ayub/`, `driving_log_ayub.csv`) committed directly to
  `main` (tens of thousands of image files). Used a `git sparse-checkout`
  (cone mode, root files + `model/` only) so the pipeline code could be
  added and merged without re-downloading or disturbing the existing
  multi-gigabyte image history.

## Deliverables

- All Python scripts: data preprocessing (`preprocessing.py`), augmentation
  (`augmentation.py`), batching (`batching.py`), data loading/balancing
  (`data_loader.py`), training (`train.py`), inference/testing
  (`TestSimulation.py`).
- A screen recording of the trained model driving successfully in the
  simulator (not included in this repo — add separately).
- This Git repository, with commit history showing individual contributions.
- This documentation.

Groups of up to 3 individuals are allowed for this project.
