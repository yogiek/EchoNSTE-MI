# EchoNSTE-MI: baseline code

Code accompanying the **EchoNSTE-MI** dataset, a multi-view transthoracic echocardiography video dataset for
classifying **non-ST-elevation myocardial infarction (NSTEMI)** versus **unstable angina pectoris (UAP)** in
patients with non-ST-elevation acute coronary syndrome (NSTE-ACS).

This repository contains the scripts used to check the dataset and to train and evaluate the baseline models
reported in the Data in Brief article (technical validation).

- **Dataset (Mendeley Data):** https://data.mendeley.com/datasets/y6z64ntstg/2 (doi:[10.17632/y6z64ntstg.2](https://doi.org/10.17632/y6z64ntstg.2))
- **Article:** *EchoNSTE-MI: A multi-view echocardiography video dataset for myocardial infarction classification
  in patients with non-ST-elevation acute coronary syndrome*, Data in Brief (under review)

---

## Dataset at a glance

| Item | Value |
|---|---|
| Patients | 720 (311 NSTEMI, 409 UAP) |
| Views per patient | 3: apical two-chamber (2CH), apical four-chamber (4CH), parasternal long-axis (PLAX) |
| Videos | 2,160 MP4 files, 590 × 370 pixels, 45–60 frames/s, 2–5 cardiac cycles each |
| Source | National Cardiovascular Center Harapan Kita, Jakarta, Indonesia (admissions 2019–2023) |
| Acquisition | GE Vivid E9 with M5Sc-D transducer |
| Labels | Final clinical diagnosis based on clinical data, ECG, and high-sensitivity troponin T |
| Licence | CC BY 4.0 |

Folder structure after downloading and extracting the dataset:

```
EchoNSTE-MI/
├── 2CH/
│   ├── NSTEMI/   P001_2CH.mp4 ... P311_2CH.mp4
│   └── UAP/      P312_2CH.mp4 ... P720_2CH.mp4
├── 4CH/
│   ├── NSTEMI/   P001_4CH.mp4 ...
│   └── UAP/      P312_4CH.mp4 ...
└── PLAX/
    ├── NSTEMI/   P001_PLAX.mp4 ...
    └── UAP/      P312_PLAX.mp4 ...
```

The patient identifier (`P001`–`P720`) is unique across the dataset and shared by the three views of the same
patient. All videos are de-identified; clinical variables are not distributed.

---

## Repository contents

| File | Purpose |
|---|---|
| `check_env.py` | Checks Python packages, GPU availability, and shared memory |
| `check_dataset.py` | Counts videos per folder, checks that every patient has all three views, and test-decodes videos |
| `train.py` | Trains and evaluates one baseline model on one view |
| `run_all.sh` | Runs all six experiments (2 models × 3 views) and writes the summary table |
| `summarize.py` | Collects test metrics into `summary_test_metrics.csv` |
| `verify_results.py` | Recomputes metrics from saved predictions, checks for train/test leakage, and computes bootstrap 95% CIs of the AUC |
| `inspect_inputs.py` | Saves the frames that the model actually receives and measures video decoding time |
| `Dockerfile` | Optional container definition |
| `requirements.txt` | Python dependencies for a non-Docker installation |

---

## Baseline models

| Name in code | Architecture | Pretrained weights | Input |
|---|---|---|---|
| `cnn2d` | ResNet-18 applied to every frame, features averaged over time, linear classifier | ImageNet | 16 frames × 112 × 112 |
| `cnn3d` | R(2+1)D-18 applied to the whole clip | Kinetics-400 | 16 frames × 112 × 112 |

Common settings:

- 16 frames sampled uniformly from each video, converted to greyscale and replicated to three channels,
  resized to 112 × 112 pixels (aspect ratio not preserved), scaled to [0, 1], and normalised with the mean
  and standard deviation of the corresponding pretraining dataset.
- One model per view. Binary task with NSTEMI as the positive class.
- Patient-level random split into training (70%), validation (15%), and test (15%) subsets with seed 42.
  The same patients are assigned to the same subset in all three views.
- Adam optimiser (learning rate 1 × 10⁻⁴), batch size 8, 30 epochs, binary cross-entropy loss,
  mixed-precision training.
- The epoch with the highest validation AUC is selected and evaluated once on the test subset.

---

## Installation

### Option A: NVIDIA NGC PyTorch container (used for the reported results)

```bash
docker run -t -d --name echonste --runtime=nvidia --net=host --ipc=host \
  -v /path/to/workdir:/workspace nvcr.io/nvidia/pytorch:23.12-py3
docker exec -it echonste bash

cd /workspace/EchoNSTE-MI-code
pip install "numpy==1.26.4" "opencv-python-headless==4.10.0.84" \
            "pandas==2.2.2" "scikit-learn==1.5.1" tabulate
```

> **Note:** keep NumPy below 2.0 in this container. The bundled PyTorch 2.2 build is compiled against NumPy 1.x.
> If OpenCV fails to import, remove duplicate OpenCV installations (`pip list | grep -i opencv`) and reinstall
> a single `opencv-python-headless` package.

### Option B: plain Python environment

Python 3.10 and a CUDA-capable GPU are recommended.

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# 1. Check the environment and the dataset
python check_env.py                         # expected: RESULT: ready
python check_dataset.py /path/to/EchoNSTE-MI # expected: RESULT: OK

# 2. Quick test (one epoch)
python train.py --data /path/to/EchoNSTE-MI --out test_run --model cnn3d --view 4CH --epochs 1

# 3. Run all experiments
DATA=/path/to/EchoNSTE-MI OUT=/path/to/results CUDA_VISIBLE_DEVICES=0 bash run_all.sh

# 4. Verify the results
python verify_results.py /path/to/results
```

Main options of `train.py`:

| Option | Default | Description |
|---|---|---|
| `--model` | (required) | `cnn2d` or `cnn3d` |
| `--view` | (required) | `2CH`, `4CH`, or `PLAX` |
| `--frames` | 16 | Frames sampled per video |
| `--size` | 112 | Frame size in pixels |
| `--epochs` | 30 | Training epochs |
| `--batch` | 8 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--seed` | 42 | Random seed for the split and training |
| `--workers` | 4 | DataLoader workers (use 0 if shared memory is limited) |
| `--shuffle_labels` | off | Sanity check: trains on permuted labels; the test AUC should fall to about 0.5 |

### Output files

| File | Content |
|---|---|
| `split_patient_level.csv` | Assignment of every patient to the training, validation, or test subset |
| `<model>_<view>_log.csv` | Loss and validation metrics per epoch |
| `<model>_<view>_best.pt` | Weights of the selected epoch |
| `<model>_<view>_test_predictions.csv` | Predicted probability for every test video |
| `<model>_<view>_test_metrics.json` | AUC, accuracy, precision, sensitivity, specificity, F1-score, and confusion matrix |
| `summary_test_metrics.csv` | Summary table of all runs |

---

## Results

Test-set performance (n = 108 patients per view). AUC is reported with its bootstrap 95% confidence interval.

| View | Model | AUC (95% CI) | Accuracy | Precision | Sensitivity | Specificity | F1-score |
|---|---|---|---|---|---|---|---|
| 2CH | ResNet-18 (2D CNN) | – | – | – | – | – | – |
| 2CH | R(2+1)D-18 (3D CNN) | – | – | – | – | – | – |
| 4CH | ResNet-18 (2D CNN) | – | – | – | – | – | – |
| 4CH | R(2+1)D-18 (3D CNN) | – | – | – | – | – | – |
| PLAX | ResNet-18 (2D CNN) | – | – | – | – | – | – |
| PLAX | R(2+1)D-18 (3D CNN) | – | – | – | – | – | – |

The test subset is small, so differences between models of a few AUC points fall within the confidence
intervals. These results are reference baselines, not optimised models.

---

## Citation

If you use the dataset or this code, please cite:

```bibtex
@article{echonste_mi_2026,
  title   = {EchoNSTE-MI: A multi-view echocardiography video dataset for myocardial infarction
             classification in patients with non-ST-elevation acute coronary syndrome},
  author  = {Kurniawan, Yogiek Indra and Fadhlillah, Maulana Kholis and Mantau, Aprinaldi Jasa and
             Tandayu, Kevin Moses Hanky Jr and Sukmawan, Renan and Liastuti, Lies Dina and
             Pratama, Mahardhika and Jatmiko, Wisnu},
  journal = {Data in Brief},
  year    = {2026},
  note    = {Under review}
}

@misc{echonste_mi_data_2026,
  author    = {Kurniawan, Yogiek Indra},
  title     = {EchoNSTE-MI},
  year      = {2026},
  publisher = {Mendeley Data},
  version   = {1},
  doi       = {10.17632/y6z64ntstg.1}
}
```

---

## Ethics

The study was approved by the Ethical Committee of the National Cardiovascular Center Harapan Kita
(approval number DP.04.03/KEP054/EC021/2024). All videos were de-identified before release. Users must not
attempt to re-identify any individual.

## Licence

- **Code:** MIT licence (see `LICENSE`).
- **Dataset:** CC BY 4.0, distributed through Mendeley Data. The dataset is not stored in this repository.

## Acknowledgements

This work was supported by the Faculty of Computer Science, Universitas Indonesia; Universitas Indonesia under
the PUTI Hi-Impact Scheme (Contract No. PKS-32/UN2.RST/HKP.05.00/2025); and the Indonesia Endowment Fund for
Education (LPDP RI, grant no. LOG-19136/LPDP.3/2024). We thank the National Cardiovascular Center Harapan Kita
and the Department of Cardiology and Vascular Medicine, Faculty of Medicine, Universitas Indonesia.

## Contact
Yogiek Indra Kurniawan
yogiek@unsoed.ac.id
