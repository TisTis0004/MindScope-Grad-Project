# ▶️ MindScope: Two-Stage Cascade EEG Seizure Detection (V2.1)

This repository contains the data generation and training pipeline for
the **MindScope** project. It implements a **Two-Stage Cascade
approach** for EEG seizure detection, transitioning from raw TUH EDF
files to efficient PyTorch tensor streams.

Currently, the pipeline covers: - Data Setup - Stage 1 (Binary
Classification) Training

------------------------------------------------------------------------

## 📁 Project Structure Overview

    Cascade_MindScope-Grad-Project_V2.1
    │
    ├── Binary_Balanced_Run_1_history.csv
    ├── Binary_Balanced_Run_1.pt
    ├── Binary_Imbalanced_Finetune_Run_2_history.csv
    ├── Binary_Imbalanced_Finetune_Run_2.pt
    ├── cache_windows_eval_8_classes
    │   ├── aaaaaaaq_s006_t000.pt
    │   ├── aaaaaarq_s014_t001.pt
    │   ├── ...
    │   ├── label_map.json
    │   └── manifest.jsonl
    ├── cache_windows_train_8_classes
    │   ├── aaaaaaac_s001_t000.pt
    │   ├── ...
    │   ├── label_map.json
    │   └── manifest.jsonl
    ├── data
    │   ├── cache_windows.py
    │   ├── dataloader.py
    │   ├── dataset.py
    │   ├── __init__.py
    │   ├── minfest_effient.py
    │   ├── ratio.py
    │   └── total_disk_size.py
    ├── eeg_seizure_only_eval.json
    ├── eeg_seizure_only_train.json
    ├── filter_stage.py
    ├── helper
    │   ├── train_helper.py
    ├── label_map.json
    ├── models
    │   ├── models.py
    ├── purify_manifest.py
    ├── README.md
    ├── train.py
    ├── tuh_eval_index.json
    └── tuh_train_index.json

------------------------------------------------------------------------

## 🚀 Pipeline Instructions

Follow these steps in order to reproduce the data pipeline and Stage 1
model training.

------------------------------------------------------------------------

### Step 0 --- Download the Dataset

https://isip.piconepress.com/projects/tuh_eeg/

-   Extract locally (SSD recommended)

------------------------------------------------------------------------

### Step 1 --- Generate Dataset Metadata (JSON)

``` bash
python data/dataset.py
```

**What this does:** - Scans dataset structure - Validates `.edf`,
`.csv`, `.csv_bi` - Generates JSON metadata

------------------------------------------------------------------------

### Step 2 --- Cache EEG Windows (.pt files)

``` bash
python data/cache_windows.py
```

**What this does:** 
- Loads EEG via MNE
- Applies bandpass (0.5--40Hz) 
- Extracts 1s windows (21 channels)
- Labels windows
- Saves `.pt` tensors

------------------------------------------------------------------------

### Step 3 --- Build Efficient Manifest Files

``` bash
python data/minfest_effient.py
```

**What this does:**
- Scans cached files
- Builds `manifest.jsonl`
- Enables streaming dataset

------------------------------------------------------------------------

### Step 4 --- Train Stage 1 (Binary Detector)

#### Phase 1: Balanced Initialization

``` bash
python train.py
```

- Uses 1:1 undersampling
- Produces `Binary_Balanced_Run_1.pt`

------------------------------------------------------------------------

#### Phase 2: Imbalanced Fine-Tuning

``` bash
python train.py
```

- Uses full dataset
- Dynamic class weighting
- Lower LR (e.g., 5e-5)
- Produces `Binary_Imbalanced_Finetune_Run_2.pt`

------------------------------------------------------------------------

## ⏭️ Next Steps (Stage 2)

``` bash
python purify_manifest.py
```

- Filters background using high-confidence predictions
- Produces seizure-focused dataset
- Used for multiclass training

------------------------------------------------------------------------

## ⚠️ Important Notes

- Run from project root
- Step 2 → CPU heavy
- Step 4 → GPU recommended
- Store `.pt` on SSD/NVMe
- Avoid window-level shuffling
