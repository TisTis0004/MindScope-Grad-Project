
# 🧠 EEG Experiment Report: eegtcn_afterSampling.pt

## 📋 1. Meta Data
- **Target Metric:** f1_macro
- **Best Score achieved:** **0.7196106691049794**
- **Total Training Epochs:** 2
- **AMP Used:** Yes

## ⚙️ 2. Model Configuration (Hyperparameters)
| Parameter | Value |
| :--- | :--- |
| **Learning Rate** | 0.001 |
| **Number of Classes** | 2 |
| **Random Seed** | 3025 |
| **Patience** | 60 |
| **Input Shape** | 17 channels x 2500 samples |

## 📊 3. Performance Breakdown
*Extracted from the last recorded training step.*

| Metric | Training | Validation |
| :--- | :--- | :--- |
| **Final f1_macro** | N/A | N/A |

## 🛠️ 4. Training State Verification
- [x] **Model State:** `model_state_dict` loaded
- [x] **Optimizer State:** `optimizer_state_dict` loaded
- [x] **Scheduler State:** `scheduler_state_dict` loaded

---
### 📝 Notes for Paper
* Architecture: **EEGNet** (Braindecode implementation).
* The model used `0.001` as the initial learning rate.
* The best performance was achieved at epoch `2`.

---
*Generated via eeg_tracker*
