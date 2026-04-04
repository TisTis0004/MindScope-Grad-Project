# gradcam_all.py  —  True Grad-CAM on ALL windows, no PNGs
# Runs a real Grad-CAM backward pass on every window across
# all 195 .pt files. Saves only summary.csv and summary.png.


# Output: gradcam_all_results/
#   summary.csv — one row per window with true Grad-CAM peak band
#   summary.png — comparison figure with correct attention statistics
import os, json, csv, shutil, tempfile, warnings, time
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom
from pathlib import Path

warnings.filterwarnings("ignore")
from dataloader import STFTSpec, SpecZNorm

MODEL_PATH    = "model_weights.pth"
EVAL_DIR      = r"G:\.shortcut-targets-by-id\1IS7vV1RQpfSoVy_vC4cp3EmiZ-sVdd6t\data V1\binary data\cache_windows_eval"
MANIFEST_PATH = os.path.join(EVAL_DIR, "manifest.jsonl")
OUT_DIR       = Path("gradcam_all_results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_pt(drive_path):
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        shutil.copy2(drive_path, tmp_path)
        data = torch.load(tmp_path, map_location="cpu", weights_only=False)
        return data
    except OSError:
        return None
    finally:
        try: Path(tmp_path).unlink(missing_ok=True)
        except: pass

def remap(pt_path_from_manifest):
    return os.path.join(EVAL_DIR, Path(pt_path_from_manifest).name)

def build_model():
    m = models.resnet18()
    m.conv1   = nn.Conv2d(41, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, 1)
    return m

transform = nn.Sequential(
    STFTSpec(fs=250, n_fft=256, hop_length=64, fmin=1.0, fmax=40.0),
    SpecZNorm(),
).eval()

# TRUE GRAD-CAM (one backward pass per window)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model      = model
        self._features  = None
        self._gradients = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_features", o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_gradients", go[0].detach()))

    def run(self, inp):
        """
        inp: [1, 41, 39, 40] on DEVICE
        Returns: cam [39, 40] numpy, prob float
        """
        inp   = inp.float().requires_grad_(True)
        logit = self.model(inp)
        prob  = torch.sigmoid(logit).item()
        self.model.zero_grad()
        logit.backward()

        w   = self._gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((w * self._features).sum(dim=1)).squeeze().cpu().numpy()

        if cam.ndim < 2 or cam.shape[0] == 0:
            return np.zeros((39, 40)), prob
        cam = zoom(cam, (39 / cam.shape[0], 40 / cam.shape[1]), order=1)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, prob


def bin_to_hz(bin_idx):
    return round(bin_idx * (250 / 256) + 1.95)

def band_name(hz):
    if hz <= 4:   return "Delta"
    if hz <= 8:   return "Theta"
    if hz <= 13:  return "Alpha"
    if hz <= 30:  return "Beta"
    return "Gamma"

def save_summary(all_rows, out_path):
    sei_hz = [r["cam_peak_hz"] for r in all_rows if r["true_label"] == 1]
    bkg_hz = [r["cam_peak_hz"] for r in all_rows if r["true_label"] == 0]
    sig_hz_sei = [r["sig_peak_hz"] for r in all_rows if r["true_label"] == 1]
    sig_hz_bkg = [r["sig_peak_hz"] for r in all_rows if r["true_label"] == 0]
    bins = list(range(2, 42, 2))

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    def dark(ax):
        ax.set_facecolor("#0f0f1a"); ax.tick_params(colors="white")

    # Panel 1 — True Grad-CAM attention: seizure vs background band %
    ax1 = fig.add_subplot(gs[0, :2]); dark(ax1)
    br  = [(1,4),(4,8),(8,13),(13,30),(30,40)]
    bl  = ["Delta\n1-4 Hz","Theta\n4-8 Hz","Alpha\n8-13 Hz","Beta\n13-30 Hz","Gamma\n30+ Hz"]
    sp  = [100*sum(1 for hz in sei_hz if lo<=hz<hi)/max(len(sei_hz),1) for lo,hi in br]
    bp  = [100*sum(1 for hz in bkg_hz if lo<=hz<hi)/max(len(bkg_hz),1) for lo,hi in br]
    x = np.arange(5); w = 0.35
    b1 = ax1.bar(x-w/2, sp, w, label="Seizure",    color="#ff6666", edgecolor="black", lw=0.5)
    b2 = ax1.bar(x+w/2, bp, w, label="Background", color="#6699ff", edgecolor="black", lw=0.5)
    for bar, val in zip(list(b1)+list(b2), sp+bp):
        if val > 1:
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f"{val:.1f}%", ha="center", color="white", fontsize=8, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(bl, color="white", fontsize=10)
    ax1.set_ylabel("% of windows with Grad-CAM peak in this band", color="white", fontsize=9)
    ax1.set_title("TRUE Grad-CAM Attention — Where the model looked\n"
                  "(backward pass on every window, not signal power)",
                  color="white", fontweight="bold", fontsize=11)
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)

    # Panel 2 — Signal power vs Grad-CAM attention for seizure windows
    ax2 = fig.add_subplot(gs[0, 2]); dark(ax2)
    sp_sig = [100*sum(1 for hz in sig_hz_sei if lo<=hz<hi)/max(len(sig_hz_sei),1) for lo,hi in br]
    ax2.barh(np.arange(5)+0.2, sp,     0.35, label="Grad-CAM attention", color="#ff6666", edgecolor="black", lw=0.5)
    ax2.barh(np.arange(5)-0.2, sp_sig, 0.35, label="Signal power",       color="#ffaa44", edgecolor="black", lw=0.5)
    ax2.set_yticks(np.arange(5)); ax2.set_yticklabels(bl, color="white", fontsize=9)
    ax2.set_xlabel("% of seizure windows", color="white", fontsize=9)
    ax2.set_title("Seizure windows:\nGrad-CAM attention vs signal power",
                  color="white", fontweight="bold", fontsize=10)
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    # Panel 3 — Accuracy breakdown
    ax3 = fig.add_subplot(gs[1, 0]); dark(ax3)
    sc = sum(1 for r in all_rows if r["true_label"]==1 and r["correct"])
    sw = sum(1 for r in all_rows if r["true_label"]==1 and not r["correct"])
    bc = sum(1 for r in all_rows if r["true_label"]==0 and r["correct"])
    bw = sum(1 for r in all_rows if r["true_label"]==0 and not r["correct"])
    total = len(all_rows)
    cats  = ["Seizure\nDetected","Seizure\nMissed","Background\nCorrect","Background\nFalse Alarm"]
    cnts  = [sc, sw, bc, bw]
    clrs  = ["#88cc88","#ff6666","#88aaff","#ffaa44"]
    bars  = ax3.bar(cats, cnts, color=clrs, edgecolor="black", lw=0.5)
    for bar, cnt in zip(bars, cnts):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                 str(cnt), ha="center", color="white", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Number of windows", color="white")
    ax3.set_title("Prediction Accuracy", color="white", fontweight="bold")
    ax3.set_xlabel(f"Accuracy {100*(sc+bc)/total:.1f}%  |  "
                   f"Sensitivity {100*sc/max(sc+sw,1):.1f}%  |  "
                   f"Specificity {100*bc/max(bc+bw,1):.1f}%",
                   color="white", fontsize=9)

    # Panel 4 — Grad-CAM histogram: seizure
    ax4 = fig.add_subplot(gs[1, 1]); dark(ax4)
    ax4.hist(sei_hz, bins=bins, color="#ff6666", edgecolor="black", lw=0.4)
    ax4.set_title("Seizure — Grad-CAM peak frequency distribution",
                  color="#ff6666", fontweight="bold")
    ax4.set_xlabel("Hz", color="white"); ax4.set_ylabel("Windows", color="white")
    for x_line in [4,8,13,30]: ax4.axvline(x_line, color="gray", lw=0.5, ls="--")

    # Panel 5 — Grad-CAM histogram: background
    ax5 = fig.add_subplot(gs[1, 2]); dark(ax5)
    ax5.hist(bkg_hz, bins=bins, color="#6699ff", edgecolor="black", lw=0.4)
    ax5.set_title("Background — Grad-CAM peak frequency distribution",
                  color="#6699ff", fontweight="bold")
    ax5.set_xlabel("Hz", color="white"); ax5.set_ylabel("Windows", color="white")
    for x_line in [4,8,13,30]: ax5.axvline(x_line, color="gray", lw=0.5, ls="--")

    fig.suptitle(
        f"MindScope ResNet18 — TRUE Grad-CAM Results\n"
        f"Full evaluation set: {total} windows  |  "
        f"{len(sei_hz)} seizure  |  {len(bkg_hz)} background",
        color="white", fontsize=14, fontweight="bold"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print(f"Summary figure saved → {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device : {DEVICE}" +
          (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    print("Mode   : TRUE Grad-CAM (backward pass every window) — no PNGs saved")
    print()

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    gradcam = GradCAM(model, model.layer4[-1].conv2)
    print("Model loaded.\n")

    with open(MANIFEST_PATH) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    print(f"Manifest: {len(entries)} files\n")

    all_rows      = []
    csv_rows      = []
    global_idx    = 0
    skipped_files = 0
    t_start       = time.time()

    for file_num, entry in enumerate(entries, 1):
        pt = remap(entry["pt_path"])
        if not os.path.exists(pt):
            skipped_files += 1
            continue

        data = load_pt(pt)
        if data is None:
            skipped_files += 1
            continue

        fname  = Path(pt).name
        x_all  = data["x"]          # [N, 41, 2500] CPU
        y_all  = data["y"].tolist()
        N      = len(y_all)

        # Pre-compute all spectrograms for this file (CPU, fast)
        with torch.no_grad():
            specs = torch.stack([transform(x_all[i]) for i in range(N)])  # [N,41,39,40]

        # True Grad-CAM: one backward pass per window
        for win_idx in range(N):
            true_label = int(y_all[win_idx])
            global_idx += 1
            label_str  = "SEI" if true_label == 1 else "BKG"
            window_id  = f"{label_str}_{global_idx:05d}"

            inp      = specs[win_idx].unsqueeze(0).to(DEVICE)   # [1,41,39,40]
            cam, prob = gradcam.run(inp)                         # backward pass here

            predicted = "SEIZURE" if prob > 0.5 else "BACKGROUND"
            correct   = (prob > 0.5) == (true_label == 1)

            # TRUE Grad-CAM peak: row-averaged heatmap argmax
            cam_peak_bin = int(cam.mean(axis=1).argmax())
            cam_peak_hz  = bin_to_hz(cam_peak_bin)
            cam_peak_band = band_name(cam_peak_hz)

            # Signal power peak (for comparison column in CSV)
            sig_power    = specs[win_idx].numpy().mean(axis=(0, 2))  # [39]
            sig_peak_hz  = bin_to_hz(int(sig_power.argmax()))
            sig_peak_band = band_name(sig_peak_hz)

            row = dict(
                window_id     = window_id,
                file          = fname,
                win_idx       = win_idx,
                true_label    = true_label,
                predicted     = predicted,
                prob          = round(prob, 4),
                correct       = correct,
                cam_peak_hz   = cam_peak_hz,
                cam_peak_band = cam_peak_band,
                sig_peak_hz   = sig_peak_hz,
                sig_peak_band = sig_peak_band,
            )
            all_rows.append(row)
            csv_rows.append([
                window_id, fname, win_idx, true_label, predicted,
                round(prob, 4), correct,
                cam_peak_hz, cam_peak_band,
                sig_peak_hz, sig_peak_band,
            ])

        elapsed = time.time() - t_start
        rate    = global_idx / max(elapsed, 0.001)
        remaining = (30479 - global_idx) / max(rate, 0.001)
        print(f"  File {file_num:3d}/195 | {fname} | {N} windows | "
              f"total {global_idx} | {rate:.0f} win/s | ETA {remaining/60:.1f} min")

        del data, x_all, specs   # free RAM before next file

    # Save CSV
    csv_path = OUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window_id","file","win_idx","true_label","predicted","prob","correct",
            "cam_peak_hz","cam_peak_band",   # TRUE Grad-CAM
            "sig_peak_hz","sig_peak_band",   # raw signal power (for comparison)
        ])
        writer.writerows(csv_rows)
    print(f"\nCSV saved → {csv_path}")

    save_summary(all_rows, OUT_DIR / "summary.png")

    n_total  = len(all_rows)
    n_ok     = sum(1 for r in all_rows if r["correct"])
    n_sei    = sum(1 for r in all_rows if r["true_label"] == 1)
    n_bkg    = sum(1 for r in all_rows if r["true_label"] == 0)
    n_sei_ok = sum(1 for r in all_rows if r["true_label"] == 1 and r["correct"])
    n_bkg_ok = sum(1 for r in all_rows if r["true_label"] == 0 and r["correct"])

    cam_sei_bands, cam_bkg_bands = {}, {}
    sig_sei_bands, sig_bkg_bands = {}, {}
    for r in all_rows:
        cb = r["cam_peak_band"]; sb = r["sig_peak_band"]
        if r["true_label"] == 1:
            cam_sei_bands[cb] = cam_sei_bands.get(cb, 0) + 1
            sig_sei_bands[sb] = sig_sei_bands.get(sb, 0) + 1
        else:
            cam_bkg_bands[cb] = cam_bkg_bands.get(cb, 0) + 1
            sig_bkg_bands[sb] = sig_bkg_bands.get(sb, 0) + 1

    total_time = time.time() - t_start
    print("\n")
    print("FINAL SUMMARY — TRUE GRAD-CAM (full 195-file eval set)")
    print("\n")
    print(f"Total windows : {n_total}  |  Time: {total_time/60:.1f} min")
    if skipped_files: print(f"Skipped files : {skipped_files} (cloud-only/missing)")
    print(f"Accuracy      : {n_ok}/{n_total} = {100*n_ok/max(n_total,1):.1f}%")
    print(f"Sensitivity   : {n_sei_ok}/{n_sei} = {100*n_sei_ok/max(n_sei,1):.1f}%  (seizure recall)")
    print(f"Specificity   : {n_bkg_ok}/{n_bkg} = {100*n_bkg_ok/max(n_bkg,1):.1f}%  (background recall)")

    print(f"\n── TRUE Grad-CAM peak band ──────────────────────────────────────")
    print("SEIZURE windows:")
    for b, c in sorted(cam_sei_bands.items(), key=lambda x: -x[1]):
        print(f"  {b:10s}: {c:5d}  ({100*c/max(n_sei,1):.1f}%)")
    print("BACKGROUND windows:")
    for b, c in sorted(cam_bkg_bands.items(), key=lambda x: -x[1]):
        print(f"  {b:10s}: {c:5d}  ({100*c/max(n_bkg,1):.1f}%)")

    print(f"\n── Signal power peak (for comparison) ──────────────────────────")
    print("SEIZURE windows:")
    for b, c in sorted(sig_sei_bands.items(), key=lambda x: -x[1]):
        print(f"  {b:10s}: {c:5d}  ({100*c/max(n_sei,1):.1f}%)")
    print("BACKGROUND windows:")
    for b, c in sorted(sig_bkg_bands.items(), key=lambda x: -x[1]):
        print(f"  {b:10s}: {c:5d}  ({100*c/max(n_bkg,1):.1f}%)")


if __name__ == "__main__":
    main()