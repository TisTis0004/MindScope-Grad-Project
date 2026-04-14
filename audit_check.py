"""Full pipeline audit script — run once to check all consistency issues."""
import torch
import glob

def check_cache(name, pattern):
    files = glob.glob(pattern)
    print(f"\n{'='*50}")
    print(f"  {name}: {len(files)} .pt files")
    print(f"{'='*50}")
    if not files:
        print("  *** EMPTY — no cached data found! ***")
        return None
    
    d = torch.load(files[0], weights_only=False)
    x = d["x"]
    y = d["y"]
    m = d.get("meta", {})
    
    info = {
        "n_files": len(files),
        "channels": x.shape[1],
        "time_samples": x.shape[2],
        "fs": m.get("fs", "UNKNOWN"),
    }
    
    print(f"  x.shape       = {list(x.shape)}")
    print(f"  y.shape       = {list(y.shape)}")
    print(f"  x range       = [{x.min():.4f}, {x.max():.4f}]")
    print(f"  x mean/std    = {x.mean():.4f} / {x.std():.4f}")
    print(f"  Has NaN       = {torch.isnan(x).any().item()}")
    print(f"  Has Inf       = {torch.isinf(x).any().item()}")
    print(f"  fs (meta)     = {info['fs']}")
    
    ch_names = m.get("final_montage_channels", m.get("target_montage_channels", "N/A"))
    print(f"  Montage       = {ch_names}")
    print(f"  Labels        = {dict(zip(*torch.unique(y, return_counts=True)))}")
    
    return info


print("=" * 60)
print("  FULL PIPELINE AUDIT")
print("=" * 60)

# 1. Check cached data
train_info = check_cache("TRAIN CACHE", "cache_windows_binary_10_sec/*.pt")
eval_info = check_cache("EVAL CACHE", "cache_windows_binary_10_sec_eval/*.pt")

# 2. Consistency check
print(f"\n{'='*50}")
print("  TRAIN vs EVAL CONSISTENCY")
print(f"{'='*50}")
if train_info and eval_info:
    for key in ["channels", "time_samples", "fs"]:
        tv = train_info[key]
        ev = eval_info[key]
        status = "OK" if tv == ev else "*** MISMATCH ***"
        print(f"  {key:15s}: train={tv}, eval={ev}  {status}")
else:
    print("  Cannot compare — one or both caches are empty")

# 3. Model check
print(f"\n{'='*50}")
print("  MODEL vs DATA CONSISTENCY")
print(f"{'='*50}")
try:
    from helper.models import Spectrogram_CNN_LSTM
    model = Spectrogram_CNN_LSTM()
    model_channels = model.cnn[0].in_channels
    print(f"  Model expects {model_channels} input channels (from CNN first layer)")
    if train_info:
        print(f"  Train data has {train_info['channels']} channels", 
              "OK" if model_channels == train_info["channels"] else "*** MISMATCH ***")
except Exception as e:
    print(f"  Could not load model: {e}")

# 4. Transform check
print(f"\n{'='*50}")
print("  TRANSFORM SETTINGS")
print(f"{'='*50}")
from helper.T import EEGToSpectrogram
t = EEGToSpectrogram()
mel = t.spectrogram
print(f"  MelSpec sample_rate = {mel.sample_rate}")
print(f"  MelSpec n_fft       = {mel.n_fft}")
print(f"  MelSpec f_min       = {mel.f_min}")
print(f"  MelSpec f_max       = {mel.f_max}")
print(f"  MelSpec n_mels      = {mel.n_mels}")

if train_info:
    fs_match = mel.sample_rate == train_info["fs"]
    print(f"  sample_rate vs data fs: {mel.sample_rate} vs {train_info['fs']}",
          "OK" if fs_match else "*** MISMATCH ***")

# 5. Test forward pass
print(f"\n{'='*50}")
print("  FORWARD PASS TEST")
print(f"{'='*50}")
if train_info:
    C = train_info["channels"]
    T_len = train_info["time_samples"]
    dummy = torch.randn(2, C, T_len)
    
    t.train()
    out_train = t(dummy)
    print(f"  Train mode: input {list(dummy.shape)} -> output {list(out_train.shape)}")
    print(f"  Output NaN: {torch.isnan(out_train).any().item()}")
    print(f"  Output Inf: {torch.isinf(out_train).any().item()}")
    
    t.eval()
    out_eval = t(dummy)
    print(f"  Eval mode:  input {list(dummy.shape)} -> output {list(out_eval.shape)}")
    print(f"  Output NaN: {torch.isnan(out_eval).any().item()}")
    
    # Check model accepts the spectrogram
    try:
        model = Spectrogram_CNN_LSTM()
        logits = model(out_eval)
        print(f"  Model output: {list(logits.shape)}, NaN: {torch.isnan(logits).any().item()}")
        print(f"  FULL PIPELINE: OK")
    except Exception as e:
        print(f"  Model forward FAILED: {e}")

print(f"\n{'='*50}")
print("  AUDIT COMPLETE")
print(f"{'='*50}")
