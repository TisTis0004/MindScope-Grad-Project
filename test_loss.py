import sys
sys.path.insert(0, '.')
from train_eegnet import build_eegnet, EEG1DAugmentation, FocalLoss
import torch
from data.dataloaderV2 import Loader

dl = Loader(ds='cache_windows_binary_10_sec/manifest.jsonl', batch_size=64, num_workers=0).return_Loader()
device = torch.device('cuda')
m = build_eegnet(device)
m.train()
aug = EEG1DAugmentation().to(device)
crit = FocalLoss(gamma=2.0).to(device)
scaler = torch.amp.GradScaler('cuda')

it = iter(dl)
b = next(it)
x = b['x'].to(device)
y = b['y'].to(device).long()

print(f"Max before clamp: {x.abs().max().item()}")
x = torch.clamp(x, -20.0, 20.0)
print(f"Max after clamp: {x.abs().max().item()}")

x = aug(x)

with torch.amp.autocast('cuda'):
    out = m(x)
    loss = crit(out, y)

print(f'Max logit: {out.abs().max().item()}')
print(f'Loss: {loss.item()}')

scaler.scale(loss).backward()
print('Backward OK')
