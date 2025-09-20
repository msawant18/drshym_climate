import argparse, yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio as rio

from models.encoder_backbones import make_encoder
from models.unet import UNet
from utils.seed import fix_seeds

class TiffMaskDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        img_tif, mask_tif = self.pairs[i]
        with rio.open(img_tif) as s: x = s.read(1).astype('float32')
        with rio.open(mask_tif) as s: y = s.read(1).astype('float32')
        import numpy as np
        return torch.from_numpy(x)[None], torch.from_numpy(y)[None]

def dice_bce_loss(logits, y):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    num = 2*(p*y).sum(dim=(2,3))
    den = (p+y).sum(dim=(2,3))+1e-6
    dice = 1 - (num/den).mean()
    return bce + dice

def main(cfg):
    fix_seeds(cfg['seed'])
    train_pairs = []
    for p in Path('data/tiles/train').glob('*.tif'):
        m = p.with_name(p.stem + '_mask.tif')
        if m.exists(): train_pairs.append((str(p), str(m)))
    val_pairs = []
    for p in Path('data/tiles/val').glob('*.tif'):
        m = p.with_name(p.stem + '_mask.tif')
        if m.exists(): val_pairs.append((str(p), str(m)))

    dl_tr = DataLoader(TiffMaskDataset(train_pairs), batch_size=cfg['model']['batch_size'], shuffle=True, num_workers=2)
    dl_va = DataLoader(TiffMaskDataset(val_pairs), batch_size=cfg['model']['batch_size'], shuffle=False, num_workers=2)

    layers, chs = make_encoder(cfg['model']['encoder'], pretrained=False)
    model = UNet(layers, chs)
    opt = optim.AdamW(model.parameters(), lr=cfg['model']['lr'])

    best_iou = -1
    for ep in range(cfg['model']['epochs']):
        model.train(); tl=0
        for xb, yb in dl_tr:
            opt.zero_grad();
            logits = model(xb)
            loss = dice_bce_loss(logits, yb)
            loss.backward(); opt.step(); tl += loss.item()
        model.eval();
        iou_sum=0; n=0
        with torch.inference_mode():
            for xb, yb in dl_va:
                p = torch.sigmoid(model(xb))
                pred = (p>0.5).float()
                inter = (pred*yb).sum()
                union = pred.sum()+yb.sum()-inter+1e-6
                iou_sum += float(inter/union); n+=1
        miou = iou_sum/max(n,1)
        print(f"epoch {ep}: train_loss={tl/len(dl_tr):.4f} val_iou={miou:.3f}")
        if miou>best_iou:
            best_iou=miou
            import torch as _t
            Path('artifacts/checkpoints').mkdir(parents=True, exist_ok=True)
            _t.save(model.state_dict(), 'artifacts/checkpoints/best.pt')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/flood.yaml')
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    main(cfg)
