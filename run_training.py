"""Train segmentation model."""
import argparse, os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from unet import AttentionUNet

class SatelliteDataset(Dataset):
    def __init__(self, data_dir, patch_size=256):
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        self.patch_size = patch_size
    def __len__(self): return len(self.files) * 4
    def __getitem__(self, idx):
        data = np.load(self.files[idx // 4])
        img = data['image'].astype(np.float32)
        masks = data['masks'].astype(np.float32)
        # random crop
        h, w = img.shape[:2]
        ps = self.patch_size
        y = np.random.randint(0, max(h - ps, 1))
        x = np.random.randint(0, max(w - ps, 1))
        img_patch = img[y:y+ps, x:x+ps].transpose(2, 0, 1)  # CHW
        mask_patch = masks[y:y+ps, x:x+ps].transpose(2, 0, 1)
        # normalize
        for ch in range(img_patch.shape[0]):
            m = img_patch[ch].max()
            if m > 0: img_patch[ch] /= m
        return torch.from_numpy(img_patch), torch.from_numpy(mask_patch)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionUNet(in_channels=3, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    dataset = SatelliteDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
            torch.save(model.state_dict(), f'checkpoints/epoch{epoch+1}.pt')
    torch.save(model.state_dict(), 'checkpoints/final.pt')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/processed')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    train(p.parse_args())
