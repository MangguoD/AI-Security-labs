# utils.py
import os
import logging
import torch
import numpy as np

def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(save_dir, 'train.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

def numpy2tensor(arr, device):
    return torch.from_numpy(arr).float().to(device)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = outputs.max(1)
            total += y.size(0)
            correct += (preds == y).sum().item()
    return correct / total