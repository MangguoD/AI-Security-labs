# visualize.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
from utils import load_model, evaluate
from torchvision import datasets, transforms
from argument import parser

def test_and_plot(args, epsilons=[0,0.05,0.1,0.2,0.3]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    results = []
    for eps in epsilons:
        model = LeNet().to(device)
        model = load_model(model, os.path.join(args.save_dir, f"lenet_epoch{args.epochs}.pth"), device)
        # 对不同 ε 用 PGD 攻击并评估
        correct = total = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if eps > 0:
                delta = torch.zeros_like(data).uniform_(-eps, eps).to(device)
                adv_data = (data + delta).clamp(0,1)
            else:
                adv_data = data
            output = model(adv_data)
            _, pred = output.max(1)
            total += target.size(0)
            correct += (pred == target).sum().item()
        acc = correct / total
        results.append((eps, acc*100))
        print(f"Epsilon {eps:.2f} -> Accuracy: {acc*100:.2f}%")

    # 绘图
    eps_vals, acc_vals = zip(*results)
    plt.figure()
    plt.plot(eps_vals, acc_vals, marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Robustness under PGD-like Noise')
    plt.grid(True)
    os.makedirs(args.save_dir, exist_ok=True)
    plt.savefig(os.path.join(args.save_dir, 'robustness.png'))
    plt.show()

if __name__ == '__main__':
    args = parser()
    test_and_plot(args)