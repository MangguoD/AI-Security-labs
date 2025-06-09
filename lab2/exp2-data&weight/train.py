# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from argument import parser
from model import LeNet
from utils import setup_logging, save_model, evaluate
import logging

def pgd_attack(model, images, labels, device, epsilon, alpha, iters):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon).to(device)
    delta.requires_grad = True
    for _ in range(iters):
        outputs = model(images + delta)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = (delta + alpha * torch.sign(grad)).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.model = LeNet().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = nn.CrossEntropyLoss()
        setup_logging(args.save_dir)

    def train_epoch(self, loader, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            if self.args.adv_train:
                delta = pgd_attack(self.model, data, target, self.device,
                                   self.args.epsilon, self.args.alpha, self.args.pgd_steps)
                data = data + delta
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.args.log_interval == 0:
                msg = f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(loader.dataset)}] Loss: {loss.item():.4f}"
                print(msg)
                logging.info(msg)

    def train(self):
        # dataloaders
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False)

        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(train_loader, epoch)
            acc_clean = evaluate(self.model, test_loader, self.device)
            msg = f"Epoch {epoch}: Clean Accuracy: {acc_clean*100:.2f}%"
            print(msg); logging.info(msg)
            # 保存模型
            if epoch % self.args.save_model_interval == 0:
                save_path = os.path.join(self.args.save_dir, f"lenet_epoch{epoch}.pth")
                save_model(self.model, save_path)
        print("Training complete.")

if __name__ == '__main__':
    args = parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(args, device)
    trainer.train()