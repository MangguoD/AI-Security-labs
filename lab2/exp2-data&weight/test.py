import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# 确保权重目录存在
os.makedirs("./weights", exist_ok=True)

# 预训练模型路径
pretrained_model = "./weights/lenet_mnist_model.pth"
use_cuda = False  # 不使用GPU


# ----------------------
# LeNet模型定义
# ----------------------
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ----------------------
# FGSM攻击实现
# ----------------------
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)


# ----------------------
# PGD攻击实现
# ----------------------
def pgd_attack(image, epsilon, alpha, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + alpha * sign_data_grad
    # 投影到ε邻域内
    eta = torch.clamp(perturbed_image - image, min=-epsilon, max=epsilon)
    return torch.clamp(image + eta, 0, 1)


# ----------------------
# 训练模型函数（如果预训练模型不存在）
# ----------------------
def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# ----------------------
# 测试函数
# ----------------------
def test(model, device, test_loader, epsilon, attack='fgsm', steps=1):
    correct = 0
    total = 0
    adv_examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        total += 1

        # 原始预测
        with torch.no_grad():
            output = model(data)
            init_pred = output.argmax(dim=1)

        # 只处理初始预测正确的样本
        if init_pred.item() != target.item():
            continue

        # 执行攻击
        if attack == 'fgsm':
            # 启用梯度计算
            data.requires_grad = True
            output = model(data)
            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = fgsm_attack(data, epsilon, data_grad)

        elif attack == 'pgd':
            perturbed_data = data.clone().detach()
            perturbed_data.requires_grad = True

            for _ in range(steps):
                # 确保梯度计算已启用
                with torch.enable_grad():
                    output = model(perturbed_data)
                    loss = F.nll_loss(output, target)

                    # 计算梯度
                    grad = torch.autograd.grad(loss, perturbed_data)[0]

                # 更新扰动数据
                perturbed_data = pgd_attack(
                    perturbed_data,
                    epsilon,
                    0.01,  # alpha
                    grad
                ).detach()

                # 确保数据在合理范围内
                perturbed_data = torch.clamp(perturbed_data, 0, 1)
                perturbed_data.requires_grad = True

        # 攻击后预测
        with torch.no_grad():
            output = model(perturbed_data)
            final_pred = output.argmax(dim=1)

            # 统计结果
            if final_pred.item() == target.item():
                correct += 1
            elif len(adv_examples) < 5:  # 保存示例
                adv_ex = perturbed_data.squeeze().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    accuracy = correct / total if total > 0 else 0
    print(f"Epsilon: {epsilon}\tAttack: {attack}\tAccuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy, adv_examples


# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    # 初始化模型
    model = LeNet().to(device)

    # 如果预训练模型不存在，则训练一个新模型
    if not os.path.exists(pretrained_model):
        print("Training model...")
        optimizer = torch.optim.Adam(model.parameters())
        for epoch in range(1, 6):  # 训练5个epoch
            train_model(model, device, train_loader, optimizer, epoch)
        torch.save(model.state_dict(), pretrained_model)
        print(f"Model saved to {pretrained_model}")
    else:
        model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

    model.eval()  # 评估模式

    # 参数设置
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    accuracies_fgsm, examples_fgsm = [], []
    accuracies_pgd, examples_pgd = [], []

    # FGSM测试
    print("\n===== Running FGSM attacks =====")
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps, attack='fgsm')
        accuracies_fgsm.append(acc)
        examples_fgsm.append(ex)

    # PGD测试 (默认迭代7步)
    print("\n===== Running PGD attacks =====")
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps, attack='pgd', steps=7)
        accuracies_pgd.append(acc)
        examples_pgd.append(ex)

    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, accuracies_fgsm, 'o-', label='FGSM')
    plt.plot(epsilons, accuracies_pgd, 's-', label='PGD (7 steps)')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 0.35, step=0.05))
    plt.title("Model Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_vs_epsilon.png')
    plt.show()

    # 可视化对抗样本 (FGSM)
    print("\nVisualizing FGSM adversarial examples...")
    plt.figure(figsize=(15, 10))
    for i, eps in enumerate(epsilons[:3]):
        for j in range(min(5, len(examples_fgsm[i]))):
            orig, adv, ex = examples_fgsm[i][j]
            plt.subplot(3, 5, i * 5 + j + 1)
            plt.imshow(ex, cmap='gray')
            plt.title(f"ε={eps}\n{orig}→{adv}", color='red' if orig != adv else 'green')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('fgsm_adversarial_examples.png')
    plt.show()

    # 可视化对抗样本 (PGD)
    print("\nVisualizing PGD adversarial examples...")
    plt.figure(figsize=(15, 10))
    for i, eps in enumerate(epsilons[:3]):
        for j in range(min(5, len(examples_pgd[i]))):
            orig, adv, ex = examples_pgd[i][j]
            plt.subplot(3, 5, i * 5 + j + 1)
            plt.imshow(ex, cmap='gray')
            plt.title(f"ε={eps}\n{orig}→{adv}", color='red' if orig != adv else 'green')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('pgd_adversarial_examples.png')
    plt.show()

    # 输出结果表格
    print("\n===== Results Summary =====")
    print("Epsilon\tFGSM Accuracy\tPGD Accuracy")
    for i, eps in enumerate(epsilons):
        print(f"{eps:.3f}\t{accuracies_fgsm[i]:.4f}\t\t{accuracies_pgd[i]:.4f}")

    # 保存结果到文件
    with open("results.txt", "w") as f:
        f.write("Epsilon\tFGSM Accuracy\tPGD Accuracy\n")
        for i, eps in enumerate(epsilons):
            f.write(f"{eps:.3f}\t{accuracies_fgsm[i]:.4f}\t\t{accuracies_pgd[i]:.4f}\n")