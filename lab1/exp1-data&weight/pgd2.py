import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# LeNet模型定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# FGSM攻击函数
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# PGD攻击函数
def pgd_attack(image, epsilon, alpha, data_grad):
    perturbed_image = image + alpha * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# 测试函数
def test(model, device, test_loader, epsilon, attack='fgsm', step=1):
    correct = 0
    adv_examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        if attack == 'fgsm':
            data.requires_grad = True
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]

            if init_pred.item() != target.item():
                continue

            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = fgsm_attack(data, epsilon, data_grad)

        elif attack == 'pgd':
            perturbed_data = data.clone().detach()
            perturbed_data.requires_grad = True

            for _ in range(step):
                output = model(perturbed_data)
                init_pred = output.max(1, keepdim=True)[1]

                if init_pred.item() != target.item():
                    break

                loss = F.nll_loss(output, target)
                model.zero_grad()
                perturbed_data.grad = None
                loss.backward()
                data_grad = perturbed_data.grad.data
                perturbed_data = pgd_attack(perturbed_data, epsilon, 0.01, data_grad)
                perturbed_data = perturbed_data.detach()
                perturbed_data.requires_grad = True

        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct}/{len(test_loader)} = {final_acc}")

    return final_acc, adv_examples
def test_pgd(model, device, test_loader, epsilon, alpha, steps):
    correct = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        perturbed_data = data.clone().detach()
        perturbed_data.requires_grad = True

        for _ in range(steps):
            output = model(perturbed_data)
            init_pred = output.max(1, keepdim=True)[1]
            if init_pred.item() != target.item():
                break

            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = perturbed_data.grad.data
            perturbed_data = pgd_attack(perturbed_data, epsilon, alpha, data_grad)
            perturbed_data = perturbed_data.detach()
            perturbed_data.requires_grad = True

        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1

    acc = correct / float(len(test_loader))
    print(f"ε={epsilon:.2f}, α={alpha}, steps={steps} -> acc={acc:.4f}")
    return acc

def plot_accuracy_vs_step(model, device, test_loader, epsilon, alpha, step_list):
    accuracies = []
    for step in step_list:
        acc = test_pgd(model, device, test_loader, epsilon, alpha, step)
        accuracies.append(acc)

    plt.figure(figsize=(8, 5))
    plt.plot(step_list, accuracies, marker='o', color='blue', label=f"ε={epsilon}, α={alpha}")
    plt.title("PGD Accuracy vs Steps")
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(step_list)
    plt.yticks(np.arange(0.0, 1.01, 0.05))
    plt.legend()
    plt.tight_layout()
    plt.savefig("pgd_accuracy_vs_steps.png")
    plt.show()
def plot_accuracy_vs_alpha(model, device, test_loader, epsilon, alpha_list, steps):
    accuracies = []
    for alpha in alpha_list:
        acc = test_pgd(model, device, test_loader, epsilon, alpha, steps)
        accuracies.append(acc)

    plt.figure(figsize=(8, 5))
    plt.plot(alpha_list, accuracies, marker='o', color='green', label=f"ε={epsilon}, step={steps}")
    plt.title("PGD Accuracy vs Alpha")
    plt.xlabel("Alpha (Step Size)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(alpha_list)
    plt.yticks(np.arange(0.0, 1.01, 0.05))
    plt.legend()
    plt.tight_layout()
    plt.savefig("pgd_accuracy_vs_alpha.png")
    plt.show()

def main():
    epsilon = 0.2
    pretrained_model = "./weights/lenet_mnist_model.pth"
    use_cuda = False

    print("CUDA Available:", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model = Net().to(device)

    try:
        model.load_state_dict(torch.load(pretrained_model, map_location=device))
    except FileNotFoundError:
        print(f"错误：找不到预训练模型文件 {pretrained_model}")
        return

    model.eval()

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=1, shuffle=True
    )

    # 绘图1：横轴为 steps，alpha 固定
    # alpha_fixed = 0.02
    # step_list = [1, 2, 4, 6, 8, 10]
    # plot_accuracy_vs_step(model, device, test_loader, epsilon, alpha_fixed, step_list)

    # 绘图2：横轴为 alpha，step 固定
    # step_fixed = 4
    # alpha_list = [0.005, 0.01, 0.02, 0.03, 0.04]
    # plot_accuracy_vs_alpha(model, device, test_loader, epsilon, alpha_list, step_fixed)

    # 绘图3：同时比较多组 (alpha, step) 对某个 epsilon 的影响（仍然只有一个点，但多条线）
    pgd_param_sets = [
    #     (0.05, 1),
    #     (0.075, 2),
    #     (0.1, 2),
    #     (0.125, 4),
    #     (0.15, 6),
    #     (0.175, 7),
    # ]

    # # 单个 epsilon 观察不同组合下的准确率
    # accuracies = []
    # labels = []
    #
    # for alpha, steps in pgd_param_sets:
    #       acc = test_pgd(model, device, test_loader, epsilon, alpha, steps)
    #       accuracies.append(acc)
    #       labels.append(f"α={alpha}, step={steps}")
    #
    # # 条形图形式展示对比（每个组合一个柱子）
    # plt.figure(figsize=(10, 6))
    # plt.barh(labels, accuracies, color='orange')
    # plt.xlabel("Accuracy")
    # plt.title(f"PGD Accuracy for ε={epsilon} with Different α and Steps")
    # plt.xlim(0, 1.0)
    # plt.grid(axis='x', linestyle='--', alpha=0.6)
    # plt.tight_layout()
    plt.savefig("pgd_param_sets.png")
    plt.show()
    #

if __name__ == '__main__':
    main()