import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')


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


def main():
    # 参数设置
    epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    pgd_param_sets = [
         (0.05, 1),
         (0.075, 2),
         (0.1, 2),
         (0.125, 4),
         (0.15, 6),
         (0.175, 7),
    ]
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

    # 绘图初始化
    plt.figure(figsize=(10, 6))

    # 针对每组 (alpha, steps) 参数组合，画出不同 epsilon 下的 PGD 精度曲线
    for alpha, steps in pgd_param_sets:
        accuracies = []
        for eps in epsilons:
            acc = test_pgd(model, device, test_loader, eps, alpha, steps)
            accuracies.append(acc)
        plt.plot(epsilons, accuracies, marker='o', label=f"α={alpha}, step={steps}")

    # 绘制图像
    plt.title("PGD Accuracy vs Epsilon（多组 α 和 step）")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(0, 0.65 + 0.05, 0.05))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True)
    plt.legend()
    plt.savefig("pgd_accuracy_vs_epsilon_multi.png")
    plt.show()




if __name__ == '__main__':
    main()