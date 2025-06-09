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
                perturbed_data = pgd_attack(perturbed_data, epsilon, 0.5, data_grad)
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


def main():
    # 参数设置
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    pretrained_model = "./weights/lenet_mnist_model.pth"
    use_cuda = False

    # 初始化模型
    print("CUDA Available:", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model = Net().to(device)

    # 加载预训练模型
    try:
        model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    except FileNotFoundError:
        print(f"错误：找不到预训练模型文件 {pretrained_model}")
        print("请先训练模型或下载预训练模型")
        return

    model.eval()

    # 加载MNIST测试集
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=1, shuffle=True)

    # 运行FGSM攻击
    print("\nRunning FGSM Attack")
    accuracies_fgsm = []
    examples_fgsm = []

    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps, attack='fgsm')
        accuracies_fgsm.append(acc)
        examples_fgsm.append(ex)


    # 绘制结果
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, accuracies_fgsm, "*-", label="FGSM")

    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 0.35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('accuracy_vs_epsilon.png')  # 保存图像
    plt.show()

    # 展示对抗样本示例
    plt.figure(figsize=(10, 8))
    for i in range(len(epsilons[:3])):
        for j in range(len(examples_fgsm[0])):
            plt_idx = i * len(examples_fgsm[0]) + j + 1
            plt.subplot(len(epsilons[:3]), len(examples_fgsm[0]), plt_idx)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"ε={epsilons[i]}", fontsize=12)
            orig, adv, ex = examples_fgsm[i][j]
            plt.title(f"{orig}→{adv}", color=("green" if orig == adv else "red"))
            plt.imshow(ex, cmap='gray')
    plt.tight_layout()
    plt.savefig('adversarial_examples.png')  # 保存图像
    plt.show()


if __name__ == '__main__':
    main()