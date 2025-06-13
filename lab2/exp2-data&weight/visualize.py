import os
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import load_model
from argument import parser
from attack import PGD_attack
from model import Model

if __name__ == '__main__':
    img_folder = './img'
    os.makedirs(img_folder, exist_ok=True)
    args = parser()

    # 加载测试数据
    ts_dataset = torchvision.datasets.MNIST(args.data_root, train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())

    # 采样100张图像
    sample_index = [i for i in range(100)]
    image_test = []
    label_test = []
    for i in sample_index:
        image = ts_dataset[i][0]
        image_test.append(image)
        label = ts_dataset[i][1]
        label_test.append(label)
    sampled_test_data = [(image, label) for image, label in zip(image_test, label_test)]

    ts_loader = DataLoader(sampled_test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    for data, label in ts_loader:
        break

    # 定义要比较的模型
    types = ['Original', 'Standard_k10', 'Standard_k40']
    model_checkpoints = [
        './checkpoint/mnist_k10/checkpoint_500.pth',
        './checkpoint/mnist_k40/checkpoint_500.pth'
    ]
    adv_list = []
    pred_list = []

    max_epsilon = 0.8
    perturbation_type = 'linf'

    # 获取不同模型的分类结果
    with torch.no_grad():
        for checkpoint in model_checkpoints:
            model = Model(i_c=1, n_c=10)
            load_model(model, checkpoint)
            attack = PGD_attack(model, max_epsilon, args.alpha, args.k,
                                min_val=0, max_val=1, _type=perturbation_type)
            adv_data = attack.perturb(data, label, 'mean', False)

            output = model(adv_data, _eval=True)
            pred = torch.max(output, dim=1)[1]
            adv_list.append(adv_data.cpu().detach().numpy().squeeze())
            pred_list.append(pred.cpu().numpy())

    # 准备可视化数据
    data = data.cpu().numpy().squeeze()  # (N, 28, 28)
    data *= 255.0
    label = label.cpu().numpy()
    adv_list.insert(0, data)
    pred_list.insert(0, label)

    # 可视化
    out_num = args.batch_size
    fig, _axs = plt.subplots(nrows=len(adv_list), ncols=out_num, figsize=(15, 8))
    axs = _axs

    for j, _type in enumerate(types):
        axs[j, 0].set_ylabel(_type)
        for i in range(out_num):
            axs[j, i].set_title('%d' % pred_list[j][i])
            axs[j, i].imshow(adv_list[j][i], cmap='gray')
            axs[j, i].get_xaxis().set_ticks([])
            axs[j, i].get_yaxis().set_ticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'mnist_compare_%s.jpg' % perturbation_type))
    plt.show()

    model = LeNet().to(device)
    model.load_state_dict(torch.load('checkpoints/model_epoch20.pth'))  # 替换为你保存的模型
    model.eval()
