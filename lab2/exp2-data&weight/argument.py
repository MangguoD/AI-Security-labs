# argument.py
import argparse

def parser():
    p = argparse.ArgumentParser(description="Adversarial Training with PGD on MNIST")
    # 路径参数
    p.add_argument('--save_dir', type=str, default='./outputs', help='实验输出目录')
    # 训练超参数
    p.add_argument('--batch_size', type=int, default=64, help='训练批大小')
    p.add_argument('--epochs', type=int, default=10, help='训练轮数')
    p.add_argument('--lr', type=float, default=1e-3, help='学习率')
    # PGD 攻击参数
    p.add_argument('--adv_train', action='store_true', help='是否启用对抗训练')
    p.add_argument('--epsilon', type=float, default=0.1, help='扰动最大范数 ε')
    p.add_argument('--alpha', type=float, default=0.05, help='PGD 步长 α')
    p.add_argument('--pgd_steps', type=int, default=40, help='PGD 迭代步数')
    # 日志与保存频率
    p.add_argument('--log_interval', type=int, default=10, help='日志打印频率')
    p.add_argument('--save_model_interval', type=int, default=1, help='模型保存轮数间隔')
    return p.parse_args()

if __name__ == '__main__':
    args = parser()
    print(args)