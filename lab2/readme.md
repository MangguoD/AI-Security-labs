# 实验2：对抗网络训练及模型脆弱性分析

本项目包含以下脚本：
- `argument.py`：参数定义与解析
- `model.py`：LeNet 网络结构
- `utils.py`：工具函数（日志记录、模型保存/加载、性能评估）
- `train.py`：训练脚本，支持普通训练和 PGD 对抗训练
- `visualize.py`：绘制不同 ε 强度下模型鲁棒性曲线

## 环境依赖
- Python 3.7+
- PyTorch
- Torchvision
- NumPy
- Matplotlib

安装依赖（或者直接用我配好的虚拟环境![img.png](img.png)）：
```bash
pip install torch torchvision numpy matplotlib
```

## 目录结构
```
.
├── argument.py
├── model.py
├── utils.py
├── train.py
└── visualize.py
```

> **提示**：建议不要在项目目录名中使用 `&`、空格等特殊字符，以免在 shell 中引起解析错误。

## 使用方法

### 查看参数说明
```bash
python argument.py
```
- `--save_dir`：输出目录，默认 `./outputs`
- `--batch_size`：训练批大小，默认 64
- `--epochs`：训练轮数，默认 10
- `--lr`：学习率，默认 1e-3
- `--adv_train`：启用对抗训练（PGD）
- `--epsilon`：PGD 扰动最大范数 ε（仅在对抗训练时使用）
- `--alpha`：PGD 步长 α
- `--pgd_steps`：PGD 迭代步数
- `--log_interval`：日志打印频率
- `--save_model_interval`：模型保存频率

### 普通训练
```bash
python train.py --save_dir ./out_default --epochs 5 --batch_size 128
```

### 对抗训练
```bash
python train.py --save_dir ./out_adv --epochs 5 --batch_size 128 --adv_train --epsilon 0.3 --alpha 0.01 --pgd_steps 40
```
- 模型权重将保存在 `./out_adv/lenet_epoch{n}.pth`
- 日志文件：`./out_adv/train.log`

### 可视化模型鲁棒性
训练完成后，运行：
```bash
python visualize.py --save_dir ./out_adv --batch_size 128 --epochs 5
```
- 输出鲁棒性曲线：`./out_adv/robustness.png`

## 结果分析
- `train.log`：记录每个 batch 和每个 epoch 的训练损失与测试集准确率
- `robustness.png`：展示模型在不同对抗强度 ε 下的测试准确率

