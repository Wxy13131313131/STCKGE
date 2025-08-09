# IncDE-BoxE 详细使用指南


## 安装依赖
```bash
pip install -r requirements.txt
```


## 数据准备

```bash
# 检查数据一致性
python data_preprocess.py --dataset YOUR_DATASET 

```

## 快速开始

### 方法 1: 使用脚本（推荐）
```bash
# 运行完整实验
./main.sh ENTITY 0 my_experiment

```

### 方法 2: 直接运行
```bash
# 基础训练
python main.py --dataset ENTITY --gpu 0

# 自定义参数
python main.py --dataset ENTITY --gpu 0 \
    --embed_dim 200 --learning_rate 0.001 \
    --batch_size 1024 --epochs 100
```

