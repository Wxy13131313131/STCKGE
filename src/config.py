#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的配置文件
"""

import argparse
import os
import torch
from datetime import datetime


class Config:
    """简化的配置类"""

    def __init__(self):
        # 基础路径
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_root, 'data')
        self.checkpoint_dir = os.path.join(self.project_root, 'checkpoints')
        self.log_dir = os.path.join(self.project_root, 'logs')

        # 确保目录存在
        for directory in [self.checkpoint_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)

        # 模型参数
        self.embed_dim = 100
        self.margin = 6.0
        self.p_norm = 1
        self.init_size = 0.0001  # 嵌入初始化大小
        # BoxE特定参数
        self.box_softplus = True  # 是否使用softplus确保offset为正
        self.min_offset = 1e-3    # 最小偏移量，防止盒子退化为点
        self.max_offset = 2.0     # 最大偏移量，防止盒子过大
        self.learnable_temperature = True  # 是否让temperature可学习

        # IncDE特定参数
        self.distill_weight = 0.5
        self.temperature = 4.0
        self.importance_weight = 0.1

        # 优化器参数
        self.learning_rate = 0.001
        self.weight_decay = 1e-5

        # 训练参数
        self.epochs = 30
        self.batch_size = 512
        self.negative_sampling_ratio = 10
        self.grad_clip = 1.0

        # 自适应负采样
        self.adversarial_temperature = 1.0  # 0表示使用最困难的负样本

        # 正则化参数
        self.regularization_weight = 1e-6
        self.box_volume_regularization = 1e-7  # 盒子体积正则化

        # 验证和早停
        self.valid_freq = 1  # 每个epoch都验证
        self.patience = 5  # 早停patience

        # 日志参数
        self.log_freq = 100

        # 评估参数
        self.eval_batch_size = 512

        # 实验参数
        self.dataset = 'ENTITY'
        self.exp_name = None
        self.gpu = 0
        self.test_only = False

    def update_from_args(self, args):
        """从命令行参数更新配置"""
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

        # 生成实验名称
        if not self.exp_name:
            self.exp_name = f'stckge_{self.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        # 更新日志目录
        self.log_dir = os.path.join(self.log_dir, self.exp_name)
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_device(self):
        """获取设备配置"""
        if torch.cuda.is_available() and self.gpu >= 0:
            if self.gpu < torch.cuda.device_count():
                return f'cuda:{self.gpu}'
            else:
                print(f"Warning: GPU {self.gpu} not available, using GPU 0")
                return 'cuda:0'
        else:
            print("CUDA not available, using CPU")
            return 'cpu'

    def print_config(self):
        """打印配置信息"""
        print("=" * 30)
        print("配置信息")
        print("=" * 30)
        print(f"数据集: {self.dataset}")
        print(f"实验名称: {self.exp_name}")
        print(f"嵌入维度: {self.embed_dim}")
        print(f"学习率: {self.learning_rate}")
        print(f"批大小: {self.batch_size}")
        print(f"训练轮数: {self.epochs}")
        print(f"负采样比例: {self.negative_sampling_ratio}")
        print(f"蒸馏权重: {self.distill_weight}")
        print(f"验证频率: {self.valid_freq}")
        print(f"早停patience: {self.patience}")
        print("=" * 30)


def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='简化的IncDE训练')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='ENTITY',
                        help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help='数据目录路径')

    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=200,
                        help='嵌入维度')
    parser.add_argument('--margin', type=float, default=6.0,
                        help='损失函数的margin')

    # 训练参数
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='批大小')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--negative_sampling_ratio', type=int, default=10,
                        help='负采样比例')

    # IncDE参数
    parser.add_argument('--distill_weight', type=float, default=0.1,
                        help='蒸馏损失权重')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='蒸馏温度')

    # 设备参数
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备ID (-1表示使用CPU)')

    # 实验参数
    parser.add_argument('--exp_name', type=str, default=None,
                        help='实验名称')
    parser.add_argument('--test_only', action='store_true',
                        help='仅运行测试')

    return parser.parse_args()