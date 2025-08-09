#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的主程序
参考IncDE框架，保持持续学习的核心流程
"""

import os
import sys
import torch
import random
import numpy as np
import logging
import traceback

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 导入模块
from src.config import Config, get_args
from src.utils.data_loader import create_data_loader
from src.trainer.continual_trainer import create_trainer
# 导入训练器（需要确保正确的导入路径）





def setup_logging(log_dir):
    """设置日志"""
    log_file = os.path.join(log_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)

def log_config(config, logger):
    """将配置信息输出到日志文件"""
    logger.info("=" * 60)
    logger.info("配置信息")
    logger.info("=" * 60)

    # 获取配置对象的所有属性
    config_dict = vars(config)

    # 按类别组织配置项
    categories = {
        "基础配置": ["dataset", "data_dir", "test_only"],
        "模型配置": ["model_name", "embedding_dim", "num_layers", "dropout", "l2_reg"],
        "训练配置": ["learning_rate", "batch_size", "num_epochs", "patience", "min_delta"],
        "持续学习配置": ["method", "memory_size", "replay_ratio", "reg_lambda"],
        "路径配置": ["log_dir", "checkpoint_dir", "result_dir"],
        "硬件配置": ["device", "num_workers"],
        "验证配置": ["eval_batch_size", "max_valid_samples", "validate_every"]
    }

    # 按类别输出配置
    for category, keys in categories.items():
        logger.info(f"\n{category}:")
        for key in keys:
            if hasattr(config, key):
                value = getattr(config, key)
                logger.info(f"  {key}: {value}")

    # 输出其他未分类的配置项
    logged_keys = set()
    for keys in categories.values():
        logged_keys.update(keys)

    other_configs = []
    for key, value in config_dict.items():
        if key not in logged_keys and not key.startswith('_'):
            other_configs.append((key, value))

    if other_configs:
        logger.info(f"\n其他配置:")
        for key, value in other_configs:
            logger.info(f"  {key}: {value}")

    logger.info("=" * 60)

def set_seed(seed=55):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子设置为: {seed}")


def check_data_directory(data_dir, dataset):
    """检查数据目录"""
    processed_data_dir = os.path.join(data_dir, f'{dataset}_processed')

    if not os.path.exists(processed_data_dir):
        print(f"错误: 找不到预处理数据目录 {processed_data_dir}")
        print("请先运行数据预处理脚本 data_preprocess.py")
        return None

    # 检查必要文件
    required_files = ['global_mappings.pkl', 'statistics.json']
    for file in required_files:
        file_path = os.path.join(processed_data_dir, file)
        if not os.path.exists(file_path):
            print(f"错误: 找不到必要文件 {file_path}")
            return None

    # 检查快照目录
    snapshot_dirs = []
    for item in os.listdir(processed_data_dir):
        item_path = os.path.join(processed_data_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            snapshot_dirs.append(int(item))

    if not snapshot_dirs:
        print(f"错误: 在 {processed_data_dir} 中找不到快照目录")
        return None

    print(f"✅ 找到预处理数据，包含 {len(snapshot_dirs)} 个快照")
    return processed_data_dir


def run_training(config, processed_data_dir, device, logger):
    """运行训练"""
    logger.info("开始训练...")

    try:
        # 创建数据加载器
        logger.info("创建数据加载器...")
        data_loader = create_data_loader(
            processed_data_dir
            # use_memory_mapping=False,  # 简化：不使用内存映射
            # cache_snapshots=1,  # 简化：只缓存1个快照
            # enable_async_loading=False  # 简化：不使用异步加载
        )

        # 打印数据统计
        logger.info(f"数据集: {config.dataset}")
        logger.info(f"快照数量: {len(data_loader.snapshots)}")
        logger.info(f"总实体数: {data_loader.num_entities}")
        logger.info(f"总关系数: {data_loader.num_relations}")

        # 创建训练器
        logger.info("创建训练器...")
        trainer = create_trainer(config, data_loader, device)

        # 开始训练
        logger.info("开始持续学习训练...")
        results = trainer.train_all_snapshots()

        logger.info("✅ 训练完成!")
        return results

    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        traceback.print_exc()
        return None


def run_testing(config, processed_data_dir, device, logger):
    """运行测试"""
    logger.info("开始测试...")

    try:
        # 创建数据加载器
        data_loader = create_data_loader(processed_data_dir)

        # 创建训练器
        trainer = create_trainer(config, data_loader, device)

        # 查找最新的检查点
        checkpoint_files = []
        if os.path.exists(config.checkpoint_dir):
            for file in os.listdir(config.checkpoint_dir):
                if file.startswith('best_model_snapshot_') and file.endswith('.pth'):
                    checkpoint_files.append(file)

        if not checkpoint_files:
            logger.error("没有找到训练好的模型检查点!")
            return None

        # 使用最后一个快照的模型
        latest_checkpoint = sorted(checkpoint_files)[-1]
        checkpoint_path = os.path.join(config.checkpoint_dir, latest_checkpoint)

        logger.info(f"加载检查点: {checkpoint_path}")

        # 加载模型
        checkpoint = torch.load(checkpoint_path, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])

        # 在所有测试集上评估
        num_snapshots = len(data_loader.snapshots)
        test_results = trainer._evaluate_on_all_test_sets(num_snapshots - 1)

        logger.info("✅ 测试完成!")
        return test_results

    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        traceback.print_exc()
        return None


def main():
    """主函数"""
    print("=" * 80)
    print("简化的IncDE持续学习知识图谱嵌入")
    print("=" * 80)

    try:
        # 解析命令行参数
        args = get_args()

        # 创建配置
        config = Config()
        config.update_from_args(args)

        # 设置日志
        logger = setup_logging(config.log_dir)

        # 打印配置
        config.print_config()
        # 将配置信息输出到日志文件
        log_config(config, logger)
        # 设置随机种子
        set_seed(42)
        logger.info(f"随机种子设置为: 42")
        # 设置设备
        device = config.get_device()
        logger.info(f"使用设备: {device}")

        # 检查数据目录
        processed_data_dir = check_data_directory(config.data_dir, config.dataset)
        if processed_data_dir is None:
            return

        # 运行训练或测试
        if config.test_only:
            results = run_testing(config, processed_data_dir, device, logger)
        else:
            results = run_training(config, processed_data_dir, device, logger)

        if results:
            logger.info("程序执行成功!")
        else:
            logger.error("程序执行失败!")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        traceback.print_exc()


if __name__ == '__main__':
    main()