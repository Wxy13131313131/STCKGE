#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import json
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging
from ..utils.data_loader import create_data_loader
from ..models.stckge import ContinualModel
logger = logging.getLogger(__name__)


class ContinualTrainer:
    """持续学习训练器"""

    def __init__(self, config, data_loader, device='cuda'):
        self.config = config
        self.data_loader = data_loader
        self.device = device


        self.model = ContinualModel(
            num_entities=data_loader.num_entities,
            num_relations=data_loader.num_relations,
            embed_dim=config.embed_dim,
            margin=config.margin,
            distill_weight=getattr(config, 'distill_weight', 0.5),
            temperature=getattr(config, 'temperature', 4.0),
            device=device
        ).to(device)

        # 创建优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=getattr(config, 'weight_decay', 1e-5)
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=5, verbose=True
        )

        # 训练状态
        self.best_metrics = {}
        self.training_history = []

        # 时间统计
        self.total_train_time = 0.0
        self.total_test_time = 0.0
        self.snapshot_train_times = []
        self.snapshot_test_times = []

        # 确保目录存在
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def train_all_snapshots(self):
        """训练所有快照"""
        logger.info("🚀 开始持续学习训练...")
        overall_start_time = time.time()

        num_snapshots = len(self.data_loader.snapshots)
        all_test_results = {}

        for snapshot_id in range(num_snapshots):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"处理快照 {snapshot_id}/{num_snapshots - 1}")
            logger.info(f"{'=' * 60}")

            # 获取快照信息
            snapshot_info = self._get_snapshot_info(snapshot_id)
            self._log_snapshot_info(snapshot_id, snapshot_info)

            # 检查是否已存在训练好的模型
            model_path = os.path.join(self.config.checkpoint_dir, f'best_model_snapshot_{snapshot_id}.pth')
            if os.path.exists(model_path):
                logger.info(f"🔍 发现已存在的快照 {snapshot_id} 模型，直接加载进行评估...")

                # 确保模型扩展到合适大小
                if self._should_expand_model(snapshot_id):
                    self._expand_model(snapshot_id)

                # 加载已有模型
                self._load_best_model(snapshot_id)

                # 记录时间（跳过的训练时间为0）
                self.snapshot_train_times.append(0.0)

                # 直接评估
                test_start_time = time.time()
                test_results = self._evaluate_on_all_test_sets_with_weighted_avg(snapshot_id)
                test_time = time.time() - test_start_time
                self.snapshot_test_times.append(test_time)
                self.total_test_time += test_time

                all_test_results[f'after_snapshot_{snapshot_id}'] = test_results

                # 记录到历史
                self.training_history.append({
                    'snapshot_id': snapshot_id,
                    'training_results': {'skipped': True, 'reason': 'model_exists'},
                    'test_results': test_results,
                    'train_time': 0.0,
                    'test_time': test_time
                })

                logger.info(f"✅ 快照 {snapshot_id} 评估完成（跳过训练，测试耗时: {test_time:.2f}s）")
                continue

            # 检查是否需要扩展模型
            if self._should_expand_model(snapshot_id):
                self._expand_model(snapshot_id)

            # 设置蒸馏模型（从第二个快照开始）
            if snapshot_id > 0:
                self._setup_distillation_model()

            # 训练当前快照
            train_start_time = time.time()
            snapshot_results = self._train_single_snapshot(snapshot_id)
            train_time = time.time() - train_start_time
            self.snapshot_train_times.append(train_time)
            self.total_train_time += train_time

            # 保存最佳模型
            self._save_best_model(snapshot_id)

            # 评估当前模型在所有之前测试集上的性能（带加权平均）
            test_start_time = time.time()
            test_results = self._evaluate_on_all_test_sets_with_weighted_avg(snapshot_id)
            test_time = time.time() - test_start_time
            self.snapshot_test_times.append(test_time)
            self.total_test_time += test_time

            all_test_results[f'after_snapshot_{snapshot_id}'] = test_results

            # 记录训练历史
            self.training_history.append({
                'snapshot_id': snapshot_id,
                'training_results': snapshot_results,
                'test_results': test_results,
                'train_time': train_time,
                'test_time': test_time
            })

            logger.info(f"✅ 快照 {snapshot_id} 训练和评估完成（训练: {train_time:.2f}s, 测试: {test_time:.2f}s）")

        # 计算总时间
        total_time = time.time() - overall_start_time

        # 保存最终结果（包含时间统计）
        self._save_final_results(all_test_results, total_time)

        # 显示时间统计
        self._print_time_summary(total_time)

        return all_test_results

    def _get_snapshot_info(self, snapshot_id):
        """获取快照信息"""
        # 获取当前快照的增量信息
        incremental_info = self.data_loader.get_incremental_info(snapshot_id)

        # 获取三元组数量
        train_dataset = self.data_loader.get_snapshot_dataset(snapshot_id, mode='train')
        valid_dataset = self.data_loader.get_snapshot_dataset(snapshot_id, mode='valid')
        test_dataset = self.data_loader.get_snapshot_dataset(snapshot_id, mode='test')

        return {
            'num_train_triples': len(train_dataset),
            'num_valid_triples': len(valid_dataset),
            'num_test_triples': len(test_dataset),
            'num_new_entities': incremental_info['num_new_entities'],
            'num_new_relations': incremental_info['num_new_relations'],
            'new_entities': incremental_info['new_entities'],
            'old_entities': incremental_info['old_entities'],
            'new_relations': incremental_info['new_relations'],
            'old_relations': incremental_info['old_relations']
        }

    def _log_snapshot_info(self, snapshot_id, info):
        """记录快照信息"""
        logger.info(f"快照 {snapshot_id} 信息:")
        logger.info(f"  训练三元组: {info['num_train_triples']}")
        logger.info(f"  验证三元组: {info['num_valid_triples']}")
        logger.info(f"  测试三元组: {info['num_test_triples']}")
        logger.info(f"  新实体数: {info['num_new_entities']}")
        logger.info(f"  新关系数: {info['num_new_relations']}")
        logger.info(f"  当前总实体数: {self.model.num_entities}")
        logger.info(f"  当前总关系数: {self.model.num_relations}")

    def _should_expand_model(self, snapshot_id):
        """判断是否需要扩展模型"""
        if snapshot_id == 0:
            return False

        incremental_info = self.data_loader.get_incremental_info(snapshot_id)
        return (incremental_info['num_new_entities'] > 0 or
                incremental_info['num_new_relations'] > 0)

    def _expand_model(self, snapshot_id):
        """扩展模型以容纳新实体和关系"""
        incremental_info = self.data_loader.get_incremental_info(snapshot_id)

        # 计算新的实体和关系数量
        all_entities = set(incremental_info['new_entities']) | set(incremental_info['old_entities'])
        all_relations = set(incremental_info['new_relations']) | set(incremental_info['old_relations'])

        new_num_entities = max(max(all_entities) + 1 if all_entities else 0, self.model.num_entities)
        new_num_relations = max(max(all_relations) + 1 if all_relations else 0, self.model.num_relations)

        if new_num_entities > self.model.num_entities or new_num_relations > self.model.num_relations:
            logger.info(f"扩展模型: 实体 {self.model.num_entities} -> {new_num_entities}, "
                        f"关系 {self.model.num_relations} -> {new_num_relations}")

            # 扩展模型
            self.model.expand_model(new_num_entities, new_num_relations)

            # 重新创建优化器（参数数量改变了）
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=getattr(self.config, 'weight_decay', 1e-5)
            )

            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.8, patience=5, verbose=True
            )

    def _setup_distillation_model(self):
        """设置蒸馏模型"""
        previous_model = self.model.copy_model_for_distillation()
        self.model.set_previous_model(previous_model)
        logger.info("设置蒸馏模型完成")

    def _train_single_snapshot(self, snapshot_id):
        """训练单个快照（修正过拟合问题）"""
        # 获取数据加载器
        train_loader = self.data_loader.get_snapshot_dataloader(
            snapshot_id,
            mode='train',
            batch_size=self.config.batch_size,
            shuffle=True
        )

        valid_loader = self.data_loader.get_snapshot_dataloader(
            snapshot_id,
            mode='valid',
            batch_size=self.config.batch_size,
            shuffle=False
        )

        # 获取蒸馏信息
        incremental_info = self.data_loader.get_incremental_info(snapshot_id)
        old_entities = torch.LongTensor(incremental_info['old_entities']).to(self.device) if incremental_info[
            'old_entities'] else None
        old_relations = torch.LongTensor(incremental_info['old_relations']).to(self.device) if incremental_info[
            'old_relations'] else None

        best_valid_mrr = 0.0
        patience_counter = 0
        epoch_losses = []

        logger.info(f"开始训练快照 {snapshot_id}，共 {self.config.epochs} 个 epoch")
        logger.info(f"验证频率: 每 {getattr(self.config, 'valid_freq', 3)} 个epoch验证一次")
        logger.info("=" * 80)
        logger.info(
            f"{'Epoch':<6} {'Loss':<10} {'LR':<12} {'MRR':<8} {'Hit@1':<8} {'Hit@3':<8} {'Hit@10':<8} {'Best':<8}")
        logger.info("=" * 80)

        # 创建epoch进度条
        epoch_pbar = tqdm(range(self.config.epochs),
                          desc=f"训练快照 {snapshot_id}",
                          unit="epoch",
                          position=0,
                          leave=True)

        for epoch in epoch_pbar:
            # 训练一个epoch
            epoch_loss = self._train_epoch(
                train_loader, old_entities, old_relations, epoch
            )
            epoch_losses.append(epoch_loss)

            # 按照设定频率进行验证
            valid_freq = getattr(self.config, 'valid_freq', 3)
            should_validate = (epoch + 1) % valid_freq == 0 or (epoch + 1) == self.config.epochs

            if should_validate:
                # 验证
                valid_metrics = self._validate_with_filtering_parallel(valid_loader,snapshot_id)
                current_mrr = valid_metrics['MRR']
                current_lr = self.optimizer.param_groups[0]['lr']

                # 更新学习率
                self.scheduler.step(current_mrr)

                # 检查是否是最佳模型
                is_best = False
                if current_mrr > best_valid_mrr:
                    best_valid_mrr = current_mrr
                    patience_counter = 0
                    self._save_temp_best_model(snapshot_id)
                    is_best = True
                else:
                    patience_counter += 1

                # 打印详细信息
                status = "✓" if is_best else " "
                logger.info(f"{epoch + 1:<6} {epoch_loss:<10.4f} {current_lr:<12.2e} "
                            f"{valid_metrics['MRR']:<8.4f} {valid_metrics['Hit@1']:<8.4f} "
                            f"{valid_metrics['Hit@3']:<8.4f} {valid_metrics['Hit@10']:<8.4f} "
                            f"{best_valid_mrr:<8.4f} {status}")

                # 更新进度条显示
                epoch_pbar.set_postfix({
                    'Loss': f"{epoch_loss:.4f}",
                    'MRR': f"{current_mrr:.4f}",
                    'Best': f"{best_valid_mrr:.4f}",
                    'Pat': f"{patience_counter}/{getattr(self.config, 'patience', 15)}"
                })

                # 早停检查
                if patience_counter >= getattr(self.config, 'patience', 15):
                    logger.info(f"早停在 epoch {epoch + 1}，patience达到 {getattr(self.config, 'patience', 15)}")
                    epoch_pbar.close()
                    break
            else:
                # 不验证的epoch，只显示训练损失
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"{epoch + 1:<6} {epoch_loss:<10.4f} {current_lr:<12.2e} "
                            f"{'---':<8} {'---':<8} {'---':<8} {'---':<8} "
                            f"{best_valid_mrr:<8.4f} ")

                # 更新进度条显示
                epoch_pbar.set_postfix({
                    'Loss': f"{epoch_loss:.4f}",
                    'MRR': f"未验证",
                    'Best': f"{best_valid_mrr:.4f}",
                    'Pat': f"{patience_counter}/{getattr(self.config, 'patience', 15)}"
                })
        else:
            # 正常完成所有epoch
            epoch_pbar.close()

        logger.info("=" * 80)

        # 加载最佳模型
        self._load_temp_best_model(snapshot_id)

        # 最终验证
        final_valid_metrics = self._validate_with_filtering_parallel(valid_loader,snapshot_id)
        logger.info(f"✅ 最终验证结果: MRR={final_valid_metrics['MRR']:.4f}, "
                    f"Hit@1={final_valid_metrics['Hit@1']:.4f}, "
                    f"Hit@10={final_valid_metrics['Hit@10']:.4f}")

        return {
            'losses': epoch_losses,
            'best_valid_mrr': best_valid_mrr,
            'final_valid_metrics': final_valid_metrics,
            'total_epochs': len(epoch_losses)
        }

    def _train_epoch(self, train_loader, old_entities, old_relations, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (pos_triples, neg_triples) in enumerate(train_loader):
            # 数据移到设备
            pos_triples = pos_triples.to(self.device)
            if neg_triples.dim() == 3:
                neg_triples = neg_triples.view(-1, 3)
            neg_triples = neg_triples.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()

            # 准备蒸馏样本
            sample_triples = None
            if self.model.previous_model is not None and len(pos_triples) > 0:
                sample_size = min(len(pos_triples), 16)
                sample_indices = torch.randperm(len(pos_triples))[:sample_size]
                sample_triples = pos_triples[sample_indices]

            # 计算损失
            loss_dict = self.model.compute_total_loss(
                pos_triples, neg_triples, old_entities, old_relations, sample_triples
            )

            total_loss_batch = loss_dict['total_loss']

            # 反向传播
            total_loss_batch.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.config, 'grad_clip', 1.0))

            # 优化器步骤
            self.optimizer.step()

            # 更新重要性权重
            entities_in_batch = torch.cat([pos_triples[:, 0], pos_triples[:, 2]])
            relations_in_batch = pos_triples[:, 1]
            self.model.current_model.update_importance_weights(entities_in_batch, relations_in_batch)

            total_loss += total_loss_batch.item()

        return total_loss / num_batches

    def _validate_with_filtering_parallel(self, valid_loader, snapshot_id):
        """使用负过滤的严格验证（并行批处理版本，适用于大规模数据集）"""
        self.model.eval()

        # 收集所有已知的正样本三元组（用于过滤）
        # 使用字典加速查找
        positive_triples_dict = defaultdict(set)
        for i in range(snapshot_id + 1):
            if i < len(self.data_loader.snapshots):
                snapshot_data = self.data_loader.snapshots[i]
                for dataset_name in ['train_triples', 'valid_triples', 'test_triples']:
                    for triple in snapshot_data.get(dataset_name, []):
                        if len(triple) >= 3:
                            h, r, t = triple[0], triple[1], triple[2]
                            positive_triples_dict[(h, r)].add(t)

        total_reciprocal_rank = 0.0
        total_samples = 0
        hit_counts = {1: 0, 3: 0, 10: 0}
        total_mr = 0.0

        # 设置评估批大小
        eval_batch_size = getattr(self.config, 'eval_batch_size', 32)

        with torch.no_grad():
            for pos_triples, _ in valid_loader:
                pos_triples = pos_triples.to(self.device)
                num_triples = pos_triples.shape[0]

                # 批量处理所有三元组
                for batch_start in range(0, num_triples, eval_batch_size):
                    batch_end = min(batch_start + eval_batch_size, num_triples)
                    batch_triples = pos_triples[batch_start:batch_end]

                    # 批量验证索引有效性
                    batch_heads = batch_triples[:, 0]
                    batch_relations = batch_triples[:, 1]
                    batch_tails = batch_triples[:, 2]

                    valid_mask = (
                            (batch_heads < self.model.num_entities) &
                            (batch_relations < self.model.num_relations) &
                            (batch_tails < self.model.num_entities)
                    )

                    if not valid_mask.any():
                        continue

                    # 提取有效三元组
                    valid_triples = batch_triples[valid_mask]

                    # 使用IncDE风格的排名计算
                    batch_ranks = self._compute_batch_ranks(
                        valid_triples, positive_triples_dict
                    )

                    # 更新统计信息
                    for rank in batch_ranks:
                        total_reciprocal_rank += 1.0 / rank
                        total_mr += rank
                        total_samples += 1

                        for k in [1, 3, 10]:
                            if rank <= k:
                                hit_counts[k] += 1

                # 限制验证样本数量
                if total_samples >= getattr(self.config, 'max_valid_samples', 1000):
                    break

        # 计算最终指标
        if total_samples == 0:
            return {'MRR': 0.0, 'Hit@1': 0.0, 'Hit@3': 0.0, 'Hit@10': 0.0, 'MR': 0.0}

        metrics = {
            'MRR': total_reciprocal_rank / total_samples,
            'MR': total_mr / total_samples,
            'Hit@1': hit_counts[1] / total_samples,
            'Hit@3': hit_counts[3] / total_samples,
            'Hit@10': hit_counts[10] / total_samples,
            'num_evaluated': total_samples
        }

        return metrics

    def _compute_batch_ranks(self, batch_triples, positive_triples_dict):
        """
        排名计算方法
        """
        batch_size = batch_triples.shape[0]
        batch_heads = batch_triples[:, 0]
        batch_relations = batch_triples[:, 1]
        batch_tails = batch_triples[:, 2]

        # 创建所有可能的尾实体候选
        all_entities = torch.arange(self.model.num_entities, device=self.device)

        # 初始化预测得分矩阵
        pred = torch.zeros(batch_size, self.model.num_entities, device=self.device)

        # 为批次中的每个三元组计算对所有尾实体的得分
        for i in range(batch_size):
            h = batch_heads[i]
            r = batch_relations[i]

            # 扩展头实体和关系以匹配所有候选尾实体
            heads_expanded = torch.full((self.model.num_entities,), h, device=self.device)
            relations_expanded = torch.full((self.model.num_entities,), r, device=self.device)

            # 计算当前头实体和关系对所有尾实体的得分
            scores = self.model.predict(heads_expanded, relations_expanded, all_entities)
            pred[i] = scores

        # 创建label矩阵，标记所有已知的正确tail
        label = torch.zeros(batch_size, self.model.num_entities, device=self.device, dtype=torch.bool)

        for i in range(batch_size):
            h = batch_heads[i].item()
            r = batch_relations[i].item()
            # 标记所有已知的正确tail
            if (h, r) in positive_triples_dict:
                for t in positive_triples_dict[(h, r)]:
                    if t < self.model.num_entities:
                        label[i, t] = True

        # 按照IncDE的过滤方式进行过滤
        batch_size_range = torch.arange(batch_size, device=self.device)
        target_pred = pred[batch_size_range, batch_tails]  # 取出当前三元组中tail的得分

        # 将所有其他已知正确tail的得分设为负无穷
        pred = torch.where(label, -torch.ones_like(pred) * 10000000, pred)

        # 恢复当前三元组中tail的得分
        pred[batch_size_range, batch_tails] = target_pred

        # 排名计算
        ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
            batch_size_range, batch_tails]

        return ranks.cpu().numpy().tolist()

    def _create_label_matrix_for_filtering(self, batch_heads, batch_relations, positive_triples_dict):
        """
        创建用于过滤的label矩阵
        """
        batch_size = len(batch_heads)
        label = torch.zeros(batch_size, self.model.num_entities, device=self.device, dtype=torch.bool)

        for i in range(batch_size):
            h = batch_heads[i].item()
            r = batch_relations[i].item()

            # 标记所有已知的正确tail
            if (h, r) in positive_triples_dict:
                for t in positive_triples_dict[(h, r)]:
                    if t < self.model.num_entities:
                        label[i, t] = True

        return label

    def _validate_with_filtering(self, valid_loader, snapshot_id):
        """使用负过滤的严格验证（批量处理优化版本）"""
        self.model.eval()

        # 收集所有已知的正样本三元组（用于过滤）
        all_positive_triples = set()
        for i in range(snapshot_id + 1):
            if i < len(self.data_loader.snapshots):
                snapshot_data = self.data_loader.snapshots[i]
                for triple in snapshot_data.get('train_triples', []):
                    if len(triple) >= 3:
                        all_positive_triples.add((triple[0], triple[1], triple[2]))
                for triple in snapshot_data.get('valid_triples', []):
                    if len(triple) >= 3:
                        all_positive_triples.add((triple[0], triple[1], triple[2]))
                for triple in snapshot_data.get('test_triples', []):
                    if len(triple) >= 3:
                        all_positive_triples.add((triple[0], triple[1], triple[2]))

        total_reciprocal_rank = 0.0
        total_samples = 0
        hit_counts = {1: 0, 3: 0, 10: 0}

        # 设置评估批大小
        eval_batch_size = getattr(self.config, 'eval_batch_size', 16)

        with torch.no_grad():
            for pos_triples, _ in valid_loader:
                pos_triples = pos_triples.to(self.device)

                # 分批处理当前批次的三元组
                num_triples = pos_triples.shape[0]

                for batch_start in range(0, num_triples, eval_batch_size):
                    batch_end = min(batch_start + eval_batch_size, num_triples)
                    batch_triples = pos_triples[batch_start:batch_end]
                    batch_size = batch_triples.shape[0]

                    # 提取批次中的头实体、关系和尾实体
                    batch_heads = batch_triples[:, 0]
                    batch_relations = batch_triples[:, 1]
                    batch_tails = batch_triples[:, 2]

                    # 检查索引是否有效
                    valid_mask = (
                            (batch_heads < self.model.num_entities) &
                            (batch_relations < self.model.num_relations) &
                            (batch_tails < self.model.num_entities)
                    )

                    if not valid_mask.any():
                        continue

                    # 只处理有效的三元组
                    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                    valid_heads = batch_heads[valid_indices]
                    valid_relations = batch_relations[valid_indices]
                    valid_tails = batch_tails[valid_indices]
                    valid_batch_size = len(valid_indices)

                    # 为批次中的每个三元组计算排名
                    for idx in range(valid_batch_size):
                        h = valid_heads[idx].item()
                        r = valid_relations[idx].item()
                        t = valid_tails[idx].item()

                        # 批量计算所有候选实体的得分
                        # 方法1：对所有实体进行评分（内存友好但计算密集）
                        if self.model.num_entities <= 10000:  # 实体数量较少时
                            all_entities = torch.arange(self.model.num_entities, device=self.device)
                            heads = torch.full((self.model.num_entities,), h, device=self.device)
                            relations = torch.full((self.model.num_entities,), r, device=self.device)

                            # 批量计算得分
                            scores = self.model.predict(heads, relations, all_entities)

                        else:  # 实体数量较多时，分块计算
                            chunk_size = 1000
                            scores = torch.zeros(self.model.num_entities, device=self.device)

                            for chunk_start in range(0, self.model.num_entities, chunk_size):
                                chunk_end = min(chunk_start + chunk_size, self.model.num_entities)
                                chunk_entities = torch.arange(chunk_start, chunk_end, device=self.device)
                                chunk_heads = torch.full((len(chunk_entities),), h, device=self.device)
                                chunk_relations = torch.full((len(chunk_entities),), r, device=self.device)

                                chunk_scores = self.model.predict(chunk_heads, chunk_relations, chunk_entities)
                                scores[chunk_start:chunk_end] = chunk_scores

                        # 过滤掉其他已知正样本
                        # 使用向量化操作提高效率
                        for entity_id in range(self.model.num_entities):
                            if entity_id != t and (h, r, entity_id) in all_positive_triples:
                                scores[entity_id] = float('-inf')

                        # 获取目标实体的得分
                        target_score = scores[t]

                        # 计算排名
                        rank = torch.sum(scores > target_score).item() + 1

                        # 处理得分相同的情况
                        num_ties = torch.sum(scores == target_score).item()
                        if num_ties > 1:
                            rank = rank + (num_ties - 1) / 2.0

                        # 更新统计信息
                        total_reciprocal_rank += 1.0 / rank
                        total_samples += 1

                        for k in [1, 3, 10]:
                            if rank <= k:
                                hit_counts[k] += 1

                    # 定期打印进度（可选）
                    if total_samples % 100 == 0 and total_samples > 0:
                        current_mrr = total_reciprocal_rank / total_samples
                        logger.debug(f"验证进度: {total_samples} 样本, 当前MRR: {current_mrr:.4f}")

                # 限制验证样本数量
                if total_samples >= getattr(self.config, 'max_valid_samples', 1000):
                    break

        # 计算最终指标
        if total_samples == 0:
            return {'MRR': 0.0, 'Hit@1': 0.0, 'Hit@3': 0.0, 'Hit@10': 0.0}

        metrics = {
            'MRR': total_reciprocal_rank / total_samples,
            'Hit@1': hit_counts[1] / total_samples,
            'Hit@3': hit_counts[3] / total_samples,
            'Hit@10': hit_counts[10] / total_samples,
            'num_evaluated': total_samples
        }

        return metrics

    def _evaluate_on_all_test_sets_with_weighted_avg(self, current_snapshot_id):
        """在所有之前的测试集上评估模型，并计算加权平均"""
        logger.info(f"\n📊 在所有测试集上评估模型（快照 0 到 {current_snapshot_id}）")
        logger.info("=" * 60)

        all_test_results = {}
        total_mrr = 0.0
        total_hit1 = 0.0
        total_hit3 = 0.0
        total_hit10 = 0.0
        total_samples = 0

        # 用于存储每个快照的样本数量（用于加权）
        snapshot_weights = []
        snapshot_metrics = []

        for test_snapshot_id in range(current_snapshot_id + 1):
            test_loader = self.data_loader.get_snapshot_dataloader(
                test_snapshot_id,
                mode='test',
                batch_size=self.config.batch_size,
                shuffle=False
            )

            # 使用严格的过滤评估
            test_metrics = self._validate_with_filtering_parallel(test_loader, current_snapshot_id)
            test_dataset = self.data_loader.get_snapshot_dataset(test_snapshot_id, mode='test')
            num_test_samples = len(test_dataset)

            all_test_results[f'snapshot_{test_snapshot_id}'] = test_metrics
            all_test_results[f'snapshot_{test_snapshot_id}_samples'] = num_test_samples

            # 累积加权统计
            weight = num_test_samples
            snapshot_weights.append(weight)
            snapshot_metrics.append(test_metrics)

            total_mrr += test_metrics['MRR'] * weight
            total_hit1 += test_metrics['Hit@1'] * weight
            total_hit3 += test_metrics['Hit@3'] * weight
            total_hit10 += test_metrics['Hit@10'] * weight
            total_samples += weight

            logger.info(f"快照 {test_snapshot_id:2d} ({num_test_samples:4d} 样本): "
                        f"MRR={test_metrics['MRR']:.4f} "
                        f"Hit@1={test_metrics['Hit@1']:.4f} "
                        f"Hit@3={test_metrics['Hit@3']:.4f} "
                        f"Hit@10={test_metrics['Hit@10']:.4f}")

        # 计算加权平均
        if total_samples > 0:
            weighted_avg_metrics = {
                'weighted_avg_MRR': total_mrr / total_samples,
                'weighted_avg_Hit@1': total_hit1 / total_samples,
                'weighted_avg_Hit@3': total_hit3 / total_samples,
                'weighted_avg_Hit@10': total_hit10 / total_samples
            }
        else:
            weighted_avg_metrics = {
                'weighted_avg_MRR': 0.0,
                'weighted_avg_Hit@1': 0.0,
                'weighted_avg_Hit@3': 0.0,
                'weighted_avg_Hit@10': 0.0
            }

        # 计算简单平均（不加权）
        if len(snapshot_metrics) > 0:
            simple_avg_metrics = {
                'simple_avg_MRR': sum(m['MRR'] for m in snapshot_metrics) / len(snapshot_metrics),
                'simple_avg_Hit@1': sum(m['Hit@1'] for m in snapshot_metrics) / len(snapshot_metrics),
                'simple_avg_Hit@3': sum(m['Hit@3'] for m in snapshot_metrics) / len(snapshot_metrics),
                'simple_avg_Hit@10': sum(m['Hit@10'] for m in snapshot_metrics) / len(snapshot_metrics)
            }
        else:
            simple_avg_metrics = {
                'simple_avg_MRR': 0.0,
                'simple_avg_Hit@1': 0.0,
                'simple_avg_Hit@3': 0.0,
                'simple_avg_Hit@10': 0.0
            }

        all_test_results.update(weighted_avg_metrics)
        all_test_results.update(simple_avg_metrics)
        all_test_results['total_test_samples'] = total_samples

        logger.info("-" * 60)
        logger.info(f"📈 加权平均性能 (基于样本数量权重，使用负过滤):")
        logger.info(f"   MRR: {weighted_avg_metrics['weighted_avg_MRR']:.4f}")
        logger.info(f"   Hit@1: {weighted_avg_metrics['weighted_avg_Hit@1']:.4f}")
        logger.info(f"   Hit@3: {weighted_avg_metrics['weighted_avg_Hit@3']:.4f}")
        logger.info(f"   Hit@10: {weighted_avg_metrics['weighted_avg_Hit@10']:.4f}")

        logger.info(f"📊 简单平均性能 (等权重):")
        logger.info(f"   MRR: {simple_avg_metrics['simple_avg_MRR']:.4f}")
        logger.info(f"   Hit@1: {simple_avg_metrics['simple_avg_Hit@1']:.4f}")
        logger.info(f"   Hit@3: {simple_avg_metrics['simple_avg_Hit@3']:.4f}")
        logger.info(f"   Hit@10: {simple_avg_metrics['simple_avg_Hit@10']:.4f}")

        logger.info(f"📋 总测试样本: {total_samples}")
        logger.info("=" * 60)

        return all_test_results

    def diagnose_data_leakage(self):
        """诊断数据是否存在泄露问题"""
        logger.info("\n" + "=" * 80)
        logger.info("🔍 数据泄露诊断报告")
        logger.info("=" * 80)

        issues_found = []
        warnings_found = []

        for snapshot_id in range(len(self.data_loader.snapshots)):
            snapshot_data = self.data_loader.snapshots[snapshot_id]

            # 检查1: 验证/测试数据是否为空或来自训练数据
            train_triples = set(tuple(t) for t in snapshot_data.get('train_triples', []))
            valid_triples = set(tuple(t) for t in snapshot_data.get('valid_triples', []))
            test_triples = set(tuple(t) for t in snapshot_data.get('test_triples', []))

            # 验证数据检查
            if not valid_triples:
                warnings_found.append(f"快照 {snapshot_id}: 没有验证数据")
            else:
                overlap = train_triples & valid_triples
                if overlap:
                    issues_found.append(f"快照 {snapshot_id}: 验证数据与训练数据重叠 ({len(overlap)} 个三元组)")

            # 测试数据检查
            if not test_triples:
                warnings_found.append(f"快照 {snapshot_id}: 没有测试数据")
            else:
                overlap = train_triples & test_triples
                if overlap:
                    issues_found.append(f"快照 {snapshot_id}: 测试数据与训练数据重叠 ({len(overlap)} 个三元组)")

                # 验证和测试数据重叠检查
                if valid_triples:
                    val_test_overlap = valid_triples & test_triples
                    if val_test_overlap:
                        issues_found.append(
                            f"快照 {snapshot_id}: 验证数据与测试数据重叠 ({len(val_test_overlap)} 个三元组)")

            # 检查2: 数据集大小合理性
            train_size = len(train_triples)
            valid_size = len(valid_triples)
            test_size = len(test_triples)

            if valid_size > 0 and valid_size < train_size * 0.05:
                warnings_found.append(f"快照 {snapshot_id}: 验证集过小 ({valid_size} vs {train_size} 训练样本)")

            if test_size > 0 and test_size < train_size * 0.05:
                warnings_found.append(f"快照 {snapshot_id}: 测试集过小 ({test_size} vs {train_size} 训练样本)")

            # 检查3: 实体/关系范围
            entities_in_snapshot = set()
            relations_in_snapshot = set()

            for triple_set in [train_triples, valid_triples, test_triples]:
                for h, r, t in triple_set:
                    entities_in_snapshot.add(h)
                    entities_in_snapshot.add(t)
                    relations_in_snapshot.add(r)

            max_entity = max(entities_in_snapshot) if entities_in_snapshot else 0
            max_relation = max(relations_in_snapshot) if relations_in_snapshot else 0

            if max_entity >= self.data_loader.num_entities:
                issues_found.append(
                    f"快照 {snapshot_id}: 实体ID超出范围 ({max_entity} >= {self.data_loader.num_entities})")

            if max_relation >= self.data_loader.num_relations:
                issues_found.append(
                    f"快照 {snapshot_id}: 关系ID超出范围 ({max_relation} >= {self.data_loader.num_relations})")

        # 打印诊断结果
        if issues_found:
            logger.error("❌ 发现严重问题:")
            for issue in issues_found:
                logger.error(f"   • {issue}")
        else:
            logger.info("✅ 未发现严重的数据泄露问题")

        if warnings_found:
            logger.warning("⚠️  发现潜在问题:")
            for warning in warnings_found:
                logger.warning(f"   • {warning}")

        # 建议
        logger.info("\n📋 建议:")
        if not valid_triples or not test_triples:
            logger.info("   • 使用真实的验证/测试数据，避免从训练数据中划分")

        if issues_found:
            logger.info("   • 重新预处理数据，确保数据集之间没有重叠")

        logger.info("=" * 80)

        return len(issues_found) == 0

    def _load_best_model(self, snapshot_id):
        """加载最佳模型"""
        model_path = os.path.join(self.config.checkpoint_dir, f'best_model_snapshot_{snapshot_id}.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"已加载快照 {snapshot_id} 的最佳模型: {model_path}")
            return True
        else:
            logger.warning(f"找不到快照 {snapshot_id} 的最佳模型: {model_path}")
            return False

    def _save_temp_best_model(self, snapshot_id):
        """保存临时最佳模型"""
        temp_path = os.path.join(self.config.checkpoint_dir, f'temp_best_snapshot_{snapshot_id}.pth')
        torch.save(self.model.state_dict(), temp_path)

    def _load_temp_best_model(self, snapshot_id):
        """加载临时最佳模型"""
        temp_path = os.path.join(self.config.checkpoint_dir, f'temp_best_snapshot_{snapshot_id}.pth')
        if os.path.exists(temp_path):
            self.model.load_state_dict(torch.load(temp_path, map_location=self.device))

    def _save_best_model(self, snapshot_id):
        """保存最佳模型"""
        model_path = os.path.join(self.config.checkpoint_dir, f'best_model_snapshot_{snapshot_id}.pth')

        # 保存完整的检查点
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'snapshot_id': snapshot_id,
            'num_entities': self.model.num_entities,
            'num_relations': self.model.num_relations
        }

        torch.save(checkpoint, model_path)
        logger.info(f"最佳模型已保存: {model_path}")

        # 删除临时文件
        temp_path = os.path.join(self.config.checkpoint_dir, f'temp_best_snapshot_{snapshot_id}.pth')
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def _print_time_summary(self, total_time):
        """打印时间统计汇总"""
        logger.info("\n" + "=" * 80)
        logger.info("⏱️  时间统计汇总")
        logger.info("=" * 80)

        # 格式化时间显示函数
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.2f}s"
            elif seconds < 3600:
                mins = int(seconds // 60)
                secs = seconds % 60
                return f"{mins}m {secs:.1f}s"
            else:
                hours = int(seconds // 3600)
                mins = int((seconds % 3600) // 60)
                secs = seconds % 60
                return f"{hours}h {mins}m {secs:.1f}s"

        # 总体时间统计
        logger.info(f"🚀 总执行时间: {format_time(total_time)}")
        logger.info(f"🎓 总训练时间: {format_time(self.total_train_time)}")
        logger.info(f"🧪 总测试时间: {format_time(self.total_test_time)}")

        if total_time > 0:
            train_ratio = (self.total_train_time / total_time) * 100
            test_ratio = (self.total_test_time / total_time) * 100
            other_ratio = 100 - train_ratio - test_ratio

            logger.info(f"📊 时间分布: 训练 {train_ratio:.1f}% | 测试 {test_ratio:.1f}% | 其他 {other_ratio:.1f}%")

        # 各快照详细时间
        if self.snapshot_train_times or self.snapshot_test_times:
            logger.info("\n📋 各快照详细时间:")
            logger.info("-" * 60)
            logger.info(f"{'快照':<6} {'训练时间':<15} {'测试时间':<15} {'总时间':<15}")
            logger.info("-" * 60)

            for i in range(max(len(self.snapshot_train_times), len(self.snapshot_test_times))):
                train_time = self.snapshot_train_times[i] if i < len(self.snapshot_train_times) else 0.0
                test_time = self.snapshot_test_times[i] if i < len(self.snapshot_test_times) else 0.0
                total_snap_time = train_time + test_time

                train_str = "跳过" if train_time == 0.0 else format_time(train_time)

                logger.info(f"{i:<6} {train_str:<15} {format_time(test_time):<15} {format_time(total_snap_time):<15}")

            logger.info("-" * 60)

            # 平均时间统计
            trained_snapshots = [t for t in self.snapshot_train_times if t > 0]
            if trained_snapshots:
                avg_train_time = sum(trained_snapshots) / len(trained_snapshots)
                logger.info(f"📈 平均训练时间 (已训练快照): {format_time(avg_train_time)}")

            if self.snapshot_test_times:
                avg_test_time = sum(self.snapshot_test_times) / len(self.snapshot_test_times)
                logger.info(f"🧪 平均测试时间: {format_time(avg_test_time)}")

        logger.info("=" * 80)

    def _print_final_summary(self, all_test_results):
        """打印最终性能总结（优化的表格格式）"""
        logger.info("\n" + "=" * 90)
        logger.info("🎯 最终性能总结（使用负过滤评估）")
        logger.info("=" * 90)

        # 获取最后一次评估的结果
        last_evaluation = all_test_results[list(all_test_results.keys())[-1]]

        # 收集快照数据
        snapshot_data = []
        for snapshot_key, metrics in last_evaluation.items():
            if snapshot_key.startswith('snapshot_') and not snapshot_key.endswith('_samples'):
                snapshot_id = int(snapshot_key.split('_')[1])
                samples_key = f'snapshot_{snapshot_id}_samples'
                num_samples = last_evaluation.get(samples_key, 0)

                # 获取训练信息
                training_info = "已训练"
                train_time = 0.0
                test_time = 0.0

                for history in self.training_history:
                    if history['snapshot_id'] == snapshot_id:
                        if history['training_results'].get('skipped', False):
                            training_info = "跳过"
                        train_time = history.get('train_time', 0.0)
                        test_time = history.get('test_time', 0.0)
                        break

                snapshot_data.append({
                    'id': snapshot_id,
                    'samples': num_samples,
                    'mrr': metrics['MRR'],
                    'hit1': metrics['Hit@1'],
                    'hit3': metrics['Hit@3'],
                    'hit10': metrics['Hit@10'],
                    'status': training_info,
                    'train_time': train_time,
                    'test_time': test_time
                })

        # 按快照ID排序
        snapshot_data.sort(key=lambda x: x['id'])

        # 打印详细表格
        logger.info("📊 各快照性能详情:")
        logger.info("=" * 90)
        header = f"{'快照':<4} {'状态':<6} {'样本数':<7} {'MRR':<8} {'Hit@1':<8} {'Hit@3':<8} {'Hit@10':<8} {'训练时间':<10} {'测试时间':<10}"
        logger.info(header)
        logger.info("=" * 90)

        def format_time_short(seconds):
            if seconds == 0.0:
                return "跳过"
            elif seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{int(seconds // 60)}m{int(seconds % 60)}s"
            else:
                return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60)}m"

        for data in snapshot_data:
            train_time_str = format_time_short(data['train_time'])
            test_time_str = format_time_short(data['test_time'])

            row = (f"{data['id']:<4} {data['status']:<6} {data['samples']:<7} "
                   f"{data['mrr']:<8.4f} {data['hit1']:<8.4f} {data['hit3']:<8.4f} "
                   f"{data['hit10']:<8.4f} {train_time_str:<10} {test_time_str:<10}")
            logger.info(row)

        logger.info("=" * 90)
        # 新增：5个快照加权均值汇总表
        logger.info("🏆 5个快照加权均值汇总表:")
        logger.info("=" * 70)
        logger.info(f"{'训练轮次':<10} {'加权MRR':<12} {'加权Hit@1':<12} {'加权Hit@3':<12} {'加权Hit@10':<12}")
        logger.info("=" * 70)

        # 遍历所有训练轮次的结果
        for round_key in sorted(all_test_results.keys()):
            if round_key.startswith('after_snapshot_'):
                snapshot_num = int(round_key.split('_')[-1])
                round_results = all_test_results[round_key]

                # 获取该轮次的加权平均结果
                weighted_mrr = round_results.get('weighted_avg_MRR', 0.0)
                weighted_hit1 = round_results.get('weighted_avg_Hit@1', 0.0)
                weighted_hit3 = round_results.get('weighted_avg_Hit@3', 0.0)
                weighted_hit10 = round_results.get('weighted_avg_Hit@10', 0.0)

                logger.info(f"快照 {snapshot_num:<5} {weighted_mrr:<12.4f} {weighted_hit1:<12.4f} "
                            f"{weighted_hit3:<12.4f} {weighted_hit10:<12.4f}")

        logger.info("=" * 70)
        # 打印汇总统计
        logger.info("📈 汇总统计 (⚠️  注意：使用了负过滤评估，结果更严格):")
        logger.info("-" * 90)

        # 性能汇总
        if 'weighted_avg_MRR' in last_evaluation:
            logger.info("🏆 加权平均性能 (按样本数量加权):")
            logger.info(f"    MRR: {last_evaluation['weighted_avg_MRR']:.4f} | "
                        f"Hit@1: {last_evaluation['weighted_avg_Hit@1']:.4f} | "
                        f"Hit@3: {last_evaluation['weighted_avg_Hit@3']:.4f} | "
                        f"Hit@10: {last_evaluation['weighted_avg_Hit@10']:.4f}")

        if 'simple_avg_MRR' in last_evaluation:
            logger.info("📊 简单平均性能 (等权重):")
            logger.info(f"    MRR: {last_evaluation['simple_avg_MRR']:.4f} | "
                        f"Hit@1: {last_evaluation['simple_avg_Hit@1']:.4f} | "
                        f"Hit@3: {last_evaluation['simple_avg_Hit@3']:.4f} | "
                        f"Hit@10: {last_evaluation['simple_avg_Hit@10']:.4f}")

        # 数据汇总
        total_samples = sum(data['samples'] for data in snapshot_data)
        trained_snapshots = len([d for d in snapshot_data if d['status'] == "已训练"])
        skipped_snapshots = len([d for d in snapshot_data if d['status'] == "跳过"])

        logger.info(f"📋 数据汇总:")
        logger.info(f"    总快照数: {len(snapshot_data)} | "
                    f"已训练: {trained_snapshots} | "
                    f"跳过: {skipped_snapshots} | "
                    f"总样本数: {total_samples:,}")

        # 重要提示
        logger.info("\n⚠️  重要说明:")
        logger.info("   • 此次评估使用了负过滤，过滤掉了已知的正样本三元组")

        logger.info("=" * 90)

    def _save_final_results(self, all_test_results, total_time):
        """保存最终结果（包含时间统计）"""
        results_file = os.path.join(self.config.log_dir, 'final_results.json')

        final_results = {
            'training_history': self.training_history,
            'all_test_results': all_test_results,
            'time_statistics': {
                'total_time': total_time,
                'total_train_time': self.total_train_time,
                'total_test_time': self.total_test_time,
                'snapshot_train_times': self.snapshot_train_times,
                'snapshot_test_times': self.snapshot_test_times,
                'train_time_ratio': (self.total_train_time / total_time * 100) if total_time > 0 else 0,
                'test_time_ratio': (self.total_test_time / total_time * 100) if total_time > 0 else 0
            },
            'config': vars(self.config)
        }

        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        logger.info(f"💾 最终结果已保存: {results_file}")

        # 打印最终性能总结
        self._print_final_summary(all_test_results)


def create_trainer(config, data_loader, device='cuda'):
    """创建简化的持续学习训练器"""
    return ContinualTrainer(config, data_loader, device)
