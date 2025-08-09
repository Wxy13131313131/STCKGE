#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的数据加载器
去掉复杂的缓存和异步加载功能，保持核心功能
"""

import torch
import numpy as np
import os
import pickle
import json
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class KGDataset(Dataset):
    """知识图谱数据集"""

    def __init__(self, triples: List[List[int]], negative_samples: List[List[int]],
                 num_entities: int, num_relations: int, negative_sampling_ratio: int = 10):
        self.triples = triples
        self.negative_samples = negative_samples
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.negative_sampling_ratio = negative_sampling_ratio

        # 构建正样本集合
        self.positive_set = set(tuple(triple) for triple in triples)

        logger.info(f"数据集初始化完成: {len(triples)} 个正样本")

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        """获取数据项"""
        if idx >= len(self.triples):
            idx = idx % len(self.triples)

        pos_triple = torch.LongTensor(self.triples[idx])

        # 生成负样本
        if self.negative_samples and len(self.negative_samples) > 0:
            # 使用预生成的负样本
            neg_triples = self._get_pregenerated_negatives(idx)
        else:
            # 动态生成负样本
            neg_triples = self._generate_dynamic_negatives(self.triples[idx])

        return pos_triple, neg_triples

    def _get_pregenerated_negatives(self, idx):
        """获取预生成的负样本"""
        negatives = []
        start_idx = idx * self.negative_sampling_ratio

        for i in range(self.negative_sampling_ratio):
            neg_idx = (start_idx + i) % len(self.negative_samples)
            negatives.append(self.negative_samples[neg_idx])

        return torch.LongTensor(negatives)

    # def _generate_dynamic_negatives(self, pos_triple):
    #     """动态生成负样本"""
    #     h, r, t = pos_triple
    #     negatives = []
    #
    #     # 头实体替换和尾实体替换各占一半
    #     half_negatives = self.negative_sampling_ratio // 2
    #
    #     # 头实体替换
    #     for _ in range(half_negatives):
    #         attempts = 0
    #         while attempts < 50:  # 最多尝试50次
    #             neg_h = random.randint(0, self.num_entities - 1)
    #             if (neg_h, r, t) not in self.positive_set:
    #                 negatives.append([neg_h, r, t])
    #                 break
    #             attempts += 1
    #
    #     # 尾实体替换
    #     remaining = self.negative_sampling_ratio - len(negatives)
    #     for _ in range(remaining):
    #         attempts = 0
    #         while attempts < 50:
    #             neg_t = random.randint(0, self.num_entities - 1)
    #             if (h, r, neg_t) not in self.positive_set:
    #                 negatives.append([h, r, neg_t])
    #                 break
    #             attempts += 1
    #
    #     # 如果生成的负样本不足，用随机采样补充
    #     while len(negatives) < self.negative_sampling_ratio:
    #         if random.random() < 0.5:
    #             neg_h = random.randint(0, self.num_entities - 1)
    #             negatives.append([neg_h, r, t])
    #         else:
    #             neg_t = random.randint(0, self.num_entities - 1)
    #             negatives.append([h, r, neg_t])
    #
    #     return torch.LongTensor(negatives[:self.negative_sampling_ratio])
    def _generate_dynamic_negatives(self, pos_triple):
        """动态生成负样本（批量生成+过滤）"""
        h, r, t = pos_triple
        negatives = []

        # 批量生成候选负样本
        replace_head = np.random.rand(self.negative_sampling_ratio) < 0.5
        neg_entities = np.random.randint(0, self.num_entities, self.negative_sampling_ratio)

        # 创建候选负样本
        candidates = [
            [neg_entities[i], r, t] if replace_head[i] else [h, r, neg_entities[i]]
            for i in range(self.negative_sampling_ratio)
        ]

        # 过滤条件：不在正样本集中且不与原实体相同
        filtered = [
            cand for cand in candidates
            if tuple(cand) not in self.positive_set and
               (cand[0] != h if cand[2] == t else cand[2] != t)
        ]

        # 如果过滤后数量不足，补充生成
        if len(filtered) < self.negative_sampling_ratio:
            needed = self.negative_sampling_ratio - len(filtered)
            extra = []
            for _ in range(needed * 2):  # 生成双倍以防过滤
                if np.random.rand() < 0.5:
                    neg_h = np.random.randint(0, self.num_entities)
                    if neg_h != h:
                        extra.append([neg_h, r, t])
                else:
                    neg_t = np.random.randint(0, self.num_entities)
                    if neg_t != t:
                        extra.append([h, r, neg_t])
            # 再次过滤并取所需数量
            extra_filtered = [
                                 cand for cand in extra
                                 if tuple(cand) not in self.positive_set
                             ][:needed]
            filtered.extend(extra_filtered)

        # 最终确保数量正确
        negatives = filtered[:self.negative_sampling_ratio]

        # 极端情况下仍不足，使用最后一个有效样本填充
        while len(negatives) < self.negative_sampling_ratio:
            if negatives:
                negatives.append(negatives[-1])
            else:
                # 创建保证不同的样本
                neg_h = (h + 1) % self.num_entities
                negatives.append([neg_h, r, t])

        return torch.LongTensor(negatives)
    # def _generate_dynamic_negatives(self, pos_triple):
    #     """动态生成负样本（修正正样本泄露问题）"""
    #     h, r, t = pos_triple
    #     negatives = []
    #
    #     # 头实体替换和尾实体替换各占一半
    #     half_negatives = self.negative_sampling_ratio // 2
    #
    #     # 头实体替换
    #     head_attempts = 0
    #     head_neg_count = 0
    #     while head_neg_count < half_negatives and head_attempts < 1000:  # 增加最大尝试次数
    #         neg_h = random.randint(0, self.num_entities - 1)
    #         candidate_triple = (neg_h, r, t)
    #         # 修正：确保不生成已知正样本，且不与原三元组相同
    #         if candidate_triple not in self.positive_set and neg_h != h:
    #             negatives.append([neg_h, r, t])
    #             head_neg_count += 1
    #         head_attempts += 1
    #
    #     # 尾实体替换
    #     tail_attempts = 0
    #     tail_neg_count = 0
    #     remaining = self.negative_sampling_ratio - len(negatives)
    #     while tail_neg_count < remaining and tail_attempts < 1000:
    #         neg_t = random.randint(0, self.num_entities - 1)
    #         candidate_triple = (h, r, neg_t)
    #         # 修正：确保不生成已知正样本，且不与原三元组相同
    #         if candidate_triple not in self.positive_set and neg_t != t:
    #             negatives.append([h, r, neg_t])
    #             tail_neg_count += 1
    #         tail_attempts += 1
    #
    #     # 如果生成的负样本不足，用更宽松的条件补充
    #     while len(negatives) < self.negative_sampling_ratio:
    #         if random.random() < 0.5:
    #             # 头实体替换
    #             neg_h = random.randint(0, self.num_entities - 1)
    #             if neg_h != h:  # 至少确保不与原实体相同
    #                 negatives.append([neg_h, r, t])
    #         else:
    #             # 尾实体替换
    #             neg_t = random.randint(0, self.num_entities - 1)
    #             if neg_t != t:  # 至少确保不与原实体相同
    #                 negatives.append([h, r, neg_t])
    #
    #     # 确保返回正确数量的负样本
    #     negatives = negatives[:self.negative_sampling_ratio]
    #
    #     # 如果还是不够，使用最后一个元素填充
    #     while len(negatives) < self.negative_sampling_ratio:
    #         if negatives:
    #             negatives.append(negatives[-1])
    #         else:
    #             # 最后的备选方案：创建一个不同的三元组
    #             backup_h = (h + 1) % self.num_entities
    #             negatives.append([backup_h, r, t])
    #
    #     return torch.LongTensor(negatives[:self.negative_sampling_ratio])

class STDataLoader:
    """简化的时序知识图谱数据加载器"""

    def __init__(self, processed_data_dir: str):
        self.processed_data_dir = processed_data_dir

        self.snapshots = []
        self.global_mappings = {}
        self.statistics = {}

        # 加载数据
        self._load_processed_data()

        logger.info(f"数据加载器初始化完成，包含 {len(self.snapshots)} 个快照")

    def _load_processed_data(self):
        """加载预处理数据"""
        # 加载全局映射
        mappings_file = os.path.join(self.processed_data_dir, 'global_mappings.pkl')
        if os.path.exists(mappings_file):
            with open(mappings_file, 'rb') as f:
                self.global_mappings = pickle.load(f)
        else:
            logger.warning(f"找不到全局映射文件: {mappings_file}")
            self.global_mappings = {}

        # 加载统计信息
        stats_file = os.path.join(self.processed_data_dir, 'statistics.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                self.statistics = json.load(f)
        else:
            logger.warning(f"找不到统计信息文件: {stats_file}")
            self.statistics = {'total_entities': 1000, 'total_relations': 100, 'snapshots': []}

        # 加载快照数据
        self._load_snapshots()

    def _load_snapshots(self):
        """加载所有快照数据"""
        snapshot_dirs = []

        # 查找所有数字目录
        for item in os.listdir(self.processed_data_dir):
            item_path = os.path.join(self.processed_data_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                snapshot_dirs.append(int(item))

        # 按编号排序
        snapshot_dirs.sort()

        for snapshot_id in snapshot_dirs:
            snapshot_data = self._load_single_snapshot(snapshot_id)
            self.snapshots.append(snapshot_data)

    def _load_single_snapshot(self, snapshot_id: int) -> Dict:
        """加载单个快照数据"""
        snapshot_dir = os.path.join(self.processed_data_dir, str(snapshot_id))

        if not os.path.exists(snapshot_dir):
            logger.warning(f"快照目录不存在: {snapshot_dir}")
            return self._create_empty_snapshot_data(snapshot_id)

        # 加载基础数据
        data = {}

        # 加载三元组
        train_file = os.path.join(snapshot_dir, 'train_triples.npy')
        if os.path.exists(train_file):
            data['train_triples'] = np.load(train_file).tolist()
        else:
            data['train_triples'] = []

        valid_file = os.path.join(snapshot_dir, 'valid_triples.npy')
        if os.path.exists(valid_file):
            data['valid_triples'] = np.load(valid_file).tolist()
        else:
            data['valid_triples'] = []

        test_file = os.path.join(snapshot_dir, 'test_triples.npy')
        if os.path.exists(test_file):
            data['test_triples'] = np.load(test_file).tolist()
        else:
            data['test_triples'] = []

        # 加载负样本
        neg_file = os.path.join(snapshot_dir, 'negative_samples.npy')
        if os.path.exists(neg_file):
            data['negative_samples'] = np.load(neg_file).tolist()
        else:
            data['negative_samples'] = []

        # 加载元数据
        metadata_file = os.path.join(snapshot_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                data.update(metadata)

        data['snapshot_id'] = snapshot_id

        logger.info(f"加载快照 {snapshot_id}: "
                    f"{len(data['train_triples'])} 训练, "
                    f"{len(data['valid_triples'])} 验证, "
                    f"{len(data['test_triples'])} 测试")

        return data

    def _create_empty_snapshot_data(self, snapshot_id: int) -> Dict:
        """创建空的快照数据"""
        return {
            'snapshot_id': snapshot_id,
            'train_triples': [],
            'valid_triples': [],
            'test_triples': [],
            'negative_samples': [],
            'entities': [],
            'relations': [],
            'new_entities': [],
            'old_entities': [],
            'new_relations': [],
            'old_relations': []
        }

    @property
    def num_entities(self) -> int:
        return self.statistics.get('total_entities', 1000)

    @property
    def num_relations(self) -> int:
        return self.statistics.get('total_relations', 100)

    def get_snapshot_dataset(self, snapshot_id: int, mode: str = 'train',
                             negative_sampling_ratio: int = 10) -> KGDataset:
        """获取快照数据集"""
        if snapshot_id >= len(self.snapshots):
            logger.warning(f"快照 {snapshot_id} 不存在，使用快照 0")
            snapshot_id = 0

        snapshot_data = self.snapshots[snapshot_id]

        # 选择数据模式
        if mode == 'train':
            triples = snapshot_data['train_triples']
        elif mode == 'valid':
            triples = snapshot_data.get('valid_triples', [])
            if not triples and snapshot_data['train_triples']:
                # 如果没有验证数据，使用部分训练数据
                logger.warning(f"快照 {snapshot_id} 不存在val数据")
        elif mode == 'test':
            triples = snapshot_data.get('test_triples', [])
            if not triples and snapshot_data['train_triples']:
                # 如果没有测试数据，使用部分训练数据
                logger.warning(f"快照 {snapshot_id} 不存在test数据")
        else:
            raise ValueError(f"无效的模式: {mode}")

        # 如果三元组为空
        if not triples:
            logger.warning(f"快照 {snapshot_id} 的 {mode} 数据为空")

        # 创建数据集
        return KGDataset(
            triples=triples,
            negative_samples=snapshot_data['negative_samples'],
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            negative_sampling_ratio=negative_sampling_ratio
        )

    def get_snapshot_dataloader(self, snapshot_id: int, mode: str = 'train',
                                batch_size: int = 512, shuffle: bool = True,
                                negative_sampling_ratio: int = 10,
                                num_workers: int = 0) -> DataLoader:
        """获取数据加载器"""
        dataset = self.get_snapshot_dataset(snapshot_id, mode, negative_sampling_ratio)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        """批处理函数"""
        pos_triples = []
        neg_triples = []

        for pos_triple, neg_triple_batch in batch:
            pos_triples.append(pos_triple)
            neg_triples.append(neg_triple_batch)

        pos_triples = torch.stack(pos_triples)
        neg_triples = torch.stack(neg_triples)

        return pos_triples, neg_triples

    def get_incremental_info(self, snapshot_id: int) -> Dict:
        """获取快照增量信息"""
        if snapshot_id >= len(self.snapshots):
            return {
                'num_new_entities': 0,
                'num_new_relations': 0,
                'num_old_entities': 0,
                'num_old_relations': 0,
                'new_entities': [],
                'old_entities': [],
                'new_relations': [],
                'old_relations': []
            }

        snapshot_data = self.snapshots[snapshot_id]

        # 从快照数据中获取增量信息
        new_entities = snapshot_data.get('new_entities', [])
        old_entities = snapshot_data.get('old_entities', [])
        new_relations = snapshot_data.get('new_relations', [])
        old_relations = snapshot_data.get('old_relations', [])

        # 如果没有预计算的增量信息，从当前和之前的快照计算
        if not new_entities and not old_entities:
            current_entities = set(snapshot_data.get('entities', []))

            previous_entities = set()
            for i in range(snapshot_id):
                if i < len(self.snapshots):
                    previous_entities.update(self.snapshots[i].get('entities', []))

            new_entities = list(current_entities - previous_entities)
            old_entities = list(current_entities & previous_entities)

        if not new_relations and not old_relations:
            current_relations = set(snapshot_data.get('relations', []))

            previous_relations = set()
            for i in range(snapshot_id):
                if i < len(self.snapshots):
                    previous_relations.update(self.snapshots[i].get('relations', []))

            new_relations = list(current_relations - previous_relations)
            old_relations = list(current_relations & previous_relations)

        return {
            'num_new_entities': len(new_entities),
            'num_new_relations': len(new_relations),
            'num_old_entities': len(old_entities),
            'num_old_relations': len(old_relations),
            'new_entities': new_entities,
            'old_entities': old_entities,
            'new_relations': new_relations,
            'old_relations': old_relations
        }


def create_data_loader(processed_data_dir: str, **kwargs) -> STDataLoader:
    """创建简化的数据加载器"""
    return STDataLoader(processed_data_dir)