#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import json
import random
from collections import defaultdict, deque, Counter
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Set, Optional
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    层次化数据处理器
    """
    
    def __init__(self, data_dir: str, dataset_name: str = 'ENTITY'):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.output_dir = os.path.join(data_dir, f'{dataset_name}_processed')
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 全局映射
        self.global_entity2id = {}
        self.global_relation2id = {}
        self.global_id2entity = {}
        self.global_id2relation = {}
        
        # 快照数据
        self.snapshots = []
        self.snapshot_graphs = []  # 每个快照的图结构
        
        # 统计信息
        self.statistics = {
            'total_entities': 0,
            'total_relations': 0,
            'total_triples': 0,
            'snapshots': []
        }
        
    def load_raw_data(self):
        """加载原始数据"""
        logger.info(f"Loading raw data from {self.data_dir}/{self.dataset_name}")
        
        dataset_path = os.path.join(self.data_dir, self.dataset_name)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # 找到所有时间快照目录
        snapshot_dirs = []
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path) and item.isdigit():
                snapshot_dirs.append(int(item))
        
        snapshot_dirs.sort()
        logger.info(f"Found {len(snapshot_dirs)} snapshots: {snapshot_dirs}")
        
        # 首先构建全局映射
        self._build_global_mappings(dataset_path, snapshot_dirs)
        
        # 然后加载每个快照
        for snapshot_id in snapshot_dirs:
            snapshot_data = self._load_single_snapshot(dataset_path, snapshot_id)
            self.snapshots.append(snapshot_data)
            
            # 构建图结构
            graph = self._build_graph_from_triples(snapshot_data['all_triples'])
            self.snapshot_graphs.append(graph)
        
        logger.info(f"Data loading completed. Total snapshots: {len(self.snapshots)}")
        
    def _build_global_mappings(self, dataset_path: str, snapshot_dirs: List[int]):
        """构建全局实体和关系映射，直接从三元组文件中收集"""
        logger.info("Building global entity and relation mappings from triple files...")
        
        all_entities = set()
        all_relations = set()
        
        # 直接从三元组文件中收集所有实体和关系
        for snapshot_id in snapshot_dirs:
            snapshot_path = os.path.join(dataset_path, str(snapshot_id))
            
            # 检查所有三元组文件
            triple_files = ['train.txt', 'valid.txt', 'test.txt']
            for triple_file in triple_files:
                file_path = os.path.join(snapshot_path, triple_file)
                if os.path.exists(file_path):
                    logger.info(f"Scanning {file_path} for entities and relations...")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:
                                try:
                                    h_entity = parts[0].strip()
                                    r_relation = parts[1].strip()
                                    t_entity = parts[2].strip()
                                    
                                    all_entities.add(h_entity)
                                    all_entities.add(t_entity)
                                    all_relations.add(r_relation)
                                except Exception as e:
                                    if line_num <= 5:  # 只记录前5个错误
                                        logger.warning(f"Error scanning line {line_num} in {file_path}: {e}")
        
        # 构建全局映射
        self.global_entity2id = {entity: idx for idx, entity in enumerate(sorted(all_entities))}
        self.global_relation2id = {relation: idx for idx, relation in enumerate(sorted(all_relations))}
        self.global_id2entity = {idx: entity for entity, idx in self.global_entity2id.items()}
        self.global_id2relation = {idx: relation for relation, idx in self.global_relation2id.items()}
        
        self.statistics['total_entities'] = len(self.global_entity2id)
        self.statistics['total_relations'] = len(self.global_relation2id)
        
        logger.info(f"Global mappings built: {self.statistics['total_entities']} entities, "
                   f"{self.statistics['total_relations']} relations")
    
    def _load_single_snapshot(self, dataset_path: str, snapshot_id: int) -> Dict:
        """加载单个快照数据"""
        snapshot_path = os.path.join(dataset_path, str(snapshot_id))
        
        # 加载三元组数据 - 直接传递None，因为我们不使用本地映射
        train_triples = self._load_triples(snapshot_path, 'train.txt', None, None)
        valid_triples = self._load_triples(snapshot_path, 'valid.txt', None, None)
        test_triples = self._load_triples(snapshot_path, 'test.txt', None, None)
        
        # 所有三元组
        all_triples = train_triples + valid_triples + test_triples
        
        # 收集当前快照中的实体和关系
        snapshot_entities = set()
        snapshot_relations = set()
        
        for h, r, t in all_triples:
            snapshot_entities.add(h)
            snapshot_entities.add(t)
            snapshot_relations.add(r)
        
        # 计算增量信息
        new_entities, old_entities, new_relations, old_relations = self._compute_incremental_info_from_ids(
            snapshot_id, snapshot_entities, snapshot_relations
        )
        
        snapshot_data = {
            'snapshot_id': snapshot_id,
            'train_triples': train_triples,
            'valid_triples': valid_triples,
            'test_triples': test_triples,
            'all_triples': all_triples,
            'entities': snapshot_entities,
            'relations': snapshot_relations,
            'new_entities': new_entities,
            'old_entities': old_entities,
            'new_relations': new_relations,
            'old_relations': old_relations
        }
        
        # 更新统计信息
        snapshot_stats = {
            'snapshot_id': snapshot_id,
            'num_train_triples': len(train_triples),
            'num_valid_triples': len(valid_triples),
            'num_test_triples': len(test_triples),
            'num_entities': len(snapshot_entities),
            'num_relations': len(snapshot_relations),
            'num_new_entities': len(new_entities),
            'num_new_relations': len(new_relations)
        }
        self.statistics['snapshots'].append(snapshot_stats)
        
        # 更新总三元组数
        self.statistics['total_triples'] += len(all_triples)
        
        logger.info(f"Loaded snapshot {snapshot_id}: "
                   f"{len(train_triples)} train, {len(valid_triples)} valid, "
                   f"{len(test_triples)} test triples")
        
        return snapshot_data
    
    def _load_triples(self, snapshot_path: str, filename: str, 
                     local_id2entity: Dict, local_id2relation: Dict) -> List[List[int]]:
        """加载三元组并转换为全局ID，直接使用字符串ID"""
        triples = []
        file_path = os.path.join(snapshot_path, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return triples
        
        logger.info(f"Loading triples from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    try:
                        # 直接使用字符串作为实体和关系ID
                        h_entity = parts[0].strip()
                        r_relation = parts[1].strip()
                        t_entity = parts[2].strip()
                        
                        # 检查是否存在于全局映射中
                        if h_entity in self.global_entity2id and r_relation in self.global_relation2id and t_entity in self.global_entity2id:
                            h_global = self.global_entity2id[h_entity]
                            r_global = self.global_relation2id[r_relation]
                            t_global = self.global_entity2id[t_entity]
                            triples.append([h_global, r_global, t_global])
                        else:
                            # 记录第一次遇到但数量限制在100次以内
                            if line_num <= 100:
                                missing_items = []
                                if h_entity not in self.global_entity2id:
                                    missing_items.append(f"head: {h_entity}")
                                if r_relation not in self.global_relation2id:
                                    missing_items.append(f"relation: {r_relation}")
                                if t_entity not in self.global_entity2id:
                                    missing_items.append(f"tail: {t_entity}")
                                logger.debug(f"Missing mapping at line {line_num}: {', '.join(missing_items)}")
                    
                    except Exception as e:
                        if line_num <= 10:  # 只记录前10个错误
                            logger.warning(f"Error processing line {line_num}: {line.strip()}, Error: {e}")
                        continue
                else:
                    if line_num <= 10:  # 只记录前10个格式错误
                        logger.warning(f"Invalid format at line {line_num}: {line.strip()}")
        
        logger.info(f"Successfully loaded {len(triples)} triples from {filename}")
        return triples
    
    def _compute_incremental_info_from_ids(self, snapshot_id: int, current_entities: Set[int], 
                                          current_relations: Set[int]) -> Tuple[Set, Set, Set, Set]:
        """从全局ID计算增量信息"""
        if snapshot_id == 0 or len(self.snapshots) == 0:
            # 第一个快照，所有都是新的
            return current_entities, set(), current_relations, set()
        
        # 计算之前快照的实体和关系
        previous_entities = set()
        previous_relations = set()
        for i in range(len(self.snapshots)):  # 使用已加载的快照数量
            previous_entities.update(self.snapshots[i]['entities'])
            previous_relations.update(self.snapshots[i]['relations'])
        
        new_entities = current_entities - previous_entities
        old_entities = current_entities & previous_entities
        new_relations = current_relations - previous_relations
        old_relations = current_relations & previous_relations
        
        return new_entities, old_entities, new_relations, old_relations
    
    def _build_graph_from_triples(self, triples: List[List[int]]) -> nx.Graph:
        """从三元组构建图结构"""
        graph = nx.Graph()
        
        for h, r, t in triples:
            graph.add_edge(h, t, relation=r)
        
        return graph
    
    def apply_hierarchical_ordering(self):
        """
        """
        logger.info("Applying hierarchical ordering to snapshots...")
        
        for i, snapshot_data in enumerate(self.snapshots):
            if i == 0:
                # 第一个快照不需要层次化排序
                snapshot_data['hierarchical_groups'] = [snapshot_data['train_triples']]
                continue
            
            logger.info(f"Processing hierarchical ordering for snapshot {i}")
            
            # 获取增量三元组（只包含新的三元组）
            delta_triples = self._get_delta_triples(i)
            
            if not delta_triples:
                snapshot_data['hierarchical_groups'] = []
                continue
            
            # Inter-hierarchical ordering: 使用BFS从旧图扩展
            hierarchical_groups = self._inter_hierarchical_ordering(delta_triples, i)
            
            # Intra-hierarchical ordering: 每层内部排序
            for group_idx, group in enumerate(hierarchical_groups):
                hierarchical_groups[group_idx] = self._intra_hierarchical_ordering(group)
            
            snapshot_data['hierarchical_groups'] = hierarchical_groups
            
            logger.info(f"Snapshot {i} divided into {len(hierarchical_groups)} hierarchical groups")
    
    def _get_delta_triples(self, snapshot_id: int) -> List[List[int]]:
        """获取增量三元组（相对于之前快照的新三元组）"""
        current_triples = set(tuple(triple) for triple in self.snapshots[snapshot_id]['train_triples'])
        
        # 收集之前所有快照的三元组
        previous_triples = set()
        for i in range(snapshot_id):
            for triple in self.snapshots[i]['train_triples']:
                previous_triples.add(tuple(triple))
        
        # 计算增量三元组
        delta_triples = current_triples - previous_triples
        return [list(triple) for triple in delta_triples]
    
    def _inter_hierarchical_ordering(self, delta_triples: List[List[int]], 
                                   snapshot_id: int) -> List[List[List[int]]]:
        """
        Inter-hierarchical ordering: 使用BFS从旧图扩展
        将增量三元组按照与旧图的距离分层
        """
        if snapshot_id == 0:
            return [delta_triples]
        
        # 构建旧图（之前所有快照的图）
        old_graph = nx.Graph()
        for i in range(snapshot_id):
            for h, r, t in self.snapshots[i]['train_triples']:
                old_graph.add_edge(h, t, relation=r)
        
        # 构建增量图
        delta_graph = nx.Graph()
        for h, r, t in delta_triples:
            delta_graph.add_edge(h, t, relation=r)
        
        # 使用BFS进行分层
        layers = []
        visited_triples = set()
        
        # 第一层：直接连接到旧图的三元组
        current_layer_triples = []
        for h, r, t in delta_triples:
            if (h in old_graph.nodes) or (t in old_graph.nodes):
                current_layer_triples.append([h, r, t])
                visited_triples.add((h, r, t))
        
        if current_layer_triples:
            layers.append(current_layer_triples)
        
        # 后续层：通过BFS扩展
        current_nodes = set()
        for h, r, t in current_layer_triples:
            current_nodes.add(h)
            current_nodes.add(t)
        
        while len(visited_triples) < len(delta_triples):
            next_layer_triples = []
            next_nodes = set()
            
            for h, r, t in delta_triples:
                if (h, r, t) not in visited_triples:
                    if (h in current_nodes) or (t in current_nodes):
                        next_layer_triples.append([h, r, t])
                        visited_triples.add((h, r, t))
                        next_nodes.add(h)
                        next_nodes.add(t)
            
            if not next_layer_triples:
                # 处理孤立的三元组
                remaining_triples = []
                for h, r, t in delta_triples:
                    if (h, r, t) not in visited_triples:
                        remaining_triples.append([h, r, t])
                        visited_triples.add((h, r, t))
                
                if remaining_triples:
                    layers.append(remaining_triples)
                break
            
            layers.append(next_layer_triples)
            current_nodes = next_nodes
        
        return layers
    
    def _intra_hierarchical_ordering(self, triples: List[List[int]]) -> List[List[int]]:
        """
        Intra-hierarchical ordering: 层内排序
        根据图结构特性对三元组进行排序
        """
        if len(triples) <= 1:
            return triples
        
        # 构建临时图
        temp_graph = nx.Graph()
        for h, r, t in triples:
            temp_graph.add_edge(h, t, relation=r)
        
        # 计算节点度数
        node_degrees = dict(temp_graph.degree())
        
        # 按度数和连通性排序
        def triple_priority(triple):
            h, r, t = triple
            h_degree = node_degrees.get(h, 0)
            t_degree = node_degrees.get(t, 0)
            # 优先处理度数高的节点对应的三元组
            return -(h_degree + t_degree)
        
        sorted_triples = sorted(triples, key=triple_priority)
        return sorted_triples
    
    def generate_enhanced_negative_samples(self):
        """
        生成增强的负采样
        考虑图结构和实体频率的智能负采样
        """
        logger.info("Generating enhanced negative samples...")
        
        for snapshot_data in self.snapshots:
            snapshot_id = snapshot_data['snapshot_id']
            train_triples = snapshot_data['train_triples']
            
            if not train_triples:
                snapshot_data['negative_samples'] = []
                continue
            
            # 构建正样本集合
            positive_set = set(tuple(triple) for triple in train_triples)
            
            # 统计实体和关系频率
            entity_freq = Counter()
            relation_freq = Counter()
            
            for h, r, t in train_triples:
                entity_freq[h] += 1
                entity_freq[t] += 1
                relation_freq[r] += 1
            
            # 生成负样本
            negative_samples = []
            num_negatives_per_positive = 5  # 每个正样本生成5个负样本
            
            for h, r, t in tqdm(train_triples, desc=f"Generating negatives for snapshot {snapshot_id}"):
                triple_negatives = []
                
                # 头实体替换（考虑频率分布）
                for _ in range(num_negatives_per_positive // 2):
                    attempts = 0
                    while attempts < 100:  # 最多尝试100次
                        if random.random() < 0.7:
                            # 70%概率使用频率采样
                            neg_h = self._frequency_based_sampling(entity_freq, exclude=h)
                        else:
                            # 30%概率使用均匀采样
                            neg_h = random.randint(0, self.statistics['total_entities'] - 1)
                        
                        neg_triple = (neg_h, r, t)
                        if neg_triple not in positive_set:
                            triple_negatives.append([neg_h, r, t])
                            break
                        attempts += 1
                
                # 尾实体替换
                for _ in range(num_negatives_per_positive - len(triple_negatives)):
                    attempts = 0
                    while attempts < 100:
                        if random.random() < 0.7:
                            neg_t = self._frequency_based_sampling(entity_freq, exclude=t)
                        else:
                            neg_t = random.randint(0, self.statistics['total_entities'] - 1)
                        
                        neg_triple = (h, r, neg_t)
                        if neg_triple not in positive_set:
                            triple_negatives.append([h, r, neg_t])
                            break
                        attempts += 1
                
                negative_samples.extend(triple_negatives)
            
            snapshot_data['negative_samples'] = negative_samples
            logger.info(f"Generated {len(negative_samples)} negative samples for snapshot {snapshot_id}")
    
    def _frequency_based_sampling(self, freq_counter: Counter, exclude: int = None) -> int:
        """基于频率的采样"""
        candidates = list(freq_counter.keys())
        if exclude is not None and exclude in candidates:
            candidates.remove(exclude)
        
        if not candidates:
            return random.randint(0, self.statistics['total_entities'] - 1)
        
        # 使用逆频率作为权重（低频实体有更高概率被采样）
        weights = [1.0 / (freq_counter[candidate] + 1) for candidate in candidates]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(candidates)
        
        # 轮盘赌选择
        r = random.uniform(0, total_weight)
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return candidates[i]
        
        return candidates[-1]
    
    def save_processed_data(self):
        """保存处理后的数据"""
        logger.info(f"Saving processed data to {self.output_dir}")
        
        # 保存全局映射
        mappings = {
            'global_entity2id': self.global_entity2id,
            'global_relation2id': self.global_relation2id,
            'global_id2entity': self.global_id2entity,
            'global_id2relation': self.global_id2relation
        }
        
        with open(os.path.join(self.output_dir, 'global_mappings.pkl'), 'wb') as f:
            pickle.dump(mappings, f)
        
        # 保存统计信息
        with open(os.path.join(self.output_dir, 'statistics.json'), 'w') as f:
            json.dump(self.statistics, f, indent=2)
        
        # 保存每个快照的处理数据
        for i, snapshot_data in enumerate(self.snapshots):
            snapshot_dir = os.path.join(self.output_dir, str(i))
            os.makedirs(snapshot_dir, exist_ok=True)
            
            # 保存基础数据
            if snapshot_data['train_triples']:
                np.save(os.path.join(snapshot_dir, 'train_triples.npy'), 
                       np.array(snapshot_data['train_triples']))
            if snapshot_data['valid_triples']:
                np.save(os.path.join(snapshot_dir, 'valid_triples.npy'), 
                       np.array(snapshot_data['valid_triples']))
            if snapshot_data['test_triples']:
                np.save(os.path.join(snapshot_dir, 'test_triples.npy'), 
                       np.array(snapshot_data['test_triples']))
            
            # 保存负样本
            if 'negative_samples' in snapshot_data and snapshot_data['negative_samples']:
                np.save(os.path.join(snapshot_dir, 'negative_samples.npy'), 
                       np.array(snapshot_data['negative_samples']))
            
            # 保存层次化分组
            if 'hierarchical_groups' in snapshot_data:
                with open(os.path.join(snapshot_dir, 'hierarchical_groups.pkl'), 'wb') as f:
                    pickle.dump(snapshot_data['hierarchical_groups'], f)
            
            # 保存元数据
            metadata = {
                'snapshot_id': snapshot_data['snapshot_id'],
                'entities': list(snapshot_data['entities']),
                'relations': list(snapshot_data['relations']),
                'new_entities': list(snapshot_data['new_entities']),
                'old_entities': list(snapshot_data['old_entities']),
                'new_relations': list(snapshot_data['new_relations']),
                'old_relations': list(snapshot_data['old_relations'])
            }
            
            with open(os.path.join(snapshot_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info("Data processing and saving completed!")
    
    def generate_analysis_report(self):
        """生成数据分析报告"""
        logger.info("Generating analysis report...")
        
        report = {
            'dataset_name': self.dataset_name,
            'total_snapshots': len(self.snapshots),
            'total_entities': self.statistics['total_entities'],
            'total_relations': self.statistics['total_relations'],
            'total_triples': self.statistics['total_triples'],
            'snapshot_analysis': []
        }
        
        # 分析每个快照
        for i, stats in enumerate(self.statistics['snapshots']):
            snapshot_analysis = {
                'snapshot_id': i,
                'num_train_triples': stats['num_train_triples'],
                'num_valid_triples': stats['num_valid_triples'],
                'num_test_triples': stats['num_test_triples'],
                'num_entities': stats['num_entities'],
                'num_relations': stats['num_relations'],
                'num_new_entities': stats['num_new_entities'],
                'num_new_relations': stats['num_new_relations'],
                'entity_growth_rate': stats['num_new_entities'] / max(1, stats['num_entities'] - stats['num_new_entities']),
                'relation_growth_rate': stats['num_new_relations'] / max(1, stats['num_relations'] - stats['num_new_relations'])
            }
            
            # 分析层次化分组信息
            if 'hierarchical_groups' in self.snapshots[i]:
                groups = self.snapshots[i]['hierarchical_groups']
                snapshot_analysis['num_hierarchical_groups'] = len(groups)
                snapshot_analysis['group_sizes'] = [len(group) for group in groups]
            
            report['snapshot_analysis'].append(snapshot_analysis)
        
        # 保存报告
        with open(os.path.join(self.output_dir, 'analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # 打印简要报告
        print("\n" + "="*60)
        print("DATA ANALYSIS REPORT")
        print("="*60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Total Snapshots: {report['total_snapshots']}")
        print(f"Total Entities: {report['total_entities']}")
        print(f"Total Relations: {report['total_relations']}")
        print(f"Total Triples: {report['total_triples']}")
        print("\nSnapshot Details:")
        for snapshot in report['snapshot_analysis']:
            print(f"  Snapshot {snapshot['snapshot_id']}: "
                  f"Train={snapshot['num_train_triples']}, "
                  f"Valid={snapshot['num_valid_triples']}, "
                  f"Test={snapshot['num_test_triples']}, "
                  f"Entities={snapshot['num_entities']}, "
                  f"New_Entities={snapshot['num_new_entities']}")
        
        logger.info("Analysis report generated successfully!")


def main():
    """主函数：处理data目录下的所有数据集"""
    parser = argparse.ArgumentParser(description='层次化数据预处理器')
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='数据目录路径 (默认: ./data)')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       help='要处理的数据集名称列表 (如不指定则处理所有数据集)')
    parser.add_argument('--output_dir', type=str, 
                       help='输出目录 (如不指定则在各数据集目录下生成)')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式，显示更详细的日志信息')
    
    args = parser.parse_args()
    
    # 设置调试模式
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    data_dir = args.data_dir
    
    if not os.path.exists(data_dir):
        logger.error(f"数据目录不存在: {data_dir}")
        return
    
    # 如果没有指定数据集，则自动发现所有数据集
    if args.datasets is None:
        datasets = []
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path) and item not in ['processed', '__pycache__']:
                # 检查是否包含时间快照目录
                has_snapshots = False
                try:
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isdir(subitem_path) and subitem.isdigit():
                            # 进一步检查是否包含必要的文件
                            required_files = ['entity2id.txt', 'relation2id.txt']
                            if any(os.path.exists(os.path.join(subitem_path, f)) for f in required_files):
                                has_snapshots = True
                                break
                except PermissionError:
                    continue
                    
                if has_snapshots:
                    datasets.append(item)
        
        if not datasets:
            logger.warning("未找到任何有效的数据集")
            logger.info("数据集目录结构应该为: data/DATASET_NAME/SNAPSHOT_ID/[entity2id.txt, relation2id.txt, train.txt, etc.]")
            return
            
        logger.info(f"自动发现数据集: {datasets}")
    else:
        datasets = args.datasets
    
    # 处理每个数据集
    for dataset_name in datasets:
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"开始处理数据集: {dataset_name}")
            logger.info(f"{'='*80}")
            
            # 创建处理器
            processor = DataProcessor(data_dir, dataset_name)
            
            # 执行完整的处理流程
            logger.info("步骤 1: 加载原始数据...")
            processor.load_raw_data()
            
            # 检查是否成功加载数据
            if not processor.snapshots:
                logger.warning(f"数据集 {dataset_name} 没有成功加载任何快照，跳过处理")
                continue
            
            logger.info("步骤 2: 应用层次化排序...")
            processor.apply_hierarchical_ordering()
            
            logger.info("步骤 3: 生成增强负样本...")
            processor.generate_enhanced_negative_samples()
            
            logger.info("步骤 4: 保存处理后的数据...")
            processor.save_processed_data()
            
            logger.info("步骤 5: 生成分析报告...")
            processor.generate_analysis_report()
            
            logger.info(f"数据集 {dataset_name} 处理完成!")
            
        except Exception as e:
            logger.error(f"处理数据集 {dataset_name} 时出错: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()
            continue
    
    logger.info(f"\n{'='*80}")
    logger.info("所有数据集处理完成!")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()