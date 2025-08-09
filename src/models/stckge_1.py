"""
简化的BoxE模型，支持持续学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SimpleBoxE(nn.Module):
    """简化的BoxE模型"""

    def __init__(self, num_entities: int, num_relations: int, embed_dim: int,
                 margin: float = 6.0, device: str = 'cuda'):
        super(SimpleBoxE, self).__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        self.margin = margin
        self.device = device

        # 实体嵌入 (点表示)
        self.entity_embeddings = nn.Embedding(num_entities, embed_dim)

        # 关系嵌入 (盒子表示: 中心点 + 偏移量)
        self.relation_base = nn.Embedding(num_relations, embed_dim)
        self.relation_delta = nn.Embedding(num_relations, embed_dim)

        # 实体和关系的重要性权重（用于蒸馏）
        self.entity_importance = nn.Parameter(torch.ones(num_entities), requires_grad=False)
        self.relation_importance = nn.Parameter(torch.ones(num_relations), requires_grad=False)

        self.init_embeddings()

    def init_embeddings(self):
        """初始化嵌入参数"""
        # 实体嵌入初始化
        nn.init.xavier_uniform_(self.entity_embeddings.weight)

        # 关系基点初始化
        nn.init.xavier_uniform_(self.relation_base.weight)

        # 关系偏移量初始化（确保为正数）
        nn.init.uniform_(self.relation_delta.weight, 0.01, 0.1)

    def get_box_bounds(self, relations):
        """获取关系盒子的边界"""
        base = self.relation_base(relations)
        delta = torch.abs(self.relation_delta(relations))  # 确保delta为正数

        # 添加小的epsilon防止盒子退化为点
        epsilon = 1e-6
        delta = torch.clamp(delta, min=epsilon)

        lower_bounds = base - delta
        upper_bounds = base + delta

        return lower_bounds, upper_bounds

    def point_to_box_distance(self, points, lower_bounds, upper_bounds):
        """计算点到盒子的距离"""
        # 计算点到盒子外部的距离
        dist_to_lower = torch.clamp(lower_bounds - points, min=0)
        dist_to_upper = torch.clamp(points - upper_bounds, min=0)
        distances = dist_to_lower + dist_to_upper

        # 使用L1范数
        return torch.sum(distances, dim=-1)

    def score_function(self, heads, relations, tails):
        """BoxE评分函数"""
        head_emb = self.entity_embeddings(heads)
        tail_emb = self.entity_embeddings(tails)

        lower_bounds, upper_bounds = self.get_box_bounds(relations)

        head_distance = self.point_to_box_distance(head_emb, lower_bounds, upper_bounds)
        tail_distance = self.point_to_box_distance(tail_emb, lower_bounds, upper_bounds)

        total_distance = head_distance + tail_distance

        # 距离越小分数越高
        scores = -total_distance

        return scores

    def forward(self, triples):
        """前向传播"""
        heads, relations, tails = triples[:, 0], triples[:, 1], triples[:, 2]
        return self.score_function(heads, relations, tails)

    def compute_loss(self, pos_triples, neg_triples):
        """计算边际损失"""
        pos_scores = self.forward(pos_triples)
        neg_scores = self.forward(neg_triples)

        # 边际损失
        loss = torch.clamp(self.margin + neg_scores - pos_scores, min=0)

        return torch.mean(loss)

    def update_importance_weights(self, entities: torch.Tensor, relations: torch.Tensor,
                                  learning_rate: float = 0.01):
        """更新实体和关系的重要性权重"""
        with torch.no_grad():
            # 更新实体重要性
            for entity in entities:
                if entity < len(self.entity_importance):
                    self.entity_importance[entity] += learning_rate

            # 更新关系重要性
            for relation in relations:
                if relation < len(self.relation_importance):
                    self.relation_importance[relation] += learning_rate

            # 归一化
            self.entity_importance.data = F.normalize(self.entity_importance.data, p=1, dim=0)
            self.relation_importance.data = F.normalize(self.relation_importance.data, p=1, dim=0)

    def expand_embeddings(self, new_num_entities: int, new_num_relations: int):
        """动态扩展嵌入层"""
        if new_num_entities <= self.num_entities and new_num_relations <= self.num_relations:
            return

        # 保存旧参数
        old_entity_emb = self.entity_embeddings.weight.data.clone()
        old_rel_base = self.relation_base.weight.data.clone()
        old_rel_delta = self.relation_delta.weight.data.clone()
        old_entity_importance = self.entity_importance.data.clone()
        old_relation_importance = self.relation_importance.data.clone()

        # 扩展实体嵌入
        if new_num_entities > self.num_entities:
            self.entity_embeddings = nn.Embedding(new_num_entities, self.embed_dim)
            self.entity_importance = nn.Parameter(torch.ones(new_num_entities), requires_grad=False)

            # 初始化新嵌入
            nn.init.xavier_uniform_(self.entity_embeddings.weight)

            # 复制旧参数
            self.entity_embeddings.weight.data[:self.num_entities] = old_entity_emb
            self.entity_importance.data[:self.num_entities] = old_entity_importance

        # 扩展关系嵌入
        if new_num_relations > self.num_relations:
            self.relation_base = nn.Embedding(new_num_relations, self.embed_dim)
            self.relation_delta = nn.Embedding(new_num_relations, self.embed_dim)
            self.relation_importance = nn.Parameter(torch.ones(new_num_relations), requires_grad=False)

            # 初始化新嵌入
            nn.init.xavier_uniform_(self.relation_base.weight)
            nn.init.uniform_(self.relation_delta.weight, 0.01, 0.1)

            # 复制旧参数
            self.relation_base.weight.data[:self.num_relations] = old_rel_base
            self.relation_delta.weight.data[:self.num_relations] = old_rel_delta
            self.relation_importance.data[:self.num_relations] = old_relation_importance

        # 更新模型大小
        self.num_entities = new_num_entities
        self.num_relations = new_num_relations

        # 移动到设备
        self.to(self.device)

        logger.info(f"模型扩展为 {new_num_entities} 个实体和 {new_num_relations} 个关系")


class SimpleContinualModel(nn.Module):
    """简化的支持持续学习的模型"""

    def __init__(self, num_entities: int, num_relations: int, embed_dim: int,
                 margin: float = 6.0, distill_weight: float = 0.5,
                 temperature: float = 4.0, device: str = 'cuda'):
        super(SimpleContinualModel, self).__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        self.margin = margin
        self.distill_weight = distill_weight
        self.temperature = temperature
        self.device = device

        # 当前模型
        self.current_model = SimpleBoxE(
            num_entities, num_relations, embed_dim, margin, device
        )

        # 历史模型（用于蒸馏）
        self.previous_model = None

        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def set_previous_model(self, previous_model):
        """设置用于蒸馏的历史模型"""
        self.previous_model = previous_model
        if self.previous_model:
            for param in self.previous_model.parameters():
                param.requires_grad = False

    def forward(self, triples):
        """前向传播"""
        return self.current_model(triples)

    def predict(self, heads, relations, tails):
        """预测函数"""
        return self.current_model.score_function(heads, relations, tails)

    def compute_main_loss(self, pos_triples, neg_triples):
        """计算主要任务损失"""
        pos_scores = self.current_model(pos_triples)

        # 处理负样本的形状
        if neg_triples.dim() == 3:
            batch_size, neg_ratio, _ = neg_triples.shape
            neg_triples = neg_triples.view(-1, 3)
        else:
            batch_size = pos_triples.shape[0]
            neg_ratio = neg_triples.shape[0] // batch_size if batch_size > 0 else 1

        neg_scores = self.current_model(neg_triples)

        # 重新整形负样本分数
        if batch_size > 0:
            neg_scores = neg_scores.view(batch_size, neg_ratio)
            pos_scores_expanded = pos_scores.unsqueeze(1).expand(-1, neg_ratio)
        else:
            neg_scores = neg_scores.unsqueeze(0)
            pos_scores_expanded = pos_scores.unsqueeze(0)

        # 边际损失
        margin_loss = torch.clamp(self.margin + neg_scores - pos_scores_expanded, min=0)
        main_loss = torch.mean(margin_loss)

        return main_loss

    def compute_distillation_loss(self, old_entities: torch.Tensor,
                                  old_relations: torch.Tensor,
                                  sample_triples: torch.Tensor = None):
        """计算蒸馏损失"""
        if self.previous_model is None:
            return torch.tensor(0.0, device=self.device)

        distill_loss = 0.0

        # 1. 硬蒸馏：实体和关系嵌入保持稳定
        if old_entities is not None and len(old_entities) > 0:
            valid_entities = old_entities[old_entities < self.current_model.num_entities]
            if len(valid_entities) > 0:
                current_entity_emb = self.current_model.entity_embeddings(valid_entities)
                with torch.no_grad():
                    previous_entity_emb = self.previous_model.entity_embeddings(valid_entities)

                # 重要性加权的蒸馏损失
                entity_weights = self.current_model.entity_importance[valid_entities]
                weighted_entity_loss = torch.mean(entity_weights.unsqueeze(-1) *
                                                  torch.sum((current_entity_emb - previous_entity_emb) ** 2, dim=-1,
                                                            keepdim=True))
                distill_loss += weighted_entity_loss

        if old_relations is not None and len(old_relations) > 0:
            valid_relations = old_relations[old_relations < self.current_model.num_relations]
            if len(valid_relations) > 0:
                current_rel_base = self.current_model.relation_base(valid_relations)
                current_rel_delta = self.current_model.relation_delta(valid_relations)

                with torch.no_grad():
                    previous_rel_base = self.previous_model.relation_base(valid_relations)
                    previous_rel_delta = self.previous_model.relation_delta(valid_relations)

                # 关系重要性加权
                relation_weights = self.current_model.relation_importance[valid_relations]
                base_loss = torch.mean(relation_weights.unsqueeze(-1) *
                                       torch.sum((current_rel_base - previous_rel_base) ** 2, dim=-1, keepdim=True))
                delta_loss = torch.mean(relation_weights.unsqueeze(-1) *
                                        torch.sum((current_rel_delta - previous_rel_delta) ** 2, dim=-1, keepdim=True))

                distill_loss += base_loss + delta_loss

        # 2. 软蒸馏：预测分布的一致性
        if sample_triples is not None and len(sample_triples) > 0:
            current_scores = self.current_model(sample_triples)
            with torch.no_grad():
                previous_scores = self.previous_model(sample_triples)

            # 温度缩放的KL散度
            current_probs = F.softmax(current_scores / self.temperature, dim=0)
            previous_probs = F.softmax(previous_scores / self.temperature, dim=0)

            # 避免log(0)
            epsilon = 1e-8
            current_log_probs = torch.log(current_probs + epsilon)

            soft_loss = self.kl_loss(current_log_probs, previous_probs) * (self.temperature ** 2)
            distill_loss += soft_loss

        return distill_loss

    def compute_total_loss(self, pos_triples, neg_triples, old_entities=None,
                           old_relations=None, sample_triples=None):
        """计算总损失"""
        # 主任务损失
        main_loss = self.compute_main_loss(pos_triples, neg_triples)

        # 蒸馏损失
        if self.previous_model is not None and old_entities is not None:
            distill_loss = self.compute_distillation_loss(old_entities, old_relations, sample_triples)
        else:
            distill_loss = torch.tensor(0.0, device=self.device)

        # 总损失
        total_loss = main_loss + self.distill_weight * distill_loss

        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'distill_loss': distill_loss
        }

    def expand_model(self, new_num_entities: int, new_num_relations: int):
        """扩展模型以支持新实体和关系"""
        self.current_model.expand_embeddings(new_num_entities, new_num_relations)
        self.num_entities = new_num_entities
        self.num_relations = new_num_relations

    def copy_model_for_distillation(self):
        """复制当前模型用于下一轮蒸馏"""
        previous_model = SimpleBoxE(
            self.num_entities, self.num_relations, self.embed_dim,
            self.margin, self.device
        )

        previous_model.load_state_dict(self.current_model.state_dict())
        previous_model.to(self.device)

        return previous_model

    def save_checkpoint(self, filepath: str):
        """保存模型检查点"""
        checkpoint = {
            'current_model_state': self.current_model.state_dict(),
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'embed_dim': self.embed_dim,
            'margin': self.margin,
            'distill_weight': self.distill_weight,
            'temperature': self.temperature
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """加载模型检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)

        # 更新模型配置
        self.num_entities = checkpoint['num_entities']
        self.num_relations = checkpoint['num_relations']
        self.embed_dim = checkpoint['embed_dim']
        self.margin = checkpoint['margin']
        self.distill_weight = checkpoint['distill_weight']
        self.temperature = checkpoint['temperature']

        # 重新创建模型
        self.current_model = SimpleBoxE(
            self.num_entities, self.num_relations, self.embed_dim,
            self.margin, self.device
        )

        # 加载模型参数
        self.current_model.load_state_dict(checkpoint['current_model_state'])
        self.current_model.to(self.device)


# 为了保持与原代码的兼容性，提供别名
HierarchicalSTCKGE = SimpleContinualModel