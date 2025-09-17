#!/usr/bin/env python3
"""
Layerwise Distillation Training Script
层级知识蒸馏训练脚本

基于Fisher信息矩阵的权重分配策略，从llama3教师模型蒸馏知识到轻量级学生模型
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
import time
from typing import Dict, List, Tuple
from pathlib import Path
import logging
from tqdm import tqdm

from layerwise_distillation import (
    DistillationConfig,
    StudentRecommenderModel,
    TeacherModelProxy,
    FisherInformationCalculator,
    LayerwiseDistillationLoss
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RecommendationDataset(Dataset):
    """推荐数据集"""
    
    def __init__(self, data_file: str, max_samples: int = 1000):
        self.data = self._load_recommendation_data(data_file, max_samples)
        self.tokenizer = SimpleTokenizer()
        
    def _load_recommendation_data(self, data_file: str, max_samples: int) -> List[Dict]:
        """加载推荐数据"""
        logger.info(f"加载推荐数据: {data_file}")
        
        if not os.path.exists(data_file):
            logger.warning(f"数据文件不存在，生成模拟数据: {data_file}")
            return self._generate_mock_data(max_samples)
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换为训练格式
            training_data = []
            for item in data[:max_samples]:
                if isinstance(item, dict):
                    training_data.append(self._convert_to_training_format(item))
            
            logger.info(f"成功加载 {len(training_data)} 条训练数据")
            return training_data
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return self._generate_mock_data(max_samples)
    
    def _convert_to_training_format(self, item: Dict) -> Dict:
        """将原始数据转换为训练格式"""
        # 提取用户画像
        user_profile = ""
        if 'user_analysis' in item:
            user_profile = str(item['user_analysis'])
        elif 'user_profile' in item:
            user_profile = str(item['user_profile'])
        
        # 提取推荐商品
        recommendations = item.get('recommendations', [])
        if recommendations:
            rec_text = " ".join([str(rec) for rec in recommendations[:3]])
        else:
            rec_text = "暂无推荐"
        
        # 构建输入文本
        input_text = f"用户画像: {user_profile} 推荐商品: {rec_text}"
        
        # 模拟标签（实际应用中应该有真实的用户反馈）
        label = 1 if len(recommendations) > 0 else 0
        
        return {
            'input_text': input_text[:512],  # 限制长度
            'label': label,
            'user_profile': user_profile,
            'recommendations': recommendations
        }
    
    def _generate_mock_data(self, num_samples: int) -> List[Dict]:
        """生成模拟训练数据"""
        logger.info(f"生成 {num_samples} 条模拟训练数据")
        
        mock_data = []
        categories = ['电子产品', '美妆护肤', '服装饰品', '家居用品', '图书音像']
        
        for i in range(num_samples):
            category = np.random.choice(categories)
            
            if category == '电子产品':
                user_profile = "喜欢科技产品，追求性能和创新"
                items = ['智能手机', '平板电脑', '无线耳机', '智能手表']
            elif category == '美妆护肤':
                user_profile = "注重护肤保养，偏爱天然成分"
                items = ['面膜', '精华液', '防晒霜', '口红']
            elif category == '服装饰品':
                user_profile = "时尚达人，喜欢潮流单品"
                items = ['连衣裙', '运动鞋', '手表', '包包']
            elif category == '家居用品':
                user_profile = "注重生活品质，喜欢实用美观的家居"
                items = ['收纳盒', '餐具套装', '床上用品', '装饰画']
            else:
                user_profile = "爱好阅读，喜欢文学和学习"
                items = ['小说', '专业书籍', '电子书', 'audiobook']
            
            selected_items = np.random.choice(items, size=2, replace=False)
            input_text = f"用户画像: {user_profile} 推荐商品: {' '.join(selected_items)}"
            
            mock_data.append({
                'input_text': input_text,
                'label': 1,  # 假设都是正样本
                'user_profile': user_profile,
                'recommendations': selected_items.tolist()
            })
        
        return mock_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 分词编码
        input_ids = self.tokenizer.encode(item['input_text'])
        attention_mask = [1] * len(input_ids)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(item['label'], dtype=torch.float),
            'input_text': item['input_text']
        }

class SimpleTokenizer:
    """简单的分词器"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.build_vocab()
    
    def build_vocab(self):
        """构建词汇表"""
        # 特殊token
        self.char_to_id['[PAD]'] = 0
        self.char_to_id['[UNK]'] = 1
        self.char_to_id['[CLS]'] = 2
        self.char_to_id['[SEP]'] = 3
        
        # ASCII字符
        for i in range(256):
            char = chr(i)
            if char not in self.char_to_id:
                self.char_to_id[char] = len(self.char_to_id)
        
        # 常用中文字符（简化处理）
        common_chars = "的了是我你他她它们这那有在不为和与或但因为所以如果那么时候地方人们工作学习生活"
        for char in common_chars:
            if char not in self.char_to_id:
                self.char_to_id[char] = len(self.char_to_id)
    
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """编码文本"""
        token_ids = [self.char_to_id['[CLS]']]
        
        for char in text[:max_length-2]:
            token_id = self.char_to_id.get(char, self.char_to_id['[UNK]'])
            token_ids.append(token_id)
        
        token_ids.append(self.char_to_id['[SEP]'])
        
        # 填充到固定长度
        while len(token_ids) < max_length:
            token_ids.append(self.char_to_id['[PAD]'])
        
        return token_ids[:max_length]

class DistillationTrainer:
    """蒸馏训练器"""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self.student_model = StudentRecommenderModel(config).to(self.device)
        self.teacher_proxy = TeacherModelProxy(config)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # 训练统计
        self.training_stats = {
            'epoch_losses': [],
            'task_losses': [],
            'distill_losses': [],
            'layer_losses': []
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """执行蒸馏训练"""
        logger.info("开始蒸馏训练...")
        
        # 计算Fisher权重
        sample_data = self._get_sample_data(train_loader)
        fisher_calc = FisherInformationCalculator(self.config)
        fisher_weights = fisher_calc.calculate_fisher_weights(sample_data).to(self.device)
        
        # 初始化损失函数
        distill_loss_fn = LayerwiseDistillationLoss(self.config, fisher_weights).to(self.device)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # 训练
            train_loss = self._train_epoch(train_loader, distill_loss_fn)
            
            # 验证
            if val_loader:
                val_loss = self._validate_epoch(val_loader, distill_loss_fn)
                logger.info(f"Validation Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_model(f"best_student_model_epoch_{epoch+1}.pt")
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录统计信息
            self.training_stats['epoch_losses'].append(train_loss)
            
            logger.info(f"Epoch {epoch+1} completed. Train Loss: {train_loss:.4f}")
        
        # 保存最终模型
        self._save_model("final_student_model.pt")
        self._save_training_stats()
        
        logger.info("蒸馏训练完成!")
    
    def _train_epoch(self, train_loader: DataLoader, loss_fn: LayerwiseDistillationLoss) -> float:
        """训练一个epoch"""
        self.student_model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            input_texts = batch['input_text']
            
            # 学生模型前向传播
            student_outputs = self.student_model(input_ids, attention_mask)
            
            # 获取教师模型输出
            with torch.no_grad():
                teacher_outputs = self.teacher_proxy.get_teacher_outputs(input_texts)
                # 移动到正确设备
                for key in teacher_outputs:
                    if isinstance(teacher_outputs[key], torch.Tensor):
                        teacher_outputs[key] = teacher_outputs[key].to(self.device)
                    elif isinstance(teacher_outputs[key], list):
                        teacher_outputs[key] = [t.to(self.device) for t in teacher_outputs[key]]
            
            # 计算损失
            loss_dict = loss_fn(student_outputs, teacher_outputs, labels)
            loss = loss_dict['total_loss']
            
            # 反向传播
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            loss.backward()
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Task': f"{loss_dict['task_loss'].item():.4f}",
                'Distill': f"{loss_dict['distill_loss'].item():.4f}"
            })
            
            # 记录详细统计
            self.training_stats['task_losses'].append(loss_dict['task_loss'].item())
            self.training_stats['distill_losses'].append(loss_dict['distill_loss'].item())
            self.training_stats['layer_losses'].append(loss_dict['layer_distill_loss'].item())
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader, loss_fn: LayerwiseDistillationLoss) -> float:
        """验证一个epoch"""
        self.student_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                input_texts = batch['input_text']
                
                # 前向传播
                student_outputs = self.student_model(input_ids, attention_mask)
                teacher_outputs = self.teacher_proxy.get_teacher_outputs(input_texts)
                
                # 移动到设备
                for key in teacher_outputs:
                    if isinstance(teacher_outputs[key], torch.Tensor):
                        teacher_outputs[key] = teacher_outputs[key].to(self.device)
                    elif isinstance(teacher_outputs[key], list):
                        teacher_outputs[key] = [t.to(self.device) for t in teacher_outputs[key]]
                
                # 计算损失
                loss_dict = loss_fn(student_outputs, teacher_outputs, labels)
                total_loss += loss_dict['total_loss'].item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _get_sample_data(self, train_loader: DataLoader) -> List[Dict]:
        """获取样本数据用于Fisher权重计算"""
        sample_data = []
        for batch in train_loader:
            for i, text in enumerate(batch['input_text'][:5]):  # 只取前5个样本
                sample_data.append({
                    'user_profile': text,
                    'candidate_items': ['item1', 'item2', 'item3']  # 简化处理
                })
            break  # 只要第一个batch
        return sample_data
    
    def _save_model(self, filename: str):
        """保存模型"""
        save_path = Path("../results") / filename
        save_path.parent.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': self.student_model.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, save_path)
        
        logger.info(f"模型已保存: {save_path}")
    
    def _save_training_stats(self):
        """保存训练统计"""
        stats_path = Path("../results") / "training_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练统计已保存: {stats_path}")

def create_data_loaders(config: DistillationConfig) -> Tuple[DataLoader, DataLoader]:
    """创建数据加载器"""
    # 使用已有的实验结果作为训练数据
    train_data_file = "../results/multi_category_recommendations_20250909_170318.json"
    
    # 创建数据集
    full_dataset = RecommendationDataset(train_data_file, max_samples=200)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # 在某些环境中需要设为0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    return train_loader, val_loader

def main():
    """主函数"""
    logger.info("启动层级知识蒸馏训练...")
    
    # 配置
    config = DistillationConfig(
        student_hidden_dim=512,  # 减小模型尺寸
        num_layers=8,            # 减少层数
        num_heads=8,             # 减少注意力头数
        batch_size=4,            # 减小批次大小
        num_epochs=3,            # 减少训练轮数
        learning_rate=1e-4
    )
    
    try:
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(config)
        
        # 初始化训练器
        trainer = DistillationTrainer(config)
        
        # 开始训练
        trainer.train(train_loader, val_loader)
        
        logger.info("✅ 蒸馏训练成功完成!")
        
    except Exception as e:
        logger.error(f"❌ 训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()
