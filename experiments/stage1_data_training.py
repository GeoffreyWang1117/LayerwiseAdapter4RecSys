#!/usr/bin/env python3
"""
阶段1：诚实的数据加载和基础模型训练
确保数据完全真实，模型训练稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HonestDataValidator:
    """数据诚实性验证器"""
    
    @staticmethod
    def validate_real_data(df, source_name="Amazon"):
        """验证数据是真实的，不是模拟的"""
        logger.info(f"🔍 验证{source_name}数据真实性...")
        
        # 基本统计检查
        logger.info(f"数据形状: {df.shape}")
        logger.info(f"列名: {list(df.columns)}")
        
        # 检查文本多样性（真实数据应该有很高的多样性）
        if 'text' in df.columns:
            unique_texts = df['text'].nunique()
            total_texts = len(df)
            diversity_ratio = unique_texts / total_texts
            
            logger.info(f"文本唯一性: {unique_texts:,}/{total_texts:,} = {diversity_ratio:.3f}")
            
            if diversity_ratio < 0.5:
                logger.warning(f"文本多样性低 ({diversity_ratio:.3f})，可能存在重复或模拟数据")
            else:
                logger.info("✅ 文本多样性验证通过")
            
            # 检查文本长度分布（真实数据应该有自然的长度分布）
            text_lengths = df['text'].str.len()
            logger.info(f"文本长度统计:")
            logger.info(f"  最小: {text_lengths.min()}")
            logger.info(f"  最大: {text_lengths.max()}")
            logger.info(f"  平均: {text_lengths.mean():.1f}")
            logger.info(f"  中位数: {text_lengths.median():.1f}")
            logger.info(f"  标准差: {text_lengths.std():.1f}")
        
        # 检查评分分布
        if 'rating' in df.columns:
            rating_dist = df['rating'].value_counts().sort_index()
            logger.info(f"评分分布:")
            for rating, count in rating_dist.items():
                percentage = count / len(df) * 100
                logger.info(f"  {rating}星: {count:,} ({percentage:.1f}%)")
        
        # 检查时间戳（如果有）
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
            time_range = timestamps.max() - timestamps.min()
            logger.info(f"时间跨度: {time_range}")
        
        logger.info("✅ 数据真实性验证完成")
        return True

class StableDataLoader:
    """稳定的真实数据加载器"""
    
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = Path(data_dir)
        self.vocab = {}
        self.vocab_size = 0
        self.validator = HonestDataValidator()
        
    def load_verified_amazon_data(self, category="Electronics", max_samples=30000, min_text_length=10):
        """加载并验证真实Amazon数据"""
        logger.info(f"📊 加载真实Amazon {category}数据...")
        
        reviews_file = self.data_dir / "amazon" / f"{category}_reviews.parquet"
        
        if not reviews_file.exists():
            raise FileNotFoundError(f"真实数据文件不存在: {reviews_file}")
        
        # 加载数据
        df = pd.read_parquet(reviews_file)
        logger.info(f"原始数据: {len(df):,} 条记录")
        
        # 验证数据真实性
        self.validator.validate_real_data(df, f"Amazon {category}")
        
        # 数据清洗（保持严格标准）
        original_size = len(df)
        
        # 必须字段检查
        required_fields = ['text', 'rating']
        for field in required_fields:
            if field not in df.columns:
                raise ValueError(f"缺少必要字段: {field}")
        
        # 数据质量过滤
        df = df.dropna(subset=required_fields)
        df = df[df['rating'].between(1, 5)]
        df = df[df['text'].str.len() >= min_text_length]
        df = df[df['text'].str.len() <= 1000]  # 过滤异常长文本
        
        # 去除明显重复
        df = df.drop_duplicates(subset=['text'], keep='first')
        
        logger.info(f"质量过滤: {original_size:,} -> {len(df):,} ({len(df)/original_size:.1%}保留)")
        
        # 平衡采样
        positive_samples = df[df['rating'] >= 4]
        negative_samples = df[df['rating'] <= 2]
        
        logger.info(f"正例候选: {len(positive_samples):,}")
        logger.info(f"负例候选: {len(negative_samples):,}")
        
        # 确保有足够样本
        min_samples_per_class = max_samples // 2
        if len(positive_samples) < min_samples_per_class:
            logger.warning(f"正例样本不足: {len(positive_samples)} < {min_samples_per_class}")
            min_samples_per_class = len(positive_samples)
        
        if len(negative_samples) < min_samples_per_class:
            logger.warning(f"负例样本不足: {len(negative_samples)} < {min_samples_per_class}")
            min_samples_per_class = len(negative_samples)
        
        # 随机采样
        df_positive = positive_samples.sample(n=min_samples_per_class, random_state=42)
        df_negative = negative_samples.sample(n=min_samples_per_class, random_state=42)
        
        # 合并和随机化
        df_balanced = pd.concat([df_positive, df_negative]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 提取文本和标签
        texts = df_balanced['text'].astype(str).tolist()
        labels = (df_balanced['rating'] >= 4).astype(int).tolist()
        
        # 最终验证
        logger.info(f"✅ 最终数据集:")
        logger.info(f"   总样本: {len(texts):,}")
        logger.info(f"   正例: {sum(labels):,} ({sum(labels)/len(labels):.1%})")
        logger.info(f"   负例: {len(labels)-sum(labels):,} ({(len(labels)-sum(labels))/len(labels):.1%})")
        logger.info(f"   平均文本长度: {np.mean([len(t.split()) for t in texts]):.1f} 词")
        
        return texts, labels, df_balanced
    
    def create_stable_vocabulary(self, texts, max_vocab_size=15000, min_freq=2):
        """创建稳定的词汇表"""
        logger.info("📝 构建稳定词汇表...")
        
        # 文本预处理和词频统计
        word_freq = defaultdict(int)
        total_words = 0
        
        for text in texts:
            # 简单但有效的文本清理
            clean_text = text.lower()
            # 保留基本标点但分离
            for punct in '.,!?;:"()[]{}':
                clean_text = clean_text.replace(punct, f' {punct} ')
            
            words = clean_text.split()
            for word in words:
                word = word.strip()
                if word and len(word) <= 20:  # 过滤异常长词
                    word_freq[word] += 1
                    total_words += 1
        
        logger.info(f"词汇统计: {len(word_freq):,} 唯一词, {total_words:,} 总词数")
        
        # 构建词汇表
        special_tokens = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK>']
        vocab_words = special_tokens.copy()
        
        # 按频率排序，添加高频词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 统计覆盖率
        covered_freq = 0
        for word, freq in sorted_words:
            if len(vocab_words) >= max_vocab_size:
                break
            if freq >= min_freq:
                vocab_words.append(word)
                covered_freq += freq
        
        coverage = covered_freq / total_words
        
        self.vocab = {word: idx for idx, word in enumerate(vocab_words)}
        self.vocab_size = len(vocab_words)
        
        logger.info(f"✅ 词汇表构建完成:")
        logger.info(f"   词汇表大小: {self.vocab_size:,}")
        logger.info(f"   词汇覆盖率: {coverage:.1%}")
        logger.info(f"   OOV率: {1-coverage:.1%}")
        
        return self.vocab
    
    def tokenize_texts(self, texts, max_seq_length=256):
        """文本tokenization"""
        logger.info(f"🔤 文本tokenization (序列长度: {max_seq_length})...")
        
        if not self.vocab:
            raise ValueError("词汇表未构建，请先调用create_stable_vocabulary")
        
        tokenized_texts = []
        
        for i, text in enumerate(texts):
            if i % 5000 == 0:
                logger.info(f"   处理进度: {i:,}/{len(texts):,}")
            
            # 预处理
            clean_text = text.lower()
            for punct in '.,!?;:"()[]{}':
                clean_text = clean_text.replace(punct, f' {punct} ')
            
            # Tokenize
            tokens = [self.vocab['<CLS>']]
            words = clean_text.split()
            
            for word in words[:max_seq_length-2]:  # 保留位置给特殊token
                word = word.strip()
                if word:
                    token_id = self.vocab.get(word, self.vocab['<UNK>'])
                    tokens.append(token_id)
            
            tokens.append(self.vocab['<SEP>'])
            
            # 填充或截断
            if len(tokens) < max_seq_length:
                tokens.extend([self.vocab['<PAD>']] * (max_seq_length - len(tokens)))
            else:
                tokens = tokens[:max_seq_length]
            
            tokenized_texts.append(tokens)
        
        logger.info(f"✅ Tokenization完成: {len(tokenized_texts)} 样本")
        
        return torch.tensor(tokenized_texts, dtype=torch.long)

class StableTransformer(nn.Module):
    """稳定的Transformer模型"""
    
    def __init__(self, vocab_size=15000, d_model=512, nhead=8, num_layers=12, max_seq_length=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
        # 权重初始化
        self.apply(self._init_weights)
        
        # 记录模型信息
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"📊 模型参数统计:")
        logger.info(f"   总参数: {total_params:,}")
        logger.info(f"   可训练参数: {trainable_params:,}")
        logger.info(f"   模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    def _init_weights(self, module):
        """Xavier初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                module.weight.data[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None):
        """前向传播"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 创建attention mask
        if attention_mask is None:
            attention_mask = (input_ids != 0)  # PAD token is 0
        
        # 转换为transformer期望的格式 (True表示要attend的位置)
        src_key_padding_mask = ~attention_mask  # Invert for transformer
        
        # 嵌入
        token_embeds = self.token_embedding(input_ids)
        
        # 位置嵌入
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # 组合嵌入
        hidden_states = self.embedding_dropout(token_embeds + position_embeds)
        
        # Transformer编码
        try:
            encoded = self.transformer(hidden_states, src_key_padding_mask=src_key_padding_mask)
        except Exception as e:
            logger.warning(f"Transformer编码出错，使用无mask模式: {e}")
            encoded = self.transformer(hidden_states)
        
        # 使用CLS token进行分类
        cls_output = encoded[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_output)
        
        return {
            'logits': logits,
            'hidden_states': encoded,
            'attention_mask': attention_mask
        }

class StableTrainer:
    """稳定的模型训练器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.training_history = []
        
    def train_stable(self, train_data, train_labels, val_data, val_labels, 
                    max_epochs=10, batch_size=32, learning_rate=2e-4, patience=3):
        """稳定训练模型"""
        logger.info("🏋️ 开始稳定训练...")
        
        # 优化器和调度器
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,
            eps=1e-6
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=max_epochs,
            steps_per_epoch=len(train_data) // batch_size + 1,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100
        )
        
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            # 打乱训练数据
            indices = torch.randperm(len(train_data))
            
            num_batches = 0
            for i in range(0, len(train_data), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_data = train_data[batch_indices].to(self.device)
                batch_labels = torch.tensor([train_labels[idx] for idx in batch_indices], 
                                          dtype=torch.long).to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(batch_data)
                loss = criterion(outputs['logits'], batch_labels)
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                # 统计
                train_loss += loss.item()
                num_batches += 1
                
                predictions = torch.argmax(outputs['logits'], dim=1)
                train_correct += (predictions == batch_labels).sum().item()
                train_total += batch_labels.size(0)
                
                # 定期输出进度
                if num_batches % 50 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(f"   Batch {num_batches}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
            
            # 验证阶段
            val_acc, val_loss = self.evaluate(val_data, val_labels, batch_size)
            train_acc = train_correct / train_total if train_total > 0 else 0
            avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
            
            # 记录历史
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': scheduler.get_last_lr()[0]
            }
            self.training_history.append(epoch_stats)
            
            logger.info(f"Epoch {epoch+1}/{max_epochs}:")
            logger.info(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")
            logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            logger.info(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 保存最佳模型
                self.save_checkpoint(f"best_model_epoch_{epoch+1}.pt", epoch_stats)
                logger.info(f"  ✅ 新的最佳验证准确率: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                logger.info(f"  ⏳ 耐心计数: {patience_counter}/{patience}")
                
            if patience_counter >= patience:
                logger.info(f"  🛑 早停于epoch {epoch+1}")
                break
        
        logger.info(f"🎯 训练完成! 最佳验证准确率: {best_val_acc:.4f}")
        return best_val_acc, self.training_history
    
    def evaluate(self, data, labels, batch_size=64):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i+batch_size].to(self.device)
                batch_labels = torch.tensor(labels[i:i+batch_size], dtype=torch.long).to(self.device)
                
                outputs = self.model(batch_data)
                loss = criterion(outputs['logits'], batch_labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                predictions = torch.argmax(outputs['logits'], dim=1)
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return accuracy, avg_loss
    
    def save_checkpoint(self, filename, epoch_stats):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch_stats': epoch_stats,
            'training_history': self.training_history,
            'model_config': {
                'vocab_size': self.model.token_embedding.num_embeddings,
                'd_model': self.model.d_model,
                'num_layers': self.model.num_layers,
                'max_seq_length': self.model.max_seq_length
            }
        }
        
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        torch.save(checkpoint, checkpoint_dir / filename)
        logger.info(f"💾 检查点已保存: {filename}")

def run_stage1_data_and_training():
    """阶段1：数据加载和基础训练"""
    logger.info("🚀 开始阶段1：数据加载和基础训练")
    
    # 1. 数据加载
    logger.info("📁 步骤1: 加载真实数据")
    data_loader = StableDataLoader()
    
    try:
        texts, labels, raw_df = data_loader.load_verified_amazon_data(
            category="Electronics", 
            max_samples=20000
        )
        
        # 保存原始数据摘要
        data_summary = {
            'total_samples': len(texts),
            'positive_samples': sum(labels),
            'negative_samples': len(labels) - sum(labels),
            'avg_text_length': np.mean([len(t.split()) for t in texts]),
            'data_source': 'Amazon Electronics Reviews',
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = Path("results") / "stage1_data_summary.json"
        summary_file.parent.mkdir(exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump(data_summary, f, indent=2)
        
        logger.info(f"📋 数据摘要已保存: {summary_file}")
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise
    
    # 2. 词汇表构建
    logger.info("📚 步骤2: 构建词汇表")
    vocab = data_loader.create_stable_vocabulary(texts, max_vocab_size=12000)
    
    # 3. Tokenization
    logger.info("🔤 步骤3: 文本tokenization")
    input_ids = data_loader.tokenize_texts(texts, max_seq_length=256)
    
    # 4. 数据分割
    logger.info("✂️ 步骤4: 数据分割")
    n_total = len(texts)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val
    
    # 随机分割
    indices = torch.randperm(n_total)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    train_data = input_ids[train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_data = input_ids[val_indices]
    val_labels = [labels[i] for i in val_indices]
    test_data = input_ids[test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    logger.info(f"✅ 数据分割完成:")
    logger.info(f"   训练集: {len(train_data):,} 样本")
    logger.info(f"   验证集: {len(val_data):,} 样本")
    logger.info(f"   测试集: {len(test_data):,} 样本")
    
    # 5. 模型创建
    logger.info("🏗️ 步骤5: 创建Transformer模型")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = StableTransformer(
        vocab_size=data_loader.vocab_size,
        d_model=512,
        nhead=8,
        num_layers=12,
        max_seq_length=256,
        dropout=0.1
    ).to(device)
    
    # 6. 模型训练
    logger.info("🏋️ 步骤6: 模型训练")
    trainer = StableTrainer(model, device)
    
    best_val_acc, history = trainer.train_stable(
        train_data, train_labels, 
        val_data, val_labels,
        max_epochs=8,
        batch_size=24,
        learning_rate=1e-4,
        patience=3
    )
    
    # 7. 测试集评估
    logger.info("🧪 步骤7: 测试集最终评估")
    test_acc, test_loss = trainer.evaluate(test_data, test_labels)
    
    logger.info(f"🎯 最终测试结果:")
    logger.info(f"   测试准确率: {test_acc:.4f}")
    logger.info(f"   测试损失: {test_loss:.4f}")
    
    # 8. 保存结果
    stage1_results = {
        'data_summary': data_summary,
        'model_config': {
            'vocab_size': data_loader.vocab_size,
            'd_model': 512,
            'num_layers': 12,
            'max_seq_length': 256
        },
        'training_results': {
            'best_val_acc': best_val_acc,
            'final_test_acc': test_acc,
            'final_test_loss': test_loss,
            'training_history': history
        },
        'stage1_completed': datetime.now().isoformat()
    }
    
    results_file = Path("results") / "stage1_complete_results.json"
    with open(results_file, 'w') as f:
        json.dump(stage1_results, f, indent=2, default=str)
    
    logger.info(f"💾 阶段1结果已保存: {results_file}")
    logger.info("🎉 阶段1完成！模型已训练稳定，准备进入阶段2")
    
    return {
        'model': model,
        'trainer': trainer,
        'data_splits': {
            'train': (train_data, train_labels),
            'val': (val_data, val_labels), 
            'test': (test_data, test_labels)
        },
        'data_loader': data_loader,
        'results': stage1_results
    }

if __name__ == "__main__":
    # 运行阶段1
    stage1_outputs = run_stage1_data_and_training()
    
    logger.info("✅ 阶段1成功完成！")
    logger.info("🔜 准备运行阶段2: 层重要性分析")
