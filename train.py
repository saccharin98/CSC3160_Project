# train.py
"""
Transformer语音情感识别训练脚本

完整训练流程:
1. 加载数据集
2. 创建模型
3. 定义损失函数和优化器
4. 训练循环
5. 验证和保存
6. TensorBoard可视化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import json

from config import config
from dataset import RAVDESSDataset
from model import create_model


# ============================================
# 第1部分: 训练器类
# ============================================

class Trainer:
    """
    训练器 - 管理整个训练过程
    """
    
    def __init__(self, model, train_loader, val_loader, device):
        """
        参数:
            model: Transformer模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 运行设备 (cuda/cpu)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 损失函数: 交叉熵
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器: AdamW (带权重衰减的Adam)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器: ReduceLROnPlateau
        # 当验证损失不下降时，自动降低学习率
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        
        # TensorBoard记录器
        self.writer = SummaryWriter(log_dir=config.LOG_DIR)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # 最佳模型记录
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        print(f"✓ 训练器初始化完成")
        print(f"  设备: {device}")
        print(f"  优化器: AdamW (lr={config.LEARNING_RATE})")
        print(f"  损失函数: CrossEntropyLoss")
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        参数:
            epoch: 当前epoch编号
        
        返回:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        self.model.train()  # 设置为训练模式
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{config.NUM_EPOCHS} [Train]')
        
        for batch_idx, (features, labels) in enumerate(pbar):
            # 1. 数据移到设备
            features = features.to(self.device)  # (batch, time, feature_dim)
            labels = labels.to(self.device)      # (batch,)
            
            # 2. 前向传播
            outputs = self.model(features)  # (batch, num_classes)
            loss = self.criterion(outputs, labels)
            
            # 3. 反向传播
            self.optimizer.zero_grad()  # 清空梯度
            loss.backward()             # 计算梯度
            
            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()       # 更新参数
            
            # 4. 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # 5. 更新进度条
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
        
        # 计算平均值
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100.0 * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, epoch):
        """
        验证一个epoch
        
        参数:
            epoch: 当前epoch编号
        
        返回:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        self.model.eval()  # 设置为评估模式
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 进度条
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{config.NUM_EPOCHS} [Val]  ')
        
        with torch.no_grad():  # 不计算梯度
            for batch_idx, (features, labels) in enumerate(pbar):
                # 1. 数据移到设备
                features = features.to(self.device)  # (batch, time, feature_dim)
                labels = labels.to(self.device)      # (batch,)
                
                # 2. 前向传播
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # 3. 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # 4. 更新进度条
                current_loss = total_loss / (batch_idx + 1)
                current_acc = 100.0 * correct / total
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%'
                })
        
        # 计算平均值
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        avg_acc = 100.0 * correct / total if total > 0 else 0
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """
        保存检查点
        
        参数:
            epoch: epoch编号
            val_acc: 验证准确率
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'config': vars(config)
        }
        
        # 保存最新检查点
        latest_path = config.CHECKPOINT_DIR / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = config.CHECKPOINT_DIR / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ 保存最佳模型 (准确率: {val_acc:.2f}%)")
    
    def train(self):
        """
        完整训练流程
        """
        print("\n" + "=" * 70)
        print("开始训练")
        print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(1, config.NUM_EPOCHS + 1):
            # 1. 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 2. 验证
            val_loss, val_acc = self.validate(epoch)
            
            # 3. 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 4. 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # 5. TensorBoard记录
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # 6. 打印摘要
            print(f"\nEpoch {epoch}/{config.NUM_EPOCHS} 摘要:")
            print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"  学习率: {current_lr:.6f}")
            
            # 7. 保存检查点
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # 8. 早停检查 (可选)
            if config.EARLY_STOPPING and epoch - self.best_epoch > config.EARLY_STOPPING_PATIENCE:
                print(f"\n早停! 最佳epoch: {self.best_epoch}, 准确率: {self.best_val_acc:.2f}%")
                break
        
        # 训练完成
        total_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("训练完成!")
        print("=" * 70)
        print(f"总耗时: {total_time / 60:.2f} 分钟")
        print(f"最佳Epoch: {self.best_epoch}")
        print(f"最佳准确率: {self.best_val_acc:.2f}%")
        
        # 保存训练历史
        self._save_history()
        
        # 关闭TensorBoard
        self.writer.close()
    
    def _save_history(self):
        """
        保存训练历史到JSON
        """
        history_path = config.CHECKPOINT_DIR / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"✓ 训练历史保存到: {history_path}")


# ============================================
# 第2部分: 主函数
# ============================================

def main():
    """
    主函数 - 训练入口
    """
    print("=" * 70)
    print("Transformer语音情感识别 - 训练")
    print("=" * 70)
    
    # 1. 设置随机种子 (保证可复现)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)
    
    # 2. 设置设备
    device = torch.device(
        'cuda' if config.DEVICE == 'cuda' and torch.cuda.is_available() else 'cpu'
    )
    print(f"\n设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 3. 加载数据集
    print("\n加载数据集...")
    full_dataset = RAVDESSDataset(config.DATA_PATH)
    
    # 4. 划分训练集和验证集
    train_size = int(config.TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    print(f"✓ 数据集加载完成")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    
    # 5. 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"✓ 数据加载器创建完成")
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")
    
    # 6. 创建模型
    print("\n创建模型...")
    model = create_model()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ 模型创建完成")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 7. 创建训练器
    trainer = Trainer(model, train_loader, val_loader, device)
    
    # 8. 开始训练
    trainer.train()
    
    print("\n" + "=" * 70)
    print("全部完成!")
    print("=" * 70)
    print(f"\n查看训练曲线:")
    print(f"  tensorboard --logdir={config.LOG_DIR}")


# ============================================
# 第3部分: 程序入口
# ============================================

if __name__ == '__main__':
    main()