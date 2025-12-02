# model.py
"""
语音情感识别的Transformer模型

架构:
1. 输入: Mel频谱 (batch, time, n_mels)
2. 位置编码
3. Transformer编码器层
4. 分类头
5. 输出: 情感类别 (batch, num_classes)
"""

import torch
import torch.nn as nn
import math
from config import config


# ============================================
# 第1部分: 位置编码
# ============================================

class PositionalEncoding(nn.Module):
    """
    位置编码 - 让模型知道时间顺序
    
    为什么需要?
    - Transformer没有循环结构，不知道顺序
    - 位置编码告诉模型"第1帧"和"第100帧"的位置信息
    
    使用正弦/余弦函数:
    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        参数:
            d_model: 模型维度 (256)
            max_len: 最大序列长度 (5000足够了)
            dropout: Dropout比例
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 分母项 (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 计算正弦和余弦
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos
        
        # 增加batch维度 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 注册为buffer（不是参数，但会保存到模型）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        参数:
            x: (batch, seq_len, d_model)
        
        返回:
            加上位置编码后的x
        """
        # 取出对应长度的位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================
# 第2部分: Transformer编码器层
# ============================================

class TransformerEncoderLayer(nn.Module):
    """
    单个Transformer编码器层
    
    结构:
    1. 多头自注意力 (Multi-Head Self-Attention)
    2. 残差连接 + LayerNorm
    3. 前馈网络 (Feed-Forward Network)
    4. 残差连接 + LayerNorm
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        参数:
            d_model: 模型维度 (256)
            num_heads: 注意力头数 (8)
            d_ff: 前馈网络维度 (1024)
            dropout: Dropout比例
        """
        super().__init__()
        
        # 1. 多头自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 输入格式: (batch, seq, feature)
        )
        
        # 2. 前馈网络 (两层全连接)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 3. LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 4. Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        参数:
            x: (batch, seq_len, d_model)
            mask: 注意力掩码（可选）
        
        返回:
            输出: (batch, seq_len, d_model)
        """
        # 1. 多头自注意力 + 残差连接 + LayerNorm
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


# ============================================
# 第3部分: 完整的Transformer模型
# ============================================

class SpeechEmotionTransformer(nn.Module):
    """
    语音情感识别的Transformer模型
    
    完整流程:
    输入Mel频谱 → 线性投影 → 位置编码 → Transformer编码器 → 全局池化 → 分类
    """
    
    def __init__(
        self,
        input_dim=80,           # Mel频谱维度
        d_model=256,            # 模型维度
        num_heads=8,            # 注意力头数
        num_layers=6,           # Transformer层数
        d_ff=1024,              # 前馈网络维度
        num_classes=config.NUM_CLASSES,  # 情感类别数
        dropout=0.1,            # Dropout比例
        max_len=5000            # 最大序列长度
    ):
        super().__init__()
        
        # 保存配置
        self.d_model = d_model
        self.num_classes = num_classes
        
        # 1. 输入投影层: 将Mel频谱投影到d_model维度
        # (batch, time, 80) -> (batch, time, 256)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # 3. Transformer编码器层 (堆叠num_layers层)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 256 -> 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)  # 128 -> num_classes
        )
        
        # 5. 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化模型权重
        使用Xavier初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x: Mel频谱 (batch, time, n_mels)
            mask: 注意力掩码 (可选)
        
        返回:
            logits: (batch, num_classes)
        """
        # 1. 输入投影
        # (batch, time, 80) -> (batch, time, 256)
        x = self.input_projection(x)
        
        # 2. 位置编码
        x = self.pos_encoder(x)
        
        # 3. 通过所有Transformer层
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # 4. 全局平均池化 (在时间维度上)
        # (batch, time, 256) -> (batch, 256)
        x = torch.mean(x, dim=1)
        
        # 5. 分类
        # (batch, 256) -> (batch, num_classes)
        logits = self.classifier(x)
        
        return logits


# ============================================
# 第4部分: 模型创建函数
# ============================================

def create_model():
    """
    根据config创建模型
    
    返回:
        model: SpeechEmotionTransformer模型
    """
    model = SpeechEmotionTransformer(
        input_dim=config.N_MELS,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        d_ff=config.D_FF,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
        max_len=config.MAX_TIME_STEPS
    )
    
    return model


# ============================================
# 第5部分: 测试代码
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("测试Transformer模型")
    print("=" * 60)
    
    # 1. 创建模型
    model = create_model()
    
    # 2. 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型信息:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 3. 测试前向传播
    batch_size = 4
    time_steps = 300
    n_mels = 80
    
    # 创建随机输入
    x = torch.randn(batch_size, time_steps, n_mels)
    
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.2f}, {output.max():.2f}]")
    
    # 4. 测试概率输出
    probs = torch.softmax(output, dim=1)
    print(f"\n概率分布 (第一个样本):")
    for i, emotion in enumerate(config.EMOTION_LABELS):
        print(f"  {emotion}: {probs[0, i]:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ 模型测试通过！")
    print("=" * 60)