"""
RAVDESS语音情感数据集加载器

RAVDESS数据集说明:
- 24个演员 (12男12女)
- 每个演员60个音频文件
- 总共1440个音频文件

文件命名格式: 03-01-01-01-01-01-01.wav
              |  |  |  |  |  |  |
              |  |  |  |  |  |  +- 重复次数 (01/02)
              |  |  |  |  |  +---- 演员ID (01-24)
              |  |  |  |  +------- 情感强度 (01=normal, 02=strong)
              |  |  |  +---------- 语句ID (01/02)
              |  |  +------------- 情感ID (01-08)
              |  +---------------- 声音通道 (01=speech)
              +------------------- 模态 (03=audio-video)

情感ID对应:
01 = neutral    (中性)
02 = calm       (平静)
03 = happy      (快乐)
04 = sad        (悲伤)
05 = angry      (愤怒)
06 = fearful    (恐惧)
07 = disgust    (厌恶)
08 = surprised  (惊讶)
"""

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import config
import os


class RAVDESSDataset(Dataset):
    """
    RAVDESS语音情感识别数据集
    
    功能:
    1. 加载.wav音频文件
    2. 提取特征（Wav2Vec2 或 Mel频谱图）
    3. 归一化处理
    4. 返回(特征, 标签)对
    """
    
    def __init__(self, data_path, transform=None):
        """
        参数:
            data_path: 数据集根目录路径 (包含Actor_XX文件夹)
            transform: 可选的数据增强变换
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.use_pretrained = config.USE_PRETRAINED
        
        # 检查数据路径
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据路径不存在: {self.data_path}")
        
        # 加载所有音频文件路径和标签
        self.audio_files = []
        self.labels = []
        
        self._load_dataset()
        
        # 根据配置选择特征提取方式
        if self.use_pretrained:
            print("使用 Wav2Vec2 预训练特征")
            self._init_wav2vec2()
        else:
            print("使用 Mel 频谱图特征")
            self._init_mel_transform()
    
    def _init_wav2vec2(self):
        """
        初始化 Wav2Vec2 模型（使用本地模型）
        """
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        
        # 使用本地模型路径
        model_path = "./models/wav2vec2-base-960h"
        
        # 检查本地模型是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"本地模型不存在: {model_path}\n"
                f"请先运行 download_wav2vec2_offline.py 下载模型"
            )
        
        print(f"✓ 使用本地模型: {model_path}")
        
        # 从本地加载模型（不联网）
        self.processor = Wav2Vec2Processor.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        print("✓ Wav2Vec2 模型加载成功")
        
        self.wav2vec2.eval()  # 设置为评估模式
        
        # 冻结 Wav2Vec2 的参数
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        # 移到 GPU（如果可用）
        if torch.cuda.is_available():
            self.wav2vec2 = self.wav2vec2.cuda()
            print("✓ Wav2Vec2 已移到 GPU")
        else:
            print("✓ Wav2Vec2 使用 CPU")
    
    def _init_mel_transform(self):
        """
        初始化 Mel 频谱图转换器
        """
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.F_MIN,
            f_max=config.F_MAX
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        
    def _load_dataset(self):
        """
        加载数据集，扫描所有Actor文件夹
        """
        print(f"加载数据集: {self.data_path}")
        
        # 查找所有Actor_XX文件夹
        actor_folders = sorted(self.data_path.glob('Actor_*'))
        
        if len(actor_folders) == 0:
            raise FileNotFoundError(
                f"没有找到Actor_XX文件夹！\n"
                f"请确保数据集已解压到: {self.data_path}\n"
                f"应该包含: Actor_01, Actor_02, ..., Actor_24"
            )
        
        print(f"找到 {len(actor_folders)} 个演员文件夹")
        
        # 遍历每个演员文件夹
        for actor_folder in tqdm(actor_folders, desc="加载演员数据"):
            # 获取该文件夹下所有.wav文件
            wav_files = list(actor_folder.glob('*.wav'))
            
            for wav_file in wav_files:
                # 从文件名提取情感标签
                emotion_id = self._extract_emotion_from_filename(wav_file.name)
                
                if emotion_id is not None:
                    self.audio_files.append(wav_file)
                    self.labels.append(emotion_id)
        
        print(f"✓ 成功加载 {len(self.audio_files)} 个音频文件")
        print(f"  情感分布: {self._get_emotion_distribution()}")
    
    def _extract_emotion_from_filename(self, filename):
        """
        从文件名提取情感ID
        
        文件名格式: 03-01-01-01-01-01-01.wav
                          ^^
                          情感ID (第3个字段)
        
        参数:
            filename: 文件名
        
        返回:
            emotion_id: 情感ID (0-7)，失败返回None
        """
        try:
            parts = filename.split('-')
            if len(parts) >= 3:
                emotion_code = int(parts[2])  # 第3个字段是情感
                # 情感ID从1开始，转换为0-based索引
                emotion_id = emotion_code - 1
                
                # 验证范围 (0-7 对应8种情感)
                if 0 <= emotion_id < config.NUM_CLASSES:
                    return emotion_id
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _get_emotion_distribution(self):
        """
        获取情感分布统计
        
        返回:
            dict: {情感名: 数量}
        """
        distribution = {}
        for label in self.labels:
            emotion_name = config.EMOTION_LABELS[label]
            distribution[emotion_name] = distribution.get(emotion_name, 0) + 1
        
        return distribution
    
    def _load_audio(self, audio_path):
        """
        加载音频文件
        
        参数:
            audio_path: 音频文件路径
        
        返回:
            waveform: 音频波形 (1, samples)
        """
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重采样（如果需要）
        if sample_rate != config.SAMPLE_RATE:
            resampler = T.Resample(sample_rate, config.SAMPLE_RATE)
            waveform = resampler(waveform)
        
        return waveform
    
    def _extract_wav2vec2_features(self, waveform):
        """
        使用 Wav2Vec2 提取特征
        
        参数:
            waveform: 音频波形 (1, samples)
        
        返回:
            features: Wav2Vec2 特征 (time, 768)
        """
        with torch.no_grad():
            # 处理音频
            inputs = self.processor(
                waveform.squeeze(0).numpy(),
                sampling_rate=config.SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )
            
            # 移到 GPU（如果可用）
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 提取特征
            outputs = self.wav2vec2(**inputs)
            features = outputs.last_hidden_state  # (1, time, 768)
            features = features.squeeze(0)  # (time, 768)
            
            # 移回 CPU
            features = features.cpu()
        
        return features
    
    def _extract_mel_spectrogram(self, waveform):
        """
        提取Mel频谱图特征
        
        参数:
            waveform: 音频波形 (1, samples)
        
        返回:
            mel_spec: Mel频谱图 (n_mels, time)
        """
        # 1. 计算Mel频谱
        mel_spec = self.mel_transform(waveform)  # (1, n_mels, time)
        
        # 2. 转换为分贝
        mel_spec = self.amplitude_to_db(mel_spec)  # (1, n_mels, time)
        
        # 3. 去掉batch维度
        mel_spec = mel_spec.squeeze(0)  # (n_mels, time)
        
        return mel_spec
    
    def _normalize(self, features):
        """
        归一化特征
        
        参数:
            features: 特征张量
        
        返回:
            normalized: 归一化后的特征
        """
        mean = features.mean()
        std = features.std()
        
        if std > 0:
            normalized = (features - mean) / std
        else:
            normalized = features - mean
        
        return normalized
    
    def _resize_features(self, features, target_length=300):
        """
        调整特征到固定长度 - 完全修复版本
        
        参数:
            features: 特征 (time, feature_dim) - 任意时间长度
            target_length: 目标时间长度 (固定为300)
        
        返回:
            resized: 调整后的特征 (target_length, feature_dim)
        """
        current_length = features.shape[0]  # 当前时间步数
        feature_dim = features.shape[1]     # 特征维度
        
        if current_length == target_length:
            # 长度已经正确，直接返回
            return features
        
        elif current_length > target_length:
            # 情况1: 当前长度 > 目标长度 → 截断（取中间部分）
            start = (current_length - target_length) // 2
            features = features[start:start + target_length, :]
        
        else:
            # 情况2: 当前长度 < 目标长度 → 填充（两边补零）
            pad_length = target_length - current_length
            pad_left = pad_length // 2
            pad_right = pad_length - pad_left
            
            # 使用 F.pad: (左, 右, 上, 下)
            # 对于 (time, feature_dim)，我们只在时间维度(第0维)填充
            features = torch.nn.functional.pad(
                features,
                (0, 0, pad_left, pad_right),  # (feature维度不填充, 时间维度填充)
                mode='constant',
                value=0
            )
        
        # 最终验证
        assert features.shape == (target_length, feature_dim), \
            f"调整后形状错误: {features.shape}, 期望: ({target_length}, {feature_dim})"
        
        return features
    
    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        参数:
            idx: 索引
        
        返回:
            features: 特征 (time, feature_dim) - 固定为 (300, 768) 或 (300, 80)
            label: 情感标签 (0-7)
        """
        # 1. 加载音频
        audio_path = self.audio_files[idx]
        waveform = self._load_audio(audio_path)
        
        # 2. 提取特征
        if self.use_pretrained:
            # 使用 Wav2Vec2: (time_variable, 768)
            features = self._extract_wav2vec2_features(waveform)
        else:
            # 使用 Mel 频谱图: (n_mels, time_variable)
            mel_spec = self._extract_mel_spectrogram(waveform)
            # 转置为 (time_variable, n_mels)
            features = mel_spec.transpose(0, 1)
        
        # 3. 调整到固定大小: (300, feature_dim)
        features = self._resize_features(features, config.MEL_TIME_STEPS)
        
        # 4. 归一化
        features = self._normalize(features)
        
        # 5. 获取标签
        label = self.labels[idx]
        
        # 6. 可选的数据增强
        if self.transform:
            features = self.transform(features)
        
        return features, label


# ============================================
# 测试代码
# ============================================

if __name__ == '__main__':
    """
    测试数据集加载
    """
    print("=" * 70)
    print("测试RAVDESS数据集")
    print("=" * 70)
    
    # 创建数据集
    dataset = RAVDESSDataset(config.DATA_PATH)
    
    print(f"\n数据集信息:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  情感类别: {config.NUM_CLASSES}")
    print(f"  情感标签: {config.EMOTION_LABELS}")
    print(f"  特征类型: {'Wav2Vec2' if config.USE_PRETRAINED else 'Mel频谱图'}")
    
    # 测试获取一个样本
    print(f"\n测试样本获取:")
    features, label = dataset[0]
    
    print(f"  特征形状: {features.shape}")
    print(f"  标签: {label} ({config.EMOTION_LABELS[label]})")
    print(f"  数据类型: {features.dtype}")
    print(f"  数值范围: [{features.min():.2f}, {features.max():.2f}]")
    
    # 测试多个样本的形状一致性
    print(f"\n测试形状一致性:")
    for i in range(5):
        f, l = dataset[i]
        print(f"  样本 {i}: {f.shape} - {config.EMOTION_LABELS[l]}")
    
    # 测试批量加载
    print(f"\n测试批量加载:")
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    batch_features, batch_labels = next(iter(dataloader))
    print(f"  批次特征形状: {batch_features.shape}")
    print(f"  批次标签形状: {batch_labels.shape}")
    print(f"  批次标签: {batch_labels.tolist()}")
    
    # 验证所有特征维度一致
    expected_shape = (config.MEL_TIME_STEPS, config.INPUT_DIM)
    for i in range(batch_features.shape[0]):
        assert batch_features[i].shape == expected_shape, \
            f"样本 {i} 形状不一致: {batch_features[i].shape}"
    
    print(f"  ✓ 所有样本形状一致: {expected_shape}")
    
    print("\n" + "=" * 70)
    print("✓ 数据集测试完成")
    print("=" * 70)