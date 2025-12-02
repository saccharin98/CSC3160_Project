# dataset.py
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


class RAVDESSDataset(Dataset):
    """
    RAVDESS语音情感识别数据集
    
    功能:
    1. 加载.wav音频文件
    2. 提取Mel频谱图特征
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
        
        # 检查数据路径
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据路径不存在: {self.data_path}")
        
        # 加载所有音频文件路径和标签
        self.audio_files = []
        self.labels = []
        
        self._load_dataset()
        
        # Mel频谱图转换器
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.F_MIN,
            f_max=config.F_MAX
        )
        
        # 振幅转分贝
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
            sample_rate: 采样率
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
    
    def _normalize(self, mel_spec):
        """
        归一化Mel频谱图
        
        参数:
            mel_spec: Mel频谱图 (n_mels, time)
        
        返回:
            normalized: 归一化后的频谱图
        """
        # 标准化到 [-1, 1]
        mel_mean = mel_spec.mean()
        mel_std = mel_spec.std()
        
        if mel_std > 0:
            normalized = (mel_spec - mel_mean) / mel_std
        else:
            normalized = mel_spec - mel_mean
        
        return normalized
    
    def _resize_mel(self, mel_spec, target_length=128):
        """
        调整Mel频谱图到固定长度
        
        参数:
            mel_spec: Mel频谱图 (n_mels, time)
            target_length: 目标时间长度
        
        返回:
            resized: 调整后的频谱图 (n_mels, target_length)
        """
        current_length = mel_spec.shape[1]
        
        if current_length > target_length:
            # 截断：取中间部分
            start = (current_length - target_length) // 2
            mel_spec = mel_spec[:, start:start + target_length]
        
        elif current_length < target_length:
            # 填充：两边补零
            pad_length = target_length - current_length
            pad_left = pad_length // 2
            pad_right = pad_length - pad_left
            
            mel_spec = torch.nn.functional.pad(
                mel_spec,
                (pad_left, pad_right),
                mode='constant',
                value=0
            )
        
        return mel_spec
    
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
            mel_spec: Mel频谱图 (n_mels, time)
            label: 情感标签 (0-7)
        """
        # 1. 加载音频
        audio_path = self.audio_files[idx]
        waveform = self._load_audio(audio_path)
        
        # 2. 提取Mel频谱图
        mel_spec = self._extract_mel_spectrogram(waveform)
        
        # 3. 调整到固定大小
        mel_spec = self._resize_mel(mel_spec, config.MEL_TIME_STEPS)
        
        # 4. 归一化
        mel_spec = self._normalize(mel_spec)
        
        # 5. 转置：(time, n_mels) - Transformer输入格式
        mel_spec = mel_spec.transpose(0, 1)
        
        # 6. 获取标签
        label = self.labels[idx]
        
        # 7. 可选的数据增强
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        return mel_spec, label


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
    
    # 测试获取一个样本
    print(f"\n测试样本获取:")
    mel_spec, label = dataset[0]
    
    print(f"  Mel频谱形状: {mel_spec.shape}")
    print(f"  标签: {label} ({config.EMOTION_LABELS[label]})")
    print(f"  数据类型: {mel_spec.dtype}")
    print(f"  数值范围: [{mel_spec.min():.2f}, {mel_spec.max():.2f}]")
    
    # 测试批量加载
    print(f"\n测试批量加载:")
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    batch_mel, batch_labels = next(iter(dataloader))
    print(f"  批次Mel频谱形状: {batch_mel.shape}")
    print(f"  批次标签形状: {batch_labels.shape}")
    print(f"  批次标签: {batch_labels.tolist()}")
    
    print("\n" + "=" * 70)
    print("✓ 数据集测试完成")
    print("=" * 70)