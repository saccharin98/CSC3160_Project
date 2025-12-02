# dataset.py
"""
简单的数据加载
不用缓存，不用复杂增强
先跑通再优化
"""

import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from pathlib import Path
from config import config


class SimpleEmotionDataset(Dataset):
    """
    简单的情感数据集
    只做最基本的处理
    """
    
    def __init__(self, audio_files, labels, is_train=True):
        """
        参数:
            audio_files: 音频文件路径列表
            labels: 标签列表
            is_train: 是否训练集（用于数据增强）
        """
        self.audio_files = audio_files
        self.labels = labels
        self.is_train = is_train
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """获取一个样本"""
        # 1. 加载音频
        audio_path = self.audio_files[idx]
        y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, duration=config.AUDIO_DURATION)
        
        # 2. 填充或截断到固定长度
        target_length = int(config.AUDIO_DURATION * config.SAMPLE_RATE)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
        
        # 3. 提取Mel频谱
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=config.SAMPLE_RATE,
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH
        )
        
        # 4. 转换为dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 5. 归一化到[-1, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        # 6. 调整时间维度
        if mel_spec_db.shape[1] > config.MAX_TIME_STEPS:
            mel_spec_db = mel_spec_db[:, :config.MAX_TIME_STEPS]
        else:
            pad_width = config.MAX_TIME_STEPS - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        
        # 7. 转换为tensor: (time, n_mels)
        mel_tensor = torch.FloatTensor(mel_spec_db.T)  # 转置：(time, n_mels)
        
        # 8. 标签
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return mel_tensor, label


def load_ravdess_data():
    """
    加载RAVDESS数据集
    
    文件名格式: 03-01-05-01-01-01-12.wav
    第3个数字是情感: 01=neutral, 03=happy, 04=sad, 05=angry
    
    返回:
        audio_files: 音频路径列表
        labels: 标签列表
    """
    data_path = Path(config.DATA_PATH)
    
    audio_files = []
    labels = []
    
    # 情感映射
    emotion_map = {
        '01': 0,  # neutral
        '03': 1,  # happy
        '04': 2,  # sad
        '05': 3   # angry
    }
    
    # 遍历所有Actor文件夹
    for actor_folder in sorted(data_path.glob('Actor_*')):
        for audio_file in actor_folder.glob('*.wav'):
            # 解析文件名
            parts = audio_file.stem.split('-')
            emotion_code = parts[2]
            
            # 只选择我们需要的4种情感
            if emotion_code in emotion_map:
                audio_files.append(str(audio_file))
                labels.append(emotion_map[emotion_code])
    
    print(f"✓ 加载了 {len(audio_files)} 个音频文件")
    print(f"  标签分布: {np.bincount(labels)}")
    
    return audio_files, labels


def create_dataloaders():
    """
    创建训练和测试数据加载器
    
    返回:
        train_loader, test_loader
    """
    from sklearn.model_selection import train_test_split
    
    # 加载数据
    audio_files, labels = load_ravdess_data()
    
    # 划分训练集和测试集（80%-20%）
    train_files, test_files, train_labels, test_labels = train_test_split(
        audio_files, labels,
        test_size=0.2,
        random_state=config.SEED,
        stratify=labels  # 保持标签分布
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(train_files)} 样本")
    print(f"  测试集: {len(test_files)} 样本")
    
    # 创建数据集
    train_dataset = SimpleEmotionDataset(train_files, train_labels, is_train=True)
    test_dataset = SimpleEmotionDataset(test_files, test_labels, is_train=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True  # 加速GPU传输
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader


# 测试代码
if __name__ == '__main__':
    # 测试数据加载
    train_loader, test_loader = create_dataloaders()
    
    # 查看一个batch
    for mel_spec, label in train_loader:
        print(f"\nBatch形状:")
        print(f"  Mel频谱: {mel_spec.shape}")  # (batch, time, n_mels)
        print(f"  标签: {label.shape}")         # (batch,)
        break