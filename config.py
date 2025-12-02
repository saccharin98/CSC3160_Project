# config.py
"""
所有配置都在这里
修改参数很方便
"""

class Config:
    # ============ 数据配置 ============
    DATA_PATH = './data'          # RAVDESS数据路径
    SAMPLE_RATE = 16000                    # 采样率（降低到16k节省显存）
    AUDIO_DURATION = 3.0                   # 音频时长（秒）
    
    # ============ 特征配置 ============
    N_MELS = 80                            # Mel频谱bins（降低维度）
    N_FFT = 512                            # FFT窗口
    HOP_LENGTH = 160                       # 帧移
    MAX_TIME_STEPS = 300                   # 最大时间步（3秒音频）
    
    # ============ 模型配置 ============
    D_MODEL = 256                          # 模型维度（适中）
    NUM_HEADS = 8                          # 注意力头数
    NUM_LAYERS = 6                         # Transformer层数
    D_FF = 1024                            # FFN维度
    DROPOUT = 0.1
    
    # ============ 训练配置 ============
    BATCH_SIZE = 32                        # 批次大小
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001
    WARMUP_STEPS = 500
    
    # ============ 其他 ============
    NUM_CLASSES = 4                        # 情感类别数
    EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry']
    DEVICE = 'cuda'                        # A100 GPU
    SEED = 42

config = Config()