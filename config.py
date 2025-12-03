from pathlib import Path


class Config:
    DATA_PATH = Path('./data')
    SAMPLE_RATE = 16000
    AUDIO_DURATION = 3.0

    # ============================================
    # Mel 频谱图配置（备用）
    # ============================================
    N_MELS = 80
    N_FFT = 512
    HOP_LENGTH = 160
    MAX_TIME_STEPS = 300
    MEL_TIME_STEPS = MAX_TIME_STEPS
    F_MIN = 0.0
    F_MAX = SAMPLE_RATE / 2

    # ============================================
    # 特征提取配置
    # ============================================
    USE_PRETRAINED = True  # 是否使用 Wav2Vec2 预训练特征
    WAV2VEC2_MODEL = "facebook/wav2vec2-base-960h"  # 预训练模型
    
    # Wav2Vec2 输出维度是 768
    INPUT_DIM = 768 if USE_PRETRAINED else N_MELS

    # ============================================
    # Transformer 模型配置（增强版）
    # ============================================
    D_MODEL = 256          # 从 256 增加到 512
    NUM_HEADS = 8
    NUM_LAYERS = 4         # 从 6 增加到 8
    D_FF = 1024            # 从 1024 增加到 2048
    DROPOUT = 0.2          # 从 0.1 增加到 0.2

    # ============================================
    # 训练配置
    # ============================================
    BATCH_SIZE = 1        # 减小 batch size（因为特征提取更重）
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-4   # 降低学习率（预训练特征需要更小的学习率）
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 1e-4
    TRAIN_SPLIT = 0.8
    NUM_WORKERS = 2

    # ============================================
    # 数据集配置
    # ============================================
    NUM_CLASSES = 8
    EMOTION_LABELS = [
        'neutral',
        'calm',
        'happy',
        'sad',
        'angry',
        'fearful',
        'disgust',
        'surprised'
    ]
    
    # ============================================
    # 设备配置
    # ============================================
    DEVICE = 'cuda'
    SEED = 42

    # ============================================
    # 保存路径
    # ============================================
    CHECKPOINT_DIR = Path('./checkpoints')
    LOG_DIR = Path('./runs')
    EARLY_STOPPING = False
    EARLY_STOPPING_PATIENCE = 10

    @classmethod
    def init_dirs(cls):
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)


Config.init_dirs()
config = Config()