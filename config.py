from pathlib import Path


class Config:
    DATA_PATH = Path('./data')
    SAMPLE_RATE = 16000
    AUDIO_DURATION = 3.0

    N_MELS = 80
    N_FFT = 512
    HOP_LENGTH = 160
    MAX_TIME_STEPS = 300
    MEL_TIME_STEPS = MAX_TIME_STEPS
    F_MIN = 0.0
    F_MAX = SAMPLE_RATE / 2

    D_MODEL = 256
    NUM_HEADS = 8
    NUM_LAYERS = 6
    D_FF = 1024
    DROPOUT = 0.1

    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 1e-4
    TRAIN_SPLIT = 0.8
    NUM_WORKERS = 2

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
    DEVICE = 'cuda'
    SEED = 42

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
