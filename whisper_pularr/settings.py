from __future__ import annotations

DEFAULT_DATASET_NAME = "google/WaxalNLP"
DEFAULT_DATASET_CONFIG = "ful_asr"
DEFAULT_TRAINABLE_MODEL = "openai/whisper-small"
DEFAULT_TEACHER_MODEL = "openai/whisper-large-v3"
DEFAULT_WHISPER_LANGUAGE_HINT = None
DEFAULT_AUDIO_SAMPLING_RATE = 16_000
DEFAULT_MAX_TRAIN_DURATION_SECONDS = 30.0
DEFAULT_EVAL_CHUNK_LENGTH_SECONDS = 30
DEFAULT_TRAINING_LANGUAGE_CANDIDATES = ("ful", "fula", "fulah", "fulfulde", "pulaar", "pular", "ff")
DEFAULT_STOP_WER = 20.0
DEFAULT_STOP_CER = 20.0
DEFAULT_HARD_STOP_WER = 15.0
DEFAULT_HARD_STOP_CER = 15.0
DEFAULT_EARLY_STOP_PATIENCE_EPOCHS = 4

SUPERVISED_PRESETS = {
    "trial_a": {
        "learning_rate": 1e-5,
        "label_smoothing_factor": 0.1,
        "apply_spec_augment": True,
    },
    "trial_b": {
        "learning_rate": 5e-6,
        "label_smoothing_factor": 0.1,
        "apply_spec_augment": True,
    },
    "trial_c": {
        "learning_rate": 1e-5,
        "label_smoothing_factor": 0.0,
        "apply_spec_augment": False,
    },
}
