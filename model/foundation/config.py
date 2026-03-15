"""Configuration for Somali ASR model training."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model architecture settings (CNN + Transformer + CTC)."""

    # Audio feature extraction
    n_mels: int = 80

    # CNN front-end
    cnn_channels: list[int] = field(default_factory=lambda: [32, 32])
    cnn_kernel_sizes: list[int] = field(default_factory=lambda: [41, 21])
    cnn_strides: list[int] = field(default_factory=lambda: [2, 2])

    # Transformer encoder
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 6
    dim_feedforward: int = 1024
    transformer_dropout: float = 0.1

    # Output
    output_dir: str = "./model/foundation/output"
    vocab_path: str = "./model/foundation/output/vocab.json"


@dataclass
class LMConfig:
    """Character-level language model settings."""

    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    # Training
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    seq_length: int = 256
    weight_decay: float = 1e-5
    seed: int = 42

    # Paths
    text_corpus: str = "./data/text/somali-phrases-corpus.txt"
    output_path: str = "./model/foundation/output/lm.pt"


@dataclass
class DataConfig:
    """Dataset and audio processing settings."""

    data_dir: str = "./data"
    train_manifest: str = "./data/train.tsv"
    eval_manifest: str = "./data/eval.tsv"
    test_manifest: str = "./data/test.tsv"
    eval_split_pct: float = 0.1
    sampling_rate: int = 16_000
    max_audio_length_sec: float = 15.0
    min_audio_length_sec: float = 0.5
    num_workers: int = 4


@dataclass
class TrainingConfig:
    """Hyperparameters for ASR training."""

    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 5.0
    log_interval: int = 50
    eval_interval: int = 1
    save_interval: int = 5
    patience: int = 10
    seed: int = 42
