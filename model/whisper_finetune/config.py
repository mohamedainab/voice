"""Configuration for Whisper fine-tuning on Somali speech data."""

from dataclasses import dataclass, field


@dataclass
class WhisperConfig:
    """Whisper model and LoRA fine-tuning settings."""

    # Base model
    model_name: str = "openai/whisper-large-v3"
    language: str = "so"  # Somali ISO 639-1
    task: str = "transcribe"

    # LoRA settings (parameter-efficient fine-tuning)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Paths
    output_dir: str = "./model/whisper_finetune/output"


@dataclass
class WhisperDataConfig:
    """Dataset settings."""

    data_dir: str = "./data"
    train_manifest: str = "./data/train.tsv"
    eval_manifest: str = "./data/eval.tsv"
    test_manifest: str = "./data/test.tsv"
    eval_split_pct: float = 0.1
    sampling_rate: int = 16_000
    max_audio_length_sec: float = 30.0
    min_audio_length_sec: float = 0.5
    num_workers: int = 4


@dataclass
class WhisperTrainingConfig:
    """Training hyperparameters."""

    epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # effective batch = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True  # mixed precision
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 3
    seed: int = 42
    patience: int = 5
