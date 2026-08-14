# Video Sentiment Model

A multimodal deep learning model that predicts **emotions** (anger, disgust, fear, joy, neutral, sadness, surprise) and **sentiment** (negative, neutral, positive) from short video clips of people speaking.

## What it does

Given a short video of a person speaking, the model analyzes three modalities at once and produces two predictions:

- **Emotion** — one of 7 classes: anger, disgust, fear, joy, neutral, sadness, surprise
- **Sentiment** — one of 3 classes: negative, neutral, positive

It is trained end-to-end on the [MELD dataset](https://affective-meld.github.io/) (a multimodal extension of the EmotionLines corpus drawn from *Friends* TV episodes), where every utterance is a short video clip labeled with both an emotion and a sentiment.

## Why it does it

Human communication is inherently multimodal — meaning lives in *what* is said (text), *how* it is said (audio prosody), and *how* it looks (facial expressions and body language). Relying on any single modality is brittle:

- Text alone misses sarcasm, tone, and facial cues.
- Audio alone loses the lexical content.
- Video alone loses what was actually said.

By fusing all three signals, the model can disambiguate cases that a unimodal system would get wrong (e.g., a sarcastic "great" spoken flatly). The result is a single model capable of emotion/sentiment classification of natural conversational video.

## How it does it

### Architecture

The model is a fusion of three frozen pretrained feature extractors feeding a small trainable head:

```
Text    ──► BERT (bert-base-uncased, frozen) ──► [CLS] ──► Linear(768→128)
Video   ──► ResNet3D-18 (frozen)               ──► Linear(→128)
Audio   ──► Conv1d mel-spectrogram CNN (frozen) ──► Linear(→128)
                                              │
                      concatenate  (128 × 3 = 384)
                                              │
                              Fusion layer: Linear(384→256) + BN + ReLU + Dropout
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
       Emotion head (Linear→64→7)                                    Sentiment head (Linear→64→3)
```

- **TextEncoder** — `bert-base-uncased` with all parameters frozen; the pooled `[CLS]` embedding (768-d) is projected down to 128-d.
- **VideoEncoder** — a `ResNet3D-18` video backbone, frozen, with its final FC layer replaced by a 128-d head. It consumes 30 resized 224×224 frames per clip.
- **AudioEncoder** — a small frozen CNN over 64-band mel-spectrograms (300 time-steps), pooled to 128-d.
- **Fusion & heads** — the three 128-d embeddings are concatenated, fused, and passed to two classification heads producing logits for 7 emotions and 3 sentiments. Only the projection/fusion/classification layers are trained (~the frozen backbones are never fine-tuned).

### Data pipeline

`MELDDataset` (`training/meld_dataset.py`) turns each CSV row + raw video into a training sample:

- **Text** — the utterance is tokenized with the BERT tokenizer (padded/truncated to 128 tokens).
- **Video** — OpenCV reads up to 30 frames, resized to 224×224 and normalized; short clips are zero-padded, longer ones truncated.
- **Audio** — FFmpeg extracts a 16 kHz mono PCM track, which is converted to a 64-band mel-spectrogram (300 time-steps), normalized, and padded/truncated.
- Samples that fail to load are filtered out by the custom `collate_fn`.

### Training

`MultimodalTrainer` (`training/models.py`) handles the training loop:

- **Loss** — sum of two `CrossEntropyLoss` terms (emotion + sentiment), each with 5% label smoothing and inverse-frequency **class weights** computed from the training set (MELD is heavily imbalanced — "neutral"/"joy" dominate, "disgust"/"fear" are rare).
- **Optimizer** — Adam with per-module learning rates (very low for the frozen encoders' heads, higher for fusion/classifiers) and weight decay; `ReduceLROnPlateau` schedules on validation loss.
- **Regularization** — dropout throughout, BatchNorm, and gradient clipping (max norm 1.0).
- **Logging** — TensorBoard (train/val losses and per-task accuracy/precision), plus SageMaker-compatible JSON metrics emitted each epoch.
- **Checkpointing** — the best model (lowest validation loss) is saved as `model.pth`.
- Final evaluation runs on the held-out test split.

### Running

The project is split into two environments:

- `backend/training/` — training code and requirements (PyTorch, torchaudio, torchvision, transformers, OpenCV, SageMaker SDK). Run locally with `python train.py` or on SageMaker.
- `backend/deployment/` — a slimmed-down set of requirements (and models/inference stubs) for serving the trained model in production.

SageMaker is the primary training target:

1. Upload the MELD CSV files and video splits to S3 (`train`, `dev`, `test` channels).
2. Configure `backend/train_sagemaker.py` with your bucket and execution role.
3. `estimator.fit()` launches a PyTorch training job on a GPU instance (e.g. `ml.g5.xlarge`); the container auto-installs a static FFmpeg binary (`install_ffmpeg.py`) so audio extraction works in the sandboxed environment.

### Repository layout

```
backend/
├── train_sagemaker.py        # SageMaker estimator launcher
├── training/
│   ├── train.py              # Entry point: dataloaders, training loop, SageMaker metrics
│   ├── models.py             # Encoders, MultimodalSentimentModel, MultimodalTrainer
│   ├── meld_dataset.py       # MELD dataset + audio/video/text preprocessing
│   ├── install_ffmpeg.py     # Static FFmpeg bootstrap for the container
│   ├── count_parameters.py   # Trainable-parameter audit by component
│   └── test_logging.py       # Smoke test for metric logging
└── deployment/               # Serving-side requirements/models (work in progress)
```