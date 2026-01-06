import os
import torch
import torch.nn as nn
from torch.utils.data import dataset
from transformers import BertModel
from torchvision import models as vision_models
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from meld_dataset import MELDDataset


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # Extract BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)


class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_fts = self.backbone.fc.in_features
        # self.backbone.fc = nn.Sequential(
        #     nn.Linear(num_fts, 128), nn.ReLU(), nn.Dropout(0.2)
        # )

        # for typecheck error
        head = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        setattr(self.backbone, "fc", head)

    def forward(self, x):
        # [batch_size, frames, channels, height, width]->[batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            # Lower level feactures
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Higher level feactures
            nn.Conv1d(64, 128, kernel_size=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        for param in self.conv_layer.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = x.squeeze(1)

        features = self.conv_layer(x)
        # Features output : [batch_size, 128, 1]

        return self.projection(features.squeeze(-1))


class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classification heads
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7),  # Sadness, anger
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),  # Negative, positive, neutral
        )

    def forward(self, text_inputs, video_frames, audio_features):
        text_features = self.text_encoder(
            text_inputs["input_ids"],
            text_inputs["attention_mask"],
        )
        video_features = -self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate multimodal features
        combined_features = torch.cat(
            [
                text_features,
                video_features,
                audio_features,
            ],
            dim=1,
        )
        # [batch_size, 128 * 3]

        fused_features = self.fusion_layer(combined_features)

        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)

        return {
            "emotions": emotion_output,
            "sentiments": sentiment_output,
        }


def compute_class_weights(dataset):
    emotion_counts = torch.zeros(7)
    sentiment_counts = torch.zeros(3)
    skipped = 0
    total = len(dataset)

    print("\n Counting class distributions...")
    for i in range(total):
        sample = dataset[i]

        if sample is None:
            skipped += 1
            continue

        emotion_label = sample["emotion_label"]
        sentiment_label = sample["sentiment_label"]

        emotion_counts[emotion_label] += 1
        sentiment_counts[sentiment_label] += 1

    valid = total - skipped
    print(f"skipped sample: {skipped}/{total}")

    print("\n Class distrubution")
    print("Emotions:")
    emotion_map = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "joy",
        4: "neutral",
        5: "sadness",
        6: "surprise",
    }

    for i, count in enumerate(emotion_counts):
        print(f"{emotion_map[i]}: {count / valid:.2f}")

    # Calculate class weights
    emotion_weights = 1.0 / emotion_counts
    sentiment_weights = 1.0 / sentiment_counts

    # Normalize weights
    emotion_weights = emotion_weights / emotion_weights.sum()
    sentiment_weights = sentiment_weights / sentiment_weights.sum()

    return emotion_weights, sentiment_weights


class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Log dataset sized
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        print("\nDataset sized:")
        print(f"Training samples: {train_size:, }")
        print(f"Validation samples: {val_size:, }")
        print(f"Batches per epoch: {len(train_loader):, }")

        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")  # Dec17_14-22-35
        base_dir = (
            "/opt/ml/output/tensorboard" if "SM_MODEL_DIR" in os.environ else "runs"
        )
        log_dir = f"{base_dir}/run_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        # Very high: 1, high: 0.1-0.01, medium: 1e-1, low: 1e-4, very low: 1e-5
        self.optimizer = torch.optim.Adam(
            [
                {"params": model.text_encoder.parameters(), "lr": 8e-6},
                {"params": model.video_encoder.parameters(), "lr": 8e-6},
                {"params": model.audio_encoder.parameters(), "lr": 8e-6},
                {"params": model.fusion_layer.parameters(), "lr": 5e-4},
                {"params": model.segment_classifier.parameters(), "lr": 5e-4},
            ],
            weight_decay=1e-5,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=2,
        )

        self.current_train_losses = None

        # Calculate class weights
        print("\n Calculating class weight...")
        emotion_weights, sentiment_weights = compute_class_weights(train_loader.dataset)

        device = next(model.parameters()).device

        self.emotion_weights = emotion_weights.to(device)
        self.sentiment_weights = sentiment_weights.to(device)

        print(f"Emotion weight on device: {self.emotion_weights.device}")
        print(f"Sentiment weight on device: {self.sentiment_weights.device}")

        self.emotion_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05,
            weight=self.emotion_weights,
        )

        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05,
            weight=self.sentiment_weights,
        )


if __name__ == "__main__":
    dataset = MELDDataset(
        "../dataset/train/train_sent_emo.csv", "../dataset/train/train_splits"
    )

    sample = dataset[0]

    model = MultimodalSentimentModel()
    model.eval()

    text_inputs = {
        "input_ids": sample["text_inputs"]["input_ids"].unsqueeze(0),
        "attention_mask": sample["text_inputs"]["attention_mask"].unsqueeze(0),
    }
    video_frames = sample["video_frames"].unsqueeze(0)
    audio_features = sample["audio_features"].unsqueeze(0)

    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)

        emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
        sentiment_probs = torch.softmax(outputs["sentiments"], dim=1)[0]

    emotion_map = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "joy",
        4: "neutral",
        5: "sadness",
        6: "surprise",
    }

    sentiment_map = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }

    for i, prob in enumerate(emotion_probs):
        print(f"{emotion_map[i]}: {prob:.2f}")

    for i, prob in enumerate(sentiment_probs):
        print(f"{sentiment_map[i]}: {prob:.2f}")

    print("Predictions for utterance")
