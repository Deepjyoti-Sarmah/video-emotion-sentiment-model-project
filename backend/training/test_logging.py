# from collections import namedtuple

import torch
from models import MultimodalSentimentModel, MultimodalTrainer
from torch.utils.data import DataLoader, Dataset


class MockDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def test_logging():
    # Batch = namedtuple("Batch", ["text_inputs", "video_frames", "audio_features"])
    # mock_batch = Batch(
    #     text_inputs={"input_ids": torch.ones(1), "attention_mask": torch.ones(1)},
    #     video_frames=torch.ones(1),
    #     audio_features=torch.ones(1),
    # )
    #
    # mock_dataset = MockDataset([mock_batch])
    # # mock_loader = DataLoader([mock_batch])
    # mock_loader = DataLoader(mock_dataset)
    # print(f"mock_loader: ${mock_loader}")
    #
    # model = MultimodalSentimentModel()
    # trainer = MultimodalTrainer(model, mock_loader, mock_loader)
    mock_batch = {
        "text_inputs": {
            "input_ids": torch.ones(1, 128, dtype=torch.long),
            "attention_mask": torch.ones(1, 128, dtype=torch.long),
        },
        "video_frames": torch.ones(30, 3, 224, 224),
        "audio_features": torch.ones(1, 64, 300),
        # 2. Add these labels so compute_class_weights doesn't fail
        "emotion_label": torch.tensor(0),
        "sentiment_label": torch.tensor(1),
    }

    mock_dataset = MockDataset([mock_batch])
    mock_loader = DataLoader(mock_dataset, batch_size=1)

    # TODO: remove later
    print(f"mock_loader: {mock_loader}")

    model = MultimodalSentimentModel()
    # This will now work because train_loader.dataset[0] returns a dict
    trainer = MultimodalTrainer(model, mock_loader, mock_loader)

    train_losses = {"total": 2.5, "emotion": 1.0, "sentiment": 1.5}

    trainer.log_metrics(train_losses, phase="train")

    val_losses = {"total": 1.5, "emotion": 0.5, "sentiment": 1.0}
    val_metrics = {
        "emotion_precision": 0.65,
        "emotion_accuracy": 0.75,
        "sentiment_precision": 0.85,
        "sentiment_accuracy": 0.95,
    }

    trainer.log_metrics(val_losses, val_metrics, phase="val")


if __name__ == "__main__":
    test_logging()
