import os


# AWS SageMaker
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", ".")
SM_CHANNEL_TRAINING = os.environ.get(
    "SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"
)
SM_CHANNEL_VALIDATION = os.environ.get(
    "SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"
)
SM_CHANNEL_TEST = os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

