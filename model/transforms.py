import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEFAULT_TRANSFORM = A.Compose(
    [
        A.Resize(160, 160),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]
)
