import os

import cv2
import gdown
import torch
from dotenv import load_dotenv

from model.model import LightningEyesClassifier
from model.transforms import DEFAULT_TRANSFORM

load_dotenv()
DRIVE_ID = os.getenv("DRIVE_ID")


class OpenEyesClassificator:
    def __init__(
        self,
        model_checkpoint="./checkpoints/final_best.ckpt",
        transform=DEFAULT_TRANSFORM,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_checkpoint(model_checkpoint)
        self.model = LightningEyesClassifier.load_from_checkpoint(
            checkpoint_path=model_checkpoint, map_location=self.device
        )
        self.model.eval()
        self.transform = transform

    def load_checkpoint(self, model_checkpoint):
        folder_path = "/".join(model_checkpoint.split("/")[:-1])
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            gdown.download(id=DRIVE_ID, output=model_checkpoint)

    def data_process(self, inpIm):
        image = cv2.imread(inpIm)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image

    def predict(self, inpIm):
        self.model.eval()
        img_tensor = self.data_process(inpIm)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(img_tensor).detach()
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
        is_open_score = probabilities[1].item()
        return is_open_score


if __name__ == "__main__":
    classificator = OpenEyesClassificator()
    is_open_score = classificator.predict("./test_data/test.png")
    print(f"is_open_score = {is_open_score}")
