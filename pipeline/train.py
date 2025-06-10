import wandb
import yaml
import pandas as pd
import argparse
import os

from models.setfit import SBERTModel
from utils.data_loader import DataLoader
from models.bert import BERTModel
from models.distilroberta import DistilRoBERTaModel
import argparse

from models.roberta import RoBERTaModel
from models.distilroberta import DistilRoBERTaModel
from utils.data_loader import DataLoader
from utils.seed import set_seed

os.environ["WANDB_SILENT"] = "true"

class TrainPipeline:
    def __init__(self, config=None, model_override=None):
        # === Load config ===
        if config is not None:
            self.config = config
        else:
            with open("config.yaml", "r") as f:
                self.config = yaml.safe_load(f)

        if model_override:
            valid_models = ["bert", "roberta", "sbert", "distilroberta"]
            if model_override not in valid_models:
                raise ValueError(f"Invalid model name: {model_override}")
            self.config["train"]["model"] = model_override

        self.model_type = self.config["train"]["model"]
        self.data_loader = DataLoader(self.config)

    def run(self):
        set_seed(42)

        model_type = self.config["train"]["model"]
        run_mode = self.config.get("wandb", {}).get("run_mode", "manual")

        if "sweep" in run_mode.lower():
            print("🔁 Running in SWEEP mode.")
            wandb.config.update(self.config)
        else:
            print("🚀 Running single training run.")
            wandb.init(project=self.config["wandb"]["project_name"])
            wandb.config.update(self.config)

        print("\n📦 Final training configuration:")
        print(f"Model: {model_type}")
        print(f"Hyperparameters: {self.config.get(model_type, {})}")
        print(f"📂 Dataset path: {self.data_loader.classify_filepath}")

        df = self.data_loader.load_classification_data()
        print(f"📊 Loaded dataset with {len(df)} samples")
        print(df.head(2))

    
        df["text"] = df["text"].apply(lambda x: self.data_loader.clean_text(x))

        train_df, test_df = self.data_loader.train_test_split(df)

        model_map = {
            "bert": BERTModel,
            "roberta": RoBERTaModel,
            "distilroberta": DistilRoBERTaModel,
            "sbert": SBERTModel,
        }

        model_class = model_map.get(model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {model_type}")

        model = model_class(self.config)
        model.train(train_df, test_df)

        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ESG Models")
    parser.add_argument("--model", type=str, required=False, help="Model to train: bert, roberta, sbert, socialbert, etc.")
    args = parser.parse_args()

    pipeline = TrainPipeline(model_override=args.model)
    pipeline.run()