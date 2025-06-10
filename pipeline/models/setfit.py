import wandb
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from setfit import SetFitModel, Trainer
from models.base import BaseModel  


def compute_metrics(predictions, references):
    accuracy = np.mean(np.array(predictions) == np.array(references))
    precision, recall, f1, _ = precision_recall_fscore_support(references, predictions, average="binary")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


class SBERTModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = SetFitModel.from_pretrained(config["sbert"]["model_name"])

    def tokenize_data(self, df):
        return Dataset.from_pandas(df[["text", "label"]])

    def few_shot_split(self, df, shots_per_class=4, seed=42):
        pos = df[df["label"] == 1]
        neg = df[df["label"] == 0]

        n_pos = min(shots_per_class, len(pos))
        n_neg = min(shots_per_class, len(neg))

        pos_sampled = pos.sample(n=n_pos, random_state=seed)
        neg_sampled = neg.sample(n=n_neg, random_state=seed)

        train_df = pd.concat([pos_sampled, neg_sampled]).sample(frac=1, random_state=seed)
        test_df = df.drop(train_df.index).reset_index(drop=True)

        return train_df.reset_index(drop=True), test_df

    def train(self, train_df, test_df=None, few_shot_mode=False):
        if wandb.run is None:
            wandb.init(
                project=self.config["wandb"]["project_name"],
            )
            wandb.config.update(self.config)

        if not few_shot_mode:
            train_df, _ = self.few_shot_split(
                train_df,
                shots_per_class=self.config["sbert"]["shots_per_class"],
                seed=self.config["sbert"]["seed"]
            )

        train_dataset = self.tokenize_data(train_df)
        eval_dataset = self.tokenize_data(test_df) if test_df is not None else None

        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train(
            num_iterations=self.config["sbert"]["num_iterations"],
            batch_size=self.config["sbert"]["batch_size"]
        )

        self.model.save_pretrained(self.config["sbert"]["model_save_path"])
        print("SBERT model saved")

        if test_df is not None:
            self.evaluate(test_df)

        if wandb.run is not None:
            wandb.finish()

    def evaluate(self, test_df):
        test_dataset = self.tokenize_data(test_df)
        predictions = self.model.predict([example["text"] for example in test_dataset])
        references = [example["label"] for example in test_dataset]

        eval_metrics = compute_metrics(predictions, references)
        
        if wandb.run is not None:
            wandb.log(eval_metrics)

        print("Evaluation Results:", eval_metrics)
        return eval_metrics

    def run_multiple_seeds(self, df, shot_counts, seeds):
        all_results = []
        for shots in shot_counts:
            for seed in seeds:
                print(f"Running {shots}-shot with seed {seed}")
                train_df, test_df = self.few_shot_split(df, shots_per_class=shots, seed=seed)
                self.train(train_df, test_df, few_shot_mode=True)
                metrics = self.evaluate(test_df)
                all_results.append({
                    "shots": shots,
                    "seed": seed,
                    "f1_score": metrics["f1_score"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"]
                })
        return pd.DataFrame(all_results)

    def cross_validate_few_shot(self, df, shot_counts, seeds, k=5):
        all_results = []

        for shots in shot_counts:
            for seed in seeds:
                print(f"\n🎯 Running {shots}-shot with seed {seed}")
                skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

                fold = 1
                for train_index, test_index in skf.split(df["text"], df["label"]):
                    print(f"🔁 Fold {fold}/{k} (shots={shots}, seed={seed})")

                    fold_train_df = df.iloc[train_index].reset_index(drop=True)
                    fold_test_df = df.iloc[test_index].reset_index(drop=True)

                    few_shot_train_df, _ = self.few_shot_split(fold_train_df, shots_per_class=shots, seed=seed)

                    if wandb.run is None:
                        wandb.init(project=self.config["wandb"]["project_name"])
                        wandb.config.update(self.config)
                    else:
                        wandb.run.name = f"cv_seed{seed}_shots{shots}_fold{fold}"

                    self.train(few_shot_train_df, fold_test_df, few_shot_mode=True)
                    metrics = self.evaluate(fold_test_df)
                    metrics.update({
                        "fold": fold,
                        "seed": seed,
                        "shots": shots
                    })
                    all_results.append(metrics)

                    fold += 1
                    if wandb.run is not None:
                        wandb.finish()
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("few_shot_cv_results.csv", index=False)
        return results_df