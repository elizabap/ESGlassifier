import wandb
import numpy as np
import evaluate
import os

from sklearn.metrics import precision_recall_fscore_support
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from models.base import BaseModel
from utils.memory_logs import MemoryLoggingCallback

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1 score"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

class DistilRoBERTaModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
        self.model = RobertaForSequenceClassification.from_pretrained(
            "distilroberta-base",
            num_labels=config["distilroberta"]["num_labels"]
        )

    def tokenize_data(self, df):
        """Tokenize using Hugging Face tokenizer"""
        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

        dataset = Dataset.from_pandas(df)
        return dataset.map(preprocess_function, batched=True)

    def train(self, train_df, test_df):
        train_dataset = self.tokenize_data(train_df)
        test_dataset = self.tokenize_data(test_df)

        print(f"\n🚀 Training DistilRoBERTa with parameters:")
        for k, v in self.config["distilroberta"].items():
            print(f"  {k}: {v}")

        flat_config = {f"distilroberta_{k}": v for k, v in self.config["distilroberta"].items()}
        if wandb.run is None:
            wandb.init(project=self.config["wandb"]["project_name"], config=flat_config)

        training_args = TrainingArguments(
            seed=42,
            output_dir="./results",
            num_train_epochs=self.config["distilroberta"]["epochs"],
            per_device_train_batch_size=self.config["distilroberta"]["batch_size"],
            per_device_eval_batch_size=self.config["distilroberta"]["batch_size"],
            evaluation_strategy="epoch",
            logging_dir="./logs",
            report_to="wandb",
            save_total_limit=2,
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_strategy="epoch",
            logging_steps=1,
            learning_rate=float(self.config["distilroberta"]["learning_rate"]),
            weight_decay=0.01,
            fp16=True,
            gradient_accumulation_steps=1
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            callbacks=[MemoryLoggingCallback()]

        )

        trainer.train()

        output_dir = f"./saved_distilroberta_model_{wandb.run.id}"
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)

        self.evaluate(test_df, trainer)

    def evaluate(self, test_df, trainer):
        """Evaluate and log results to W&B"""
        test_dataset = self.tokenize_data(test_df)
        results = trainer.evaluate(test_dataset)

        if "f1_score" in results:
            results["eval_f1_score"] = results.pop("f1_score")
        if "accuracy" in results:
            results["eval_accuracy"] = results.pop("accuracy")
        if "precision" in results:
            results["eval_precision"] = results.pop("precision")
        if "recall" in results:
            results["eval_recall"] = results.pop("recall")

        wandb.log(results)
        print("📊 Evaluation Results:", results)
        return results
