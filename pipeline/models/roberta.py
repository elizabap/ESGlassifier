from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from models.base import BaseModel
import wandb
import numpy as np
import evaluate
from sklearn.metrics import precision_recall_fscore_support
import os

from utils.memory_logs import MemoryLoggingCallback


accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
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

class RoBERTaModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = RobertaTokenizer.from_pretrained(config["roberta"]["model_name"])
        self.model = RobertaForSequenceClassification.from_pretrained(
            config["roberta"]["model_name"], num_labels=config["roberta"]["num_labels"]
        )
        self.config = config

    def tokenize_data(self, df):
        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        dataset = Dataset.from_pandas(df)
        return dataset.map(preprocess_function, batched=True)

    def train(self, train_df, test_df):
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.1,
            stratify=train_df["label"], 
            random_state=42
        )

        train_dataset = self.tokenize_data(train_df)
        val_dataset = self.tokenize_data(val_df)
        test_dataset = self.tokenize_data(test_df)

        print(f"\n🧪 Train set size: {len(train_dataset)}") 
        print(f"🧪 Test set size:  {len(test_dataset)}")
        print("🚀 Starting training...")

        print("\n🚀 Training with hyperparameters:")
        print(f"  learning_rate: {self.config['roberta']['learning_rate']}")
        print(f"  batch_size:    {self.config['roberta']['batch_size']}")
        print(f"  epochs:        {self.config['roberta']['epochs']}")

        flat_config = {f"roberta_{k}": v for k, v in self.config["roberta"].items()}
        if wandb.run is None:
            wandb.init(project=self.config["wandb"]["project_name"], config=flat_config)

        training_args = TrainingArguments(
            seed=42,
            output_dir="./results",
            num_train_epochs=self.config["roberta"]["epochs"],
            per_device_train_batch_size=self.config["roberta"]["batch_size"],
            per_device_eval_batch_size=self.config["roberta"]["batch_size"],
            evaluation_strategy="epoch",
            logging_dir="./logs",
            report_to="wandb",
            save_total_limit=2,
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_strategy="epoch",
            logging_steps=1,
            learning_rate=float(self.config["roberta"]["learning_rate"]),
            weight_decay=0.01,
            fp16=True,  
            gradient_accumulation_steps=1,  
            
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset, 
            compute_metrics=compute_metrics,
            callbacks=[MemoryLoggingCallback()]

        )

        trainer.train()
        output_dir = f"./saved_roberta_model_{wandb.run.id}"
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        self.evaluate(test_df, trainer)

        
    def evaluate(self, test_df, trainer):
        test_dataset = self.tokenize_data(test_df)
        results = trainer.evaluate(test_dataset)

        print("📊 Evaluation completed. Results:") 
        print(results)

        wandb.log(results)
        print("Evaluation Results:", results)
