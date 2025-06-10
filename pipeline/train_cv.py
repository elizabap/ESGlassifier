import pandas as pd
import numpy as np
import torch
import wandb
import evaluate
import yaml
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from datasets import Dataset
from utils.data_loader import DataLoader
from utils.seed import set_seed

accuracy_metric = evaluate.load("accuracy")

def sanitize_model_name(model_name):
    return model_name.replace("/", "_").replace("-", "_")

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

def tokenize_dataset(df, tokenizer):
    dataset = Dataset.from_pandas(df)
    return dataset.map(lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512), batched=True)

def run_cv_experiment(
    df: pd.DataFrame,
    model_name="distilroberta-base",
    epochs=3, 
    batch_size=16,
    learning_rate=5e-5,
    seeds=[42],
    n_splits=5,
    project_name="esg-cv"
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    safe_model_name = sanitize_model_name(model_name)
    metrics_all = []

    for seed in seeds:
        set_seed(seed)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for fold, (train_idx, test_idx) in enumerate(skf.split(df, df["label"])):
            print(f"\n🚀 Fold {fold + 1}/{n_splits} (seed={seed}, model={model_name})")

            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            train_dataset = tokenize_dataset(train_df, tokenizer)
            test_dataset = tokenize_dataset(test_df, tokenizer)

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            run_name = f"{safe_model_name}_seed{seed}_fold{fold + 1}"

            wandb.init(
                project=project_name,
                name=run_name,
                reinit=True,
                config={
                    "fold": fold + 1,
                    "seed": seed,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "model_name": model_name
                }
            )

            training_args = TrainingArguments(
                output_dir=f"./results/{run_name}",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=100,
                seed=seed,
                report_to="wandb",
                fp16=torch.cuda.is_available()
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=compute_metrics
            )

            trainer.train()
            eval_results = trainer.evaluate()
            eval_results.update({
                "seed": seed,
                "fold": fold + 1,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs
            })
            print(f"📊 Fold {fold+1}, Seed {seed} Results: {eval_results}")
            wandb.log(eval_results)

            metrics_all.append(eval_results)
            wandb.finish()

            del trainer, model, train_dataset, test_dataset

    results_df = pd.DataFrame(metrics_all)
    os.makedirs("results", exist_ok=True)

    results_df.to_csv(f"results/raw_{safe_model_name}_cv.csv", index=False)
    summary = results_df.groupby(["seed"])[["eval_accuracy", "eval_f1_score", "eval_precision", "eval_recall"]].agg(["mean", "std"])
    summary.to_csv(f"results/summary_{safe_model_name}_cv.csv")

    print("\n Cross-validation summary (mean ± std per seed):")
    print(summary)

    wandb.init(project=project_name, name=f"{safe_model_name}_cv_summary", reinit=True)
    wandb.log({"cv_summary_table": wandb.Table(dataframe=results_df)})
    wandb.finish()

    return summary

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--model", type=str, default="distilroberta-base")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--project", type=str, default="esg-cv")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataloader = DataLoader(config)
    df = dataloader.load_classification_data()
    df["text"] = df["text"].apply(lambda x: dataloader.clean_text(x))

    run_cv_experiment(
        df=df,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seeds=args.seeds,
        project_name=args.project
    )
