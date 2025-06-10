import pandas as pd
import torch
import wandb
import evaluate
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import yaml
from pipeline.utils.data_loader import DataLoader
from utils.seed import set_seed

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

def tokenize_dataset(df, tokenizer):
    dataset = Dataset.from_pandas(df)
    return dataset.map(lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512), batched=True)

def run_cv_experiment(train_df, test_df, model_name, epochs, batch_size, lr, seeds, project_name, experiment_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    metrics_all = []

    for seed in seeds:
        set_seed(seed)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
            print(f"\n🚀 Fold {fold + 1}/5 (seed={seed})")
            train_fold = train_df.iloc[train_idx].reset_index(drop=True)
            val_fold = train_df.iloc[val_idx].reset_index(drop=True)

            train_dataset = tokenize_dataset(train_fold, tokenizer)
            val_dataset = tokenize_dataset(val_fold, tokenizer)
            final_test_dataset = tokenize_dataset(test_df, tokenizer)

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

            run_name = f"combined_{model_name}_seed{seed}_fold{fold + 1}_{experiment_name}"
            wandb.init(
                project=project_name,
                name=run_name,
                reinit=True,
                config={
                    "fold": fold + 1,
                    "seed": seed,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "model_name": model_name,
                    "experiment": experiment_name
                }
            )

            args = TrainingArguments(
                output_dir=f"./results/{run_name}",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                learning_rate=lr,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=100,
                seed=seed,
                report_to="wandb",
                fp16=torch.cuda.is_available()
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics
            )

            trainer.train()
            eval_results = trainer.evaluate(final_test_dataset)

            eval_results.update({
                "seed": seed,
                "fold": fold + 1,
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs
            })
            print(f"📊 Fold {fold+1}, Seed {seed} Test Results: {eval_results}")
            wandb.log(eval_results)

            metrics_all.append(eval_results)
            wandb.finish()

            del trainer, model, train_dataset, val_dataset, final_test_dataset
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(metrics_all)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(f"results/raw_combined_{model_name}_cv.csv", index=False)

    summary = results_df.groupby(["seed"])[["eval_accuracy", "eval_f1_score", "eval_precision", "eval_recall"]].agg(["mean", "std"])
    summary.to_csv(f"results/summary_combined_{model_name}_cv.csv")

    print("\n✅ Cross-validation summary (mean ± std per seed):")
    print(summary)

    return summary

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo_dataset", type=str, required=True)
    parser.add_argument("--human_dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="roberta-base")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--project", type=str, default="Governance-Combined-CV-2")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataloader = DataLoader(config)

    pseudo_df = pd.read_csv(args.pseudo_dataset)
    if "sentence" in pseudo_df.columns:
        pseudo_df = pseudo_df.rename(columns={"sentence": "text"})
    pseudo_df = pseudo_df.rename(columns={"gov": "label", "soc": "label", "env": "label"})
    pseudo_df = pseudo_df[["text", "label"]].dropna()
    pseudo_df["label"] = pseudo_df["label"].astype(int)
    pseudo_df["text"] = pseudo_df["text"].apply(lambda x: dataloader.clean_text(x))

    human_df = pd.read_csv(args.human_dataset)
    if "sentence" in human_df.columns:
        human_df = human_df.rename(columns={"sentence": "text"})
    human_df = human_df.rename(columns={"gov": "label", "soc": "label", "env": "label"})
    human_df = human_df[["text", "label"]].dropna()
    human_df["label"] = human_df["label"].astype(int)
    human_df["text"] = human_df["text"].apply(lambda x: dataloader.clean_text(x))

    run_cv_experiment(
        train_df=pseudo_df,
        test_df=human_df,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seeds=args.seeds,
        project_name=args.project,
        experiment_name="combined"
    )
