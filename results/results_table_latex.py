import wandb
import pandas as pd
import re

SWEEP_PATH = "mastersveenpan/domain/sweep_id"

api = wandb.Api()
sweep = api.sweep(SWEEP_PATH)
runs = sweep.runs

records = []
for run in runs:
    config = run.config
    summary = run.summary

    match = re.search(r"(\d+)$", run.name)
    if match and "eval_f1_score" in summary:
        run_number = int(match.group(1))
        lr = config.get("learning_rate")
        formatted_lr = f"{int(lr * 1e6 / 10)}e-5"

        records.append({
            "run_id": run.id,
            "run_name": run.name,
            "run_number": run_number,
            "learning_rate": formatted_lr,
            "batch_size": config.get("batch_size"),
            "epochs": config.get("epochs"),
            "eval_accuracy": summary.get("eval_accuracy"),
            "eval_precision": summary.get("eval_precision"),
            "eval_recall": summary.get("eval_recall"),
            "eval_f1_score": summary.get("eval_f1_score")
        })

df = pd.DataFrame(records)
df_sorted = df.sort_values(by="run_number").reset_index(drop=True)

# ⭐ Find index of best F1-score
best_f1_index = df_sorted["eval_f1_score"].idxmax()

# 📋 Format all values as strings (for LaTeX)
def format_row(row, bold=False):
    def bf(val):
        return f"\\textbf{{{val}}}" if bold else str(val)
    return {
        "Run": bf(row["run_name"]),
        "LR": bf(row["learning_rate"]),
        "Batch": bf(row["batch_size"]),
        "Epochs": bf(row["epochs"]),
        "Accuracy": bf(f"{row['eval_accuracy']:.4f}"),
        "Precision": bf(f"{row['eval_precision']:.4f}"),
        "Recall": bf(f"{row['eval_recall']:.4f}"),
        "F1": bf(f"{row['eval_f1_score']:.4f}")
    }

formatted_rows = []
for i, row in df_sorted.iterrows():
    bold = (i == best_f1_index)
    formatted_rows.append(format_row(row, bold=bold))

latex_df = pd.DataFrame(formatted_rows, columns=[
    "Run", "LR", "Batch", "Epochs", "Accuracy", "Precision", "Recall", "F1"
])

# 🧾 Generate LaTeX table with bolded best row
latex_table = latex_df.to_latex(index=False, escape=False, caption="Results sorted after run-number", label="tab:results")

# 💾 Save to file
with open("latex_tables/experiment_2/Gov_BERT_2k_sentiment.tex", "w") as f:
    f.write(latex_table)

print("Table created named results_table_sorted_by_run.tex")
