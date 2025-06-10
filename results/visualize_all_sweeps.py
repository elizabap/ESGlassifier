import wandb
import pandas as pd
import matplotlib.pyplot as plt

# 🛠️ CONFIG — Set your sweep path and metric to sort by
SWEEP_PATH = "mastersveenpan/Governance/0wdge5el"  
SORT_BY = "eval_f1_score"  

# 🎣 Load sweep and runs
api = wandb.Api()
sweep = api.sweep(SWEEP_PATH)
runs = sweep.runs

# 🧾 Collect metrics from each run
records = []
for run in runs:
    config = run.config
    summary = run.summary
    if "eval_loss" in summary and "eval_accuracy" in summary:
        records.append({
            "run_id": run.id,
            "run_name": run.name,
            "learning_rate": config.get("learning_rate"),
            "batch_size": config.get("batch_size"),
            "epochs": config.get("epochs"),
            "eval_loss": summary.get("eval_loss"),
            "eval_accuracy": summary.get("eval_accuracy"),
            "eval_precision": summary.get("eval_precision"),
            "eval_recall": summary.get("eval_recall"),
            "eval_f1_score": summary.get("eval_f1_score")
        })

df = pd.DataFrame(records)

# 📊 Sort by selected metric
df_sorted = df.sort_values(by=SORT_BY, ascending=False).reset_index(drop=True)

# 📋 Print full table
print(f"\n📋 All runs sorted by {SORT_BY}:")
print(df_sorted.to_string(index=False))

# 💾 Save to CSV
filename = f"sweep_runs_sorted_by_{SORT_BY}.csv"
df_sorted.to_csv(filename, index=False)
print(f"\n💾 Saved to '{filename}'")

# 🎯 Top 5
print(f"\n🎯 Top 5 runs by {SORT_BY}:")
print(df_sorted.head(5).to_string(index=False))

# 🏆 Best run summary
best = df_sorted.iloc[0]
print(f"\n🏆 Best run by {SORT_BY}:")
print(f"  ID:          {best['run_id']}")
print(f"  Accuracy:    {best['eval_accuracy']:.4f}")
print(f"  F1 Score:    {best['eval_f1_score']:.4f}")
print(f"  Precision:   {best['eval_precision']:.4f}")
print(f"  Recall:      {best['eval_recall']:.4f}")
print(f"  LR:          {best['learning_rate']}")
print(f"  Batch size:  {best['batch_size']}")
print(f"  Epochs:      {best['epochs']}")

# 📈 Plot F1 Score vs Learning Rate for each batch size
plt.figure(figsize=(10, 6))

for batch in sorted(df["batch_size"].dropna().unique()):
    subset = df[df["batch_size"] == batch]
    plt.plot(
        subset["learning_rate"],
        subset[SORT_BY],
        marker="o",
        linestyle='-',
        label=f"Batch size {batch}"
    )

# 🔺 Highlight best run
plt.scatter(
    best["learning_rate"],
    best[SORT_BY],
    s=150,
    color='red',
    label="🏆 Best run",
    zorder=5
)

plt.title(f"{SORT_BY.replace('_', ' ').title()} by Learning Rate and Batch Size")
plt.xlabel("Learning Rate")
plt.ylabel(SORT_BY.replace("_", " ").title())
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
