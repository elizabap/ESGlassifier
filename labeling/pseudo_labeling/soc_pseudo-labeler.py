import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import argparse
import os

def classify_sentences(model_dir, base_model, data_path, output_path, text_column="sentence",
                        batch_size=32, threshold=0.0, add_source_column=True, minimal_output=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📡 Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Could not find dataset at: {data_path}")

    df = pd.read_csv(data_path)
    if text_column not in df.columns:
        raise ValueError(f"❌ Column '{text_column}' not found in dataset. Available columns: {df.columns.tolist()}")

    texts = df[text_column].astype(str).tolist()
    predicted_labels = []
    all_probs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="🔮 Classifying", ncols=80):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

            all_probs.extend(probs.tolist())
            predicted_labels.extend(probs.argmax(axis=1).tolist())

    df["pseudo_label"] = predicted_labels
    for i in range(len(all_probs[0])):
        df[f"prob_label_{i}"] = [prob[i] for prob in all_probs]

    if threshold > 0.0:
        max_conf = df[[f"prob_label_{i}" for i in range(len(all_probs[0]))]].max(axis=1)
        df = df[max_conf >= threshold]
        print(f"📉 Filtered by threshold={threshold}: {len(df)} rows retained.")

    if minimal_output:
        df[[text_column, "pseudo_label"]].rename(columns={"pseudo_label": "soc_label"}).to_csv(output_path, index=False)
        print(f"\n✅ Done! Saved in minimal CSV format to: {output_path}")
    else:
        if add_source_column:
            df["label_source"] = "pseudo"
        df.to_csv(output_path, index=False)
        print(f"\n✅ Done! Full classification saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify sentences using a fine-tuned Social model for pseudo-labeling.")
    parser.add_argument("--model_dir", type=str, default="../finetuned_cv_models/soc_roberta_cv/checkpoint-300",
                        help="Path to the fine-tuned model to be used.")
    parser.add_argument("--base_model", type=str, default="roberta-base",
                        help="Name of the base model used during training.")
    parser.add_argument("--data_path", type=str, default="../data/base_data_train_S.csv",
                        help="Path to the CSV file to be classified.")
    parser.add_argument("--output_path", type=str, default="../data/base_data_train_S-pseudo.csv",
                        help="Path to save the pseudo-labeled data.")
    parser.add_argument("--text_column", type=str, default="sentence",
                        help="Name of the column containing the sentences.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for classification.")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Minimum probability required to keep a pseudo-label (0.0 = no filtering).")
    parser.add_argument("--no_source_column", action="store_true",
                        help="Do not add the 'label_source' column.")
    parser.add_argument("--minimal_output", action="store_true",
                        help="Only save the 'sentence' and 'soc_label' columns.")

    args = parser.parse_args()

    classify_sentences(
        model_dir=args.model_dir,
        base_model=args.base_model,
        data_path=args.data_path,
        output_path=args.output_path,
        text_column=args.text_column,
        batch_size=args.batch_size,
        threshold=args.threshold,
        add_source_column=not args.no_source_column,
        minimal_output=args.minimal_output
    )
