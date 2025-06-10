import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer


class DataLoader:
    def __init__(self, config):
        self.config = config

        self.classify_filepath = config["data"]["filepath"]
        self.test_size = config["data"].get("test_size", 0.2)


    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


    def load_pretrain_data(self):
        """Loads dataset for self-supervised pretraining (unlabeled)"""
        df = pd.read_csv(self.pretrain_filepath)
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        if "sentence" in df.columns:  
            df = df.rename(columns={"sentence": "text"})  # Adjust column name
        df = df[["text"]].dropna()  # Only 'text' column for unsupervised learning
        return df

    def load_classification_data(self):
        print("✅ Trying to load:", self.config["data"]["filepath"])

        """Loads dataset for fine-tuning the classifier (human-annotated)"""
        df = pd.read_csv(self.classify_filepath)
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        
        # Ensure the dataset contains 'text' and 'label'
        df = df.rename(columns={"soc": "label", "gov": "label", "env": "label"})  
        df = df[["text", "label"]].dropna()
        df["label"] = df["label"].astype(int)


        return df

    def train_test_split(self, df):
        """Splits dataset into train and test sets with proper size checks."""
        if len(df) > 1:
            test_size = min(self.test_size, max(1 / len(df), 0.5))  # ✅ Ensure valid test size
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        else:
            train_df, test_df = df, df  # ✅ Use full dataset if too small

        return train_df, test_df