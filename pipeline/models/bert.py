import wandb
import numpy as np
import evaluate
from sklearn.metrics import precision_recall_fscore_support

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

from models.base import BaseModel
import numpy as np

from utils.memory_logs import MemoryLoggingCallback

accuracy_metric = evaluate.load("accuracy") 

def compute_metrics(eval_pred):
    """Computes accuracy, precision, recall, and F1-score"""
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

class BERTModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = BertTokenizer.from_pretrained(config["bert"]["model_name"]) #("yiyanghkust/finbert-esg") #
        self.model = BertForSequenceClassification.from_pretrained(
           config["bert"]["model_name"], num_labels=config["bert"]["num_labels"]
        )

    def tokenize_data(self, df):
        """Efficiently tokenizes dataset and converts it to Hugging Face Dataset format"""
        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

        dataset = Dataset.from_pandas(df)  
        return dataset.map(preprocess_function, batched=True)    




    def train(self, train_df, test_df):
        train_dataset = self.tokenize_data(train_df)
        test_dataset = self.tokenize_data(test_df)


        if wandb.run is None:
            wandb.init(project=self.config["wandb"]["project_name"], config=self.config["bert"])

        training_args = TrainingArguments(
            seed=42, 
            output_dir="./results",
            num_train_epochs=self.config["bert"]["epochs"],
            per_device_train_batch_size=self.config["bert"]["batch_size"],
            per_device_eval_batch_size=self.config["bert"]["batch_size"],
            evaluation_strategy="epoch", 
            logging_dir="./logs",
            report_to="wandb",
            save_total_limit=2,
            save_strategy="epoch",
            load_best_model_at_end=True,  
            logging_strategy="epoch",
            logging_steps=1,  
            learning_rate=float(self.config["bert"]["learning_rate"]),
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

        trainer.save_model("./saved_bert_model") 

        print("Fine-tuned model saved successfully!")

        self.evaluate(test_df, trainer)
    

    def evaluate(self, test_df, trainer):
        """Evaluates the trained model and logs evaluation metrics"""
        test_dataset = self.tokenize_data(test_df)

        results = trainer.evaluate(test_dataset) 
        wandb.log(results)  
        print("Evaluation Results:", results)
        return results

  