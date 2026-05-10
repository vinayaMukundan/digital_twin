import pandas as pd
import torch
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def evaluate_model():
    model_path = "./models/bert_mental_health"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        return

    # 1. Load the model and tokenizer
    print("Loading model...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

   # 2. Load the dedicated Test Set
    print("Loading test data...")
    df_test = pd.read_csv("data/test.csv") # Points to the specific test file
    df_test['label'] = df_test['label'].astype(int)
    
    def tokenize_func(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing data...")
    test_dataset = Dataset.from_pandas(df_test)
    test_dataset = test_dataset.map(tokenize_func, batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # 3. Define Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        acc = accuracy_score(labels, predictions)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    # 4. Explicitly define TrainingArguments (This prevents the Trainer from guessing)
    training_args = TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=16,
        do_train=False,
        do_eval=True,
        report_to="none" # Prevents trying to log to wandb/tensorboard
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    # 6. Run Evaluation
    print("Running evaluation...")
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    
    print("\n" + "="*30)
    print("VITA-TWIN MODEL PERFORMANCE")
    print("="*30)
    print(f"Accuracy:  {metrics['eval_accuracy']:.4f}")
    print(f"Precision: {metrics['eval_precision']:.4f}")
    print(f"Recall:    {metrics['eval_recall']:.4f}")
    print(f"F1 Score:  {metrics['eval_f1']:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_model()
