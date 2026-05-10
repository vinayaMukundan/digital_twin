import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup 
from torch.optim import AdamW 
# from sklearn.model_selection import train_test_split

# --- 1. Configuration & Hyperparameters ---
DATA_PATH = 'data/preprocessed.csv'
MODEL_SAVE_PATH = 'models/bert_mental_health'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# NEW: Data Limit Variable
# Set to None for full data, or an integer (e.g., 500) for a subset
DATA_LIMIT = None 

# UPDATED: GPU Logic for MacBook Air (MPS)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Ensure save directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# --- 2. Dataset Class ---
class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            str(self.texts[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

# --- 3. Data Loading & Preprocessing ---
def load_data():
    # Load specific files created by preprocess.py
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')
    
    if DATA_LIMIT is not None:
        train_df = train_df.head(DATA_LIMIT)
        val_df = val_df.head(int(DATA_LIMIT * 0.1))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # No more train_test_split here!
    train_data = MentalHealthDataset(train_df['text'].values, train_df['label'].values, tokenizer, MAX_LEN)
    val_data = MentalHealthDataset(val_df['text'].values, val_df['label'].values, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    
    return train_loader, val_loader, tokenizer

# --- 4. Training Function ---
def train_model():
    train_loader, val_loader, tokenizer = load_data()

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=2
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    print(f"Starting training on {device}...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # These will now be sent to the Mac GPU (mps)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")

    # --- 5. Save the Model ---
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
