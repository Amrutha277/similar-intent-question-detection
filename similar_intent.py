# siamese_bert_intent.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import argparse
import os
import random

# 1) Dataset Definition
class FAQDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_length: int = 128):
        """
        df must have columns: ['question1','question2','label']
        label: 1 for similar intents, 0 otherwise
        """
        self.q1 = df['question1'].tolist()
        self.q2 = df['question2'].tolist()
        self.labels = df['label'].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        q1, q2, label = self.q1[idx], self.q2[idx], self.labels[idx]
        enc1 = self.tokenizer(
            q1,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        enc2 = self.tokenizer(
            q2,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids1': enc1['input_ids'].squeeze(),
            'attention_mask1': enc1['attention_mask'].squeeze(),
            'input_ids2': enc2['input_ids'].squeeze(),
            'attention_mask2': enc2['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }

# 2) Model Definition
class SiameseBert(nn.Module):
    def __init__(self, pretrained_model_name: str = 'bert-base-uncased', dropout: float = 0.1):
        super(SiameseBert, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size
        # classifier on [h1; h2; |h1-h2|; h1*h2]
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        o1 = self.bert(input_ids=input_ids1, attention_mask=attention_mask1).pooler_output
        o2 = self.bert(input_ids=input_ids2, attention_mask=attention_mask2).pooler_output

        diff = torch.abs(o1 - o2)
        mult = o1 * o2
        features = torch.cat([o1, o2, diff, mult], dim=1)
        logits = self.classifier(features).squeeze(-1)  # [batch]
        return logits

# 3) Training & Evaluation Functions
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(device)
        logits = model(**inputs)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_model(model, dataloader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            logits = model(**inputs)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).long().cpu().numpy()
            preds.extend(pred)
            true.extend(labels.cpu().numpy().astype(int))
    f1 = f1_score(true, preds)
    return f1

# 4) Main Script
def main(args):
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    df = pd.read_csv(args.train_csv)  # expects columns question1, question2, label
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    dataset = FAQDataset(df, tokenizer, max_length=args.max_length)
    # train/val split
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # model
    model = SiameseBert(pretrained_model_name=args.bert_model, dropout=args.dropout)
    model.to(device)

    # optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_f1 = eval_model(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} — train_loss: {train_loss:.4f} — val_f1: {val_f1:.4f}")

        # save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), args.output_path)
            print(f"→ New best F1! Model saved to {args.output_path}")

    print(f"Training complete. Best val F1: {best_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese BERT for FAQ intent similarity")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to CSV with question1,question2,label")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--output_path", type=str, default="best_model.pt", help="Where to save best model")
    parser.add_argument("--max_length", type=int, default=128, help="Max token length")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    main(args)
