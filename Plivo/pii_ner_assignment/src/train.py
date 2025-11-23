import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from labels import LABELS        # list of BIO tags, e.g. ["O","B-PERSON_NAME",...]
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=8)

    # 🔥 tuned hyperparameters
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--max_length", type=int, default=128)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Dataset with our chosen max_length
    train_ds = PIIDataset(
        args.train,
        tokenizer,
        LABELS,
        max_length=args.max_length,
        is_train=True,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    # Model with custom dropout
    model = create_model(
        args.model_name,
        dropout=args.dropout,
    )
    model.to(args.device)
    model.train()

    # 🔥 Build class weights for PII emphasis
    pii_base_labels = {"CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE"}

    weights = []
    num_labels = len(LABELS)
    for lab in LABELS:
        if lab == "O":
            # downweight non-entity tokens a bit
            weights.append(0.5)
        else:
            # strip BIO prefix if present
            base = lab.split("-")[-1]
            if base in pii_base_labels:
                # upweight PII tokens
                weights.append(2.0)
            else:
                # other entities (CITY, LOCATION, etc.)
                weights.append(1.0)

    class_weights = torch.tensor(weights, dtype=torch.float, device=args.device)
    loss_fct = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=-100,  # standard ignore index for masked tokens
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):

            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            # HF model forward: we'll ignore its internal loss and compute our own
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = outputs.logits  # (batch, seq_len, num_labels)

            # 🔥 our weighted loss
            loss = loss_fct(
                logits.view(-1, num_labels),
                labels.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_dl))
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved model + tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()