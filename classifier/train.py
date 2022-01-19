from torch.optim import Adam
import torch, os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from utils.model import BertClassifier
from utils.datasets import Dataset


def pred_acc(original, predicted):
    return torch.round(predicted).eq(original).sum().numpy() / len(original)


def train(
    model,
    name,
    texts,
    labels,
    learning_rate=1e-6,
    epochs=2,
    save_dir="output",
    classes=2,
    multi_label=False,
):
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(name)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.33, random_state=42
    )
    train, val = Dataset(tokenizer, X_train, y_train), Dataset(
        tokenizer, X_test, y_test
    )

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    if multi_label:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:

        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)
            if multi_label:
                output = output.to(torch.float32)
                train_label = train_label.to(torch.float32)
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            if not multi_label:
                acc = (output.argmax(dim=1) == train_label).sum().item()
            else:
                preds = torch.where(output > 0.5, 1, 0)
                acc = pred_acc(output, train_label)
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)

                output = model(input_id, mask)

                if multi_label:
                    output = output.to(torch.float32)
                    val_label = val_label.to(torch.float32)
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                if not multi_label:
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                else:
                    preds = torch.where(output > 0.5, 1, 0)
                    acc = pred_acc(output, val_label)
            total_acc_train += acc

        print(
            "\n".join(
                [
                    f"Epochs: {epoch_num + 1}",
                    f"Train Loss: {total_loss_train / len(X_train): .3f}",
                    f"Train Accuracy: {total_acc_train / len(X_train): .3f}",
                    f"Val Loss: {total_loss_val / len(X_test): .3f}",
                    f"Val Accuracy: {total_acc_val / len(X_test): .3f}",
                ]
            )
        )
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))


def flatten_list(l):
    out = []
    for item in l:
        if isinstance(item, list):
            out.extend(flatten_list(item))
        else:
            out.append(item)
    return out


if __name__ == "__main__":
    import pandas as pd
    import re
    from ast import literal_eval

    df = pd.read_csv("TrainData.csv", index_col=False)
    df = df[df["labels"] != "[0, 0, 0, 0, 0, 0]"]
    df["labels"] = df["labels"].apply(lambda x: literal_eval(x))
    name = "bert-base-cased"
    texts = ["Good", "Bad"]*10 #df["Text"].tolist()
    labels = [1, 0]*10 #df["labels"].tolist()
    model = BertClassifier(classes=2)
    train(
        model,
        name,
        texts,
        labels,
        save_dir="output",
        classes=2,
        multi_label=False,
    )