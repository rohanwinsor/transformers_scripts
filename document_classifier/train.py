from transformers import LayoutLMForSequenceClassification, AdamW
import torch
from utils.datasets import create_data


def train(data_path, model_path, epochs, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        train_size,
        validation_size,
        test_size,
        label2idx,
    ) = create_data(data_path)
    model = LayoutLMForSequenceClassification.from_pretrained(
        "microsoft/layoutlm-base-uncased", num_labels=len(label2idx)
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    global_step = 0
    num_train_epochs = epochs
    t_total = len(train_dataloader) * num_train_epochs  # total number of training steps

    for epoch in range(num_train_epochs):
        print("Epoch:", epoch)
        running_loss = 0.0
        correct = 0
        model.train()
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            bbox = batch["bbox"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            # forward pass
            outputs = model(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss

            running_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            correct += (predictions == labels).float().sum()

            # backward pass to get the gradients
            loss.backward()

            # update
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        print("Loss:", running_loss / batch["input_ids"].shape[0])
        accuracy = 100 * correct / train_size
        print("Training accuracy:", accuracy.item())

        if epoch % 5 == 0:
            model.eval()

            correct = 0
            for batch in validation_dataloader:
                input_ids = batch["input_ids"].to(device)
                bbox = batch["bbox"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["label"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    bbox=bbox,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                predictions = outputs.logits.argmax(-1)
                correct += (predictions == labels).float().sum()

            accuracy = 100 * correct / validation_size
            print("Validation accuracy:", accuracy.item())

    model.eval()

    correct = 0
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        bbox = batch["bbox"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)
        outputs = model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        predictions = outputs.logits.argmax(-1)
        correct += (predictions == labels).float().sum()

    accuracy = 100 * correct / test_size
    print("Testing accuracy:", accuracy.item())
    model.save_pretrained(model_path)


if __name__ == "__main__":
    train(
        data_path="data/train.json",
        model_path="output",
        epochs=10,
    )
