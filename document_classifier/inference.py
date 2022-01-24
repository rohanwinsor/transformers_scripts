import os
import json
from transformers import LayoutLMForSequenceClassification
import torch
import pandas as pd
from utils.dataloader import Dataset, apply_ocr
from utils.encode import encode_example
from datasets import Features, Sequence, ClassLabel, Value, Array2D


features = Features(
    {
        "input_ids": Sequence(feature=Value(dtype="int64")),
        "bbox": Array2D(dtype="int64", shape=(512, 4)),
        "attention_mask": Sequence(Value(dtype="int64")),
        "token_type_ids": Sequence(Value(dtype="int64")),
        "label": ClassLabel(names=["refuted", "entailed"]),
        "image_path": Value(dtype="string"),
        "words": Sequence(feature=Value(dtype="string")),
    }
)


def inference(model_path, input_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(model_path, "label2idx.json"), "r") as f:
        label2idx = json.load(f)
    model = LayoutLMForSequenceClassification.from_pretrained(
        "microsoft/layoutlm-base-uncased", num_labels=len(label2idx)
    )
    model.to(device)
    if type(input_path) == str:
        input_path = [input_path]
    test_data = pd.DataFrame(
        {
            "image_path": input_path,
            "label": [list(label2idx.keys())[0]] * len(input_path),
        }
    )
    test_size = len(test_data)
    test_dataset = Dataset.from_pandas(test_data)
    updated_test_dataset = test_dataset.map(apply_ocr)
    # updated_test_dataset = updated_test_dataset.remove_columns("__index_level_0__")

    encoded_test_dataset = updated_test_dataset.map(
        lambda example: encode_example(example, label2idx), features=features
    )

    encoded_test_dataset.set_format(
        type="torch",
        columns=["input_ids", "bbox", "attention_mask", "token_type_ids", "label"],
    )
    test_dataloader = torch.utils.data.DataLoader(
        encoded_test_dataset, batch_size=1, shuffle=True
    )
    idx2label = {y: x for x, y in label2idx.items()}
    model.eval()
    output = []
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        bbox = batch["bbox"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        outputs = model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        predictions = outputs.logits.argmax(-1)
        output.extend(predictions.tolist())
    return [idx2label[i] for i in output]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--input_path", action="store", type=str, required=True)
    parser.add_argument("--model_path", action="store", type=str, required=True)
    args = parser.parse_args()

    out = inference(args.model_path, args.input_path, device=None)
    print("OUTPUT ::", out)
