import os
import json
from tranformers import LayoutLMForSequenceClassification
import torch

def inference(model_path):
    
    with open(os.path.join(model_path, "labels.json"), "r") as f:
        num_labels = json.load()["count"]
    model = LayoutLMForSequenceClassification.from_pretrained(
        "microsoft/layoutlm-base-uncased", num_labels=num_labels
    )
    test_data = pd.concat([X_test, y_test], axis=1)
    test_size = len(test_data)
    test_dataset = Dataset.from_pandas(test_data)
    updated_test_dataset = test_dataset.map(apply_ocr)
    updated_test_dataset = updated_test_dataset.remove_columns("__index_level_0__")

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