from transformers import LayoutLMTokenizer
import torch
from datasets import Features, Sequence, ClassLabel, Value, Array2D

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")


def encode_example(example, max_seq_length=512, pad_token_box=[0, 0, 0, 0]):
    words = example["words"]
    normalized_word_boxes = example["bbox"]

    assert len(words) == len(normalized_word_boxes)

    token_boxes = []
    for word, box in zip(words, normalized_word_boxes):
        word_tokens = tokenizer.tokenize(word)
        token_boxes.extend([box] * len(word_tokens))

    # Truncation of token_boxes
    special_tokens_count = 2
    if len(token_boxes) > max_seq_length - special_tokens_count:
        token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

    # add bounding boxes of cls + sep tokens
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

    encoding = tokenizer(" ".join(words), padding="max_length", truncation=True)
    # Padding of token_boxes up the bounding boxes to the sequence length.
    input_ids = tokenizer(" ".join(words), truncation=True)["input_ids"]
    padding_length = max_seq_length - len(input_ids)
    token_boxes += [pad_token_box] * padding_length
    encoding["bbox"] = token_boxes
    encoding["label"] = label2idx[example["label"]]

    assert len(encoding["input_ids"]) == max_seq_length
    assert len(encoding["attention_mask"]) == max_seq_length
    assert len(encoding["token_type_ids"]) == max_seq_length
    assert len(encoding["bbox"]) == max_seq_length

    return encoding


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
