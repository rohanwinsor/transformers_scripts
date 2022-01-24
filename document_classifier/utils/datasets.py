import pandas as pd
import os, torch
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import pytesseract
from encode import encode_example
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D


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


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def apply_ocr(example):
    # get the image
    image = Image.open(example["image_path"])

    width, height = image.size

    # apply ocr to the image
    ocr_df = pytesseract.image_to_data(image, output_type="data.frame")
    float_cols = ocr_df.select_dtypes("float").columns
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r"^\s*$", np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)

    # get the words and actual (unnormalized) bounding boxes
    # words = [word for word in ocr_df.text if str(word) != 'nan'])
    words = list(ocr_df.text)
    words = [str(w) for w in words]
    coordinates = ocr_df[["left", "top", "width", "height"]]
    actual_boxes = []
    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
        actual_box = [
            x,
            y,
            x + w,
            y + h,
        ]  # we turn it into (left, top, left+width, top+height) to get the actual box
        actual_boxes.append(actual_box)

    # normalize the bounding boxes
    boxes = []
    for box in actual_boxes:
        boxes.append(normalize_box(box, width, height))

    # add as extra columns
    assert len(words) == len(boxes)
    example["words"] = words
    example["bbox"] = boxes
    return example


def create_data(dataset_path):
    labels = [label for label in os.listdir(dataset_path)]
    label2idx = {k: v for v, k in enumerate(labels)}
    data = pd.DataFrame()
    i = 0
    for label in os.listdir(dataset_path):
        count = 0
        for filename in os.listdir(dataset_path + "/" + label):

            if ".ipynb" not in filename and count < 200:
                data.at[i, "image_path"] = dataset_path + "/" + label + "/" + filename
                data.at[i, "label"] = label
                i = i + 1
                count = count + 1

    data = shuffle(data)
    data.head()
    X = data[["image_path"]]
    y = data[["label"]]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, stratify=y)

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_val, y_val, test_size=0.5, stratify=y_val
    )
    train_data = pd.concat([X_train, y_train], axis=1)
    valid_data = pd.concat([X_valid, y_valid], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_size = len(train_data)
    validation_size = len(valid_data)
    test_size = len(test_data)

    train_dataset = Dataset.from_pandas(train_data)
    updated_train_dataset = train_dataset.map(apply_ocr)

    valid_dataset = Dataset.from_pandas(valid_data)
    updated_valid_dataset = valid_dataset.map(apply_ocr)

    test_dataset = Dataset.from_pandas(test_data)
    updated_test_dataset = test_dataset.map(apply_ocr)

    # TODO: Check This
    updated_train_dataset = updated_train_dataset.remove_columns("__index_level_0__")
    updated_valid_dataset = updated_valid_dataset.remove_columns("__index_level_0__")
    updated_test_dataset = updated_test_dataset.remove_columns("__index_level_0__")

    encoded_train_dataset = updated_train_dataset.map(
        lambda example: encode_example(example), features=features
    )

    encoded_train_dataset.set_format(
        type="torch",
        columns=["input_ids", "bbox", "attention_mask", "token_type_ids", "label"],
    )

    encoded_valid_dataset = updated_valid_dataset.map(
        lambda example: encode_example(example), features=features
    )

    encoded_valid_dataset.set_format(
        type="torch",
        columns=["input_ids", "bbox", "attention_mask", "token_type_ids", "label"],
    )

    encoded_test_dataset = updated_test_dataset.map(
        lambda example: encode_example(example), features=features
    )

    encoded_test_dataset.set_format(
        type="torch",
        columns=["input_ids", "bbox", "attention_mask", "token_type_ids", "label"],
    )
    train_dataloader = torch.utils.data.DataLoader(
        encoded_train_dataset, batch_size=5, shuffle=True
    )
    validation_dataloader = torch.utils.data.DataLoader(
        encoded_valid_dataset, batch_size=2, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        encoded_test_dataset, batch_size=1, shuffle=True
    )
    return (
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        train_size,
        validation_size,
        test_size,
        label2idx,
    )
