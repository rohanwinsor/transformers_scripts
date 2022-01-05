import os
import torch
import numpy as np
from transformers import BertTokenizer
from utils.model import BertClassifier
from utils.datasets import Dataset


class ClassifyModel:
    def __init__(self, path, classes=2, multi_label=False):
        self.multi_label = multi_label
        self.model = BertClassifier(
            classes=classes, name="bert-base-cased", dropout=0.5
        )
        self.model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        self.model.eval()

    def inference(self, text):
        label = 0 if isinstance(text, str) else [0] * len(text)
        test = Dataset(self.tokenizer, text, label)

        test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:

            self.model = self.model.cuda()

        total_acc_test = 0
        output = []
        with torch.no_grad():

            for test_input, _ in test_dataloader:

                mask = test_input["attention_mask"].to(device)
                input_id = test_input["input_ids"].squeeze(1).to(device)
                if self.multi_label:
                    output.append(
                        [
                            torch.where(i > 0.5, 1, 0).tolist()
                            for i in self.model(input_id, mask)
                        ]
                    )
                else:
                    output.append(
                        [
                            np.array(i.argmax()).tolist()
                            for i in self.model(input_id, mask)
                        ]
                    )
            return output


if __name__ == "__main__":
    model = ClassifyModel(os.path.abspath("output"), 6, True)
    string = "Explanation Why the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove the template from the talk page since I'm retired now.89.205.38.27"
    out = model.inference(string)
    print("out ::", out)
