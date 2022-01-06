from torch import nn
from utils.model_classes import MODEL_CLASSES


class TransformersClassifier(nn.Module):
    def __init__(
        self, name="bert", model_name="bert-base-cased", num_labels=2, dropout=0.5
    ):
        config, model, _ = MODEL_CLASSES[name]
        super(TransformersClassifier, self).__init__()

        self.model = model.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_id, mask):

        out = self.model(input_ids=input_id, attention_mask=mask, return_dict=False)
        # model outputs are always tuple
        return out[0]
