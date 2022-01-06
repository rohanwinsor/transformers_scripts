from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (
        RobertaConfig,
        RobertaForSequenceClassification,
        RobertaTokenizer,
    ),
}
