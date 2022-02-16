import torch
from transformers import DistilBertTokenizer

from net import DistilBertModelClass


labels = {0: 'Sincere', 1: 'Insincere'}


class TextClassifier():

    def __init__(self, model_path: str):
        self._tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModelClass()
        self.model.load_state_dict(torch.load(model_path), strict=False)


    def predict(self, text: str):
        self.model.eval()

        inputs = self._tokenizer(
            [text],
            truncation=True, 
            return_tensors="pt",
            max_length=150,
            padding='max_length'
        )

        ids = inputs['input_ids'].squeeze(1)
        mask = inputs['attention_mask'].squeeze(1)

        output = self.model(ids, mask).squeeze()

        return labels[torch.argmax(output).item()]