import torch.nn as nn
from transformers import  DistilBertModel


class DistilBertModelClass(nn.Module):

    def __init__(self):
        super(DistilBertModelClass, self).__init__()
        self.distil_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.linear1 = nn.Linear(768, 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, ids, mask):
        bert_out = self.distil_bert(ids, mask)
        x = bert_out.last_hidden_state[:, -1, :] # get bert last hidden state
        x = self.linear1(x)
        x = self.sigmoid(x)
        return x