# %%
import sys
sys.path.append('../')
from pathlib import Path
from datetime import datetime

import pandas as pd

import torch
import torch.nn as nn
from torch import cuda
from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertTokenizer, DistilBertModel

from config import settings


print("Starting")
# %% [markdown]
# ### Prepare

# %%
device = 'cuda:1' if cuda.is_available() else 'cpu'

MAX_LEN = 150
BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 1e-05
DISTIL_BERT_CHECKPOINT = 'distilbert-base-uncased'
RUN_NAME = 'ROS'
TEST_PATH = '../data/processed/quick_test.csv'
TRAIN_PATH = '../data/ros/train.csv'
MODEL_SAVE = '../models/'

tokenizer = DistilBertTokenizer.from_pretrained(DISTIL_BERT_CHECKPOINT)

# %% [markdown]
# ### Dataset and dataloader

# %%
class QuoraDataset(Dataset):

    def __init__(self, file_path, tokenizer, max_len):
        self._dataset = pd.read_csv(file_path, low_memory=False)
        self._tokenizer = tokenizer 
        self._max_len = max_len

    def __getitem__(self, index):
        text = self._dataset.iloc[index]["question_text"]
        inputs = self._tokenizer(
            [text],
            truncation=True, 
            return_tensors="pt",
            max_length=self._max_len,
            padding='max_length'
        )

        return {
            "ids": inputs["input_ids"],
            "mask": inputs["attention_mask"],
            "target": torch.tensor(self._dataset.iloc[index]["target"], dtype=torch.long)
        }

    def __len__(self):
        return len(self._dataset)

# %%
print("Load datasets...")
train_dataset = QuoraDataset(TRAIN_PATH, tokenizer, MAX_LEN)
test_dataset = QuoraDataset(TEST_PATH, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# %% [markdown]
# ### DistilBert Model

# %%
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

print("Load model...")
model = DistilBertModelClass()
model.to(device);

# %% [markdown]
# ### Training

# %%
# Creating the loss function and optimizer
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE) 

# %%
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from collections import defaultdict

def accuracy(model, loader):
    model.eval()

    with torch.no_grad():
        y_pred = []
        y_true = []

        classname = {0: 'Sincere', 1: 'Insincere'}
        correct_pred = defaultdict(lambda: 0)
        total_pred = defaultdict(lambda: 0)

        for inputs in loader:
            ids = inputs['ids'].squeeze(1).to(device)
            mask = inputs['mask'].squeeze(1).to(device)
            targets = inputs['target'].to(device)

            output = model(ids, mask).squeeze()

            _, predictions = torch.max(output, 1)
            
            y_pred += list(predictions.to('cpu'))
            y_true += list(targets.to('cpu'))

            for target, prediction in zip(targets, predictions):
                if target.item() == prediction.item():
                    correct_pred[classname[target.item()]] += 1
                total_pred[classname[prediction.item()]] += 1

        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred)
        }

        for classname, correct_count in correct_pred.items():
            results['precision_' + classname] = 100 * float(correct_count) / total_pred[classname]

        return results

#results = accuracy(model, test_loader)
#results

# %%
def train(epoch=1):
    model.train()

    for idx, inputs in enumerate(train_loader):
        
        ids = inputs['ids'].squeeze(1).to(device)
        mask = inputs['mask'].squeeze(1).to(device)
        target = inputs['target'].to(device)

        output = model(ids, mask).squeeze()

        optimizer.zero_grad()

        l = loss(output, target)
        l.backward()

        optimizer.step()

        # Log Loss
        run["train/loss"].log(l.item(), step=idx)

        if idx % 10 == 0:
            print(f'Epoch: {epoch}, {idx}/{len(train_loader)}, Loss:  {l.item()}')

        if idx % 20 == 0:
            results = accuracy(model, test_loader) 
            run["train/accuracy"].log(results['accuracy'], step=idx)
            run["train/f1"].log(results['f1'], step=idx)
            run["train/roc_auc"].log(results['roc_auc'], step=idx)
            run["train/precision_Sincere"].log(results['precision_Sincere'], step=idx)
            run["train/precision_Insincere"].log(results['precision_Insincere'], step=idx)
            print(results)
            print("Saving model...")
            torch.save(model.state_dict(), Path(MODEL_SAVE) / f'ftbert_{idx}_{datetime.now()}' )

# %% [markdown]
# ### Training

# %%
# track training and results...
print("Starting neptune tracking...")
import neptune.new as neptune

run = neptune.init(
    project=settings.project,
    api_token=settings.api_token,
    name='RandomOversampling'
)  

print("Training...")
train(epoch=EPOCHS)

run.stop()
print("Finished!")