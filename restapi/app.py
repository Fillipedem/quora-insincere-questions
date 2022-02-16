from typing import List, Tuple
from pydantic import BaseModel

from fastapi import FastAPI
from classifier import TextClassifier

print("Starting...")

MODEL_PATH = './model/model_dict'

print("Loading Models...")
text_classifier = TextClassifier(MODEL_PATH)

print("Starting RestAPI...")

with open("README.md", "r") as f:
    description = f.read()

app = FastAPI(
    description=description,
    version="0.0.1"
)


class RequestText(BaseModel):
    texts: List[str]

class RequestAnswer(BaseModel):
    labels: List[Tuple[str, str]]


@app.post("/run")
def run(body: RequestText = None):
    ans = []
    for text in body.texts:
        ans.append(
            (text, text_classifier.predict(text))
        )
    return ans