from typing import List, Tuple
from pydantic import BaseModel

from fastapi import FastAPI
from classifier import TextClassifier

print("Starting...")

MODEL_PATH = './model/model_weights'

print("Loading Models...")
text_classifier = TextClassifier(MODEL_PATH)

print("Starting RestAPI...")
app = FastAPI()


class RequestText(BaseModel):
    texts: List[str]

class RequestAnswer(BaseModel):
    labels: List[Tuple[str, str]]


@app.post("/run", response_form=RequestText)
def run(request: RequestText):
    ans = []
    for text in request.texts:
        ans.append(
            (text, text_classifier.predict(text))
        )
    return ans