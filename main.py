from fastapi import FastAPI, Request
from pydantic import BaseModel
import modal
from dataclasses import dataclass
import json, os
import requests
import hashlib 

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to my GPU server!"}


class Item(BaseModel):
    item_name: str
    price: float

# In-memory storage for demonstration
items = {}

@app.get("/items/")
async def create_item(item: Item):
    items[item.item_name] = item.price
    return item

@app.get("/items/{item_name}")
async def read_item(item_name: str):
    return items.get(item_name)


class Prompt(BaseModel):
    prompt: str = "What is your name?"
    model: str = "01-ai/Yi-6B"
    system_message: str = "You are an exert in Artificial Intelligence"
    token: str = "abcd"


@app.post("/llm_prompt/")
async def llm_prompt(data: Prompt):

    # to call modal directly, make sure the MODAL tokens are in the environment
    # on render.com, I did put them in a .env file, so I have to load them manually
    # f = modal.Function.lookup("GPU_server", "llm_prompt")
    # answer = f.remote(prompt=data.prompt, model=data.model)
    
    if hashlib.sha256(data.token.encode()).hexdigest() != "1b055d8da2ab450b22cb8b10ed28a64c47c5a3417c0dfddc5665f45f703b3ab3":
        return {"answer": f"Wrong token"}

    url = "https://jpbianchi--gpu-server-llm-prompt.modal.run/"
    answer = requests.post(url, 
                           json={"prompt":data.prompt, 
                                 "model":data.model,
                                 "system_message":data.system_message},
                           )
    return {"answer": answer.text}


@app.post("/test/")
async def test(request: Request):
    """ Pass only one string parameter"""
    body_str = await request.body()  # body for 1 parameter
    body_str = body_str.decode("utf-8")  # Decode bytes to string

    return body_str
    
    
@app.post("/test2/")
async def llm_prompt2(data: Prompt):
    """ Pass as many parameters as you want, as defined in the Pydantic class"""
    return data


# To deploy simply click:
# [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/render-examples/fastapi)
