from fastapi import FastAPI
from pydantic import BaseModel
import modal

app = FastAPI()

# A simple model to structure the data you receive
class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

# In-memory storage for demonstration
items = {1:2}

@app.post("/items/")
async def create_item(item: Item):
    items[item.name] = item
    return item

@app.get("/items/{item_name}")
async def read_item(item_name: str):
    return items.get(item_name)

@app.post("/llm_prompt/")
async def llm_prompt(prompt: str = "What is your name?", modelname: str ="01-ai/Yi-6B"):

    f = modal.Function.lookup("GPU_server", "llm_prompt")
    answer = f.remote(modelname, prompt)
    return {"answer": answer}
