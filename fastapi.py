from fastapi import FastAPI
from pydantic import BaseModel

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
