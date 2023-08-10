from __future__ import annotations

from typing import Any

from celery.result import AsyncResult
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from src.celery.celery import generate_text_task

load_dotenv()

app = FastAPI()


class Item(BaseModel):
    prompt: str


@app.post("/generate/")
async def generate_text(item: Item) -> Any:
    task = generate_text_task.delay(item.prompt)
    return {"task_id": task.id}


@app.get("/task/{task_id}")
async def get_task(task_id: str) -> Any:
    result = AsyncResult(task_id)
    if result.ready():
        res = result.get()
        return {
            "result": res[0],
            "time": res[1],
            "memory": res[2],
        }
    else:
        return {"status": "Task not completed yet"}
