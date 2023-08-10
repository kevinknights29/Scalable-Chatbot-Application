from __future__ import annotations

from celery import Celery
from celery import signals

from src.models.loader import ModelLoader
from src.utils.utils import generate_output

model_loader = None
model_path = "meta-llama/Llama-2-7b-chat-hf"


def make_celery(app_name=__name__):
    backend = broker = "redis://llama2_redis_1:6379/0"
    return Celery(app_name, backend=backend, broker=broker)


celery_app = make_celery()


@signals.worker_process_init.connect
def setup_model(signal, sender, **kwargs):
    global model_loader
    model_loader = ModelLoader(model_path)


@celery_app.task
def generate_text_task(prompt):
    time, memory, outputs = generate_output(
        prompt,
        model_loader.model,
        model_loader.tokenizer,
    )
    return model_loader.tokenizer.decode(outputs[0]), time, memory


if __name__ == "__main__":
    celery_app.start()
