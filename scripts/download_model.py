from __future__ import annotations

import argparse
import os

from dotenv import find_dotenv
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


_ = load_dotenv(find_dotenv())
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


def download_model(model_id, model_dir="./models"):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        trust_remote_code=True,
    )
    model.save_pretrained(model_dir)


def download_tokenizer(model_id, model_dir="./models"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model ID to download",
    )
    args = parser.parse_args()

    login(token=HUGGINGFACE_TOKEN)
    download_model(args.model_id)
    download_tokenizer(args.model_id)
