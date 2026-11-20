"""Commit MemT-1.0 history, then save and restore memory without a transcript."""

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from memt_sglang.client import (
    MemTSession,
    NativeClient,
    load_chat_format,
)


def main():
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        "model_path", help="Local Nano, Flash or Pro HF model directory"
    )
    arg_parser.add_argument("--url", default="http://127.0.0.1:30000")
    args = arg_parser.parse_args()
    model_type = json.loads((Path(args.model_path) / "config.json").read_text())[
        "model_type"
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=False,
        # Flash uses raw assets; load_chat_format() supplies its native encoder.
        tokenizer_type="deepseek_v4" if model_type == "memt_deepseek_v4" else None,
    )
    # chat_format builds inputs; parser extracts reasoning and tool calls.
    chat_format, parser = load_chat_format(
        args.model_path,
        tokenizer=tokenizer,
        chat_template_kwargs={"enable_thinking": False},
    )
    client = NativeClient(args.url, api_key=os.environ.get("MEMT_API_KEY"))
    with MemTSession(client, chat_format, parser) as session:
        result = session.chat(
            "I have a three-month-old kitten named memmy who loves eating "
            "small shrimp and is excellent at remembering things."
        )
        print(result)
        session.commit_history()
        print(
            session.chat("What are my kitten's name, age, favorite food, and special trait?")
        )
        # save_state() also commits the remaining recall turn.
        snapshot = session.save_state()
        print("Snapshot:", snapshot["path"])

    with MemTSession(client, chat_format, parser, state=snapshot["path"]) as restored:
        print("Restored messages:", len(restored.messages))
        print(restored.chat("How old is my kitten?"))


if __name__ == "__main__":
    main()
