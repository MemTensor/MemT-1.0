"""Two ordinary MemT-1.0 chat turns, without reading or writing native memory."""

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
        session.set_memory_read(False)
        print(session.chat("Type 'I love MemT' exactly.", enable_memory=False))
        print(
            session.chat(
                "Translate your previous answer into Chinese.", enable_memory=False
            )
        )


if __name__ == "__main__":
    main()
