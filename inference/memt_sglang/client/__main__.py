"""Minimal interactive client: python -m memt_sglang.client --help."""

import argparse
import json
import os
import sys

from memt_sglang.client import (
    MemTClientError,
    MemTSession,
    NativeClient,
    load_chat_format,
)
from memt_sglang.client.native import DEFAULT_MAX_NEW_TOKENS
from memt_sglang.models.common.hf_config import MODEL_TYPES


def _print_error(error, *, uncertain=False):
    print(
        json.dumps(
            {
                "error": str(error),
                "uncertain": uncertain or bool(getattr(error, "uncertain", False)),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )


def main():
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument("--url", default="http://127.0.0.1:30000")
    arg_parser.add_argument(
        "--tokenizer", required=True, help="Local tokenizer directory"
    )
    arg_parser.add_argument(
        "--model",
        required=True,
        choices=tuple(MODEL_TYPES.values()),
        help="MemT-1.0-Nano, MemT-1.0-Flash or MemT-1.0-Pro",
    )
    deepseek = arg_parser.add_mutually_exclusive_group()
    deepseek.add_argument(
        "--chat-template-kwargs", help="DeepSeek template JSON, e.g. {}"
    )
    deepseek.add_argument(
        "--deepseek-encoder", help="Trusted local checkpoint encoding_dsv4.py"
    )
    arg_parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    arg_parser.add_argument(
        "--enable-memory",
        action="store_true",
        help="Commit unsaved past and current turns after the reply completes",
    )
    args = arg_parser.parse_args()
    if args.max_new_tokens < 1:
        arg_parser.error("max-new-tokens must be positive")
    if args.model == "memt-flash" and not (
        args.deepseek_encoder or args.chat_template_kwargs is not None
    ):
        arg_parser.error(
            "--model memt-flash requires --deepseek-encoder or --chat-template-kwargs"
        )
    if args.model in {"memt-nano", "memt-pro"} and (
        args.deepseek_encoder or args.chat_template_kwargs is not None
    ):
        arg_parser.error("DeepSeek encoder options require --model memt-flash")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
        # Flash uses raw tokenizer assets; chat encoding stays in chat_format.
        tokenizer_type="deepseek_v4" if args.model == "memt-flash" else None,
    )
    # Input formatting and output parsing are independent client components.
    chat_format, parser = load_chat_format(
        args.tokenizer,
        tokenizer=tokenizer,
        chat_template_kwargs=(
            json.loads(args.chat_template_kwargs)
            if args.chat_template_kwargs is not None
            else None
        ),
        chat_encoding_spec="auto" if args.deepseek_encoder else None,
        deepseek_encoder=args.deepseek_encoder,
    )
    client = NativeClient(
        args.url,
        api_key=os.environ.get("MEMT_API_KEY"),
    )
    print(
        "Commands: /read on|off, /commit, /save [name.safetensors], /load ID, "
        "/status, /reset, /quit. Other lines are chat messages.",
        file=sys.stderr,
    )
    session = None
    try:
        with MemTSession(client, chat_format, parser) as session:
            print(json.dumps(session.status(), ensure_ascii=False))
            for line in sys.stdin:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                command, _, value = line.partition(" ")
                if command == "/quit" and not value:
                    break
                try:
                    if command == "/read" and value in {"on", "off"}:
                        result = session.set_memory_read(value == "on")
                    elif command == "/commit" and not value:
                        session.commit_history()
                        result = session.status()
                    elif command == "/save":
                        result = session.save_state(value or None)
                    elif command == "/load" and value:
                        result = session.load_state(value)
                    elif command == "/status" and not value:
                        result = session.status()
                    elif command == "/reset" and not value:
                        session.reset()
                        result = session.status()
                    elif line.startswith("/"):
                        raise ValueError("Unknown command or invalid arguments")
                    else:
                        result = session.chat(
                            line,
                            enable_memory=args.enable_memory,
                            return_dict=True,
                            max_new_tokens=args.max_new_tokens,
                        )
                except (MemTClientError, ValueError) as error:
                    if session.uncertain or getattr(error, "uncertain", False):
                        raise
                    # A rejected command leaves this session usable. Let the user
                    # correct it without closing memory or replaying the operation.
                    _print_error(error)
                    continue
                print(json.dumps(result, ensure_ascii=False), flush=True)
    except (MemTClientError, OSError, ValueError) as error:
        _print_error(error, uncertain=session is not None and session.uncertain)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
