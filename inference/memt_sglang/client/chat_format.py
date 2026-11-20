"""Chat templates/native chat encoding, separate from response parsing."""

import json
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from memt_sglang.models.common.snapshot_identity import snapshot_identity


def token_ids(value):
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(type(token_id) is not int or token_id < 0 for token_id in value)
    ):
        raise ValueError("Expected nonempty token IDs")
    return tuple(value)


def text(value):
    if not isinstance(value, str):
        raise ValueError("Expected text")
    return value


def load_chat_format(
    path,
    *,
    tokenizer,
    chat_template_kwargs=None,
    chat_encoding_spec="auto",
    deepseek_encoder=None,
):
    """Return (chat_format, parser) for a local Nano, Flash or Pro checkpoint.

    The tokenizer is supplied separately. Following SGLang's separation,
    chat_format applies the HF chat template or native chat_encoding; parser
    handles reasoning and function calls. These lightweight client objects do
    not import SGLang's server managers.

    Flash defaults to its trusted local encoding_dsv4.py. Set
    chat_encoding_spec=None to explicitly use its HF chat template instead.
    chat_template_kwargs supplies defaults such as enable_thinking; chat()
    can override thinking for an individual turn.
    """
    from memt_sglang.client.parsers import DeepSeekResponseParser, QwenResponseParser
    from memt_sglang.models.common.hf_config import memt_model

    path = Path(path).resolve(strict=True)
    config = json.loads((path / "config.json").read_text())
    model = memt_model(config)
    if model not in {"memt-nano", "memt-flash", "memt-pro"}:
        raise ValueError("Expected a MemT-1.0-Nano, Flash or Pro model directory")
    if chat_encoding_spec not in {"auto", None, "dsv4"}:
        raise ValueError(f"Unsupported chat_encoding_spec: {chat_encoding_spec!r}")
    if model in {"memt-nano", "memt-pro"}:
        if deepseek_encoder is not None or chat_encoding_spec == "dsv4":
            raise ValueError("DeepSeek encoding requires MemT-1.0-Flash")
        format_type = NemotronChatFormat if model == "memt-pro" else QwenChatFormat
        chat_format = format_type(
            tokenizer, model_config=config, chat_template_kwargs=chat_template_kwargs
        )
        return chat_format, QwenResponseParser(
            tokenizer, eos_token_ids=chat_format.eos_token_ids
        )
    if chat_encoding_spec is None:
        if deepseek_encoder is not None:
            raise ValueError("Choose a DeepSeek encoder or the HF chat template")
        chat_format = DeepSeekChatFormat(
            tokenizer,
            chat_template_kwargs=chat_template_kwargs or {},
            model_config=config,
        )
        return chat_format, DeepSeekResponseParser(
            tokenizer, eos_token_ids=chat_format.eos_token_ids
        )
    encoder = _load_deepseek_encoder(deepseek_encoder or path / "encoding_dsv4.py")
    chat_format = DeepSeekChatFormat(
        tokenizer, encode_messages=encoder.encode_messages, model_config=config
    )
    chat_format.chat_template_kwargs.update(chat_template_kwargs or {})
    return chat_format, DeepSeekResponseParser(
        tokenizer,
        eos_token_ids=chat_format.eos_token_ids,
        parse_message=encoder.parse_message_from_completion_text,
    )


class ChatFormat:
    """Input templates/native encoding and exact memory replay boundaries.

    Tokenizer ownership remains explicit; output parsing lives in ResponseParser.
    chat_template_kwargs follows SGLang/HF naming.
    """

    chat_encoding_spec = None

    def __init__(self, tokenizer, *, chat_template_kwargs, model_config=None):
        self.tokenizer = tokenizer
        self.chat_template_kwargs = deepcopy(dict(chat_template_kwargs))
        if {
            "tokenize",
            "add_generation_prompt",
            "tools",
        } & self.chat_template_kwargs.keys():
            raise ValueError("Reserved template argument")
        if model_config is None:
            path = Path(getattr(tokenizer, "name_or_path", "")) / "config.json"
            if path.is_file():
                model_config = json.loads(path.read_text())
        self.model_config = deepcopy(model_config)
        self.eos_token_ids = {tokenizer.eos_token_id}
        generation_path = (
            Path(getattr(tokenizer, "name_or_path", "")) / "generation_config.json"
        )
        if generation_path.is_file():
            eos = json.loads(generation_path.read_text()).get("eos_token_id")
            if eos is not None:
                self.eos_token_ids = set(eos if isinstance(eos, list) else [eos])

    @property
    def identity(self):
        if self.model_config is None:
            raise ValueError("Memory snapshots require the model's HF config.json")
        return snapshot_identity(self.model_config, self.tokenizer)

    @property
    def chat_template(self):
        return self.tokenizer.chat_template

    def encode(self, value):
        return token_ids(self.tokenizer.encode(text(value), add_special_tokens=False))

    def render(
        self, messages, *, generation=False, thinking=None, system=None, tools=None
    ):
        messages = deepcopy(list(messages))
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        options = dict(self.chat_template_kwargs, preserve_thinking=True)
        if thinking is not None:
            options["enable_thinking"] = thinking
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=generation,
            tools=tools,
            **options,
        )

    def prefix(self, messages, *, thinking=None):
        return self.encode(self.render(messages, generation=True, thinking=thinking))

    def prompt(self, messages, *, tools=None, thinking=None):
        return self.encode(
            self.render(
                messages,
                generation=True,
                thinking=thinking,
                tools=tools,
            )
        )

    def query(self, turns, *, system=None, tools=None, has_memory=False):
        messages = [message for turn in turns for message in turn.messages]
        return self.encode(
            self.render(
                messages,
                generation=True,
                thinking=turns[-1].thinking,
                system=system,
                tools=tools,
            )
        )

    def replay(self, ids, *, has_memory):
        return token_ids(ids)

    def document(self, value, *, has_memory=False):
        return self.encode(self.render([{"role": "user", "content": text(value)}]))

    def turn(self, question, answer):
        return self.encode(
            self.render(
                [
                    {"role": "user", "content": text(question)},
                    {"role": "assistant", "content": text(answer)},
                ]
            )
        )

    def history(self, messages):
        """Encode an explicitly supplied transcript; Session uses raw turn IDs."""
        groups = []
        for message in messages:
            if message.get("role") == "user":
                groups.append([])
            if not groups or message.get("role") not in {"user", "assistant", "tool"}:
                raise ValueError(
                    "History requires user turns with assistant/tool messages"
                )
            groups[-1].append(message)
        if not groups or any(group[-1]["role"] != "assistant" for group in groups):
            raise ValueError("History requires completed assistant turns")
        return tuple(self.encode(self.render(group)) for group in groups)

    def tool_suffix(self, messages, *, context, thinking):
        dummy = [{"role": "user", "content": ""}]
        prefix = self.render(dummy)
        suffix = self.render(dummy + messages, generation=True, thinking=thinking)
        if not suffix.startswith(prefix):
            raise ValueError("Tool template changed the retained prefix")
        return self.encode("\n" + suffix[len(prefix) :])


class QwenChatFormat(ChatFormat):
    encoding = "qwen-text-raw-v1"

    def __init__(self, tokenizer, *, chat_template_kwargs=None, model_config=None):
        super().__init__(
            tokenizer,
            chat_template_kwargs=chat_template_kwargs or {},
            model_config=model_config,
        )


class NemotronChatFormat(ChatFormat):
    """Native Nemotron Jinja, with HF Pro's context/replay boundaries."""

    encoding = "nemotron-text-raw-v1"

    def __init__(self, tokenizer, *, chat_template_kwargs=None, model_config=None):
        options = dict(chat_template_kwargs or {})
        if options.get("truncate_history_thinking", False) is not False:
            raise ValueError("Pro sessions require truncate_history_thinking=False")
        options["truncate_history_thinking"] = False
        super().__init__(
            tokenizer, chat_template_kwargs=options, model_config=model_config
        )

    def render(
        self,
        messages,
        *,
        generation=False,
        thinking=None,
        system=None,
        tools=None,
        context=False,
    ):
        messages = deepcopy(list(messages))
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        options = dict(self.chat_template_kwargs, truncate_history_thinking=False)
        if thinking is not None:
            options["enable_thinking"] = thinking
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=generation,
            tools=tools,
            **options,
        )
        if not context:
            empty_system = "<|im_start|>system\n<|im_end|>\n"
            if not rendered.startswith(empty_system):
                raise ValueError(
                    "Nemotron replay expects the native empty system prefix"
                )
            rendered = rendered[len(empty_system) :]
        return rendered

    def prompt(self, messages, *, tools=None, thinking=None):
        return self.encode(
            self.render(
                messages, generation=True, thinking=thinking, tools=tools, context=True
            )
        )

    def query(self, turns, *, system=None, tools=None, has_memory=False):
        messages = [message for turn in turns for message in turn.messages]
        return self.encode(
            self.render(
                messages,
                generation=True,
                thinking=turns[-1].thinking,
                system=system,
                tools=tools,
                context=True,
            )
        )


class DeepSeekChatFormat(ChatFormat):
    encoding = "deepseek-v4-native-raw-v1"

    def __init__(
        self,
        tokenizer,
        *,
        encode_messages=None,
        chat_template_kwargs=None,
        model_config=None,
    ):
        if (encode_messages is None) == (chat_template_kwargs is None):
            raise ValueError("Choose a DeepSeek encoder or explicit template options")
        if encode_messages is not None and not callable(encode_messages):
            raise ValueError("DeepSeek encode_messages must be callable")
        super().__init__(
            tokenizer,
            chat_template_kwargs=chat_template_kwargs or {},
            model_config=model_config,
        )
        self.chat_encoding_spec = "dsv4" if encode_messages is not None else None
        self.encode_messages = encode_messages

    def render(
        self, messages, *, generation=False, thinking=None, system=None, tools=None
    ):
        if self.encode_messages is None:
            return super().render(
                messages,
                generation=generation,
                thinking=thinking,
                system=system,
                tools=tools,
            )
        messages = deepcopy(list(messages))
        if system is not None or tools:
            messages.insert(
                0, {"role": "system", "content": system or "", "tools": tools or []}
            )
        for message in messages:
            content = message.get("content")
            if message.get("role") == "assistant" and isinstance(content, str):
                if content.startswith("<think>") and "</think>" in content:
                    reasoning, answer = content[len("<think>") :].split("</think>", 1)
                    message.update(content=answer, reasoning_content=reasoning)
        rendered = self.encode_messages(
            messages,
            "thinking" if thinking else "chat",
            drop_thinking=False,
            reasoning_effort=self.chat_template_kwargs.get("reasoning_effort"),
        )
        if (
            not generation
            and messages
            and messages[-1]["role"] in {"user", "developer", "tool"}
        ):
            for suffix in ("<｜Assistant｜><think>", "<｜Assistant｜></think>"):
                if rendered.endswith(suffix):
                    rendered = rendered[: -len(suffix)]
                    break
        return rendered

    def query(self, turns, *, system=None, tools=None, has_memory=False):
        if self.encode_messages is None:
            return super().query(
                turns, system=system, tools=tools, has_memory=has_memory
            )
        prefix = []
        if system is not None or tools:
            prefix = [{"role": "system", "content": system or "", "tools": tools or []}]
        ids = list(self.encode(self.encode_messages(prefix, "chat")))
        bos = self.tokenizer.bos_token_id
        if has_memory and ids and ids[0] == bos:
            ids = ids[1:]
        for turn in turns:
            tokens = turn.replay_ids
            ids.extend(tokens[1:] if tokens and tokens[0] == bos else tokens)
        return token_ids(ids)

    def replay(self, ids, *, has_memory):
        ids = token_ids(ids)
        if has_memory and ids[0] == self.tokenizer.bos_token_id:
            ids = ids[1:]
        return token_ids(ids)

    def document(self, value, *, has_memory=False):
        return self.replay(super().document(value), has_memory=has_memory)

    def tool_suffix(self, messages, *, context, thinking):
        if self.encode_messages is None:
            return super().tool_suffix(messages, context=context, thinking=thinking)
        return self.encode(
            self.encode_messages(
                deepcopy(messages),
                "thinking" if thinking else "chat",
                context=deepcopy(context),
                drop_thinking=False,
                add_default_bos_token=False,
            )
        )


def _load_deepseek_encoder(path):
    """Execute only the explicitly selected, trusted local encoder module."""
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.suffix != ".py":
        raise ValueError("Expected a local DeepSeek Python encoder file")
    spec = spec_from_file_location("memt_deepseek_checkpoint_encoder", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the DeepSeek checkpoint encoder")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunks(ids, *, chunk_tokens):
    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError("chunk_tokens must be a positive integer")
    exact = token_ids(ids)
    return tuple(
        exact[start : start + chunk_tokens]
        for start in range(0, len(exact), chunk_tokens)
    )
