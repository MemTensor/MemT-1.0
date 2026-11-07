"""Read-only batch planning against SGLang 0.5.16's duck-typed fields."""

from dataclasses import dataclass

from memt_sglang.protocol import MemoryRequest, is_memt_rid
from memt_sglang.runtime.features import commit_chunk_policy

# A tail policy keeps whole ordinary blocks and may fragment only the tail.
# The block is the model's memory write-event size, not a scheduler constant:
# install_partial_prefill() publishes the resolved geometry here so a checkpoint
# with a different chunk_tokens cannot be validated against the wrong block.
_COMMIT_BLOCK_TOKENS = None


def set_commit_block_tokens(tokens):
    """Publish the resolved write-event geometry; later calls must agree."""
    global _COMMIT_BLOCK_TOKENS
    if type(tokens) is not int or tokens < 1:
        raise ValueError("Commit block must be a positive integer")
    if _COMMIT_BLOCK_TOKENS is not None and _COMMIT_BLOCK_TOKENS != tokens:
        raise ValueError("Commit block geometry is immutable once resolved")
    _COMMIT_BLOCK_TOKENS = tokens


def commit_block_tokens():
    if _COMMIT_BLOCK_TOKENS is None:
        raise ValueError("Commit block geometry has not been resolved")
    return _COMMIT_BLOCK_TOKENS


def validate_commit_span(request, prefix, count, policy):
    """Validate logical offsets before any layer mutates private state."""
    if (
        type(prefix) is not int
        or type(count) is not int
        or prefix < 0
        or count <= 0
        or prefix + count > request.length
    ):
        raise ValueError("Commit fragment is outside its logical input")
    if policy == "atomic":
        if prefix != 0 or count != request.length:
            raise ValueError("Commit was split or has a cached prefix")
        return
    if policy == "tail":
        block = commit_block_tokens()
        tail_start = request.length // block * block
        if prefix < tail_start and (prefix % block != 0 or count != block):
            raise ValueError(
                f"Tail policy requires complete {block}-token blocks before the tail"
            )
        return
    if policy != "global":
        raise ValueError(f"Unknown commit chunk policy {policy!r}")


@dataclass(frozen=True)
class Sample:
    request: MemoryRequest
    start: int
    stop: int
    prefix_length: int
    decode: bool


@dataclass(frozen=True)
class StepPlan:
    samples: tuple[Sample, ...]
    token_count: int

    @classmethod
    def from_batch(cls, batch) -> "StepPlan":
        # fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1:
        # forward_batch_info.py:454,476-477,493; DP idle conversion:1273-1324.
        mode = batch.forward_mode
        original_mode = getattr(batch, "_original_forward_mode", None) or mode
        rids = batch.rids or []
        if mode.is_idle() or original_mode.is_idle() or not any(map(is_memt_rid, rids)):
            return cls((), 0)
        decode = mode.is_decode()
        if not decode and not mode.is_extend():
            raise ValueError("MemT requires eager extend or decode batches")
        if decode:
            lengths, prefixes = [1] * len(rids), [0] * len(rids)
        else:
            lengths = batch.extend_seq_lens_cpu
            prefixes = batch.extend_prefix_lens_cpu
        if (
            lengths is None
            or prefixes is None
            or len(lengths) < len(rids)
            or len(prefixes) < len(rids)
        ):
            raise ValueError("Missing CPU request lengths for MemT planning")
        samples, touched, offset = [], set(), 0
        for rid, length, prefix in zip(rids, lengths, prefixes, strict=False):
            if (
                type(length) is not int
                or length < 1
                or type(prefix) is not int
                or prefix < 0
            ):
                raise ValueError("Invalid request token span")
            if is_memt_rid(rid):
                request = MemoryRequest.from_rid(rid)
                if request.session in touched:
                    raise ValueError("A step must contain independent MemT sessions")
                touched.add(request.session)
                if request.operation == "commit" and decode:
                    raise ValueError("commit cannot decode")
                if request.operation == "commit":
                    # The policy decides whether a split is legal; `atomic` keeps
                    # exactly the old guarantee, so its behaviour cannot change.
                    validate_commit_span(
                        request, prefix, length, commit_chunk_policy()
                    )
                samples.append(Sample(request, offset, offset + length, prefix, decode))
            offset += length
        non_padding = getattr(batch, "num_token_non_padded_cpu", None)
        if non_padding is not None and offset > non_padding:
            raise ValueError("Request spans include padding tokens")
        return cls(tuple(samples), offset)
