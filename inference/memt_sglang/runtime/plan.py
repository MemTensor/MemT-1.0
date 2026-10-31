"""Read-only batch planning against SGLang 0.5.16's duck-typed fields."""

from dataclasses import dataclass

from memt_sglang.protocol import MemoryRequest, is_memt_rid
from memt_sglang.runtime.features import commit_chunk_policy


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
    elif policy == "tail":
        tail_start = request.length // 2048 * 2048
        if prefix < tail_start and (prefix % 2048 != 0 or count != 2048):
            raise ValueError("Tail policy requires complete ordinary 2048-token blocks")
    elif policy != "global":
        raise ValueError("Unknown commit chunk policy")


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
                    validate_commit_span(request, prefix, length, commit_chunk_policy())
                samples.append(Sample(request, offset, offset + length, prefix, decode))
            offset += length
        non_padding = getattr(batch, "num_token_non_padded_cpu", None)
        if non_padding is not None and offset > non_padding:
            raise ValueError("Request spans include padding tokens")
        return cls(tuple(samples), offset)
