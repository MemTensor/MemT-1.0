"""Read packed samples from fixed memory snapshots with batched matrix products."""

import torch


def snapshot_read(query, states, lengths, scale, *, bf16_accumulator_output=False):
    # Padded *reader workspace* only. Native backbone tokens remain packed.
    _, _, heads, dim = query.shape
    width, count = max(lengths), len(lengths)
    destination = []
    for sample, length in enumerate(lengths):
        destination.extend(range(sample * width, sample * width + length))
    index = torch.tensor(destination, device=query.device)
    padded = query.new_zeros(count * width, heads, dim)
    padded.index_copy_(0, index, query[0])
    q = (
        padded.view(count, width, heads, dim)
        .transpose(1, 2)
        .reshape(count * heads, width, dim)
    )
    state_matrix = states.reshape(count * heads, dim, states.shape[-1]).to(q.dtype)
    if bf16_accumulator_output:
        result = torch.baddbmm(q.new_empty(()), q, state_matrix, beta=0, alpha=scale)
    else:
        result = torch.bmm(q, state_matrix) * scale
    result = (
        result.view(count, heads, width, -1)
        .transpose(1, 2)
        .reshape(count * width, heads, -1)
    )
    return result.index_select(0, index).unsqueeze(0)


def packed_snapshot_read(query, states, lengths, scale):
    """Skip identity padding/index operations for one-token decode samples."""
    if all(length == 1 for length in lengths):
        count, heads, dim = len(lengths), query.shape[2], query.shape[3]
        q = query.reshape(count * heads, 1, dim)
        state_matrix = states.reshape(count * heads, dim, states.shape[-1]).to(q.dtype)
        result = torch.baddbmm(q.new_empty(()), q, state_matrix, beta=0, alpha=scale)
        return result.reshape(1, count, heads, states.shape[-1])
    return snapshot_read(query, states, lengths, scale, bf16_accumulator_output=True)
