"""Experimental native query chunking, preserving one-forward atomic commits."""

from functools import wraps

from memt_sglang.protocol import MemoryRequest, is_memt_rid


def install_atomic_commit_guard():
    from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder

    original = PrefillAdder.add_one_req
    if getattr(original, "_memt_atomic_commits", False):
        return

    @wraps(original)
    def add_one_req(self, req, *args, **kwargs):
        commit = None
        if is_memt_rid(req.rid):
            request = MemoryRequest.from_rid(req.rid)
            if request.operation == "commit":
                commit = request
                if (
                    len(req.prefix_indices)
                    or len(req.full_untruncated_fill_ids) != commit.length
                ):
                    raise ValueError(
                        "Atomic commit must begin at zero and carry its complete input"
                    )
                required = self.ceil_paged_tokens(commit.length)
                if (
                    self.rem_chunk_tokens is not None
                    and required > self.rem_chunk_tokens
                ):
                    if self.can_run_list or self.new_chunked_req is not None:
                        return AddReqResult.OTHER
                    # Release commits may exceed the query chunk budget. Admit
                    # one whole commit in an otherwise empty prefill, retaining
                    # native total-token and SWA capacity checks. Charge the
                    # actual input against the original budget afterwards.
                    remaining = self.rem_chunk_tokens
                    self.rem_chunk_tokens = None
                    try:
                        result = original(self, req, *args, **kwargs)
                    finally:
                        self.rem_chunk_tokens = remaining - (
                            required if any(r is req for r in self.can_run_list) else 0
                        )
                    if any(r is req for r in self.can_run_list):
                        if (
                            self.new_chunked_req is req
                            or req.extend_range.length != commit.length
                        ):
                            raise RuntimeError(
                                "Scheduler attempted to split an atomic commit"
                            )
                    return result
        result = original(self, req, *args, **kwargs)
        if commit is not None and any(r is req for r in self.can_run_list):
            if self.new_chunked_req is req or req.extend_range.length != commit.length:
                raise RuntimeError(
                    "Scheduler attempted to split an atomic commit; no forward permitted"
                )
        return result

    add_one_req._memt_atomic_commits = True
    PrefillAdder.add_one_req = add_one_req
