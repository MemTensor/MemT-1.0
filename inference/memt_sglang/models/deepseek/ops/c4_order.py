"""Permute native C4 selections without changing their membership."""


def canonicalize_(mapped, raw):
    if mapped.shape[0] == 0:
        return
    from ._triton.c4_order import canonicalize_kernel

    canonicalize_kernel[(mapped.shape[0],)](mapped, raw, num_warps=4, num_stages=1)
