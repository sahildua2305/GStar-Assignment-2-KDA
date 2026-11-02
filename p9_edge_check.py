"""Check the required combined P9 segmented-state edge case on CUDA."""

import argparse

import torch

from autograder import make_case, verify_segmented
from problem_9 import chunk_kimi_delta_attention


def main(dtype_name: str) -> None:
    torch.manual_seed(2026)
    dtype = getattr(torch, dtype_name)

    q, k, v, kwargs = make_case(
        "kda",
        (1, 1, 129, 128, 64),
        "cuda",
        dtype,
    )

    if not torch.count_nonzero(kwargs["initial_state"]):
        raise AssertionError("The edge case requires a nonzero initial state")
    if not torch.any(kwargs["gate"] != kwargs["gate"][..., :1]):
        raise AssertionError("The edge case requires unequal channel gates")

    verify_segmented(
        chunk_kimi_delta_attention,
        q,
        k,
        v,
        kwargs,
        split=65,
    )
    print(
        "PASS P9 exact edge case: "
        f"dtype={dtype_name}, T=129, Dk=128, Dv=64, split=65, "
        "nonzero initial state, unequal channel gates"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    main(parser.parse_args().dtype)
