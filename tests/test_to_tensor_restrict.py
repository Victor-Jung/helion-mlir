"""Regression: tile-load `to_tensor` ops must carry `restrict`.

Without it, One-Shot Bufferization fails with
    'bufferization.to_tensor' op to_tensor ops without `restrict` are not
    supported by One-Shot Analysis
which surfaces at the materialization stage of the Loom pipeline, not here, so
this guards the emission site directly.
"""

from __future__ import annotations

from helion_mlir.mlir_utils import format_to_tensor


def test_format_to_tensor_emits_restrict() -> None:
    out = format_to_tensor(
        "%tile",
        "%subview",
        "memref<?x1xf16, strided<[1, 1], offset: ?>>",
        "tensor<?x1xf16>",
    )
    assert out == (
        "%tile = bufferization.to_tensor %subview restrict : "
        "memref<?x1xf16, strided<[1, 1], offset: ?>> to tensor<?x1xf16>"
    )


def test_format_to_tensor_restrict_precedes_colon() -> None:
    # `restrict` is an operand-position attribute: it must come before the ':'
    # or the op does not parse.
    out = format_to_tensor("%t", "%s", "memref<4x4xf16>", "tensor<4x4xf16>")
    assert " restrict :" in out
    assert out.index("restrict") < out.index(":")
