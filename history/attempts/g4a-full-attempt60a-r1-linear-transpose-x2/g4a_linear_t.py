import torch

from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from vllm_ascend.utils import enable_custom_op


enable_custom_op()
_lib = torch.library.Library("g4a_linear_t", "DEF")
_lib.define("mm_t(Tensor self, Tensor weight) -> Tensor")


def _mm_t_npu(self: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mm(self, weight.transpose(0, 1))


def _mm_t_meta(self: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (self.shape[0], weight.shape[0]), dtype=self.dtype, device="meta"
    )


_lib.impl("mm_t", _mm_t_npu, "PrivateUse1")
_lib.impl("mm_t", _mm_t_meta, "Meta")


@register_fx_node_ge_converter(torch.ops.g4a_linear_t.mm_t.default)
def convert_mm_t(
    self: Tensor,
    weight: Tensor,
    *,
    meta_outputs: TensorSpec = None,
):
    return ge.MatMulV2(
        self,
        weight,
        None,
        None,
        transpose_x2=True,
        node_name="LinearTransposeX2",
    )
