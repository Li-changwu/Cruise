import torch

from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from vllm_ascend.utils import enable_custom_op


enable_custom_op()
_lib = torch.library.Library("g4a_matmulv2", "DEF")
_lib.define("mm(Tensor self, Tensor mat2) -> Tensor")


def _mm_npu(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    return torch.mm(self, mat2)


def _mm_meta(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (self.shape[0], mat2.shape[1]), dtype=self.dtype, device="meta"
    )


_lib.impl("mm", _mm_npu, "PrivateUse1")
_lib.impl("mm", _mm_meta, "Meta")


@register_fx_node_ge_converter(torch.ops.g4a_matmulv2.mm.default)
def convert_mm(
    self: Tensor,
    mat2: Tensor,
    *,
    meta_outputs: TensorSpec = None,
):
    return ge.MatMulV2(
        self,
        mat2,
        None,
        None,
        node_name="GateMatMulV2",
    )
