import itertools

import torch
import torch.nn.functional as F

from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from vllm_ascend.utils import enable_custom_op


enable_custom_op()
_lib = torch.library.Library("g4a_linear_t_bias", "DEF")
_lib.define("linear(Tensor self, Tensor weight, Tensor bias) -> Tensor")
_node_ids = itertools.count()


def _linear_npu(
    self: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return F.linear(self, weight, bias)


def _linear_meta(
    self: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return torch.empty(
        (self.shape[0], weight.shape[0]), dtype=self.dtype, device="meta"
    )


_lib.impl("linear", _linear_npu, "PrivateUse1")
_lib.impl("linear", _linear_meta, "Meta")


@register_fx_node_ge_converter(torch.ops.g4a_linear_t_bias.linear.default)
def convert_linear(
    self: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    meta_outputs: TensorSpec = None,
):
    return ge.MatMulV2(
        self,
        weight,
        bias,
        None,
        transpose_x2=True,
        node_name=f"QkvLinearTransposeX2_{next(_node_ids):03d}",
    )
