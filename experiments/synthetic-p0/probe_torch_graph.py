#!/usr/bin/env python3
import json

import torch
import torch_npu


def main() -> None:
    graph_names = sorted(name for name in dir(torch.npu) if "graph" in name.lower())
    result = {
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "has_npu": hasattr(torch, "npu"),
        "has_npugraph": hasattr(torch.npu, "NPUGraph"),
        "has_graph_context": hasattr(torch.npu, "graph"),
        "graph_names": graph_names,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
