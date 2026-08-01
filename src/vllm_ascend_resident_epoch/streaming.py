"""Cruise-specific incremental streaming support for resident epochs."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any


logger = logging.getLogger(__name__)


def install_strict_delta_collector() -> None:
    """Keep each Cruise DELTA output as an independently consumable event.

    vLLM's standard collector intentionally merges DELTA outputs while a
    producer is ahead of its HTTP consumer. A resident epoch can commit
    multiple decode tokens at once, so that behavior would hide the token
    boundaries required by the incremental streaming contract. The patch is
    installed only by the Cruise general plugin and leaves all non-DELTA
    collectors on vLLM's implementation.
    """
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.output_processor import RequestOutputCollector

    if getattr(RequestOutputCollector, "_cruise_strict_delta_installed", False):
        return

    original_init = RequestOutputCollector.__init__
    original_put = RequestOutputCollector.put
    original_get = RequestOutputCollector.get
    original_get_nowait = RequestOutputCollector.get_nowait

    def strict_init(self: Any, output_kind: Any, request_id: str) -> None:
        original_init(self, output_kind, request_id)
        if output_kind == RequestOutputKind.DELTA:
            self._cruise_strict_delta_outputs = deque()
            self._cruise_strict_delta_error = False

    def strict_put(self: Any, output: Any) -> None:
        outputs = getattr(self, "_cruise_strict_delta_outputs", None)
        if outputs is None:
            original_put(self, output)
            return
        if self._cruise_strict_delta_error:
            return
        if isinstance(output, Exception):
            outputs.clear()
            self._cruise_strict_delta_error = True
        outputs.append(output)
        self.ready.set()

    async def strict_get(self: Any) -> Any:
        outputs = getattr(self, "_cruise_strict_delta_outputs", None)
        if outputs is None:
            return await original_get(self)
        while not outputs:
            await self.ready.wait()
        output = outputs.popleft()
        if not outputs:
            self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    def strict_get_nowait(self: Any) -> Any:
        outputs = getattr(self, "_cruise_strict_delta_outputs", None)
        if outputs is None:
            return original_get_nowait(self)
        if not outputs:
            return None
        output = outputs.popleft()
        if not outputs:
            self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    RequestOutputCollector.__init__ = strict_init
    RequestOutputCollector.put = strict_put
    RequestOutputCollector.get = strict_get
    RequestOutputCollector.get_nowait = strict_get_nowait
    RequestOutputCollector._cruise_strict_delta_installed = True
    logger.info("Cruise enabled strict DELTA output collection")
