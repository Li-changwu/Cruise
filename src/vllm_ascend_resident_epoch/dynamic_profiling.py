"""Configure CANN dynamic profiling in the process that owns the NPU."""

from __future__ import annotations

import os


_ENABLE_ENV = "VLLM_ASCEND_RESIDENT_EPOCH_DYNAMIC_PROFILING"
_BINDING_FILE_PREFIX = "dynamic-profiling-binding-"
_ORIGINAL_ENGINE_CORE_ENTRYPOINT = (
    "_resident_epoch_original_dynamic_profiling_engine_core_entrypoint"
)
_ENGINE_CORE_REBIND_INSTALLED = "_resident_epoch_dynamic_profiling_rebind_installed"


def _dynamic_profiling_enabled() -> bool:
    return (
        os.getenv(_ENABLE_ENV) == "1"
        or os.getenv("PROFILING_MODE") == "dynamic"
    )


def configure_dynamic_profiling() -> None:
    """Bind CANN's dynamic profiling key to this process when requested.

    The EngineCore process must set this before constructing its model executor
    so CANN creates the matching dynamic socket.
    """

    if not _dynamic_profiling_enabled():
        return
    pid = os.getpid()
    os.environ["PROFILING_MODE"] = "dynamic"
    os.environ["DYNAMIC_PROFILING_KEY_PID"] = str(pid)

    # /proc/<pid>/environ only reflects the exec-time environment on this
    # platform. Record the runtime value that CANN actually reads so the
    # hardware runner can verify a post-spawn rebind without trusting stale
    # process metadata.
    temporary_directory = os.getenv("TMPDIR")
    if temporary_directory:
        binding_path = os.path.join(
            temporary_directory, f"{_BINDING_FILE_PREFIX}{pid}.tsv"
        )
        with open(binding_path, "w", encoding="ascii") as binding_file:
            binding_file.write("key\tvalue\n")
            binding_file.write("profiling_mode\tdynamic\n")
            binding_file.write(f"dynamic_profiling_key_pid\t{pid}\n")


def _run_engine_core_with_dynamic_profiling(*args, **kwargs):
    """Rebind immediately inside vLLM's EngineCore process.

    vLLM normally uses ``fork`` here. A fork inherits the API server's
    environment and does not re-run ``sitecustomize`` or the server launcher,
    so the dynamic key must be corrected at this exact entry point. This module
    level function remains pickleable when vLLM instead uses ``spawn``.
    """

    configure_dynamic_profiling()

    from vllm.v1.engine.core import EngineCoreProc

    original = getattr(EngineCoreProc, _ORIGINAL_ENGINE_CORE_ENTRYPOINT, None)
    if original is None:
        # A spawn child imports this function without the parent-side patch.
        original = EngineCoreProc.run_engine_core
    return original(*args, **kwargs)


def install_engine_core_dynamic_profiling_rebind() -> None:
    """Install the fork-safe EngineCore entrypoint wrapper when profiling."""

    if not _dynamic_profiling_enabled():
        return

    from vllm.v1.engine.core import EngineCoreProc

    if getattr(EngineCoreProc, _ENGINE_CORE_REBIND_INSTALLED, False):
        return

    original = EngineCoreProc.run_engine_core
    setattr(EngineCoreProc, _ORIGINAL_ENGINE_CORE_ENTRYPOINT, original)
    EngineCoreProc.run_engine_core = staticmethod(
        _run_engine_core_with_dynamic_profiling
    )
    setattr(EngineCoreProc, _ENGINE_CORE_REBIND_INSTALLED, True)
