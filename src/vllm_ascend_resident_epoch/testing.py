from .backend import NativeEpochOutput, NativeWarmupOutput
from .contract import ResidentEpochPlan


class DeterministicTestEngine:
    """Control-plane test double. It never represents NPU evidence."""

    def __init__(self) -> None:
        self.warmup_calls = 0
        self.execute_calls = 0

    def warm_up(self) -> NativeWarmupOutput:
        self.warmup_calls += 1
        return NativeWarmupOutput(
            status=0,
            model_calls=1,
            feed_calls=1,
            fetch_calls=1,
            wall_us=10,
            native_cpu_us=1,
            declared_input_bytes=260,
            declared_output_bytes=368,
        )

    def execute(self, plan: ResidentEpochPlan) -> NativeEpochOutput:
        self.execute_calls += 1
        token_ids = {
            request.req_id: [
                (request.token_id + step + 1) % 152064
                for step in range(plan.max_steps)
            ]
            for request in plan.requests
        }
        return NativeEpochOutput(
            status=0,
            model_calls=plan.max_steps,
            token_ids=token_ids,
            row_generations=plan.row_generations,
            feed_calls=1,
            fetch_calls=1,
            wall_us=10,
            native_cpu_us=1,
            declared_input_bytes=260,
            declared_output_bytes=368,
        )


def create_test_engine() -> DeterministicTestEngine:
    return DeterministicTestEngine()
