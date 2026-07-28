from .backend import NativeEpochOutput
from .contract import ResidentEpochPlan


class DeterministicTestEngine:
    """Control-plane test double. It never represents NPU evidence."""

    def execute(self, plan: ResidentEpochPlan) -> NativeEpochOutput:
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
            feed_calls=1,
            fetch_calls=1,
        )


def create_test_engine() -> DeterministicTestEngine:
    return DeterministicTestEngine()
