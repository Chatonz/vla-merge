"""Integrate capture into an existing rollout loop; observations are already preprocessed."""

from pathlib import Path

from tcr_merging.calibration.collection import CaptureConfig, ReplayCollector


def capture(policy, episode_runner, *, name, checkpoint, output, selection, expected_tasks=10):
    """episode_runner must reset the policy per task and run complete episodes.

    It receives a capture_task_texts callback: call it with the raw task text
    after each reset and before processing observations. Then call
    policy.predict_action_chunk with native preprocessed batches
    (batch size one), not teacher-forced actions. No success filtering is applied.
    Use distinct episodes for A and B; selection is reservoir for A, quantiles for B.
    """
    collector = ReplayCollector(
        policy,
        CaptureConfig(
            name, Path(checkpoint), Path(output), selection=selection, expected_tasks=expected_tasks
        ),
    )
    collector.install()
    try:
        episode_runner(policy, collector.capture_task_texts)
        return collector.finish()
    finally:
        collector.close()
