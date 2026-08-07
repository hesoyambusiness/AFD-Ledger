"""Scheduling equations shared by search and frontier perturbation code."""

from __future__ import annotations

from typing import Mapping


def afd_iteration_schedule(stage_times: Mapping[str, float], microbatch_num: int) -> tuple[float, str]:
    attn = stage_times["attention"]
    dispatch = stage_times["dispatch"]
    moe = stage_times["moe"]
    combine = stage_times["combine"]
    stage_sum = attn + dispatch + moe + combine
    max_stage_time = max(stage_times.values())
    steady_state_time = max(stage_sum, microbatch_num * max_stage_time)

    if microbatch_num == 1:
        return steady_state_time, "afd_no_overlap"

    if microbatch_num < 4:
        if stage_sum >= microbatch_num * max_stage_time:
            return steady_state_time, "afd_steady_state_underfilled"
        return steady_state_time, "afd_steady_state_pipeline"

    return steady_state_time, "afd_deep_pipeline"
