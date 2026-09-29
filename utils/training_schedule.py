"""Exact zero-based update schedules used for the two training stages."""


def training_phases(stage, step):
    if step < 0:
        raise ValueError("step must be non-negative")
    if stage == "rmd":
        generator = step % 6 == 5
        return generator, not generator
    if stage == "video_dmd":
        return step % 5 == 0, True
    raise ValueError(f"Unknown stage: {stage}")
