# Third-party notices

This repository is derived from Self Forcing and preserves its Apache-2.0
LICENSE. Wan model components retain their source attribution. Additional
research-code changes provide rollout-marginal scoring, replay, and NPU support.

- Self Forcing: https://github.com/guandeh17/Self-Forcing
- Wan 2.1: https://github.com/Wan-Video/Wan2.1
- FramePack memory helpers: https://github.com/lllyasviel/FramePack
  (original attribution remains in demo_utils/memory.py)
- Causal Forcing: causal ODE generator initialization, credited in the paper.
- Solaris: checkpointed self-forcing / replay formulation, credited in the paper.

Pretrained model weights are separate assets. The code LICENSE does not override
the terms attached to Wan base models or third-party initialization checkpoints.
This release does not bundle the training prompt corpus or generated videos.
