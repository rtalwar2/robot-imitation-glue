"""Minimal in-repo policies in plain pytorch.

- `policy.py` / `train.py` / `agent.py`: a behavior-cloning baseline that regresses a single action.
- `dit_policy.py` / `train_dit.py` / `dit_agent.py`: a flow-matching diffusion transformer that predicts an
  action chunk.

The policies and the training loops do not depend on lerobot or openPI.
Only the training scripts read the recorded datasets, which are stored on disk in the lerobot format.
"""
