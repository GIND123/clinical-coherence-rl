"""Coherence as a Reward: label-free RL for rational belief revision in
clinical reasoning models.

Track A (``ccrl.panel``, ``ccrl.bedrock``, ``ccrl.prep``): a frozen, fully
archived panel of hosted models on Amazon Bedrock.
Track B (``ccrl.reward``, ``ccrl.train``): local GRPO training with the
coherence-gated reward.
Shared (``ccrl.data``, ``ccrl.prompts``, ``ccrl.analysis``): cases, prompts,
the posterior parser and the estimands, identical for both tracks.
"""
__version__ = "1.0.0"
