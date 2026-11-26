"""vLLM generation for Track B: GRPO rollouts and local evaluation.

The transformers `generate` loop is the bottleneck of a GRPO step (160 rollouts of up to
512 tokens per step). vLLM generates the same rollouts many times faster:

  * one vLLM engine holds the frozen base weights; the current policy is the LoRA adapter,
    saved after every optimiser step and loaded as a new `LoRARequest` (a fresh id per step,
    so vLLM never serves a stale adapter);
  * prompts are tokenised by the training tokenizer (`encode_chat`) and passed as token ids,
    so the rollout and the training forward pass see identical tokens;
  * vLLM returns the log-probability of each sampled token; `pg_kl_loss` uses it for the
    truncated importance-sampling correction (`tis_cap`), which absorbs the small numerical
    differences between the vLLM and transformers kernels;
  * with `sleep: true` the engine frees its KV cache and parks its weights in CPU memory
    during the backward pass (needed for Qwen3-8B on a 48 GB card; not on 80 GB cards).

Training samples at temperature 1.0 / top-p 1.0 (config/train/base.yaml), so the sampling
distribution equals the policy the loss differentiates. Evaluation keeps the registered
decoding (temperature 0.7, top-p 0.95) with a fixed seed per item.

vLLM is imported lazily: nothing here is needed on the CPU-only analysis machine.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


class VLLMEngine:
    def __init__(self, model: str, *, max_new_tokens: int = 512, temperature: float = 1.0, top_p: float = 1.0,
                 gpu_memory_utilization: float = 0.3, max_model_len: int = 4096, lora_rank: int | None = None,
                 enable_sleep: bool = False, seed: int = 0, dtype: str = "bfloat16",
                 eos_token_ids: list[int] | None = None):
        from vllm import LLM

        kw = dict(model=model, dtype=dtype, gpu_memory_utilization=float(gpu_memory_utilization),
                  max_model_len=int(max_model_len), seed=int(seed), enable_prefix_caching=True)
        if lora_rank:
            kw.update(enable_lora=True, max_lora_rank=int(lora_rank), max_loras=1)
        if enable_sleep:
            kw["enable_sleep_mode"] = True
        self.llm = LLM(**kw)
        self.max_new, self.temp, self.top_p = int(max_new_tokens), float(temperature), float(top_p)
        self.enable_sleep = bool(enable_sleep)
        self.asleep = False
        self.eos = set(int(x) for x in (eos_token_ids or []))
        self._lora_id = 0

    # -------------------------------------------------------------- memory
    def sleep(self) -> None:
        if self.enable_sleep and not self.asleep:
            self.llm.sleep(level=1)
            self.asleep = True

    def wake(self) -> None:
        if self.asleep:
            self.llm.wake_up()
            self.asleep = False

    # -------------------------------------------------------------- generation
    def _lora(self, path: str | Path | None):
        if not path:
            return None
        from vllm.lora.request import LoRARequest
        self._lora_id += 1
        return LoRARequest(f"policy-{self._lora_id}", self._lora_id, str(path))

    def generate(self, prompt_ids: list[list[int]], *, lora_path: str | Path | None = None,
                 seeds: list[int] | None = None, temperature: float | None = None, top_p: float | None = None,
                 max_new_tokens: int | None = None, logprobs: bool = True
                 ) -> list[tuple[list[int], str, list[float] | None]]:
        """(completion token ids, text, per-token log-probs of the sampled tokens) per prompt, in order."""
        from vllm import SamplingParams

        self.wake()
        t = self.temp if temperature is None else float(temperature)
        p = self.top_p if top_p is None else float(top_p)
        n = self.max_new if max_new_tokens is None else int(max_new_tokens)

        def sp(seed):
            return SamplingParams(temperature=t, top_p=p, max_tokens=n, seed=seed,
                                  logprobs=1 if logprobs else None)

        params = [sp(int(s)) for s in seeds] if seeds is not None else sp(None)
        outs = self.llm.generate([{"prompt_token_ids": list(x)} for x in prompt_ids], params,
                                 lora_request=self._lora(lora_path), use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            ids = [int(x) for x in c.token_ids]
            lps = None
            if logprobs and c.logprobs is not None:
                lps = []
                for tok, d in zip(ids, c.logprobs):
                    lp = d.get(tok) if d else None
                    lps.append(float(lp.logprob) if lp is not None else math.nan)
            # stopped on EOS but the EOS id is not in the output: append it, so the policy is
            # trained to stop exactly as the transformers backend trains it (no log-prob -> TIS weight 1)
            if (getattr(c, "finish_reason", None) == "stop" and getattr(c, "stop_reason", None) is None
                    and self.eos and (not ids or ids[-1] not in self.eos)):
                ids.append(min(self.eos))
                if lps is not None:
                    lps.append(math.nan)
            res.append((ids, c.text, lps))
        return res


class VLLMRolloutGenerator:
    """Drop-in for grpo.HFGenerator: saves the current LoRA policy, generates with vLLM,
    then lets the engine sleep (if enabled) while the trainer runs the backward pass."""

    def __init__(self, engine: VLLMEngine, model, tok, adapter_dir: Path, chat_kwargs: dict):
        self.engine, self.model, self.tok = engine, model, tok
        self.adapter_dir = Path(adapter_dir)
        self.kw = chat_kwargs

    def __call__(self, convs: list[list[dict]]):
        from ccrl.train.common import encode_chat

        prompts = [encode_chat(self.tok, c, self.kw) for c in convs]
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.adapter_dir)          # PEFT: adapter weights + config only
        outs = self.engine.generate(prompts, lora_path=self.adapter_dir)
        self.engine.sleep()
        return [(list(p), ids, text, lps) for p, (ids, text, lps) in zip(prompts, outs)]


def adapter_rank(adapter: str | None) -> int | None:
    if not adapter:
        return None
    cfg = Path(adapter) / "adapter_config.json"
    return int(json.loads(cfg.read_text())["r"]) if cfg.exists() else 64
