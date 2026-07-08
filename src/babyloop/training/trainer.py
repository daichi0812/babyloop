"""学習ループ。scripts/train.py から呼ばれる。"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import AdamW

from babyloop.data.dataloader import MultimodalDataModule, TextDataModule
from babyloop.data.tokenizer import load_tokenizer
from babyloop.models.configuration_babyloop import BabyloopConfig
from babyloop.models.modeling_babyloop import LoopedForCausalLM
from babyloop.training.checkpointing import CheckpointManager
from babyloop.training.run_record import RunRecorder


def _select_device() -> torch.device:
    override = os.environ.get("BABYLOOP_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _output_dir(cfg) -> Path:
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        return Path("outputs") / cfg.name / f"seed_{cfg.train.seed}"


def cosine_lr_words(words_seen: float, peak: float, min_ratio: float,
                    warmup_words: float, max_words: float) -> float:
    """words_seen 駆動の線形 warmup → cosine 減衰（地平線=max_words）。

    packed text は 1 step がほぼ一定語数なので、step 駆動 cosine と語数換算で一致する
    （①③ 再現性。tests/test_lr_schedule.py で tol 内一致を固定）。
    """
    if words_seen < warmup_words:
        return peak * (words_seen / max(1.0, warmup_words))
    progress = min(1.0, (words_seen - warmup_words) / max(1.0, max_words - warmup_words))
    return peak * (min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress)))


class Trainer:
    """合成済みconfigからモデル・データ・最適化を組み立てて学習を実行する。

    モデルは LoopedForCausalLM、データは BaseDataModule のインターフェース
    のみに依存し、2×2のどのセルでも同一の学習コードが走る。

    Args:
        cfg: Hydraで合成された全体config（model / data / train を含む）。
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def _build_model(self, tokenizer) -> tuple[LoopedForCausalLM, BabyloopConfig]:
        m = self.cfg.model
        ve = m.get("vision_encoder", {}) or {}      # 視覚（②④）。text では未使用
        conn = m.get("connector", {}) or {}
        config = BabyloopConfig(
            d_model=int(m.d_model),
            n_layers=int(m.n_layers),
            n_heads=int(m.n_heads),
            ffn_hidden=int(m.ffn_hidden),
            n_prelude=int(m.n_prelude),
            n_core=int(m.n_core),
            n_coda=int(m.n_coda),
            k=int(m.k),
            inject_input=bool(m.inject_input),
            vocab_size=len(tokenizer),
            max_seq_len=int(m.max_seq_len),
            rope_base=float(m.rope_base),
            rms_eps=float(m.rms_eps),
            tie_embeddings=bool(m.tie_embeddings),
            bias=bool(m.bias),
            dropout=float(m.dropout),
            fusion=m.get("fusion", None),
            vision_feature_dim=int(ve.get("feature_dim", 768)),
            n_visual_tokens=int(ve.get("n_visual_tokens", 0)),
            connector_type=str(conn.get("type", "mlp")),
            visual_inject_iters=int(conn.get("inject_iters", 0)),
            visual_inject_mode=str(conn.get("inject_mode", "prefix_refresh")),
            vision_layers=list(ve.get("layers") or []) or None,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return LoopedForCausalLM(config), config

    def _build_optimizer(self, model) -> AdamW:
        t = self.cfg.train
        decay, no_decay = [], []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (decay if p.ndim >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": float(t.weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return AdamW(groups, lr=float(t.lr), betas=(float(t.adam_beta1), float(t.adam_beta2)))

    def _build_lr_fn(self):
        """words_seen 駆動の cosine LR を返す（地平線=max_words_seen＝全セル共通）。

        step 駆動だと cosine 地平線が max_steps（セルごとにバッチ構成で変わる量）に乗り、
        max_steps を誤るとデータ予算だけでなく LR の形まで列間で食い違う（MM で実際に発生）。
        words_seen 駆動なら packed text（1 step≒一定語数）では step 駆動軌道をほぼ再現し
        （①③は no-op・再ラン不要、tests/test_lr_schedule.py で固定）、可変長 caption の MM だけ
        正しく直る。warmup も語数比（warmup_frac）で固定し max_steps↔warmup の手動連動を解消。
        max_steps は安全キャップに降格（停止は words_seen>=max_words_seen が主）。
        """
        t = self.cfg.train
        peak = float(t.lr)
        min_ratio = float(t.min_lr_ratio)
        max_words = max(1, int(t.max_words_seen))
        warmup_words = float(t.get("warmup_frac", 0.015)) * max_words
        return lambda words_seen: cosine_lr_words(words_seen, peak, min_ratio, warmup_words, max_words)

    @staticmethod
    def _unpack_batch(batch, device):
        """Text(tuple) / MM(dict) の両バッチ形を正規化する。

        Returns: ``(input_ids, attention_mask, labels, vision_features, batch_words)``。
        text 経路は attention_mask=None, vision_features=None, labels=input_ids となり、
        ①の ``model(input_ids=, labels=)`` 呼び出しと数値的に同一（回帰不変）。
        """
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels")
            labels = labels.to(device) if labels is not None else input_ids
            attn = batch.get("attention_mask")
            attn = attn.to(device) if attn is not None else None
            vis = batch.get("vision_features")
            vis = vis.to(device) if vis is not None else None
            return input_ids, attn, labels, vis, batch["batch_words"]
        input_ids, words = batch
        input_ids = input_ids.to(device)
        return input_ids, None, input_ids, None, words

    @staticmethod
    def _save_state(path: Path, model, optimizer, step: int, words_seen: int) -> None:
        """resume 用の学習状態を1ファイルに保存（rolling overwrite）。

        LR は words_seen の純関数（cosine_lr_words）なので scheduler state は保存不要＝
        resume 時は words_seen から LR を再計算すれば軌道が一致する。
        """
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "words_seen": words_seen,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    @staticmethod
    def _load_state(path: Path, model, optimizer, device) -> tuple[int, int]:
        """学習状態を復元し (step, words_seen) を返す（LR は words_seen から再計算するので不要）。"""
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng = state.get("rng", {})
        if rng.get("torch") is not None:
            # map_location で device に移っている可能性があるため CPU ByteTensor に戻す。
            torch.set_rng_state(rng["torch"].to("cpu", torch.uint8))
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.to("cpu", torch.uint8) for s in rng["cuda"]])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        return int(state["step"]), int(state["words_seen"])

    def fit(self) -> None:
        """学習を実行する。

        - grad accumulation で effective batch を稼ぐ（``step`` は optimizer-step を数える）。
        - 累積word数（packing メタの実語数を積算）でマイルストーン checkpoint を保存。
        - ``+resume=<training_state.pt or run dir>`` で optimizer/step/words/RNG を復元して継続
          （LR は words_seen の純関数なので scheduler state は不要）。
        """
        cfg = self.cfg
        t = cfg.train
        _seed_everything(int(t.seed))
        device = _select_device()

        resume_arg = cfg.get("resume", None)
        rec = RunRecorder(cfg.name, int(t.seed), resume=bool(resume_arg))
        rec.log(f"[train] device={device} experiment={cfg.name} seed={t.seed}")

        tokenizer = load_tokenizer(str(cfg.data.tokenizer_path))
        if cfg.data.get("modality", "text") == "multimodal":
            datamodule = MultimodalDataModule(
                cfg.data, cfg.train,
                n_visual_tokens=int(cfg.model.get("vision_encoder", {}).get("n_visual_tokens", 0)),
                pad_token_id=tokenizer.pad_token_id or 0,
                ablate_zero_vision=bool(cfg.data.get("ablate_zero_vision", False)),
            )
        else:
            datamodule = TextDataModule(cfg.data, cfg.train)
        datamodule.setup()
        dataloader = datamodule.train_dataloader()

        model, _ = self._build_model(tokenizer)
        model.to(device)
        model.train()

        optimizer = self._build_optimizer(model)
        lr_at = self._build_lr_fn()  # words_seen → lr（地平線=max_words_seen）

        out_dir = _output_dir(cfg)
        state_path = out_dir / "training_state.pt"
        ckpt = CheckpointManager(
            milestones=list(t.checkpoint_milestones), output_dir=str(out_dir), tokenizer=tokenizer
        )

        step = 0
        words_seen = 0
        if resume_arg:
            rp = Path(resume_arg)
            if rp.is_dir():
                rp = rp / "training_state.pt"
            step, words_seen = self._load_state(rp, model, optimizer, device)
            rec.log(f"[train] resumed from {rp} at step={step} words_seen={words_seen}")

        use_amp = device.type == "cuda"
        accum = max(1, int(t.get("grad_accum_steps", 1)))
        max_steps = int(t.max_steps)
        max_words = int(t.max_words_seen)
        log_every = int(t.log_every)
        save_state_every = int(t.get("save_state_every", 1000))
        grad_clip = float(t.grad_clip)

        micro = 0
        loss_val = float("nan")
        optimizer.zero_grad(set_to_none=True)
        done = False
        while not done:
            for batch in dataloader:
                input_ids, attn, labels, vision_features, batch_words = self._unpack_batch(batch, device)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(input_ids=input_ids, attention_mask=attn,
                                     vision_features=vision_features, labels=labels).loss
                else:
                    loss = model(input_ids=input_ids, attention_mask=attn,
                                 vision_features=vision_features, labels=labels).loss
                (loss / accum).backward()
                loss_val = loss.item()
                words_seen += int(batch_words.sum())
                micro += 1
                if micro < accum:
                    continue

                # --- optimizer step（accum 個のmicro-batch分） ---
                micro = 0
                lr = lr_at(words_seen)  # words_seen 駆動（地平線=max_words_seen）
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                saved = ckpt.maybe_save(model, words_seen, step)
                if saved:
                    rec.log(f"[train] step={step} words_seen={words_seen} saved={saved}")
                if step % log_every == 0:
                    rec.log(f"[train] step={step} loss={loss_val:.4f} lr={lr:.2e} words={words_seen}")
                    rec.metric(step=step, loss=loss_val, lr=float(lr), words_seen=words_seen)
                if step % save_state_every == 0:
                    self._save_state(state_path, model, optimizer, step, words_seen)

                if step >= max_steps or words_seen >= max_words:
                    done = True
                    break

        self._save_state(state_path, model, optimizer, step, words_seen)
        final = ckpt.save_final(model, words_seen)
        rec.log(f"[train] done. steps={step} words_seen={words_seen} final={final}")
        # 完了時チェック: データ予算未達（max_steps 頭打ち・早期停止・max_words 誤設定）を即検知。
        # 根本原因#2（最初の1本で words_seen を見ず5本に伝播）の再発防止（全原因に効く）。
        if words_seen < 0.98 * max_words:
            rec.log(
                f"[train] WARNING: words_seen={words_seen} < 0.98×max_words_seen={max_words}"
                f"（max_steps={max_steps} で頭打ちの可能性＝データ露出不足。max_steps を上げて再ラン推奨。"
                f"LR は words 駆動なので形は正しいが露出が足りない）"
            )
        rec.finalize(
            config=OmegaConf.to_container(cfg, resolve=True),
            summary={
                "steps": step,
                "words_seen": words_seen,
                "final_loss": loss_val,
                "grad_accum_steps": accum,
                "output_dir": str(out_dir),
                "final_checkpoint": final,
                "training_state": str(state_path),
                "saved_milestones": sorted(ckpt._saved),
            },
            device=str(device),
        )
