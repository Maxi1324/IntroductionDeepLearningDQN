from __future__ import annotations
import numpy as np
import mlflow
import torch
from pathlib import Path
import uuid
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from dqn.parameter import Parameter
    from dqn.utils.training_logger import TrainingLogger
    from dqn.optimizer import Optimizer


class TrainingFinalizer:
    def __init__(
        self,
        param: Parameter,
        logger: TrainingLogger,
        optimizer: Optimizer,
        evaluate_fn: Callable[..., float],
        epoch_avg_lengths: list[float],
    ):
        self.param = param
        self.logger = logger
        self.optimizer = optimizer
        self.evaluate_fn = evaluate_fn
        self.epoch_avg_lengths = epoch_avg_lengths

    def finalize_training(self, env_steps_total: int):
        # Final metrics: average of last 100 epoch-level episode-length means
        if self.epoch_avg_lengths:
            avg_last_100 = float(np.mean(self.epoch_avg_lengths[-100:]))
            if self.logger.enable_mlflow:
                mlflow.log_metric("avg_episode_length_last_100", avg_last_100, step=self.param.epochs)
            self.logger._console(f"Avg episode length (last 100 epoch means): {avg_last_100:.2f}")

        # Evaluate final policy on fresh env rollouts (greedy-ish at final epsilon)
        try:
            eval_avg_len = self.evaluate_fn(
                episodes=20,
                max_episode_steps=10_000,
                env_steps_total=env_steps_total,
            )
            if self.logger.enable_mlflow:
                mlflow.log_metric("eval_avg_episode_length", eval_avg_len, step=self.param.epochs)
            self.logger._console(f"Eval avg episode length: {eval_avg_len:.2f}")
        except Exception as e:
            self.logger._console(f"Warning: evaluation failed: {e}")

        # Always save final model locally
        try:
            local_dir = Path("artifacts")
            local_dir.mkdir(parents=True, exist_ok=True)
            fname_parts = [getattr(self.param, "name", "dqn-run") or "dqn-run"]
            if getattr(self.param, "seed", None) is not None:
                fname_parts.append(f"seed{self.param.seed}")
            fname_parts.append(uuid.uuid4().hex[:6])
            local_path = local_dir / ("_".join(fname_parts) + ".pt")
            torch.save(self.optimizer.getOnlineNetwork().state_dict(), local_path)
            self.logger._console(f"Saved model locally to {local_path}")
            if self.logger.enable_mlflow:
                # Log path as parameter, as artifact logging may fail with some backends
                resolved_path = local_path.resolve()
                mlflow.log_param("model_path", str(resolved_path))
                self.logger._console(f"Logged model path to MLflow parameters: {resolved_path}")
        except Exception as e:
            self.logger._console(f"Warning: failed to save local model or log path: {e}")
