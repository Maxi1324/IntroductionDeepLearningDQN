from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional

import mlflow


class TrainingLogger:
    """Thin wrapper around MLflow logging so Agent stays clean."""

    def __init__(self, run_name: str = "dqn-run", tracking_uri: str = "file:mlruns") -> None:
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self._active: bool = False
        self._epoch_losses: list[float] = []
        self._current_epoch: Optional[int] = None
        self._total_epochs: Optional[int] = None

    @contextmanager
    def start_run(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_name=self.run_name):
            self._active = True
            try:
                yield
            finally:
                self._active = False

    def log_params(self, params: Dict[str, Any]) -> None:
        if not params:
            return
        filtered = {k: v for k, v in params.items() if v is not None}
        if filtered:
            mlflow.log_params(filtered)

    def log_all_params(self, param: Any) -> None:
        params = {
            "env": getattr(param, "env", None),
            "envsCount": getattr(param, "envsCount", None),
            "learningRate": getattr(param, "learningRate", None),
            "discount": getattr(param, "discount", None),
            "collectSteps": getattr(param, "collectSteps", None),
            "replayBufferSize": getattr(param, "replayBufferSize", None),
            "epochs": getattr(param, "epochs", None),
            "optimizationPerEpoch": getattr(param, "optimizationPerEpoch", None),
            "batchSize": getattr(param, "batchSize", None),
            "prewarm": getattr(param, "prewarm", None),
            "device": getattr(param, "Device", None),
            "network": getattr(param, "Network", None),
            "policy": getattr(param, "Policy", None),
            "epsilon": getattr(param, "epsilon", None),
            "run_name": getattr(param, "name", None),
        }
        self.log_params(params)

    def start_epoch(self, epoch: int, total_epochs: int) -> None:
        self._epoch_losses = []
        self._current_epoch = epoch
        self._total_epochs = total_epochs

    def log_loss(self, loss: float, step: int) -> None:
        mlflow.log_metric("loss", float(loss), step=step)
        self._epoch_losses.append(float(loss))

    def log_episode_stats(self, avg_len: Optional[float], avg_ret: Optional[float]) -> None:
        if self._current_epoch is None:
            return
        if avg_len is not None:
            mlflow.log_metric("avg_episode_length", avg_len, step=self._current_epoch)
        if avg_ret is not None:
            mlflow.log_metric("avg_episode_reward", avg_ret, step=self._current_epoch)

    def end_epoch(self, global_step: int) -> None:
        if self._current_epoch is None:
            return
        if self._epoch_losses:
            avg_loss = float(sum(self._epoch_losses) / len(self._epoch_losses))
            mlflow.log_metric("epoch_loss", avg_loss, step=self._current_epoch)
            total = self._total_epochs or 0
            self._console(
                f"Epoch {self._current_epoch + 1}/{total} | avg_loss={avg_loss:.4f} | steps={global_step}"
            )
        else:
            total = self._total_epochs or 0
            self._console(f"Epoch {self._current_epoch + 1}/{total} | no optimization steps run.")

    def _console(self, msg: str) -> None:
        print(msg)
