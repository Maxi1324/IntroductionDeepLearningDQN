from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional
import time
import logging
import os

import mlflow

LOG_TO_MLFLOW = True


class TrainingLogger:

    def __init__(
        self,
        run_name: str = "dqn-run",
        experiment_name: str | None = None,
        tracking_uri: str | None = None,
        consoleLogging: bool = False,
        enable_mlflow: bool = True,
    ) -> None:
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "postgresql://postgres@localhost:5432/mlflow"
        )
        self.enable_mlflow = enable_mlflow and LOG_TO_MLFLOW
        self._active: bool = False
        self._epoch_losses: list[float] = []
        self._current_epoch: Optional[int] = None
        self._total_epochs: Optional[int] = None
        self.consoleLogging = consoleLogging
        self._run_start: Optional[float] = None
        self._quiet_logs_configured = False
        self._buffered_epoch_metrics: list[dict[str, float | int]] = []
        self._flush_interval_epochs: int = 20
        self._pending_avg_len: Optional[float] = None
        self._pending_avg_ret: Optional[float] = None
        self._epoch_start: Optional[float] = None

    @contextmanager
    def start_run(self):
        if not self.enable_mlflow:
            yield
            return
        if not self._quiet_logs_configured:
            logging.getLogger("mlflow").setLevel(logging.ERROR)
            logging.getLogger("alembic").setLevel(logging.ERROR)
            self._quiet_logs_configured = True
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.run_name):
            self._active = True
            self._run_start = time.perf_counter()
            try:
                yield
            finally:
                if self._run_start is not None:
                    duration = time.perf_counter() - self._run_start
                    mlflow.log_metric("run_duration_seconds", duration)
                    self._run_start = None
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
            "epsilonEnd": getattr(param, "epsilonEnd", None),
            "seed": getattr(param, "seed", None),
            "run_name": getattr(param, "name", None),
        }
        self.log_params(params)

    def start_epoch(self, epoch: int, total_epochs: int) -> None:
        self._epoch_losses = []
        self._current_epoch = epoch
        self._total_epochs = total_epochs
        self._pending_avg_len = None
        self._pending_avg_ret = None
        self._epoch_start = time.perf_counter()

    def log_loss(self, loss: float, step: int) -> None:
        self._epoch_losses.append(float(loss))

    def log_episode_stats(self, avg_len: Optional[float], avg_ret: Optional[float]) -> None:
        if self._current_epoch is None:
            return
        self._pending_avg_len = avg_len
        self._pending_avg_ret = avg_ret

    def end_epoch(self, global_step: int) -> None:
        if self._current_epoch is None:
            return
        if self._epoch_losses:
            avg_loss = float(sum(self._epoch_losses) / len(self._epoch_losses))
            metrics: dict[str, float | int] = {"epoch": self._current_epoch, "epoch_loss": avg_loss}
            if self._pending_avg_len is not None:
                metrics["avg_episode_length"] = self._pending_avg_len
            if self._pending_avg_ret is not None:
                metrics["avg_episode_reward"] = self._pending_avg_ret
            metrics["global_step"] = global_step
            if self._epoch_start is not None:
                metrics["epoch_duration_seconds"] = time.perf_counter() - self._epoch_start
            self._buffered_epoch_metrics.append(metrics)
            total = self._total_epochs or 0
            self._console(
                f"Epoch {self._current_epoch + 1}/{total} | avg_loss={avg_loss:.4f} | steps={global_step}"
            )
        else:
            total = self._total_epochs or 0
            self._console(f"Epoch {self._current_epoch + 1}/{total} | no optimization steps run.")

        if self.enable_mlflow and self._total_epochs is not None:
            if ((self._current_epoch + 1) % self._flush_interval_epochs == 0) or (
                self._current_epoch + 1 == self._total_epochs
            ):
                for m in self._buffered_epoch_metrics:
                    mlflow.log_metric("epoch_loss", float(m["epoch_loss"]), step=int(m["epoch"]))
                    if "avg_episode_length" in m:
                        mlflow.log_metric("avg_episode_length", float(m["avg_episode_length"]), step=int(m["epoch"]))
                    if "avg_episode_reward" in m:
                        mlflow.log_metric("avg_episode_reward", float(m["avg_episode_reward"]), step=int(m["epoch"]))
                    if "epoch_duration_seconds" in m:
                        mlflow.log_metric(
                            "epoch_duration_seconds", float(m["epoch_duration_seconds"]), step=int(m["epoch"])
                        )
                self._buffered_epoch_metrics.clear()
                self._pending_avg_len = None
                self._pending_avg_ret = None
                self._epoch_start = None

    def _console(self, msg: str) -> None:
        if self.consoleLogging:
            print(msg)
