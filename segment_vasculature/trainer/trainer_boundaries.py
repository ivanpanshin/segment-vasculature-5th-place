import logging
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from segment_vasculature.callbacks.base import Callback
from segment_vasculature.metrics.base import BaseMetric


class Trainer:
    """
    Trainer class that covers the whole model training cycle via callbacks and parameters.
    """

    def __init__(
        self,
        cfg: DictConfig = None,
        model: torch.nn.Module = None,
        callbacks: Optional[List[Callback]] = None,
        metric: Optional[BaseMetric] = None,
        loaders: Dict[str, torch.utils.data.DataLoader] = None,
        optimizer: torch.optim.Optimizer = None,
        criterion: torch.nn.Module = None,
        scheduler: torch.optim.lr_scheduler = None,
    ) -> None:
        """Set-up the trainer.
        Args:
            TBA
        Returns:
            None
        """
        self.device = f'cuda:{int(os.environ["LOCAL_RANK"])}'
        self.model = model
        self.cfg = cfg
        self.callbacks = callbacks if callbacks is not None else []
        self.loaders = loaders if loaders is not None else {}
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        if metric is None:
            raise ValueError("Metric must not be None")
        self.metric = metric
        self.objective_metric_best = self.metric.first_objective_metric_best

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.accelerate.amp if self.cfg else True)

        if self.cfg.accelerate.amp:
            if self.cfg.accelerate.dtype == "torch.bfloat16":
                logging.info("Setting amp dtype to bfloat16")
                self.amp_dtype = torch.bfloat16
            else:
                logging.info("Setting amp dtype to auto")
                self.amp_dtype = None

        self.start_epoch = 0
        self.current_iter = 0

        self._call_callback_hooks("on_init_end")

    def _call_callback_hooks(
        self,
        hook_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Execute callbacks at corresponding places.
        Args:
            hook_name: Name of the place to execute callback (on_init_start, on_fit_end, etc).
        Returns:
            None
        """
        if hasattr(self, "callbacks"):
            for callback in self.callbacks:
                fn: Callable = getattr(callback, hook_name)
                if callable(fn):
                    fn(self, *args, **kwargs)

    def _train_step(
        self,
        input: torch.Tensor,
        labels: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> Dict[str, float]:
        self._call_callback_hooks("on_train_step_start")

        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.cfg.accelerate.amp, dtype=self.amp_dtype):
            model_predictions = self.model(input)

            loss = self.criterion(
                model_predictions,
                labels,
                boundaries=boundaries,
            )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler and not self.cfg.scheduler.scheduler_by_epoch:
                self.scheduler.step()

        self._call_callback_hooks("on_train_step_end")
        return {"model_predictions": model_predictions, "loss_item": loss.item()}

    @torch.no_grad()
    def _val_step(
        self,
        input: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        self._call_callback_hooks("on_val_step_start")

        with torch.cuda.amp.autocast(enabled=self.cfg.accelerate.amp, dtype=self.amp_dtype):
            model_predictions = self.model.module(input)

            loss = None
            if self.cfg.metric.track_val_loss and self.criterion and labels is not None:
                loss = self.criterion(
                    model_predictions,
                    labels,
                ).item()

        self._call_callback_hooks("on_val_step_end")
        return {"model_predictions": model_predictions, "loss_item": loss}

    def _logging_step(self, batch_index: int) -> None:
        """Add stats for future logging (mean loss, occupied memory, current LRs)
        Args:
            batch_index: Index of the current batch of the epoch.
        Returns:
            Dict with loss value.
        """
        if not hasattr(self, "logging_stats"):
            self.logging_stats = {k: [] for k in ["loss_raw", "loss_aggregated", "memory"]}

        self.logging_stats["loss_raw"] = self.losses[-1]
        self.logging_stats["loss_aggregated"] = round(np.mean(self.losses), 6)
        self.logging_stats["memory"] = round(torch.cuda.memory_reserved() / 1e9, 2)
        lrs = [_["lr"] for _ in self.optimizer.param_groups]
        for param_group_index, param_group_lr in enumerate(lrs):
            self.logging_stats[f"lr_param_group_{param_group_index}"] = round(param_group_lr, 10)

        if self.cfg.trainer.trainer_hyps.debug_iters:
            self.current_iter = batch_index + self.cfg.trainer.trainer_hyps.debug_iters * self.current_epoch
        else:
            self.current_iter = batch_index + len(self.loaders["train"]) * self.current_epoch

    def _train_epoch(self) -> Dict[str, float]:
        """Train a single epoch (train_step across the whole train loader)
        Returns:
            Dict with mean loss value.
        """
        self._call_callback_hooks("on_train_start")

        self.losses = []
        for batch_index, (input, labels, boundaries) in enumerate(tqdm(self.loaders["train"])):
            if (batch_index == 0) and hasattr(input, "shape"):
                logging.info(f"input shape {input.shape}")
            if (batch_index == 0) and hasattr(labels, "shape"):
                logging.info(f"labels shape: {labels.shape}")

            if self.cfg.trainer.trainer_hyps.debug_iters and self.cfg.trainer.trainer_hyps.debug_iters == batch_index:
                break

            input, labels, boundaries = input.to(self.device), labels.to(self.device), boundaries.to(self.device)
            preds = self._train_step(
                input=input,
                labels=labels,
                boundaries=boundaries,
            )
            if (batch_index == 0) and hasattr(preds["model_predictions"], "shape"):
                logging.info(f"preds shape {preds['model_predictions'].shape}")

            self.losses.append(preds["loss_item"])

            if batch_index % self.cfg.logging.iters == 0:
                self._logging_step(batch_index=batch_index)
                self._call_callback_hooks("on_logging_by_iter")

        if self.scheduler and self.cfg.scheduler.scheduler_by_epoch:
            self.scheduler.step()

        return {"loss_mean": np.mean(self.losses)}

    @torch.no_grad()
    def val_epoch(self, loader_name: str) -> Dict[str, Any]:
        """Evaluate model based on a particular loader defined by loader_name.
        Args:
            loader_name: Key from self.loaders for the evaluation loader.
        Returns:
            Dict with evaluation metrics.
        """
        self._call_callback_hooks("on_val_start")

        self.metric.prepare_predictions_init(self, loader_name)
        for batch_index, (input, labels, coordinates, paddings) in enumerate(tqdm(self.loaders[loader_name])):
            input, labels = input.to(self.device), labels.to(self.device)
            preds = self._val_step(
                input=input,
                labels=labels,
            )

            self.metric.prepare_predictions_batch(
                batch_index=batch_index,
                preds=preds,
                labels=labels,
                coordinates=coordinates,
                paddings=paddings,
            )

        dataset_preds = self.metric.calculate()
        self.metric.cleanup()

        self._call_callback_hooks("on_val_end")
        return dataset_preds

    def fit(
        self,
    ) -> None:
        """Run fit based on provided arguments (model, dataset, optimizer, loss, etc).
        Returns:
            None.
        """
        self._call_callback_hooks("on_fit_start")

        for epoch in range(self.start_epoch, self.cfg.trainer.trainer_hyps.num_epochs):
            self.loaders["train"].sampler.set_epoch(epoch)
            self.current_epoch = epoch
            self._call_callback_hooks("on_epoch_start")
            self.train_metrics = self._train_epoch()
            val_loaders_names = [key for key in self.loaders if key.startswith("val_")]

            self.val_metrics = {}

            for name in val_loaders_names:
                val_metrics_dataset = self.val_epoch(loader_name=name).items()
                for k, v in val_metrics_dataset:
                    if self.device == "cuda:0":
                        self.val_metrics[f"{name}/{k}"] = v

            if self.cfg.logging.track_train_metrics:
                train_metrics_dataset = self.val_epoch(loader_name="train")
                if self.device == "cuda:0":
                    for k, v in train_metrics_dataset:
                        self.train_metrics[f"{name}/{k}"] = v

            self._call_callback_hooks("on_epoch_end")

            torch.distributed.barrier()

        self._call_callback_hooks("on_fit_end")

    def calculate_val_metrics(
        self,
    ) -> None:
        self._call_callback_hooks("on_val_start")
        val_loaders_names = [key for key in self.loaders if key.startswith("val_")]
        val_metrics = {}

        for name in val_loaders_names:
            val_metrics_dataset = self.val_epoch(loader_name=name).items()
            if self.device == "cuda:0":
                for k, v in val_metrics_dataset:
                    val_metrics[f"{name}/{k}"] = v

                logging.info(f"Val metrics: {val_metrics}")
