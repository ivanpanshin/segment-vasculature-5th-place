import logging
import os
import time
from glob import glob

import mlflow

from .base import Callback


class LogMetrics_Mlflow(Callback):
    def __init__(self):
        super().__init__()

    @staticmethod
    def run_only_on_cuda0(func):
        def wrapper(self, trainer, *args, **kwargs):
            if trainer.device == "cuda:0":
                return func(self, trainer, *args, **kwargs)

        return wrapper

    @staticmethod
    def _log_metrics(trainer, metrics, prefix, step):
        mlflow.log_metrics({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)

    @staticmethod
    def _log_transforms(trainer, loader_name):
        if (
            hasattr(trainer.loaders[loader_name].dataset, "transform")
            and trainer.loaders[loader_name].dataset.transform
        ):
            k = f"{loader_name}_transform"
            v = str(trainer.loaders[loader_name].dataset.transform)
            if len(str(v)) <= 500:
                mlflow.log_params({k: v})
            else:
                for i in range(len(str(v)) // 500 + 1):
                    mlflow.log_params({f"{k}_{i}": str(v)[i * 500 : (i + 1) * 500]})

    @run_only_on_cuda0
    def on_init_end(self, trainer):
        mlflow.set_tracking_uri(trainer.cfg.logging.tracking_url)
        mlflow.set_experiment(trainer.cfg.logging.experiment_name)

        mlflow.start_run(run_name=trainer.cfg.logging.run_name)

        for k, v in dict(trainer.cfg).items():
            if len(str(v)) <= 500:
                mlflow.log_params({k: v})
            else:
                for i in range(len(str(v)) // 500 + 1):
                    mlflow.log_params({f"{k}_{i}": str(v)[i * 500 : (i + 1) * 500]})

        # val_loaders_names = [key for key in trainer.loaders if key.startswith("val_")]
        # self._log_transforms(trainer=trainer, loader_name="train")
        # self._log_transforms(trainer=trainer, loader_name=val_loaders_names[0])

    @run_only_on_cuda0
    def on_fit_end(self, trainer):
        if hasattr(trainer, "best_epoch"):
            mlflow.log_metric("best_epoch", trainer.best_epoch)

        mlflow.end_run()

    @run_only_on_cuda0
    def on_epoch_end(self, trainer):
        logging.info("Uploading all models to MLFlow")
        for file_path in glob(f"{trainer.cfg.logging.logging_dir}/*.pt"):
            mlflow.log_artifact(file_path)
            os.remove(file_path)

        if trainer.cfg.logging.track_train_metrics:
            self._log_metrics(
                trainer=trainer, metrics=trainer.train_metrics, prefix="train", step=trainer.current_epoch
            )
        self._log_metrics(trainer=trainer, metrics=trainer.val_metrics, prefix="val", step=trainer.current_epoch)
        logging.info(f"Epoch {trainer.current_epoch} Train {trainer.train_metrics} Val {trainer.val_metrics}")

    @run_only_on_cuda0
    def on_logging_by_iter(self, trainer):
        self._log_metrics(trainer=trainer, metrics=trainer.logging_stats, prefix="train", step=trainer.current_iter)

        info = f"Epoch {trainer.current_epoch} Iter {trainer.current_iter} Stats {trainer.logging_stats}"
        if hasattr(trainer, "eta"):
            info = f"ETA {trainer.eta} hours " + info
        logging.info(info)


class CalculateETA(Callback):
    def __init__(self):
        super().__init__()

    def on_init_end(self, trainer):
        trainer.start_time = time.time()

    def on_logging_by_iter(self, trainer):
        if trainer.current_iter > 0:
            total_number_iters = trainer.cfg.trainer.trainer_hyps.num_epochs * len(trainer.loaders["train"])
            elapsed_time = time.time() - trainer.start_time
            time_per_iter = elapsed_time / trainer.current_iter
            # counting the validation time as 10% of training epoch time
            eta_remaining_iters = 1.1 * (total_number_iters - trainer.current_iter) * time_per_iter

            trainer.eta = round(eta_remaining_iters / 3600, 2)
