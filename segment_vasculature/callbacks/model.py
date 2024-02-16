import heapq
import logging
import os

import mlflow
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_ema import ExponentialMovingAverage

from .base import Callback


class TrainEval(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer):
        trainer.model.train()

    def on_val_start(self, trainer):
        trainer.model.eval()

    def on_train_step_start(self, trainer):
        assert trainer.model.training

    def on_val_step_start(self, trainer):
        assert not trainer.model.training


class HeapItem:
    def __init__(self, metric, path):
        self.metric = metric
        self.path = path

    def __lt__(self, other):
        return self.metric < other.metric


class SaveModel(Callback):
    def __init__(self, top_k):
        super().__init__()
        self.top_items = []
        self.top_k = top_k

    @staticmethod
    def run_only_on_cuda0(func):
        def wrapper(self, trainer):
            if trainer.device == "cuda:0":
                return func(self, trainer)

        return wrapper

    @staticmethod
    def with_ema_check(func):
        def wrapper(self, trainer, metric_improved, item, deleted_item_path):
            if hasattr(trainer, "ema"):
                with trainer.ema.average_parameters():
                    return func(self, trainer, metric_improved, item, deleted_item_path)
            else:
                return func(self, trainer, metric_improved, item, deleted_item_path)

        return wrapper

    @run_only_on_cuda0
    def on_init_end(self, trainer):
        run_number = 0

        logging_dir_run_name = os.path.join(
            trainer.cfg.logging.logging_dir, trainer.cfg.logging.experiment_name, trainer.cfg.logging.run_name
        )
        while os.path.isdir(f"{logging_dir_run_name}_{str(run_number)}"):
            run_number += 1

        logging_dir_run_name_run_number = f"{logging_dir_run_name}_{str(run_number)}"
        logging.info(f"Creating logging dir {logging_dir_run_name_run_number}")
        os.makedirs(logging_dir_run_name_run_number, exist_ok=False)
        trainer.cfg.logging.logging_dir = logging_dir_run_name_run_number

    def _save_model(self, trainer, logging_name) -> None:
        if logging_name == "best.pt":
            trainer.best_epoch = trainer.current_epoch

        torch.save(
            {
                "model": trainer.model.module.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "scheduler": trainer.scheduler.state_dict(),
                "epoch": trainer.current_epoch,
                "iter": trainer.current_iter,
            },
            os.path.join(trainer.cfg.logging.logging_dir, logging_name),
        )
        logging.info("Saved new best model")

    @with_ema_check
    def _save_models(self, trainer, metric_improved, item, deleted_item_path):
        logging.info("Model was configured with EMA - saving EMA weights")
        self._save_model(trainer=trainer, logging_name="last.pt")
        if metric_improved:
            self._save_model(trainer=trainer, logging_name="best.pt")

        if trainer.cfg.logging.save_all_epochs:
            self._save_model(trainer=trainer, logging_name=item.path)
        elif item in self.top_items:
            self._save_model(trainer=trainer, logging_name=item.path)

        if deleted_item_path and not trainer.cfg.logging.save_all_epochs:
            os.remove(deleted_item_path)

    @run_only_on_cuda0
    def on_epoch_end(self, trainer):
        value = round(trainer.val_metrics[trainer.metric.main_metric], 5)
        item = HeapItem(
            trainer.val_metrics[trainer.metric.main_metric],
            f"epoch_{trainer.current_epoch}_{trainer.metric.main_metric.split('/')[-1]}_{value}.pt",
        )
        deleted_item_path = None
        if len(self.top_items) < self.top_k:
            heapq.heappush(self.top_items, item)
        elif trainer.val_metrics[trainer.metric.main_metric] > self.top_items[0].metric:
            deleted_item_path = mlflow.artifacts.download_artifacts(
                run_id=mlflow.active_run().info.run_id, artifact_path=self.top_items[0].path
            )
            heapq.heapreplace(self.top_items, item)

        metric_improved = False
        if trainer.metric.main_metric_improved(
            prev_value=trainer.objective_metric_best,
            current_value=trainer.val_metrics[trainer.metric.main_metric],
        ):
            trainer.objective_metric_best = trainer.val_metrics[trainer.metric.main_metric]
            metric_improved = True

        self._save_models(
            trainer=trainer,
            metric_improved=metric_improved,
            item=item,
            deleted_item_path=deleted_item_path,
        )


class EMA(Callback):
    def __init__(self, ema_decay_per_epoch):
        super().__init__()
        self.ema_decay_per_epoch = ema_decay_per_epoch

    def on_init_end(self, trainer):
        num_iters = len(trainer.loaders["train"])
        ema_decay_per_iter = self.ema_decay_per_epoch ** (1 / num_iters)
        trainer.ema = ExponentialMovingAverage(trainer.model.parameters(), decay=ema_decay_per_iter)
        trainer.ema.to(trainer.device)
        logging.info("Confgured EMA")

    def on_val_start(self, trainer):
        trainer.ema.store()
        trainer.ema.copy_to()

    def on_val_end(self, trainer):
        trainer.ema.restore()

    def on_train_step_end(self, trainer):
        trainer.ema.update()


class ChangeModelDevice(Callback):
    def __init__(self):
        super().__init__()

    def on_init_end(self, trainer):
        if not isinstance(trainer.model, torch.nn.Module):
            raise ValueError("Model should be of type torch.nn.Module")
        else:
            trainer.model.to(trainer.device)
            # trainer.model = DDP(trainer.model, device_ids=[trainer.device], find_unused_parameters=True)
            trainer.model = DDP(trainer.model, device_ids=[trainer.device])


class ChangeModelDtype(Callback):
    def __init__(self, dtype: str):
        super().__init__()
        dtype_dict = dict(
            float32=torch.float32,
            float64=torch.float64,
            double=torch.double,
            float16=torch.float16,
            bfloat16=torch.bfloat16,
            half=torch.half,
            uint8=torch.uint8,
            int8=torch.int8,
            int16=torch.int16,
            short=torch.short,
            int32=torch.int32,
            int=torch.int,
            int64=torch.int64,
            long=torch.long,
        )

        self.dtype = dtype_dict[dtype]

    def on_init_end(self, trainer):
        if not isinstance(trainer.model, torch.nn.Module):
            raise ValueError("Model should be of type torch.nn.Module")
        else:
            trainer.model.to(dtype=self.dtype)


class Replace8bitLinear(Callback):
    def __init__(
        self,
    ):
        pass

    def on_init_end(self, trainer):
        if not isinstance(trainer.model, torch.nn.Module):
            raise ValueError("Model should be of type torch.nn.Module")

        from transformers.utils.bitsandbytes import replace_8bit_linear as replace_8bit_linear_fn

        trainer.model = replace_8bit_linear_fn(trainer.model)
        trainer.model.is_loaded_in_8bit = True


class ResumeTraining(Callback):
    def __init__(self, mlflow_run_id):
        super().__init__()
        self.mlflow_run_id = mlflow_run_id

    def on_init_end(self, trainer):
        if trainer.device == "cuda:0":
            old_mlflow_run_id = mlflow.active_run().info.run_id
            mlflow.end_run()
            mlflow.delete_run(old_mlflow_run_id)
        else:
            mlflow.set_tracking_uri(trainer.cfg.logging.tracking_url)
            mlflow.set_experiment(trainer.cfg.logging.experiment_name)

        checkpoint_path = mlflow.artifacts.download_artifacts(run_id=self.mlflow_run_id, artifact_path="last.pt")
        checkpoint = torch.load(checkpoint_path)

        logging.info("Resuming training from checkpoint")
        model_keys = trainer.model.module.state_dict().keys()
        checkpoint_keys = checkpoint["model"].keys()
        logging.info(f"The following model keys won't be loaded {set(checkpoint_keys).difference(set(model_keys))}")

        trainer.model.module.load_state_dict(checkpoint["model"], strict=False)
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler"])
        trainer.start_epoch = checkpoint["epoch"]
        trainer.current_iter = checkpoint["iter"]

        if trainer.device == "cuda:0":
            mlflow.start_run(run_id=self.mlflow_run_id)


class LoadModelFromDisk(Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def on_init_end(self, trainer):
        checkpoint = torch.load(self.path)

        logging.info(f"Loading checkpoint {self.path}")
        model_keys = trainer.model.module.state_dict().keys()
        checkpoint_keys = checkpoint["model"].keys()
        logging.info(f"The following model keys won't be loaded {set(checkpoint_keys).difference(set(model_keys))}")

        trainer.model.module.load_state_dict(checkpoint["model"], strict=True)
