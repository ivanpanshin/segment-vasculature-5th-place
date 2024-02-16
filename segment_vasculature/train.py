import os

import cv2
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.distributed import destroy_process_group, init_process_group

from segment_vasculature.helpers import build_callbacks, build_loaders, build_metric, build_optim, seed_everything

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


@hydra.main(config_path="../configs", config_name="train")
def hydra_run(
    cfg: DictConfig,
) -> None:
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    seed_everything()

    model = instantiate(cfg.model)
    loaders = build_loaders(cfg=cfg)
    optim = build_optim(cfg=cfg, model=model)
    callbacks = build_callbacks(cfg=cfg)
    metric = build_metric(cfg=cfg)

    trainer = instantiate(
        cfg.trainer.trainer,
        cfg=cfg,
        model=optim["model"],
        callbacks=callbacks,
        metric=metric,
        loaders=loaders,
        optimizer=optim["optimizer"],
        criterion=optim["criterion"],
        scheduler=optim["scheduler"],
    )

    trainer.fit()

    destroy_process_group()


if __name__ == "__main__":
    hydra_run()
