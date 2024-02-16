import logging
import os

import hydra
import numpy as np
import tifffile
from omegaconf import DictConfig
from skimage.segmentation import find_boundaries
from tqdm import tqdm


def create_boundaries(label_path, boundaries_path):
    label = tifffile.memmap(label_path, mode="r")

    boundaries = np.zeros(label.shape, dtype=np.float16)
    for slice_index in tqdm(range(label.shape[0])):
        boundaries[slice_index] = find_boundaries(label[slice_index].astype(np.uint8), mode="inner").astype(np.float16)

    base_path = "/".join(boundaries_path.split("/")[:-1])
    os.makedirs(base_path, exist_ok=True)
    tifffile.imwrite(boundaries_path, boundaries)
    logging.info(f"Boundaries calculated and saved to {boundaries_path}")


@hydra.main(config_path="../../configs/preprocessing", config_name="create_boundaries")
def hydra_run(
    cfg: DictConfig,
) -> None:
    for label_path, boundaries_path in zip(cfg.label_paths, cfg.boundaries_paths):
        create_boundaries(
            label_path=label_path,
            boundaries_path=boundaries_path,
        )


if __name__ == "__main__":
    hydra_run()
