import logging
import os
from glob import glob

import cv2
import hydra
import numpy as np
import tifffile
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path="../../configs/preprocessing", config_name="create_3d_tensors")
def hydra_run(
    cfg: DictConfig,
) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/kidney_1_dense", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/kidney_3_dense", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/kidney_2_sparse", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/kidney_external", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/spleen_external", exist_ok=True)

    # kidney_1
    kidney_1_dense_path = f"{cfg.kaggle_root}/kidney_1_dense"
    kidney_1_dense_images_paths = sorted(glob(f"{kidney_1_dense_path}/images/*.tif"))
    kidney_1_dense_labels_paths = sorted(glob(f"{kidney_1_dense_path}/labels/*.tif"))
    kidney_1_depth = len(kidney_1_dense_images_paths)
    kidney_1_height, kidney_1_width = tifffile.imread(kidney_1_dense_images_paths[0]).shape
    kidney_1_full_image, kidney_1_full_label = np.zeros(
        (kidney_1_depth, kidney_1_height, kidney_1_width), dtype=np.uint8
    ), np.zeros((kidney_1_depth, kidney_1_height, kidney_1_width), dtype=np.uint8)

    for index, (image_path, label_path) in tqdm(
        enumerate(zip(kidney_1_dense_images_paths, kidney_1_dense_labels_paths))
    ):
        kidney_1_full_image[index] = (tifffile.imread(image_path) / 256).astype(np.uint8)
        kidney_1_full_label[index] = (tifffile.imread(label_path) / 255).astype(np.uint8)

    assert kidney_1_full_image.shape == kidney_1_full_label.shape

    logging.info(f"kidney 1 full shape: {kidney_1_full_image.shape}")
    tifffile.imwrite(f"{cfg.output_dir}/kidney_1_dense/full_image.tif", kidney_1_full_image)
    tifffile.imwrite(f"{cfg.output_dir}/kidney_1_dense/full_label.tif", kidney_1_full_label)
    logging.info(f"Kidney 1 saved to {cfg.output_dir}/kidney_1_dense/full_*")

    # kidney_2
    kidney_2_dense_path = f"{cfg.kaggle_root}/kidney_2"
    kidney_2_dense_images_paths = sorted(glob(f"{kidney_2_dense_path}/images/*.tif"))
    kidney_2_dense_labels_paths = sorted(glob(f"{kidney_2_dense_path}/labels/*.tif"))

    kidney_2_depth = len(kidney_2_dense_images_paths)
    kidney_2_height, kidney_2_width = tifffile.imread(kidney_2_dense_images_paths[0]).shape
    kidney_2_full_image, kidney_2_full_label = np.zeros(
        (kidney_2_depth, kidney_2_height, kidney_2_width), dtype=np.uint8
    ), np.zeros((kidney_2_depth, kidney_2_height, kidney_2_width), dtype=np.uint8)

    for index, (image_path, label_path) in tqdm(
        enumerate(zip(kidney_2_dense_images_paths, kidney_2_dense_labels_paths))
    ):
        kidney_2_full_image[index] = (tifffile.imread(image_path) / 256).astype(np.uint8)
        kidney_2_full_label[index] = (tifffile.imread(label_path) / 256).astype(np.uint8)

    assert kidney_2_full_image.shape == kidney_2_full_label.shape

    logging.info(f"kidney 2 full shape: {kidney_2_full_image.shape}")
    tifffile.imwrite(f"{cfg.output_dir}/kidney_2_sparse/full_image.tif", kidney_2_full_image)
    tifffile.imwrite(f"{cfg.output_dir}/kidney_2_sparse/full_label.tif", kidney_2_full_label)
    logging.info(f"Kidney 2 saved to {cfg.output_dir}/kidney_2_sparse/full_*")

    # kidney_3
    kidney_3_dense_images_paths = sorted(glob(f"{cfg.kaggle_root}/kidney_3_sparse/images/*.tif"))
    kidney_3_dense_labels_paths = sorted(glob(f"{cfg.kaggle_root}/kidney_3_dense/labels/*.tif"))

    kidney_3_dense_images_paths_filtered, kidney_3_dense_labels_paths_filtered = [], []
    for image_path in kidney_3_dense_images_paths:
        if image_path.split("/")[-1] in [_.split("/")[-1] for _ in kidney_3_dense_labels_paths]:
            kidney_3_dense_images_paths_filtered.append(image_path)
            kidney_3_dense_labels_paths_filtered.append(
                f'{cfg.kaggle_root}/kidney_3_dense/labels/{image_path.split("/")[-1]}'
            )

    kidney_3_depth = len(kidney_3_dense_images_paths_filtered)
    kidney_3_height, kidney_3_width = tifffile.imread(kidney_3_dense_images_paths_filtered[0]).shape
    kidney_3_full_image, kidney_3_full_label = np.zeros(
        (kidney_3_depth, kidney_3_height, kidney_3_width), dtype=np.uint8
    ), np.zeros((kidney_3_depth, kidney_3_height, kidney_3_width), dtype=np.uint8)

    for index, (image_path, label_path) in tqdm(
        enumerate(zip(kidney_3_dense_images_paths_filtered, kidney_3_dense_labels_paths_filtered))
    ):
        kidney_3_full_image[index] = (tifffile.imread(image_path) / 256).astype(np.uint8)
        kidney_3_full_label[index] = (tifffile.imread(label_path) / 256).astype(np.uint8)

    assert kidney_3_full_image.shape == kidney_3_full_label.shape
    logging.info(f"kidney 3 full shape: {kidney_3_full_image.shape}")
    tifffile.imwrite(f"{cfg.output_dir}/kidney_3_dense/full_image.tif", kidney_3_full_image)
    tifffile.imwrite(f"{cfg.output_dir}/kidney_3_dense/full_label.tif", kidney_3_full_label)
    logging.info(f"Kidney 3 saved to {cfg.output_dir}/kidney_3_dense/full_*")

    # kidney external
    kidney_external_dense_path = f"{cfg.external_root}/_50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2_/50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2_"  # noqa: E501
    kidney_external_dense_images_paths = sorted(glob(f"{kidney_external_dense_path}/*.jp2"))
    kidney_external_depth = len(kidney_external_dense_images_paths)
    kidney_external_height, kidney_external_width = cv2.imread(
        kidney_external_dense_images_paths[0], cv2.IMREAD_UNCHANGED
    ).shape[:2]

    kidney_external_full_image = np.zeros(
        (kidney_external_depth, kidney_external_height, kidney_external_width), dtype=np.uint8
    )
    for index, path in tqdm(enumerate(kidney_external_dense_images_paths)):
        kidney_external_full_image[index] = (cv2.imread(path, cv2.IMREAD_UNCHANGED) / 256).astype(np.uint8)

    logging.info(f"kidney external full shape: {kidney_external_full_image.shape}")
    tifffile.imwrite(f"{cfg.output_dir}/kidney_external/full_image.tif", kidney_external_full_image)
    logging.info(f"Kidney external saved to {cfg.output_dir}/kidney_external/full_image.tif")

    # spleen external
    spleen_external_dense_path = (
        f"{cfg.external_root}/_50.16um_LADAF_2020-27_spleen_pag_jp2_/50.16um_LADAF_2020-27_spleen_pag_jp2_"
    )
    spleen_external_dense_images_paths = sorted(glob(f"{spleen_external_dense_path}/*.jp2"))
    spleen_external_depth = len(spleen_external_dense_images_paths)
    spleen_external_height, spleen_external_width = cv2.imread(
        spleen_external_dense_images_paths[0], cv2.IMREAD_UNCHANGED
    ).shape[:2]

    spleen_external_full_image = np.zeros(
        (spleen_external_depth, spleen_external_height, spleen_external_width), dtype=np.uint8
    )
    for index, path in tqdm(enumerate(spleen_external_dense_images_paths)):
        spleen_external_full_image[index] = (cv2.imread(path, cv2.IMREAD_UNCHANGED) / 256).astype(np.uint8)

    logging.info(f"spleen external full shape: {spleen_external_full_image.shape}")
    tifffile.imwrite(f"{cfg.output_dir}/spleen_external/full_image.tif", spleen_external_full_image)
    logging.info(f"spleen external saved to {cfg.output_dir}/spleen_external/full_image.tif")


if __name__ == "__main__":
    hydra_run()
