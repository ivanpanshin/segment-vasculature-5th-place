import logging

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


class Dataset2DMultiPlanes(Dataset):
    def __init__(
        self,
        dataset_root,
        crop_size,
        overlap_size,
        split,
        multiplier=1,
        transform=None,
        pseudo_annotations_path=None,
    ):
        super().__init__()
        self.crop_size = crop_size
        self.image = tifffile.memmap(f"{dataset_root}/full_image.tif", mode="r")

        if split != "test":
            if pseudo_annotations_path:
                logging.info("Reading pseudo annotations")
                self.label = tifffile.memmap(pseudo_annotations_path, mode="r")
            else:
                self.label = tifffile.memmap(f"{dataset_root}/full_label.tif", mode="r")

            assert self.label.min() >= 0 and self.label.max() <= 1
            assert self.image.shape == self.label.shape

        assert self.image.min() >= 0 and self.image.max() <= 256
        logging.info(f"Full image shape: {self.image.shape}")

        step_size = crop_size - overlap_size
        self.depth, self.height, self.width = self.image.shape

        # calculate XY coordinates
        xy_coordinates = []
        for z in range(self.depth):
            for y in range(0, self.height - step_size, step_size):
                for x in range(0, self.width - step_size, step_size):
                    crop_end_y = min(y + crop_size, self.height)
                    crop_end_x = min(x + crop_size, self.width)

                    xy_coordinates.append((z, z + 1, y, crop_end_y, x, crop_end_x))

        # calculate XZ coordinates
        xz_coordinates = []
        for z in range(0, self.depth - step_size, step_size):
            for y in range(self.height):
                for x in range(0, self.width - step_size, step_size):
                    crop_end_z = min(z + crop_size, self.depth)
                    crop_end_x = min(x + crop_size, self.width)

                    xz_coordinates.append((z, crop_end_z, y, y + 1, x, crop_end_x))

        # calculate YZ coordinates
        yz_coordinates = []
        for z in range(0, self.depth - step_size, step_size):
            for y in range(0, self.height - step_size, step_size):
                for x in range(self.width):
                    crop_end_z = min(z + crop_size, self.depth)
                    crop_end_y = min(y + crop_size, self.height)

                    yz_coordinates.append((z, crop_end_z, y, crop_end_y, x, x + 1))

        logging.info(
            f"num xy slices: {len(xy_coordinates)}\
            num xz slices: {len(xz_coordinates)}\
            num yz slices: {len(yz_coordinates)}"
        )
        self.coordinates = xy_coordinates + xz_coordinates + yz_coordinates

        logging.info(f"total num of coordinates across 3 planes: {len(self.coordinates)}")

        self.split = split
        self.transform = transform

        self.coordinates *= multiplier

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        coordinates = self.coordinates[idx]
        z1, z2, y1, y2, x1, x2 = coordinates

        image_crop = self.image[z1:z2, y1:y2, x1:x2].copy().squeeze()
        if self.split != "test":
            label_crop = self.label[z1:z2, y1:y2, x1:x2].copy().squeeze()

            assert image_crop.shape == label_crop.shape

        height_pad_before = height_pad_after = width_pad_before = width_pad_after = 0
        if image_crop.shape[0] != self.crop_size:
            height_pad_size = self.crop_size - image_crop.shape[0]
            height_pad_before = height_pad_size // 2
            height_pad_after = height_pad_size - height_pad_before

        if image_crop.shape[1] != self.crop_size:
            width_pad_size = self.crop_size - image_crop.shape[1]
            width_pad_before = width_pad_size // 2
            width_pad_after = width_pad_size - width_pad_before

        image_crop = np.pad(
            image_crop,
            ((height_pad_before, height_pad_after), (width_pad_before, width_pad_after)),
            mode="constant",
            constant_values=0,
        )
        if self.split != "test":
            label_crop = np.pad(
                label_crop,
                ((height_pad_before, height_pad_after), (width_pad_before, width_pad_after)),
                mode="constant",
                constant_values=0,
            )

        if self.transform and (self.split != "test"):
            sample = self.transform(image=image_crop, mask=label_crop)
            image_crop = sample["image"]
            label_crop = sample["mask"].unsqueeze(0).to(dtype=torch.float16)
        elif self.transform:
            sample = self.transform(image=image_crop)
            image_crop = sample["image"]

        image_mean = torch.mean(image_crop.float())
        image_std = torch.std(image_crop.float())

        image_crop = (image_crop - image_mean) / (image_std + 1e-4)

        if self.split == "train":
            return image_crop, label_crop
        elif self.split == "val":
            return (
                image_crop,
                label_crop,
                torch.tensor([z1, z2, y1, y2, x1, x2]),
                torch.tensor([height_pad_before, height_pad_after, width_pad_before, width_pad_after]),
            )
        else:
            return (
                image_crop,
                torch.tensor([z1, z2, y1, y2, x1, x2]),
                torch.tensor([height_pad_before, height_pad_after, width_pad_before, width_pad_after]),
            )


class Dataset2DMultiPlanesSeveralKidneys(Dataset):
    def __init__(
        self,
        dataset_roots,
        pseudo_annotations_paths,
        crop_size,
        overlap_size,
        split,
        multiplier=1,
        transform=None,
    ):
        super().__init__()
        self.crop_size = crop_size
        self.images, self.labels, self.depths, self.heights, self.widths = [], [], [], [], []
        for dataset_root, pseudo_annotations_path in zip(dataset_roots, pseudo_annotations_paths):
            image = tifffile.memmap(f"{dataset_root}/full_image.tif", mode="r")
            if split != "test":
                if pseudo_annotations_path:
                    label = tifffile.memmap(pseudo_annotations_path, mode="r")
                else:
                    label = tifffile.memmap(f"{dataset_root}/full_label.tif", mode="r")

                assert image.shape == label.shape
                assert label.min() >= 0 and label.max() <= 1
                self.labels.append(label)

            assert image.min() >= 0 and image.max() <= 256

            self.images.append(image)

            logging.info(f"Full image shape: {image.shape}")
            depth, height, width = image.shape
            self.depths.append(depth)
            self.heights.append(height)
            self.widths.append(width)

        logging.info(f"Num of kidneys: {len(self.images)}")

        step_size = crop_size - overlap_size

        self.coordinates = []
        for kidney_index, (depth, height, width) in enumerate(zip(self.depths, self.heights, self.widths)):
            # calculate XY coordinates
            xy_coordinates = []
            for z in range(depth):
                for y in range(0, height - step_size, step_size):
                    for x in range(0, width - step_size, step_size):
                        crop_end_y = min(y + crop_size, height)
                        crop_end_x = min(x + crop_size, width)

                        xy_coordinates.append((kidney_index, z, z + 1, y, crop_end_y, x, crop_end_x))

            xz_coordinates = []
            for z in range(0, depth - step_size, step_size):
                for y in range(height):
                    for x in range(0, width - step_size, step_size):
                        crop_end_z = min(z + crop_size, depth)
                        crop_end_x = min(x + crop_size, width)

                        xz_coordinates.append((kidney_index, z, crop_end_z, y, y + 1, x, crop_end_x))

            # calculate YZ coordinates
            yz_coordinates = []
            for z in range(0, depth - step_size, step_size):
                for y in range(0, height - step_size, step_size):
                    for x in range(width):
                        crop_end_z = min(z + crop_size, depth)
                        crop_end_y = min(y + crop_size, height)

                        yz_coordinates.append((kidney_index, z, crop_end_z, y, crop_end_y, x, x + 1))

            logging.info(
                f"kidney_index: {kidney_index}\
                num xy slices: {len(xy_coordinates)}\
                num xz slices: {len(xz_coordinates)}\
                num yz slices: {len(yz_coordinates)}"
            )
            logging.info(
                f"total num of coordinates across 3 planes across {kidney_index}\
                kidney: {len(xy_coordinates + xz_coordinates + yz_coordinates)}"
            )

            self.coordinates += xy_coordinates + xz_coordinates + yz_coordinates

        logging.info(f"total num of coordinates across 3 planes across all kidneys: {len(self.coordinates)}")

        self.split = split
        self.multiplier = multiplier
        self.transform = transform

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        coordinates = self.coordinates[idx]
        kidney_index, z1, z2, y1, y2, x1, x2 = coordinates

        image_crop = self.images[kidney_index][z1:z2, y1:y2, x1:x2].copy().squeeze()
        if self.split != "test":
            label_crop = self.labels[kidney_index][z1:z2, y1:y2, x1:x2].copy().squeeze()
            assert image_crop.shape == label_crop.shape

        height_pad_before = height_pad_after = width_pad_before = width_pad_after = 0
        if image_crop.shape[0] != self.crop_size:
            height_pad_size = self.crop_size - image_crop.shape[0]
            height_pad_before = height_pad_size // 2
            height_pad_after = height_pad_size - height_pad_before

        if image_crop.shape[1] != self.crop_size:
            width_pad_size = self.crop_size - image_crop.shape[1]
            width_pad_before = width_pad_size // 2
            width_pad_after = width_pad_size - width_pad_before

        image_crop = np.pad(
            image_crop,
            ((height_pad_before, height_pad_after), (width_pad_before, width_pad_after)),
            mode="constant",
            constant_values=0,
        )
        if self.split != "test":
            label_crop = np.pad(
                label_crop,
                ((height_pad_before, height_pad_after), (width_pad_before, width_pad_after)),
                mode="constant",
                constant_values=0,
            )

        if self.transform and (self.split != "test"):
            sample = self.transform(image=image_crop, mask=label_crop)
            image_crop = sample["image"]
            label_crop = sample["mask"].unsqueeze(0)
        elif self.transform:
            sample = self.transform(image=image_crop)
            image_crop = sample["image"]

        image_mean = torch.mean(image_crop.float())
        image_std = torch.std(image_crop.float())

        image_crop = (image_crop - image_mean) / (image_std + 1e-4)

        if self.split == "train":
            return image_crop, label_crop
        elif self.split == "val":
            return (
                image_crop,
                label_crop,
                torch.tensor([z1, z2, y1, y2, x1, x2]),
                torch.tensor([height_pad_before, height_pad_after, width_pad_before, width_pad_after]),
            )
        else:
            return (
                image_crop,
                torch.tensor([z1, z2, y1, y2, x1, x2]),
                torch.tensor([height_pad_before, height_pad_after, width_pad_before, width_pad_after]),
            )


class Dataset2DMultiPlanesSeveralKidneysBoundaries(Dataset):
    def __init__(
        self,
        dataset_roots,
        pseudo_annotations_paths,
        boundaries_paths,
        crop_size,
        overlap_size,
        split,
        multiplier=1,
        transform=None,
    ):
        super().__init__()
        self.crop_size = crop_size
        self.images, self.labels, self.boundaries, self.depths, self.heights, self.widths = [], [], [], [], [], []
        for dataset_root, pseudo_annotations_path, boundaries_path in zip(
            dataset_roots, pseudo_annotations_paths, boundaries_paths
        ):
            image = tifffile.memmap(f"{dataset_root}/full_image.tif", mode="r")
            if pseudo_annotations_path:
                label = tifffile.memmap(pseudo_annotations_path, mode="r")
            else:
                label = tifffile.memmap(f"{dataset_root}/full_label.tif", mode="r")

            boundaries = tifffile.memmap(boundaries_path, mode="r")

            assert image.shape == label.shape == boundaries.shape
            assert (
                image.min() >= 0
                and image.max() <= 256
                and label.min() >= 0
                and label.max() <= 1
                and boundaries.min() >= 0
                and boundaries.max() <= 1
            )

            self.images.append(image)
            self.labels.append(label)
            self.boundaries.append(boundaries)

            logging.info(f"Full image shape: {image.shape}")
            depth, height, width = image.shape
            self.depths.append(depth)
            self.heights.append(height)
            self.widths.append(width)

        logging.info(f"Num of kidneys: {len(self.images)}")

        step_size = crop_size - overlap_size

        self.coordinates = []
        for kidney_index, (depth, height, width) in enumerate(zip(self.depths, self.heights, self.widths)):
            # calculate XY coordinates
            xy_coordinates = []
            for z in range(depth):
                for y in range(0, height - step_size, step_size):
                    for x in range(0, width - step_size, step_size):
                        crop_end_y = min(y + crop_size, height)
                        crop_end_x = min(x + crop_size, width)

                        xy_coordinates.append((kidney_index, z, z + 1, y, crop_end_y, x, crop_end_x))

            xz_coordinates = []
            for z in range(0, depth - step_size, step_size):
                for y in range(height):
                    for x in range(0, width - step_size, step_size):
                        crop_end_z = min(z + crop_size, depth)
                        crop_end_x = min(x + crop_size, width)

                        xz_coordinates.append((kidney_index, z, crop_end_z, y, y + 1, x, crop_end_x))

            # calculate YZ coordinates
            yz_coordinates = []
            for z in range(0, depth - step_size, step_size):
                for y in range(0, height - step_size, step_size):
                    for x in range(width):
                        crop_end_z = min(z + crop_size, depth)
                        crop_end_y = min(y + crop_size, height)

                        yz_coordinates.append((kidney_index, z, crop_end_z, y, crop_end_y, x, x + 1))

            logging.info(
                f"kidney_index: {kidney_index}\
                num xy slices: {len(xy_coordinates)}\
                num xz slices: {len(xz_coordinates)}\
                num yz slices: {len(yz_coordinates)}"
            )
            logging.info(
                f"total num of coordinates across 3 planes across {kidney_index}\
                kidney: {len(xy_coordinates + xz_coordinates + yz_coordinates)}"
            )

            self.coordinates += xy_coordinates + xz_coordinates + yz_coordinates

        logging.info(f"total num of coordinates across 3 planes across all kidneys: {len(self.coordinates)}")

        self.split = split
        self.multiplier = multiplier
        self.transform = transform

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        coordinates = self.coordinates[idx]
        kidney_index, z1, z2, y1, y2, x1, x2 = coordinates

        image_crop = self.images[kidney_index][z1:z2, y1:y2, x1:x2].copy().squeeze()
        label_crop = self.labels[kidney_index][z1:z2, y1:y2, x1:x2].copy().squeeze()
        boundaries_crop = self.boundaries[kidney_index][z1:z2, y1:y2, x1:x2].copy().squeeze().astype(np.float64)

        assert image_crop.shape == label_crop.shape == boundaries_crop.shape

        height_pad_before = height_pad_after = width_pad_before = width_pad_after = 0
        if image_crop.shape[0] != self.crop_size:
            height_pad_size = self.crop_size - image_crop.shape[0]
            height_pad_before = height_pad_size // 2
            height_pad_after = height_pad_size - height_pad_before

        if image_crop.shape[1] != self.crop_size:
            width_pad_size = self.crop_size - image_crop.shape[1]
            width_pad_before = width_pad_size // 2
            width_pad_after = width_pad_size - width_pad_before

        image_crop = np.pad(
            image_crop,
            ((height_pad_before, height_pad_after), (width_pad_before, width_pad_after)),
            mode="constant",
            constant_values=0,
        )
        label_crop = np.pad(
            label_crop,
            ((height_pad_before, height_pad_after), (width_pad_before, width_pad_after)),
            mode="constant",
            constant_values=0,
        )

        boundaries_crop = np.pad(
            boundaries_crop,
            ((height_pad_before, height_pad_after), (width_pad_before, width_pad_after)),
            mode="constant",
            constant_values=0,
        )

        if self.transform:
            sample = self.transform(image=image_crop, mask=label_crop, boundaries=boundaries_crop)
            image_crop = sample["image"]
            label_crop = sample["mask"].unsqueeze(0)
            boundaries_crop = sample["boundaries"].unsqueeze(0)

        image_mean = torch.mean(image_crop.float())
        image_std = torch.std(image_crop.float())

        image_crop = (image_crop - image_mean) / (image_std + 1e-4)

        if self.split != "train":
            return (
                image_crop,
                label_crop,
                boundaries_crop,
                torch.tensor([z1, z2, y1, y2, x1, x2]),
                torch.tensor([height_pad_before, height_pad_after, width_pad_before, width_pad_after]),
            )

        return image_crop, label_crop, boundaries_crop
