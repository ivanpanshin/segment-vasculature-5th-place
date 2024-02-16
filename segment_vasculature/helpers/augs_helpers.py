import albumentations as A
import albumentations.pytorch as AT
import cv2
import numpy as np


class ToInt(object):
    def __call__(self, **kwargs):
        kwargs["mask"] = np.round(kwargs["mask"] * 255).astype(np.uint8)
        return kwargs


class ToFloat(object):
    def __call__(self, **kwargs):
        kwargs["mask"] = np.clip(kwargs["mask"], 0, 255).astype(np.float32) / 255.0
        return kwargs


class ToIntBoundaries(object):
    def __call__(self, **kwargs):
        kwargs["mask"] = np.round(kwargs["mask"] * 255).astype(np.uint8)
        return kwargs

    def add_targets(self, additional_targets):
        self._additional_targets = additional_targets


class ToFloatBoundaries(object):
    def __call__(self, **kwargs):
        kwargs["mask"] = np.clip(kwargs["mask"], 0, 255).astype(np.float32) / 255.0
        return kwargs

    def add_targets(self, additional_targets):
        self._additional_targets = additional_targets


class ToInt3D(object):
    def __call__(self, **kwargs):
        kwargs["mask"] = np.round(kwargs["mask"] * 255).astype(np.uint8)

        for index in range(1, 64):
            kwargs[f"mask_{index}"] = np.round(kwargs[f"mask_{index}"] * 255).astype(np.uint8)

        return kwargs

    def add_targets(self, additional_targets):
        self._additional_targets = additional_targets


class ToFloat3D(object):
    def __call__(self, **kwargs):
        kwargs["mask"] = np.clip(kwargs["mask"], 0, 255).astype(np.float32) / 255.0

        for index in range(1, 64):
            kwargs[f"mask_{index}"] = np.clip(kwargs[f"mask_{index}"], 0, 255).astype(np.float32) / 255.0

        return kwargs

    def add_targets(self, additional_targets):
        self._additional_targets = additional_targets


def rsna_2023():
    # Constructing the additional targets for {crop_size} images and masks

    return A.Compose(
        [
            A.ShiftScaleRotate(
                p=0.7,
                shift_limit_x=(-0.1, 0.1),
                shift_limit_y=(-0.1, 0.1),
                scale_limit=(-0.25, 0.25),
                rotate_limit=(-25, 25),
                border_mode=cv2.BORDER_CONSTANT,
                # rotate_method="largest_box",
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25),
                contrast_limit=(-0.25, 0.25),
                p=0.5,
            ),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.OneOf(
                [
                    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, distort_limit=0.1),
                    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
                ],
                p=0.2,
            ),
            AT.ToTensorV2(),
        ],
    )


def rsna_2023_float():
    # Constructing the additional targets for {crop_size} images and masks

    return A.Compose(
        [
            ToInt(),
            A.ShiftScaleRotate(
                p=0.7,
                shift_limit_x=(-0.1, 0.1),
                shift_limit_y=(-0.1, 0.1),
                scale_limit=(-0.25, 0.25),
                rotate_limit=(-25, 25),
                border_mode=cv2.BORDER_CONSTANT,
                # rotate_method="largest_box",
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25),
                contrast_limit=(-0.25, 0.25),
                p=0.5,
            ),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.OneOf(
                [
                    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, distort_limit=0.1),
                    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
                ],
                p=0.2,
            ),
            ToFloat(),
            AT.ToTensorV2(),
        ],
    )


def rsna_2023_float_boundaries():
    # Constructing the additional targets for {crop_size} images and masks

    additional_targets = {"boundaries": "mask"}

    return A.Compose(
        [
            ToIntBoundaries(),
            A.ShiftScaleRotate(
                p=0.7,
                shift_limit_x=(-0.1, 0.1),
                shift_limit_y=(-0.1, 0.1),
                scale_limit=(-0.25, 0.25),
                rotate_limit=(-25, 25),
                border_mode=cv2.BORDER_CONSTANT,
                # rotate_method="largest_box",
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25),
                contrast_limit=(-0.25, 0.25),
                p=0.5,
            ),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.OneOf(
                [
                    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, distort_limit=0.1),
                    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
                ],
                p=0.2,
            ),
            ToFloatBoundaries(),
            AT.ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def to_tensor():
    # Constructing the additional targets for {crop_size} images and masks

    return A.Compose(
        [
            AT.ToTensorV2(),
        ],
    )


def rsna_2023_3d(temporal_size=64):
    # Constructing the additional targets for {crop_size} images and masks
    additional_targets = {
        **{f"image_{i}": "image" for i in range(1, temporal_size)},
        **{f"mask_{i}": "mask" for i in range(1, temporal_size)},
    }

    return A.Compose(
        [
            A.ShiftScaleRotate(
                p=0.7,
                shift_limit_x=(-0.1, 0.1),
                shift_limit_y=(-0.1, 0.1),
                scale_limit=(-0.25, 0.25),
                rotate_limit=(-25, 25),
                border_mode=cv2.BORDER_CONSTANT,
                # rotate_method="largest_box",
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25),
                contrast_limit=(-0.25, 0.25),
                p=0.5,
            ),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.OneOf(
                [
                    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, distort_limit=0.1),
                    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
                ],
                p=0.2,
            ),
            AT.ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def rsna_2023_3d_float(temporal_size=64):
    # Constructing the additional targets for {crop_size} images and masks
    additional_targets = {
        **{f"image_{i}": "image" for i in range(1, temporal_size)},
        **{f"mask_{i}": "mask" for i in range(1, temporal_size)},
    }

    return A.Compose(
        [
            ToInt3D(),
            A.ShiftScaleRotate(
                p=0.7,
                shift_limit_x=(-0.1, 0.1),
                shift_limit_y=(-0.1, 0.1),
                scale_limit=(-0.25, 0.25),
                rotate_limit=(-25, 25),
                border_mode=cv2.BORDER_CONSTANT,
                # rotate_method="largest_box",
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25),
                contrast_limit=(-0.25, 0.25),
                p=0.5,
            ),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.OneOf(
                [
                    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, distort_limit=0.1),
                    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
                ],
                p=0.2,
            ),
            ToFloat3D(),
            AT.ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def to_tensor_3d(temporal_size=96):
    # Constructing the additional targets for {crop_size} images and masks
    additional_targets = {
        **{f"image_{i}": "image" for i in range(1, temporal_size)},
        **{f"mask_{i}": "mask" for i in range(1, temporal_size)},
    }

    return A.Compose(
        [
            AT.ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )
