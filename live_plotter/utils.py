import math
from datetime import datetime
from typing import Optional, Tuple

import numpy as np


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


def datetime_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def compute_n_rows_n_cols(
    n_plots: int,
    n_rows: Optional[int],
    n_cols: Optional[int],
) -> Tuple[int, int]:
    assert n_plots > 0

    if n_rows is not None and n_cols is not None:
        assert n_rows * n_cols >= n_plots
        return n_rows, n_cols
    elif n_rows is None and n_cols is None:
        n_rows = math.ceil(math.sqrt(n_plots))
        n_cols = math.ceil(n_plots / n_rows)
        return n_rows, n_cols
    elif n_cols is None:
        assert n_rows is not None
        n_cols = math.ceil(n_plots / n_rows)
        return n_rows, n_cols
    elif n_rows is None:
        assert n_cols is not None
        n_rows = math.ceil(n_plots / n_cols)
        return n_rows, n_cols
    else:
        raise ValueError("This should not happen")


DEFAULT_IMAGE_HEIGHT = 100
DEFAULT_IMAGE_WIDTH = 100
DEFAULT_IMAGE_SHAPE = (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)


def preprocess_image_data_if_needed(image_data: np.ndarray) -> np.ndarray:
    assert len(image_data.shape) in [2, 3], f"image_data.shape = {image_data.shape}"

    NUM_RGB = 3
    NUM_RGBA = 4
    if len(image_data.shape) == 2:
        image_data = image_data[..., None].repeat(NUM_RGB, axis=-1)

    channels = image_data.shape[-1]
    assert channels in [NUM_RGB, NUM_RGBA], f"channels = {channels}"

    if not is_valid_image_data_content(image_data=image_data):
        print(
            f"WARNING: image_data range in [{image_data.min()}, {image_data.max()}], rescaling"
        )
        image_data = scale_image(image_data=image_data)

    return image_data


def is_valid_image_data_content(image_data: np.ndarray) -> bool:
    # If float, check that values are in [0, 1]
    if np.issubdtype(image_data.dtype, np.floating):
        return image_data.min() >= 0 and image_data.max() <= 1

    # If integer, check that values are in [0, 255]
    elif np.issubdtype(image_data.dtype, np.integer):
        return image_data.min() >= 0 and image_data.max() <= 255

    return False


def validate_image_data_content(image_data: np.ndarray) -> None:
    # If float, check that values are in [0, 1]
    if np.issubdtype(image_data.dtype, np.floating):
        assert (
            image_data.min() >= 0 and image_data.max() <= 1
        ), f"dtype = {image_data.dtype}, image_data range in [{image_data.min()}, {image_data.max()}], should be in [0, 1]"
        return

    # If integer, check that values are in [0, 255]
    elif np.issubdtype(image_data.dtype, np.integer):
        assert (
            image_data.min() >= 0 and image_data.max() <= 255
        ), f"dtype = {image_data.dtype}, image_data range in [{image_data.min()}, {image_data.max()}], should be in [0, 255]"
        return

    raise ValueError(f"Invalid image_data.dtype = {image_data.dtype}")


def validate_image_data(image_data: np.ndarray) -> None:
    assert_equals(len(image_data.shape), 3)
    channel_dim = image_data.shape[2]

    NUM_RGB = 3
    NUM_RGBA = 4
    assert channel_dim in [NUM_RGB, NUM_RGBA], f"channel_dim = {channel_dim}"

    validate_image_data_content(image_data=image_data)


def scale_image(
    image_data: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    eps: float = 1e-5,
    validate: bool = True,
) -> np.ndarray:
    if min_val is None:
        min_val = image_data.min()
    if max_val is None:
        max_val = image_data.max()

    assert min_val is not None and max_val is not None

    assert min_val <= max_val, f"min_val = {min_val}, max_val = {max_val}"

    if np.issubdtype(image_data.dtype, np.floating):
        output_image_data = (image_data - min_val) / (max_val - min_val + eps)

    elif np.issubdtype(image_data.dtype, np.integer):
        output_image_data = (
            (image_data - min_val) / (max_val - min_val + eps) * 255
        ).astype(np.uint8)

    else:
        raise ValueError(f"Invalid image_data.dtype = {image_data.dtype}")

    if validate:
        original_image_min, original_image_max = image_data.min(), image_data.max()
        assert (
            min_val <= original_image_min
        ), f"min_val = {min_val}, original_image_min = {original_image_min}"
        assert (
            max_val >= original_image_max
        ), f"max_val = {max_val}, original_image_max = {original_image_max}"
        validate_image_data_content(image_data=output_image_data)

    return output_image_data
