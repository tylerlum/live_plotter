from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from typing import Union, List, Optional, Tuple
import math
import sys

import seaborn as sns

from live_plotter.utils import (
    assert_equals,
    datetime_str,
    convert_to_list_str_fixed_len,
)

sns.set_theme()

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


def fast_plot_images_helper(
    fig: Figure,
    image_data_list: List[np.ndarray],
    axes: List[Axes],
    axes_images: List[AxesImage],
) -> None:
    """Plot data on existing figure onto existing axes and axes_images"""
    # Shape checks
    n_plots = len(image_data_list)
    max_n_plots = len(axes)
    assert_equals(len(axes_images), max_n_plots)
    assert n_plots <= max_n_plots, f"{n_plots} > {max_n_plots}"

    for i in range(n_plots):
        axes_image, ax = axes_images[i], axes[i]
        image_data = image_data_list[i]

        validate_image_data(image_data=image_data)
        axes_image.set_data(image_data)

    fig.tight_layout()
    plt.pause(0.001)


class FastLiveImagePlotter:
    def __init__(
        self,
        title: str = "",
        save_to_file_on_close: bool = False,
        save_to_file_on_exception: bool = False,
    ) -> None:
        self.title = title
        self.save_to_file_on_close = save_to_file_on_close
        self.save_to_file_on_exception = save_to_file_on_exception

        self.n_rows = 1
        self.n_cols = 1
        self.n_plots = self.n_rows * self.n_cols
        plt.show(block=False)

        ax_idx = 1
        self.fig = plt.figure()
        ax = self.fig.add_subplot(self.n_rows, self.n_cols, ax_idx)
        ax.set_title(title)
        ax.grid(False)
        self.axes = [ax]
        self.axes_images = [ax.imshow(np.zeros(DEFAULT_IMAGE_SHAPE))]

        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.001)

        if self.save_to_file_on_exception:
            self._setup_exception_hook()

    def plot(
        self,
        image_data: np.ndarray,
    ) -> None:
        image_data = preprocess_image_data_if_needed(image_data=image_data)
        fast_plot_images_helper(
            fig=self.fig,
            image_data_list=[image_data],
            axes=self.axes,
            axes_images=self.axes_images,
        )

    def _save_to_file(self) -> None:
        filename = (
            f"{datetime_str()}_{self.title}.png"
            if len(self.title) > 0
            else f"{datetime_str()}.png"
        )
        print(f"Saving to {filename}")
        self.fig.savefig(filename)
        print(f"Saved to {filename}")

    def __del__(self) -> None:
        if self.save_to_file_on_close:
            self._save_to_file()

    def _setup_exception_hook(self) -> None:
        original_excepthook = sys.excepthook

        def exception_hook(exctype, value, traceback):
            print(f"Exception hook called ({self.__class__.__name__})")
            self._save_to_file()
            original_excepthook(exctype, value, traceback)

        sys.excepthook = exception_hook


class FastLiveImagePlotterGrid:
    def __init__(
        self,
        title: Union[str, List[str]] = "",
        n_rows: int = 1,
        n_cols: int = 1,
        save_to_file_on_close: bool = False,
        save_to_file_on_exception: bool = False,
    ) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.save_to_file_on_close = save_to_file_on_close
        self.save_to_file_on_exception = save_to_file_on_exception
        self.n_plots = n_rows * n_cols

        self.titles = convert_to_list_str_fixed_len(
            str_or_list_str=title, fixed_length=self.n_plots
        )
        assert len(self.titles) == self.n_plots

        plt.show(block=False)

        self.fig = plt.figure()
        self.axes = []
        self.axes_images = []
        for i, _title in enumerate(self.titles):
            ax_idx = i + 1
            ax = self.fig.add_subplot(n_rows, n_cols, ax_idx)
            adjusted_title = (
                " ".join([_title, f"(Plot {i})"]) if self.n_plots > 1 else _title
            )
            ax.set_title(adjusted_title)
            ax.grid(False)
            axes_image = ax.imshow(np.zeros(DEFAULT_IMAGE_SHAPE))
            self.axes.append(ax)
            self.axes_images.append(axes_image)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.001)

        if self.save_to_file_on_exception:
            self._setup_exception_hook()

    @classmethod
    def from_desired_n_plots(
        cls,
        title: Union[str, List[str]] = "",
        save_to_file_on_close: bool = False,
        save_to_file_on_exception: bool = False,
        desired_n_plots: int = 1,
    ) -> FastLiveImagePlotterGrid:
        n_rows = math.ceil(math.sqrt(desired_n_plots))
        n_cols = math.ceil(desired_n_plots / n_rows)

        return cls(
            title=title,
            n_rows=n_rows,
            n_cols=n_cols,
            save_to_file_on_close=save_to_file_on_close,
            save_to_file_on_exception=save_to_file_on_exception,
        )

    def plot_grid(
        self,
        image_data_list: List[np.ndarray],
    ) -> None:
        image_data_list = [
            preprocess_image_data_if_needed(image_data=image_data)
            for image_data in image_data_list
        ]
        fast_plot_images_helper(
            fig=self.fig,
            image_data_list=image_data_list,
            axes=self.axes,
            axes_images=self.axes_images,
        )

    def _save_to_file(self) -> None:
        filename = (
            f"{datetime_str()}_{self.titles}.png"
            if len("".join(self.titles)) > 0
            else f"{datetime_str()}.png"
        )
        print(f"Saving to {filename}")
        self.fig.savefig(filename)
        print(f"Saved to {filename}")

    def __del__(self) -> None:
        if self.save_to_file_on_close:
            self._save_to_file()

    def _setup_exception_hook(self) -> None:
        original_excepthook = sys.excepthook

        def exception_hook(exctype, value, traceback):
            print(f"Exception hook called ({self.__class__.__name__})")
            self._save_to_file()
            original_excepthook(exctype, value, traceback)

        sys.excepthook = exception_hook


def main() -> None:
    import time

    live_plotter = FastLiveImagePlotter(title="sin")

    x_data = []
    for i in range(25):
        x_data.append(0.5 * i)
        image_data = (
            np.sin(x_data)[None, ...]
            .repeat(DEFAULT_IMAGE_HEIGHT, 0)
            .repeat(DEFAULT_IMAGE_WIDTH, 1)
        )
        live_plotter.plot(image_data=scale_image(image_data, min_val=-1.0, max_val=1.0))

    time.sleep(2)

    live_plotter_grid = FastLiveImagePlotterGrid(
        title=["sin", "cos"], n_rows=2, n_cols=1
    )
    x_data = []
    for i in range(25):
        x_data.append(i)
        image_data_1 = (
            np.sin(x_data)[None, ...]
            .repeat(DEFAULT_IMAGE_HEIGHT, 0)
            .repeat(DEFAULT_IMAGE_WIDTH, 1)
        )
        image_data_2 = (
            np.cos(x_data)[None, ...]
            .repeat(DEFAULT_IMAGE_HEIGHT, 0)
            .repeat(DEFAULT_IMAGE_WIDTH, 1)
        )
        live_plotter_grid.plot_grid(
            image_data_list=[
                scale_image(image_data_1, min_val=-1.0, max_val=1.0),
                image_data_2,
            ],
        )

    time.sleep(2)

    y_data_dict = {
        "exp(-x/10)": [],
        "ln(x + 1)": [],
        "x^2": [],
        "4x^4": [],
        "ln(2^x)": [],
    }
    plot_names = list(y_data_dict.keys())
    live_plotter_grid = FastLiveImagePlotterGrid.from_desired_n_plots(
        title=plot_names, desired_n_plots=len(plot_names)
    )
    for i in range(25):
        y_data_dict["exp(-x/10)"].append(np.exp(-i / 10))
        y_data_dict["ln(x + 1)"].append(np.log(i + 1))
        y_data_dict["x^2"].append(np.power(i, 2))
        y_data_dict["4x^4"].append(4 * np.power(i, 4))
        y_data_dict["ln(2^x)"].append(np.log(np.power(2, i)))

        image_data_list = [
            scale_image(
                np.array(y_data_dict[plot_name])[None, ...]
                .repeat(DEFAULT_IMAGE_HEIGHT, 0)
                .repeat(DEFAULT_IMAGE_WIDTH, 1)
            )
            for plot_name in plot_names
        ]
        live_plotter_grid.plot_grid(
            image_data_list=image_data_list,
        )


if __name__ == "__main__":
    main()
