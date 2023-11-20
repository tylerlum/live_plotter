from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from typing import Union, List
import math
import sys

import seaborn as sns

from live_plotter.utils import (
    assert_equals,
    datetime_str,
    convert_to_list_str_fixed_len,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_IMAGE_SHAPE,
    preprocess_image_data_if_needed,
    validate_image_data,
    scale_image,
)

sns.set_theme()


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

        x_min, x_max = 0, image_data.shape[1]
        y_min, y_max = 0, image_data.shape[0]
        axes_image.set_extent((x_min, x_max, y_min, y_max))
        ax.set_xlim(left=x_min, right=x_max)
        ax.set_ylim(bottom=y_min, top=y_max)

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
