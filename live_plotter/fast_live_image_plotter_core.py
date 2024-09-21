from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from live_plotter.utils import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_SHAPE,
    DEFAULT_IMAGE_WIDTH,
    assert_equals,
    compute_n_rows_n_cols,
    preprocess_image_data_if_needed,
    scale_image,
    validate_image_data,
)

sns.set_theme()


class FastLiveImagePlotter:
    def __init__(
        self,
        n_plots: int = 1,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        titles: Optional[List[Optional[str]]] = None,
        xlabels: Optional[List[Optional[str]]] = None,
        ylabels: Optional[List[Optional[str]]] = None,
    ) -> None:
        """
        Create a FastLiveImagePlotter object consisting of n_plots image subplots arranged in a grid of shape n_rows x n_cols (or automatically computed if not given).

        Args:
            n_plots: int, number of image subplots
            n_rows: Optional[int], number of rows in the grid of subplots
                    If n_rows is None, then n_rows will be automatically computed
            n_cols: Optional[int], number of columns in the grid of subplots
                    If n_cols is None, then n_cols will be automatically computed
            titles: Optional[List[Optional[str]], where each element is the title for a subplot
                    If titles is None, then the default titles are used
                    If titles[i] is None, then the default title is used for subplot i
            xlabels: Optional[List[Optional[str]], where each element is the x label for a subplot
                     If xlabels is None, then the default x labels are used
                     If xlabels[i] is None, then the default x label is used for subplot i
            ylabels: Optional[List[Optional[str]], where each element is the y label for a subplot
                     If ylabels is None, then the default y labels are used
                     If ylabels[i] is None, then the default y label is used for subplot i
        """
        self.n_plots = n_plots

        # Infer n_rows and n_cols if not given
        self.n_rows, self.n_cols = compute_n_rows_n_cols(
            n_plots=n_plots, n_rows=n_rows, n_cols=n_cols
        )
        assert (
            self.n_plots <= self.n_rows * self.n_cols
        ), f"n_plots = {self.n_plots}, n_rows = {self.n_rows}, n_cols = {self.n_cols}"

        # Validate other inputs
        if titles is None:
            titles = [None for _ in range(self.n_plots)]
        assert_equals(len(titles), self.n_plots)

        if xlabels is None:
            xlabels = [None for _ in range(self.n_plots)]
        assert_equals(len(xlabels), self.n_plots)

        if ylabels is None:
            ylabels = [None for _ in range(self.n_plots)]
        assert_equals(len(ylabels), self.n_plots)

        self.titles = titles
        self.xlabels = xlabels
        self.ylabels = ylabels

        plt.show(block=False)

        self.fig = plt.figure()
        self.axes = []
        self.axes_images = []
        for i, (title, xlabel, ylabel) in enumerate(zip(titles, xlabels, ylabels)):
            ax_idx = i + 1
            ax = self.fig.add_subplot(self.n_rows, self.n_cols, ax_idx)

            ax.grid(False)
            if title is not None:
                ax.set_title(title)
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)

            axes_image = ax.imshow(np.zeros(DEFAULT_IMAGE_SHAPE))

            self.axes.append(ax)
            self.axes_images.append(axes_image)
        self.fig.tight_layout()
        self.fig.canvas.draw()

        # Replace plt.pause(0.001) to avoid focus stealing
        # https://github.com/tylerlum/live_plotter/issues/2
        plt.pause(0.001)

    def plot(
        self,
        image_data_list: List[np.ndarray],
    ) -> None:
        """
        Update the plot with new data.

        Args:
          image_data_list: List[np.ndarray], where each element is the image data to be plotted
                           image_data is expected to be 2D of shape (H, W) or 3D of shape (H, W, C), where H is the height, W is the width, and C is the number of channels (C = 3 RGB or 4 RGBA)
                           image_data is either of type float in [0, 1] or int in [0, 255]
        """
        n_plots = len(image_data_list)
        assert_equals(n_plots, self.n_plots)

        image_data_list = [
            preprocess_image_data_if_needed(image_data=image_data)
            for image_data in image_data_list
        ]

        for i, (image_data, axes_image, ax) in enumerate(
            zip(
                image_data_list,
                self.axes_images,
                self.axes,
            )
        ):
            validate_image_data(image_data=image_data)
            axes_image.set_data(image_data)

            x_min, x_max = 0, image_data.shape[1]
            y_min, y_max = 0, image_data.shape[0]
            axes_image.set_extent((x_min, x_max, y_min, y_max))
            ax.set_xlim(left=x_min, right=x_max)
            ax.set_ylim(bottom=y_min, top=y_max)

        self.fig.tight_layout()

        # Replace plt.pause(0.001) to avoid focus stealing
        # https://github.com/tylerlum/live_plotter/issues/2
        # plt.pause(0.001)
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.001)


def main() -> None:
    import time

    N = 25

    live_plotter = FastLiveImagePlotter(
        titles=["sin", "cos"], n_plots=2, n_rows=2, n_cols=1
    )
    x_data = []
    for i in range(N):
        x_data.append(i)
        image_data_1 = (
            np.sin(x_data)[None, ...]
            .repeat(DEFAULT_IMAGE_HEIGHT, 0)
            .repeat(DEFAULT_IMAGE_WIDTH // N, 1)
        )
        image_data_2 = (
            np.cos(x_data)[None, ...]
            .repeat(DEFAULT_IMAGE_HEIGHT, 0)
            .repeat(DEFAULT_IMAGE_WIDTH // N, 1)
        )
        live_plotter.plot(
            image_data_list=[
                scale_image(image_data_1, min_val=-1.0, max_val=1.0),
                scale_image(image_data_2),
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
    live_plotter = FastLiveImagePlotter(titles=plot_names, n_plots=len(plot_names))
    for i in range(N):
        y_data_dict["exp(-x/10)"].append(np.exp(-i / 10))
        y_data_dict["ln(x + 1)"].append(np.log(i + 1))
        y_data_dict["x^2"].append(np.power(i, 2))
        y_data_dict["4x^4"].append(4 * np.power(i, 4))
        y_data_dict["ln(2^x)"].append(np.log(np.power(2, i)))

        image_data_list = [
            scale_image(
                np.array(y_data_dict[plot_name])[None, ...]
                .repeat(DEFAULT_IMAGE_HEIGHT, 0)
                .repeat(DEFAULT_IMAGE_WIDTH // N, 1)
            )
            for plot_name in plot_names
        ]
        live_plotter.plot(
            image_data_list=image_data_list,
        )


if __name__ == "__main__":
    main()
