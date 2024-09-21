from __future__ import annotations

import math
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from live_plotter.utils import (
    assert_equals,
    compute_n_rows_n_cols,
)

sns.set_theme()


def compute_axes_min_max(axes_min: float, axes_max: float) -> Tuple[float, float]:
    if not math.isclose(axes_min, axes_max):
        return axes_min, axes_max

    if math.isclose(axes_min, 0.0) or math.isclose(axes_max, 0.0):
        return -0.05, 0.05

    return 0.95 * axes_min, 1.05 * axes_max


class FastLivePlotter:
    def __init__(
        self,
        n_plots: int = 1,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        titles: Optional[List[Optional[str]]] = None,
        xlabels: Optional[List[Optional[str]]] = None,
        ylabels: Optional[List[Optional[str]]] = None,
        xlims: Optional[List[Optional[Tuple[float, float]]]] = None,
        ylims: Optional[List[Optional[Tuple[float, float]]]] = None,
        legends: Optional[List[Optional[List[str]]]] = None,
    ) -> None:
        """
        Create a FastLivePlotter object consisting of n_plots subplots arranged in a grid of shape n_rows x n_cols (or automatically computed if not given).

        Args:
            n_plots: int, number of subplots
            n_rows: Optional[int], number of rows in the grid of subplots
                    If n_rows is None, then n_rows will be automatically computed
            n_cols: Optional[int], number of columns in the grid of subplots
                    If n_cols is None, then n_cols will be automatically computed
            titles: Optional[List[Optional[str]]], where each element is the title for a subplot
                    If titles is None, then the default titles are used
                    If titles[i] is None, then the default title is used for subplot i
            xlabels: Optional[List[Optional[str]]], where each element is the x label for a subplot
                     If xlabels is None, then the default x labels are used
                     If xlabels[i] is None, then the default x label is used for subplot i
            ylabels: Optional[List[Optional[str]]], where each element is the y label for a subplot
                     If ylabels is None, then the default y labels are used
                     If ylabels[i] is None, then the default y label is used for subplot i
            xlims: Optional[List[Optional[Tuple[float, float]]], where each element is the x limits for a subplot
                   If xlims is None, then the default x limits are used
                   If xlims[i] is None, then the default x limits are used for subplot i
            ylims: Optional[List[Optional[Tuple[float, float]]], where each element is the y limits for a subplot
                   If ylims is None, then the default y limits are used
                   If ylims[i] is None, then the default y limits are used for subplot i
            legends: Optional[List[Optional[List[str]]], where each element is the legend for a subplot
                     If legends is None, then the default legends are used
                     If legends[i] is None, then the default legend is used for subplot i
                     If legends[i] is not None, it must be of length N, where N is the number of plots in subplot i
                     Requires y_data to be 2D
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

        if xlims is None:
            xlims = [None for _ in range(self.n_plots)]
        assert_equals(len(xlims), self.n_plots)

        if ylims is None:
            ylims = [None for _ in range(self.n_plots)]
        assert_equals(len(ylims), self.n_plots)

        if legends is None:
            legends = [None for _ in range(self.n_plots)]
        assert_equals(len(legends), self.n_plots)

        self.titles = titles
        self.xlabels = xlabels
        self.ylabels = ylabels
        self.xlims = xlims
        self.ylims = ylims
        self.legends = legends

        plt.show(block=False)

        self.fig = plt.figure()
        self.axes = []
        self.lines_list = []

        for i, (title, xlabel, ylabel, xlim, ylim, legend) in enumerate(
            zip(
                self.titles,
                self.xlabels,
                self.ylabels,
                self.xlims,
                self.ylims,
                self.legends,
            )
        ):
            ax_idx = i + 1
            ax = self.fig.add_subplot(self.n_rows, self.n_cols, ax_idx)

            if title is not None:
                ax.set_title(title)
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            if xlim is not None:
                left, right = xlim
                ax.set_xlim(left=left, right=right)
            if ylim is not None:
                bottom, top = ylim
                ax.set_ylim(bottom=bottom, top=top)

            if legend is not None:
                N = len(legend)
                lines = ax.plot(np.zeros((0, N)), label=legend)
                ax.legend()
            else:
                lines = ax.plot([])
            self.axes.append(ax)
            self.lines_list.append(lines)
        self.fig.tight_layout()
        self.fig.canvas.draw()

        plt.pause(0.001)

    def plot(
        self,
        y_data_list: List[np.ndarray],
        x_data_list: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        """
        Update the plot with new data.

        Args:
          y_data_list: List[np.ndarray], where each element is the y_data for subplot i
                       y_data is expected to be 1D of shape (D,) or 2D of shape (D, N), where D is the data dimension for 1 plot and N is the number of plots in this subplot
                       If legends was provided in the constructor, then y_data must be 2D
          x_data_list: Optional[List[Optional[np.ndarray]]], where each element is the x_data for a subplot
                       If x_data_list is None, then x_data is assumed to be default 0, 1, 2, ..., D-1 for all subplots
                       If x_data_list[i] is None, then x_data is assumed to be default 0, 1, 2, ..., D-1 for subplot i
        """
        n_plots = len(y_data_list)
        assert_equals(n_plots, self.n_plots)

        # Validate y_data
        for y_data in y_data_list:
            assert y_data.ndim in [1, 2], f"y_data.ndim = {y_data.ndim}"

        # Validate other inputs
        if x_data_list is None:
            x_data_list = [None for _ in range(n_plots)]
        assert_equals(len(x_data_list), n_plots)

        # Shape checks
        n_plots = len(x_data_list)
        assert_equals(len(y_data_list), n_plots)

        for i, (x_data, y_data, xlim, ylim, lines, ax) in enumerate(
            zip(
                x_data_list,
                y_data_list,
                self.xlims,
                self.ylims,
                self.lines_list,
                self.axes,
            )
        ):
            if x_data is None:
                if y_data.ndim == 2:
                    D, N = y_data.shape
                    x_data = np.arange(D).reshape(-1, 1).repeat(N, axis=1)
                else:
                    D = y_data.shape[0]
                    x_data = np.arange(D)
            assert_equals(x_data.shape, y_data.shape)

            if y_data.ndim == 2:
                D, N = y_data.shape
                assert_equals(len(lines), N)
                for j, line in enumerate(lines):
                    line.set_data(x_data[:, j], y_data[:, j])
            else:
                assert_equals(len(lines), 1)
                lines[0].set_data(x_data, y_data)

            # Keep the same xlim and ylim if provided
            if xlim is not None:
                left, right = xlim
                ax.set_xlim(left=left, right=right)
            else:
                # Handle case when min == max
                x_min, x_max = compute_axes_min_max(
                    axes_min=np.min(x_data), axes_max=np.max(x_data)
                )
                ax.set_xlim(left=x_min, right=x_max)

            if ylim is not None:
                bottom, top = ylim
                ax.set_ylim(bottom=bottom, top=top)
            else:
                # Handle case when min == max
                y_min, y_max = compute_axes_min_max(
                    axes_min=np.min(y_data), axes_max=np.max(y_data)
                )
                ax.set_ylim(bottom=y_min, top=y_max)

        self.fig.tight_layout()

        # Replace plt.pause(0.001) to avoid focus stealing
        # https://github.com/tylerlum/live_plotter/issues/2
        # plt.pause(0.001)
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.001)


def main() -> None:
    import time

    live_plotter = FastLivePlotter(titles=["sin", "cos"], n_plots=2, n_rows=2, n_cols=1)
    x_data = []
    for i in range(25):
        x_data.append(i)
        live_plotter.plot(
            y_data_list=[np.sin(x_data), np.cos(x_data)],
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
    live_plotter = FastLivePlotter(titles=plot_names, n_plots=len(plot_names))
    for i in range(25):
        y_data_dict["exp(-x/10)"].append(np.exp(-i / 10))
        y_data_dict["ln(x + 1)"].append(np.log(i + 1))
        y_data_dict["x^2"].append(np.power(i, 2))
        y_data_dict["4x^4"].append(4 * np.power(i, 4))
        y_data_dict["ln(2^x)"].append(np.log(np.power(2, i)))

        live_plotter.plot(
            y_data_list=[np.array(y_data_dict[plot_name]) for plot_name in plot_names],
        )

    new_x_data = []
    live_plotter = FastLivePlotter(
        n_plots=1,
        titles=["sin and cos"],
        xlabels=["x"],
        ylabels=["y"],
        ylims=[(-2, 2)],
        legends=[["sin", "cos"]],
    )
    for i in range(25):
        new_x_data.append(i)
        y_data = np.stack([np.sin(new_x_data), np.cos(new_x_data)], axis=1)
        live_plotter.plot(
            y_data_list=[y_data],
        )


if __name__ == "__main__":
    main()
