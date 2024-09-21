from __future__ import annotations

import math
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from live_plotter.utils import (
    assert_equals,
)

sns.set_theme()


def compute_axes_min_max(axes_min: float, axes_max: float) -> Tuple[float, float]:
    if not math.isclose(axes_min, axes_max):
        return axes_min, axes_max

    if math.isclose(axes_min, 0.0) or math.isclose(axes_max, 0.0):
        return -0.05, 0.05

    return 0.95 * axes_min, 1.05 * axes_max


def fast_plot_helper(
    fig: Figure,
    x_data_list: List[np.ndarray],
    y_data_list: List[np.ndarray],
    axes: List[Axes],
    lines: List[Line2D],
    xlims: Optional[List[Tuple[float, float]]] = None,
    ylims: Optional[List[Tuple[float, float]]] = None,
) -> None:
    """Plot data on existing figure onto existing axes and lines"""
    # Shape checks
    n_plots = len(x_data_list)
    assert_equals(len(y_data_list), n_plots)

    max_n_plots = len(axes)
    assert_equals(len(lines), max_n_plots)

    assert n_plots <= max_n_plots, f"{n_plots} > {max_n_plots}"

    for i in range(n_plots):
        line, ax = lines[i], axes[i]
        x_data, y_data = x_data_list[i], y_data_list[i]

        assert_equals(x_data.shape, y_data.shape)
        assert_equals(len(x_data.shape), 1)

        line.set_data(x_data, y_data)

        if xlims is not None and i < len(xlims):
            left, right = xlims[i]
            ax.set_xlim(left=left, right=right)
        else:
            # Handle case when min == max
            x_min, x_max = compute_axes_min_max(
                axes_min=np.min(x_data), axes_max=np.max(x_data)
            )
            ax.set_xlim(left=x_min, right=x_max)

        if ylims is not None and i < len(ylims):
            bottom, top = ylims[i]
            ax.set_ylim(bottom=bottom, top=top)
        else:
            # Handle case when min == max
            y_min, y_max = compute_axes_min_max(
                axes_min=np.min(y_data), axes_max=np.max(y_data)
            )
            ax.set_ylim(bottom=y_min, top=y_max)

    fig.tight_layout()

    # Replace plt.pause(0.001) to avoid focus stealing
    # https://github.com/tylerlum/live_plotter/issues/2
    # plt.pause(0.001)
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.001)


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
        self.n_plots = n_plots
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.titles = titles
        self.xlabels = xlabels
        self.ylabels = ylabels
        self.xlims = xlims
        self.ylims = ylims
        self.legends = legends

        # TODO: replace with comute_n_rows_n_cols
        # Infer n_rows and n_cols if not given
        if self.n_rows is None and self.n_cols is None:
            self.n_rows = math.ceil(math.sqrt(n_plots))
            self.n_cols = math.ceil(n_plots / self.n_rows)
        elif self.n_cols is None:
            assert self.n_rows is not None
            self.n_cols = math.ceil(n_plots / self.n_rows)
        elif self.n_rows is None:
            self.n_rows = math.ceil(n_plots / self.n_cols)

        self.total_n_plots = self.n_rows * self.n_cols

        if self.titles is None:
            self.titles = [None for _ in range(self.n_plots)]
        assert_equals(len(self.titles), self.n_plots)

        if self.xlabels is None:
            self.xlabels = [None for _ in range(self.n_plots)]
        assert_equals(len(self.xlabels), self.n_plots)

        if self.ylabels is None:
            self.ylabels = [None for _ in range(self.n_plots)]
        assert_equals(len(self.ylabels), self.n_plots)

        if self.xlims is None:
            self.xlims = [None for _ in range(self.n_plots)]
        assert_equals(len(self.xlims), self.n_plots)

        if self.ylims is None:
            self.ylims = [None for _ in range(self.n_plots)]
        assert_equals(len(self.ylims), self.n_plots)

        if self.legends is None:
            self.legends = [None for _ in range(self.n_plots)]
        assert_equals(len(self.legends), self.n_plots)

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
        n_plots = len(y_data_list)

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
