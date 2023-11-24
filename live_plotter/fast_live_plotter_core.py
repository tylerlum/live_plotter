from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
import numpy as np
from typing import List, Optional, Tuple
import math
import sys

import seaborn as sns

from live_plotter.utils import (
    assert_equals,
    datetime_str,
    convert_to_list_str_fixed_len,
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

        # Handle case when min == max
        x_min, x_max = compute_axes_min_max(
            axes_min=np.min(x_data), axes_max=np.max(x_data)
        )
        y_min, y_max = compute_axes_min_max(
            axes_min=np.min(y_data), axes_max=np.max(y_data)
        )
        ax.set_xlim(left=x_min, right=x_max)
        ax.set_ylim(bottom=y_min, top=y_max)

    fig.tight_layout()
    plt.pause(0.001)


class FastLivePlotter:
    def __init__(
        self,
        titles: Optional[List[str]] = None,
        xlabels: Optional[List[str]] = None,
        ylabels: Optional[List[str]] = None,
        xlims: Optional[List[Tuple[float, float]]] = None,
        ylims: Optional[List[Tuple[float, float]]] = None,
        n_rows: int = 1,
        n_cols: int = 1,
        save_to_file_on_close: bool = False,
        save_to_file_on_exception: bool = False,
    ) -> None:
        self.xlims = xlims
        self.ylims = ylims
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.save_to_file_on_close = save_to_file_on_close
        self.save_to_file_on_exception = save_to_file_on_exception
        self.n_plots = n_rows * n_cols

        self.titles = convert_to_list_str_fixed_len(
            list_str=titles, fixed_length=self.n_plots
        )
        self.xlabels = convert_to_list_str_fixed_len(
            list_str=xlabels, fixed_length=self.n_plots
        )
        self.ylabels = convert_to_list_str_fixed_len(
            list_str=ylabels, fixed_length=self.n_plots
        )
        assert (
            len(self.titles) == len(self.xlabels) == len(self.ylabels) == self.n_plots
        )

        plt.show(block=False)

        self.fig = plt.figure()
        self.axes = []
        self.lines = []
        for i, (_title, _xlabel, _ylabel) in enumerate(
            zip(self.titles, self.xlabels, self.ylabels)
        ):
            ax_idx = i + 1
            ax = self.fig.add_subplot(n_rows, n_cols, ax_idx)

            PLOT_MODIFIED_TITLE = False
            if PLOT_MODIFIED_TITLE:
                ax.set_title(
                    " ".join([_title, f"(Plot {i})"]) if self.n_plots > 1 else _title
                )
            else:
                ax.set_title(_title)

            ax.set_xlabel(_xlabel)
            ax.set_ylabel(_ylabel)

            if self.xlims is not None and i < len(self.xlims):
                left, right = self.xlims[i]
                ax.set_xlim(left=left, right=right)
            if self.ylims is not None and i < len(self.ylims):
                bottom, top = self.ylims[i]
                ax.set_ylim(bottom=bottom, top=top)

            line = ax.plot([], [])[0]
            self.axes.append(ax)
            self.lines.append(line)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.001)

        if self.save_to_file_on_exception:
            self._setup_exception_hook()

    @classmethod
    def from_desired_n_plots(
        cls,
        desired_n_plots: int,
        titles: Optional[List[str]] = None,
        xlabels: Optional[List[str]] = None,
        ylabels: Optional[List[str]] = None,
        xlims: Optional[List[Tuple[float, float]]] = None,
        ylims: Optional[List[Tuple[float, float]]] = None,
        save_to_file_on_close: bool = False,
        save_to_file_on_exception: bool = False,
    ) -> FastLivePlotter:
        n_rows = math.ceil(math.sqrt(desired_n_plots))
        n_cols = math.ceil(desired_n_plots / n_rows)

        return cls(
            titles=titles,
            xlabels=xlabels,
            ylabels=ylabels,
            xlims=xlims,
            ylims=ylims,
            n_rows=n_rows,
            n_cols=n_cols,
            save_to_file_on_close=save_to_file_on_close,
            save_to_file_on_exception=save_to_file_on_exception,
        )

    def plot(
        self,
        y_data_list: List[np.ndarray],
        x_data_list: Optional[List[np.ndarray]] = None,
    ) -> None:
        for y_data in y_data_list:
            assert_equals(len(y_data.shape), 1)

        if x_data_list is None:
            x_data_list = [np.arange(len(y_data)) for y_data in y_data_list]

        assert x_data_list is not None
        assert_equals(len(x_data_list), len(y_data_list))
        for x_data, y_data in zip(x_data_list, y_data_list):
            assert_equals(x_data.shape, y_data.shape)

        fast_plot_helper(
            fig=self.fig,
            x_data_list=x_data_list,
            y_data_list=y_data_list,
            axes=self.axes,
            lines=self.lines,
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

    live_plotter = FastLivePlotter(titles=["sin", "cos"], n_rows=2, n_cols=1)
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
    live_plotter = FastLivePlotter.from_desired_n_plots(
        titles=plot_names, desired_n_plots=len(plot_names)
    )
    for i in range(25):
        y_data_dict["exp(-x/10)"].append(np.exp(-i / 10))
        y_data_dict["ln(x + 1)"].append(np.log(i + 1))
        y_data_dict["x^2"].append(np.power(i, 2))
        y_data_dict["4x^4"].append(4 * np.power(i, 4))
        y_data_dict["ln(2^x)"].append(np.log(np.power(2, i)))

        live_plotter.plot(
            y_data_list=[np.array(y_data_dict[plot_name]) for plot_name in plot_names],
        )


if __name__ == "__main__":
    main()
