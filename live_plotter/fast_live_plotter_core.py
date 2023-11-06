from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List, Optional, Tuple
import math

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
    fig: plt.Figure,
    x_data_list: List[np.ndarray],
    y_data_list: List[np.ndarray],
    n_rows: int,
    n_cols: int,
    axes: List[plt.Axes],
    lines: List[plt.Line2D],
) -> None:
    """Plot data on existing figure onto existing axes and lines"""
    # Shape checks
    n_plots = len(x_data_list)
    assert_equals(len(y_data_list), n_plots)
    assert n_plots <= n_rows * n_cols, f"{n_plots} > {n_rows} * {n_cols}"

    assert_equals(len(lines), n_rows * n_cols)
    assert_equals(len(axes), n_rows * n_cols)

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
        title: str = "",
        xlabel: str = "x",
        ylabel: str = "y",
        save_to_file_on_close: bool = False,
        save_to_file_on_exception: bool = False,
    ) -> None:
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
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
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.axes = [ax]
        self.lines = [ax.plot([], [])[0]]

        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.001)

        if self.save_to_file_on_exception:
            self._setup_exception_hook()

    def plot(
        self,
        y_data: np.ndarray,
        x_data: Optional[np.ndarray] = None,
    ) -> None:
        assert_equals(len(y_data.shape), 1)

        if x_data is None:
            x_data = np.arange(len(y_data))

        assert x_data is not None
        assert_equals(x_data.shape, y_data.shape)

        fast_plot_helper(
            fig=self.fig,
            x_data_list=[x_data],
            y_data_list=[y_data],
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            axes=self.axes,
            lines=self.lines,
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
        # Note this is hacky because excepthook may be overwritten by others
        import sys

        def exception_hook(exctype, value, traceback):
            print("Exception hook called")
            self._save_to_file()
            sys.__excepthook__(exctype, value, traceback)

        sys.excepthook = exception_hook


class FastLivePlotterGrid:
    def __init__(
        self,
        title: Union[str, List[str]] = "",
        xlabel: Union[str, List[str]] = "x",
        ylabel: Union[str, List[str]] = "y",
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
        self.xlabels = convert_to_list_str_fixed_len(
            str_or_list_str=xlabel, fixed_length=self.n_plots
        )
        self.ylabels = convert_to_list_str_fixed_len(
            str_or_list_str=ylabel, fixed_length=self.n_plots
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
            adjusted_title = (
                " ".join([_title, f"(Plot {i})"]) if self.n_plots > 1 else _title
            )
            ax.set_title(adjusted_title)
            ax.set_xlabel(_xlabel)
            ax.set_ylabel(_ylabel)
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
        title: Union[str, List[str]] = "",
        xlabel: Union[str, List[str]] = "x",
        ylabel: Union[str, List[str]] = "y",
        save_to_file_on_close: bool = False,
        save_to_file_on_exception: bool = False,
        desired_n_plots: int = 1,
    ) -> FastLivePlotterGrid:
        n_rows = math.ceil(math.sqrt(desired_n_plots))
        n_cols = math.ceil(desired_n_plots / n_rows)

        return cls(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            n_rows=n_rows,
            n_cols=n_cols,
            save_to_file_on_close=save_to_file_on_close,
            save_to_file_on_exception=save_to_file_on_exception,
        )

    def plot_grid(
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
            n_rows=self.n_rows,
            n_cols=self.n_cols,
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
        # Note this is hacky because excepthook may be overwritten by others
        import sys

        def exception_hook(exctype, value, traceback):
            print("Exception hook called")
            self._save_to_file()
            sys.__excepthook__(exctype, value, traceback)

        sys.excepthook = exception_hook


def main() -> None:
    import time

    live_plotter = FastLivePlotter(title="sin")

    x_data = []
    for i in range(25):
        x_data.append(0.5 * i)
        live_plotter.plot(x_data=np.array(x_data), y_data=np.sin(x_data))

    time.sleep(2)

    live_plotter_grid = FastLivePlotterGrid(title=["sin", "cos"], n_rows=2, n_cols=1)
    x_data = []
    for i in range(25):
        x_data.append(i)
        live_plotter_grid.plot_grid(
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
    live_plotter_grid = FastLivePlotterGrid.from_desired_n_plots(
        title=plot_names, desired_n_plots=len(plot_names)
    )
    for i in range(25):
        y_data_dict["exp(-x/10)"].append(np.exp(-i / 10))
        y_data_dict["ln(x + 1)"].append(np.log(i + 1))
        y_data_dict["x^2"].append(np.power(i, 2))
        y_data_dict["4x^4"].append(4 * np.power(i, 4))
        y_data_dict["ln(2^x)"].append(np.log(np.power(2, i)))

        live_plotter_grid.plot_grid(
            y_data_list=[np.array(y_data_dict[plot_name]) for plot_name in plot_names],
        )


if __name__ == "__main__":
    main()
