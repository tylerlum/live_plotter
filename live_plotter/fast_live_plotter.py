from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List
import math
from datetime import datetime

import seaborn as sns

sns.set_theme()

EPSILON = 1e-5


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


def datetime_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def convert_to_list_str_fixed_len(
    str_or_list_str: Union[str, List[str]], fixed_length: int
) -> List[str]:
    if isinstance(str_or_list_str, str):
        return [str_or_list_str] * fixed_length

    if len(str_or_list_str) < fixed_length:
        return str_or_list_str + [""] * (fixed_length - len(str_or_list_str))

    return str_or_list_str[:fixed_length]


def plot_helper(
    x_data_list: List[np.ndarray],
    y_data_list: List[np.ndarray],
    n_rows: int,
    n_cols: int,
    lines: List[plt.Line2D],
    axes: List[plt.Axes],
    fig: plt.Figure,
) -> None:
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

        # Add EPSILON to avoid case when min == max
        ax.set_xlim([np.min(x_data), np.max(x_data) + EPSILON])
        ax.set_ylim([np.min(y_data), np.max(y_data) + EPSILON])

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
        self.fig = plt.figure()
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
        x_data: np.ndarray,
        y_data: np.ndarray,
    ) -> None:
        assert_equals(x_data.shape, y_data.shape)
        assert_equals(len(x_data.shape), 1)

        plot_helper(
            x_data_list=[x_data],
            y_data_list=[y_data],
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            lines=self.lines,
            axes=self.axes,
            fig=self.fig,
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
        self.fig = plt.figure()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.save_to_file_on_close = save_to_file_on_close
        self.save_to_file_on_exception = save_to_file_on_exception
        self.n_plots = n_rows * n_cols

        self.title = convert_to_list_str_fixed_len(
            str_or_list_str=title, fixed_length=self.n_plots
        )
        self.xlabel = convert_to_list_str_fixed_len(
            str_or_list_str=xlabel, fixed_length=self.n_plots
        )
        self.ylabel = convert_to_list_str_fixed_len(
            str_or_list_str=ylabel, fixed_length=self.n_plots
        )
        plt.show(block=False)

        self.axes = []
        self.lines = []
        for i in range(self.n_plots):
            ax_idx = i + 1
            ax = self.fig.add_subplot(n_rows, n_cols, ax_idx)
            ax.set_title(
                " ".join([self.title[i]] + ([str(ax_idx)] if self.n_plots > 1 else []))
            )
            ax.set_xlabel(self.xlabel[i])
            ax.set_ylabel(self.ylabel[i])
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
        x_data_list: List[np.ndarray],
        y_data_list: List[np.ndarray],
    ) -> None:
        assert_equals(len(x_data_list), len(y_data_list))

        plot_helper(
            x_data_list=x_data_list,
            y_data_list=y_data_list,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            lines=self.lines,
            axes=self.axes,
            fig=self.fig,
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


def main() -> None:
    import time

    live_plotter = FastLivePlotter(title="sin")

    x_data = []
    for i in range(25):
        x_data.append(i)
        live_plotter.plot(x_data=np.array(x_data), y_data=np.sin(x_data))

    time.sleep(2)

    live_plotter_grid = FastLivePlotterGrid(title="sin cos", n_rows=2, n_cols=1)
    x_data = []
    x_data2 = []
    for i in range(25):
        x_data.append(i)
        x_data2.append(i)

        live_plotter_grid.plot_grid(
            x_data_list=[np.array(x_data), np.array(x_data2)],
            y_data_list=[np.sin(x_data), np.cos(x_data2)],
        )

    time.sleep(2)
    NUM_DATAS = 7
    live_plotter_grid = FastLivePlotterGrid.from_desired_n_plots(
        title="exp", desired_n_plots=NUM_DATAS
    )
    x_data_list = [[] for _ in range(NUM_DATAS)]
    y_data_list = [[] for _ in range(NUM_DATAS)]
    for i in range(25):
        for j in range(NUM_DATAS):
            x_data_list[j].append(i)
            y_data_list[j].append(np.exp(-j / 10 * i))
        live_plotter_grid.plot_grid(
            x_data_list=[np.array(x_data) for x_data in x_data_list],
            y_data_list=[np.array(y_data) for y_data in y_data_list],
        )


if __name__ == "__main__":
    main()
