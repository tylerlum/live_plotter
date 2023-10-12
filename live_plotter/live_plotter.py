import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Union
import math
from datetime import datetime

import seaborn as sns

sns.set_theme()


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
    fig: plt.Figure,
    x_data_list: List[np.ndarray],
    y_data_list: List[np.ndarray],
    n_rows: int,
    n_cols: int,
    titles: Optional[List[str]],
    xlabels: Optional[List[str]],
    ylabels: Optional[List[str]],
) -> None:
    n_plots = len(x_data_list)
    assert_equals(len(y_data_list), n_plots)
    assert n_plots <= n_rows * n_cols, f"{n_plots} > {n_rows} * {n_cols}"

    if titles is not None:
        assert_equals(len(titles), n_plots)
    if xlabels is not None:
        assert_equals(len(xlabels), n_plots)
    if ylabels is not None:
        assert_equals(len(ylabels), n_plots)

    plt.clf()

    for i in range(n_plots):
        ax_idx = i + 1
        ax = fig.add_subplot(n_rows, n_cols, ax_idx)

        ax.plot(x_data_list[i], y_data_list[i])
        if titles is not None:
            ax.set_title(titles[i])
        if xlabels is not None:
            ax.set_xlabel(xlabels[i])
        if ylabels is not None:
            ax.set_ylabel(ylabels[i])

    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.001)


class LivePlotter:
    def __init__(
        self,
        default_title: str = "",
        default_xlabel: str = "x",
        default_ylabel: str = "y",
        save_to_file_on_close: bool = False,
        save_to_file_on_exception: bool = False,
    ) -> None:
        self.fig = plt.figure()
        self.default_title = default_title
        self.default_xlabel = default_xlabel
        self.default_ylabel = default_ylabel
        self.save_to_file_on_close = save_to_file_on_close
        self.save_to_file_on_exception = save_to_file_on_exception
        plt.show(block=False)

        if self.save_to_file_on_exception:
            self._setup_exception_hook()

    def plot(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ) -> None:
        """Plot 1D data"""
        assert_equals(x_data.shape, y_data.shape)
        assert_equals(len(x_data.shape), 1)
        plot_helper(
            fig=self.fig,
            x_data_list=[x_data],
            y_data_list=[y_data],
            n_rows=1,
            n_cols=1,
            titles=[self.default_title if title is None else title],
            xlabels=[self.default_xlabel if xlabel is None else xlabel],
            ylabels=[self.default_ylabel if ylabel is None else ylabel],
        )

    def _save_to_file(self) -> None:
        filename = (
            f"{datetime_str()}_{self.default_title}.png"
            if len(self.default_title) > 0
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


class LivePlotterGrid:
    def __init__(
        self,
        default_title: Union[str, List[str]] = "",
        default_xlabel: Union[str, List[str]] = "x",
        default_ylabel: Union[str, List[str]] = "y",
        save_to_file_on_close: bool = False,
        save_to_file_on_exception: bool = False,
    ) -> None:
        self.fig = plt.figure()
        self.default_title = default_title
        self.default_xlabel = default_xlabel
        self.default_ylabel = default_ylabel
        self.save_to_file_on_close = save_to_file_on_close
        self.save_to_file_on_exception = save_to_file_on_exception
        plt.show(block=False)

        if self.save_to_file_on_exception:
            self._setup_exception_hook()

    def plot_grid(
        self,
        x_data_list: List[np.ndarray],
        y_data_list: List[np.ndarray],
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        title: Optional[Union[str, List[str]]] = None,
        xlabel: Optional[Union[str, List[str]]] = None,
        ylabel: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Plot multiple 1D datas in a grid"""
        assert_equals(len(x_data_list), len(y_data_list))
        n_plots = len(x_data_list)

        # Infer n_rows and n_cols if not given
        if n_rows is None and n_cols is None:
            n_rows = math.ceil(math.sqrt(n_plots))
            n_cols = math.ceil(n_plots / n_rows)
        elif n_cols is None:
            assert n_rows is not None
            n_cols = math.ceil(n_plots / n_rows)
        elif n_rows is None:
            n_rows = math.ceil(n_plots / n_cols)

        titles = convert_to_list_str_fixed_len(
            str_or_list_str=(title if title is not None else self.default_title),
            fixed_length=n_plots,
        )
        xlabels = convert_to_list_str_fixed_len(
            str_or_list_str=(xlabel if xlabel is not None else self.default_xlabel),
            fixed_length=n_plots,
        )
        ylabels = convert_to_list_str_fixed_len(
            str_or_list_str=(ylabel if ylabel is not None else self.default_ylabel),
            fixed_length=n_plots,
        )

        plot_helper(
            fig=self.fig,
            x_data_list=x_data_list,
            y_data_list=y_data_list,
            n_rows=n_rows,
            n_cols=n_cols,
            titles=titles,
            xlabels=xlabels,
            ylabels=ylabels,
        )

    def _save_to_file(self) -> None:
        filename = (
            f"{datetime_str()}_{self.default_title}.png"
            if len(self.default_title) > 0
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

    live_plotter = LivePlotter(default_title="sin")

    x_data = []
    for i in range(25):
        x_data.append(i)
        live_plotter.plot(x_data=np.array(x_data), y_data=np.sin(x_data))

    time.sleep(2)

    live_plotter_grid = LivePlotterGrid(default_title="sin")
    x_data = []
    x_data2 = []
    for i in range(25):
        x_data.append(i)
        x_data2.append(i)

        live_plotter_grid.plot_grid(
            x_data_list=[np.array(x_data), np.array(x_data2)],
            y_data_list=[np.sin(x_data), np.cos(x_data2)],
            n_rows=2,
            title=["sin", "cos"],
        )

    time.sleep(2)

    live_plotter_grid.default_title = "exp"
    NUM_DATAS = 7
    x_data_list = [[] for _ in range(NUM_DATAS)]
    y_data_list = [[] for _ in range(NUM_DATAS)]
    for i in range(25):
        for j in range(NUM_DATAS):
            x_data_list[j].append(i)
            y_data_list[j].append(np.exp(-j / 10 * i))
        live_plotter_grid.plot_grid(
            x_data_list=[np.array(x_data) for x_data in x_data_list],
            y_data_list=[np.array(y_data) for y_data in y_data_list],
            xlabel=[f"x{i}" for i in range(NUM_DATAS)],
            ylabel=[f"y{i}" for i in range(NUM_DATAS)],
        )


if __name__ == "__main__":
    main()
