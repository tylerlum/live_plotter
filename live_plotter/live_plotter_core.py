import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Optional, List, Tuple
import math
import sys

import seaborn as sns

from live_plotter.utils import (
    assert_equals,
    datetime_str,
    convert_to_list_str_fixed_len,
)


sns.set_theme()


def plot_helper(
    fig: Figure,
    x_data_list: List[np.ndarray],
    y_data_list: List[np.ndarray],
    n_rows: int,
    n_cols: int,
    titles: Optional[List[str]],
    xlabels: Optional[List[str]],
    ylabels: Optional[List[str]],
    xlims: Optional[List[Tuple[float, float]]],
    ylims: Optional[List[Tuple[float, float]]],
) -> None:
    """Plot data on existing figure"""
    n_plots = len(x_data_list)
    assert_equals(len(y_data_list), n_plots)
    assert n_plots <= n_rows * n_cols, f"{n_plots} > {n_rows} * {n_cols}"

    if titles is not None:
        assert_equals(len(titles), n_plots)
    if xlabels is not None:
        assert_equals(len(xlabels), n_plots)
    if ylabels is not None:
        assert_equals(len(ylabels), n_plots)
    if xlims is not None:
        assert_equals(len(xlims), n_plots)
    if ylims is not None:
        assert_equals(len(ylims), n_plots)

    plt.clf()

    for i in range(n_plots):
        ax_idx = i + 1
        ax = fig.add_subplot(n_rows, n_cols, ax_idx)

        ax.plot(x_data_list[i], y_data_list[i])

        if titles is not None:
            PLOT_MODIFIED_TITLE = False
            if PLOT_MODIFIED_TITLE:
                ax.set_title(
                    " ".join([titles[i], f"(Plot {i})"]) if n_plots > 1 else titles[i]
                )
            else:
                ax.set_title(titles[i])

        if xlabels is not None:
            ax.set_xlabel(xlabels[i])
        if ylabels is not None:
            ax.set_ylabel(ylabels[i])
        if xlims is not None:
            left, right = xlims[i]
            ax.set_xlim(left=left, right=right)
        if ylims is not None:
            bottom, top = ylims[i]
            ax.set_ylim(bottom=bottom, top=top)

    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.001)


class LivePlotter:
    def __init__(
        self,
        default_titles: Optional[List[str]] = None,
        default_xlabels: Optional[List[str]] = None,
        default_ylabels: Optional[List[str]] = None,
        default_xlims: Optional[List[Tuple[float, float]]] = None,
        default_ylims: Optional[List[Tuple[float, float]]] = None,
        save_to_file_on_close: bool = False,
        save_to_file_on_exception: bool = False,
    ) -> None:
        self.default_titles = default_titles
        self.default_xlabels = default_xlabels
        self.default_ylabels = default_ylabels
        self.default_xlims = default_xlims
        self.default_ylims = default_ylims
        self.save_to_file_on_close = save_to_file_on_close
        self.save_to_file_on_exception = save_to_file_on_exception

        self.fig = plt.figure()
        plt.show(block=False)

        if self.save_to_file_on_exception:
            self._setup_exception_hook()

    def plot(
        self,
        y_data_list: List[np.ndarray],
        x_data_list: Optional[List[np.ndarray]] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        titles: Optional[List[str]] = None,
        xlabels: Optional[List[str]] = None,
        ylabels: Optional[List[str]] = None,
        xlims: Optional[List[Tuple[float, float]]] = None,
        ylims: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Plot multiple 1D datas in a grid"""
        for y_data in y_data_list:
            assert_equals(len(y_data.shape), 1)

        if x_data_list is None:
            x_data_list = [np.arange(len(y_data)) for y_data in y_data_list]

        assert x_data_list is not None
        assert_equals(len(x_data_list), len(y_data_list))
        for x_data, y_data in zip(x_data_list, y_data_list):
            assert_equals(x_data.shape, y_data.shape)

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
            list_str=(titles if titles is not None else self.default_titles),
            fixed_length=n_plots,
        )
        xlabels = convert_to_list_str_fixed_len(
            list_str=(xlabels if xlabels is not None else self.default_xlabels),
            fixed_length=n_plots,
        )
        ylabels = convert_to_list_str_fixed_len(
            list_str=(ylabels if ylabels is not None else self.default_ylabels),
            fixed_length=n_plots,
        )
        if xlims is None:
            xlims = self.default_xlims
        if ylims is None:
            ylims = self.default_ylims
        if xlims is not None:
            assert_equals(len(xlims), n_plots)
        if ylims is not None:
            assert_equals(len(ylims), n_plots)

        plot_helper(
            fig=self.fig,
            x_data_list=x_data_list,
            y_data_list=y_data_list,
            n_rows=n_rows,
            n_cols=n_cols,
            titles=titles,
            xlabels=xlabels,
            ylabels=ylabels,
            xlims=xlims,
            ylims=ylims,
        )

    def _save_to_file(self) -> None:
        filename = (
            f"{datetime_str()}_{self.default_titles}.png"
            if self.default_titles is not None
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

    live_plotter = LivePlotter()
    x_data = []
    for i in range(25):
        x_data.append(i)
        live_plotter.plot(
            y_data_list=[np.sin(x_data), np.cos(x_data)],
            titles=["sin", "cos"],
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
    for i in range(25):
        y_data_dict["exp(-x/10)"].append(np.exp(-i / 10))
        y_data_dict["ln(x + 1)"].append(np.log(i + 1))
        y_data_dict["x^2"].append(np.power(i, 2))
        y_data_dict["4x^4"].append(4 * np.power(i, 4))
        y_data_dict["ln(2^x)"].append(np.log(np.power(2, i)))

        live_plotter.plot(
            y_data_list=[np.array(y_data_dict[plot_name]) for plot_name in plot_names],
            titles=plot_names,
        )


if __name__ == "__main__":
    main()
