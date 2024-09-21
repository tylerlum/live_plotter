from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from live_plotter.utils import (
    assert_equals,
    compute_n_rows_n_cols,
)

sns.set_theme()


class LivePlotter:
    def __init__(
        self,
    ) -> None:
        """
        Create a LivePlotter object.
        """
        self.fig = plt.figure()
        plt.show(block=False)

    def plot(
        self,
        y_data_list: List[np.ndarray],
        x_data_list: Optional[List[Optional[np.ndarray]]] = None,
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
        Create a grid of subplots of shape n_rows x n_cols (or automatically computed if not given).

        Args:
          y_data_list: List[np.ndarray], where each element is the y_data for subplot i
                       y_data is expected to be 1D of shape (D,) or 2D of shape (D, N), where D is the data dimension for 1 plot and N is the number of plots in this subplot
          x_data_list: Optional[List[Optional[np.ndarray]]], where each element is the x_data for a subplot
                       If x_data_list is None, then x_data is assumed to be default 0, 1, 2, ..., D-1 for all subplots
                       If x_data_list[i] is None, then x_data is assumed to be default 0, 1, 2, ..., D-1 for subplot i
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
                   Ignored if y_data is 1D
        """
        n_plots = len(y_data_list)

        # Infer n_rows and n_cols if not given
        n_rows, n_cols = compute_n_rows_n_cols(
            n_plots=n_plots,
            n_rows=n_rows,
            n_cols=n_cols,
        )
        assert (
            n_plots <= n_rows * n_cols
        ), f"n_plots = {n_plots}, n_rows = {n_rows}, n_cols = {n_cols}"

        # Validate y_data
        for y_data in y_data_list:
            assert y_data.ndim in [1, 2], f"y_data.ndim = {y_data.ndim}"

        # Validate other inputs
        if x_data_list is None:
            x_data_list = [None for _ in range(n_plots)]
        assert_equals(len(x_data_list), n_plots)

        if titles is None:
            titles = [None for _ in range(n_plots)]
        assert_equals(len(titles), n_plots)

        if xlabels is None:
            xlabels = [None for _ in range(n_plots)]
        assert_equals(len(xlabels), n_plots)

        if ylabels is None:
            ylabels = [None for _ in range(n_plots)]
        assert_equals(len(ylabels), n_plots)

        if xlims is None:
            xlims = [None for _ in range(n_plots)]
        assert_equals(len(xlims), n_plots)

        if ylims is None:
            ylims = [None for _ in range(n_plots)]
        assert_equals(len(ylims), n_plots)

        if legends is None:
            legends = [None for _ in range(n_plots)]
        assert_equals(len(legends), n_plots)

        plt.clf()

        for i, (x_data, y_data, title, xlabel, ylabel, xlim, ylim, legend) in enumerate(
            zip(
                x_data_list,
                y_data_list,
                titles,
                xlabels,
                ylabels,
                xlims,
                ylims,
                legends,
            )
        ):
            ax_idx = i + 1
            ax = self.fig.add_subplot(n_rows, n_cols, ax_idx)

            if x_data is None:
                if y_data.ndim == 2:
                    D, N = y_data.shape
                    x_data = np.arange(D).reshape(-1, 1).repeat(N, axis=1)
                else:
                    D = y_data.shape[0]
                    x_data = np.arange(D)
            assert_equals(x_data.shape, y_data.shape)

            if y_data.ndim == 2 and legend is not None:
                _, N = y_data.shape
                assert_equals(len(legend), N)
                ax.plot(x_data, y_data, label=legend)
                ax.legend()
            else:
                ax.plot(x_data, y_data)

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

        self.fig.tight_layout()
        self.fig.canvas.draw()

        # Replace plt.pause(0.001) to avoid focus stealing
        # https://github.com/tylerlum/live_plotter/issues/2
        # plt.pause(0.001)
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.001)


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

    new_x_data = []
    for i in range(25):
        new_x_data.append(i)
        y_data = np.stack([np.sin(new_x_data), np.cos(new_x_data)], axis=1)
        live_plotter.plot(
            y_data_list=[y_data],
            titles=["sin and cos"],
            xlabels=["x"],
            ylabels=["y"],
            ylims=[(-2, 2)],
            legends=[["sin", "cos"]],
        )


if __name__ == "__main__":
    main()
