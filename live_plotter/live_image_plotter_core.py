import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Optional, List
import math
import sys

import seaborn as sns

from live_plotter.utils import (
    assert_equals,
    datetime_str,
    convert_to_list_str_fixed_len,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    preprocess_image_data_if_needed,
    validate_image_data,
    scale_image,
)


sns.set_theme()


def plot_images_helper(
    fig: Figure,
    image_data_list: List[np.ndarray],
    n_rows: int,
    n_cols: int,
    titles: Optional[List[str]],
) -> None:
    """Plot data on existing figure"""
    n_plots = len(image_data_list)
    assert n_plots <= n_rows * n_cols, f"{n_plots} > {n_rows} * {n_cols}"

    if titles is not None:
        assert_equals(len(titles), n_plots)

    plt.clf()

    for i in range(n_plots):
        ax_idx = i + 1
        ax = fig.add_subplot(n_rows, n_cols, ax_idx)

        validate_image_data(image_data=image_data_list[i])
        ax.imshow(image_data_list[i])
        ax.grid(False)
        if titles is not None:
            PLOT_MODIFIED_TITLE = False
            if PLOT_MODIFIED_TITLE:
                ax.set_title(
                    " ".join([titles[i], f"(Plot {i})"]) if n_plots > 1 else titles[i]
                )
            else:
                ax.set_title(titles[i])

    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.001)


class LiveImagePlotter:
    def __init__(
        self,
        default_titles: Optional[List[str]] = None,
        save_to_file_on_close: bool = False,
        save_to_file_on_exception: bool = False,
    ) -> None:
        self.default_titles = default_titles
        self.save_to_file_on_close = save_to_file_on_close
        self.save_to_file_on_exception = save_to_file_on_exception

        self.fig = plt.figure()
        plt.show(block=False)

        if self.save_to_file_on_exception:
            self._setup_exception_hook()

    def plot(
        self,
        image_data_list: List[np.ndarray],
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        titles: Optional[List[str]] = None,
    ) -> None:
        """Plot multiple 1D datas in a grid"""
        image_data_list = [
            preprocess_image_data_if_needed(image_data=image_data)
            for image_data in image_data_list
        ]

        n_plots = len(image_data_list)

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

        plot_images_helper(
            fig=self.fig,
            image_data_list=image_data_list,
            n_rows=n_rows,
            n_cols=n_cols,
            titles=titles,
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

    N = 25

    live_plotter = LiveImagePlotter()
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
            titles=plot_names,
        )


if __name__ == "__main__":
    main()
