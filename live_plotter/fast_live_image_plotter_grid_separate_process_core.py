from multiprocessing import Process, Event, Manager
from typing import List
import numpy as np
import sys

from live_plotter.fast_live_image_plotter_core import FastLiveImagePlotterGrid


class FastLiveImagePlotterGridSeparateProcess:
    def __init__(
        self,
        plot_names: List[str],
        save_to_file_on_close: bool = False,
        save_to_file_on_exception: bool = False,
    ) -> None:
        self.plot_names = plot_names

        self.live_image_plotter = FastLiveImagePlotterGrid.from_desired_n_plots(
            title=plot_names,
            desired_n_plots=len(plot_names),
            save_to_file_on_close=save_to_file_on_close,
            save_to_file_on_exception=save_to_file_on_exception,
        )

        self.manager = Manager()
        self.data_dict = self.manager.dict()
        for plot_name in plot_names:
            self.data_dict[plot_name] = self.manager.list()

        self.update_event = Event()
        self.process = Process(target=self._run_task, daemon=True)
        self._setup_exception_hook()

    def start(self) -> None:
        self.process.start()

    def update(self) -> None:
        self.update_event.set()

    def _run_task(self) -> None:
        while True:
            self.update_event.wait()
            self.update_event.clear()

            self.live_image_plotter.plot_grid(
                image_data_list=[
                    np.array(self.data_dict[plot_name]) for plot_name in self.plot_names
                ],
            )

    def __del__(self) -> None:
        print(f"__del__ called ({self.__class__.__name__})")
        self.process.terminate()

    def _setup_exception_hook(self) -> None:
        original_excepthook = sys.excepthook

        def exception_hook(exctype, value, traceback):
            print(f"Exception hook called ({self.__class__.__name__})")
            self.process.terminate()
            original_excepthook(exctype, value, traceback)

        sys.excepthook = exception_hook


def main() -> None:
    import time

    N_ITERS = 100
    SIMULATED_COMPUTATION_TIME_S = 0.1
    OPTIMAL_TIME_S = N_ITERS * SIMULATED_COMPUTATION_TIME_S

    DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH = 100, 100

    # Slower when plotting is on same process
    live_image_plotter = FastLiveImagePlotterGrid.from_desired_n_plots(
        title=["sin", "cos"], desired_n_plots=2
    )
    x_data = []
    start_time_same_process = time.time()
    for i in range(N_ITERS):
        x_data.append(i)
        time.sleep(SIMULATED_COMPUTATION_TIME_S)
        live_image_plotter.plot_grid(
            image_data_list=[
                np.sin(x_data)[None, ...]
                .repeat(DEFAULT_IMAGE_HEIGHT, 0)
                .repeat(DEFAULT_IMAGE_WIDTH, 1),
                np.cos(x_data)[None, ...]
                .repeat(DEFAULT_IMAGE_HEIGHT, 0)
                .repeat(DEFAULT_IMAGE_WIDTH, 1),
            ]
        )
    time_taken_same_process = time.time() - start_time_same_process

    # Faster when plotting is on separate process
    live_image_plotter_separate_process = FastLiveImagePlotterGridSeparateProcess(
        plot_names=["sin", "cos"]
    )
    live_image_plotter_separate_process.start()
    x_data = []
    start_time_separate_process = time.time()
    for i in range(N_ITERS):
        x_data.append(i)
        time.sleep(SIMULATED_COMPUTATION_TIME_S)
        live_image_plotter_separate_process.data_dict["sin"] = (
            np.sin(x_data)[None, ...]
            .repeat(DEFAULT_IMAGE_HEIGHT, 0)
            .repeat(DEFAULT_IMAGE_WIDTH, 1)
        )
        live_image_plotter_separate_process.data_dict["cos"] = (
            np.cos(x_data)[None, ...]
            .repeat(DEFAULT_IMAGE_HEIGHT, 0)
            .repeat(DEFAULT_IMAGE_WIDTH, 1)
        )
        live_image_plotter_separate_process.update()
    time_taken_separate_process = time.time() - start_time_separate_process

    print(f"Time taken same process: {round(time_taken_same_process, 1)} s")
    print(f"Time taken separate process: {round(time_taken_separate_process, 1)} s")
    print(f"OPTIMAL_TIME_S: {round(OPTIMAL_TIME_S, 1)} s")

    assert time_taken_separate_process < time_taken_same_process


if __name__ == "__main__":
    main()
