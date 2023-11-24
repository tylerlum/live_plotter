from multiprocessing import Process, Event, Manager
from typing import List, Union
import numpy as np
import sys

from live_plotter.live_image_plotter_core import LiveImagePlotter
from live_plotter.live_plotter_core import LivePlotter


class SeparateProcessLivePlotter:
    def __init__(
        self,
        live_plotter: Union[LivePlotter, LiveImagePlotter],
        plot_names: List[str],
    ) -> None:
        self.live_plotter = live_plotter
        self.plot_names = plot_names

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
        try:
            while True:
                self.update_event.wait()
                self.update_event.clear()

                self.live_plotter.plot(
                    [np.array(self.data_dict[plot_name]) for plot_name in self.plot_names],
                    title=self.plot_names,
                )
        except Exception as e:
            print(f"Exception in {self.__class__.__name__}: {e}")

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


def test_live_image_plotter() -> None:
    import time

    N_ITERS = 100
    SIMULATED_COMPUTATION_TIME_S = 0.1
    OPTIMAL_TIME_S = N_ITERS * SIMULATED_COMPUTATION_TIME_S

    DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH = 100, 100

    # Slower when plotting is on same process
    live_image_plotter = LiveImagePlotter(default_title=["sin", "cos"])
    x_data = []
    start_time_same_process = time.time()
    for i in range(N_ITERS):
        x_data.append(i)
        time.sleep(SIMULATED_COMPUTATION_TIME_S)
        live_image_plotter.plot(
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
    live_image_plotter_separate_process = SeparateProcessLivePlotter(
        live_plotter=LiveImagePlotter(), plot_names=["sin", "cos"]
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


def test_live_plotter() -> None:
    import time

    N_ITERS = 100
    SIMULATED_COMPUTATION_TIME_S = 0.1
    OPTIMAL_TIME_S = N_ITERS * SIMULATED_COMPUTATION_TIME_S

    # Slower when plotting is on same process
    live_plotter = LivePlotter(default_title=["sin", "cos"])
    x_data = []
    start_time_same_process = time.time()
    for i in range(N_ITERS):
        x_data.append(i)
        time.sleep(SIMULATED_COMPUTATION_TIME_S)
        live_plotter.plot(
            y_data_list=[np.sin(x_data), np.cos(x_data)],
        )
    time_taken_same_process = time.time() - start_time_same_process

    # Faster when plotting is on separate process
    live_plotter_separate_process = SeparateProcessLivePlotter(
        live_plotter=LivePlotter(), plot_names=["sin", "cos"]
    )
    live_plotter_separate_process.start()
    start_time_separate_process = time.time()
    for i in range(N_ITERS):
        time.sleep(SIMULATED_COMPUTATION_TIME_S)
        live_plotter_separate_process.data_dict["sin"].append(np.sin(i))
        live_plotter_separate_process.data_dict["cos"].append(np.cos(i))
        live_plotter_separate_process.update()
    time_taken_separate_process = time.time() - start_time_separate_process

    print(f"Time taken same process: {round(time_taken_same_process, 1)} s")
    print(f"Time taken separate process: {round(time_taken_separate_process, 1)} s")
    print(f"OPTIMAL_TIME_S: {round(OPTIMAL_TIME_S, 1)} s")

    assert time_taken_separate_process < time_taken_same_process


def main() -> None:
    test_live_image_plotter()
    test_live_plotter()


if __name__ == "__main__":
    main()
