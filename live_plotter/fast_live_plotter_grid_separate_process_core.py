from multiprocessing import Process, Event, Manager
from multiprocessing.managers import DictProxy
from typing import List
import numpy as np

from live_plotter.fast_live_plotter_core import FastLivePlotterGrid


class FastLivePlotterGridSeparateProcess:
    def __init__(
        self,
        plot_names: List[str],
        save_to_file_on_close: bool = False,
        save_to_file_on_exception: bool = False,
    ) -> None:
        self.plot_names = plot_names

        self.live_plotter = FastLivePlotterGrid.from_desired_n_plots(
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

    def start(self) -> None:
        self.process.start()

    def update(self) -> None:
        self.update_event.set()

    def _run_task(self) -> None:
        while True:
            self.update_event.wait()
            self.update_event.clear()

            self.live_plotter.plot_grid(
                y_data_list=[
                    np.array(self.data_dict[plot_name]) for plot_name in self.plot_names
                ],
            )


def main() -> None:
    import time

    N_ITERS = 100
    SIMULATED_COMPUTATION_TIME_S = 0.1
    OPTIMAL_TIME_S = N_ITERS * SIMULATED_COMPUTATION_TIME_S

    # Slower when plotting is on same process
    live_plotter = FastLivePlotterGrid.from_desired_n_plots(
        title=["sin", "cos"], desired_n_plots=2
    )
    x_data = []
    start_time_same_process = time.time()
    for i in range(N_ITERS):
        x_data.append(i)
        time.sleep(SIMULATED_COMPUTATION_TIME_S)
        live_plotter.plot_grid(
            y_data_list=[np.sin(x_data), np.cos(x_data)],
        )
    time_taken_same_process = time.time() - start_time_same_process

    # Faster when plotting is on separate process
    live_plotter_separate_process = FastLivePlotterGridSeparateProcess(
        plot_names=["sin", "cos"]
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


if __name__ == "__main__":
    main()
