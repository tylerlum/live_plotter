# live_plotter

Plot live data that updates in real time using matplotlib backend

# Installing

Install:

```
pip install live_plotter
```

# Usage

In this library, we have two axes of variation. The first axis of variation is using either `LivePlotter` or `LivePlotterGrid`. `LivePlotter` creates 1 plot, while `LivePlotterGrid` creates a grid of plots. The second axis of variation is using either `LivePlotterGrid` or `FastLivePlotterGrid`. `LivePlotterGrid` is more flexible and dynamic, but this results in slower updates. `FastLivePlotterGrid` requires that the user specify the number of plots in the figure from the beginning, but this allows it to update faster by modifying an existing plot rather than creating a new plot from scratch. Please refer to the associated example code for more details.

Lastly, you can add `save_to_file_on_close=True` to save the figure to a file when the live plotter is deleted (either out of scope or end of script). You can add `save_to_file_on_exception=True` to save the figure to a file when an exception occurs. Note this feature is experimental.

New feature: we have added `FastLivePlotterGridSeparateProcess`, which is a wrapper around `FastLivePlotterGrid` but puts the plotting code in another process. Plotting takes time, so running the plotting code in the same process as the main process can significantly slow things down, especially as plots get larger. This must be done on a new process instead of a new thread because the GUI does not work on non-main threads.

Options:

- `LivePlotter`

- `LivePlotterGrid`

- `FastLivePlotter`

- `FastLivePlotterGrid`

- `FastLivePlotterGridSeparateProcess`

## Live Plotter

![live_plotter](https://github.com/tylerlum/live_plotting/assets/26510814/919532a7-3d6d-47c2-b2e6-4aebb66d2591)

## Fast Live Plotter

![fast_live_plotter](https://github.com/tylerlum/live_plotting/assets/26510814/6c9c1647-e4b2-4589-ba91-ba3f5947843c)

## Example Usage of `LivePlotter`

```
import numpy as np
from live_plotter import LivePlotter

live_plotter = LivePlotter(default_title="sin")

x_data = []
for i in range(25):
    x_data.append(2 * i)
    live_plotter.plot(x_data=np.array(x_data), y_data=np.sin(x_data))
```

## Example Usage of `FastLivePlotter`

```
import numpy as np
from live_plotter import FastLivePlotter

live_plotter = FastLivePlotter(title="sin")

x_data = []
for i in range(25):
    x_data.append(2 * i)
    live_plotter.plot(x_data=np.array(x_data), y_data=np.sin(x_data))
```

## Example Usage of `LivePlotterGrid`

```
import numpy as np
from live_plotter import LivePlotterGrid

live_plotter_grid = LivePlotterGrid(default_title="sin")

x_data = []
for i in range(25):
    x_data.append(i)
    live_plotter_grid.plot_grid(
        y_data_list=[np.sin(x_data), np.cos(x_data)],
        title=["sin", "cos"],
    )
```

## Example Usage of `FastLivePlotterGrid`

```
import numpy as np

from live_plotter import FastLivePlotterGrid

live_plotter_grid = FastLivePlotterGrid(title=["sin", "cos"], n_rows=2, n_cols=1)
x_data = []
for i in range(25):
    x_data.append(i)
    live_plotter_grid.plot_grid(
        y_data_list=[np.sin(x_data), np.cos(x_data)],
    )
```

## Example Usage of `FastLivePlotterGrid` using `from_desired_n_plots` (recommended method for more complex use-cases)

```
import numpy as np

from live_plotter import FastLivePlotterGrid

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
```

## Example Usage of `FastLivePlotterGridSeparateProcess` (recommended method to minimize plotting time impacting main code performance)

```
import numpy as np
import time

from live_plotter import FastLivePlotterGridSeparateProcess

N_ITERS = 100
SIMULATED_COMPUTATION_TIME_S = 0.1
OPTIMAL_TIME_S = N_ITERS * SIMULATED_COMPUTATION_TIME_S

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

print(f"Time taken separate process: {round(time_taken_separate_process, 1)} s")
print(f"OPTIMAL_TIME_S: {round(OPTIMAL_TIME_S, 1)} s")
```
Output:
```
Time taken separate process: 10.3 s
OPTIMAL_TIME_S: 10.0 s
```
You may get an error `ConnectionResetError: [Errno 104] Connection reset by peer` at the end. This is not a problem, as it means the main process ended before the new process could be killed.


Note how this runs much faster than the equivalent same process code
```
import numpy as np
import time

from live_plotter import FastLivePlotterGrid

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

print(f"Time taken same process: {round(time_taken_same_process, 1)} s")
print(f"OPTIMAL_TIME_S: {round(OPTIMAL_TIME_S, 1)} s")
```
Output:
```
Time taken same process: 18.3 s
OPTIMAL_TIME_S: 10.0 s
```