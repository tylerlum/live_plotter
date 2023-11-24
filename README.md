# live_plotter

Plot live data that updates in real time using matplotlib backend

# Installing

Install:

```
pip install live_plotter
```

# Usage

In this library, we have two axes of variation:

* The first axis of variation is using either `LivePlotter` or `LiveImagePlotter`. `LivePlotter` create line plots. `LiveImagePlotter` creates image plots.

* The second axis of variation is using either `LivePlotter` or `FastLivePlotter`. `LivePlotter` is more flexible and dynamic, but this results in slower updates. `FastLivePlotter` requires that the user specify the number of plots in the figure from the beginning, but this allows it to update faster by modifying an existing plot rather than creating a new plot from scratch.

Additionally, we have a wrapper `SeparateProcessLivePlotter` that takes in any of the above plotters and creates a separate process to update the plot. The above plotters run on the same process as the main process, but `SeparateProcessLivePlotter` is run on another process so there is much less performance overhead on the main process from plotting. Plotting takes time, so running the plotting code in the same process as the main process can significantly slow things down, especially as plots get larger. This must be done on a new process instead of a new thread because the GUI does not work on non-main threads.

Lastly, you can add `save_to_file_on_close=True` to save the figure to a file when the live plotter is deleted (either out of scope or end of script). You can add `save_to_file_on_exception=True` to save the figure to a file when an exception occurs. Note this feature is experimental.

Please refer to the associated example code for more details.

Options:

- `LivePlotter`

- `FastLivePlotter`

- `LiveImagePlotter`

- `FastLiveImagePlotter`

- `SeparateProcessLivePlotter`

## Live Plotter

![2023-11-23_16-55-40_live_plot](https://github.com/tylerlum/live_plotter/assets/26510814/5481f062-743a-40f9-8e1a-31a2d8dee24e)

## Fast Live Plotter

![2023-11-23_16-55-46_fast_live_plot](https://github.com/tylerlum/live_plotter/assets/26510814/133093bc-6503-470d-b531-ab1b7948f13a)

## Live Image Plotter

![2023-11-23_16-55-54_live_image_plot](https://github.com/tylerlum/live_plotter/assets/26510814/6051c114-d537-4e1a-8889-34bc0c067fe5)

## Example Usage of `LivePlotter`

```
import numpy as np
from live_plotter import LivePlotter

live_plotter = LivePlotter(default_title="sin")
x_data = []
for i in range(25):
    x_data.append(i)
    live_plotter.plot(
        y_data_list=[np.sin(x_data), np.cos(x_data)],
        title=["sin", "cos"],
    )
```

## Example Usage of `FastLivePlotter`

```
import numpy as np
from live_plotter import FastLivePlotter

live_plotter = FastLivePlotter(title=["sin", "cos"], n_rows=2, n_cols=1)
x_data = []
for i in range(25):
    x_data.append(i)
    live_plotter.plot(
        y_data_list=[np.sin(x_data), np.cos(x_data)],
    )
```

## Example Usage of `FastLivePlotter` using `from_desired_n_plots` (recommended method for more complex use-cases)

```
import numpy as np

from live_plotter import FastLivePlotter

y_data_dict = {
    "exp(-x/10)": [],
    "ln(x + 1)": [],
    "x^2": [],
    "4x^4": [],
    "ln(2^x)": [],
}
plot_names = list(y_data_dict.keys())
live_plotter = FastLivePlotter.from_desired_n_plots(
    title=plot_names, desired_n_plots=len(plot_names)
)
for i in range(25):
    y_data_dict["exp(-x/10)"].append(np.exp(-i / 10))
    y_data_dict["ln(x + 1)"].append(np.log(i + 1))
    y_data_dict["x^2"].append(np.power(i, 2))
    y_data_dict["4x^4"].append(4 * np.power(i, 4))
    y_data_dict["ln(2^x)"].append(np.log(np.power(2, i)))

    live_plotter.plot(
        y_data_list=[np.array(y_data_dict[plot_name]) for plot_name in plot_names],
    )
```

## Example Usage of `SeparateProcessLivePlotter` (recommended method to minimize plotting time impacting main code performance)

```
import numpy as np
import time

from live_plotter import SeparateProcessLivePlotter, FastLivePlotter

N_ITERS = 100
SIMULATED_COMPUTATION_TIME_S = 0.1
OPTIMAL_TIME_S = N_ITERS * SIMULATED_COMPUTATION_TIME_S

# Slower when plotting is on same process
live_plotter = FastLivePlotter.from_desired_n_plots(
    desired_n_plots=2, title=["sin", "cos"]
)
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
    live_plotter=live_plotter, plot_names=["sin", "cos"]
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
```
Output:
```
Time taken same process: 19.0 s
Time taken separate process: 10.4 s
OPTIMAL_TIME_S: 10.0 s
```

Note how this runs much faster than the same process code

## Example Usage of `LiveImagePlotter`

Note:

* images must be (M, N) or (M, N, 3) or (M, N, 4)

* Typically images must either be floats in [0, 1] or ints in [0, 255]. If not in this range, we will automatically scale it and print a warning. We recommend using the scale_image function as shown below.

```
import numpy as np
from live_plotter import LiveImagePlotter, scale_image

N = 25
DEFAULT_IMAGE_HEIGHT = 100
DEFAULT_IMAGE_WIDTH = 100

live_plotter = LiveImagePlotter(default_title="sin")

x_data = []
for i in range(N):
    x_data.append(0.5 * i)
    image_data = (
        np.sin(x_data)[None, ...]
        .repeat(DEFAULT_IMAGE_HEIGHT, 0)
        .repeat(DEFAULT_IMAGE_WIDTH // N, 1)
    )
    live_plotter.plot(image_data_list=[scale_image(image_data, min_val=-1.0, max_val=1.0)])
```
