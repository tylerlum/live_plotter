# live_plotter

Plot live data that updates in real time using matplotlib backend

# Installing

Install:
```
pip install live_plotter
```

# Usage

In this library, we have `LivePlotter` and `FastLivePlotter`. `LivePlotter` is more flexible and dynamic, but this results in slower updates. `FastLivePlotter` requires that the user specify the figure's shape from the beginning, but this allows it to update faster by modifying an existing plot rather than creating a new plot from scratch. Please refer to the associated example code for more details. 

## Live Plotter
```
python live_plotter.py
```

![live_plotter](https://github.com/tylerlum/live_plotting/assets/26510814/919532a7-3d6d-47c2-b2e6-4aebb66d2591)

## Fast Live Plotter
```
python fast_live_plotter.py
```

![fast_live_plotter](https://github.com/tylerlum/live_plotting/assets/26510814/6c9c1647-e4b2-4589-ba91-ba3f5947843c)

## Example Usage

```
import matplotlib.pyplot as plt
import numpy as np

from live_plotter import LivePlotterGrid

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
```

```
import matplotlib.pyplot as plt
import numpy as np

from live_plotter import FastLivePlotterGrid

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
```

```
import matplotlib.pyplot as plt
import numpy as np

from live_plotter import FastLivePlotterGrid

NUM_DATAS = 7
live_plotter_grid = FastLivePlotterGrid.from_desired_n_plots(
    title="exp", desired_n_plots=NUM_DATAS
)
x_data_list = [[] for _ in range(NUM_DATAS)]
y_data_list = [[] for _ in range(NUM_DATAS)]
for i in range(25):
    # Add new data
    for j in range(NUM_DATAS):
        x_data_list[j].append(i)
        y_data_list[j].append(np.exp(-j / 10 * i))

    live_plotter_grid.plot_grid(
        x_data_list=[np.array(x_data) for x_data in x_data_list],
        y_data_list=[np.array(y_data) for y_data in y_data_list],
    )
```

