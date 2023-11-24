# Avoid using GUI for testing
import matplotlib

matplotlib.use("Agg")
from live_plotter.fast_live_plotter_core import main


def test_fast_live_plotter() -> None:
    main()
