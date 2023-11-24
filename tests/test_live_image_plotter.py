# Avoid using GUI for testing
import matplotlib

matplotlib.use("Agg")
from live_plotter.live_image_plotter_core import main


def test_live_image_plotter() -> None:
    main()
