# Avoid using GUI for testing
import matplotlib

matplotlib.use("Agg")
from live_plotter.separate_process_live_plotter_core import (
    main_live_image_plotter,
    main_live_plotter,
)


def test_separate_process_live_image_plotter() -> None:
    main_live_image_plotter()


def test_separate_process_live_plotter() -> None:
    main_live_plotter()
