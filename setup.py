from setuptools import setup, find_packages
from pathlib import Path

VERSION = "0.0.3"
DESCRIPTION = "Plot live data that updates in real time using matplotlib backend"
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="live_plotter",
    version=VERSION,
    author="Tyler Lum",
    author_email="tylergwlum@gmail.com",
    url="https://github.com/tylerlum/live_plotter",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["matplotlib", "numpy", "seaborn"],
    keywords=["python", "matplotlib", "plot", "live", "real time"],
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
