from pathlib import Path
import setuptools

VERSION = "0.2.25"

NAME = "SynthOpt"

INSTALL_REQUIRES = [
    "sdmetrics",
    "sdv",
    "synthcity",
    "anonymeter",
    "seaborn",
    "reportlab",
    "distfit",
    "tqdm",
]

setuptools.setup(
    name=NAME,
    version=VERSION,
    description="A package for synthetic data generation, evaluation and optimisation.",
    url="https://github.com/LewisHotchkissDPUK/SynthOpt",
    project_urls={
        "Source Code": "https://github.com/LewisHotchkissDPUK/SynthOpt",
    },
    author="Lewis Hotchkiss", 
    author_email="lewishotchkiss123@gmail.com",
    license="Apache License 2.0",
    install_requires=INSTALL_REQUIRES,
    packages=setuptools.find_packages(),  # Automatically discover your package
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
)
