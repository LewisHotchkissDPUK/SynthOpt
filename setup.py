from pathlib import Path
import setuptools

VERSION = "0.1.0"  # PEP-440

NAME = "SynthOpt"

INSTALL_REQUIRES = [
    "sdmetrics",
    "sdv",
    "synthcity",
    "anonymeter",
    "seaborn",
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
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Data Science",
    ],
    install_requires=INSTALL_REQUIRES,
    packages=setuptools.find_packages(),  # Automatically discover your package
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
)
