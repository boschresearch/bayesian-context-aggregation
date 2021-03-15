# Always prefer setuptools over distutils
# To use a consistent encoding
from codecs import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the readme file
with open(path.join(here, "readme.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bayesian_aggregation",
    version="0.0.1",
    description="bayesian_aggregation",
    long_description=long_description,
    url="https://github.com/boschresearch/bayesian-context-aggregation",
    author="Robert Bosch GmbH, Michael Volpp",
    author_email="michael.volpp@de.bosch.com",
    classifiers=["Programming Language :: Python :: 3"],
    python_requires=">3.7,<3.9",
    install_requires=[
        "matplotlib==3.3.2",
        "numpy==1.19.2",
        "pyyaml==5.3.1",
        "scipy==1.5.2",
        "tensorboard==2.3.0",
        "torch==1.4.0",
        "torchvision==0.5.0",
        "tqdm==4.50.0",
    ],
    packages=find_packages(),
    include_package_data=True,
)
