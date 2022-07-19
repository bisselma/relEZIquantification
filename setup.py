# -*- coding: utf-8 -*-

"""The setup script."""
import sys
from setuptools import find_packages, setup

__author__ = """Ben Isselmann"""
__email__ = "ben.isselmann@gmail.com"
__version__ = "0.0.6"


#with open("README.rst") as readme_file:
 #   readme = readme_file.read()

#with open("HISTORY.rst") as history_file:
 #   history = history_file.read()

requirements = ["eyepie==0.3.7", "numpy>=1.22", "opencv-python", "opencv-python-headless", "Pillow",
    "xlsxwriter"]
setup_requirements = ["pytest-runner"]

test_requirements = ["pytest>=3"]


setup(
    author=__author__,
    author_email=__email__,
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="The Python package to analyse the relative elipsoid zone intensity (relEZI) by oct imaging",
    install_requires=requirements,
    license="MIT license",
    #long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords=["rel_ez_intensity"],
    name="rel_ez_intensity",
    packages=find_packages(include=["rel_ez_intensity", "rel_ez_intensity.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/bisselma/rel_ez_intensity",
    version=__version__,
    zip_safe=False,
)
