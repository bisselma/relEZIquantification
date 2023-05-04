# -*- coding: utf-8 -*-

"""The setup script."""
import sys
from setuptools import find_packages, setup

__author__ = """Ben Isselmann"""
__email__ = "ben.isselmann@gmail.com"
__version__ = "0.0.45" 


#with open("README.rst") as readme_file:
 #   readme = readme_file.read()

#with open("HISTORY.rst") as history_file:
 #   history = history_file.read()10
requirements = ["numpy>=1.22", "opencv-python==4.5.3.56", "pillow==9.0.1",
    "xlsxwriter", "read-roi", "imgaug==0.4.0", "matplotlib==3.6", "PyYAML==6.0", "scikit-image", "scikit_learn==1.1.1", "scipy==1.8.0",
     "tqdm==4.64.0", "pandas", "ipywidgets", "SimpleITK"]
    # "torch==1.8.1", "torchvision==0.9.1",
    #"http://gitlab.grade-rc.de/rc-weinz/heyex_tools.git",
    #"git+http://gitlab.grade-rc.de/rc-weinz/macustarpredictor.git"

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest>=3"]


setup(
    author=__author__,
    author_email=__email__,
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers"
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"],
    description="The Python package to analyse the relative elipsoid zone intensity (relEZI) by oct imaging provided by HEYEX Software",
    install_requires=requirements,
    license="MIT license",
    #long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords=["relEZIquantification"],
    name="relEZIquantification",
    packages=find_packages(include=["relEZIquantification", "relEZIquantification.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/bisselma/relEZIquantification",
    version=__version__,
    zip_safe=False,
)
