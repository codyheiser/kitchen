# -*- coding: utf-8 -*-
"""
package setup

@author: C Heiser
"""
import sys
import os
import io
import setuptools
from setuptools import setup


def read(fname):
    with io.open(
        os.path.join(os.path.dirname(__file__), fname), encoding="utf-8"
    ) as _in:
        return _in.read()


if __name__ == "__main__":
    import versioneer

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setup(
        name="kitchen",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        description="Manipulate counts matrix files and cook scRNA-seq data from command line",
        long_description=long_description,
        author="Cody Heiser",
        author_email="codyheiser49@gmail.com",
        url="https://github.com/codyheiser/kitchen",
        install_requires=read("requirements.txt").splitlines(),
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
        ],
        python_requires=">=3.6",
        entry_points={
            "console_scripts": ["kitchen = kitchen.kitchen:main"]
        },
    )
