"""
Setup script for PyMBO - Python Multi-objective Bayesian Optimization
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pymbo",
    version="3.0.0",
    author="Jakub Jagielski",
    author_email="your.email@example.com",
    description="PyMBO - Python Multi-objective Bayesian Optimization framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/pymbo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pymbo=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pymbo": ["*.md"],
    },
)