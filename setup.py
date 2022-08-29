"""Package Setup instructions."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme_content = readme_file.read()

requirements = [
    "jax",
    "jaxlib",
    "numpy",
    "tqdm"
]

dev_requirements = [
    "matplotlib",
    "ipykernel",
    "jupyter",
    "sklearn",
    "pandas"
]

setup(
    name="cgc",
    description="CGC: Computational Graph Completion",
    author="Mostafa Samir",
    author_email="mostafa.3210@gmail.com",
    python_requires=">=3.8",
    classifiers=[
        "Private :: Do Not Upload",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License"
    ],
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
    license="MIT",
    long_description_content_type="text/markdown",
    long_description=readme_content,
    keywords=["matrix completion", "computational graph", "gaussian process", "regression"],
    packages=find_packages(include=["cgc", "cgc.*"]),
    url="https://github.com/Mostafa-Samir/CGC",
    version="0.1.0"
)