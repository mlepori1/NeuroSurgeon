from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    README = fh.read()

setup(
    name="NeuroSurgeon",
    version="0.1.0",
    description="A toolkit for subnetwork analysis",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/mlepori1/NeuroSurgeon",
    author="Michael Lepori",
    author_email="michael_lepori@brown.edu",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.1",
        "transformers",
        "matplotlib",
    ],
    extras_require={
        "tests": ["pytest", "datasets"],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.9",
    ],
)
