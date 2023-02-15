import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cooperative-transport",
    version="0.0.1",
    author="Eley Ng, Albert Li",
    author_email="eleyng@stanford.edu, ahli@stanford.edu",
    description="Custom environment and model for training RL agents for cooperative transport.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eley-ng/cooperative-transport",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    license="MIT",
    install_requires=[
        "gym",
        "matplotlib",
        "numpy",
        "pillow",
        "pygame",
        "stable-baselines3",
        "tensorboard",
        "torch",
    ],
)
