import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cooperative-transport",
    version="0.0.1",
    author="Eley Ng, Ziang Liu",
    author_email="eleyng@stanford.edu",
    description="Custom environment and model for training RL agents for cooperative transport.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eleyng/table-carrying-ai",
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
        "opencv-python",
    ],
)
