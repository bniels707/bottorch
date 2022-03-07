from setuptools import find_packages, setup

setup(
    name="bottorch",
    version="0.0.1",
    description="Crude fighting robot bracketology",
    long_description="Applies primitive machine learning to assist with populating fighting robot brackets",
    author="Brandon Nielsen",
    author_email="bniels707@gmail.com",
    url="https://github.com/bniels707/bottorch/",
    entry_points={
        "console_scripts": [
            "bottorch = bottorch.bottorch:main",
        ]
    },
    python_requires=">=3.7",
    install_requires=["torch", "torchvision", "pandas", "numpy"],
    packages=find_packages(),
    include_package_data=True
)
