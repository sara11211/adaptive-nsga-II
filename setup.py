from setuptools import find_packages, setup

setup(
    name="nsga2",
    version="1.0.0",
    description="NSGA-II: A Fast and Elitist Multiobjective Genetic Algorithm (Deb et al., 2002)",
    author="",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
    ],
)
