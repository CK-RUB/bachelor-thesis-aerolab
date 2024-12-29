from setuptools import setup, find_packages

setup(
    name="meaningful_complexity",
    version="0.1",
    description="A module to compute meaningful image complexity.",
    author="Louis Mahon and Thomas Lukasiewicz",
    long_description="""
    This module computes meaningful image complexity using the Minimum Description Length (MDL) principle.

    Reference:
    @article{mahon2024minimum,
        title = {Minimum description length clustering to measure meaningful image complexity},
        journal = {Pattern Recognition},
        volume = {145},
        pages = {109889},
        year = {2024},
        issn = {0031-3203},
        doi = {https://doi.org/10.1016/j.patcog.2023.109889},
        url = {https://www.sciencedirect.com/science/article/pii/S0031320323005873},
        author = {Louis Mahon and Thomas Lukasiewicz},
    }
    """,
    long_description_content_type="text/plain",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scipy",
        "matplotlib",
        "scikit-fuzzy",
        "scikit-image",
        "opencv-python",
        "dl-utils385",
    ],
)
