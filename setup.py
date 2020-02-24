
from setuptools import setup, find_packages

setup(
    name="photobooth",
    version="0.0.1",
    description="Photo Booth is a repository for Image Restauration (SR, Colorization)",
    license="Apache2",
    author="Bernardo Lourenço",
    author_email="bernardo.lourenco@ua.pt",
    python_requires=">=3.6.0",
    url="https://github.com/bernardomig/photobooth",
    packages=find_packages(exclude=["tests", ".tests", "tests_*"]),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
