from setuptools import find_packages, setup

readme = open("README.md", "r")

setup(
    name="merx",
    packages=find_packages(include=["merx"]),
    requires=["pandas"],
    version="0.0.1",
    description="Python library to more easily utilize technical analysis indicators.",
    long_description=readme.read(),
    long_description_content_type='text/markdown',
    author="bigweavers",
    license="GPL-3.0"
)