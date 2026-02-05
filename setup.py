import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="automunge",
    version="9.0",
    author="Nicholas Teague",
    author_email="automunge@gmail.com",
    description="platform for preparing tabular data for machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Automunge/AutoMunge",
    packages=setuptools.find_packages(),
    python_requires=">=3.11,<4",
    install_requires=[
        "numpy>=2.0,<3",
        "pandas>=2.2.2,<3",
        "scikit-learn>=1.8,<2",
        "scipy>=1.13,<2"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)