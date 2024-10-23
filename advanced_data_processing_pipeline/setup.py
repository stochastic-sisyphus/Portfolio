from setuptools import setup, find_packages

setup(
    name="advanced_data_processing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "dask",
        "dask-ml",
        "scikit-learn",
        "nltk",
        "spacy",
        "gensim",
        "matplotlib",
        "seaborn",
        "imbalanced-learn",
        # Add any other dependencies from your requirements.txt
    ],
    entry_points={
        'console_scripts': [
            'adp=data_processing.main:main',
        ],
    },
    author="Vanessa Beck",
    description="An advanced data processing pipeline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/stochastic-sisyphus/Portfolio/tree/main/adv_data_processing_pipeline",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

