"""
Setup script for NLTK Bigram Analyzer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nltk-bigram-analyzer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A powerful toolkit for text tokenization and bigram semantic analysis using NLTK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nltk-bigram-analyzer",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "nltk>=3.8.1",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "matplotlib>=3.7.2",
        "scikit-learn>=1.3.0",
        "wordcloud>=1.9.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bigram-analyze=examples.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["data/*.txt"],
    },
)
