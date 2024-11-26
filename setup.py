from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="trendmaster",
    version="0.2.3",
    author="Hemang Joshi",
    author_email="hemangjoshi37a@gmail.com",
    description="Stock Price Prediction using Transformer Deep Learning Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hemangjoshi37a/TrendMaster",
    packages=find_packages(exclude=["Training","Inference"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "matplotlib",
        "tqdm",
        "joblib",
        "scikit-learn",
        "jugaad-trader",
    ],
    entry_points={
        "console_scripts": [
            "trendmaster=trendmaster.cli:main",
        ],
    },
)
