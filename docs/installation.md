# Installation

Follow these steps to install TrendMaster and its dependencies.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installing TrendMaster

You can install TrendMaster using pip:

```bash
pip install trendmaster
```

This will install TrendMaster and all its required dependencies.

## Installing from source

If you want to install the latest development version, you can install directly from the GitHub repository:

```bash
git clone https://github.com/hemangjoshi37a/TrendMaster.git
cd TrendMaster
pip install -e .
```

## Verifying the installation

After installation, you can verify that TrendMaster is installed correctly by running:

```python
import trendmaster
print(trendmaster.__version__)
```

This should print the version number of TrendMaster.

## Next steps

Now that you have TrendMaster installed, check out the [Quick Start](quickstart.md) guide to begin using the library!