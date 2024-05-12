from setuptools import setup, find_packages

setup(
    name='TrendMaster',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'transformers'
    ],
    python_requires='>=3.6',
    author='Hemang Joshi',
    author_email='hemangjoshi37a@gmail.com',
    description='Stock Price Prediction using Transformer Deep Learning Architecture',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    entry_points={
        'console_scripts': [
            'trendmaster = trendmaster:main',
        ],
    },
)