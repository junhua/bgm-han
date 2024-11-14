from setuptools import setup, find_packages

setup(
    name='bgmhan',
    version='0.1.0',
    description='BGM-HAN: A PyTorch implementation of the BGM-HAN model',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'torch',
        'transformers',
        'scikit-learn',
        'seaborn',
        'matplotlib',
        'tqdm',
        'sentencepiece'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)