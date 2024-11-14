from setuptools import setup, find_packages

setup(
    name='bgmhan',
    version='0.1.0',
    description='BGM-HAN: A PyTorch implementation of the BGM-HAN model for text classification',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Junhua Liu',
    author_email='j@forth.ai',
    packages=find_packages(),
    url='https://github.com/junhua/bgmhan',
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
    python_requires='>3.6',
)