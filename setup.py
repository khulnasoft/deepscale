from setuptools import setup, find_packages

setup(
    name='deepscale',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=1.2',
        'tqdm',
        'tensorboardX==1.8',
        'ninja',
        'numpy',
        'psutil',
        'packaging'
    ],
    extras_require={
        'dev': [
            'pytest>=3.7',
            'check-manifest',
            'flake8',
        ],
        'docs': [
            'Sphinx>=1.8.1',
            'sphinx_rtd_theme',
        ],
    },
)
