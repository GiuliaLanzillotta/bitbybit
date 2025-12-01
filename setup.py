from setuptools import setup, find_packages

setup(
    name='bitbybit',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'avalanche-lib',   # Continual Learning library
        'fvcore',
        'matplotlib',
        'numpy',
        'pandas',
        'Pillow',
        'python-dotenv',
        'pytorch_warmup',
        'seaborn',
        'torch',
        'torchvision',
        'wandb',
    ]
    # Metadata
    author='Giulia Lanzillotta',
)