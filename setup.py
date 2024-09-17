from os import path
from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))

REQUIREMENTS_FILE = 'requirements.txt'

# read the contents of requirements.txt
with open(path.join(this_directory, REQUIREMENTS_FILE), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='rosae',
    version="0.1",
    description='Anomaly detection with robust autoencoders',
    author='',
    author_email='',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    setup_requires=['setuptools>=38.6.0'],
)