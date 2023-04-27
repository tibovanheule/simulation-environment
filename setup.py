from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()
    
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='Simulation environment',
    version='0.0.1',
    description='The Simulation Environment',
    long_description=readme,
    url='https://github.com/sel3-2023-group-6/simulation-environment',
    license='license',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=required
)
