#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

from setuptools import setup

requires = [
    'vizdoom',
    'gym',
    'numpy',
]

setup(
    name='vizdoomgym_duel',
    version='1.4',
    install_requires=requires,
    packages=['vizdoomgym_duel'],
    package_data={'vizdoomgym_duel': ['scenarios/*']},
    include_package_data=True,
)
