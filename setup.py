#!/usr/bin/env python

from setuptools import (
    setup as install,
    find_packages,
)

VERSION = '0.1.0'

install(
    name='hierarchical_envs',
    version=VERSION,
    description="Hierarchical Environments for RL",
    long_description=open('README.md').read(),
    author='Liyu Chen',
    author_email='na',
    url='http://github.com/lchenat',
    download_url='',
    license='License :: OSI Approved :: Apache Software License',
    packages=find_packages(exclude=["tests"]),
    classifiers=[]
)
