## a simple setup.py file.

from os import path
import sys
from setuptools import setup,find_packages
from setuptools.extension import Extension
    
setup(name='san',
      version='0.1',
      description="Feature ranking with self-attention networks",
      url='http://github.com/skblaz/san',
      author='Blaž Škrlj and Matej Petković',
      author_email='blaz.skrlj@ijs.si',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True)


