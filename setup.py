# #!/usr/bin/env python3
# import io
# import os
# import re

# from setuptools import setup

# requirements = []

# # Read the version from the __init__.py file without importing it
# def read(*names, **kwargs):
#     with io.open(
#             os.path.join(os.path.dirname(__file__), *names),
#             encoding=kwargs.get("encoding", "utf8")
#     ) as fp:
#         return fp.read()


# def find_version(*file_paths):
#     version_file = read(*file_paths)
#     version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
#                               version_file, re.M)
#     if version_match:
#         return version_match.group(1)
#     raise RuntimeError("Unable to find version string.")


# from setuptools import find_packages, setup


# here = os.path.abspath(os.path.dirname(__file__))

# # Get the long description from the README file
# with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()

# # Arguments marked as "Required" below must be included for upload to PyPI.
# # Fields marked as "Optional" may be commented out.
# setup(
#     name='power-gridworld',
#     version=find_version("gridworld", "__init__.py"),
#     description='A framework for multi-agent reinforcement learning for power systems',
#     long_description=long_description,
#     long_description_content_type='text/markdown',
#     url='https://github.com/NREL/PowerGridworld',  # Optional
#     author='David Biagioni',
#     author_email='dave.biagioni@nrel.gov',  # Optional
#     classifiers=[
#         # Development status of the project
#         'Development Status :: 3 - Alpha',
#         # Indicate who your project is intended for
#         'Intended Audience :: Science/Research',
#         # Pick your license as you wish
#         'License :: OSI Approved :: BSD 3-Clause',
#         # Specify the Python versions you support here. In particular, ensure
#         # that you indicate whether you support Python 2, Python 3 or both.
#         'Programming Language :: Python :: 3',
#     ],
#     packages=find_packages(),  # Required
#     install_requires=[],  # TODO
#     project_urls={
#         'Source': 'https://github.com/NREL/PowerGridworld',
#     },
# )


import io
import os
import re

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requirements = []


# Read the version from the __init__.py file without importing it
def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='power-gridworld',
    version=find_version("gridworld", "__init__.py"),
    description='A package for running AUMC "lite" as a gym environment',
    author='Dave Biagioni',
    author_email='dave.biagioni@nrel.gov',
    url='https://github.nrel.gov/dbiagion/aumc-gridworld/',
    packages=['gridworld'],
    install_requires=requirements,
)
