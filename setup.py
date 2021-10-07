from io import open

from setuptools import setup, find_packages
import os

with open('h5mapper/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.0.1'

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), "r", encoding="utf-8") as f:
    REQUIRES = [ln.strip() for ln in f.readlines() if ln.strip()]

PACKAGES = find_packages(exclude=('tests', 'tests.*'))

kwargs = {
    'name': 'h5mapper',
    'version': version,
    'description': 'pythonic ORM tool for reading and writing HDF5 data',
    'long_description': readme,
    "long_description_content_type": "text/markdown",
    'author': 'Antoine Daurat',
    'author_email': 'antoinedaurat@gmail.com',
    'url': 'https://github.com/ktonal/h5mapper',
    'download_url': 'https://github.com/ktonal/h5mapper',
    'license': 'MIT License',
    'classifiers': [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        "Intended Audience :: Science/Research",
        "Intended Audience :: Other Audience",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    "keywords": "hdf5 h5py ORM deep-learning machine-learning",
    'python_requires': '>=3.6',
    'install_requires': REQUIRES,
    'tests_require': ['coverage', 'pytest'],
    'packages': PACKAGES,
    "entry_points": {
        'console_scripts': [
            'sound-bank=h5mapper.typed_files:sound_bank',
            'image-bank=h5mapper.typed_files:image_bank'
        ]}

}

setup(**kwargs)
