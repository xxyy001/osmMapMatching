import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="smopy",
    version='0.0.7',
    author="Cyrille Rossant",
    author_email="rossant@users.noreply.github.com",
    description="OpenStreetMap image tiles in Python",
    license="BSD",
    keywords="openstreetmap matplotlib map maps ipython",
    url="https://github.com/rossant/smopy",
    py_modules=['smopy'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Framework :: IPython",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=['numpy', 'ipython', 'pillow', 'six', 'matplotlib'],
)
