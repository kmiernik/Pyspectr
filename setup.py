from distutils.core import setup

setup(
    name='Pyspectr',
    version='0.1.0',
    author='K. Miernik',
    author_email='kamiernik@gmail.com',
    packages=['Pyspectr'],
    scripts=['bin/pydamm.py','bin/py_grow_decay.py'],
    license='LICENSE.txt',
    description='Useful spectroscopic tools',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy",
        "matplotlib",
        "lmfit"
    ],
)
