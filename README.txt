===========
Pyspectr
===========

Pyspectr provides nuclear spectroscopy tools, specifically targeted
to use with the his/drr histogram files used by the upak library. Apart from
reading the binary input files, it provides some tools like half-life fitting,
peak-fitting and pydamm program, mimicking the DAMM program from the upak.
Users are encouraged to use pydamm within ipython3 shell, as it offers a great
advantages over standard python3 shell, such as tab-completion with history
search, input/outputs registry (as in Maxima or Mathematica), etc.  However, it
is also possible to work with pydamm with the standard python3 shell.


Instalation
===========
This package requires the following modules:
* numpy (http://www.numpy.org/)
* matplotlib (http://matplotlib.org/, 
              https://github.com/matplotlib/matplotlib)
* lmfit (https://github.com/newville/lmfit-py,
         http://cars9.uchicago.edu/software/python/lmfit/)

It is also recommended to install the ipython shell:
* ipython (http://ipython.org/)
However, the standard python shell will also work.

In a typical Linux distrubution the numpy, matplotlib and ipython should be
incuded in the package manager repositories (note that python3 version is
needed). If they are missing the github repositories include information about
the building and installation procedure (it is very simple). The lmfit library
on the github includes the standard pythons distutils setup script and is also
very easy to install.

Ones the required libraries are in place, install the Pyspectr with:
    python3 setup.py build
    sudo python3 setup.py install

Usage
=====

pydamm
------
Pydamm is a replacement for DAMM, so a typical session starts with importing
the pydamm module:
>>> from Pyspectr.pydamm import *

The main class for the data analysis is the Experiment, it requires a file
name (.his) to be given in the constructor:
>>> e = Experiment('data_file.his')
or tar gzipped file (.tgz, .tar.gz):
>>> e = Experiment('data_file.tgz')

Ones the Experiment object is created follow DAMM-like syntax to display
and analyze the data:
>>> e.d(100)
>>> e.dl(0, 1000)
>>> e.gx(1000, (212, 214))
...

However, there are some useful things that the DAMM couldn't do easyli. Check
functions like Experiment.show_registry(), Experiment.gamma_gamma_spectra(),
Experiment.fit_decay(), ...

Finally remember about the python's build-in help(), that should allow you to
investigate the avaiable variables and methods. While the documentation is
far from being perfect, at least it should give you a hint about possibilites.

spectrum_fitter
---------------

This script fits the peaks in a .his or .txt spectrum file. The peak function
include skewed Gaussian and other more sophistacated models. The fit
configuration is done via XML config file, see spectrum_fitter_example.xml


py_grow_decay
-------------

This script fits the grow-in/decay pattern, typical in the experiments with the
Moving Tape Collector. Available models include 1st and 2nd isotope in the
chain, isomeric decay, diffusion corrected decay and more. See
grow_decay_example.xml for XML config file structure.


