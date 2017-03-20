===========
Pyspectr
===========

Pyspectr provides nuclear spectroscopy tools, specifically targeted to use with
the his/drr histogram files used by the upak library. Apart from reading the
binary input files, it provides tools like half-life fitting, peak-fitting and
a pydamm program, mimicking the DAMM program from the upak library.  Users are
encouraged to use pydamm within ipython3 shell, as it offers a great advantages
over standard python3 shell, such as tab-completion with history search,
input/outputs registry (as in Maxima or Mathematica), etc.  However, it is also
possible to work with pydamm with the standard python3 shell.


Instalation
============

This package requires the following modules:
* numpy (http://www.numpy.org/)
* matplotlib (http://matplotlib.org/, 
              https://github.com/matplotlib/matplotlib)
* lmfit (https://github.com/newville/lmfit-py,
         http://cars9.uchicago.edu/software/python/lmfit/)
  Note: if github version doesn't work see the second link

It is also recommended to install the ipython shell:
* ipython (http://ipython.org/)
However, the standard python shell will also work.

In a typical Linux distribution the numpy, matplotlib and ipython should be
included in the package manager repositories (note that python3 version is
needed). If they are missing, the github repositories include information about
the building and installation procedure. The lmfit library on the github
includes the standard pythons distutils setup script. 

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

NOTE: If you are using ipython 5.0.0, it is currently affected by a bug causing
plot window to be frozen. In order to avoid that you should type:
>>> matplotlib tk
before loading an experiment file.

The main class for the data analysis is the Experiment, it requires a file
name (.his) to be given in the constructor:

>>> e = Experiment('data_file.his')

or tar gzipped file (.tgz, .tar.gz):

>>> e = Experiment('data_file.tgz')

When the Experiment object is created follow DAMM-like syntax to display
and analyze the data. The most common operations are listed below

* Display a 1D histogram or histograms
>>> e.d(100)
>>> e.d(100, 101)
>>> e.d('100-110')

* Display a 2D histogram use
>>> e.dd(1000)

* Check list of available histograms
>>> e.list()

* Learn details about selected histogram
>>> e.list(100)

* Search for histogram title that contain a keyword, or keywords (logical and)
>>> e.list('gamma')
>>> e.list('gamma', 'beta')

* Change X scale of a 1D histogram
>>> e.dl(0, 1000)

* Use automatic X scale
>>> e.dl()

* Change Y scale of a 1D histogram
>>> e.dmm(0, 100)

* Use automatic Y scale
>>> e.dmm()

* Change Y scale to logarithmic
>>> e.log()

* Change Y scale to linear
>>> e.lin()

*  Change X and Y scale of a 2D histogram
>>> e.xc(0, 100)
>>> e.yc(0, 100)

* Make projection of a 2D histogram on a Y axis, with gate on X axis
>>> e.gx(1000, (200, 203))

* Make projection of a 2D histogram on a Y axis, with gate on X axis, and
subtract a background gate
>>> e.gx(1000, (200, 203), bg_gate=(205, 207))

* Projections on X axis, with gates on Y axis
>>> e.gy(1000, (200, 205))

* Add peak or peaks for fitting
>>> e.pk(202)
>>> e.pk(202, 210, 250)

* Fit peaks in current view
>>> e.fit_peaks()

* Fit peaks in histogram 100, in range [10, 210]
>>> e.fit_peaks(100, [10, 210])

* Use a cursor mode to print the position of some selected points in the 
spectrum. Use 3rd button to quit this mode.
>>> e.c()

There are also some useful things that the DAMM couldn't do easily. Check
functions like show_registry(), gamma_gamma_spectra(), fit_decay().

Finally, remember about the python's build-in help() (also the ? in ipython),
that should allow you to investigate the available variables and methods. While
the documentation is far from being perfect, at least it should give you a hint
about possibilities.

Histogram manipulations
~~~~~~~~~~~~~~~~~~~~~~~

The plotting functions return Plot objects (or list of Plots). Each Plot
object contains bin size, normalization and a Histogram object. Histogram
object contains X and Y axis bins positions and a 'weights' table with numbers
of counts per bin (a numpy array). This can be used to manipulate histrogram.

Below is an example of gamma-gamma analysis with a background substraction. The
background is taken in two separate regions that togheter have the same width
as the gate set on the transition.

Suppose that spectrum 2660 is a gamma-gamma matrix. The line is in channels
305 to 308, the background in 302-302 and 310-311

>>> line = e.gy(2660, (305, 308))
>>> bg1 = e.gy(2660, (302, 303))
>>> bg2 = e.gy(2660, (310, 311))
>>> line.histogram.weights -= (bg1.histogram.weights + bg2.histogram.weights)

Now the "line" object is a background subtracted spectrum that can be plotted

>>> e.d(line)

Notice that access to 'weights' allow also to apply normalizations 
calculated in selected regions and other more sophisticated operations.

spectrum_fitter
---------------

This script fits the peaks in a .his or .txt spectrum file. The peak function
include the Gaussian function, skewed Gaussian and more. The fit
configuration is done via XML config file, see spectrum_fitter_example.xml


py_grow_decay
-------------

This script fits the grow-in/decay pattern, typical in the experiments with the
Moving Tape Collector. Available models include 1st and 2nd isotope in the
chain, isomeric decay, diffusion corrected decay and more. See
grow_decay_example.xml for XML config file structure.


