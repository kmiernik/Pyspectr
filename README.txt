===========
Pyspectr
===========

Pyspectr provides nuclear spectroscopy tools, specifically targeted
to use with his/drr histogram files from upak library. Apart from reading
the binary input files it provides some loosely bound tools like half-life
fitting, peak-fitting and pydamm program, mimicing the damm program from upak.
Typical usage often looks like this::

    #!/usr/bin/env python

    from Pyspectr import hisfile
