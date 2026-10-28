PSD
===

PSD compares spatial power spectra. Regular latitude/longitude fields use the
a two-dimensional FFT plus radial averaging; native HEALPix fields use
``healpy.anafast`` after conversion to ring ordering. Configure
``psd_calculations`` to select time-averaged spectra and/or the spectrum of the
time-mean field. Run ``python -m aqua.diagnostics.emulator.psd.cli_psd --config CONFIG``.
