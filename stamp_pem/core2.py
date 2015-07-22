from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict
from gwpy.spectrum import Spectrum
from glue import datafind
from gwpy.spectrogram import Spectrogram
import numpy as np
from astropy import units as u


def fftgram(timeseries, stride, pad=False):
    """
    calculates fourier-gram with automatic
    50% overlapping hann windowing.
    Parameters
    ----------
        timeseries : gwpy TimeSeries
            time series to take fftgram of
        stride : `int`
            number of seconds in single PSD
    Returns
    -------
        fftgram : gwpy spectrogram
            a fourier-gram
    """

    fftlength = stride
    dt = stride
    df = 1. / fftlength
    # number of values in a step
    stride *= timeseries.sample_rate.value
    # number of steps
    nsteps = 2 * int(timeseries.size // stride) - 1
    # only get positive frequencies
    if pad:
        nfreqs = int(fftlength * timeseries.sample_rate.value)
    else:
        nfreqs = int(fftlength * timeseries.sample_rate.value) / 2
    dtype = np.complex
    # initialize the spectrogram
    out = Spectrogram(np.zeros((nsteps, nfreqs), dtype=dtype),
                      name=timeseries.name, epoch=timeseries.epoch,
                      f0=df / 2, df=df / 2, dt=dt, copy=True,
                      unit=1 / u.Hz**0.5, dtype=dtype)
    # stride through TimeSeries, recording FFTs as columns of Spectrogram
    for step in range(nsteps):
        # indexes for this step
        idx = (stride / 2) * step
        idx_end = idx + stride
        stepseries = timeseries[idx:idx_end]
        # zeropad, window, fft, shift zero to center, normalize
        if pad:
            stepseries.pad(stepseries.size)
        # window
        stepseries = np.multiply(stepseries,
                                 np.hanning(stepseries.value.size))
        # take fft
        tempfft = stepseries.fft(stepseries.size)
        tempfft.override_unit(out.unit)

        out[step] = tempfft[1:]
    return out


def psdgram(timeseries, stride, adjacent=1):
    """
    calculates PSD from timeseries
    properly using welch's method by averaging adjacent non-ovlping
    segments. Since default fftgram overlaps segments
    we have to be careful here...
    Parameters
    ----------
        fftgram : Spectrogram object
            complex fftgram
        adjacent : `int`
            number of adjacent segments
            to calculate PSD of middle segment
    Returns
    -------
        psdgram : Spectrogram object
            psd spectrogram calculated in
            spirit of STAMP/stochastic
    """
    fftlength = stride
    dt = stride
    df = 1. / fftlength
    # number of values in a step
    stride *= timeseries.sample_rate.value
    # number of steps
    nsteps = 2 * int(timeseries.size // stride) - 1
    # only get positive frequencies
    nfreqs = int(fftlength * timeseries.sample_rate.value) / 2.
    # initialize the spectrogram
    if timeseries.unit:
        unit = timeseries.unit / u.Hz
    else:
        unit = 1 / u.Hz
    out = Spectrogram(np.zeros((nsteps, int(nfreqs))),
                      name=timeseries.name, epoch=timeseries.epoch,
                      f0=df / 2, df=df, dt=dt, copy=True,
                      unit=unit)
    # stride through TimeSeries, recording FFTs as columns of Spectrogram
    for step in range(nsteps):
        # indexes for this step
        idx = (stride / 2) * step
        idx_end = idx + stride
        stepseries = timeseries[idx:idx_end]
        steppsd = stepseries.psd()[1:]
        out.value[step, :] = steppsd.value

    psdleft = np.hstack((out.T, np.zeros((out.shape[1], 4))))
    psdright = np.hstack((np.zeros((out.shape[1], 4)), out.T))
    # psd we want is average of adjacent, non-ovlped segs. don't include
    # middle segment for now. throw away edges.
    psd_temp = ((psdleft + psdright) / 2).T
    psd = Spectrogram(psd_temp.value[4:-4],
                  name=timeseries.name, epoch=timeseries.epoch.value+2*dt,
                  f0=df, df=df, dt=dt, copy=True,
                  unit=unit)
    return psd
