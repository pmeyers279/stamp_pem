from gwpy.timeseries import TimeSeries
from gwpy.spectrum import Spectrum
from gwpy.spectrogram import Spectrogram
import numpy as np
from astropy import units as u


def fftgram(timeseries, stride, pad=False, deltaF=None):
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
    df = 1. / stride
    dt = stride
    # number of values in a step
    stride *= timeseries.sample_rate.value
    # number of steps
    nsteps = 2 * int(timeseries.size // stride) - 1
    # only get positive frequencies
    if pad:
        nfreqs = int(1. / df * timeseries.sample_rate.value)
    else:
        nfreqs = int(1. / df * timeseries.sample_rate.value) / 2
    dtype = np.complex
    # initialize the spectrogram
    out = Spectrogram(np.zeros((nsteps, nfreqs), dtype=dtype),
                      name=timeseries.name, epoch=timeseries.epoch,
                      f0=df / 2., df=df / 2., dt=dt, copy=True,
                      unit=1 / u.Hz**0.5, dtype=dtype)
    # stride through TimeSeries, recording FFTs as columns of Spectrogram
    for step in range(nsteps):
        # indexes for this step
        idx = (stride / 2) * step
        idx_end = idx + stride
        stepseries = timeseries[idx:idx_end]
        # zeropad, window, fft, shift zero to center, normalize
        # window
        stepseries = np.multiply(stepseries,
                                 np.hanning(stepseries.value.size))
        # take fft
        if pad:
            stepseries = TimeSeries(np.hstack((stepseries,
                                               np.zeros(stepseries.size))),
                                    name=stepseries.name, x0=stepseries.x0,
                                    dx=timeseries.dx)
            tempfft = stepseries.fft(stepseries.size)
        else:
            tempfft = stepseries.fft(stepseries.size)
        tempfft.override_unit(out.unit)
        # get rid of dc part.
        tempfft = tempfft[1:]

        out[step] = tempfft

    return out


def psdgram(timeseries, stride, adjacent=1, deltaF=None):
    """
    calculates one-sided PSD from timeseries
    properly using welch's method by averaging adjacent non-ovlping
    segments. Since default fftgram overlaps segments
    we have to be careful here...
    Parameters
    ----------
        timeseries : timeseries object
            timeseries to create pdsgram
        adjacent : `int`
            number of adjacent segments
            to calculate PSD of middle segment
    Returns
    -------
        psdgram : Spectrogram object
            psd spectrogram calculated in
            spirit of STAMP/stochastic
    """
    dt = stride
    if not deltaF:
        deltaF = 1. / stride
    min_df = 1. / stride
    df = deltaF
    # number of values in a step
    stride *= timeseries.sample_rate.value
    # number of steps
    nsteps = 2 * int(timeseries.size // stride) - 1
    # only get positive frequencies
    if not (min_df == deltaF):
        nfreqs = int(1. / deltaF * timeseries.sample_rate.value) / 2 - 1.
    else:
        nfreqs = int(1. / deltaF * timeseries.sample_rate.value) / 2
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
        if not steppsd.df.value == deltaF:
            steppsd = coarseGrain(steppsd, deltaF, deltaF, int(nfreqs))
        out.value[step, :] = steppsd.value

    psdleft = np.hstack((out.T, np.zeros((out.shape[1], 4))))
    psdright = np.hstack((np.zeros((out.shape[1], 4)), out.T))
    psd_temp = ((psdleft + psdright) / 2).T
    # create spectrogram object. multiply by 2 for one-sided.
    psd = Spectrogram(psd_temp.value[4:-4],
                      name=timeseries.name, epoch=timeseries.epoch.value +
                      2 * dt,
                      f0=df, df=df, dt=dt, copy=True,
                      unit=unit)
    return psd


def window_factors(window1, window2):
    """
    calculates window factors
    necessary for stamp analysis
    Parameters
    ----------
        window1 : array
            window used for PSDs
    Returns
    -------
        w1w2bar : array
            average of product of windows
        w1w2squaredbar : array
            average of product of squares of windows
        w1w2ovlsquaredbar : array
            average of product of first and second
            halves of window1 and window2.
    """

    Nred = window1.size
    if Nred == 1:
        raise ValueError('windows are not compatible')

    N1 = window1.size
    N2 = window2.size
    # get reduced windows
    window1red = window1[0:(N1 - N1 / Nred) + 1]
    window2red = window2[0:(N2 - N2 / Nred) + 1]
    idx = int(np.floor(Nred / 2.))

    w1w2bar = np.mean(np.multiply(window1red, window2red))
    w1w2squaredbar = np.mean(
        np.multiply(np.power(window1red, 2), np.power(window2red, 2)))
    w1w2ovlsquaredbar = np.mean(np.multiply(
        np.multiply(window1red[0:idx], window2red[idx:]),
        np.multiply(window1red[idx:], window2red[0:idx])))
    return w1w2bar, w1w2squaredbar, w1w2ovlsquaredbar


def coarseGrain(spectrum, f0, df, N):
    """
    coarse grain frequency spectrum
    Parameters
    ----------
        spectrum : Spectrum object
        df : new df for new spectrum
        N : number of frequencies in resultant spectrum
    Returns
    -------
        spectrumCG : Spectrum object
            output spectrum
    """
    f0i = spectrum.f0.value
    dfi = spectrum.df.value
    Ni = spectrum.size
    freqs = np.arange(f0,spectrum.frequencies.value[-1],df)
    fhighi = f0i + (Ni - 1) * dfi
    fhigh = f0 + df * (N - 1)
    shaved = False
    if ((fhigh + 0.5 * df) > (fhighi + 0.5 * dfi)) and ((fhigh + 0.5 * df) - (fhighi + 0.5 * dfi))<df:
        N = N-1
        fhigh = f0+df*(N-1)
        shaved = True
    elif ((fhigh + 0.5 * df) > (fhighi + 0.5 * dfi)) and not ((fhigh + 0.5 * df) - (fhighi + 0.5 * dfi))<df:
        print 'we want: ' + str(fhigh + 0.5 * df)
        print 'we can get: ' + str(fhighi + 0.5 * dfi)
        raise ValueError('desired coarseGrain stop frequency too high')


    i = np.arange(0, N)

    # low/high indices for coarsegrain
    jlow = 1 + ((f0 + (i - 0.5) * df - f0i - 0.5 * f0i) / dfi)
    jhigh = 1 + ((f0 + (i + 0.5) * df - f0i - 0.5 * f0i) / dfi)
    # fractional contribution of partial bins
    fraclow = (dfi + (jlow + 0.5) * dfi - f0 - (i - 0.5) * df) / dfi
    frachigh = (df + (i + 0.5) * df - f0i - (jhigh - 0.5) * dfi) / dfi

    jtemp = jlow + 2
    y_real = np.zeros(N)
    y_imag = np.zeros(N)
    for idx in range(N):
        y_real[idx] = sum(spectrum.value.real[jtemp[idx] - 1:jhigh[idx]])
        y_imag[idx] = sum(spectrum.value.imag[jtemp[idx] - 1:jhigh[idx]])
    y = np.vectorize(complex)(y_real, y_imag)

    ya = (dfi / df) * (np.multiply(spectrum.value[jlow[:-1].astype(int) - 1],
                                   fraclow[:-1]) +
                       np.multiply(spectrum.value[jhigh[:-1].astype(int) - 1],
                                   frachigh[:-1] + y[:-1]))
    if (jhigh[N - 1] > Ni - 1):
        yb = (dfi / df) * \
            (spectrum.value[jlow[N - 1].astype(int)] * fraclow[N - 1].astype(int) + y[N - 1])
    else:
        yb = (dfi / df) * (spectrum.value[jlow[N - 1].astype(int)] * fraclow[N - 1] +
                           spectrum.value[jhigh[N - 1].astype(int)] * frachigh[N - 1] +
                           y[N - 1])
    y = np.hstack((ya, yb))
    y = Spectrum(y, df=df, f0=f0, epoch=spectrum.epoch, unit=spectrum.unit,
                 name=spectrum.name)
    return y


def csdgram(channel1, channel2, stride, deltaF=None):
    """
    calculates one-sided csd spectrogram between two timeseries
    or fftgrams. Allows for flexibility for holding DARM
    fftgram in memory while looping over others.
    Parameters
    ----------
        channel1 : TimeSeries or Spectrogram object
            timeseries from channel 1
        timeseries2 : TimeSeries or Spectrogram object
            timeseries from channel 2
    Returns
    -------
        csdgram : spectrogram object
            csd spectrogram for two objects
    """
    if isinstance(channel1, TimeSeries):
        fftgram1 = fftgram(channel1, stride, pad=True)
    elif isinstance(channel1, Spectrogram):
        fftgram1 = channel1
    else:
        raise TypeError('First arg is either TimeSeries or Spectrogram object')
    if isinstance(channel2, TimeSeries):
        fftgram2 = fftgram(channel2, stride, pad=True)
    elif isinstance(channel2, Spectrogram):
        fftgram2 = channel2
    else:
        raise TypeError('First arg is either TimeSeries or Spectrogram object')
    if not deltaF:
        deltaF = 1. / stride

    # clip off first 2 and last 2 segments to be consistent with psd
    # calculation
    out = (fftgram1.value * np.conj(fftgram2.value))[2:-2]

    csdname = 'csd spectrogram between %s and %s' % (
        fftgram1.name, fftgram2.name)
    out = Spectrogram(out, name=csdname,
                      epoch=fftgram1.epoch.value + 2 * fftgram1.dt.value,
                      df=fftgram1.df, dt=fftgram1.dt, copy=True,
                      unit=fftgram1.unit * fftgram2.unit, f0=fftgram1.f0)
    df = fftgram1.df.value * 2
    f0 = fftgram1.f0.value * 2
    csdgram = Spectrogram(np.zeros((out.shape[0], out.shape[1] / 2 - 1),
                                   dtype=np.complex), df=df,
                          dt=fftgram1.dt, copy=True, unit=out.unit, f0=f0,
                          epoch=out.epoch)

    for ii in range(csdgram.shape[0]):
        # multiply by 2 for one-sided spectrum
        temp = Spectrum(2 * out.value[ii], df=out.df,
                        f0=out.f0, epoch=out.epoch,
                        unit=out.unit)
        N = out.shape[1] / 2
        csdgram[ii] = coarseGrain(temp, df, f0, int(N))

    return csdgram


def stamp_variance(channel1, channel2, stride, deltaF=None):
    """
    calculates stamp-pem variance from two time-series.
    Parameters
    ----------
        channel1 : TimeSeries or Spectrogram object
            timeseries or PSD for channel 1
        channel2 : TimeSeries or Spectrogram object
            timeseries or PSD for channel 2
        stride : int
            fft stride
    Returns
    -------
        stamp variance : Spectrogram object
    """
    # set units
    if isinstance(channel1, TimeSeries):
        psd1 = psdgram(channel1, stride, deltaF=deltaF)
    else:
        psd1 = channel1
    if isinstance(channel2, TimeSeries):
        psd2 = psdgram(channel2, stride, deltaF=deltaF)
    else:
        psd2 = channel2
    # set units
    if psd1.unit and psd2.unit:
        var_unit = psd1.unit * psd2.unit
    else:
        var_unit = (u.Hz)**-2
    variance = Spectrogram(0.5 * psd1.value * psd2.value,
                           epoch=psd1.epoch, dt=psd1.dt,
                           copy=True, unit=var_unit, f0=psd1.f0,
                           df=psd1.df)
    return variance


def stamp_y(channel1, channel2, stride, deltaF=None):
    """
    calculates stamp statistic, Y, from two fftgrams
    or two timeseries
    Parameters
    ----------
        channel1 : TimeSeries or Spectrogram object
            timeseries or fft spectrogram for channel 1
        channel2 : TimeSeries or Spectrogram object
            timeseries or fft spectrogram for channel 2

    Returns
    -------
        y : Spectrogram object
            stamp point estimate
    """
    return 2 * np.real(csdgram(channel1, channel2, stride, deltaF=deltaF))


def stamp_snr(channel1, channel2, stride, deltaF=None):
    """
    calculates stamp snr
    Parameters
    ----------
        channel1 : TimeSeries
            channel1 timeseries
        channel2 : TimeSeries
            channel2 TimeSeries

    Returns
    -------
        snr : Spectrogram
            stamp snr spectrogram
    """
    # calculate variance
    variance = stamp_variance(channel1, channel2, stride, deltaF=deltaF)

    # calculate y
    y = stamp_y(channel1, channel2, stride, deltaF=deltaF)
    y_final = Spectrogram(np.zeros((variance.shape[0],variance.shape[1])), df=variance.df, f0=variance.f0,
                    epoch=variance.epoch, dt=variance.dt, unit=y.unit)
    if not deltaF == 1. / stride:
        for ii in range(y.shape[0]):
            # multiply by 2 for one-sided spectrum
            temp = Spectrum(2 * y.value[ii], df=y.df,
                            f0=y.f0, epoch=y.epoch,
                            unit=y.unit)
            temp = coarseGrain(temp, variance.df.value, variance.f0.value, variance[1].size)
            y_final[ii] = temp
    else:
        y_final = y
    # calculate window factors!
    w1w2bar, w1w2squaredbar, w1w2ovlsquaredbar = window_factors(
        np.hanning(channel1.size), np.hanning(channel2.size))
    # get the multiplicative window factor
    multiplicative_factor = w1w2squaredbar / w1w2bar**2
    # get snr!
    diff = np.abs(y_final[1].size - variance[1].size)
    if diff:
        variance = variance[:,:-diff]

    snr = multiplicative_factor * Spectrogram(y_final.value / variance.value**0.5,
                                              unit=None, dt=y_final.dt, f0=y_final.f0,
                                              df=y_final.df, epoch=y_final.epoch,
                                              copy=True)
    return snr
