from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict
from gwpy.spectrum import Spectrum
from glue import datafind
from gwpy.spectrogram import Spectrogram
import numpy as np
from astropy import units as u


def fftgram(timeseries, stride, overlap=None, pad=False):
    """
    calculates fourier-gram with auto 50% overlapping
    segments and hann windowed

    Parameters
    ----------
        timeseries : gwpy TimeSeries
            time series to take fftgram of
        stride : `int`
            number of seconds in single PSD
        overalp : `int`
            number of seconds of overlap
    Returns
    -------
        fftgram : gwpy spectrogram
            a fourier-gram
    """

    fftlength = stride
    if not overlap:
        overlap = stride / 2.
    dt = stride - overlap
    df = 1. / fftlength
    # number of values in a step
    stride *= timeseries.sample_rate.value
    overlap *= timeseries.sample_rate.value
    step = stride - overlap
    # number of steps
    nsteps = int(timeseries.size // step) - 1
    # only get positive frequencies
    if pad:
        nfreqs = int(fftlength * timeseries.sample_rate.value)
        df = df / 2
        f0 = df
    else:
        nfreqs = int(fftlength * timeseries.sample_rate.value) / 2
    dtype = np.complex
    # initialize the spectrogram
    out = Spectrogram(np.zeros((nsteps, nfreqs), dtype=dtype),
                      name=timeseries.name, epoch=timeseries.epoch,
                      f0=df, df=df, dt=dt, copy=True,
                      unit=1 / u.Hz**0.5, dtype=dtype)
    # stride through TimeSeries, recording FFTs as columns of Spectrogram
    for step in range(nsteps):
        # indexes for this step
        idx = (stride / 2) * step
        idx_end = idx + stride
        stepseries = timeseries[idx:idx_end]
        # zeropad, window, fft
        stepseries = np.multiply(stepseries,
                                 np.hanning(stepseries.value.size))
        # hann windowing normalization
        norm = np.sum(np.hanning(stepseries.value.size))
        if pad:
            stepseries = TimeSeries(np.hstack((stepseries, np.zeros(stepseries.size))),
                                    name=stepseries.name, x0=stepseries.x0,
                                    dx=timeseries.dx)
            tempfft = stepseries.fft(stepseries.size)
        else:
            tempfft = stepseries.fft(stepseries.size)
        # reset proper unit
        tempfft.override_unit(out.unit)
        # normalize
        out[step] = tempfft[1:] / norm

    return out


def csdgram(channel1, channel2, stride, overlap=None, pad=False):
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
        fftgram1 = fftgram(channel1, stride, pad=pad, overlap=overlap)
    elif isinstance(channel1, Spectrogram):
        fftgram1 = channel1
    else:
        raise TypeError('First arg is either TimeSeries or Spectrogram object')
    if isinstance(channel2, TimeSeries):
        fftgram2 = fftgram(channel2, stride, pad=pad, overlap=overlap)
    elif isinstance(channel2, Spectrogram):
        fftgram2 = channel2
    else:
        raise TypeError('First arg is either TimeSeries or Spectrogram object')

    # clip off first 2 and last 2 segments to be consistent with psd
    # calculation
    out = np.conj(fftgram1.value) * fftgram2.value

    if pad:
        # if zero padded, take every other frequency
        out = out[:, 0::2]
        df = fftgram1.df * 2
        f0 = fftgram1.f0 * 2
    else:
        df = fftgram1.df
        f0 = fftgram1.f0

    csdname = 'csd spectrogram between %s and %s' % (
        fftgram1.name, fftgram2.name)
    out = Spectrogram(out, name=csdname,
                      epoch=fftgram1.epoch.value,
                      df=df, dt=fftgram1.dt, copy=True,
                      unit=fftgram1.unit * fftgram2.unit, f0=f0)

    return out


def coherence(channel1, channel2, stride, overlap=None, pad=False):
    if isinstance(channel1, TimeSeries):
        fftgram1 = fftgram(channel1, stride, overlap=overlap, pad=pad)
    else:
        fftgram1 = channel1
    if isinstance(channel2, TimeSeries):
        fftgram2 = fftgram(channel2, stride, overlap=overlap, pad=pad)
    else:
        fftgram2 = channel2
    # number of segments
    N = fftgram1.shape[0]
    # max frequency we're  calculating up to
    max_f = min(fftgram1.shape[1],
                fftgram2.shape[1])
    # calculate csd
    csd12 = csdgram(fftgram1[:, 0:max_f], fftgram2[:, 0:max_f],
                    stride, overlap=None, pad=pad)
    csd12 = np.mean(csd12, 0)
    if pad:
        psd1 = np.mean(np.abs(fftgram1[:, 0::2]) ** 2, 0)
        psd2 = np.mean(np.abs(fftgram2[:, 0::2]) ** 2, 0)
    else:
        psd1 = np.mean(np.abs(fftgram1) ** 2, 0)
        psd2 = np.mean(np.abs(fftgram2) ** 2, 0)
    coherence = Spectrum(np.abs(csd12) ** 2 / (psd1[0:csd12.size]
                         * psd2[0:csd12.size]), df=fftgram1.df,
                         epoch=fftgram1.epoch, unit=None)
    return coherence, N, csd12, psd1, psd2


def coherence_spectrogram(channel1, channel2, stride, segmentDuration,
                          overlap=None, pad=False, frames=False,
                          st=None, et=None):
    """
    """
    if isinstance(channel1, TimeSeries):
        nsegs = (channel1.times.value[-1] - channel1.times.value[0]) \
            / segmentDuration
        nfreqs = min(channel1.sample_rate.value,
                     channel2.sample_rate.value) * stride * 0.5
        epoch = channel1.epoch
    elif isinstance(channel1, str):
        nsegs = int((et - st) / segmentDuration)
        test1 = _read_data(channel1, st, st + 1, frames=frames)
        test2 = _read_data(channel2, st, st + 1, frames=frames)
        nfreqs = int(min(test1.sample_rate.value,
                         test2.sample_rate.value) * (stride) * 0.5)
        epoch = test1.epoch

    df = 1. / stride

    coherence_spectrogram = Spectrogram(np.zeros((nsegs, nfreqs)),
                                        epoch=epoch, dt=segmentDuration,
                                        df=df, f0=df)
    psd1_spectrogram = Spectrogram(np.zeros((nsegs, nfreqs)),
                                   epoch=epoch, dt=segmentDuration,
                                   df=df, f0=df, unit=u.Hz**-1)
    psd2_spectrogram = Spectrogram(np.zeros((nsegs, nfreqs)),
                                   epoch=epoch, dt=segmentDuration,
                                   df=df, f0=df, unit=u.Hz**-1)
    if isinstance(channel1, str) and isinstance(channel2, str):
        for i in range(int(nsegs)):
            startTime = st + i * segmentDuration
            endTime = startTime + segmentDuration
            stepseries1 = _read_data(channel1, startTime, endTime,
                                     frames=frames)
            stepseries2 = _read_data(channel2, startTime, endTime,
                                     frames=frames)
            test, N, csd12, psd1, psd2 = coherence(stepseries1, stepseries2,
                                                   stride, overlap=None,
                                                   pad=pad)
            coherence_spectrogram[i] = test
            psd1_spectrogram[i] = psd1
            psd2_spectrogram[i] = psd2
    else:
        samples1 = segmentDuration * channel1.sample_rate.value
        samples2 = segmentDuration * channel2.sample_rate.value
        for i in range(int(nsegs)):
            idx_start1 = samples1 * i
            idx_end1 = idx_start1 + samples1
            idx_start2 = samples2 * i
            idx_end2 = idx_start2 + samples2
            stepseries1 = channel1[idx_start1:idx_end1]
            stepseries2 = channel2[idx_start2:idx_end2]
            test, N, csd12, psd1, psd2 = coherence(stepseries1, stepseries2,
                                                   stride, overlap=None,
                                                   pad=pad)
            coherence_spectrogram[i] = test
            psd1_spectrogram[i] = psd1
            psd2_spectrogram[i] = psd2

    return coherence_spectrogram, psd1_spectrogram, psd2_spectrogram, N


def _read_data(channel, st, et, frames=False):
    """
    get data, either from frames or from nds2
    """

    ifo = channel.split(':')[0]
    if frames:
        # read from frames
        connection = datafind.GWDataFindHTTPConnection()
        cache = connection.find_frame_urls(ifo[0], ifo + '_R', st, et)
        data = TimeSeries.read(cache, channel, st, et)
    else:
        data = TimeSeries.fetch(channel, st, et)

    return data


def coherence_list(channel1, channels, stride, st=None, et=None,
                   overlap=None, pad=False, frames=False):
    coh = {}
    psd1 = {}
    psd2 = {}
    csd12 = {}
    darm = _read_data(channel1, st, et)
    darm_fft = fftgram(darm, stride, pad=pad)
    for channel in channels:
        data = _read_data(channel, st, et)
        coh[channel], N, csd12[channel], psd1[channel], psd2[channel] = \
            coherence(darm_fft, data, stride, overlap=overlap, pad=pad)

    return coh, N, csd12, psd1, psd2
