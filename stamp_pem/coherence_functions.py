from gwpy.timeseries import TimeSeries
from gwpy.spectrum import Spectrum
from glue import datafind
from gwpy.spectrogram import Spectrogram
import numpy as np
import os
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import coh_io
import inspect,dis
from gwpy.segments import Segment




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
    if (timeseries.sample_rate.value * stride) % 1:
        raise ValueError('1 / stride must evenly divide sample rate of channel')
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


def coherence(channel1, channel2, stride, overlap=None, pad=False,
              segmentDuration=None):
    """
    calculates coherence from two timeseries or fft f-t maps
    depending on requested # of output variables, it returns different things.

    2 : coherence, number of segments averaged
    3 : coherence, number of segments averaged, coherence spectrogram
    5 : coherence, csd, psd1, psd2, number of segments averaged
    6 : coherence, csd, psd1, psd2, number of segments averaged,
        coherence spectrogram


    Parameters:
    -----------
      channel1 : TimeSeries or Spectrogram object
          Either time series for first channel or ffts for that channel.
      channel2 : TimeSeries or Spectrogram object
          Either time series for second channel or ffts for that channel.
      stride : int
          length of ffts to take in seconds
      overlap : int (optional)
          amount of overlap between ffts in seconds (default 0.5*stride)
      pad : bool (optional)
          decide whether or not to pad before taking ffts or
          used to indicate if input ffts were padded
      segmentDuration : float (optional)
          length of segment for creating coherence spectrogram. if 

    Returns:
    --------
      coherence : Spectrum object
          coherence between channel 1 and channel 2
      csd12 : Spectrum object
          cross spectral density between channel 1 and channel 2
      psd1 : Spectrum object
          power spectral density of channel 1
      psd2 : Spectrum object
          power spectral density of channel 2
      N : int
          number of averages done to get coherence spectrum
      coh_spectrogram : Spectrogram object
          coherence spectrogram with time segments of lengths segmentDuration

    """
    nargout = expecting()

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
                    stride, overlap=overlap, pad=pad)
    if nargout==6 or 3:
        coh_spectrogram = _coherence_spectrogram(fftgram1, fftgram2, stride, segmentDuration, pad=pad)
    csd12 = np.mean(csd12, 0)
    if pad:
        psd1 = np.mean(np.abs(fftgram1[:, 0::2]) ** 2, 0)
        psd2 = np.mean(np.abs(fftgram2[:, 0::2]) ** 2, 0)
    else:
        psd1 = np.mean(np.abs(fftgram1) ** 2, 0)
        psd2 = np.mean(np.abs(fftgram2) ** 2, 0)
    coh_name = 'coherence'
    coherence = Spectrum(np.abs(csd12) ** 2 / (psd1[0:csd12.size]
                                               * psd2[0:csd12.size]),
                         df=csd12.df,
                         epoch=csd12.epoch, unit=None, name=coh_name)
    if nargout==1:
        raise ValueError('coherence outputs 2, 3, 5 or 6 values. Not 1 or 4')
    if nargout==2:
        return coherence, N
    if nargout==3:
        return coherence, N, coh_spectrogram
    if nargout==4:
        raise ValueError('coherence outputs 2, 3, 5 or 6 values. Not 1 or 4')
    if nargout==5:
        return coherence, csd12, psd1, psd2, N
    if nargout==6:
        return coherence, csd12, psd1, psd2, N, coh_spectrogram

def _coherence_spectrogram(fftgram1, fftgram2, stride, segmentDuration, pad=False):
    max_f = min(fftgram1.shape[1], fftgram2.shape[1])
    csd12 = csdgram(fftgram1[:, 0:max_f], fftgram2[:, 0:max_f],
                    stride, pad=pad)
    if pad:
        psd1 = np.abs(fftgram1[:, 0::2]) ** 2
        psd2 = np.abs(fftgram2[:, 0::2]) ** 2
    else:
        psd1 = np.abs(fftgram1) ** 2
        psd2 = np.abs(fftgram2) ** 2

    # average over strides if necessary
    if not segmentDuration:
        segmentDuration = stride
    if segmentDuration < stride:
        raise ValueError('segmentDuration must be longer than or equal to stride')
    if segmentDuration % stride:
        raise ValueError('stride must evenly divide segmentDuration')

    navgs = segmentDuration / stride
    if not fftgram1.shape[0] % navgs:
        print 'WARNING: will cut off last segment because segmentDuration doesnt evenly divide total time window'
    nsegs = int(fftgram1.shape[0] / navgs)
    nfreqs = csd12.frequencies.size
    coh_spec = np.zeros((nsegs, nfreqs))
    for i in range(nsegs):
        idx1 = i*navgs
        idx2 = idx1+navgs
        coh_spec[i,:] = np.abs(np.mean(csd12[idx1:idx2,:],0) ** 2)\
                        / ((np.mean(psd1[idx1:idx2,:nfreqs],0))\
                        *  np.mean(psd2[idx1:idx2,:nfreqs],0))
    coh_spec = Spectrogram(coh_spec, df=csd12.df, dt=segmentDuration,
                           epoch=csd12.epoch, unit=None,
                           name='coherence spectrogram')
    return coh_spec


def coherence_spectrogram(channel1, channel2, stride, segmentDuration,
                          overlap=None, pad=False, frames=False,
                          st=None, et=None):
    """
    Calculates coherence spectrogram for two channels.
    Can be called in two ways: either channel 1 and
    channel 2 are channel names or they are time series.
    If they are strings, then each segment is loaded
    separately as a way to save memory.

    Parameters:
    -----------
        channel 1 : str or TimeSeries object
            channel 1 name or time series
        channel 2 : str or TimeSeries object
            channel 2 name or time series
        stride : int
            stride for individual ffts (in seconds)
        segmentDuration : int
            duration of segments in spectrogram (in seconds)
        overlap : int, optional
            amount of overlap between ffts (in seconds)
        pad : bool, optional
            decide whether or not to pad ffts for taking csd
        frames : bool, optional
            decide whether to use frames or nds2 to load data
        st : int, optional
            start time if we're loading channel 1 and channel 2
            segments on the fly
        et : int, optional
            end time if we're loading channel 1 and channel 2 on the fly
    Returns:
    --------
        coherence_spectrogram : Spectrogram object
            coherence spectrogram between channels 1 and 2
        psd1_spectrogram : Spectrogram object
            channel 1 psd spectrogram
        psd2_spectrogram : Spectrogram object
            channel 2 psd spectrogram
        N : int
            number of averages used for each pixel
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
            test, csd12, psd1, psd2, N = coherence(stepseries1, stepseries2,
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
            test, csd12, psd1, psd2, N = coherence(stepseries1, stepseries2,
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
	print ifo[0]
	if channel.split(':')[1] == 'GDS-CALIB_STRAIN':
	    cache = connection.find_frame_urls(ifo[0],ifo+'_HOFT_C00', st, et, urltype='file')
	else:
	    cache = connection.find_frame_urls(ifo[0], ifo + '_C', st, et,urltype='file')
        try:
            data = TimeSeries.read(cache, channel, st, et)
        except IndexError:
            cache = connection.find_frame_urls(ifo[0], ifo+'_R', st, et, urltype='file')
            data = TimeSeries.read(cache, channel, st, et)
    else:
        data = TimeSeries.fetch(channel, st, et)

    return data


def coherence_list(channel1, channels, stride, st=None, et=None,
                   overlap=None, pad=False, frames=False):
    """
    Calculates coherence between one channel and a list of other channels.
    Returns coherence, psds, and csds in dictionary form.

    Parameters:
    -----------

    Returns:
    --------

    """
    coh = {}
    psd1 = {}
    psd2 = {}
    csd12 = {}
    darm = _read_data(channel1, st, et)
    darm_fft = fftgram(darm, stride, pad=pad)
    for channel in channels:
        data = _read_data(channel, st, et, frames=frames)
        coh[channel], csd12[channel], psd1[channel], psd2[channel], N = \
            coherence(darm_fft, data, stride, overlap=overlap, pad=pad)

    return coh, csd12, psd1, psd2, N


def coherence_from_list(darm_channel, channels,
                        stride, st, et, frames=False, 
                        pad=False, fhigh=None, subsystem=None,
                        spec_fhigh=None,spec_flow=None, directory='./',
                        segmentDuration=None, tag=None):
    """
    Takes coherence from a list of channels and saves it to a file.
    Parameters:
    -----------
    darm_channel : str
        DARM channel to use for analysis
    channels : list (str)
        Auxiiliary channels to use for analysis
    stride : float
        fft length (in seconds). Must evenly divide sample rate
        of channels you plan to use (or resample to)
    st : int
        Start time for analysis
    et : int
        End time for analysis
    frames : bool (optional)
        Decides whether to read from frames or NDS2
        (default NDS2)
    pad : bool (optional)
        Decides whether or not to zero pad before taking
        coherence (defaults to False).
    fhigh : float (optional)
        High frequency for analysis. Will resample channels
        to save computational time of possible. Recommended 
        to use power of 2.
    subsystem : str (optional)
        Subsystem for analysis (used in saving file. defaults to 'CHANS')
    spec_fhigh : float (optional)
        High frequency for plotting coherence spectrogram
    spec_flow :  float (optional)
        Low frequency for plotting coherence spectrogram
    directory : str
        Base directory for saving output file
    segmentDuration : float
        Duration of coherence spectrogram pixels. must be >= stride. 
    """
    nargout = expecting()

    if not subsystem:
        subsystem='CHANS'

    channels, failed_channels = coh_io.check_channels(channels, st)

    darm = _read_data(darm_channel, st, et, frames=frames)
    if fhigh is not None:
        darm = darm.resample(fhigh * 2)
    fftgram1 = fftgram(darm, stride, pad=True)
    coherence_dict = {}

    # set up hdf5 file for output
    outputDir = coh_io.get_directory_structure(subsystem, st, directory=directory)
    fname = coh_io.create_coherence_data_filename(darm_channel, subsystem, st, et, tag=tag)
    filename = '%s/%s'%(outputDir, fname)
    f = h5py.File(filename, 'w')
    coherences = f.create_group('coherences')
    psd1 = f.create_group('psd1')
    psd2s = f.create_group('psd2s')
    csd12s = f.create_group('csd12s')

    # read data, populate hdf5 file
    for channel in channels:
        data = _read_data(channel, st, et, frames=frames)
        if fhigh is not None and data.sample_rate.value > 2 * fhigh:
            data = data.resample(fhigh * 2)

        # get coherence
        coh_temp, csd_temp, psd1_temp, psd2_temp, N, coh_spec = \
            coherence(fftgram1, data, stride, pad=pad, segmentDuration=segmentDuration)

        Nspec = segmentDuration / float(stride)

        # plot coherence spectrogram
        plot = plot_coherence_specgram(coh_spec,darm_channel, channel, st, et, Nspec, 
                                fhigh=spec_fhigh, flow=spec_flow)
        spec_name = coh_io.create_coherence_data_filename(darm_channel, channel, st, et, tag=tag)
        outDir = coh_io.get_directory_structure(subsystem, st, directory=directory, specgram=True)
        plot.savefig('%s/%s' % (outDir, spec_name))

        # add coherence to coherence dictionary
        coherence_dict[channel] = coh_temp

        # save to coherence
        coh_temp.to_hdf5(f['coherences'], name=channel)
        csd_temp.to_hdf5(f['csd12s'], name=channel)
        psd2_temp.to_hdf5(f['psd2s'], name=channel)
    psd1_temp.to_hdf5(f['psd1'], name=darm_channel)
    f['info'] = N
    f['seg'] = Segment(st, et)
    if len(failed_channels):
        f['failed_channels'] = failed_channels
    else:
        f['failed_channels'] = 'None'
    f.close()


def plot_coherence_specgram(coh_spec, darm_channel, channel, st, et, N, fhigh=None, flow=None):
    """
    Plots coherence spectrogram

    Parameters: 
    -----------
    coh_spec : Spectrogram object
        Coherence spectrogram 
    darm_channel : str
        DARM channel used in analysis
    channel : str
        Auxiliary channel used in analysis
    st : int
        start time for coherence spectrogram
    et : int 
        end time for coherence spectrogram
    fhigh : float (optional)
        high frequency for plotting
    flow : float (optional)
        low frequency for plotting

    Returns:
    --------
    plot : matplotlib object
        plot object with ylimits set to fhigh and flow
    """
    channel = channel.replace(':','-')
    chan_pname = channel.replace('_','\_')
    darm_channel = darm_channel.replace(':','-')
    darm_chan_pname=darm_channel.replace('_','\_')
    low = 1. / N ** 2
    plot = coh_spec.plot(vmin=low,vmax=1,norm='log', cmap='Spectral_r')
    ax = plot.gca()
    plot.add_colorbar(label='Coherence')
    if not fhigh:
        fhigh = coh_spec.frequencies[-1].value
    if not flow:
        flow = coh_spec.frequencies[0].value
    ax.set_ylim(flow,fhigh)
    ax.set_title('Coherence between %s and %s' % (chan_pname, darm_chan_pname),
                 fontsize=12)
    return plot


def create_matrix_from_file(coh_file, channels):
    """
    Creates coherence matrix from data that's in a file.
    Used typically as helper function for plotting

    Parameters:
    -----------
    coh_file : str
        File containing coherence data
    channels : list (str)
        channels to plot

    Returns: 
    --------
    coh_matrix : Spectrogram object
        coherence matrix in form of spectrogram object
        returns automatically in terms of coherence SNR:
        coherence * N.
        (not actually a spectrogram, though)
    frequencies : numpy array
        numpy array of frequencies associated with coherence matrix
    labels : list (str)
        labels for coherence matrix
    N : int
        Number of time segment used to create coherence spectra
    """
    labels = []
    counter = 0
    if not os.path.exists(coh_file):
	return None, None, None, None
    f = h5py.File(coh_file, 'r')
    # get number of averages
    N = f['info'].value
    channels = f['psd2s'].keys()
    First = 1
    s = 0
    for channel in channels:
        data = Spectrum.from_hdf5(f['coherences'][channel])
        s_temp = data.size
        if s_temp > s:
            s = s_temp
    for channel in channels:
        if First:
            # initialize matrix!
            darm_psd = Spectrum.from_hdf5(f['psd1'][f['psd1'].keys()[0]])
            First = 0
            s = max(s, darm_psd.size)
            coh_matrix = np.zeros((s, len(channels)))
        data = Spectrum.from_hdf5(f['coherences'][channel])
        labels.append(channel[3:-3].replace('_', '-'))
        coh_matrix[:data.size, counter] = data
        counter += 1
    coh_matrix = Spectrogram(coh_matrix)
    frequencies = (np.arange(s)+1) * (darm_psd.frequencies.value[1] - darm_psd.frequencies.value[0])
    return coh_matrix, frequencies, labels, N

def plot_coherence_matrix(coh_matrix, labels, frequencies, subsystem, nsegs, fhigh=None, flow=None):
    """
    plots coherence matrix.

    Parameters:
    -----------
        coh_matrix : spectrogram object
            Contains coherence spectra for many channels.
            Not actually a spectrogram. Just using the object.
        labels :  list
            list of labels for channels in coh_matrix
        frequencies : list
            list of frequencies in coherence spectrum
        subsystem : str
            subsystem we're plotting
        fhigh : float, optional
            highest frequency for plotting
        flow : float, optional
            lowest frequency for plotting

    Returns:
    --------
        matplotlib plot object
    """
    N = nsegs
    my_dpi = 100
    low = 1./50
    high = 50.
    for label in labels:
        label = label.replace(subsystem,'')
    plt.figure(figsize=(1200. / my_dpi, 600. / my_dpi), dpi=my_dpi)
    plt.pcolormesh(frequencies, np.arange(
        0, len(labels) + 1), N*coh_matrix.value.T,
        norm=LogNorm(vmin=low, vmax=high), cmap=plt.get_cmap('Spectral_r'))
    cbar = plt.colorbar(label='coherence')
    cbar.set_ticks(np.arange(1, 21))
    ax = plt.gca()
    plt.title(subsystem)
    plt.yticks(np.arange(1, len(labels) + 1) - 0.5, labels, fontsize=8)
    if not fhigh:
        fhigh=frequencies[-1]
    if not flow:
        flow=frequencies[0]
    ax.set_xlim(flow, fhigh)
    ax.set_xlabel('Frequency [Hz]')
    return plt


def plot_coherence_matrix_from_file(darm_channel, channels, coh_file, subsystem='CHANS', fhigh=None, flow=None):
    """
    Plots coherence matrix from file

    Parameters:
    -----------
        darm_channel : str
            channel that all other channels were xcorred against
        channel_list : str or dict
            list of channels to plot
        coh_file : str
            coherence file to load
        subsystem : str
            subsystem to plot (used in loading from list)
        fhigh : float, optional
            high frequency for plotting
        flow : float, optional
            low frequency for plotting 

    Returns:
    --------
        NONE: saves plot
    """
    labels = []
    counter = 0
    coh_matrix, frequencies, labels, N = create_matrix_from_file(coh_file, channels)
    plot = plot_coherence_matrix(coh_matrix, labels, frequencies, subsystem, N, fhigh=fhigh, flow=flow)
    outfile = coh_file
    if outfile[-3:] == 'hdf':
        outfile = outfile[:-4]
    plot.savefig(outfile)
    plot.close()


def expecting():
    """Return how many values the caller is expecting"""
    f = inspect.currentframe()
    f = f.f_back.f_back
    c = f.f_code
    i = f.f_lasti
    bytecode = c.co_code
    instruction = ord(bytecode[i+3])
    if instruction == dis.opmap['UNPACK_SEQUENCE']:
        howmany = ord(bytecode[i+4])
        return howmany
    elif instruction == dis.opmap['POP_TOP']:
        return 0
    return 1
