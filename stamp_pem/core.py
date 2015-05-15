from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict
from gwpy.spectrum import Spectrum
from glue import datafind
from gwpy.spectrogram import Spectrogram
import numpy as np
import scipy.fftpack as fft 
from astropoy import units as u



# ---------------------------------
# RETRIEVING DATA FUNCTIONS
# ---------------------------------
def datafind(ifo,type,starttime,endtime,urltype='file'):
    """uses glue datafind to get ifo data
    Parameters
    ----------
        ifo : str 
            single letter identifier for ifo
            L,H,G,V,K
        starttime : int
            start time to search for data
        endtime : int
            end time to search for data
        urltype : file scheme restriction
    Returns
    -------
        cache : cache object
            a list of cache entry representations of
            individual frame files.
    Raises
    ------ 
    """
    connection = datafind.GWDataFindHTTPConnection()
    cache = connection.find_frame_urls(ifo,type,starttime,endtime,urltype=urltype)
    return cache

def retrieve_time_series(cache,channel):
    """retrieve time series data using gwpy.TimeSeries
    given cache object and channel
    Parameters
    ----------
        cache : cache object
            cache object pointing to frame files
            to read
        channel : string
            channel name for which data will be retrieved
    Returns
    -------
        data : gwpy time series object
    Raises
    ------ 
    """
    data = TimeSeries.read(cache,channel)
    return data

def retrieve_time_series_dict(cache,channelList):
    """retrieve data for a list of channels
    Parameters
    ----------
        cache : cache object
            cache object point to frame files
            to read
        channelList : list 
            list of channels to get data for
    Returns
    -------
        dataDict : dict
            dictionary with channel names as keys
            and time series objects as values
    """
    dataDict = TimeSeriesDict.read(cache,channelList)
    return dataDict

# -----------------------------------------
# ANALYSIS FUNCTIONS
# -----------------------------------------
def fftgram(timeseries,stride):
    """calculates fourier-gram with zero
    padding
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
    df = 1./fftlength
    # number of values in a step
    stride *= timeseries.sample_rate.value
    # number of steps
    nsteps = 2*int(timeseries.size // stride) 
    # only get positive frequencies
    nfreqs = int(fftlength*timeseries.sample_rate.value) / 2. 
    dtype = np.complex
    # initialize the spectrogram
    out = Spectrogram(np.zeros((nsteps,nfreqs),dtype=dtype),
        name = timeseries.name,epoch=timeseries.epoch,f0=0,df=df,
        dt=dt,copy=True,unit=timeseries.unit/u.Hz**0.5,dtype=dtype)
    # stride through TimeSeries, recording FFTs as columns of Spectrogram
    for step in range(nsteps):
        # indexes for this step
        idx = stride * step
        idx_end = idx + stride/2
        stepseries = timeseries[idx:idx_end]
        # zeropad, window, fft, shift zero to center, normalize
        tempfft = (fft.fftshift(fft.fft(np.hstack(
            (np.multiply(stepseries.data,np.hanning(stepseries.data.size)),np.zeros(stepseries.size))
            )))*1/stride)
        # get the positive indices we want (start in middle, take very other)
        idxs_freqs = np.arange(tempfft.size/2,tempfft.size,2)
        idxs_fft = np.arange(tempfft.size/2,3*tempfft.size/4)
        out.data[step] = tempfft[idxs_fft]
        if step == 0:
            # what are the frequencies we actually have??
            out.frequencies = fft.fftshift(fft.fftfreq(int(2*stride),1./timeseries.sample_rate.value))[idxs_freqs]
    return out

def calPSD(fftgram,adjacent=1):
    """calculates PSD from fftgrams
    properly
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
    Raises
    ------
        Errors : description
    """
    psd = np.multiply(fftgram,np.conj(fftgram))
    psdleft = np.hstack((psd,np.zeros((psd.size,1))))
    psdright = np.hstack((np.zeros((psd.size,1)),psd))
    psd = np.divide(np.add(psdleft,psdright),2)
    psd = Spectrogram(psd,name=fftgram.name,epoch=fftgram.epoch,f0=fftgram.f0
        df=fftgram.df,dt=fftgram.dt,copy=True,unit=fftgram.unit**2)



