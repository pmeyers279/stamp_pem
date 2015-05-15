from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict
from glue import datafind
from gwpy.spectrogram import Spectrogram



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
    Raises
    ------
        Errors : description
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
    df = 1/fftlength
    stride *= timeseries.sample_rate.value
    nsteps = int(fftlength*timeseries.size // stride)
    nfreqs = int(fftlength*timeseries.sample_rate.value)
    dtype = numpy.complex
    out = Spectrogram(numpy.zeros((nsteps,nfreqs),dtype=dtype),
        name = timeseries.name,epoch=timeseries.epoch,f0=0,df=df,
        dt=dt,copy=True,unit=self.unit,dtype=dtype)
    # stride through TimeSeries, recording FFTs as columns of Spectrogram
    out.frequencies = np.arange(0,timeseries.sample_rate.value/2,df)
    for step in range(nsteps):
        idx = stride * step
        idx_end = idx + stride
        stepseries = timeseries.data[idx:idx_end]
        out[step] = fft.fft(np.hstack((stepseries,np.zeros(1,stepseries.size))))[0:np.floor(stride/2.+1)]


