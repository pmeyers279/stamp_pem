from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict
from gwpy.spectrum import Spectrum
from glue import datafind
from gwpy.spectrogram import Spectrogram
import numpy as np
import scipy.fftpack as fft
from astropy import units as u

# ---------------------------------
# RETRIEVING DATA FUNCTIONS
# ---------------------------------
def datafind2(ifo, type, starttime, endtime, urltype='file'):
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
    cache = connection.find_frame_urls(ifo, type, starttime, endtime, urltype=urltype)
    return cache

def retrieve_time_series(cache, channel):
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
    data = TimeSeries.read(cache, channel)
    return data

def retrieve_time_series_dict(cache, channelList):
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
    dataDict = TimeSeriesDict.read(cache, channelList)
    return dataDict

# -----------------------------------------
# ANALYSIS FUNCTIONS
# -----------------------------------------
def fftgram(timeseries, stride):
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
    nsteps = 2*int(timeseries.size // stride)  - 1
    # only get positive frequencies
    nfreqs = int(fftlength*timeseries.sample_rate.value) - 1
    dtype = np.complex
    # initialize the spectrogram
    out = Spectrogram(np.zeros((nsteps, nfreqs),dtype=dtype),
        name = timeseries.name,epoch=timeseries.epoch,f0=df/2,df=df/2,
        dt=dt,copy=True,unit=timeseries.unit/u.Hz**0.5,dtype=dtype)
    # stride through TimeSeries, recording FFTs as columns of Spectrogram
    out.starttimes = np.zeros(nsteps)
    for step in range(nsteps):
        # indexes for this step
        idx = (stride/2) * step
        idx_end = idx + stride
        out.starttimes[step]=(idx/stride)+timeseries.epoch.value
        stepseries = timeseries[idx:idx_end]
        # zeropad, window, fft, shift zero to center, normalize
        tempfft = (fft.fftshift(fft.fft(np.hstack(
            (np.multiply(stepseries.data,np.hanning(stepseries.data.size)),np.zeros(stepseries.size))
            )))*1/stride)
        # get the positive indices we want (start in middle, take very other)
        out.data[step] = tempfft[np.floor(tempfft.size/2)+1:]
    return out

def psdgram(timeseries,stride,adjacent=1):
    """calculates PSD from timeseries
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
    df = 1./fftlength
    # number of values in a step
    stride *= timeseries.sample_rate.value
    # number of steps
    nsteps = 2*int(timeseries.size // stride) - 1
    # only get positive frequencies
    nfreqs = int(fftlength*timeseries.sample_rate.value) / 2. 
    # initialize the spectrogram
    out = Spectrogram(np.zeros((nsteps, nfreqs)),
        name = timeseries.name,epoch=timeseries.epoch,f0=0,df=df,
        dt=dt,copy=True,unit=timeseries.unit / u.Hz)
    # stride through TimeSeries, recording FFTs as columns of Spectrogram
    out.starttimes = np.zeros(nsteps)
    for step in range(nsteps):
        # indexes for this step
        idx = (stride/2) * step
        idx_end = idx + stride
        out.starttimes[step] = (idx / stride) + timeseries.epoch.value
        stepseries = timeseries[idx:idx_end]
        out[step] = stepseries.psd()[:-1]

    psdleft = np.hstack((out.data.T, np.zeros((out.data.shape[1], 4))))
    psdright = np.hstack((np.zeros((out.data.shape[1], 4)), out.data.T))
    # psd we want is average of adjacent, non-ovlped segs. don't include
    # middle segment for now. throw away edges.
    psd = np.divide(np.add(psdleft, psdright),2).T[4:-4]
    psd = Spectrogram(psd, name=out.name, epoch=out.epoch,
        f0=out.f0, df=out.df, dt=out.dt,
        copy=True, unit=out.unit)
    # recall we're throwing away first and last 2 segments. 
    # to be able to do averaging
    psd.starttimes = out.starttimes[2:-2]
    psd.frequencies = out.frequencies
    return psd

def csdgram(channel1, channel2, stride):
    """calculates csd spectrogram between two timeseries
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
        fftgram1 = fftgram(channel1, stride)
    elif isinstance(channel1, Spectrogram):
        fftgram1 = channel1
    else:
        raise TypeError('First arg is either TimeSeries or Spectrogram object')
    if isinstance(channel2, TimeSeries):
        fftgram2 = fftgram(channel2, stride)
    elif isinstance(channel2, Spectrogram):
        fftgram2 = channel2
    else:
        raise TypeError('First arg is either TimeSeries or Spectrogram object')
    # clip off first 2 and last 2 segments to be consistent with psd
    # calculation
    out = np.multiply(fftgram1.data, np.conj(fftgram2.data))[2:-2]
    starttimes = fftgram2.starttimes[2:-2]

    csdname = 'csd spectrogram between %s and %s'%(fftgram1.name,fftgram2.name)
    out = Spectrogram(out,name=csdname,epoch=starttimes[0],df=fftgram1.df,
        dt=fftgram1.dt,copy=True,unit=fftgram1.unit*fftgram2.unit,f0=fftgram1.f0)
    df = fftgram1.df.value*2
    f0 = fftgram1.f0.value*2
    csdgram = Spectrogram(np.zeros((out.shape[0],out.shape[1]/2)),df=df,
        dt = fftgram1.dt,copy=True,unit=out.unit,f0=f0,epoch=out.epoch)
    
    for ii in range(csdgram.shape[0]):
        temp = Spectrum(out.data[ii],df=out.df,f0=out.f0,epoch=out.epoch)
        csdgram[ii] = coarseGrain(temp,df,f0,np.floor(out.shape[1]/2.))
    return csdgram

def stamp_sigma_gram(channel1,channel2,stride):
    """calculates STAMP sigma value
    NOT normalized with window factors
    Parameters
    ----------
        channel1 : TimeSeries or Spectrogram object
            either timeseries or psd for channel1 
        channel2 : TimeSeries or Spectrogram object
            either timeseries or psd for channel2 
        stride : `int`
            segment duration usesd
    Returns
    -------
        sigma_gram : Spectrogram object
            spectrogram of stamp sigma value
            not normalized with window factors
    """
    if isinstance(channel1,TimeSeries):
        psd1 = psdgram(channel1,stride).T[:-1].T
    elif isinstance(channel1,Spectrogram):
        psd1 = channel1
    else:
        raise TypeError('channel1 must be of correct type')

    if isinstance(channel2,TimeSeries):
        psd2 = psdgram(channel2,stride).T[:-1].T
    elif isinstance(channel2,Spectrogram):
        psd2 = channel2
    else:
        raise TypeError('channel2 must be of correct type')

    sigma_gram = (sensint(psd1,psd2,stride)*psd1.df)**-1
    # sigma_gram = np.power(np.multiply(psd1.data,psd2.data)/psd1.df,0.5)
    # sigma_gram_name = \
    # 'STAMP sigma spectrogram for %s and %s'%(psd1.name,psd2.name)
    # # sqrt(df/(psd1*psd2)) -> units of power of some kind
    # sigma_gram_unit = (psd1.unit*psd2.unit/u.Hz)**0.5
    # sigma_gram = Spectrogram(sigma_gram,name=sigma_gram_name,df=psd1.df,
    #     dt=psd1.dt,unit=sigma_gram_unit,copy=True,epoch=psd1.epoch)
    return sigma_gram

def cal_ccvar(sig):
    """calculate cross correlation variance
    integrated over frequency
    Parameters
    ----------
        sig : Spectrogram object
            sigma spectrogram

    Returns
    -------
        ccVar : timeseries object
            ccVar TimeSeries
    Raises
    ------
        Errors : description
    """
    ccVar = np.sum(np.power(sig.data,2),axis=1)*sig.df
    ccVar = TimeSeries(ccVar,name='cc variance time series',
        unit=sig.unit*u.Hz,dt=sig.dt,copy=True,epoch=sig.epoch)
    
    return ccVar

def window_factors(window1,window2):
    """calculates window factors
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
    if Nred==1:
        raise ValueError('windows are not compatible')

    N1 = window1.size
    N2 = window2.size
    # get reduced windows
    window1red = window1[0:(N1-N1/Nred)+1]
    window2red = window2[0:(N2-N2/Nred)+1]
    idx = int(np.floor(Nred/2.))

    w1w2bar = np.mean(np.multiply(window1red,window2red))
    w1w2squaredbar = np.mean(np.multiply(np.power(window1red,2),np.power(window2red,2)))
    w1w2ovlsquaredbar = np.mean(np.multiply(
        np.multiply(window1red[0:idx],window2red[idx:]),
        np.multiply(window1red[idx:],window2red[0:idx]))) 
    return w1w2bar,w1w2squaredbar,w1w2ovlsquaredbar

def coarseGrain(spectrum,f0,df,N):
    """coarse grain frequency spectrum
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
    fhighi = f0i+(Ni-1)*dfi
    fhigh = f0+df*(N-1)
    i = np.arange(0,N)

    # low/high indices for coarse=grain
    jlow = 1+( (f0 + (i-0.5)*df - f0i - 0.5*f0i)/dfi)
    jhigh = 1+( (f0 + (i+0.5)*df - f0i - 0.5*f0i)/dfi)
    # fractional contribution of partial bins 
    fraclow =  (dfi + (jlow+0.5)*dfi - f0 - (i-0.5)*df)/dfi
    frachigh = (df + (i+0.5)*df - f0i - (jhigh-0.5)*dfi)/dfi

    jtemp = jlow+2
    y_real = np.zeros(N)
    y_imag = np.zeros(N)
    for idx in range(N):
        y_real[idx] = sum(spectrum.data.real[jtemp[idx]-1:jhigh[idx]])
        y_imag[idx] = sum(spectrum.data.imag[jtemp[idx]-1:jhigh[idx]])
    y = np.vectorize(complex)(y_real,y_imag)

    ya = (dfi/df)*(np.multiply(spectrum.data[jlow[:-1].astype(int)-1],fraclow[:-1])+
                    np.multiply(spectrum.data[jhigh[:-1].astype(int)-1],frachigh[:-1]+
                        y[:-1]))
    if (jhigh[N-1]>Ni-1):
        yb = (dfi/df)*(spectrum.data[jlow[N-1]+1]*fraclow[N-1]+y[N-1])
    else:
        yb = (dfi/df)*(spectrum.data[jlow[N-1]+1]*fraclow[N-1]+
                        spectrum.data[jhigh[N-1]+1]*frachigh[N-1]+
                            y[N-1])
    y = np.hstack((ya,yb))
    y = Spectrum(y,df=df,f0=f0,epoch=spectrum.epoch,unit=spectrum.unit,name=spectrum.name)
    return y

def sens_int(channel1,channel2,stride):
    """calculate sensitivity integrand of cc
    calculation for channels 1 and 2 based on stride chosen
    Parameters
    ----------
        channel1 : TimeSeries or Spectrogram object
            psd spectrogram or time series for DARM
        channel2 : TimeSeries or Spectrogram object
            psd spectrogram or time series for AUX channel
    Returns
    -------
        sensit : Spectrogram object
            sensitivity integrand for cc calculation
    Raises
    ------
        Errors : description
    """
    if isinstance(channel1,TimeSeries):
        psd1 = psdgram(channel1,stride).T[:-1].T
    elif isinstance(channel1,Spectrogram):
        psd1 = channel1
    else:
        raise TypeError('channel1 must be of correct type')

    if isinstance(channel2,TimeSeries):
        psd2 = psdgram(channel2,stride).T[:-1].T
    elif isinstance(channel2,Spectrogram):
        psd2 = channel2
    else:
        raise TypeError('channel2 must be of correct type')
    sensint = (psd1*psd2)**-1

    return sensint

