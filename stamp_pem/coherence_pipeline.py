import coherence_functions as cf
from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict
from gwpy.spectrum import Spectrum
import numpy as np
import optparse
from glue import datafind


def parse_command_line():
    """
    parse_command_line
    """
    parser = optparse.OptionParser()
    parser.add_option(
        "--channel1", help="channel 1", default=None)
    parser.add_option(
        "--list", help="file containing channel list", default=None,
        type=str)
    parser.add_option(
        "-s", "--start-time", dest='st', help="start time", type=int,
        default=None)
    parser.add_option(
        "-e", "--end-time", dest='et', help="end time", type=int, default=None)
    parser.add_option(
        "--stride", dest='stride', help="stride for ffts",
        type=float, default=1)
    parser.add_option(
        "--delta-f", dest='df', help="frequency range", type=float, default=1)
    parser.add_option(
        "--fhigh", help="max frequency", type=float, default=None)
    parser.add_option(
        "--frames", help="read from frames as opposed to nds", type=int,
        default=False)
    parser.add_option(
        "--segment-duration", dest='sd', help="segment duration for specgram",
        type=float, default=None)
    params, args = parser.parse_args()
    return params


def _read_data(channel, st, et, frames=False):
    """
    get data, either from frames or from nds2
    """

    ifo = channel.split(':')[:3]
    if frames:
        # read from frames
        connection = datafind.GWDataFindHTTPConnection()
        cache = connection.find_frame_urls(ifo[0], ifo + '_R', st, et)
        data = TimeSeries.read(cache, channel, st, et)
    else:
        data = TimeSeries.fetch(channel, st, et)

    return data


def _read_list(file):
    """
    read channel list from file
    """
    channels = []
    f = open(file, 'r')
    for line in f:
        channels.append(line.split('\n')[0])
    f.close()
    return channels
params = parse_command_line()

channels = _read_list(params.list)
first = 1
for channel in channels:
    coherence, psd1, psd2, N = cf.coherence_spectrogram(params.channel1,
                                                        channel,
                                                        params.stride,
                                                        params.sd,
                                                        st=params.st,
                                                        et=params.et,
                                                        frames=true)
    plot = coherence.plot()
    plot.add_colorbar(label='coherence')
    plot.show()

plot = psd1.plot()
plot.add_colorbar('power in %s' % params.channel1)
plot.show()
