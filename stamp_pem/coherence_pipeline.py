import coherence_functions as cf
from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict
from gwpy.spectrum import Spectrum
import numpy as np
import optparse
import h5py
from glue import datafind


def parse_command_line():
    """
    parse_command_line
    """
    parser = optparse.OptionParser()
    parser.add_option(
        "--channel1", help="channel 1", default='L1:GDS-CALIB_STRAIN',
        type=str)
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
        "-d", help="plot directory", type=str, default="./", dest='dir')
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
        if channel is 'L1:GDS-CALIB_STRAIN':
            ftype = ifo + 'HOFT_C00'
        else:
            ftype = ifo + '_C'
        cache = connection.find_frame_urls(ifo[0], ftype, st, et)
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


def coherence_from_list(darm_channel, channel_list,
                        stride, st, et, frames=False, save=False,
                        pad=False):
    channels = _read_list(channel_list)
    # coherences = {}
    # psd1 = {}
    # psd2s = {}
    # csd12s = {}
    darm = _read_data(darm_channel, st, et, frames=frames)
    print 'darm data done loading...'
    fftgram1 = cf.fftgram(darm, stride, pad=True)
    if save:
        filename = darm_channel.replace(
            ':', '-') + '-' + str(st) + '-' + str(et - st)
        f = h5py.File(filename, 'w')
        coherences = f.create_group('coherences')
        psd1 = f.create_group('psd1')
        psd2s = f.create_group('psd2s')
        csd12s = f.create_group('csd12s')
        info = f.create_group('info')
        for channel in channels:
            data = _read_data(channel, st, et, frames=frames)
            print 'data for channel', channel, 'done loading...'
            # create new group for channel
            coherences.create_group(channel)
            psd1.create_group(channel)
            psd2s.create_group(channel)
            csd12s.create_group(channel)
            # get coherence
            coh_temp, csd_temp, psd1_temp, psd2_temp, N = \
                cf.coherence(fftgram1, data, stride, pad=pad)
            # save to coherence
            coh_temp.to_hdf5(f['coherences'][channel])
            csd_temp.to_hdf5(f['csd12s'][channel])
            psd1_temp.to_hdf5(f['psd1'][channel])
            psd2_temp.to_hdf5(f['psd2s'][channel])
        f[info] = N
        f.close()

    # print psd1
    # if save:
    #     filename = darm_channel.replace(
    #         ':', '-') + '-' + str(st) + '-' + str(et - st)
    #     f = h5py.File(filename, 'w')
    #     data = f.create_group('data')
    #     psd1.to_hdf5(data)
    # data['psd1'] = psd1
    # data['psd2s'] = psd2s
    # data['csd12s'] = csd12s
    # data['theoretical coherence'] = 1. / N
    # data['coherences'] = coherences
    #     f.close()
    #     return
    # else:
    #     return coherences, N


params = parse_command_line()

coherence_from_list(params.channel1, params.list, params.stride, params.st,
                    params.et, frames=params.frames, save=True,
                    pad=True)

# channels = _read_list(params.list)
# first = 1
# for channel in channels:
#     coherence, psd1, psd2, N = cf.coherence_spectrogram(params.channel1,
#                                                         channel,
#                                                         params.stride,
#                                                         params.sd,
#                                                         st=params.st,
#                                                         et=params.et,
#                                                         frames=params.frames)
#     f_low = 63 * params.stride
#     f_high = 65 * params.stride
#     plot = coherence.plot(vmin=1e-3, vmax=1)
#     ax = plot.gca()
#     ax.set_ylim(f_low / params.stride, f_high / params.stride)
#     plot.add_colorbar(label='coherence')
#     plot.savefig(params.dir + channel + '-coh')
#     plot.close()

#     plot = psd2.plot(vmin=psd2.value[:, f_low:f_high].min(),
#                      vmax=psd2.value[:, f_low:f_high].max(),
#                      norm='log')
#     plot.add_colorbar(label='power')
#     ax = plot.gca()
#     ax.set_ylim(f_low / params.stride, f_high / params.stride)
#     plot.savefig(params.dir + channel + '-power')
#     plot.close()

# plot = psd1.plot(vmin=psd1.value.min(), vmax=psd1.value.max())
# ax = plot.gca()
# ax.set_ylim(f_low / params.stride, f_high / params.stride)
# plot.add_colorbar(label='power')
# plot.savefig(params.dir + channel + '-power')
