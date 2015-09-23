import ConfigParser
import os


def config_section_map(c, section):
    dict1 = {}
    options = c.options(section)
    for option in options:
        dict1[option] = c.get(section, option)
    return dict1


def config_list(c):
    dict1 = {}
    for section in c.sections():
        dict1[section] = config_section_map(c, section)
    return dict1


def read_list(list):
    """
    read channel list from file
    """
    c = ConfigParser.ConfigParser()
    c.read(list)
    channels = config_list(c)
    return channels


def create_coherence_data_filename(darm_channel, subsystem, st, et,
                                   directory='./', tag=None):
    filename = darm_channel.replace(
        ':', '-') + '-' + subsystem + '-' +\
        str(st) + '-' + str(et - st)
    chan = darm_channel.replace(':', '-')
    if tag:
        filename = '%s-%s-%s-%d-%d' % (chan, subsystem, tag, st, et - st)
    else:
        filename = '%s-%s-%d-%d' % (chan, subsystem, st, et - st)
    return filename


def get_directory_structure(subsystem, st, directory='./', specgram=False):
    if specgram:
        return '%s/%s/%s/' % (directory, subsystem, str(st)[0:5])
    else:
        return '%s/%s/%s/' % (directory, subsystem, str(st)[0:5], 'plots')


def create_directory_structure(subsystems, st, directory='./'):
    for subsystem in subsystems:
        cmd = 'mkdir -p %s/%s/%s/%s' % (directory, subsystem,
                                        str(st)[0:5], 'plots')
        os.system(cmd)
