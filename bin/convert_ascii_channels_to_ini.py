import optparse

def parse_command_line():
    """
    parse command parse command line
    """
    parser = optparse.OptionParser()
    parser.add_option(
        "--list", help="channel list", default=None,
        type=str)

    params, args = parser.parse_args()
    return params

params = parse_command_line()
subsystems = []
channels = {}

f = open(params.list,'r')
# sort into subsystems
for line in f:
    if line.startswith('#') or len(line.split(':')) < 2:
        continue
    ifo = line.split(':')[0]
    chan = line.split(':')[1]
    sub = chan.split('_')[0]
    if sub not in subsystems:
        subsystems.append(sub)

f.close()
subsystems = ['ALS',
              'ASC',
              'CAL',
              'HPI',
              'IMC',
              'ISI-BS',
              'ISI-ETM',
              'ISI-ITM',
              'ISI-HAM',
              'ISI-GND',
              'LSC',
              'OMC',
              'PEM-CS',
              'PEM-EX',
              'PEM-EY',
              'PSL',
              'SUS-BS',
              'SUS-ITM',
              'SUS-ETM',
              'SUS-MC',
              'SUS-PR',
              'SUS-SR',
              'TCS'
              ]

f2 = open(params.list.replace('txt','ini'),'w')
print len(subsystems)
for sub in subsystems:
    print sub
    counter = 0
    channels[sub] = {}
    f2.write('[%s]\n' % sub)
    f = open(params.list,'r')
    for line in f:
        if line[0] == '#' or len(line.split(':')) < 2:
            continue
        ifo = line.split(':')[0]
        chan = line.split(':')[1]
        sub_temp = chan.split('_')[0]
        if sub in sub_temp:
            counter += 1
            channels[sub][str(counter)] = chan
            f2.write('%d: %s' % (counter, chan))
    f.close()
f2.close()
