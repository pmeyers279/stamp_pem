from stamp_pem import coh_io
from glue import pipeline
import optparse
from gwpy.segments import DataQualityFlag


def parse_command_line():
    """
    parse command parse command line
    """
    parser = optparse.OptionParser()
    parser.add_option(
        "--ini-file", "-i", help="pipeline ini file",
        default=None, type=str, dest='ini')
    parser.add_option(
        "-s", "--start-time", help="start time", default=None,
        type="int", dest="st")
    parser.add_option(
        "-e", "--end-time", help="end time", default=None,
        type="int", dest="et")

    params, args = parser.parse_args()
    return params


def build_arg(env_params, run_params, st, et):
    flags = {}
    flags['-list'] = env_params['list']
    flags['s'] = '$(st)'
    flags['e'] = '$(et)'
    flags['-fhigh'] = str(run_params['fhigh'])
    flags['-darm-channel'] = run_params['darm_channel']
    flags['-stride'] = str(run_params['stride'])
    flags['-subsystem'] = '$(subsystem)'
    flags['-segment-duration'] = str(run_params['segmentDuration'])
    if run_params['spec_fhigh']:
        flags['-spec-fhigh'] = str(run_params['spec_fhigh'])
    if run_params['spec_flow']:
        flags['-spec-flow'] = str(run_params['spec_flow'])
    arg = ''
    for flag in flags.keys():
        arg = arg + ' -%s %s' % (flag, flags[flag])
    return arg


params = parse_command_line()

pipeline_dict = coh_io.read_pipeline_ini(params.ini)
env_params, run_params = coh_io.check_ini_params(pipeline_dict['env'])
channel_dict = coh_io.read_list(env_params['list'])

try:
    time_file = env_params['online_time_file']
    inc = int(env_params['online_increment'])
    st = coh_io.read_time_from_file(time_file)
    et = st + inc
    coh_io.increment_datetime_in_file(time_file, inc)
except KeyError:
    st = params.st
    et = params.et

darm_channel = run_params['darm_channel']
flag = run_params['flag']


if not coh_io.check_channel_and_flag(darm_channel, flag):
    raise ValueError('channel and flag are not for same IFO!')

# get and write DQ segments:
segs = DataQualityFlag.query_dqsegdb(flag, st, et)
seg_file = coh_io.create_coherence_data_filename(flag, 'SEGMENTS', st, et)
segs.write('%s.xml.gz' % (seg_file))

for seg in segs.active:

    datajob = pipeline.CondorDAGJob('vanilla', env_params['executable'])
    datajob2 = pipeline.CondorDAGJob(
        'vanilla', env_params['combine_executable'])
    dag = pipeline.CondorDAGJob(
        '/usr1/%s/$(subsystem).log' % (env_params['user']))
    coh_io.create_directory_structure(
        channel_dict.keys(), st, directory=env_params['base_directory'])

    dag_dir = coh_io.get_directory_structure(
        'DAGS', st, env_params['base_directory'])

    for subsystem in channel_dict.keys():
        for seg in segs.active:
            seg_st = seg[0].seconds
            seg_et = seg[1].seconds
            job = pipeline.CondorDAGJob('vanilla', env_params['executable'])
            job.set_sub_file('%s/%s-%d-%d.sub',
                             dag_dir, darm_channel.replace(':', '-'),
                             seg_st, seg_et)
            job.set_stderr_file(
                '/usr1/%s/$(subsystem).err' % (env_params['user']))
            job.set_stdout_file(
                '/usr1/%s/$(subsystem).out' % (env_params['user']))
            node = pipeline.CondorDAGNode(job)
            node.add_macro("subsystem", subsystem)
            node.add_macro("st", seg_st)
            node.add_macro("et", seg_et)
            dag.add_node(node)
    for subsystem in channel_dict.keys():
        job = pipeline.CondorDAGJob(
            'vanilla', env_params['combine_executable'])
        job.set_sub_file('%s/combine_jobs-%d-%d.sub', dag_dir, st, et)
        job.set_stderr_file(
            '/usr1/%s/$(subsystem)-combine.err' % env_params['user'])
        job.set_stdout_file(
            '/usr1/%s/$(subsystem)-combine.out' % env_params['user'])
        node = pipeline.CondorDAGNode(job)
        node.add_macro('subsystem', subsystem)
        dag.add_node(node)

    dagName = '%s/%s-%d-%d' % (
        dag_dir, darm_channel.replace(':', '-'), st, et)

    # datajob info
    datajob.set_sub_file('%s/%s-%d-%d.sub',
                         darm_channel.replace(':', '-'), st, et)
    datajob.set_stderr_file(
        '/usr1/%s/$(subsystem).err' % (env_params['user']))
    datajob.set_stdout_file(
        '/usr1/%s/$(subsystem).out' % (env_params['user']))
    datajob.set_log_file(
        '/usr1/%s/$(subsystem).out' % (env_params['user']))

    # combine jobs post processing info
    datajob2.set_sub_file('%s/combine_jobs-%d-%d.sub', dag_dir, st, et)
    datajob2.set_stderr_file(
        '/usr1/%s/$(subsystem)-combine.err' % env_params['user'])
    datajob2.set_stdout_file(
        '/usr1/%s/$(subsystem)-combine.out' % env_params['user'])
    datajob2.set_log_file(
        '/usr1/%s/$(subsystem)-combine.log' % env_params['user'])
    arg = build_arg(env_params, run_params)
    print 'ARG = %s' % arg
    datajob.add_arg(arg)
    datajob2.add_arg('-s %s -e %s --subsystem $(subsystem) --darm-channel %s')
    dag.set_dag_file(dagName)
    dag.write_dag

