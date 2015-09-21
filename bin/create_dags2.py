from glue import pipeline
import optparse
from stamp_pem import coh_io


def parse_command_line():
    """
    parse command parse command line
    """
    parser = optparse.OptionParser()
    parser.add_option(
        "--list", help="channel list", default=None,
        type=str)
    parser.add_option(
        "-s", "--start-time", help="start time", default=None,
        type="int", dest="st")
    parser.add_option(
        "-e", "--end-time", help="end time", default=None,
        type="int", dest="et")
    parser.add_option(
        "--fhigh", help="max frequency", type=float, default=None)
    parser.add_option(
        "--stride", dest='stride', help="stride for ffts",
        type=float, default=1)

    params, args = parser.parse_args()
    return params

params = parse_command_line()

channels = coh_io.read_list(params.list)

datajob = pipeline.CondorDAGJob('vanilla', './coherence_pipeline.py')
dag = pipeline.CondorDAG('/usr1/meyers/$(subsystem).log')

for key in channels.keys():
    dagName = '%s-%d-%d.dag' % (params.list.split('.')[0],
                                params.st, params.et)
    job = pipeline.CondorDAGJob('vanilla', './coherence_pipeline')
    job.set_sub_file('coherence.sub')
    job.set_stderr_file('/usr1/meyers/$(subsystem).err')
    job.set_stdout_file('/usr1/meyers/$(subsystem).out')
    node = pipeline.CondorDAGNode(job)
    node.add_macro("subsystem", key)

datajob.set_sub_file('coherence.sub')
arg = '-s %d -e %d --list %s --fhigh %f --stride %d --subsystem %s --frames 1' % (
    params.st, params.et, params.list, params.fhigh, params.stride, key)
datajob.add_arg(arg)
datajob.set_stderr_file('/usr1/meyers/$(subsystem).err')
datajob.set_stdout_file('/usr1/meyers/$(subsystem).out')
datajob.set_sub_file('coherence.sub')
datajob.set_log_file('/usr1/meyers/$(subsystem).log')
datajob.write_sub_file()
dag.set_dag_file('coherence.dag')
dag.write_dag()
dag.write_script()


# for j in range(10):
#     i = j + 1
#     job = pipeline.CondorDAGJob("%d.dag" % i, str(i))
#     job.set_stderr_file('$(start).err')
#     job.set_stdout_file('$(start).out')
#     job.set_sub_file('pat.sub')
#     node = pipeline.CondorDAGNode(job)
#     node.add_macro("start", st)
#     node.add_macro("end", et)
#     dag.add_node(node)
#     st += 100
#     et = st + 100
# datajob.set_sub_file('pat.sub')
# datajob.set_stdout_file("$(subsystem).out")
# datajob.set_stdout_file("$(subsystem).out")
# arg = '-s $(start) -e $(end) --stride %d --fhigh %d --list channels.ini ' % (
#     10, 50)
# print arg
# datajob.add_arg(arg)
# datajob.set_stdout_file("$(start).out")
# datajob.set_stderr_file("$(start).err")
# datajob.set_log_file('./pat.log')
# datajob.write_sub_file()
# dag.set_dag_file('pat')
# dag.write_dag()
# dag.write_script()
