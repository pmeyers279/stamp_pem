from glue import pipeline

datajob = pipeline.CondorDAGJob('vanilla', './coherence_pipeline.py')
dag = pipeline.CondorDAG('./pat.log')

st = 1123855217
et = st + 100
for j in range(10):
    i = j + 1
    job = pipeline.CondorDAGJob("%d.dag" % i, str(i))
    job.set_stderr_file('$(start).err')
    job.set_stdout_file('$(start).out')
    job.set_sub_file('pat.sub')
    node = pipeline.CondorDAGNode(job)
    node.add_macro("start", st)
    node.add_macro("end", et)
    dag.add_node(node)
    st += 100
    et = st + 100
datajob.set_sub_file('pat.sub')
arg = '-s $(start) -e $(end) --stride %d --fhigh %d --list channels.ini' % (
	   10, 50)
print arg
datajob.add_arg(arg)
datajob.set_stdout_file("$(start).out")
datajob.set_stderr_file("$(start).err")
datajob.set_log_file('./pat.log')
datajob.write_sub_file()
dag.set_dag_file('pat')
dag.write_dag()
dag.write_script()
