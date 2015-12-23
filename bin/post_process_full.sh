
st=1134432017
st1=$st
et=1134435617
for i in `seq 1 12`;
do
#condor_workflow -i ../../config_files/ini_files/L1.ini -e $et -s $st
st=$((st+3600))
et=$((et+3600))
done
combine_coherence
st1=1134432017
et=1134590417

#./combine_full_set -s $st1 -e $et --directory /home/meyers/tools/stamppem/results2 --channel-list /home/meyers/tools/stamppem/config_files/channel_lists/L1_O1_chans_stochmon_short.ini --darm-channel L1:GDS-CALIB_STRAIN

#./plot_clipped_coherence_matrices -s $st1 -e $et --directory /home/meyers/tools/stamppem/results2 --channel-list /home/meyers/tools/stamppem/config_files/channel_lists/L1_O1_chans.ini --flow 70 --fhigh 80 --darm-channel L1:GDS-CALIB_STRAIN
make_page -i /home/meyers/tools/stamppem/config_files/ini_files/L1.ini -s $st1 -e $et -l 70 -g 80
