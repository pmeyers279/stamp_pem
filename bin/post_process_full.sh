
#./combine_full_set -s 1130803217 -e 1130810417 --directory /home/meyers/tools/stamppem/results --channel-list config_files/channel_lists/L1_O1_chans.ini

./plot_clipped_coherence_matrices -s 1130803217 -e 1130810417 --directory /home/meyers/tools/stamppem/results --channel-list config_files/channel_lists/L1_O1_chans.ini --flow 20 --fhigh 25
make_page -i config_files/ini_files/L1.ini -s 1130803217 -e 1130810417 -l 20 -g 25
./plot_clipped_coherence_matrices -s 1130803217 -e 1130810417 --directory /home/meyers/tools/stamppem/results --channel-list config_files/channel_lists/L1_O1_chans.ini --flow 25 --fhigh 30
make_page -i config_files/ini_files/L1.ini -s 1130803217 -e 1130810417 -l 25 -g 30
./plot_clipped_coherence_matrices -s 1130803217 -e 1130810417 --directory /home/meyers/tools/stamppem/results --channel-list config_files/channel_lists/L1_O1_chans.ini --flow 30 --fhigh 35
make_page -i config_files/ini_files/L1.ini -s 1130803217 -e 1130810417 -l 30 -g 35
./plot_clipped_coherence_matrices -s 1130803217 -e 1130810417 --directory /home/meyers/tools/stamppem/results --channel-list config_files/channel_lists/L1_O1_chans.ini --flow 35 --fhigh 40
make_page -i config_files/ini_files/L1.ini -s 1130803217 -e 1130810417 -l 35 -g 40
./plot_clipped_coherence_matrices -s 1130803217 -e 1130810417 --directory /home/meyers/tools/stamppem/results --channel-list config_files/channel_lists/L1_O1_chans.ini --flow 40 --fhigh 50
make_page -i config_files/ini_files/L1.ini -s 1130803217 -e 1130810417 -l 40 -g 50
