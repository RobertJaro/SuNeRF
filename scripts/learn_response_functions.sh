

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

# prepare temperature response functions
python -m sunerf.data.euv.load_aia_response_function

# STEREO
python -m sunerf.train.convert_temperature_response_function --response_file '/glade/work/rjarolim/data/sunerf/temperature_response/stereo_ahead_resonses.npz' --out_file '/glade/work/rjarolim/sunerf/response/stereo_ahead_interpolated.npz'
python -m sunerf.train.convert_temperature_response_function --response_file '/glade/work/rjarolim/data/sunerf/temperature_response/stereo_behind_resonses.npz' --out_file '/glade/work/rjarolim/sunerf/response/stereo_behind_interpolated.npz'

# AIA
# [94, 131, 171, 193, 211, 304, 335]
python -m sunerf.train.convert_temperature_response_function --response_file '/glade/work/rjarolim/sunerf/response/aia_response_functions.npz' --out_file '/glade/work/rjarolim/sunerf/response/aia_interpolated.npz'
python -m sunerf.train.convert_temperature_response_function --response_file '/glade/work/rjarolim/data/sunerf/temperature_response/aia_response.npz' --out_file '/glade/work/rjarolim/sunerf/response/aia_thin_interpolated.npz' --channels 0 1 2 3 4 6
python -m sunerf.train.convert_temperature_response_function --response_file '/glade/work/rjarolim/data/sunerf/temperature_response/stereo_ahead_response.npz' --out_file '/glade/work/rjarolim/sunerf/response/stereo_ahead_thin_interpolated.npz' --channels 0 1 2
python -m sunerf.train.convert_temperature_response_function --response_file '/glade/work/rjarolim/data/sunerf/temperature_response/stereo_behind_response.npz' --out_file '/glade/work/rjarolim/sunerf/response/stereo_behind_thin_interpolated.npz' --channels 0 1 2

# EUI
python -m sunerf.train.convert_temperature_response_function --response_file '/glade/work/rjarolim/sunerf/response/eui_response_functions.npz' --out_file '/glade/work/rjarolim/sunerf/response/eui_interpolated.npz'

# opacity model
python -m sunerf.train.learn_opacity --data_file '/glade/work/rjarolim/data/sunerf/temperature_response/opacity_table_x0.7_z0.02.txt' --out_file '/glade/work/rjarolim/sunerf/response/opacity.pt'

# NN approximation of the response function
#python -m sunerf.train.learn_temperature_response_function --response_file '/glade/work/rjarolim/data/sunerf/temperature_response/aia_resonses.npz' --out_file '/glade/work/rjarolim/sunerf/response/aia.pt'
#python -m sunerf.train.learn_temperature_response_function --response_file '/glade/work/rjarolim/data/sunerf/temperature_response/stereo_ahead_resonses.npz' --out_file '/glade/work/rjarolim/sunerf/response/stereo_ahead.pt'
#python -m sunerf.train.learn_temperature_response_function --response_file '/glade/work/rjarolim/data/sunerf/temperature_response/stereo_behind_resonses.npz' --out_file '/glade/work/rjarolim/sunerf/response/stereo_behind.pt'
#python -m sunerf.train.learn_temperature_response_function --response_file '/glade/work/rjarolim/data/sunerf/temperature_response/aia_resonses.npz' --out_file '/glade/work/rjarolim/sunerf/response/psi_mean.pt' --channels 2 3 4
#python -m sunerf.train.learn_temperature_response_function --response_file '/glade/work/rjarolim/data/sunerf/temperature_response/aia_resonses.npz' --out_file '/glade/work/rjarolim/sunerf/response/psi_193.pt' --channels 3
