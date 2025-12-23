module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

# prepare temperature response functions
# SDO/AIA
python -m sunerf.data.euv.load_aia_response_function --response_file '/glade/work/rjarolim/sunerf/response/aia_euv_resp.sav' --out_path '/glade/work/rjarolim/sunerf/response_v02/aia_response_functions.npz'
# SolO/EUI
python -m sunerf.data.euv.load_euvi_response_function --response_file '/glade/work/rjarolim/sunerf/response/ahead_sre_chianti2_fludra_mazzotta_002.geny' --out_path '/glade/work/rjarolim/sunerf/response_v02/stereo_ahead_response.npz'
python -m sunerf.data.euv.load_euvi_response_function --response_file '/glade/work/rjarolim/sunerf/response/behind_sre_chianti2_fludra_mazzotta_002.geny' --out_path '/glade/work/rjarolim/sunerf/response_v02/stereo_behind_response.npz'


