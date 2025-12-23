# create conda environment
conda create -n "sunerf" python=3.10.0
conda activate sunerf
# install required packages
cd $HOME/projects/sunerf
pip install -r requirements.txt
# download data
mkdir /glade/work/rjarolim/data/sunerf/psi_data/
gsutil -m cp -r gs://spi3s_nerf_bucket/psi_data /glade/work/rjarolim/data/sunerf/psi_data/
