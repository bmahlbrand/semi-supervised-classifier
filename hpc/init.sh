source loadmodules.sh
module list
conda env create -f environment.yml
bash extract_dataset.sh
bash config_symbolic_link.sh
