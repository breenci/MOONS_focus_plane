Installing conda environment:

conda env create -f environment.yml

Link to cooldown.4B.01.15 data:

https://www.dropbox.com/scl/fo/k8q5p26zhehkeu64y48o2/ABu7Yt3BlKOkw3Y8JZxtThs?rlkey=9ahigx8fdvorsn3rljt16b0ny&e=1&dl=0

Command to analyse frames:

python frame_analysis.py "data/raw/cool4B.01.15/*ARC*.fits" "YJ2" -d "data/raw/cool4B.01.15/cool4B.01.15.YJ2.DARK.fits" -v 0 1000

Command to fit planes:

python plane_fitting.py "data/processed/YJ2/full_table.csv" --metric "FWHMx" "FWHMy" --weights 1 1