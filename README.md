# Set-up Instructions
## Installing conda environment:

The required conda environment can be created from the environment.yml file in this repository. To do so, run the following command in the terminal:

```console
conda env create -f environment.yml
```

This will create a new conda environment called emoons_focus. To activate the environment, run:

```console
conda activate emoons_focus
```

## Dowloading the test data:

Link to cooldown.4B.01.15 data:

https://www.dropbox.com/scl/fo/k8q5p26zhehkeu64y48o2/ABu7Yt3BlKOkw3Y8JZxtThs?rlkey=9ahigx8fdvorsn3rljt16b0ny&e=1&dl=0

This data from a focus sweep of the YJ2 camera on the 4B.01.15 cooldown.


This data should be saved to a folder called "data/raw/cool4B.01.15"

## Directory structure:

The directory structure before running the code should look like this:

```
config/
    cameraConfig.yaml
    points20240508_160118_YJ2.txt
    .
    .
    .

data/
    raw/
        test_id/
            test_id.camera.ARC.X[dam position].Y[dam position].Zp[dam position].fits
            .
            .
            .
            test_id.camera.DARK.fits
    processed/
planet_fitting.py
frame_analysis.py
focus_finder_gui.py
README.md
environment.yml
```
The config folder should contain the cameraConfig.yaml file and the points files for each camera.

The data folder should contain the raw data in the format specified above. The test ID will be the name of the folder containing the raw frames. This will be used to create the processed folder name. The raw frames should be in the same format as cooldown 4B. The DAM positions should be in the file name as X[dam position].Y[dam position].Zp[dam position].fits, where [dam position] is the position of the dam in the x, y, and z directions respectively. The positive dam positions should be in the format "pXXX", and the negative dam positions should be in the format "n-XXX". The dark frame format is not as strict as it will specified by the user from the command line.

# Running the code

## Frame analysis:

The basic command to analyse the frames and characterise the line shapes is as follows:

```console
python frame_analysis.py path/to/raw/data camera_name -d path/to/dark/frame -v vmin vmax
```

Or in an example:

```console
python frame_analysis.py "data/raw/cool4B.01.15/*ARC*.fits" "YJ2" -d "data/raw/cool4B.01.15/cool4B.01.15.YJ2.DARK.fits" -v 0 1000
```

The path to the data should be a path with wildcards to the raw frames. The camera name should be the name of the camera as specified in the cameraConfig.yaml file. The user is likely to want to specifiy the dark frame to use, and the min and max cuts for the plot and gui colormaps.

Other command line options are available and can be seen by running:

```console
python frame_analysis.py --help
```

## Plane fitting:

To find the best fit focus planes for the data, the following command can be used:

```console
python plane_fitting.py path/to/processed/data --metric metric1 metric2 .. --weights weight1 weight2 ..
```

Or in an example:

```console
python plane_fitting.py "data/processed/YJ2/full_table.csv" --metric "FWHMx" "FWHMy" --weights 1 1
```

The available metrics are FWHMx, FWHMy, FWHMx1D, and FWHMy1D. The user can specify the weights for each metric to be used in the fitting.

As before, other command line options are available and can be seen by running:

```console
python plane_fitting.py --help
```

Some of the command line options likely to be used are the --option (default 4) to specify the dam offsets, the --filt_bounds to specify the in abd max pixel values to be used in the fitting, and the --max_ratio to specify the maximum ratio of the FWHMx to FWHMy to be used in the fitting.

For example:
    
```console
python plane_fitting.py "data/processed/YJ2/full_table.csv" --metric "FWHMx" "FWHMy" --weights 1 1 --option 4 --filt_bounds 2 10 --max_ratio 3
``` 