import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import pandas as pd
import re
import logging
from matplotlib.backends.backend_pdf import PdfPages


def extract_variables_and_export(filenames, pattern, column_names=None):
    """Does a regex search on filenames and extracts variables based on pattern

    :param filenames: List of filenames to be searched
    :type filenames: list
    :param pattern: Regular expression pattern to extract variables
    :type pattern: regex string
    :param column_names: pandas column names for each variable, defaults to None
    :type column_names: list of strings, optional
    :return: pandas DataFrame with extracted variables and filenames
    :rtype: pandas DataFrame
    """
    # Initialize list to store extracted variables
    variable_data = []

    # Loop through each filename and extract variables using the pattern
    for filename in filenames:
        match = re.search(pattern, filename)
        if match:
            variables = []
            for capture_group in match.groups():
                if capture_group[0] == 'p':
                    variables.append(int(capture_group[1:]))
                elif capture_group[0] == 'n':
                    variables.append(-int(capture_group[2:]))
                else:
                    variables.append(capture_group)
                    
            variable_data.append(variables)          

    # Get the number of variables
    num_variables = match.lastindex if match else 0

    # Create a Pandas DataFrame
    if column_names is None:
        column_names = [f"Variable_{i}" for i in range(1, num_variables + 1)]
    else:
        if len(column_names) != num_variables:
            raise ValueError(f"Number of column names ({len(column_names)}) must match the number of variables ({num_variables})")

    df = pd.DataFrame(variable_data, columns=column_names)
    df['filename'] = filenames

    return df

def extract_ROI(datacube, pnts, box_size):
    """This function extracts regions of interest around points in the datacube"""
    n_points = pnts.shape[0]
    n_frames = datacube.shape[0]
    
    # initialize an empty array to store the extracted regions
    n_regions = n_points * n_frames
    regions = np.zeros((n_regions, box_size*2, box_size*2))
    
    for i in range(n_points):
        x_discrete = int(pnts[i, 0])
        y_discrete = int(pnts[i, 1])
        pnt_ROIS = datacube[:, y_discrete-box_size:y_discrete+box_size,
                            x_discrete-box_size:x_discrete+box_size]
        regions[i::n_points] = pnt_ROIS
        
    return regions


if __name__ == "__main__":
    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    
    # define some parameters
    # folder with test data
    folder = "data/raw/cool4B.01.15/*ARC*.fits"
    # dark frame to subtract
    dark_frame = "data/raw/cool4B.01.15/cool4B.01.15.YJ2.DARK.fits"
    # fits extension to load
    ext = 2
    # size of analysis region
    box_size = 30 # pixels
    # points file
    pnts_file = "data/raw/cool4B.01.15/points20240527_143953.txt"
    # plotting cuts
    vmin, vmax = 0, 1000
    # number of lines to plot
    n_lines = 3
    
    # regex pattern to extract variables
    pattern = r'\.X(\w{1}\-*\d{3})\.Y(\w{1}\-*\d{3})\.Z(\w{1}\-*\d{3})'
    
    # Get a list of arc fits files in the folder 
    filenames = glob.glob(folder)
    print(len(filenames))
    # check if any files were found
    if not filenames:
        logger.error(f"No files found in folder: {folder}")
        raise FileNotFoundError(f"No files found in folder: {folder}")

    # Define custom column names (optional)
    custom_column_names = ["X", "Y", "Z"]
    
    logger.info("Extracting DAM positions from filenames....")
    # Call the function and get the extracted variables as a Pandas DataFrame
    extracted_data = extract_variables_and_export(filenames, pattern, 
                                                  column_names=custom_column_names)
    logger.info(f"Extracted {len(extracted_data)} variables from {len(filenames)} filenames")
    
    # sort the DataFrame by the X column
    # implications for analysis later?
    extracted_data = extracted_data.sort_values(by="X")
    fn_list = extracted_data['filename'].tolist()
    
    # load each file and append to data cube
    xpix, ypix = 4096, 4096
    
    # intialize an empty cube
    cube = np.zeros((len(fn_list), xpix, ypix))
    # load the file and store in the cube
    for i, fn in enumerate(fn_list):
        logger.info(f"Loading file {fn}")
        with fits.open(fn) as hdul:
            cube[i] = hdul[ext].data
        
    # dark subtraction
    logger.info("Subtracting dark frame...")
    if dark_frame is not None:
        with fits.open(dark_frame) as hdul:
            dark_data = hdul[ext].data
            dsub_cube = cube - dark_data
    logger.info("Dark subtraction complete.")
    
    # load the points file
    logger.info(f"Loading points file: {pnts_file}")
    pnts = np.loadtxt(pnts_file)
    
    # extract the regions around the points
    logger.info("Extracting regions around points...")
    ROI_arr = extract_ROI(dsub_cube, pnts, box_size)
    logger.info("Regions extracted.")
    
    # plot the regions
    n_pnts = pnts.shape[0]
    n_frames  = dsub_cube.shape[0]
    with PdfPages("regions.pdf") as pdf:
        for i in range(n_frames):
            num_rows = n_pnts // n_lines
            fig, axs = plt.subplots(num_rows, n_lines, figsize=(5*n_lines, 
                                                                5*num_rows))
            plt.subplots_adjust(hspace=1.5)
            
            for j, ax in enumerate(axs.flat):
                ax.imshow(ROI_arr[i*n_pnts+j], origin='lower', vmin=vmin, 
                          vmax=vmax)
                ax.set_title(f"Frame {i+1}, Point {j+1}")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
                
        
    
    
    
    
     