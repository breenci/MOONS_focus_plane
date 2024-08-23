import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import pandas as pd
import re
import logging
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
from lmfit.models import Gaussian2dModel
import argparse
import os


def extract_variables_and_export(filenames, pattern, column_names=None, sort_by='DAM_X'):
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
    df = df.sort_values(by=sort_by, ignore_index=True)
    df['frame_id'] = np.arange(len(df))
    
    return df


def extract_ROI(datacube, pnts, cube_ids, box_size):
    """This function extracts regions of interest around points in the datacube"""
    n_points = pnts.shape[0]
    n_frames = datacube.shape[0]
    
    # initialize an empty array to store the extracted regions
    n_regions = n_points * n_frames
    regions = np.zeros((n_regions, box_size*2, box_size*2))
    frame_idxs = np.repeat(cube_ids, n_points)
    x = np.zeros(n_regions)
    y = np.zeros(n_regions)
    
    for i in range(n_points):
        x_discrete = int(pnts[i, 0])
        y_discrete = int(pnts[i, 1])
        pnt_ROIS = datacube[:, y_discrete-box_size:y_discrete+box_size,
                            x_discrete-box_size:x_discrete+box_size]
        regions[i::n_points] = pnt_ROIS
        x[i::n_points] = x_discrete
        y[i::n_points] = y_discrete
    
    region_table = pd.DataFrame({'frame_id': frame_idxs, 'x': x, 'y': y})
        
    return regions, region_table


if __name__ == "__main__":
    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Find the best focus position")
    # add an argument for folder containing the data
    parser.add_argument("folder", help="Folder containing the data")
    # folder = "data/raw/cool4B.01.15/*ARC*.fits"
    # add optional arguments for box size, preload selection, and dark
    parser.add_argument("-b", "--box_size", type=int, default=30, help="Size of the box around each point")
    parser.add_argument("-p", "--preload_selection", help="File containing preloaded selection")
    parser.add_argument("-d", "--dark", help="Dark frame to subtract from the data")
    # add an optional argument to specify a vmin and vmax for the images
    parser.add_argument("-v", "--cmap_range", nargs=2, type=int, help='Min and max values for colormap')
    parser.add_argument("-e", "--ext", type=int, default=1, help='FITS extension to read')
    parser.add_argument("--Nlines", type=int, default=3, help='Number of lines to plot')
    # parse the arguments
    args = parser.parse_args()
    
    # regex pattern to extract variables
    pattern = r'\.X(\w{1}\-*\d{3})\.Y(\w{1}\-*\d{3})\.Z(\w{1}\-*\d{3})'
    
    # Get a list of arc fits files in the folder 
    filenames = glob.glob(args.folder)
    # check if any files were found
    if not filenames:
        logger.error(f"No files found in folder: {args.folder}")
        raise FileNotFoundError(f"No files found in folder: {args.folder}")

    # Define custom column names
    custom_column_names = ["DAM_X", "DAM_Y", "DAM_Z"]
    
    logger.info("Extracting DAM positions from filenames....")
    # Call the function and get the extracted variables as a Pandas DataFrame
    extracted_data = extract_variables_and_export(filenames, pattern, 
                                                  column_names=custom_column_names)
    logger.info(f"Extracted {len(extracted_data)} variables from {len(filenames)} filenames")
    
    # sort the DataFrame by the X column
    # implications for analysis later?
    fn_list = extracted_data['filename'].tolist()
    
    # load each file and append to data cube
    xpix, ypix = 4096, 4096
    
    # intialize an empty cube
    cube = np.zeros((len(fn_list), xpix, ypix))
    # load the file and store in the cube
    for i, fn in enumerate(fn_list):
        logger.info(f"Loading file {fn}")
        with fits.open(fn) as hdul:
            cube[i] = hdul[args.ext].data
        
    # dark subtraction
    logger.info("Subtracting dark frame...")
    if args.dark is not None:
        with fits.open(args.dark) as hdul:
            dark_data = hdul[args.ext].data
            dsub_cube = cube - dark_data
    logger.info("Dark subtraction complete.")
    
    # load the points file
    logger.info(f"Loading points file: {args.preload_selection}")
    pnts = np.loadtxt(args.preload_selection)
    
    # extract the regions around the points
    logger.info("Extracting regions around points...")
    ROI_arr, ROI_table = extract_ROI(dsub_cube, pnts, 
                                     extracted_data.loc[:,'frame_id'], args.box_size)
    full_table = pd.merge(extracted_data, ROI_table, on='frame_id')
    logger.info("Regions extracted.")
    
    # loop through each ROI and fit a 2D Gaussian
    # store the fit parameters in a DataFrame
    Xc = np.zeros(len(full_table))
    Yc = np.zeros(len(full_table))
    FWHMx = np.zeros(len(full_table))
    FWHMy = np.zeros(len(full_table))
    
    logger.info("Fitting 2D Gaussian to each region...")
    for i in range(len(full_table)):
        frame = ROI_arr[i]
        X, Y = np.meshgrid(np.arange(frame.shape[0]), np.arange(frame.shape[1]))
        # flatten X, Y and box to guess the parameters
        model = Gaussian2dModel()
        params = model.make_params(amplitude=3000, centerx=30, centery=30, 
                                   sigmax=3, sigmay=3)
        params['centerx'].set(min=0, max=60)
        params['centery'].set(min=0, max=60)
        params['fwhmx'].set(min=2, max=20)
        params['fwhmy'].set(min=2, max=20)
        
        fit_result = model.fit(frame, params, x=X, y=Y)
        Xc[i] = fit_result.params['centerx'].value
        Yc[i] = fit_result.params['centery'].value
        FWHMx[i] = fit_result.params['fwhmx'].value
        FWHMy[i] = fit_result.params['fwhmy'].value
    logger.info("Fitting complete.")
    
    full_table['Xc'] = Xc
    full_table['Yc'] = Yc
    full_table['FWHMx'] = FWHMx
    full_table['FWHMy'] = FWHMy
    
    # save the table to a csv file
    logger.info("Saving table to csv file...")
    full_table.to_csv("full_table.csv", index=False)
    
    # plot the regions
    plot = True
    if plot:
        n_pnts = pnts.shape[0]
        n_frames  = dsub_cube.shape[0]
        with PdfPages("regions.pdf") as pdf:
            for i in range(n_frames):
                num_rows = n_pnts // args.Nlines
                fig, axs = plt.subplots(num_rows, args.Nlines, figsize=(5*args.Nlines, 
                                                                    5*num_rows))
                plt.subplots_adjust(hspace=1.5)
                plt.suptitle(f"Frame: {os.path.basename(fn_list[i])}")
                for j, ax in enumerate(axs.flat):
                    ax.imshow(ROI_arr[i*n_pnts+j], origin='lower', 
                              vmin=args.cmap_range[0], vmax=args.cmap_range[1])
                    ax.scatter(full_table.loc[i*n_pnts+j, 'Xc'], 
                               full_table.loc[i*n_pnts+j, 'Yc'],
                               color='red', s=10)
                    ax.add_patch(Ellipse((full_table.loc[i*n_pnts+j, 'Xc'],
                                          full_table.loc[i*n_pnts+j, 'Yc']),
                                        full_table.loc[i*n_pnts+j, 'FWHMx'],
                                        full_table.loc[i*n_pnts+j, 'FWHMy'],
                                        edgecolor='red', facecolor='none'))
                    ax.set_title(f"Frame {i+1}, Point {j+1}")
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
                
        
    
    
    
    
     