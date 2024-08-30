import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import pandas as pd
import re
import logging
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
from lmfit.models import GaussianModel, Gaussian2dModel
import argparse
import os
import yaml
from focus_finder_gui import pointSelectGUI


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
                elif capture_group[0] == 'm':
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
    pnt_id = np.zeros(n_regions)
    
    for i in range(n_points):
        x_discrete = int(pnts[i, 0])
        y_discrete = int(pnts[i, 1])
        pnt_ROIS = datacube[:, y_discrete-box_size:y_discrete+box_size,
                            x_discrete-box_size:x_discrete+box_size]
        regions[i::n_points] = pnt_ROIS
        x[i::n_points] = x_discrete
        y[i::n_points] = y_discrete
        pnt_id[i::n_points] = i
    
    region_table = pd.DataFrame({'frame_id': frame_idxs, 'pnt_id': pnt_id, 
                                 'x': x, 'y': y})
        
    return regions, region_table


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Find the best focus position")
    # add an argument for folder containing the data
    parser.add_argument("folder", help="Folder containing the data")
    parser.add_argument("camera", type=str, help="Camera name")
    # folder = "data/raw/cool4B.01.15/*ARC*.fits"
    # add optional arguments for box size, preload selection, and dark
    parser.add_argument("-b", "--box_size", type=int, default=30, help="Size of the box around each point")
    parser.add_argument("-p", "--preload_selection", help="File containing preloaded selection")
    parser.add_argument("-d", "--dark", help="Dark frame to subtract from the data")
    # add an optional argument to specify a vmin and vmax for the images
    parser.add_argument("-v", "--cmap_range", nargs=2, type=int, help='Min and max values for colormap')
    parser.add_argument("-e", "--ext", type=int, help='FITS extension to read')
    parser.add_argument("--Nlines", type=int, default=3, help='Number of lines to plot')
    parser.add_argument("--save_folder", "-s", help="Folder to save the output files")
    parser.add_argument("--log", help="Set the logging level", default="INFO")
    parser.add_argument("--config", help="Path to the configuration file", default="config/cameraConfig.yml")
    parser.add_argument("--plot", help="Produce a plot if set", default=True)
    # parse the arguments
    args = parser.parse_args()
    
    # open the configuration file
    with open(args.config, 'r') as con:
        config = yaml.safe_load(con)
        
    # default save folder
    in_folder = os.path.dirname(args.folder)
    test_id = os.path.basename(in_folder)
    if args.save_folder is None:
        folder_name = test_id + "." + args.camera + "/"
        folder_path = "data/processed/" + folder_name
        # create a folder to save the output files
        os.makedirs(folder_path, exist_ok=True)
        args.save_folder = folder_path
    
    # Logging set up
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=numeric_level, 
                        filename=args.save_folder + "frame_analysis.log", 
                        filemode='w')
    # also log to the console
    logging.getLogger().addHandler(logging.StreamHandler())
    # Note the time in the log file
    logger.info("Frame analysis started at: " + str(pd.Timestamp.now()))
    
    # load info from the configuration file if not provided
    if args.ext is None:
        args.ext = config[args.camera]['ext']
    
    if args.preload_selection is None:
        # check if there is a default selection in the config file
        if 'pnts_path' not in config[args.camera]:
            logger.info("No default selection provided")
            args.preload_selection = None
        else:
            args.preload_selection = config[args.camera]['pnts_path']
    
    # TODO: Slit position?
    # regex pattern to extract variables
    pattern = r'\.X(\w{1}\-*\d{3})\.Y(\w{1}\-*\d{3})\.Z(\w{1}\-*\d{3})'
    # pattern = r'\.X(\w{1}\d{3})\.Y(\w{1}\d{3})\.Z(\w{1}\d{3})'
    
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
    xpix = config[args.camera]['xpix']
    ypix = config[args.camera]['ypix']
    
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
            logger.info(f"Loading dark frame: {args.dark}")
            dark_data = hdul[args.ext].data
            dsub_cube = cube - dark_data
            logger.info("Dark subtraction complete.")
    else:
        logger.info("No dark frame provided. Skipping dark subtraction.")
        
    
    # initialize the GUI
    logger.info("Initializing GUI...")
    gui = pointSelectGUI(dsub_cube, point_file=args.preload_selection, 
                            DAM_positions=extracted_data['DAM_X'].tolist(), 
                            box_size=args.box_size, vmin=args.cmap_range[0], 
                            vmax=args.cmap_range[1], output_dir=args.save_folder)
    gui.run()
    pnts = gui.selection["Selected Points"]
    logger.info("Loading points from GUI...")
    logger.info(gui.selection)

    # extract the regions around the points
    logger.info("Extracting regions around points...")
    logger.info(f"Box size: {args.box_size}")
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
    FWHMx1D = np.zeros(len(full_table))
    FWHMy1D = np.zeros(len(full_table))
    
    model2D = Gaussian2dModel()
    model1D = GaussianModel()
    
    logger.info("Fitting 2D Gaussian to each region...")
    # TODO: Clean up the fitting code
    box1D_size = config[args.camera]['box1D_size']
    for i in range(len(full_table)):
        frame = ROI_arr[i]
        X, Y = np.meshgrid(np.arange(frame.shape[0]), np.arange(frame.shape[1]))
        # flatten X, Y and box to guess the parameters
        
        params2D = model2D.make_params(amplitude=3000, centerx=30, centery=30, 
                                   sigmax=3, sigmay=3)
        params2D['centerx'].set(min=box1D_size, max=args.box_size*2 - box1D_size)
        params2D['centery'].set(min=box1D_size, max=args.box_size*2 - box1D_size)
        params2D['fwhmx'].set(min=2, max=20)
        params2D['fwhmy'].set(min=2, max=20)
        
        fit_result2D = model2D.fit(frame, params2D, x=X, y=Y)
        Xc[i] = fit_result2D.params['centerx'].value + int(full_table.loc[i, 'x']) - args.box_size
        Yc[i] = fit_result2D.params['centery'].value + int(full_table.loc[i, 'y']) - args.box_size
        FWHMx[i] = fit_result2D.params['fwhmx'].value
        FWHMy[i] = fit_result2D.params['fwhmy'].value
        
        # find a box around the fit centre
        x_min = int(fit_result2D.params['centerx'].value) - box1D_size//2
        x_max = int(fit_result2D.params['centerx'].value) + box1D_size//2
        y_min = int(fit_result2D.params['centery'].value) - box1D_size//2
        y_max = int(fit_result2D.params['centery'].value) + box1D_size//2
        
        # collapse the box in the x and y directions
        x_profile = np.sum(frame[y_min:y_max, x_min:x_max], axis=0)    
        y_profile = np.sum(frame[y_min:y_max, x_min:x_max], axis=1)
        
        # fit a 1D Gaussian to the collapsed profiles
        params1D = model1D.guess(x_profile, x=np.arange(box1D_size))
        
        fit_result1D_x = model1D.fit(x_profile, params1D, x=np.arange(box1D_size))
        fit_result1D_y = model1D.fit(y_profile, params1D, x=np.arange(box1D_size))
        
        FWHMx1D[i] = fit_result1D_x.params['fwhm'].value
        FWHMy1D[i] = fit_result1D_y.params['fwhm'].value
        
        
    logger.info("Fitting complete.")
    
    full_table['Xc'] = Xc
    full_table['Yc'] = Yc
    full_table['FWHMx'] = FWHMx
    full_table['FWHMy'] = FWHMy
    full_table['FWHMx1D'] = FWHMx1D
    full_table['FWHMy1D'] = FWHMy1D
    
    # save the table to a csv file
    logger.info("Saving table to csv file...")
    full_table.to_csv(args.save_folder + "full_table.csv", index=False)
    logger.info(f"Table saved at {args.save_folder}")
    
    # plot the regions
    # TODO: Plotting function
    if args.plot:
        logger.info("Plotting regions...")
        n_pnts = pnts.shape[0]
        n_frames  = dsub_cube.shape[0]
        with PdfPages(args.save_folder + "line_plots.pdf") as pdf:
            for i in range(n_frames):
                num_rows = n_pnts // args.Nlines
                fig, axs = plt.subplots(num_rows, args.Nlines, figsize=(5*args.Nlines, 
                                                                    3*num_rows))
                plt.subplots_adjust(hspace=1.5, wspace=1.5)
                for j, ax in enumerate(axs.flat):
                    Xc_in_box = full_table.loc[i*n_pnts+j, 'Xc'] - full_table.loc[i*n_pnts+j, 'x'] + args.box_size
                    Yc_in_box = full_table.loc[i*n_pnts+j, 'Yc'] - full_table.loc[i*n_pnts+j, 'y'] + args.box_size
                    ax.imshow(ROI_arr[i*n_pnts+j], origin='lower', 
                              vmin=args.cmap_range[0], vmax=args.cmap_range[1])
                    ax.scatter(Xc_in_box, Yc_in_box, color='red', s=10)
                    ax.add_patch(Ellipse((Xc_in_box, Yc_in_box), 
                                         full_table.loc[i*n_pnts+j, 'FWHMx'],
                                         full_table.loc[i*n_pnts+j, 'FWHMy'],
                                         edgecolor='red', facecolor='none'))
                    ax.set_title(f'Fit Centre: {full_table.loc[i*n_pnts+j, "Xc"]:.2f}, {full_table.loc[i*n_pnts+j, "Yc"]:.2f}')
                    ax.axis('off')
                plt.suptitle(f"Frame: {os.path.basename(fn_list[i])}")
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
    logger.info("Regions plotted.")        
        
    
    
    
    
     