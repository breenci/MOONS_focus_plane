import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.polynomial.polynomial import Polynomial


def DAM_to_mm(DAM_pos, DAM_offsets, DAM_step_size):
    """Conver DAM step positions to 3D coordinates in mm

    :param DAM_pos: _description_
    :type DAM_pos: _type_
    :param DAM_offsets: _description_
    :type DAM_offsets: _type_
    :param DAM_step_size: _description_
    :type DAM_step_size: _type_
    """
    DAM_z = DAM_pos * DAM_step_size + DAM_offsets[2]
    
    DAM_x = DAM_offsets[0] * np.ones_like(DAM_pos)
    DAM_y = DAM_offsets[1] * np.ones_like(DAM_pos)
    
    
    return DAM_x, DAM_y, DAM_z


def plane_from_pnts(p1, p2, p3):
    # Convert points to numpy arrays for vector operations
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Calculate vectors between points
    v1 = p2 - p1
    v2 = p3 - p1

    # Calculate the normal vectors of the planes using cross product
    normal = np.cross(v1, v2)

    # Extract coefficients (A, B, C) from the normal vectors
    A, B, C = normal[:, 0], normal[:, 1], normal[:, 2]

    # Calculate the constant terms (D) in the plane equations
    D = -np.sum(normal * p1, axis=1)

    return A, B, C, D


def find_point_on_plane(A, B, C, D, known_coords, missing_coord='z'):
    """Find the value of the missing coordinate given the coefficients of the plane"""
    
    if missing_coord == 'z':
        x, y = known_coords
        z = (-A * x - B * y - D) / C
        missing_coord_val = z
        
    if missing_coord =='y':
        x, z = known_coords
        y = (-A * x - C * z - D) / B
        missing_coord_val = y
    
    if missing_coord =='x':
        y, z = known_coords
        x = (-B * y - C * z - D) / A
        missing_coord_val = x
        
    return missing_coord_val


# write a function to fit a plane to a set of points
def plane_fitter(point_coords):
    '''Fit a plane to a set of points and return the unit normal.'''

    # make sure points are in a numpy array
    point_coords = np.array(point_coords)
    
    # Subtract the centroid from the set of points
    centroid = np.mean(point_coords, axis=0)
    cntrd_pnts = point_coords - centroid
    
    # Perform singular value decomposition
    svd, _, _ = np.linalg.svd(cntrd_pnts.T)
    
    # Final column of the svd gives the unit normal
    norm = svd[:,2]
    
    # extract the coefficients of the plane
    (A, B, C) = norm
    D = -np.sum(norm * centroid)
    
    return A, B, C, D


def sigma_clip_polyfit(x, y, order, sigma=3, max_iter=5, weights=None):
    '''Fit a polynomial to data and perform sigma clipping on the residuals.'''
    
    # initialise the mask
    mask = np.ones_like(x, dtype=bool)
    
    for i in range(max_iter):
        # fit a polynomial to the data
        poly_fit = Polynomial.fit(x[mask], y[mask], order, w=weights)
        
        # calculate the residuals
        residuals = y - poly_fit(x)
        
        # calculate the standard deviation of the residuals
        std_res = np.std(residuals[mask])
        
        # update the mask
        mask = np.abs(residuals) < sigma * std_res
        
    return poly_fit, mask



def get_score(df, metric_names, weights):
    
    # initilise array to hold weighted metrics
    wmetric_arr = np.zeros((len(df), len(metric_names)))

    for n, name in enumerate(metric_names):
        metric = df[name]/len(metric_names)
        weight = weights[n]
        weighted_metric = metric * weight
        wmetric_arr[:,n] = weighted_metric
        
    # sum the weighted metrics
    score = np.nansum(wmetric_arr, axis=1)
    
    return score
    

if __name__ == "__main__":
    # ---- Constants, config and command line arguments ----
    # 
    # Define the DAM offsets
    # Coordinate system is X, Y are in the plane of the detector and Z is along
    # the direction of travel of the motors
    # options for DAM offsets are:
    # TODO: When this is known add to the config file
    option = 4
    if option == 1:
        DAM_offsets = [[0, 263.5, 0] ,[228.2, -131.7, 0], [-228.2, -131.7, 0]]
    if option == 2:
        DAM_offsets = [[0, 263.5, 0] ,[-228.2, -131.7, 0], [228.2, -131.7, 0]]
    if option == 3:
        DAM_offsets = [[0, -263.5, 0] ,[228.2, 131.7, 0], [-228.2, 131.7, 0]]
    if option == 4:
        DAM_offsets = [[0, -263.5, 0] ,[-228.2, 131.7, 0], [228.2, 131.7, 0]]
    
    
    # TODO: Ad to a config file
    # Convert pixel coordinates to mm
    pixel_size = 0.015 #15 micron pixels
    DAM_step_size = 0.01 # 10 micron steps
    array_centre = (2048, 2048)
    
    # TODO: This should be a cl argument
    # load the data
    line_data = pd.read_csv("full_table.csv")
    
    # convert the dam positions coordinates to mm
    DAMX_x, DAMX_y, DAMX_z = DAM_to_mm(np.array(line_data.loc[:, 'DAM_X']), 
                                       DAM_offsets[0], DAM_step_size)
    DAMY_x, DAMY_y, DAMY_z = DAM_to_mm(np.array(line_data.loc[:, 'DAM_Y']), 
                                       DAM_offsets[1], DAM_step_size)
    DAMZ_x, DAMZ_y, DAMZ_z = DAM_to_mm(np.array(line_data.loc[:, 'DAM_Z']), 
                                       DAM_offsets[2], DAM_step_size)
    
    # convert the pixel coordinates to mm
    Xc_mm = (np.array(line_data.loc[:, 'Xc']) - array_centre[0]) * pixel_size
    Yc_mm = (np.array(line_data.loc[:, 'Yc']) - array_centre[1]) * pixel_size
    line_data['Xc_mm'] = Xc_mm
    line_data['Yc_mm'] = Yc_mm
    
    
    # find the missing coordinate of the line data
    # fit a plane to each set of DAM posistions and find where that plane
    # intersects the poistion of the line in real space
    (A, B, C, D) = plane_from_pnts(np.column_stack((DAMX_x, DAMX_y, DAMX_z)),
                                   np.column_stack((DAMY_x, DAMY_y, DAMY_z)),
                                   np.column_stack((DAMZ_x, DAMZ_y, DAMZ_z)))
    known_coords = (Xc_mm, Yc_mm)
    Zc_mm = find_point_on_plane(A, B, C, D, known_coords, missing_coord='z')
    line_data['Zc_mm'] = Zc_mm
    
    # ---- Metrics ----
    score = get_score(line_data, ['FWHMx', 'FWHMy'], [1, 1])
    line_data['score'] = score
    
    # filter out points with bad FWHM values
    # TODO: Change this to functions
    FWHM_ratio = line_data['FWHMx'] / line_data['FWHMy']
    filtered_data = line_data[(FWHM_ratio > 0.6) & (FWHM_ratio < 1.4)]
    filtered_data = filtered_data[filtered_data['FWHMx'] < 4.5]
    filtered_data = filtered_data[filtered_data['FWHMy'] < 4.5]
    
    # TODO: Do this differently!
    # loop through each point and plot the FWHM
    pnt_max = int(line_data['pnt_id'].max())
    pnt_min = int(line_data['pnt_id'].min())
    
    # initialise arrays to hold the min score and the Zc value at the min score
    min_score = np.zeros(pnt_max - pnt_min + 1)
    Zc_at_min_score = np.zeros(pnt_max - pnt_min + 1)
    Xc_at_min_score = np.zeros(pnt_max - pnt_min + 1)
    Yc_at_min_score = np.zeros(pnt_max - pnt_min + 1)
    pnts = np.arange(pnt_min, pnt_max+1)
    
    # This can also be cleaner!
    fig, ax = plt.subplots(3, 3)
    flat_ax = ax.flatten()
    for pnt in range(pnt_min, pnt_max+1):
        pnt_data = line_data[line_data['pnt_id'] == pnt]
        fltrd_pnt_data = filtered_data[filtered_data['pnt_id'] == pnt]
        
        poly_fit, mask = sigma_clip_polyfit(fltrd_pnt_data['Zc_mm'], 
                                            fltrd_pnt_data['score'], 2, sigma=3,
                                            max_iter=5)
        
        minz_score = poly_fit.deriv().roots()
        score_at_minz = poly_fit(minz_score)
        
        min_score[pnt - pnt_min] = score_at_minz
        Zc_at_min_score[pnt - pnt_min] = minz_score
        
        print(f"Point {pnt}: min score at {minz_score} mm is {score_at_minz}")
        Xc_at_min_score[pnt - pnt_min] = fltrd_pnt_data['Xc_mm'].iloc[np.argmin(fltrd_pnt_data['score'])]
        Yc_at_min_score[pnt - pnt_min] = fltrd_pnt_data['Yc_mm'].iloc[np.argmin(fltrd_pnt_data['score'])]
        
        
        sigma_cliped_data = fltrd_pnt_data[mask]
        
        flat_ax[pnt].plot(pnt_data['Zc_mm'], pnt_data['score'], 'bo',
                       fillstyle='none')
        flat_ax[pnt].plot(fltrd_pnt_data['Zc_mm'], fltrd_pnt_data['score'], 
                          'bo')
        flat_ax[pnt].plot(sigma_cliped_data['Zc_mm'], sigma_cliped_data['score'], 'go')
        flat_ax[pnt].plot(pnt_data['Zc_mm'], poly_fit(pnt_data['Zc_mm']), 'r-')
        
    
    pnt_df = pd.DataFrame({'pnt_id': pnts, 'min_score': min_score, 
                           'Zc_at_min_score': Zc_at_min_score, 
                           'Xc_at_min_score': Xc_at_min_score,
                           'Yc_at_min_score': Yc_at_min_score})
    
    
    # find the best fit plane to the min score points
    A, B, C, D = plane_fitter(np.column_stack((pnt_df['Xc_at_min_score'],
                                               pnt_df['Yc_at_min_score'],
                                               pnt_df['Zc_at_min_score'])))
    
    DAMX_minz = find_point_on_plane(A, B, C, D, DAM_offsets[0][:2], missing_coord='z')
    DAMY_minz = find_point_on_plane(A, B, C, D, DAM_offsets[1][:2], missing_coord='z')
    DAMZ_minz = find_point_on_plane(A, B, C, D, DAM_offsets[2][:2], missing_coord='z')
    
    print('DAMX_minz:', DAMX_minz)
    print('DAMY_minz:', DAMY_minz)
    print('DAMZ_minz:', DAMZ_minz)
     
    # plot the dam positions
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(DAMX_x, DAMX_y, DAMX_z, label='DAM1')
    ax.scatter(DAMY_x, DAMY_y, DAMY_z, label='DAM2')
    ax.scatter(DAMZ_x, DAMZ_y, DAMZ_z, label='DAM3')
    ax.scatter(pnt_df['Xc_at_min_score'], pnt_df['Yc_at_min_score'],
               pnt_df['Zc_at_min_score'], label='min score points')
    plt.show()
    
    
    