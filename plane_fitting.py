import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    
    

if __name__ == "__main__":
    # Define the DAM offsets
    # Coordinate system is X, Y are in the plane of the detector and Z is along
    # the direction of travel of the motors
    # options for DAM offsets are:
    option = 4
    if option == 1:
        DAM_offsets = [[0, 263.5, 0] ,[228.2, -131.7, 0], [-228.2, -131.7, 0]]
    if option == 2:
        DAM_offsets = [[0, 263.5, 0] ,[-228.2, -131.7, 0], [228.2, -131.7, 0]]
    if option == 3:
        DAM_offsets = [[0, -263.5, 0] ,[228.2, 131.7, 0], [-228.2, 131.7, 0]]
    if option == 4:
        DAM_offsets = [[0, -263.5, 0] ,[-228.2, 131.7, 0], [228.2, 131.7, 0]]
    
    
    # Convert pixel coordinates to mm
    pixel_size = 0.015 #15 micron pixels
    DAM_step_size = 0.01 # 10 micron steps
    array_centre = (2048, 2048)
    
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
    
    (A, B, C, D) = plane_from_pnts(np.column_stack((DAMX_x, DAMX_y, DAMX_z)),
                                   np.column_stack((DAMY_x, DAMY_y, DAMY_z)),
                                   np.column_stack((DAMZ_x, DAMZ_y, DAMZ_z)))
    
    # find the missing coordinate
    missing_coord = 'z'
    known_coords = (Xc_mm, Yc_mm)
    Zc_mm = find_point_on_plane(A, B, C, D, known_coords, missing_coord)
    line_data['Zc_mm'] = Zc_mm
    
    # loop through each point and plot the FWHM
    pnt_max = int(line_data['pnt_id'].max())
    pnt_min = int(line_data['pnt_id'].min())
    
    fig, ax = plt.subplots(3, 3)
    flat_ax = ax.flatten()
    for pnt in range(pnt_min, pnt_max+1):
        pnt_data = line_data[line_data['pnt_id'] == pnt]
        flat_ax[pnt-1].scatter(pnt_data['Zc_mm'], pnt_data['FWHMx'])
    plt.show()
        
     
    # plot the dam positions
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(DAMX_x, DAMX_y, DAMX_z, label='DAM1')
    ax.scatter(DAMY_x, DAMY_y, DAMY_z, label='DAM2')
    ax.scatter(DAMZ_x, DAMZ_y, DAMZ_z, label='DAM3')
    plt.show()
    
    
    