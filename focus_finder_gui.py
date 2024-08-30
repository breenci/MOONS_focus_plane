from mpl_point_clicker import clicker
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.widgets import Cursor, Slider, Button, TextBox
from matplotlib.gridspec import GridSpec
import numpy as np
import glob
import os
import datetime


class pointSelectGUI():
    """Matplotlib GUI for point selection"""
    
    
    def __init__(self, data_cube, point_file=None, DAM_positions=None, box_size=30,
                 vmin=0, vmax=100, output_dir="."):
        # initialize variables
        self.data_cube = data_cube
        self.point_file = point_file
        self.selection = None
        self.DAM_positions = DAM_positions
        self.box_size = box_size
        self.vmin = vmin
        self.vmax = vmax
        self.output_dir = output_dir
                
    def run(self):
        # Specify current array
        self.current_arr = self.data_cube[0]
        
        # Create a figure with grid to plot and show GUI components
        self.fig = plt.figure(figsize=(12,8))
        grid = GridSpec(7, 4)
        # plot the first frame
        imax = self.fig.add_subplot(grid[:5,:3])
        self.im = imax.imshow(self.current_arr, vmin=self.vmin, vmax=self.vmax, 
                              origin='lower')
        # add cursor for easier selection of points
        cursor = Cursor(imax, useblit=True, color='red', linewidth=.5)
        imax.set_aspect('auto')
        # create a clicker object for point selection
        # see https://mpl-point-clicker.readthedocs.io/en/latest/ for more info
        self.klicker = clicker(imax, ['Selected Points'], markers=['x'], 
                               colors=['red'])

        # Load preselected points if specified
        if self.point_file != None:
            try:
                preselection = np.loadtxt(self.point_file)
                self.klicker.set_positions({"Selected Points":preselection})
                self.klicker._update_points()
            except Exception as e:
                print(f"Error: Unable to load the file '{self.point_file}'.")
                print(f"Error message: {e}")
        
        # add a subplot to show most recent point
        subax = self.fig.add_subplot(grid[1:4, 3])
        self.subim = subax.imshow(np.array(np.zeros((self.box_size, self.box_size))), 
                                  vmin=self.vmin, vmax=self.vmax, origin='lower')
        # turn off axis
        subax.axis('off')
        # update the plot whenever a point is added
        self.klicker.on_point_added(self._on_points_changed)      
        
        # add slider for file selection
        # If DAM positions are given, add slider for DAM position selection
        if self.DAM_positions != None:
            DAM_slider_ax = self.fig.add_subplot(grid[5, :])
            DAM_slider = Slider(ax=DAM_slider_ax, valmin=0, valmax=len(self.DAM_positions), valstep=1,
                                label='DAM Position')
            DAM_slider.on_changed(self._slider_update)
        
        else:
            slideax = self.fig.add_subplot(grid[5, :])
            fn_slider = Slider(ax=slideax, valmin=0, valmax=len(self.fn_list), valstep=1,
                            label='File Index')
            fn_slider.on_changed(self._slider_update)
        
        # add button to confim selection
        bax = self.fig.add_subplot(grid[6, :3])
        brun = Button(bax, 'Run Analysis and Close')
        brun.on_clicked(self._run_button_callback)
        
        # Add check box to save points
        bax = self.fig.add_subplot(grid[6, 3])
        bsave = Button(bax, 'Save Points')
        bsave.on_clicked(self._save_button_callback)
        plt.show()
    
    # Callback functions for GUI components
    def _slider_update(self, val):
        """Change the image when the slider is moved"""
        self.im.set_data(self.data_cube[val])
        self.fig.canvas.draw_idle()
        self.current_arr = self.data_cube[val]


    def _run_button_callback(self, event):
        """Close the figure and do final update to postion list"""
        plt.close()
        self.selection = self.klicker.get_positions()
        dirname = self.output_dir
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_fn = os.path.join(dirname, 'points' + timestamp + '.txt')
        np.savetxt(save_fn, self.selection["Selected Points"])
        
        
    def _save_button_callback(self, event):
        """Save the selected points to a file"""
        pos = self.klicker.get_positions()
        dirname = self.output_dir
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_fn = os.path.join(dirname, 'points' + timestamp + '.txt')
        np.savetxt(save_fn, pos["Selected Points"])

 
    def _on_points_changed(self, position, klass):
        """Update the subimage when a point is added"""
        box = self._get_box(position)
        self.subim.set_data(box)
        self.fig.canvas.draw_idle()

    
    def _get_box(self, point):
        """Return a box around the given point"""
        x, y = point
        x = int(x)
        y = int(y)
        box = self.current_arr[y-self.box_size:y+self.box_size, x-self.box_size:x+self.box_size]
        return box
        
            
if __name__ == '__main__':
    fn_list = glob.glob('data/raw/test_3A.01.05/test*.fits')
    print(fn_list)
    gui = pointSelectGUI(fn_list)
    gui.run()
    print(gui.selection)
