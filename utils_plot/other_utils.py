import numpy as np, os, matplotlib.pyplot as plt
from glob import glob
from PIL import Image

from matplotlib import colors


class MidpointNormalize(colors.Normalize):
    """
    Created by Joe Kington.
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    # set the colormap and centre the colorbar
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def MergeImages(new_image_name, old_image_name, output_path='./', form='v', delete_old=False):
    """ Merge images togheter to create new image.
        Parameters:
            * new_image_name (string): name of the new image
            * old_image_name (string or array): name of the old images, can be a string or an array of strings, if as string then it create a list of paths matching a pathname pattern (attention to the order!)
            * output_path (string): output path save image
            * form (string or tuple): if string it can be 'v' or 'h', otherwise tuplet with the shape to optain
            * delete_old (bool): to delete old images or not
        Returns:
            nothing
    """
    if(isinstance(old_image_name, str)):
        arrsize = len(glob(output_path+old_image_name+'*.png'))
        arr_images = np.array([output_path+old_image_name+str(i)+'.png' for i in range(arrsize)])
    else:
        arr_images = np.array(old_image_name)
    # Open Images
    images = [Image.open(output_path+im_name) for im_name in arr_images]
    height, width, chan = np.shape(images[0])

    # identify which form is desired
    if(isinstance(form, str)):
        if(form == 'v'):
            total_height = height*len(images)
            total_width = width
            x_displ, y_displ = 0, height
            retbool = True
        elif(form == 'h'):
            total_height = height
            total_width = width*len(images)
            x_displ, y_displ = width, 0
            retbool = True
    else:
        total_height = height*form[0]
        total_width = width*form[1]
        x_displ, y_displ = width, height
        retbool = False

    # Create new empty image
    new_im = Image.new('RGB', size=(
        total_width, total_height), color=(255, 255, 255, 0))

    # Start paste old images on new empty image
    x_offset, y_offset = 0, 0
    if(retbool):
        for im in images:
            new_im.paste(im, (x_offset, y_offset))
            x_offset += x_displ
            y_offset += y_displ
    else:
        idx = 0
        for h in range(form[0]):
            for v in range(form[1]):
                new_im.paste(images[idx], (x_offset, y_offset))
                x_offset += x_displ
                idx += 1
            x_offset = 0
            y_offset += y_displ

    # Save new image
    new_im.save('%s.png' % (output_path+new_image_name))

    # Delete old image if required
    if(delete_old):
        if isinstance(old_image_name, (np.ndarray, list)):
            for im in old_image_name:
                os.system("rm %s" % (output_path+im))
        else:
            os.system("rm %s*.png" % (output_path+old_image_name))
    else:
        # old images not deleted.
        pass


class adjust_axis:
    def __init__(self, axis, varr, xy, to_round=10, step=5, fmt=int):
        self.axis = axis
        self.varr = varr
        self.to_round = to_round
        self.step = step
        self.fmt = fmt
        
        loc_f = self.get_axis_locs()
        if(xy == 'x'):
            plt.xticks(loc_f)
            axis.set_xticklabels([int(round(varr[i_n])) for i_n in loc_f])
        elif(xy == 'y'):
            plt.yticks(loc_f)
            axis.set_yticklabels([int(round(varr[i_n])) for i_n in loc_f])
        
    def get_axis_locs(self):    
        v_max = int(round(self.varr.max()/self.to_round)*self.to_round) if int(round(self.varr.max()/self.to_round)*self.to_round) <= self.varr.max() else int(round(self.varr.max()/self.to_round)*self.to_round)-self.to_round
        v_min = int(round(self.varr.min()/self.to_round)*self.to_round) if int(round(self.varr.min()/self.to_round)*self.to_round) >= self.varr.min() else int(round(self.varr.min()/self.to_round)*self.to_round)+self.to_round
        v_plot = np.arange(v_min, v_max+self.step, self.step)
        loc_v = np.array([np.argmin(abs(self.varr-v_plot[i])) for i in range(v_plot.size)]).astype(self.fmt)
        return loc_v