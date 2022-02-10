import numpy as np, os
from glob import glob
from PIL import Image

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
