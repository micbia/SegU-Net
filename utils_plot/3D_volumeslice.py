import numpy as np
from mayavi import mlab
from other_utils import read_cbin

size = 1
fname = '19-02T22-47-38/outputs/pred_val_image_21cm.bin'
data = read_cbin(fname)

print(data.shape)
x, y, z = np.mgrid[0:128:64j, 0:128:64j, 0:128:64j]
mlab.volume_slice(x, y, z, data, plane_opacity=0., transparent=True)
#mlab.figure(bgcolor=(0,0,0))
mlab.outline()
mlab.show()
mlab.clf()
