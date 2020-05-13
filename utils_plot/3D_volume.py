import numpy as np
from mayavi import mlab
from Xfrac import XfracFile

fname = 'data/xfrac3d_7.305.bin'
#fname = 'xfrac3d_10.023.bin'

x, y, z = np.ogrid[0:500:300j, 0:500:300j, 0:500:300j]
data = XfracFile(fname).xi

#mlab.init_notebook()
#f0 = mlab.figure(size=(600, 500), bgcolor=(0,0,0))
#f0 = mlab.figure(bgcolor=(0,0,0))
#mlab.clf()

src = mlab.pipeline.scalar_field(data)
vol = mlab.pipeline.volume(src)#, vmin=0.5, vmax=1)

#mlab.draw()
mlab.outline()
mlab.axes(ranges=[0, 500, 0, 500, 0, 500], nb_labels=6, xlabel='x', ylabel='y', zlabel='z')
mlab.view(azimuth=40, elevation=70, distance=1000, focalpoint=[150,120,100])
#mlab.savefig('test.png')
mlab.show()
