# SegU-Net
Segmentation 3D Convolutional U-Network for Identification of HI regions during Reionization in 21-cm Tomography Observations

<b>Seg U-Net Utilization:</b></br>
to train the network on data at you disposal you can change the directory path, as well as other hypeparameters, in the initial condition files <i>net.ini</i> (copy and name differently). The actual data should be stored at this location in a sub-directory called <i>data/</i>.
</br>Then run the following command:</br>

&emsp;&#9654;&emsp; python segUnet.py config/net.ini

to do some predcitions
