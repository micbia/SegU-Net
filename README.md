# SegU-Net
Segmentation 3D Convolutional U-Network for Identification of HI regions during Reionization in 21-cm Tomography Observations

<b>Seg U-Net Utilization:</b></br>
to train the network on data at you disposal you can change the directory path variable <i>PATH</i> in the initial condition files <i>net.ini</i>, as well as other hypeparameters. The actual data should be stored at this location in a sub-directory called <i>data/</i>.
</br>Then run the following command:</br>

&emsp;&#9654;&emsp; python segUnet.py config/net.ini

If you want to resume a training change the parameters <i>BEST_EPOCH</i> and <i>RESUME_EPOCH</i> in the same initial condition file, these quantities should be both zero if you a starting a new training. You also must provide the output directory <i>RESUME_PATH</i> of the interrupted training (genertaed by the code).</br>
Our code save the entire network (weights and layers) so that, in case of resumed trainig the model is already compiled (keras: load_model).</br>
Also, the number of down- and up-sampling levels are automatically scaled depending on the images size (between 64 and 128 per side, 3 levels. Above equal 128, 4 levels).

to do some predcitions:

&emsp;&#9654;&emsp; python pred_segUnet.py config/pred.ini

in the initial condition file <i>pred.ini</i>, change 

