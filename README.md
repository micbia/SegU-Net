# SegU-Net
Segmentation 2D Convolutional U-Network for Identification of HI regions during the Cosmic Epoch of Reionization in 21-cm Tomography Observations

<img src="https://github.com/micbia/SegU-Net/blob/master/utils_plot/Unet_model.png"> 

<b>Seg U-Net Training Utilization:</b></br>
to train the network on data at you disposal you can change the directory path variable <i>PATH</i> in the initial condition files <i>net.ini</i>, as well as other hypeparameters. The actual data should be stored at this location in a sub-directory called <i>data/</i>.
</br>Then run the following command:</br>

&emsp;&#9654;&emsp; python segUnet.py config/net.ini

If you want to resume a training change the parameters <i>BEST_EPOCH</i> and <i>RESUME_EPOCH</i> in the same initial condition file, the first indicates the epoch of the best saved model, the second is the restarting epoch. These quantities should be both zero if you are starting a new training. You also must provide the output directory <i>RESUME_PATH</i> of the interrupted training (genertaed by the code). Our code save the entire network (weights and layers) so that, in case of resumed trainig the model is already compiled (keras: load_model).

Also, the number of down- and up-sampling levels are automatically scales depending on the images size (between 64 and 128 per side, 4 levels. Above equal 128, 5 levels).

</br>
<b>Seg U-Net Predicts 21cm:</b></br>
to do some predictions with your best trained network, use:</br></br>

&emsp;&#9654;&emsp; python pred_segUnet.py config/pred.ini

in the initial condition file <i>pred.ini</i>, change:</br>
<ul>
  <li><i>PATH_OUT</i> is the directory of the best performing model.</li>
  <li><i>PATH_PREDIC</i> is the path of the data to predict (same structure as in <i>net.ini</i> file).</li>
  <li><i>AUGMENT</i> is the number of times you want to increase your data set (random flip of 90, 180, 270 or 360 along the x, y or z-axis).</li>
</ul> 
