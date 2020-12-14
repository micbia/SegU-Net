import zipfile, math, random, numpy as np, os, tools21cm as t2c

from datetime import datetime
from glob import glob
from keras.utils import Sequence
from tqdm import tqdm


class DataGenerator(Sequence):
    """
    Data generator of 3D data (calculate noise cube and smooth data).
    """
    def __init__(self, path='./', data_temp=None, batch_size=None, zipf=False,
                 data_shape=None, tobs=1000, shuffle=True):
        """
        Arguments:
         tobs: int
                observational time, for noise calcuation.
        """
        self.path = path
        self.indexes = data_temp
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.zipf = zipf

        self.data_size = len(self.indexes)
        self.on_epoch_end()
        
        # i, z, eff_fact, Rmfp, Tvir, np.mean(xn)
        if(self.path[-3:] == 'zip'):
            with zipfile.ZipFile(self.path) as myzip:
                self.astro_par = np.loadtxt(myzip.open('%s/astro_params.txt' %(self.path[self.path[:-5].rfind('/')+1:-4])), unpack=True)
                with myzip.open('%s/user_params.txt' %(self.path[self.path[:-5].rfind('/')+1:-4])) as myfile:
                    self.user_par = eval(myfile.read())
        else:
            self.astro_par = np.loadtxt(self.path+'astro_params.txt', unpack=True)
            with open(self.path+'user_params.txt','r') as f:
                self.user_par = eval(f.read())
        
        if(self.zipf):
            self.path_uvcov = '/home/michele/Documents/PhD_Sussex/output/ML/dataset/inputs/uv_coverage_%d.zip' %(self.user_par['HII_DIM'])
            #self.path_uvcov = '%s../uv_coverage_%d.zip' %(self.path, self.user_par['HII_DIM'])
        else:
            self.path_uvcov = '/home/michele/Documents/PhD_Sussex/output/ML/dataset/inputs/uv_coverage_%d/' %(self.user_par['HII_DIM'])
            #self.path_uvcov = '%s../uv_coverage_%d/' %(self.path, self.user_par['HII_DIM'])

        if tobs:
            self.tobs = tobs
        else:
            raise ValueError('Set observation time: tobs')


    def __len__(self):
        # number of batches
        return int(np.floor(self.data_size//self.batch_size))


    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        X = np.zeros((np.append(self.batch_size, self.data_shape)))
        y = np.zeros((np.append(self.batch_size, self.data_shape)))

        for i, idx in enumerate(indexes):
            if(self.zipf):
                if(len(self.data_shape) == 3):
                    for var in glob('%s*part*.zip' %self.path):
                        try:
                            with zipfile.ZipFile(var, 'r') as myzip:
                                f = myzip.extract(member='%s/data/dT1_21cm_i%d.bin' %(var[var[:-5].rfind('/')+1:-4], idx), path=self.path)
                                dT = t2c.read_cbin(f) 
                                f = myzip.extract(member='%s/data/xH_21cm_i%d.bin' %(var[var[:-5].rfind('/')+1:-4], idx), path=self.path)
                                xH = t2c.read_cbin(f) 
                                os.system('rm -r %s/' %var[:-4])  
                                break
                        except:
                            pass
                elif(len(self.data_shape) == 2):
                    with zipfile.ZipFile(self.path, 'r') as myzip:
                        dT = np.load(myzip.open('%s/data/image_21cm_i%d.npy' %(self.path[self.path[:-5].rfind('/')+1:-4], i)))
                        xH = np.load(myzip.open('%s/data/mask_21cm_i%d.npy' %(self.path[self.path[:-5].rfind('/')+1:-4], i)))
            else:
                dT = t2c.read_cbin('%sdata/dT1_21cm_i%d.bin' %(self.path, idx))
                xH = t2c.read_cbin('%sdata/xH_21cm_i%d.bin' %(self.path, idx))
        
            #X[i] = self._noise_smt_dT(dT1=dT, idx=idx)
            #y[i] = self._smt_xH(xH_box=xH, idx=idx)
        
        X = X[..., np.newaxis]
        y = y[..., np.newaxis]
        
        return X, y


    def _noise_smt_dT(self, dT1, idx):
        assert idx == self.astro_par[0,idx]
        z = self.astro_par[1, idx]

        # calculate uv-coverage 
        if(self.zipf):
            with zipfile.ZipFile(self.path_uvcov) as myzip: 
                with myzip.open('uv_coverage_%d/uvmap_z%.3f.npy' %(self.user_par['HII_DIM'], z)) as myfile1: 
                    uv = np.load(myfile1)
                with myzip.open('uv_coverage_%d/Nantmap_z%.3f.npy' %(self.user_par['HII_DIM'], z)) as myfile2: 
                    Nant = np.load(myfile2)
        else:
            file_uv = '%suvmap_z%.3f.npy' %(self.path_uvcov, z)
            file_Nant = '%sNantmap_z%.3f.npy' %(self.path_uvcov, z)

            if(os.path.exists(file_uv) and os.path.exists(file_Nant)):
                    uv = np.load(file_uv)
                    Nant = np.load(file_Nant)
            else:
                #SKA-Low 2016 configuration
                uv, Nant = t2c.get_uv_daily_observation(self.user_par['HII_DIM'], z, filename=None,
                                                        total_int_time=6.0, int_time=10.0,
                                                        boxsize=self.user_par['BOX_LEN'],
                                                        declination=-30.0, verbose=False)
                np.save(file_uv, uv)
                np.save(file_Nant, Nant)

        # calculate Noise cube
        random.seed(datetime.now())
        noise_cube = t2c.noise_cube_coeval(self.user_par['HII_DIM'], z, depth_mhz=None,
                                        obs_time=self.tobs, filename=None, boxsize=self.user_par['BOX_LEN'],
                                        total_int_time=6.0, int_time=10.0, declination=-30.0, 
                                        uv_map=uv, N_ant=Nant, fft_wrap=False, verbose=False)

        dT3 = t2c.smooth_coeval(dT1+noise_cube, z, 
                                box_size_mpc=self.user_par['HII_DIM'],
                                max_baseline=2.0, ratio=1.0, nu_axis=2)

        return dT3


    def _smt_xH(self, xH_box, idx):
        assert idx == self.astro_par[0,idx]
        z = self.astro_par[1, idx]

        smt_xn = t2c.smooth_coeval(xH_box, z, box_size_mpc=self.user_par['HII_DIM'], max_baseline=2.0, ratio=1.0, nu_axis=2)
        mask_xn = smt_xn>0.5

        return mask_xn.astype(int)


class RotateGenerator(Sequence):
    """
    Data generator of 3D data (only flip and rotate).

    Note:
        At the moment only one type of augmentation can be applied for one generator.
    """
    def __init__(self, data=None, label=None, batch_size=None, 
                 rotate_axis=None, rotate_angle=None, shuffle=True):
        """
        Arguments:
         flip_axis: int(0, 1, 2 or 3) or 'random'
                Integers 1, 2 and 3 mean x axis, y axis and z axis for each.
                Axis along which data is flipped.
         rotate_axis: int(1, 2 or 3) or 'random'
                Integers 1, 2 and 3 mean x axis, y axis and z axis for each.
                Axis along which data is rotated.
         rotate_angle: int or 'random'
                Angle by which data is rotated along the specified axis.
        """
        self.data = data[...,0]
        self.label = label[...,0]
        self.batch_size = batch_size
        self.rotate_axis = rotate_axis
        self.rotate_angle = rotate_angle
        self.shuffle = shuffle
        self.data_shape = self.data.shape[1:]
        self.data_size = self.data.shape[0]
        self.on_epoch_end()

        self.idx_list = np.array(range(self.data_size))

        if self.label is not None:
            self.exist_label = True
        else:
            self.exist_label = False
        

    def __len__(self):
        # number of batches
        return self.data_size//self.batch_size


    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.idx_list = np.array(range(self.data_size))
        
        if self.shuffle == True:
            np.random.shuffle(self.idx_list)
    

    def __getitem__(self, index):
        indexes = self.idx_list[index*self.batch_size:(index+1)*self.batch_size]

        # Specify axis of rotation (x, y or z)
        if self.rotate_axis in (1, 2, 3):
            rot_axis = [self.rotate_axis]*self.batch_size
        elif self.rotate_axis == 'random':
            rot_axis = [random.randint(1, 3) for i in range(self.batch_size)]
        else:
            raise ValueError('Rotate axis should be 1, 2, 3 or random')
        
        # Specify angle of rotation (90, 180, 270, 360 degree)
        if isinstance(type(self.rotate_angle), (int, float)):
            rotation = [self.rotate_angle]*self.batch_size
        elif(self.rotate_angle == 'random'):
            rotation = [random.choice([90, 180, 270, 360])//90 for i in range(self.batch_size)]
        else:
            raise ValueError('Rotate angle should be 90, 180, 270, 360 or random')
        
        X = np.zeros((np.append(self.batch_size, self.data_shape)))
        y = np.zeros((np.append(self.batch_size, self.data_shape)))

        for i, idx in enumerate(indexes):
            X[i] = self._rotate_data(self.data[idx], rot=rotation[i], rotax=rot_axis[i])
            y[i] = self._rotate_data(self.label[idx], rot=rotation[i], rotax=rot_axis[i])

        X = X[..., np.newaxis]
        y = y[..., np.newaxis]
        
        return X, y


    def _rotate_data(self, data, rot, rotax):
        if(len(self.data_shape) == 3):
            if rotax == 1:
                ax_tup = (1, 2)
            elif rotax == 2:
                ax_tup = (2, 0)
            elif rotax == 3:
                ax_tup = (0, 1)
            else:
                raise ValueError('rotate axis should be 1, 2 or 3')
            rotated_data = np.rot90(data, k=rot, axes=ax_tup)
        elif(len(self.data_shape) == 2):
            rotated_data = np.rot90(data, k=rot)
        
        return rotated_data



import tools21cm as t2c, py21cmfast as p21c
from datetime import datetime

class LightConeGenerator(Sequence):
    """
    Data generator of 3D data (calculate noise cube and smooth data).
    """
    def __init__(self, data_shape, uvpath='./', z_min=7., , tobs=1000, shuffle=False):
        """
        Arguments:
         tobs: int
                observational time, for noise calcuation.
        """

        self.user_params = {'HII_DIM':128, 'DIM':384, 'BOX_LEN':256}
        self.cosmo_params = {'OMm':0.27, 'OMb':0.046, 'SIGMA_8':0.82, 'POWER_INDEX':0.96}
        
        self.data_shape = data_shape
        self.data_size = 648

        self.path = path
        self.z_min = z_min
        self.z_max = z_max
        self.tobs = tobs
        self.uvfile = uvpath
        self.data_shape = (params['HII_DIM'], params['HII_DIM'])

        self.data_size = len(self.indexes)
        self.on_epoch_end()
        

    def __len__(self):
        # number of batches
        return int(self.data_size)


    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation()

        X = X[..., np.newaxis]
        y = y[..., np.newaxis]
        
        return X, y


    def __data_generation(self):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)

        # astro parameters for 21cmFAST
        eff_fact = random.gauss(52.5, 20.)
        Rmfp = random.gauss(12.5, 5.)
        Tvir = random.gauss(4.65, 0.5)
        astro_params = {'HII_EFF_FACTOR':self.eff_fact, 'R_BUBBLE_MAX':self.Rmfp, 'ION_Tvir_MIN':self.Tvir}

        # create seed for 21cmFAST
        str_seed = [var for var in datetime.now().strftime('%d%H%M%S')]
        np.random.shuffle(str_seed)
        seed = int(''.join(str_seed))

        # lightcone data
        lightcone = p21c.run_lightcone(redshift=self.z_min, max_redshift=self.z_max, astro_params=astro_params, cosmo_params=self.cosmo_params, user_params=self.user_params, lightcone_quantities=("brightness_temp", 'xH_box'), direc=p21c.config['direc'], random_seed=seed, cleanup=True)

        dataLC2 = np.array([lightcone.brightness_temp[:,:,i] for i in range(lightcone.brightness_temp.shape[-1])])
        redshiftLC = lightcone.lightcone_redshifts
        LC1 = t2c.subtract_mean_signal(dataLC, los_axis=2)

        noise_cone = t2c.noise_lightcone(ncells=LC1.shape[0], zs=redshiftLC, obs_time=self.tobs, boxsize=user_params['BOX_LEN'], n_jobs=1, save_uvmap=self.uvfile)
        LC3, redshiftLC3 = t2c.smooth_lightcone(lightcone=LC1+noise_cone, z_array=redshiftLC, box_size_mpc=params['BOX_LEN'])

        dataLC_xH = np.array([lightcone.xH_box[:,:,i] for i in range(lightcone.brightness_temp.shape[-1])])
        mask_LCxH = (LC_xH>0.5).astype(int)

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, y