import zipfile, tarfile, math, random, numpy as np, os, sys
import pandas as pd
import tools21cm as t2c

from datetime import datetime
from glob import glob
from tensorflow.keras.utils import Sequence

class LightConeGenerator_Reg(Sequence):
    """
    Data generator of lightcone data.

    Michele, 21 Sep 2021: up to date the class deal with LC data that are already smoothed and calculated the noise
    """
    def __init__(self, path='./', data_temp=None, batch_size=None, zipf=False,
                 data_shape=None, tobs=1000, shuffle=False):
        """
        Arguments:
         tobs: int
                observational time, for noise calcuation.
        """
        self.path = path
        self.data_temp = data_temp
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.tobs = tobs
        self.data_size = len(self.data_temp)
        self.on_epoch_end()
        
        self.astro_par = np.loadtxt('%sparameters/astro_params.txt' %self.path, usecols=(1,2,3))
        self.redshift = np.loadtxt('%s/lc_redshifts.txt' %self.path)

        with open('%sparameters/user_params.txt' %self.path,'r') as f:
            self.user_par = eval(f.read())

    def __len__(self):
        # number of batches
        return self.data_size//self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        X = np.zeros((np.append(self.batch_size, self.data_shape)))
        y1 = np.zeros((np.append(self.batch_size, self.data_shape)))
        y2 = np.zeros((np.append(self.batch_size, 4)))
        
        for i, idx in enumerate(indexes):
            # read LC
            dT = self._read_cbin(filename='%sdT3_21cm_i%d.bin' %(self.path+'data/', idx), dimensions=3)
            #xH = self._read_cbin(filename='%sxH_21cm_i%d.bin' %(self.path+'data/', idx), dimensions=3)
            xH = self._read_cbin(filename='%sdT2_21cm_i%d.bin' %(self.path+'data/', idx), dimensions=3)
            
            # apply manipolation on the LC data
            X[i], y1[i], nu = self._lc_data(x=dT, y=xH)
            y2[i] = np.append(self.astro_par[i], nu)

        # add channel dimension
        X = X[..., np.newaxis]
        y1 = y1[..., np.newaxis]

        return X, y1, y2

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = self.data_temp
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _lc_data(self, x, y):
        # for U-Net on slices
        rseed2 = random.randint(0, x.shape[-1]-1)
        dT_sampled = x[:, :, rseed2].astype(np.float32)
        xH_sampled = y[:, :, rseed2].astype(np.float32)

        #dT_sampled = self._RescaleData(arr=dT_sampled, a=1e-3, b=100.)
        #xH_sampled = self._RescaleData(arr=xH_sampled, a=1e-7, b=1.-1e-7)
        return dT_sampled, xH_sampled, t2c.z_to_nu(self.redshift[rseed2])

    def _RescaleData(self, arr, a=-1, b=1):
        scaled_arr = (arr.astype(np.float32) - np.min(arr))/(np.max(arr) - np.min(arr)) * (b-a) + a
        return scaled_arr
        
    def _read_cbin(self, filename, bits=32, order='C', dimensions=3):
        assert(bits ==32 or bits==64)
        f = open(filename)
        temp_mesh = np.fromfile(f, count=dimensions, dtype='int32')
        datatype = np.float32 if bits == 32 else np.float64
        data = np.fromfile(f, dtype=datatype, count=np.prod(temp_mesh))
        data = data.reshape(temp_mesh, order=order)
        return data


class LightConeGenerator(Sequence):
    """
    Data generator of lightcone data.

    Michele, 21 Sep 2021: up to date the class deal with LC data that are already smoothed and calculated the noise
    """
    def __init__(self, path='./', data_temp=None, batch_size=None, zipf=False,
                 data_shape=None, tobs=1000, shuffle=False):
        """
        Arguments:
         tobs: int
                observational time, for noise calcuation.
        """
        self.path = path
        self.data_temp = data_temp
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.zipf = zipf
        if(self.zipf):
            # the index of the array indicate the number of the 
            self.content = np.loadtxt(self.path+'data/content.txt', dtype=int)
            self.path_in_zip = self.path[self.path[:-1].rfind('/')+1:]+'data/'
        self.tobs = tobs
        self.data_size = len(self.data_temp)
        self.on_epoch_end()
        
        self.astro_par = np.loadtxt('%sparameters/astro_params.txt' %self.path, unpack=True)
        with open('%sparameters/user_params.txt' %self.path,'r') as f:
            self.user_par = eval(f.read())

    def __len__(self):
        # number of batches
        return self.data_size//self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        X = np.zeros((np.append(self.batch_size, self.data_shape)))
        y = np.zeros((np.append(self.batch_size, self.data_shape)))
        for i, idx in enumerate(indexes):
            if(self.zipf):
                i_tar = self.content[idx]
                name_tar = '%sdata/%s_part%d.tar.gz' %(self.path, self.path[self.path[:-1].rfind('/')+1:-1], i_tar)
                mytar = tarfile.open(name_tar, 'r')

                # load file containing TarInfo
                tar_content = np.load('%sdata/tar_content_part%d.npy' %(self.path, i_tar), allow_pickle=True) 
                tar_names = np.load('%sdata/tar_names_part%d.npy' %(self.path, i_tar))
                
                # create DataFrame to easly pick the correct TarInfo
                tar_df = pd.DataFrame(data=tar_content, index=tar_names)

                # extract file
                member = tar_df.loc['%sdT3_21cm_i%d.bin' %(self.path_in_zip, idx),0]
                temp_data = mytar.extractfile(member).read()
                temp_mesh = np.frombuffer(temp_data, count=3, dtype='int32')
                dT = np.frombuffer(temp_data, count=np.prod(temp_mesh), dtype='float32').reshape(temp_mesh, order='C')
                member = tar_df.loc['%sxH_21cm_i%d.bin' %(self.path_in_zip, idx),0]
                temp_data = mytar.extractfile(member).read()
                temp_mesh = np.frombuffer(temp_data, count=3, dtype='int32')
                xH = np.frombuffer(temp_data, count=np.prod(temp_mesh), dtype='float32').reshape(temp_mesh, order='C')
                mytar.close()
                
                # apply manipolation on the LC data
                X[i], y[i] = self._lc_data(x=dT, y=xH)
            else:
                # read LC
                dT = self._read_cbin(filename='%sdT3_21cm_i%d.bin' %(self.path+'data/', idx), dimensions=3)
                #xH = self._read_cbin(filename='%sxH_21cm_i%d.bin' %(self.path+'data/', idx), dimensions=3)
                xH = self._read_cbin(filename='%sdT2_21cm_i%d.bin' %(self.path+'data/', idx), dimensions=3)
                
                # apply manipolation on the LC data
                X[i], y[i] = self._lc_data(x=dT, y=xH)

        # add channel dimension
        X = X[..., np.newaxis]
        y = y[..., np.newaxis]

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = self.data_temp
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _lc_data(self, x, y):
        if(np.min(self.data_shape) == np.max(self.data_shape)):
            # for U-Net on slices
            rseed2 = random.randint(0, x.shape[-1]-1)
            dT_sampled = x[:, :, rseed2].astype(np.float32)
            xH_sampled = y[:, :, rseed2].astype(np.float32)
        else:
            # for LSTM U-Net on frequency range
            freq_size = np.min(self.data_shape)
            rseed2 = random.randint(freq_size, x.shape[-1]-1-freq_size) 
            dT_sampled = np.array([x[:, :, i].astype(np.float32) for i in range(rseed2-freq_size//2, rseed2+freq_size//2)])
            #xH_sampled = y[:, :, rseed2+freq_size//2-1].astype(np.float32)      # get only last slice
            xH_sampled = np.array([y[:, :, i].astype(np.float32) for i in range(rseed2-freq_size//2, rseed2+freq_size//2)])

        #dT_sampled = self._RescaleData(arr=dT_sampled, a=1e-3, b=100.)
        #xH_sampled = self._RescaleData(arr=xH_sampled, a=1e-7, b=1.-1e-7)
        return dT_sampled, xH_sampled


    def _RescaleData(self, arr, a=-1, b=1):
        scaled_arr = (arr.astype(np.float32) - np.min(arr))/(np.max(arr) - np.min(arr)) * (b-a) + a
        return scaled_arr

        
    def _read_cbin(self, filename, bits=32, order='C', dimensions=3):
        ''' Read a binary file with three inital integers (a cbin file).
        
        Parameters:
                * filename (string): the filename to read from
                * bits = 32 (integer): the number of bits in the file
                * order = 'C' (string): the ordering of the data. Can be 'C'
                        for C style ordering, or 'F' for fortran style.
                * dimensions (int): the number of dimensions of the data (default:3)
                        
        Returns:
                The data as a three dimensional numpy array.
        '''

        assert(bits ==32 or bits==64)

        f = open(filename)

        temp_mesh = np.fromfile(f, count=dimensions, dtype='int32')

        datatype = np.float32 if bits == 32 else np.float64
        data = np.fromfile(f, dtype=datatype, count=np.prod(temp_mesh))
        data = data.reshape(temp_mesh, order=order)
        return data

    #TODO: smoothing for LC data
    def _lc_noise_smt_dT(self, lc1, idx):
        assert idx == self.astro_par[0,idx]
        z = self.astro_par[1, idx]

        noise_lc = t2c.noise_lightcone(ncells=lc.brightness_temp.shape[0], zs=lc.lightcone_redshifts, obs_time=self.tobs, save_uvmap=self.uvfile, boxsize=self.user_par['BOX_LEN'])

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

    def _lc_smt_xH(self, xH_box, idx):
        assert idx == self.astro_par[0,idx]
        z = self.astro_par[1, idx]

        smt_xn = t2c.smooth_coeval(xH_box, z, box_size_mpc=self.user_par['HII_DIM'], max_baseline=2.0, ratio=1.0, nu_axis=2)
        mask_xn = smt_xn>0.5

        return mask_xn.astype(int)


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



