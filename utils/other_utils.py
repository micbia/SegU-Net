import numpy as np, random, operator, os
import zipfile
from glob import glob
from tqdm import tqdm
from datetime import datetime 
import time 


class Timer: 
    def __init__(self): 
        self._start_time = None
        self._prevlap_time = None
 
    def start(self): 
        """Start a new timer""" 
        if(self._start_time != None): 
            raise TimerError(f"Timer is running. Use .stop() to stop it") 
        self._start_time = time.perf_counter()

    def lap(self, mess=None): 
        """Stop the timer, and report the elapsed time""" 
        if(self._start_time == None): 
            raise TimerError(f"Timer is not running. Use .start() to start it") 
        lap_time = time.perf_counter()
        if(self._prevlap_time != None):
            elapsed_time = lap_time - self._prevlap_time
        else:
            elapsed_time = lap_time - self._start_time
        self._prevlap_time = lap_time
        mess = ' - '+mess if mess!=None else ''
        print("Lap time: %.4f seconds%s" %(elapsed_time, mess)) 

    def stop(self, mess=''): 
        mess = ' - '+mess if mess!='' else mess
        """Stop the timer, and report the elapsed time""" 
        if(self._start_time == None): 
            raise TimerError(f"Timer is not running. Use .start() to start it")
        time_stop = time.perf_counter()
        if(self._prevlap_time != None):
            elapsed_time = time_stop - self._prevlap_time
            print("Lap time: %.4f seconds - final lap" %(elapsed_time))
        elapsed_time = time_stop - self._start_time
        print("Elapsed time: %.4f seconds%s" %(elapsed_time, mess)) 
        self._start_time = None

class TimerError(Exception): 
    """A custom exception used to report errors in use of Timer class""" 
 

def GenerateSeed():
    # create seed for 21cmFast  
    seed = [var for var in datetime.now().strftime('%d%H%M%S')]
    np.random.shuffle(seed)
    return int(''.join(seed))


def get_dir_size(dir):
    """Returns the "dir" size in bytes."""
    comd = os.popen('du -s %s' %dir).read()
    total = float(comd[:comd.find('\t')])  / 1e6 # in GB units
    return total
    

def OrderNdimArray(arr, idx_to_sort):
    ''' Order N-dim array giving the index of sorting
    Parameters:
        * arr (array): N-dim or simple array to sort
        * idx_to_sort (int): sorteing array index
    Returns:
        sorted array by desired array index
        '''
    new_arr = np.array(sorted(arr.T, key=operator.itemgetter(idx_to_sort)))
    return new_arr.T
    

def SortArray(arr, idx_to_sort):
    ''' Order N-dim array giving the index of sorting
    Parameters:
        * arr (array): N-dim or simple array to sort
        * idx_to_sort (int): sorteing array index
    Returns:
        sorted array by desired array index
        '''
    new_arr = np.array(sorted(arr.T, key=operator.itemgetter(idx_to_sort)))
    return new_arr


def RotateCube(data, rot, rotax):
    data = data.squeeze()
    if(len(data.shape) == 3):
        ax_tup = [0,1,2]
        ax_tup.remove(rotax)
        rotated_data = np.rot90(data, k=rot, axes=ax_tup)
    elif(len(data.shape) == 2):
        rotated_data = np.rot90(data, k=rot)
    return rotated_data


def RescaleData(arr, a=-1, b=1):
    scaled_arr = (arr.astype(np.float32) - np.min(arr))/(np.max(arr) - np.min(arr)) * (b-a) + a
    return scaled_arr


def get_extend(a):
    ext = a[a.rfind('.'):]
    return ext

def get_data_lc(path, i, shuffle=False):
    lc_dT = read_cbin(filename='%sdata/dT3_21cm_i%d.bin' %(path, i), dimensions=3)
    lc_mask = read_cbin(filename='%sdata/xH_21cm_i%d.bin' %(path, i), dimensions=3)

    size = lc_dT.shape[-1]

    X = np.zeros((size, lc_dT.shape[0], lc_dT.shape[1]), dtype=np.float32)
    Y = np.zeros((size, lc_dT.shape[0], lc_dT.shape[1]), dtype=np.float32)
    
    for i in tqdm(range(size)):
        X[i] = lc_dT[...,i]
        Y[i] = lc_mask[...,i]
    
    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]

    X = RescaleData(arr=X, a=0, b=1)
    Y = RescaleData(arr=Y, a=0, b=1)

    if(shuffle):
        idxs = np.array(range(size))
        np.random.shuffle(idxs)

        X = X[idxs]
        Y = Y[idxs]

    return X, Y

# Get train images and masks
def get_data(path, img_shape, shuffle=False, norm=False):
    size = len(glob(path+'image*'))
    ext = get_extend(glob(path+'image*')[0])

    X = np.zeros(np.append(size, img_shape), dtype=np.float32)
    Y = np.zeros(np.append(size, img_shape), dtype=np.float32)
    
    for i in tqdm(range(size)):
        if(ext == '.bin'):
            X[i] = read_cbin(filename=path+'image_21cm_i%d%s' %(i, ext), dimensions=len(img_shape))
            Y[i] = read_cbin(filename=path+'mask_21cm_i%d%s' %(i, ext), dimensions=len(img_shape))
        elif(ext == '.npy'):
            X[i] = np.load(path+'image_21cm_i%d%s' %(i, ext))
            Y[i] = np.load(path+'mask_21cm_i%d%s' %(i, ext))
    
    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]

    if(norm):
        X = RescaleData(arr=X, a=-1, b=1)
        Y = RescaleData(arr=Y, a=-1, b=1)

    if(shuffle):
        idxs = np.array(range(size))
        np.random.shuffle(idxs)

        X = X[idxs]
        Y = Y[idxs]

    return X, Y



def get_batch(path, img_shape, shuffle=False, norm=False, size=32, dataset_size=1000, ext='npy'):
    X = np.zeros(np.append(size, img_shape), dtype=np.float32)
    Y = np.zeros(np.append(size, img_shape), dtype=np.float32)
    
    idx_list = random.choices(range(dataset_size), k=size)
    
    for i in tqdm(range(size)):
        if(path[-3:] == 'zip'):
            with zipfile.ZipFile(path) as myzip: 
                with myzip.open('%s/data/image_21cm_i%d.%s' %(path[path[:-5].rfind('/')+1:-4], i, ext)) as myfile1: 
                    X[i] = np.load(myfile1) 
                with myzip.open('%s/data/mask_21cm_i%d.%s' %(path[path[:-5].rfind('/')+1:-4], i, ext)) as myfile2: 
                    Y[i] = np.load(myfile2) 
        else:
            X[i] = read_cbin(filename='%simage_21cm_i%d.%s' %(path, i, ext), dimensions=len(img_shape))
            Y[i] = read_cbin(filename='%smask_21cm_i%d.%s' %(path, i, ext), dimensions=len(img_shape))

    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]

    if(norm):
        X = RescaleData(arr=X, a=-1, b=1)
        Y = RescaleData(arr=Y, a=-1, b=1)

    if(shuffle):
        idxs = np.array(range(size))
        np.random.shuffle(idxs)

        X = X[idxs]
        Y = Y[idxs]

    return X, Y


def get_subset3D(indexes, path, im_height, im_width, im_depth):
    batch_size = indexes.shape[0]

    arr_img = ['' for i in range(batch_size)]
    for j, idx in enumerate(indexes):
        arr_img[j] = '%simages_21cm_i%dj%dk%dl%d.bin' %(path, idx[0], idx[1], idx[2], idx[3])
    
    #arr_img = random.choices(arr_img, k=int(len(arr_img)/3))

    X = np.zeros((len(arr_img), im_height, im_width, im_depth), dtype=np.float32)
    Y = np.zeros((len(arr_img), im_height, im_width, im_depth), dtype=np.float32)
    
    for i, img in enumerate(arr_img):
        X[i] = read_cbin(img)
        mask = path+'masks'+img[img.rfind('_21cm'):]
        Y[i] = read_cbin(mask)
	    	
    X = X[:, :, :, :, np.newaxis]
    Y = Y[:, :, :, :, np.newaxis]

    X = RescaleData(arr=X, a=0, b=1)
    Y = RescaleData(arr=Y, a=0, b=1)

    return X, Y


def read_cbin(filename, bits=32, order='C', dimensions=3):
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


def save_cbin(filename, data, bits=32, order='C'):
    ''' Save a binary file with three inital integers (a cbin file).
    
    Parameters:
            * filename (string): the filename to save to
            * data (numpy array): the data to save
            * bits = 32 (integer): the number of bits in the file
            * order = 'C' (string): the ordering of the data. Can be 'C'
                    for C style ordering, or 'F' for fortran style.
                    
    Returns:
            Nothing
    '''
    assert(bits ==32 or bits==64)
    f = open(filename, 'wb')
    mesh = np.array(data.shape).astype('int32')
    mesh.tofile(f)
    datatype = (np.float32 if bits==32 else np.float64)
    data.flatten(order=order).astype(datatype).tofile(f)
    f.close()


