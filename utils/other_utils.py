import numpy as np, matplotlib.pyplot as plt, random, operator
from glob import glob
from tqdm import tqdm


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


def RescaleData(arr, a=-1, b=1):
    scaled_arr = (arr.astype(np.float32) - np.min(arr))/(np.max(arr) - np.min(arr)) * (b-a) + a
    return scaled_arr


# Get train images and masks
def get_data(path, img_shape, shuffle=False):
    size = len(glob(path+'images*'))
    
    X = np.zeros(np.append(size, img_shape), dtype=np.float32)
    Y = np.zeros(np.append(size, img_shape), dtype=np.float32)
    
    for i in tqdm(range(size)):
        X[i] = read_cbin(filename=path+'images_21cm_i%d.bin' %i, dimensions=len(img_shape))
        Y[i] = read_cbin(filename=path+'masks_21cm_i%d.bin' %i, dimensions=len(img_shape))
    
    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]

    X = RescaleData(arr=X, a=0, b=1)
    Y = RescaleData(arr=Y, a=0, b=1)

    if(shuffle):
        idxs = np.array(range(size))
        np.random.shuffle(idxs)

        X = X[idxs]
        y = y[idxs]

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


