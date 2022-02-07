import numpy as np
from tqdm import tqdm

def UniqueRows(arr):
    """ Remove duplicate row array in 2D data 
            * arr (narray): array with duplicate row
        
        Example:
        >> d = np.array([[0,1,2],[0,1,2],[0,0,0],[0,0,2],[0,1,2]])
        >> UniqueRows(d) 
        
        array([[0, 0, 0],
                [0, 0, 2],
                [0, 1, 2]])
    """
    arr = np.array(arr)

    if(arr.ndim == 2):
        arr = np.ascontiguousarray(arr)
        unique_arr = np.unique(arr.view([('', arr.dtype)]*arr.shape[1]))
        new_arr = unique_arr.view(arr.dtype).reshape((unique_arr.shape[0], arr.shape[1]))
    elif(arr.ndim == 1):
        new_arr = np.array(list(dict.fromkeys(arr)))

    return new_arr


def IndependentOperation():
    ''' How many operation of flip and rotation can a cube have in SegUnet?, 
        each indipendent operation is considered as an additional rappresentation
        point of the same coeval data, so that it can be considered for errorbar 
        in SegUnet '''

    data = np.array(range(4**3)).reshape((4,4,4)) 
    permut_opt = [] 
    
    operations = [lambda a: a, np.fliplr, np.flipud, lambda a: np.flipud(np.fliplr(a)), lambda a: np.fliplr(np.flipud(a))] 
    axis = [0,1,2] 
    angl_rot = [0,1,2,3] 
    
    permut_idx = np.zeros((len(operations)*len(axis)*len(angl_rot), data.size)) 
    permut_tot = {'opt%d' %k:[] for k in range(len(operations)*len(axis)*len(angl_rot))} 
    
    i = 0 
    for iopt, opt in enumerate(operations): 
        cube = opt(data)
        for rotax in axis: 
            for rot in angl_rot: 
                ax_tup = [0,1,2] 
                ax_tup.remove(rotax)         
                permut_idx[i] = np.rot90(cube, k=rot, axes=ax_tup).flatten() 
                #permut_opt.append('opt%d rotax%d rot%d' %(iopt,rotax,rot)) 
                permut_tot['opt%d' %i] = [opt, rotax, rot] 
                i += 1 
    
    idx_iter = [] 
    for j in range(0,permut_idx.shape[0]-1): 
        for k in range(j+1, permut_idx.shape[0]): 
            if (all(permut_idx[j] == permut_idx[k])): 
                idx_iter.append(k) 
    
    idx_iter = np.array(list(UniqueRows(idx_iter))) 
    idx_opt = np.sort(np.array(range(i))[~idx_iter]) 
    
    permut_opt = {'opt%d' %k: permut_tot['opt%d' %val] for k, val in enumerate(idx_opt)} 
    
    return permut_opt


def IndependentOperation_LC():
    operations = [lambda a: a, np.fliplr, np.flipud, lambda a: np.flipud(np.fliplr(a))] 
    axis = [0] 
    angl_rot = [0,1,2,3] 
    
    permut_op = {} 
    i = 0 
    for iopt, opt in enumerate(operations): 
        for rotax in axis: 
            for rot in angl_rot: 
                permut_op['opt%d' %i] = [opt, rotax, rot] 
                i += 1
    return permut_op


def SegUnet21cmPredict(unet, x, TTA=False):
    if (TTA and x.shape[0] == x.shape[-1]):
        transf_opts = IndependentOperation()
        X_tta = np.zeros((np.append(3*len(transf_opts), x.shape)))
    elif(TTA and x.shape[0] != x.shape[-1]):
        transf_opts = IndependentOperation_LC()
        X_tta = np.zeros((np.append(len(transf_opts), x.shape)))
    else:
        transf_opts = {'opt0': [lambda a: a, 0, 0]}
        X_tta = np.zeros((np.append(3, x.shape)))
    
    for iopt in tqdm(range(len(transf_opts))):
        opt, rotax, rot = transf_opts['opt%d' %iopt]
        ax_tup = [0,1,2] 
        ax_tup.remove(rotax)

        if (x.shape[0] == x.shape[-1]):
            cube = np.rot90(opt(x), k=rot, axes=ax_tup) 
            X = cube[np.newaxis, ..., np.newaxis]
        else:
            #lc = np.rot90(opt(x), k=rot, axes=ax_tup)
            X = x[np.newaxis, ..., np.newaxis]

        for j in range(x.shape[0]):
            if (x.shape[0] == x.shape[-1]):
                X_tta[iopt,j,:,:] = unet.predict(X[:,j,:,:,:], verbose=0).squeeze()
                X_tta[iopt+len(transf_opts),:,j,:] = unet.predict(X[:,:,j,:,:], verbose=0).squeeze()
                X_tta[iopt+len(transf_opts)*2,:,:,j] = unet.predict(X[:,:,:,j,:], verbose=0).squeeze()
            else:
                #x_pred = unet.predict(X[:,:,:,j,:], verbose=0).squeeze()
                #ax_tup = [0,1,2] 
                #ax_tup.remove(rotax)
                #X_tta[iopt,:,:,j] = opt(np.rot90(x_pred, k=-rot))
                X_tta[iopt,:,:,j] = unet.predict(X[:,:,:,j,:], verbose=0).squeeze()


    if (TTA and x.shape[0] == x.shape[-1]):
        transf_opts = IndependentOperation()
        for itta in range(X_tta.shape[0]):
            opt, rotax, rot = transf_opts['opt%d' %(itta%len(transf_opts))]
            ax_tup = [0,1,2] 
            ax_tup.remove(rotax)
            X_tta[itta] = opt(np.rot90(X_tta[itta], k=-rot, axes=ax_tup))

    return X_tta