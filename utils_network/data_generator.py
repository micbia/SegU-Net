import math, random, numpy as np

from keras.utils import Sequence
from tqdm import tqdm


class DataGenerator(Sequence):
    """
    Data generator of 3D data (only flip and rotate).

    Note:
        At the moment only one type of augmentation can be applied for one generator.
    """
    def __init__(self, data=None, label=None, batch_size=None, flip_axis=None, 
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
        self.data = data
        self.label = label
        self.batch_size = batch_size
        
        self.flip_axis = flip_axis
        self.rotate_axis = rotate_axis
        self.rotate_angle = rotate_angle
        self.shuffle = shuffle

        self.data_shape = data.shape[1:]
        self.data_size = data.shape[0]
        self.on_epoch_end()

        self.idx_list = np.array(range(self.data_size))

        if self.label is not None:
            self.exist_label = True
        else:
            self.exist_label = False

        self.callback = None
        if(self.rotate_axis is not None) and (self.rotate_angle is not None):
            self.callback = self._rotate_data
        elif(self.flip_axis is not None):
            self.callback = self._flip_data
        else:
            raise ValueError('No action chosen.')
        

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

        X = np.array([self.callback(self.data[idx].squeeze()) for idx in indexes])
        X = X[..., np.newaxis]
        
        y = np.array([self.callback(self.label[idx].squeeze()) for idx in indexes])
        y = y[..., np.newaxis]
        return X, y



    def _flip_data(self, data):
        """flip array along specified axis(x, y or z)"""
        if(self.flip_axis in (0, 1, 2, 3)):
            pass
        elif(self.flip_axis == 'random'):
            self.flip_axis = random.randint(0, 3)
        else:
            raise ValueError('Flip axis should be 0, 1, 2, 3 or random')

        if(self.flip_axis == 0):
            flip_data = data
        else:
            flip_data = np.flip(data, self.flip_axis)

        return flip_data


    def _rotate_data(self, data):
        """rotate array by specified range along specified axis(x, y or z)"""
        if self.rotate_axis in (1, 2, 3):
            pass
        elif self.rotate_axis == 'random':
            self.rotate_axis = random.randint(1, 3)
        else:
            raise ValueError('Rotate axis should be 1, 2, 3 or random')
        
        if isinstance(type(self.rotate_angle), (int, float)):
            pass
        elif(self.rotate_angle == 'random'):
            rotation = int(random.choice([90, 180, 270, 360]) / 90)
        else:
            raise ValueError('Rotate angle should be 90, 180, 270, 360 or random')
        
        if self.rotate_axis == 1:
            ax_tup = (1, 2)
        elif self.rotate_axis == 2:
            ax_tup = (2, 0)
        elif self.rotate_axis == 3:
            ax_tup = (0, 1)
        else:
            raise ValueError('rotate axis should be 1, 2 or 3')
        
        return np.rot90(data, k=rotation, axes=ax_tup)




class TTA_ModelWrapper():
    """A simple TTA wrapper for keras computer vision models.
    Args:
        model (keras model): A fitted keras model with a predict method.
    """

    def __init__(self, model, generator):
        self.model = model

    def predict(self, X):
        pred = []
        for i in tqdm(range(X.shape[0])):
            p0 = self.model.predict(X[i][np.newaxis, ...]).squeeze()
            p1 = self.model.predict(np.fliplr(X[i][np.newaxis, ...])).squeeze()
            p2 = self.model.predict(np.flipud(X[i][np.newaxis, ...])).squeeze()
            p3 = self.model.predict(np.fliplr(np.flipud(X[i][np.newaxis, ...]))).squeeze()
            p =  (p0 + np.fliplr(p1) + np.flipud(p2) + np.fliplr(np.flipud(p3))) * 0.25
            pred.append(p)
        return np.array(pred)
    
    def _expand(self, x):
        return np.expand_dims(np.expand_dims(x, axis=0), axis=3)

