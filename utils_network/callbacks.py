import numpy as np, os, warnings
from glob import glob

from keras import backend as K
from keras.callbacks import Callback

class HistoryCheckpoint(callbacks.Callback):
    def __init__(self, filepath='./', verbose=0, save_freq=1, in_epoch=0):
        self.verbose = verbose
        self.filepath = filepath
        self.save_freq = save_freq
        self.stor_arr = []
        self.prev_epoch = 0
        self.in_epoch = in_epoch

    def on_train_begin(self, logs=None):
        if(self.in_epoch != 0):
            print('Resuming from Epoch %d...' %self.in_epoch)
            self.prev_epoch = self.in_epoch

    def on_epoch_end(self, epoch, logs=None):
        if(epoch == self.in_epoch): self.stor_arr = [[] for i in range(len(logs))]     # initializate array
        
        fname = self.filepath+'%s_ep-%d.txt'

        if(epoch % self.save_freq == 0 and epoch != self.in_epoch): 
            for i, val in enumerate(logs):
                self.stor_arr[i] = np.append(self.stor_arr[i], logs[val])

                if(os.path.isfile(fname %(val, self.prev_epoch))):
                    chekp_arr = np.loadtxt(fname %(val, self.prev_epoch)) # load previous save
                    chekp_arr = np.append(chekp_arr, self.stor_arr[i])    # update 
                    np.savetxt(fname %(val, epoch+1), chekp_arr)            # save
                    os.remove(fname %(val, self.prev_epoch))              # delete old save
                else:
                    np.savetxt(fname %(val, epoch+1), self.stor_arr[i])
            
            self.prev_epoch = epoch+1
            self.stor_arr = [[] for i in range(len(logs))]          # empty storing array

            if(self.verbose): print('Updated Logs checkpoints for epoch %d.' %(epoch+1))
        else:
            for j, val in enumerate(logs):
                self.stor_arr[j] = np.append(self.stor_arr[j], logs[val])


class ReduceLR(Callback):
    " Copied original code, added 'wait' to init parameters for resuming training"
    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
                 verbose=0, mode='auto', min_delta=1e-4, cooldown=0, 
                 min_lr=0, wait=0, best=None, **kwargs):
        super(ReduceLR, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            warnings.warn('`epsilon` argument is deprecated and '
                          'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = wait
        self.best = best
        self.mode = mode    
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if(self.mode not in ['auto', 'min', 'max']):
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode), RuntimeWarning)
            self.mode = 'auto'
        if(self.mode == 'min' or (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            if(self.best == None):
                self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            if(self.best == None):
                self.best = -np.Inf
        self.cooldown_counter = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning)
        else:
            if(self.in_cooldown()):
                self.cooldown_counter -= 1
                self.wait = 0

            if(self.monitor_op(current, self.best)):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


class SaveModelCheckpoint(Callback):
    " Copied original code, added variable 'best' to init parameters for resuming training"
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, best=None):
        super(SaveModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.best = best

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('SaveModelCheckpoint mode %s is unknown, fallback to auto mode.' % (mode), RuntimeWarning)
            mode = 'auto'
        
        if mode == 'min':
            self.monitor_op = np.less
            if(self.best == None):
                self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            if(self.best == None):
                self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                if(self.best == None):
                    self.best = -np.Inf
            else:
                self.monitor_op = np.less
                if(self.best == None):
                    self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f, saving model to %s' %(epoch + 1, self.monitor, self.best, current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %(epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
