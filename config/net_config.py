import configparser, numpy as np

def StringOrNone(string):
    ''' convert initial condition that are 'None' in proper python none'''
    try:
        return eval(string)
    except:
        return string

class NetworkConfig:
    def __init__(self, CONFIG_FILE):
        self.config_file    = CONFIG_FILE

        config = configparser.ConfigParser()
        config.read(self.config_file)
        
        trainconfig = config['TRAINING']
        self.batch_size     = eval(trainconfig['BATCH_SIZE'])
        self.augment        = eval(trainconfig['AUGMENT'])
        self.img_shape      = tuple(np.array(eval(trainconfig['IMG_SHAPE']), dtype=int))
        self.chansize       = eval(trainconfig['CHAN_SIZE'])
        self.dropout        = eval(trainconfig['DROPOUT'])
        self.kernel_size    = eval(trainconfig['KERNEL_SIZE'])
        self.epochs         = eval(trainconfig['EPOCHS'])
        self.loss           = trainconfig['LOSS']
        if(', ' in trainconfig['METRICS']):
            self.metrics    = trainconfig['METRICS'].split(', ')
        else:
            self.metrics    = trainconfig['METRICS']
        self.learn_rate     = eval(trainconfig['LR'])
        self.recomplile     = eval(trainconfig['RECOMP'])
        self.gpus           = eval(trainconfig['GPUS'])
        if(', ' in trainconfig['PATH']):
            self.path       = trainconfig['PATH'].split(', ')
        else:
            self.path       = trainconfig['PATH']

        resumeconfig = config['RESUME']
        self.resume_path    = StringOrNone(resumeconfig['RESUME_PATH'])
        self.best_epoch     = eval(resumeconfig['BEST_EPOCH'])
        self.resume_epoch   = eval(resumeconfig['RESUME_EPOCH'])


class PredictionConfig:
    def __init__(self, CONFIG_FILE):
        self.config_file    = CONFIG_FILE

        config = configparser.ConfigParser()
        config.read(self.config_file)
 
        predconfig = config['PREDICTION']
        self.img_shape      = tuple(np.array(eval(predconfig['IMG_SHAPE']), dtype=int))
        self.model_epoch    = eval(predconfig['MODEL_EPOCH'])
        self.tta_wrap       = eval(predconfig['TTA_WRAP'])
        self.augmentation   = eval(predconfig['AUGMENT'])
        self.val            = eval(predconfig['EVAL'])
        self.path_pred      = predconfig['PATH_PREDIC']
        self.path_out       = predconfig['PATH_OUT']
        self.indexes        = np.array(eval(predconfig['INDEXES']))