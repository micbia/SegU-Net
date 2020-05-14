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
        self.kernel_size    = eval(trainconfig['KENREL_SIZE'])
        self.epochs         = eval(trainconfig['EPOCHS'])
        self.loss           = trainconfig['LOSS']
        self.learn_rate     = eval(trainconfig['LR'])
        self.recomplile     = eval(trainconfig['RECOMP'])
        self.gpus           = eval(trainconfig['GPUS'])
        try:
            self.path       = trainconfig['PATH'].split(', ')
        except:
            self.path       = trainconfig['PATH']

        resumeconfig = config['RESUME']
        self.resume_path    = StringOrNone(resumeconfig['RESUME_PATH'])
        self.best_epoch      = eval(resumeconfig['BEST_EPOCH'])
        self.resume_epoch   = eval(resumeconfig['RESUME_EPOCH'])
        
