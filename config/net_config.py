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
        self.BATCH_SIZE     = eval(trainconfig['BATCH_SIZE'])
        self.AUGMENT        = StringOrNone(trainconfig['AUGMENT'])
        self.IM_SHAPE      = tuple(np.array(eval(trainconfig['IMG_SHAPE']), dtype=int))
        self.COARSE_DIM       = eval(trainconfig['CHAN_SIZE'])
        self.DROPOUT        = eval(trainconfig['DROPOUT'])
        self.KERNEL_SIZE    = eval(trainconfig['KERNEL_SIZE'])
        self.EPOCHS         = eval(trainconfig['EPOCHS'])
        if(', ' in trainconfig['LOSS']):
            self.LOSS    = trainconfig['LOSS'].split(', ')
        else:
            self.LOSS    = trainconfig['LOSS']
        if(', ' in trainconfig['METRICS']):
            self.METRICS    = trainconfig['METRICS'].split(', ')
        else:
            self.METRICS    = trainconfig['METRICS']
        self.LR     = eval(trainconfig['LR'])
        self.RECOMPILE     = eval(trainconfig['RECOMP'])
        self.GPU           = eval(trainconfig['GPUS'])
        if(', ' in trainconfig['DATASET_PATH']):
            self.DATASET_PATH       = trainconfig['DATASET_PATH'].split(', ')
        else:
            self.DATASET_PATH       = trainconfig['DATASET_PATH']
        self.IO_PATH        = trainconfig['IO_PATH']
        
        try:
            resumeconfig = config['RESUME']
            self.RESUME_PATH    = StringOrNone(resumeconfig['RESUME_PATH'])
            self.BEST_EPOCH     = eval(resumeconfig['BEST_EPOCH'])
            self.RESUME_EPOCH   = eval(resumeconfig['RESUME_EPOCH'])
        except:
            f = open(CONFIG_FILE, 'a')
            f.write('\n\n[RESUME]')
            f.write('\nRESUME_PATH = None')
            f.write('\nBEST_EPOCH = 0')
            f.write('\nRESUME_EPOCH = 0')
            f.close()

            self.resume_path    = None
            self.BEST_EPOCH     = 0
            self.resume_epoch   = 0

class PredictionConfig:
    def __init__(self, CONFIG_FILE):
        self.config_file    = CONFIG_FILE

        config = configparser.ConfigParser()
        config.read(self.config_file)
 
        predconfig = config['PREDICTION']
        self.model_epoch    = eval(predconfig['MODEL_EPOCH'])
        self.tta_wrap       = eval(predconfig['TTA_WRAP'])
        self.augmentation   = eval(predconfig['AUGMENT'])
        self.val            = eval(predconfig['EVAL'])
        self.indexes        = np.array(eval(predconfig['INDEXES']))
        self.path_out       = StringOrNone(predconfig['MODEL_PATH'])

        trainconfig = config['TRAINING']
        self.path_pred      = trainconfig['DATASET_PATH']
        self.pred_data      = trainconfig['PRED_DATA']
        self.loss           = trainconfig['LOSS']
        self.img_shape      = tuple(np.array(eval(trainconfig['IMG_SHAPE']), dtype=int))
        if(', ' in trainconfig['METRICS']):
            self.metrics    = np.append(trainconfig['METRICS'].split(', '),trainconfig['LOSS'])
        else:
            self.metrics    = np.append(trainconfig['METRICS'], trainconfig['LOSS'])

        resumeconfig = config['RESUME']