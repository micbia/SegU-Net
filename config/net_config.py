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
        self.augment        = StringOrNone(trainconfig['AUGMENT'])
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
        if(', ' in trainconfig['DATASET_PATH']):
            self.path       = trainconfig['DATASET_PATH'].split(', ')
        else:
            self.path       = trainconfig['DATASET_PATH']

        self.train_data = trainconfig['TRAIN_DATA']
        self.pred_data = trainconfig['PRED_DATA']
        
        try:
            resumeconfig = config['RESUME']
            self.resume_path    = StringOrNone(resumeconfig['RESUME_PATH'])
            self.best_epoch     = eval(resumeconfig['BEST_EPOCH'])
            self.resume_epoch   = eval(resumeconfig['RESUME_EPOCH'])
        except:
            f = open(CONFIG_FILE, 'a')
            f.write('\n\n[RESUME]')
            f.write('\nRESUME_PATH = None')
            f.write('\nBEST_EPOCH = 0')
            f.write('\nRESUME_EPOCH = 0')
            f.close()

            self.resume_path    = None
            self.best_epoch     = 0
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