import os
import pickle5 as pickle

DATASETS = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k','pascalvoc_70','pascalvoc_140','pascalvoc_350','pascalvoc_700','pascalvoc_1400','pascalvoc_700_no_bbx',
            'pascalvoc_700_train', 'pascalvoc_700_medium', 'pascalvoc_700_medium_train','pascalvoc_700_no_bbx_train',
           'caltech101_70','caltech101_350','caltech101_700','caltech101_1400','caltech101_700_train',
            'roxford5k_drift_cnn','rparis6k_drift_cnn','pascalvoc_700_drift_cnn',
            'pascalvoc_700_medium_drift_cnn',
            'pascalvoc_700_no_bbx_drift_cnn',
            'caltech101_700_drift_cnn'
           ]

def configdataset(dataset, dir_main):

    dataset = dataset.lower()

    if dataset not in DATASETS:    
        raise ValueError('Unknown dataset: {}!'.format(dataset))
        
    if(dataset.startswith('pascalvoc')):
        dataset_name = 'pascalvoc'
    elif(dataset.startswith('caltech')):
         dataset_name = 'caltech101'
    elif(dataset.startswith('roxford5k_drift')):
         dataset_name = 'roxford5k'
    elif(dataset.startswith('rparis6k_drift')):
         dataset_name = 'rparis6k'            
    else:
        dataset_name = dataset
        

    # loading imlist, qimlist, and gnd, in cfg as a dict
    gnd_fname = os.path.join(dir_main, dataset_name, 'gnd_{}.pkl'.format(dataset))
        

    print('reading file')
    print(gnd_fname)
    with open(gnd_fname, 'rb') as f:
        cfg = pickle.load(f)
    cfg['gnd_fname'] = gnd_fname
    
    #print(">>>>>>>>>>>>>>>>>> GND HERE <<<<<<<<<<<<<<<<<<<<<")
    #print(cfg.keys())
    #print(cfg['qimlist'])
    #print(cfg['gnd_fname'])

    cfg['ext'] = '.jpg'
    cfg['qext'] = '.jpg'
    
    
    cfg['dir_data'] = os.path.join(dir_main, dataset_name)
    cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')

    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])
    
    cfg['im_fname'] = config_imname
    cfg['qim_fname'] = config_qimname

    cfg['dataset'] = dataset

    return cfg

def config_imname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])

def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])
