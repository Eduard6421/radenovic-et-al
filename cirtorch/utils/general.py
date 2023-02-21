import os
import hashlib
from pathlib import Path

def get_root():
    return Path(__file__).parents[5]


def get_data_root():
    return os.path.join(get_root(), 'Datasets')

def get_model_root():
    return os.path.join(get_root(), 'Pretrained_Models')

def get_embedding_root():
    return os.path.join(get_root(), 'Embeddings')

def get_results_root():
    return os.path.join(get_root(), 'Results')

def get_embedding_subfolder():
    return os.path.join(get_embedding_root(), 'CNN_Image_Retrieval')

def htime(c):
    c = round(c)
    
    days = c // 86400
    hours = c // 3600 % 24
    minutes = c // 60 % 60
    seconds = c % 60

    if days > 0:
        return '{:d}d {:d}h {:d}m {:d}s'.format(days, hours, minutes, seconds)
    if hours > 0:
        return '{:d}h {:d}m {:d}s'.format(hours, minutes, seconds)
    if minutes > 0:
        return '{:d}m {:d}s'.format(minutes, seconds)
    return '{:d}s'.format(seconds)


def sha256_hash(filename, block_size=65536, length=8):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()[:length-1]