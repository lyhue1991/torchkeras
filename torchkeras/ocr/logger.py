import logging 
import os 

def set_logging(name='ocr', level=None, verbose=True):
    """Sets up logging for the given name."""
    
    if level is None:
        rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
        level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format':'[%(asctime)s] %(name)s %(levelname)s: %(message)s',
                 'datefmt':"%Y-%m-%d %H:%M:%S"
            }
        },
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False}}})
    
def get_logger(name='ocr'):
    return logging.getLogger(name)
