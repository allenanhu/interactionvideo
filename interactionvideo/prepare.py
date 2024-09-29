"""
Set up folders and check requirements

"""

from os.path import join
import os


def setup_folder(work_path):
    """
    Set up folders to store temp images and audios
    Args:
        work_path (str): path to the work folder
    """
    
    if not os.path.exists(work_path):
        raise Exception('Work Path does not exist.')
    
    # Set up temp folder for audio
    if not os.path.exists(join(work_path,'audio_temp')):
        os.makedirs(join(work_path,'audio_temp'))
        print('Audio Temp folder has been set up.' + '\n')
        
    # Set up temp folder for image
    if not os.path.exists(join(work_path,'image_temp')):
        os.makedirs(join(work_path,'image_temp'))
        print('Image Temp folder has been set up.' + '\n')

    # Set up temp folder for result
    if not os.path.exists(join(work_path,'result_temp')):
        os.makedirs(join(work_path,'result_temp'))
        print('Result Temp folder has been set up.' + '\n')
        
    return True


def check_requirements():
    """
    Check whether packages required in interactionvideo are installed.
    """
    
    try:
        import tqdm, pliers, pydub
    except ImportError as e:
        print(e)
        
    print('decompose.py requirements satisfied.' + '\n')
    
    try:
        from PythonSDK.facepp import API,File
        import PIL, pandas, numpy
        
    except ImportError as e:
        print(e)
        
    print('faceppml.py requirements satisfied.' + '\n')
    
    try:
        from google.cloud import storage
        storage_client = storage.Client()  
        from google.cloud import speech_v1p1beta1 as speech
        client = speech.SpeechClient()
        import codecs

    except ImportError as e:
        print(e)
        
    print('googleml.py requirements satisfied.' + '\n')

    try:
        import pyAudioAnalysis, speechemotionrecognition

    except ImportError as e:
        print(e)
        
    print('audioml.py requirements satisfied.' + '\n')

    return True
    