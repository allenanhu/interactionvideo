"""
Use ML APIs of Face++ to process images (faces) and get facial emotions

Download the Python SDK of FacePlusPlus from
https://github.com/FacePlusPlus/facepp-python-sdk
PythonSDK folder is named to FacePlusPlusPythonSDK

"""

import os
from os.path import join

import time

import pliers as pl
pl.set_options(use_generators=True, cache_transformers=False)

from PythonSDK.facepp import API,File

from tqdm import tqdm

import pandas as pd

from interactionvideo.utils import resize_and_save_image, facepp_result_to_df,\
                                   clean_facepp_result_df


def process_image_by_facepp(video_path, work_path, \
                            facepp_key, facepp_secret, facepp_server):
    """
    Process images by ML API of Face++
    Get the key and secret from
    https://www.faceplusplus.com/
    If you register at https://console.faceplusplus.com/register
    Use https://api-us.faceplusplus.com as the server
    If you register at https://console.faceplusplus.com.cn/register
    Use https://api-cn.faceplusplus.com as the server    
    Args:
        video_path (str): path to the video file
        work_path (str): path to the work folder
        facepp_key (str): Key of Face++ API
        facepp_secret (str): Secret of Face++ API
        facepp_server (str): Server of Face++ API
    """

    # Set the image_temp and result_temp path
    ImageTempPath = join(work_path,'image_temp')
    ResultTempPath = join(work_path,'result_temp')
        
    # A template for all image files
    ImageFileName = join(ImageTempPath,'image_split-%s.png')
    
    # Process images by Face++ API
    # Construct an instance for Face++ API
    facepp_api = API(facepp_key, facepp_secret, facepp_server)
    
    # Set the attributes you want from Face++ API
    FaceppAttibutes = 'gender,age,emotion,beauty,mouthstatus'
    
    # Load in the video
    from pliers.stimuli import VideoStim
    video_stim = VideoStim(video_path)
    
    # Convert the video into a list of images
    # The default sampling rate is 10 frames per sec
    from pliers.filters import FrameSamplingFilter
    frame_filter = FrameSamplingFilter(hertz=10)
    
    slice_image_stim_list = frame_filter.transform(video_stim)
    
    # Initialize the container to store result DF of each image
    result_df_list = []
    
    print('Face++ API begins. %s images to process.' \
          %(slice_image_stim_list.n_frames) + '\n' )

    # Process list of images
    for each_image in tqdm(slice_image_stim_list.frames,\
                           total = slice_image_stim_list.n_frames):  
                
        # Check whether the image is too large
        # Face++ API does not handle very large image
        if os.path.getsize(ImageFileName % each_image.frame_num) > 2 * 1024 * 1024:
            # If large, resize and overwirte the image
            resize_and_save_image(ImageFileName % each_image.frame_num)
                
        # Upload the image file to Face++ API
        each_result = facepp_api.detect( 
                         image_file = File(ImageFileName % each_image.frame_num),\
                         image_url=None, image_base64=None, return_landmark=1,\
                         return_attributes = FaceppAttibutes)
        
        # Skip this image if no face found
        if each_result.face_num == 0:
            continue
        
        # Convert returns (dict) from Face++ API into a DataFrame
        each_result_df = facepp_result_to_df(each_result)
        temp_col_list = list(each_result_df.columns)
        
        # Add the name, onset, and duration of the image 
        each_result_df['ImageName'] = each_image.name
        each_result_df['Onset'] = round(each_image.onset, 2)
        each_result_df['Duration'] = round(each_image.duration, 2)
        each_result_df['Offset'] = each_result_df['Onset'] + \
                                   each_result_df['Duration']
                                   
        # Reorder the columns
        each_result_df = each_result_df[['ImageName','Onset','Offset','Duration'] + \
                                        temp_col_list]
        
        # Append the result for this image to the all result list
        result_df_list.append(each_result_df)
        
        # Pause and wait, Face++ has limit on number of request per second
        time.sleep(5)
        
    print('Face++ API ends. %s images processed.' \
          %(slice_image_stim_list.n_frames) + '\n' )        
    
    # Merge results from all images
    facepp_result_df = pd.concat(result_df_list).reset_index(drop=True)
    
    # Save the full result
    facepp_result_df.to_csv(join(ResultTempPath,'face_panel_facepp.csv'), \
                encoding='utf-8', sep=',', float_format='%.2f', index=False)

    # Slice the relevent columns and construct measures
    facepp_result_clean_df = clean_facepp_result_df(facepp_result_df)    

    # Save the clean result
    facepp_result_clean_df.to_csv(join(ResultTempPath,'face_panel.csv'), \
                encoding='utf-8', sep=',', float_format='%.2f', index=False)
    
    return facepp_result_df, facepp_result_clean_df
    
