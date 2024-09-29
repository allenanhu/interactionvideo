"""
Some util functions

"""
import os
from os.path import join

from PIL import Image

import pandas as pd
from pandas import DataFrame as df
import numpy as np

def resize_and_save_image(image_path):
    """
    Resize the image and overwrite the original file
    Args:
        image_path (str): path to the image file
    """
    
    img = Image.open(image_path)
    
    resize_ratio = os.path.getsize(image_path)/(2 * 1024 * 1024) * 1.2
    
    img = img.resize((int(img.size[0]/resize_ratio),\
                      int(img.size[1]/resize_ratio)), Image.ANTIALIAS)
    
    img.save(image_path)
    
    
    
def facepp_result_to_df(facepp_result):
    """
    Convert returns from Face++ API into a DataFrame
    Args:
        facepp_result (ObjectDict): return from calling Face++ API
    """
    
    # Number of faces in the image
    face_num = facepp_result.face_num
    # Image ID assigned by Face++
    image_id = facepp_result.image_id
    # Get the list of faces
    faces_list = facepp_result.faces
    
    # Initialize the container
    face_data_list = []
    
    for each_face in faces_list:
        
        # Initialize the container
        each_face_data_dict = dict()
        
        # face_token
        each_face_data_dict['face_token'] = each_face['face_token']
        
        # face_rectangle
        for each_key in each_face['face_rectangle']:
            each_face_data_dict['face_rectangle#'+each_key] = \
                                each_face['face_rectangle'][each_key]
        
        # landmark
        for each_key in each_face['landmark']:
            for each_subkey in each_face['landmark'][each_key]:
                each_face_data_dict['landmark#'+each_key+'#'+each_subkey] = \
                            each_face['landmark'][each_key][each_subkey]
               
        # attributes
        for each_key in each_face['attributes']:
            
            if each_key not in ['blur','eyestatus']:
                for each_subkey in each_face['attributes'][each_key]:  
                    each_face_data_dict['attributes#'+each_key+'#'+each_subkey] = \
                                each_face['attributes'][each_key][each_subkey]
            else:
                for each_subkey in each_face['attributes'][each_key]:  
                    for each_subsubkey in each_face['attributes'][each_key][each_subkey]:  
                        each_face_data_dict['attributes#'+each_key+'#'+each_subkey+'#'+each_subsubkey] = \
                        each_face['attributes'][each_key][each_subkey][each_subsubkey]   
        
        # Add image information
        each_face_data_dict['face_num'] = face_num
                               
        # Add image id
        each_face_data_dict['image_id'] = image_id
    
        # Add to whole image data list
        face_data_list.append(each_face_data_dict)
        
        # Merge the results
        face_result_df = df(face_data_list)
        
        # Drop some meta columns
        face_result_df.drop(columns=['face_token', 'image_id'], inplace=True)


    return face_result_df



def clean_facepp_result_df(facepp_result_df):
    """
    Clean the result DataFrame of Face++
    Only keep the relevant columns
    Args:
        facepp_result_df (DataFrame): result DataFrame of Face++
    """ 
    # Define some visual measures
    pos_emotion_col_list = ['attributes#emotion#happiness']
                           
    neg_emotion_col_list = ['attributes#emotion#anger',\
                           'attributes#emotion#disgust',\
                           'attributes#emotion#fear',\
                           'attributes#emotion#sadness']
                           
    beauty_col_list = ['attributes#beauty#male_score',\
                       'attributes#beauty#female_score']
                       
    # Get Visual Positive
    facepp_result_df['Visual-Positive'] = \
    facepp_result_df[pos_emotion_col_list].sum(axis=1) / 100

    # Get Visual Negative
    facepp_result_df['Visual-Negative'] = \
    facepp_result_df[neg_emotion_col_list].sum(axis=1) / 100  

    # Get Visual Beauty
    facepp_result_df['Visual-Beauty'] = \
    facepp_result_df[beauty_col_list].mean(axis=1) / 100  
    
    # Rename columns
    facepp_result_df.rename(columns={'attributes#gender#value': 'Gender',\
                                     'attributes#age#value': 'Age',\
                                     'face_num': 'Number of Faces'},\
                            inplace = True)
      
    # Set the columns we want to keep in the clean result DF                                   
    keep_col_list = ['Onset','Offset','Duration','Number of Faces',\
                     'Gender', 'Age', \
                     'Visual-Positive','Visual-Negative','Visual-Beauty']
    
    # Slice the relevant columns
    facepp_result_clean_df = facepp_result_df[keep_col_list]
    
    return facepp_result_clean_df
                    
  
def google_result_to_df(google_result):
    """
    Convert returns from Google Cloud Speech2Text API into a DataFrame
    Args:
        google_result (LongRunningRecognizeResponse): return from Google API
    """

    # Initialize the container 
    text_result_list = []
    
    # Go through the text one by one
    for each_text in google_result.results:
        
        # Extract results
        each_result = each_text.alternatives[0]
        words_info = each_result.words
     
        # Initialize the container
        each_text_result_df = df()
        
        # Add the text
        each_text_result_df['Text'] = [each.word for each in words_info]
        # Add the time
        each_text_result_df['Onset'] = [each.start_time.total_seconds() for each in words_info]
        
        each_text_result_df['Offset'] = [each.end_time.total_seconds() for each in words_info]
        
        each_text_result_df['Duration'] = each_text_result_df['Offset'] - \
                                          each_text_result_df['Onset']
        # Add the punctuation and check sentence end  
        each_text_result_df['Sentence End'] = [True if each[-1] in \
                            [',','.','!','?'] else False for each in \
                            each_text_result_df['Text']]
        
        # Append to the all result list
        text_result_list.append(each_text_result_df)
        
    text_result_df = pd.concat(text_result_list).reset_index(drop=True)

    return text_result_df

  
def extract_audio_feature(work_path, audio_seg):
    """
    Extract features of audios as input of speechemotionrecognition LSTM model
    Need to sample to 16KHz to feed into speechemotionrecognition LSTM model
    Args:
        work_path (str): path to the work folder
        audio_seg (AudioSegment): audio segment loaded by pydub
    """
    # Set the audio_temp path  
    AudioTempPath = join(work_path,'audio_temp')
    
    # Re-sample to 16KHz to feed into speechemotionrecognition
    if audio_seg.channels == 1:
        with open(join(AudioTempPath,'audio_temp_16KHz.wav'), 'wb') as f:
            audio_seg.set_frame_rate(16000).export(f, format='wav')        
    else:
        audio_seg_channel_list = audio_seg.split_to_mono()
        
        with open(join(AudioTempPath,'audio_temp_16KHz.wav'), 'wb') as f:
            audio_seg_channel_list[0].set_frame_rate(16000).export(f, format='wav')

    # Extract audio features
    from speechemotionrecognition.utilities import get_feature_vector_from_mfcc
    audio_feature = get_feature_vector_from_mfcc(
            join(AudioTempPath,'audio_temp_16KHz.wav'), flatten=False)
    
    # Reshape for LSTM input
    audio_feature = np.array([audio_feature])
            
    return audio_feature


 
    
    
    
    
    
    
    
    