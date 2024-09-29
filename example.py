"""
This example shows how to use `interactionvideo` package 
to process a video to study interpersonal interactions and communications. 

If you use this package in your research,
please cite and refer to our research paper: Hu and Ma (2024), 
"Persuading Investors: A Video-Based Study"

Allen Hu, UBC Sauder
Song Ma, Yale University and NBER

"""

import os
from os.path import join

# Set the path
RootPath = ''

# Set your video file path here
VideoFilePath = join(RootPath,'data','example_video.mp4')

# Set your work path here
# Work path is where to store meta files and output files
WorkPath = join(RootPath,'output')

# Set your path to the Google Cloud credential file here
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''

# Append path to ffmpeg
FFMPEGPath = ''
os.environ['PATH'] += os.pathsep + FFMPEGPath

# Set up the folders
from interactionvideo.prepare import setup_folder

setup_folder(WorkPath)

# check the requirements for interactionvideo
from interactionvideo.prepare import check_requirements

check_requirements()

# Decompose a video file into images and audios
from interactionvideo.decompose import convert_video_to_images,\
                                       convert_video_to_audios
                             
# Convert the video into a stream of images
# The default sampling rate is 10 frames per second
# Find the output at WorkPath\image_temp
convert_video_to_images(VideoFilePath, WorkPath)

# Convert the video into audios
# Find the output at WorkPath\audio_temp
convert_video_to_audios(VideoFilePath, WorkPath)



# Use Face++ ML API to process images
# Return a csv file of facial emotions, gender, predicted age
# Find the output
# - WorkPath\result_temp\face_panel_facepp.csv (full returns from Face++)
# - WorkPath\result_temp\face_panel.csv (clean results)

# Get the key and secret from
# https://www.faceplusplus.com
# If you register at https://console.faceplusplus.com/register
# Use https://api-us.faceplusplus.com as the server
# If you register at https://console.faceplusplus.com.cn/register
# Use https://api-cn.faceplusplus.com as the server
# The Python SDK of Face++ is included in this repo (Python SDK)
# Check the orginial package at
# https://github.com/FacePlusPlus/facepp-python-sdk

# NOTE:
#     This function now processes the video with one person for simplicity
#     In Hu and Ma (2024), we process videos with multiple people
#     Diarization (matching face with speech) is conducted by detecting
#     the movement of mouth


from interactionvideo.faceppml import process_image_by_facepp

# Set your Face++ key, secret, and server here                           
FaceppKey = ''
FaceppSecret = ''
FaceppServer = 'https://api-us.faceplusplus.com'

facepp_result_df, facepp_result_clean_df = \
process_image_by_facepp(VideoFilePath, WorkPath,
                        FaceppKey, FaceppSecret, FaceppServer)


print(facepp_result_df.head(10))

print(facepp_result_clean_df.head(10))

# Use Google Speech2Text API to convert audio to text
# Return a csv file of text and punctuation
# Find the output at 
# - WorkPath\result_temp\script_google.txt (full speech script)
# - WorkPath\result_temp\text_panel_google.csv (text panel from Google)

# Set up Google Cloud following
# - https://cloud.google.com/python
# - https://cloud.google.com/storage/docs/quickstart-console
# - https://developers.google.com/workspace/guides/create-credentials
# - https://cloud.google.com/storage/docs/creating-buckets
# You may need to download a json Google Cloud credential file
# and set the environment variable GOOGLE_APPLICATION_CREDENTIALS
# Remember to create a Google Cloud Storage bucket 

from interactionvideo.googleml import upload_audio_to_googlecloud,\
                                      convert_audio_to_text_by_google

# Set your Google Cloud Storage bucket name here
GoogleBucketName = ''

upload_audio_to_googlecloud(WorkPath, GoogleBucketName)

google_result_text, google_result_df = \
convert_audio_to_text_by_google(WorkPath, GoogleBucketName)

print(google_result_text)

print(google_result_df.head(10))


# Dictionary-based textual analysis to get verbal measures
# (e.g., verbal positive, negative, warmth, ability)
#
# Use Loughran-McDonald (2011, 2016) Finance Dictionary (LM)
# to get Positive and Negative
#
# For more details, please check
# https://sraf.nd.edu/textual-analysis/resources/
#
# Use Nicolas, Bai, and Fiske (2019) Social Psychology Dictionary (NBF)
# to get Warmth and Ability
#
# For more details, please check
# https://psyarxiv.com/afm8k/

from interactionvideo.textualanalysis import process_text_by_dict

DictionaryPath = join(RootPath,'data','VideoDictionary.csv')

text_result_df = process_text_by_dict(WorkPath, DictionaryPath)

print(text_result_df.head(10))

# Process audios by pre-trained SVM ML models in pyAudioAnalysis
# Get vocal arousal and vocal valence
# The pre-trained models are
#     - svmSpeechEmotion_arousal
#     - svmSpeechEmotion_arousalMEANS
#     - svmSpeechEmotion_valence
#     - svmSpeechEmotion_valenceMEANS
# For more details, please check
# https://github.com/tyiannak/pyAudioAnalysis/wiki/4.-Classification-and-Regression
# NOTE:
#     This function processes the whole audio together for simplicity
#     In Hu and Ma (2024), we split the audio into sentence-by-sentence
#     segments and construct vocal emotions sentence by sentence
 
from interactionvideo.audioml import process_audio_by_pyAudioAnalysis
                              
pyAudioAnalysisModelPath = join(RootPath,'mlmodel','pyAudioAnalysis')

audio_result_df1 = \
process_audio_by_pyAudioAnalysis(WorkPath, pyAudioAnalysisModelPath)

print(audio_result_df1.head(10))


# Process audios by pre-trained LSTM ML models in speechemotionrecognition
# Get vocal positive and vocal negative
# The pre-trained models are
#     - best_model_LSTM_39.h5
# For more details, please check
# https://github.com/hkveeranki/speech-emotion-recognition
# speechemotionrecognition requires tensorflow and Keras
# NOTE:
#     This function processes the whole audio together for simplicity
#     In Hu and Ma (2024), we split the audio into sentence-by-sentence
#     segments and construct vocal emotions sentence by sentence


from interactionvideo.audioml import process_audio_by_speechemotionrecognition

speechemotionrecognitionModelPath = join(RootPath,'mlmodel','speechemotionrecognition')

audio_result_df2 = \
process_audio_by_speechemotionrecognition(WorkPath, speechemotionrecognitionModelPath)

print(audio_result_df2.head(10))


# Aggregate information from 3V (visual, vocal, and verbal) into video level
from interactionvideo.aggregate import aggregate_3v_to_video

video_result_df = aggregate_3v_to_video(WorkPath)

print(video_result_df.T)