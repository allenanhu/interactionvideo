"""
Use Pre-trained ML models to process audios and get vocal emotions
(e.g., vocal arousal, valence, positive, negative)

Use Pre-trained models in pyAudioAnalysis to get Arousal and Valence

For more details, please check
https://github.com/tyiannak/pyAudioAnalysis

Use Pre-trained models in speech-emotion-recognition to get Positive and Negative

For more details, please check
https://github.com/hkveeranki/speech-emotion-recognition

"""

import os
from os.path import join

from pandas import DataFrame as df

from interactionvideo.utils import extract_audio_feature


def process_audio_by_pyAudioAnalysis(work_path, model_path):
    """
    Process audios by pre-trained SVM ML models in pyAudioAnalysis
    Get vocal arousal and vocal valence
    The pre-trained models are
        - svmSpeechEmotion_arousal
        - svmSpeechEmotion_arousalMEANS
        - svmSpeechEmotion_valence
        - svmSpeechEmotion_valenceMEANS
    For more details, please check
    https://github.com/tyiannak/pyAudioAnalysis/wiki/4.-Classification-and-Regression
    NOTE:
        This function processes the whole audio together for simplicity
        In Hu and Ma (2020), we split the audio into sentence-by-sentence
        segments and construct vocal emotions sentence by sentence
    Args:
        work_path (str): path to the work folder
        model_path (str): path to pyAudioAnalysis pre-trained models
    """
    
    # Set the audio_temp and result_temp path
    AudioTempPath = join(work_path,'audio_temp')
    ResultTempPath = join(work_path,'result_temp')
    
    # This is required by pyAudioAnalysis
    # We don't need a svmSpeechEmotion folder
    pyAudioAnalysisModel = join(model_path,'svmSpeechEmotion')
    
    # Check whether pre-trained models are ready
    if not os.path.isfile(join(model_path,'svmSpeechEmotion_arousal')):
        raise Exception('Pre-trained model svmSpeechEmotion_arousal \
                        does not exist.' + '\n')

    if not os.path.isfile(join(model_path,'svmSpeechEmotion_arousalMEANS')):
        raise Exception('Pre-trained model svmSpeechEmotion_arousalMEANS \
                        does not exist.' + '\n')
        
    if not os.path.isfile(join(model_path,'svmSpeechEmotion_valence')):
        raise Exception('Pre-trained model svmSpeechEmotion_valence \
                        does not exist.' + '\n')
        
    if not os.path.isfile(join(model_path,'svmSpeechEmotion_valenceMEANS')):
        raise Exception('Pre-trained model svmSpeechEmotion_valenceMEANS \
                        does not exist.' + '\n')
    
    # Get the audio meta info from the raw audio file
    from pydub import AudioSegment
    
    audio_seg = AudioSegment.from_file(join(AudioTempPath,'audio_full.wav'))
    audio_length = audio_seg.duration_seconds
    
    from pyAudioAnalysis import audioTrainTest as aT
    
    print('pyAudioAnalysis vocal emotion analysis begins. %s seconds audio to process.' \
          %(audio_length) + '\n' )
    
    # Get vocal arousal and valence
    audio_emotion = aT.file_regression(join(AudioTempPath, 'audio_full.wav'),\
                                pyAudioAnalysisModel, 'svm')                 

    print('pyAudioAnalysis ML model loaded.' + '\n' )
    
    # Save the vocal arousal and valence by pyAudioAnalysis
    audio_result_df = df()
    
    audio_result_df['Onset'] = [round(0, 2)]
    audio_result_df['Offset'] = [round(audio_length, 2)]
    audio_result_df['Duration'] = audio_result_df['Offset'] - audio_result_df['Onset']
    audio_result_df['Vocal-Arousal'] = [audio_emotion[0][0]]
    audio_result_df['Vocal-Valence'] = [audio_emotion[0][1]]
    
    audio_result_df.to_csv(join(ResultTempPath,'audio_panel_pyAudioAnalysis.csv'),\
                encoding='utf-8', sep=',', float_format='%.2f', index=False)
    
    print('pyAudioAnalysis vocal emotion analysis ends. %s seconds audio processed.' \
          %(audio_length) + '\n' )

    return audio_result_df



def process_audio_by_speechemotionrecognition(work_path, model_path):
    """
    Process audios by pre-trained LSTM ML models in speechemotionrecognition
    Get vocal positive and vocal negative
    The pre-trained models are
        - best_model_LSTM_39.h5
    For more details, please check
    https://github.com/hkveeranki/speech-emotion-recognition
    NOTE:
        This function processes the whole audio together for simplicity
        In Hu and Ma (2020), we split the audio into sentence-by-sentence
        segments and construct vocal emotions sentence by sentence
    Args:
        work_path (str): path to the work folder
        model_path (str): path to speechemotionrecognition pre-trained models
    """
    
    # Set the audio_temp and result_temp path
    AudioTempPath = join(work_path,'audio_temp')
    ResultTempPath = join(work_path,'result_temp')
    
    # Set the model to LSTM
    speechemotionrecognitionModel = \
    join(model_path,'best_model_LSTM_39.h5')
    
    # Check whether pre-trained models are ready
    if not os.path.isfile(speechemotionrecognitionModel):
        raise Exception('Pre-trained model best_model_LSTM_39.h5 \
                        does not exist.' + '\n')
        
    
    # Get the audio meta info from the raw audio file
    from pydub import AudioSegment
    
    audio_seg = AudioSegment.from_file(join(AudioTempPath,'audio_full.wav'))
    audio_length = audio_seg.duration_seconds
        
    print('speechemotionrecognition vocal emotion analysis begins. %s seconds audio to process.' \
          %(audio_length) + '\n' )
    
    # Load the pre-trained LSTM via keras
    from speechemotionrecognition.dnn import LSTM            
    
    lstm_model = LSTM(input_shape=(199, 39), num_classes=4)
    lstm_model.restore_model(speechemotionrecognitionModel)

    print('speechemotionrecognition ML model loaded.' + '\n' )
    
    # Extract audio features as input of LSTM
    audio_feature = extract_audio_feature(work_path,audio_seg)
    
    # Get vocal positive and negative
    audio_emotion = list(lstm_model.model.predict(audio_feature)[0])
    
    # Save the vocal positive and negative by speechemotionrecognition
    audio_result_df = df()                            
    
    audio_result_df['Onset'] = [round(0, 2)]
    audio_result_df['Offset'] = [round(audio_length, 2)]
    audio_result_df['Duration'] = audio_result_df['Offset'] - audio_result_df['Onset']
    audio_result_df['Vocal-Positive'] = [audio_emotion[2]]
    audio_result_df['Vocal-Negative'] = [audio_emotion[3]]
    
    audio_result_df.to_csv(join(ResultTempPath,\
                                'audio_panel_speechemotionrecognition.csv'),\
                encoding='utf-8', sep=',', float_format='%.2f', index=False)
    
    print('speechemotionrecognition vocal emotion analysis ends. %s seconds audio processed.' \
          %(audio_length) + '\n' )

    return audio_result_df


