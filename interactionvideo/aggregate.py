"""
Aggregate information from 3V (visual, vocal, and verbal) into video level

"""

from os.path import join

import pandas as pd
from pandas import DataFrame as df

def aggregate_3v_to_video(work_path):
    """
    Aggregate information from 3V (visual, vocal, and verbal) into video level
    Generate video level DataFrame
    NOTE:
        The aggregation here is simplified for the case of one person and
        full audio level analysis
        The complete aggregation requires sentence-by-sentence audio emotion
        and matching speaker with speech
    Args:
        work_path (str): path to the work folder
    """


    # Set the audio_temp and result_temp path
    ResultTempPath = join(work_path,'result_temp')
    
    # Load visual result panel
    visual_result_df = pd.read_csv(join(ResultTempPath,'face_panel.csv'))
    
    # Load vocal result panel
    vocal_result_df1 = pd.read_csv(join(ResultTempPath,\
                                        'audio_panel_pyAudioAnalysis.csv'))
    
    vocal_result_df2 = pd.read_csv(join(ResultTempPath,\
                                'audio_panel_speechemotionrecognition.csv'))
    
    # Load verbal result panel
    verbal_result_df = pd.read_csv(join(ResultTempPath,'text_panel.csv'))
    
    # Create video panel
    video_result_df = df()

    video_result_df['Number of Faces'] = [visual_result_df['Number of Faces'].median()]
    video_result_df['Gender'] = [sorted(list(visual_result_df['Gender']))\
                                [int(len(visual_result_df)/2)]]
    video_result_df['Age'] = [visual_result_df['Age'].median()]
    
    video_result_df['Visual-Positive'] = [visual_result_df['Visual-Positive'].mean()]
    video_result_df['Visual-Negative'] = [visual_result_df['Visual-Negative'].mean()]
    video_result_df['Visual-Beauty'] = [visual_result_df['Visual-Beauty'].mean()]

    video_result_df['Vocal-Positive'] = [vocal_result_df2['Vocal-Positive'].mean()]
    video_result_df['Vocal-Negative'] = [vocal_result_df2['Vocal-Negative'].mean()]    
    video_result_df['Vocal-Arousal'] = [vocal_result_df1['Vocal-Arousal'].mean()]
    video_result_df['Vocal-Valence'] = [vocal_result_df1['Vocal-Valence'].mean()]

    
    video_result_df['Verbal-Positive'] = [verbal_result_df['Verbal-Positive'].mean()]
    video_result_df['Verbal-Negative'] = [verbal_result_df['Verbal-Negative'].mean()]
    video_result_df['Verbal-Warmth'] = [verbal_result_df['Verbal-Warmth'].mean()]
    video_result_df['Verbal-Ability'] = [verbal_result_df['Verbal-Ability'].mean()]
    
    # Save the result
    video_result_df.to_csv(join(ResultTempPath,'video_panel.csv'),\
                encoding='utf-8', sep=',', float_format='%.2f', index=False)
    
    print('3V to video aggregation finished.' + '\n')
    
    return video_result_df
