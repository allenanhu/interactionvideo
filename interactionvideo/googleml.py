"""
Use ML APIs of Google Cloud to convert audio into text (Speech2Text)

To set up Google Cloud Computing, check
https://cloud.google.com/python
https://cloud.google.com/storage/docs/quickstart-console
https://cloud.google.com/speech-to-text

"""

from os.path import join

from interactionvideo.utils import google_result_to_df

def upload_audio_to_googlecloud(work_path, google_bucket_name):
    """
    Upload the audio file to Google Cloud bucket
    Check https://cloud.google.com/storage/docs/quickstart-console
    Args:
        work_path (str): path to the work folder
        google_bucket_name (str): name of the Google Cloud Storage Bucket
    """
    
    AudioTempPath = join(work_path,'audio_temp')
    
    from google.cloud import storage
    
    # Instantiates a client
    storage_client = storage.Client()

    bucket = storage_client.get_bucket(google_bucket_name)

    # Upload full channel audio
    blob = bucket.blob('audio_full.wav')
    blob.upload_from_filename(filename=join(AudioTempPath,'audio_full.wav'))

    print('Uploaded the audio file to Google Cloud.' + '\n')
    
    return True


def convert_audio_to_text_by_google(work_path, google_bucket_name):
    """
    Convert audio to text (Speech2Text) by ML API of Google Cloud
    Check https://cloud.google.com/speech-to-text
    Args:
        work_path (str): path to the work folder
        google_bucket_name (str): name of the Google Cloud Storage Bucket
    """

    # Set the audio_temp and result_temp path
    AudioTempPath = join(work_path,'audio_temp')
    ResultTempPath = join(work_path,'result_temp')
    
    from google.cloud import speech_v1p1beta1 as speech
    
    # Instantiates a client
    client = speech.SpeechClient()
    
    # Point to the audio in the Google Cloud Bucket
    audio = speech.RecognitionAudio(uri='gs://' + \
                   google_bucket_name + '/' +  'audio_full.wav')
    
    # Set some meta data to increse the accuracy
    metadata = speech.RecognitionMetadata()
    
    # Set interaction type
    metadata.interaction_type = (
        speech.RecognitionMetadata.InteractionType.PRESENTATION)
    
    # Set media type
    metadata.original_media_type = (
        speech.RecognitionMetadata.OriginalMediaType.VIDEO)
    
    # Set topic
    metadata.audio_topic = 'online application entrepreneurship accelerator'
    
    # Set context
    speech_context_list = [speech.types.SpeechContext(phrases=[each]) for each in \
                          ['online application','video pitch','business',\
                           'entrepreneurship','venture capital']]
    
    # Get the audio meta info from the raw audio file
    from pydub import AudioSegment
    
    audio_seg = AudioSegment.from_file(join(AudioTempPath,'audio_full.wav'))
    audio_samplerate = audio_seg.frame_rate
    audio_channels = audio_seg.channels
    audio_length = audio_seg.duration_seconds
    
    # Pool all config together
    config = speech.types.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=audio_samplerate,
        language_code='en-US',
        use_enhanced=True,
        model = 'video',
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
        enable_word_confidence=True,
        audio_channel_count=audio_channels,
        enable_separate_recognition_per_channel=False,
        metadata=metadata,
        speech_contexts=speech_context_list)
    
    print('Google Speech2Text begins. %s seconds audio to process.' \
          %(audio_length) + '\n' )
        
    # Call the API
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=1200)

    # Convert the return of API into a DataFrames
    google_result_df = google_result_to_df(response)

    # Save the raw transcript
    import codecs
    
    google_result_text = ' '.join(list(google_result_df['Text']))
    
    with codecs.open(join(ResultTempPath,'script_google.txt'), 'w', 'utf-8') as f:
        f.write(google_result_text)
        
    # Save the raw transcript and panel  
    google_result_df.to_csv(join(ResultTempPath,'text_panel_google.csv'), \
                            encoding='utf-8', sep=',', \
                            index=False, float_format='%.2f')
    
    print('Google Speech2Text ends. %s seconds audio processed.' \
          %(audio_length) + '\n' )
    
    return google_result_text, google_result_df




