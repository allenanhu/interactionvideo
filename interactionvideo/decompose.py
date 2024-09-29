"""
Decompose a video into a stream of images and audio

"""


from os.path import join

from tqdm import tqdm

import pliers as pl
pl.set_options(use_generators=True, cache_transformers=False)
 

def convert_video_to_images(video_path, work_path): 
    """
    Convert a video into a stream of images with a sampling rate of 
    10 frames per second
    Args:
        video_path (str): path to the video file
        work_path (str): path to the work folder
    """
    
    # Set the output path
    ImageTempPath = join(work_path,'image_temp')
    
    # Load in the video
    from pliers.stimuli import VideoStim
    video_stim = VideoStim(video_path)
    
    print('Video is %s seconds long.' %(video_stim.duration) + '\n')

    # Convert the video into a list of images
    # The default sampling rate is 10 frames per sec
    from pliers.filters import FrameSamplingFilter
    frame_filter = FrameSamplingFilter(hertz=10)
    
    slice_image_stim_list = frame_filter.transform(video_stim)
        
    # Save the images to disk
    for each_image in tqdm(slice_image_stim_list.frames,\
                           total = slice_image_stim_list.n_frames):
        
        each_image.save(join(ImageTempPath,\
                             'image_split-%s.png' % each_image.frame_num))
    
    print('Video is sampled to %d images.' \
          %(slice_image_stim_list.n_frames) + '\n')

    print('Video to images finished.' + '\n')
    
    video_stim.clip.close()
    
    return True


def convert_video_to_audios(video_path, work_path):
    """
    Convert a video into audios
    Args:
        video_path (str): path to the video file
        work_path (str): path to the work folder
    """
    
    # Set the output path
    AudioTempPath = join(work_path,'audio_temp')
    
    # Load in the video
    from pliers.stimuli import VideoStim
    video_stim = VideoStim(video_path)
    
    # Convert the video into audio files
    from pliers.converters import VideoToAudioConverter
    v2a_converter = VideoToAudioConverter()
    
    audio_stim = v2a_converter.transform(video_stim)

    # Save the full audio
    audio_stim.save(join(AudioTempPath,'audio_full.wav'))
    
    # Split the audio into mono channels for later use
    from pydub import AudioSegment
    audio_seg = AudioSegment.from_file(join(AudioTempPath,'audio_full.wav'))

    # Check the number of channels
    audio_channels = audio_seg.channels

    # Split to left and right channels
    if audio_channels == 2:
         
        audio_seg_channel_list = audio_seg.split_to_mono()
    
        # Save the channels seperately
        with open(join(AudioTempPath,'audio_full_left.wav'), 'wb') as f:
            audio_seg_channel_list[0].export(f, format='wav')
            
        with open(join(AudioTempPath,'audio_full_right.wav'), 'wb') as f:
            audio_seg_channel_list[1].export(f, format='wav')

    print('Video to audios finished.' + '\n')
    
    video_stim.clip.close()
    audio_stim.clip.close()
           
    return True




    