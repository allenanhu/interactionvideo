# Hu and Ma (2024) Video Processing Package

## Description

`interactionvideo` package processes videos to study interpersonal interactions and communications.

Please refer to and cite the research paper: Hu and Ma (2024), "Persuading Investors: A Video-Based Study", available at [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3583898), by [Allen Hu](https://www.allenanhu.com) (UBC Sauder, allen.hu@sauder.ubc.ca) and 
[Song Ma](https://songma.github.io) (Yale University and NBER, song.ma@yale.edu)

**For academic research purposes only.**

## File Structure

- `example.py` and `example.ipynb`: step-by-step tutorials
  - We recommend you start from `example.ipynb`
- `interactionvideo`: main package folder
- `data`: data input folder
- `output`: output result folder
- `mlmodel`: pre-trained ML model folder
- `PythonSDK`: Face++ Python SDK folder, downloaded directly from [Github](https://github.com/FacePlusPlus/facepp-python-sdk)

```bash
├── interactionvideo
│   ├── __pycache__
│   ├── prepare.py
│   ├── decompose.py
│   ├── faceppml.py
│   ├── googleml.py
│   ├── textualanalysis.py
│   ├── audioml.py
│   ├── aggregate.py
│   └── utils.py
├── data
│   ├── example_video.mp4
│   └── VideoDictionary.csv
├── mlmodel
│   ├── pyAudioAnalysis
│   └── speechemotionrecognition
├── output
│   ├── audio_temp
│   ├── image_temp
│   └── result_temp
├── PythonSDK
├── README.md
├── requirement.txt
├── environment.yml
├── example.py
└── example.ipynb
```

## Usage

The video processing involves the following steps:

1. Set up folders and check dependencies and requirements
2. Extract images and audios from a video using [`pliers`](https://github.com/PsychoinformaticsLab/pliers)
3. Extract text from audios using [Google Speech2Text API](https://cloud.google.com/speech-to-text)
4. Process images (faces) using [Face++ API](https://www.faceplusplus.com/)
5. Process text using [Loughran and McDonald (2011)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2010.01625.x) Finance Dictionary and [Nicolas, Bai, and Fiske (2019)](https://osf.io/preprints/psyarxiv/afm8k) Social Psychology Dictionary
6. Process audios using pre-trained ML models in [`pyAudioAnalysis`](https://github.com/tyiannak/pyAudioAnalysis) and [`speechemotionrecognition`](https://github.com/hkveeranki/speech-emotion-recognition)
7. Aggregate information from 3V (visual, vocal, and verbal) to video level

## Environment and Requirements

The package was developed and tested in Python 3.7.16 with [Anaconda](https://www.anaconda.com) under Windows 11 and Linux 4.18.0. We suggest creating a seperate environment `interactionvideo` for the use of this package.

You can create the `interactionvideo` environment and install the required packages by running the following commands in the terminal:
> conda create -n interactionvideo python=3.7.16
> conda activate interactionvideo
> pip install -r requirements.txt

Alternatively, if you want to use the `environment.yml` file to create the environment, please run the following commands in the terminal:
> conda env create -f environment.yml
> conda activate interactionvideo

### Face++

Get Face++ key and secret from https://www.faceplusplus.com. If you register with https://console.faceplusplus.com/register, use https://api-us.faceplusplus.com as the server. If you register with https://console.faceplusplus.com.cn/register, use https://api-cn.faceplusplus.com as the server.

The Python SDK of Face++ is included in this repo (folder `Python SDK`). Check the orginial package at [Github](https://github.com/FacePlusPlus/facepp-python-sdk).


### Google Cloud

Set up Google Cloud following
 - https://cloud.google.com/python
 - https://cloud.google.com/storage/docs/quickstart-console
 - https://developers.google.com/workspace/guides/create-credentials
 - https://cloud.google.com/storage/docs/creating-buckets

You may need to download a json Google Cloud credential file and set the environment variable `GOOGLE_APPLICATION_CREDENTIALS`. Remember to create a Google Cloud Storage bucket.

### FFmpeg

For Windows users, download `FFmpeg` from https://www.ffmpeg.org and add `FFmpeg` to your system path. [Here](https://phoenixnap.com/kb/ffmpeg-windows) is a tutorial.
For Linux/Mac users, install `FFmpeg` via your package manager.