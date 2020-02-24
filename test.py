from os import walk
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import warnings
warnings.filterwarnings("ignore")
from pydub import AudioSegment
from pydub.utils import make_chunks

source= "C:\\Users\\odelia\\Desktop\\deep learning"


def create_data_set_from_audio(source):
    for (dirpath, dirnames, filenames) in walk(source):
        j=0
        for filename in filenames:
            myaudio = AudioSegment.from_file(source + "\\" + filename, "wav")
            chunk_length_ms = 1000  # pydub calculates in millisec
            chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of one sec

            # Export all of the individual chunks as wav files
            size=len(chunks)
            for i, chunk in enumerate(chunks):
                if i == size-1:
                    break
                chunk_name = f"chunk{j}-{i}.wav"
                print("exporting", chunk_name)
                chunk.export(chunk_name, format="wav")

            j=j+1
        break

    return

create_data_set_from_audio(source)
