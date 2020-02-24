from os import walk
import warnings
warnings.filterwarnings("ignore")
from pydub import AudioSegment
from pydub.utils import make_chunks

source = "C:\\Users\\odelia\\Desktop\\vo\\"


# cut the wav file to 1 second
def create_data_set_from_audio(source):
    for (dirpath, dirnames, filenames) in walk(source):
        j = 0
        for filename in filenames:
            my_audio = AudioSegment.from_file(source + "\\" + filename, "wav")
            chunk_length_ms = 1000
            chunks = make_chunks(my_audio, chunk_length_ms)

            size=len(chunks)
            for i, chunk in enumerate(chunks):
                if i == size-1:
                    break
                chunk_name = f"female_clb{j}-{i}.wav"
                print("exporting", chunk_name)
                chunk.export(chunk_name, format="wav")

            j = j+1
        break

    return

create_data_set_from_audio(source)
