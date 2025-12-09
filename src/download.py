from pyriksdagen.utils import download_corpus
from settings import data_dir

data_dir.mkdir(exist_ok=True)


def dowload_speaker_metadata():
    download_corpus(partitions=["persons"])


def download_speech_data():
    download_corpus(partitions=["records"])
