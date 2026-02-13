from pyriksdagen.utils import download_corpus
from settings import data_dir


def dowload_speaker_metadata():
    pass
    if len([file for file in data_dir.iterdir() if file.is_file()]) < 25:
        download_corpus(partitions=["persons"])


def download_speech_data():
    if (
        not data_dir.exists()
        or len([subdir for subdir in data_dir.iterdir() if subdir.is_dir()]) < 158
    ):
        download_corpus(partitions=["records"])


if __name__ == "__main__":
    dowload_speaker_metadata()
    download_speech_data()
