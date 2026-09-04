from pathlib import Path
from .settings import data_dir
import httpx
from hashlib import sha256
from tqdm import tqdm


BLOCK_SIZE = 1024


def sha256sum(file_path: Path):
    with open(file_path, "rb") as file:
        bytes = file.read()
        raw_sha = sha256(bytes).hexdigest()
        return "sha256:" + raw_sha


def download(url: str, file_name: str, force=False):
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} cannot be found, aborting")

    out_file = data_dir / file_name
    if out_file.exists() and not force:
        return sha256sum(out_file)

    with httpx.stream("get", url=url, follow_redirects=True) as response:
        if response.status_code != 200:
            raise AttributeError(f"{response.status_code=} != 200")

        total_size = int(response.headers.get("content-length", 0))
        if total_size == 0:
            raise FileNotFoundError("File size is estimated to 0B")
        pbar = tqdm(
            desc=file_name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=BLOCK_SIZE,
        )
        with open(out_file, "wb") as file:
            for chunk in response.iter_raw(chunk_size=BLOCK_SIZE):
                file.write(chunk)
                pbar.update(BLOCK_SIZE)
    return sha256sum(out_file)


def download_and_check_sha(url, file, target_sha):
    returned_sha = download(url=url, file_name=file)
    if target_sha != returned_sha:
        raise ValueError(f"{target_sha=} != {returned_sha=}")


def download_speaker_metadata():
    url = "https://github.com/swerik-project/riksdagen-persons/releases/download/v1.2.2/persons.sqlite"
    target_sha = (
        "sha256:dfc0db29039715440b53d9820873c8c163fc72b5acdd207917e58bd4869ab5fd"
    )
    file = "persons.sqlite"
    download_and_check_sha(url, file, target_sha)


def download_speech_data():
    url = "https://github.com/swerik-project/riksdagen-records/releases/download/v1.6.0/records_speeches.ndjson.gz"
    target_sha = (
        "sha256:a1c1310971214d4f836b72f48503b8d86bd3ec80763738e884a02e45a8f49ebf"
    )
    file = "records_speeches.ndjson.gz"
    download_and_check_sha(url, file, target_sha)


def download_corpus_and_metadata():
    download_speaker_metadata()
    download_speech_data()


if __name__ == "__main__":
    download_corpus_and_metadata()
