from pathlib import Path


root = Path(__file__).resolve().parents[1]
data_dir = root / "data"

tmp_db = root / "tmp_db.sqlite3"
out_db = root / "ToK_data.sqlite3"

word_freq = root / "word_frequencies.csv.gz"
word_freq1 = root / "word_frequencies.1.csv.gz"
word_freq2 = root / "word_frequencies.2.csv.gz"

speakers = root / "speakers.csv.gz"
speakers1 = root / "speakers.1.csv.gz"
speakers2 = root / "speakers.2.csv.gz"

speaker_totals = root / "speaker_totals.csv.gz"
speaker_word_totals = root / "speaker_word_totals.csv.gz"

data_dir.mkdir(exist_ok=True)
persons_db = data_dir / "persons.sqlite"
data_here = Path(__file__).parent.resolve() / "data"
manual_party_overrides_csv = Path(__file__).parent.resolve() / "manual_party_overrides.csv"

# preparation config
START_YEAR = 1900
END_YEAR = 1940
EXPECTED_COUNT = 697_343
EXPECTED_MERGED_COUNT = 173_713
BATCH_SIZE = 5_000
