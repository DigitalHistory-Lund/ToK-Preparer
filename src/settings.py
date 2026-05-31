from pathlib import Path


root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
ord_dir = root / "ord"

tmp_db = root / "tmp_db.sqlite3"
tmp_db1 = root / "tmp_db.1.sqlite3"
tmp_db2 = root / "tmp_db.2.sqlite3"
out_db = root / "ToK_data.sqlite3"

data_dir.mkdir(exist_ok=True)
data_here = Path(__file__).parent.resolve() / "data"

# preparation config
EXPECTED_COUNT = 697_343
EXPECTED_MERGED_COUNT = 173_713
BATCH_SIZE = 5_000
