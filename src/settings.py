from pathlib import Path


root = Path(__file__).resolve().parents[1]
data_dir = root / "data"

tmp_db = root / "tmp_db.sqlite3"
out_db = root / "ToK_data.sqlite3"

data_dir.mkdir(exist_ok=True)
data_here = Path(__file__).parent.resolve() / "data"
