from settings import tmp_db, out_db
import sqlite3


if out_db.exists():
    raise FileExistsError(f"{out_db} already exists. Remove it before proceeding.")
elif not tmp_db.exists():
    raise FileNotFoundError(f"{tmp_db} does not exist. Run prepare_db.py first.")


if __name__ == "__main__":
    with sqlite3.connect(out_db) as out_conn:
        with sqlite3.connect(tmp_db) as conn:
            conn.backup(out_conn)

        out_conn.execute("drop table affiliation")
        out_conn.execute("drop table utterance_fts")
        out_conn.execute("drop table reverse_utterance_fts ")
        out_conn.execute("vacuum")
        out_conn.commit()

    print(f"Reduced database {tmp_db.name} to {out_db.name}.")
    print(f"size before: {tmp_db.stat().st_size / 1e6:.2f} MB")
    print(f"size after:  {out_db.stat().st_size / 1e6:.2f} MB")
