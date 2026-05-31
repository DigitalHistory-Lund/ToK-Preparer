from .settings import tmp_db, tmp_db1, tmp_db2
import sqlite3
import shutil
def make_chamber_file(number: int, force=False):
    assert number in {1, 2}
    target_file = [tmp_db1, tmp_db2][number - 1]
    assert not force and not target_file.exists(), (
        f"{target_file=}\t{target_file.exists()}"
    )

    shutil.copyfile(tmp_db, target_file)

    with sqlite3.connect(tmp_db1) as conn:
        conn.execute(f"DELETE FROM utterance where kammare != {number}")
        conn.commit()


def make_chamber_1():
    make_chamber_file(1)


def make_chamber_2():
    make_chamber_file(2)


if __name__ == "__main__":
    assert not (tmp_db1.exists() or tmp_db2.exists()), (
        f"\n{tmp_db1=}\t{tmp_db1.exists()}\n{tmp_db2=}\t{tmp_db2.exists()}"
    )

    make_chamber_1()
    make_chamber_2()
