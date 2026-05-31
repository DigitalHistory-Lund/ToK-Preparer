from .settings import tmp_db, tmp_db1, tmp_db2
import sqlite3
import shutil



def make_chamber_1():
    assert not tmp_db1.exists(), f"{tmp_db1=}\t{tmp_db1.exists()}"

    with sqlite3.connect(tmp_db1) as conn:
        conn.execute('DELETE FROM utterance where kammare != 1')
        conn.commit()

def make_chamber_2():
    assert not tmp_db2.exists(), f"{tmp_db2=}\t{tmp_db2.exists()}"

    with sqlite3.connect(tmp_db2) as conn:
        conn.execute('DELETE FROM utterance where kammare != 2')
        conn.commit()

if __name__ == '__main__':
    assert not (tmp_db1.exists() or tmp_db2.exists()), f"\n{tmp_db1=}\t{tmp_db1.exists()}\n{tmp_db2=}\t{tmp_db2.exists()}"

    shutil.copyfile(tmp_db, tmp_db1)
    shutil.copyfile(tmp_db, tmp_db2)

    make_chamber_1()
    make_chamber_2()
