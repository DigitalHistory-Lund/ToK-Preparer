from .settings import ord_dir, tmp_db
from pathlib import Path
import sqlite3
from collections import Counter
import re

def get_found_words(pattern):
    out_file = ord_dir + pattern +'.csv'
    out_file = Path(out_file.replace(' ', '_'))
    out_file.parent.mkdir(exist_ok = True, parents=True)


    if not out_file.exists():

        with sqlite3.connect(tmp_db) as conn:
            cur = conn.cursor()

            if not pattern.startswith('*'):
                term_counter = Counter((re.sub('[^a-zåäö ]', '', token.lower()) for content in cur.execute(f'select content from utterance_fts where content match "{pattern}"')
                    for token in content[0].split() if token.lower().startswith(pattern.replace('*', ''))))
            else:
                reverse_pattern = pattern[::-1]
                term_counter = Counter((re.sub('[^a-zåäö ]', '', token.lower()[::-1]) for content in cur.execute(f'select content from reverse_utterance_fts where content match "{reverse_pattern}"')
                    for token in content[0].split() if token.lower().startswith(reverse_pattern.replace('*', ''))))

        with open(out_file, 'x', encoding='utf8') as f:
            f.write('word , count\n')
            for term, count in term_counter.most_common():
                f.write(str(term) + ' , ' + str(count) + '\n')
