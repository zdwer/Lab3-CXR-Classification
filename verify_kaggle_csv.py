
import argparse, csv, hashlib
from pathlib import Path
FIELDS = ['new_filename','normal','bacteria','virus','COVID-19']
def read(path):
    with Path(path).open('r',newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError(f'Expected {FIELDS}, got {reader.fieldnames}')
        rows=list(reader)
    if len(rows)!=1182:
        raise ValueError(f'Expected 1182 test rows, got {len(rows)}')
    names=[r['new_filename'] for r in rows]
    if len(set(names))!=1182 or any(not n for n in names):
        raise ValueError('Duplicate or empty filenames')
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--submission',required=True)
    p.add_argument('--sample',required=True)
    args=p.parse_args()
    source=read(args.submission)
    sample=read(args.sample)
    if [r['new_filename'] for r in source] != [r['new_filename'] for r in sample]:
        raise ValueError('Filename order does not match original Kaggle sample')
    for ix,row in enumerate(source, start=2):
        values=[row[k].strip() for k in FIELDS[1:]]
        if any(x not in ('0','1') for x in values) or sum(int(x) for x in values)!=1:
            raise ValueError(f'Non-one-hot row {ix}: {values}')
    digest=hashlib.sha256(Path(args.submission).read_bytes()).hexdigest()
    print('PASS: 1182 rows; five exact columns; sample filename order; all one-hot.')
    print('Submission file:',args.submission)
    print('SHA256:',digest)
if __name__=='__main__': main()
