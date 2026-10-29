"""Repeatable migration: never replace existing destination files or newer epochs."""
from pathlib import Path
import hashlib
import os
import shutil


def import_missing(source, destination):
    source, destination = Path(source), Path(destination)
    # Check the immutable protocol before copying any missing file.
    protocol = destination / 'protocol.json'
    if protocol.exists() and protocol.read_bytes() != (source/'protocol.json').read_bytes():
        raise RuntimeError('Existing run protocol differs; nothing imported')
    copied, preserved = [], []
    for f in sorted(source.rglob('*')):
        if not f.is_file():
            continue
        rel = f.relative_to(source)
        dest = destination / rel
        if dest.exists():
            if not dest.is_file():
                raise RuntimeError(f'Expected file: {dest}')
            preserved.append(str(rel))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation never overwrites a file created concurrently.
        try:
            with dest.open('xb') as out, f.open('rb') as inp:
                shutil.copyfileobj(inp, out)
                out.flush()
                os.fsync(out.fileno())
        except FileExistsError:
            preserved.append(str(rel))
            continue
        if hashlib.sha256(dest.read_bytes()).digest() != hashlib.sha256(f.read_bytes()).digest():
            raise RuntimeError(f'Imported file hash mismatch: {dest}')
        copied.append(str(rel))
    return {'copied': copied, 'preserved': preserved}


def pending_epochs(remote_rows, archive_rows, *, completed):
    """Never republish an old packet over a resumed run; retain remote top two."""
    if completed:
        return []
    current = max((int(r['appProperties']['clv_epoch']) for r in remote_rows), default=0)
    for row in archive_rows:
        p = row['appProperties']
        for live in remote_rows:
            q = live['appProperties']
            if int(p['clv_epoch']) == int(q['clv_epoch']) and p['clv_sha256'] != q['clv_sha256']:
                raise RuntimeError('Same epoch has different checkpoint bytes; originals preserved')
    return sorted([r for r in archive_rows if int(r['appProperties']['clv_epoch']) > current],
                  key=lambda r: int(r['appProperties']['clv_epoch']))
