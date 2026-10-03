"""Fetch only the three explicitly selected, editable Poly Haven scene files."""
from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = next((p for p in PROJECT.parents if (p / 'runtime/bpy_runtime').is_dir()), PROJECT.parents[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', nargs='+', default=['blue_wall', 'pawn_shop', 'the_shed'])
    parser.add_argument('--archives', type=Path,
                        help='Optional reusable archive directory; default downloads are temporary.')
    args = parser.parse_args()
    archives = args.archives or WORKSPACE / 'codex_tmp/rgba_pipeline/scene_downloads'
    sources = json.loads((PROJECT / 'assets/scene_sources.json').read_text())
    for source in sources:
        if source['id'] not in args.scenes:
            continue
        output = PROJECT / 'assets/scenes' / source['id']
        marker = output / 'source.json'
        if marker.exists():
            print('READY', source['id'], flush=True)
            continue
        archives.mkdir(parents=True, exist_ok=True)
        archive = archives / (source['id'] + '.zip')
        # A valid central directory means a completed archive; curl resumes interrupted transfers.
        if not archive.exists() or not zipfile.is_zipfile(archive):
            subprocess.run(['curl', '-L', '--fail', '--silent', '--show-error', '--retry', '2',
                            '--continue-at', '-', source['url'], '-o', str(archive)], check=True)
        output.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                destination = (output / entry.filename).resolve()
                if not destination.is_relative_to(output.resolve()):
                    raise ValueError(f'archive member outside scene directory: {entry.filename}')
            bundle.extractall(output)
        blends = sorted(str(p.relative_to(PROJECT)) for p in output.rglob('*.blend'))
        marker.write_text(json.dumps({**source, 'blend_files': blends}, indent=2) + '\n')
        if args.archives is None:
            archive.unlink()
        print('SUCCESS', source['id'], blends, flush=True)


if __name__ == '__main__':
    main()
