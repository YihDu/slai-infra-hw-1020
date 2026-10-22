#!/usr/bin/env python3
"""Fetch and verify the prepared macOS arm64 / CPython 3.13 wheel set."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--direct', action='store_true', help='Bypass proxy environment variables')
parser.add_argument('--resolve', help='Optional files.pythonhosted.org IP override')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
dest = root / 'wheelhouse'
dest.mkdir(exist_ok=True)
for item in json.loads((root / 'setup/wheels-macos-arm64.json').read_text()):
    path = dest / item['filename']
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        partial = path.with_suffix('.partial')
        cmd = ['curl', '-L', '--fail', '--retry', '2', '--connect-timeout', '10', '--max-time', '120']
        if args.direct:
            cmd += ['--noproxy', '*']
        if args.resolve:
            cmd += ['--resolve', 'files.pythonhosted.org:443:' + args.resolve]
        subprocess.run(cmd + [item['url'], '-o', str(partial)], check=True)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('SHA-256 mismatch: ' + item['filename'])
        partial.replace(path)
    print(item['name'] + ' verified')
