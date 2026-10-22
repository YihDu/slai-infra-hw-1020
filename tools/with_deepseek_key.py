#!/usr/bin/env python3
"""Load only the explicitly selected DeepSeek section; never execute the file."""
import argparse
import os
import re
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--credential-file', required=True, type=Path)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command
if command and command[0] == '--':
    command = command[1:]
if not command:
    parser.error('A command is required after --')
content = args.credential_file.read_text(encoding='utf-8')
match = re.search(r'(?m)^#\s*DeepSeek-yihang\s*$', content)
if not match:
    parser.error('DeepSeek section missing')
section = content[match.end():].split('\n#', 1)[0]
key = re.search(r'(?m)^\s*key\s*=\s*[\"\']?([^\s\"\']+)', section)
if not key:
    parser.error('DeepSeek key missing')
env = dict(os.environ, DEEPSEEK_API_KEY=key.group(1))
raise SystemExit(subprocess.run(command, env=env).returncode)
