#!/bin/sh
# Run in a normal macOS terminal; no sudo required.
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
task_source="$task_root/.local/gh/gh_2.103.0_macOS_arm64/bin/gh"
task_dest="$HOME/.local/bin/gh"
task_config="$HOME/.config/slai-terminal"
task_stamp=$(date +%Y%m%d-%H%M%S)
if [ ! -x "$task_source" ]; then
    echo 'Missing downloaded GitHub CLI: install the official CLI first.' >&2
    exit 1
fi
mkdir -p "$HOME/.local/bin" "$task_config"
if [ -e "$task_dest" ] && ! cmp -s "$task_source" "$task_dest"; then
    cp -p "$task_dest" "$task_dest.backup-$task_stamp"
fi
cp "$task_source" "$task_dest"
chmod 755 "$task_dest"
if [ -f "$task_config/terminal.zsh" ]; then
    cp -p "$task_config/terminal.zsh" "$task_config/terminal.zsh.backup-$task_stamp"
fi
cp "$task_root/setup/terminal-proxy.zsh" "$task_config/terminal.zsh"
python3 - "$HOME/.zshrc" "$task_stamp" <<'PY'
from pathlib import Path
import shutil,sys
p=Path(sys.argv[1])
text=p.read_text() if p.exists() else ''
marker='# slai-terminal persistent CLI and proxy'
if marker not in text:
    if p.exists():
        shutil.copy2(p,p.with_name(p.name+'.backup-'+sys.argv[2]))
    with p.open('a') as f:
        f.write('\n'+marker+'\nsource "$HOME/.config/slai-terminal/terminal.zsh"\n')
PY
echo 'GitHub CLI and zsh proxy configuration installed. Existing files backed up.'
if [ "${1:-}" = '--login' ]; then
    # Use the standard gh configuration and OS credential store.
    unset GH_CONFIG_DIR
    export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:25662}"
    export HTTPS_PROXY="$HTTP_PROXY" ALL_PROXY="$HTTP_PROXY"
    "$task_dest" auth login --hostname github.com --git-protocol https --web
    "$task_dest" auth status
fi
echo 'Run: source ~/.config/slai-terminal/terminal.zsh'
echo 'Available commands: gh, proxy_on, proxy_off, proxy_status'
