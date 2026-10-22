#!/usr/bin/env python3
"""Validate configuration locally; --ping explicitly makes one model call."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizer.model import ModelConfig, generate

parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--ping', action='store_true')
args = parser.parse_args()
try:
    config = ModelConfig.load(args.config)
    result = dict(protocol=config.protocol, model=config.model,
                  credential_present=bool(os.environ.get(config.api_key_env)), network_called=False)
    if args.ping:
        answer = generate(config, 'Reply with OK.', max_output_tokens=64)
        result.update(network_called=True, **answer)
    print(json.dumps(result, ensure_ascii=False, indent=2))
except Exception as exc:
    # Never print HTTP response bodies or authorization headers.
    print(json.dumps({'error_type': type(exc).__name__, 'status': 'failed'}, ensure_ascii=False))
    sys.exit(1)
