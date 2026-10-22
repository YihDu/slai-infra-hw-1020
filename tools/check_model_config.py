#!/usr/bin/env python3
"""Validate configuration offline; --ping explicitly makes a text-generation call."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizer.model import ModelAPIError, ModelConfig, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--ping', action='store_true')
    args = parser.parse_args()
    try:
        config = ModelConfig.load(args.config)
        result = dict(protocol=config.protocol, model=config.model, endpoint=config.endpoint(),
                      credential_present=bool(os.environ.get(config.api_key_env)), network_called=False)
        if args.ping:
            result.update(network_called=True, **generate(config, 'Reply with OK.', max_output_tokens=1024))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ModelAPIError as exc:
        record = exc.record()
    except (ValueError, OSError) as exc:
        record = dict(error_type=type(exc).__name__, reason=str(exc) if isinstance(exc, ValueError) else 'Cannot read configuration file')
    print(json.dumps(dict(status='failed', **record), ensure_ascii=False))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
