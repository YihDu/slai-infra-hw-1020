#!/usr/bin/env python3
"""One-shot text generation using the same adapter as run.sh and the TA config."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizer.model import ModelAPIError, ModelConfig, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-config', required=True, type=Path)
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument('--prompt', default=None)
    prompts.add_argument('--prompt-file', type=Path)
    parser.add_argument('--max-output-tokens', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    try:
        config = ModelConfig.load(args.model_config)
        prompt = args.prompt_file.read_text(encoding='utf-8') if args.prompt_file else (args.prompt or 'Reply with OK.')
        response = generate(config, prompt, timeout=args.timeout, max_output_tokens=args.max_output_tokens)
        print(json.dumps(dict(protocol=config.protocol, model=config.model, **response), ensure_ascii=False, indent=2))
        return 0
    except ModelAPIError as exc:
        record = exc.record()
    except (ValueError, OSError) as exc:
        # Configuration errors contain no credential values. Avoid reflecting arbitrary paths.
        record = dict(error_type=type(exc).__name__, reason=str(exc) if isinstance(exc, ValueError) else 'Cannot read configuration or prompt file')
    print(json.dumps(dict(status='error', **record), ensure_ascii=False, indent=2), file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
