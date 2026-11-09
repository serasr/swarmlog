#!/usr/bin/env python3
"""One cheap call to confirm the API path works before spending on 10 trials.

    python check_api.py
    python check_api.py --model claude-sonnet-4-5
"""

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="claude-sonnet-4-5")
    args = p.parse_args()

    try:
        import anthropic
    except ImportError:
        print("FAIL: anthropic SDK not installed -> pip install anthropic")
        return 1

    ver = getattr(anthropic, "__version__", "unknown")
    print(f"anthropic SDK  : {ver}")
    print(f"python         : {sys.version.split()[0]}")
    print(f"key present    : {bool(os.environ.get('ANTHROPIC_API_KEY'))}")
    print(f"base url       : {os.environ.get('ANTHROPIC_BASE_URL', 'default')}")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nFAIL: ANTHROPIC_API_KEY not visible to this process.")
        print("  Run & Debug config reads .env; a plain terminal run does not.")
        print('  Terminal:  $env:ANTHROPIC_API_KEY="sk-ant-..."')
        return 1

    import inspect
    try:
        sig = inspect.signature(anthropic.Anthropic().messages.create)
        print(f"accepts temp   : {'temperature' in sig.parameters}")
    except Exception:
        pass

    print(f"\ncalling {args.model} ...")
    try:
        client = anthropic.Anthropic()
        r = client.messages.create(
            model=args.model,
            max_tokens=20,
            messages=[{"role": "user", "content": "Reply with the word: ok"}],
        )
        text = "".join(getattr(b, "text", "") for b in r.content)
        print(f"OK: {text.strip()[:40]}")
        print(f"tokens: in={r.usage.input_tokens} out={r.usage.output_tokens}")
        print("\nAPI path works. Safe to run the live experiment.")
        return 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {str(exc)[:300]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())