"""
Nimbus interactive CLI chat.

Usage:
    python chat.py                  # default (1.5B)
    python chat.py --variant 7b
    python chat.py --variant 1.5b --4bit   # quantized, fits in 6GB VRAM
"""

import argparse
import sys

from nimbus import Nimbus, NimbusConfig

BANNER = """
  _   _ _           _
 | \\ | (_)_ __ ___ | |__  _   _ ___
 |  \\| | | '_ ` _ \\| '_ \\| | | / __|
 | |\\  | | | | | | | |_) | |_| \\__ \\
 |_| \\_|_|_| |_| |_|_.__/ \\__,_|___/

 by Thundered Studios
"""

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, default="4b",
                   choices=["0.6b", "1.7b", "4b", "8b", "14b", "32b", "30b", "235b"])
    p.add_argument("--4bit",  dest="load_4bit", action="store_true",
                   help="Load in 4-bit quantization (less VRAM)")
    p.add_argument("--8bit",  dest="load_8bit", action="store_true",
                   help="Load in 8-bit quantization")
    p.add_argument("--local", type=str, default=None,
                   help="Load from a local directory")
    p.add_argument("--system", type=str, default=None,
                   help="Override system prompt")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.6)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = NimbusConfig(
        load_in_4bit=args.load_4bit,
        load_in_8bit=args.load_8bit,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    if args.system:
        cfg.system_prompt = args.system

    print(BANNER)
    nimbus = Nimbus.load(variant=args.variant, config=cfg, local_path=args.local)
    print(f"\n{nimbus}")
    print("\nType your message. Commands: /clear, /exit\n")
    print("-" * 50)

    history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print("Bye.")
            break

        if user_input.lower() == "/clear":
            history.clear()
            print("Conversation cleared.")
            continue

        print("\nNimbus: ", end="", flush=True)

        response_chunks = []
        for chunk in nimbus.stream(user_input, history=history):
            print(chunk, end="", flush=True)
            response_chunks.append(chunk)

        response = "".join(response_chunks)
        print()

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant",  "content": response})


if __name__ == "__main__":
    main()
