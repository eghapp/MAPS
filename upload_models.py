#!/usr/bin/env python3
"""
Upload models and wordlist to the MAPS Cloud Run backend.

Usage (from Google Cloud Shell):
  # Upload both models:
  python upload_models.py --url https://maps-git-268170978962.us-east5.run.app \
      --board board_classifier_v2.pkl --tray tray_classifier_v1.pkl

  # Upload wordlist:
  python upload_models.py --url https://maps-git-268170978962.us-east5.run.app \
      --wordlist NWL23.txt

  # Check health:
  python upload_models.py --url https://maps-git-268170978962.us-east5.run.app --health
"""
import argparse, requests, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--board", default="")
    p.add_argument("--tray", default="")
    p.add_argument("--wordlist", default="")
    p.add_argument("--health", action="store_true")
    a = p.parse_args()
    url = a.url.rstrip("/")

    if a.health:
        r = requests.get(f"{url}/health", timeout=30)
        print(f"Health: {r.json()}")
        return

    if a.board or a.tray:
        files = {}
        if a.board:
            files["board_classifier"] = open(a.board, "rb")
            print(f"Board model: {a.board}")
        if a.tray:
            files["tray_classifier"] = open(a.tray, "rb")
            print(f"Tray model: {a.tray}")
        r = requests.post(f"{url}/upload_models", files=files, timeout=60)
        print(f"Models: {r.json()}")

    if a.wordlist:
        r = requests.post(f"{url}/upload_wordlist",
                          files={"wordlist": open(a.wordlist, "rb")}, timeout=60)
        print(f"Wordlist: {r.json()}")

    r = requests.get(f"{url}/health", timeout=30)
    print(f"Final: {r.json()}")

if __name__ == "__main__":
    main()
