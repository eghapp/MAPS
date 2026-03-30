#!/usr/bin/env python3
"""
MAPS Model Uploader v2
Uploads .pkl model files to a running MAPS Cloud Backend instance.

Usage (from Google Cloud Shell):
    python upload_models.py <backend_url>

Example:
    python upload_models.py https://maps-git-268170978962.us-east5.run.app
"""
import sys
import os
import base64
import requests

if len(sys.argv) < 2:
    print("Usage: python upload_models.py <backend_url>")
    print("Example: python upload_models.py https://maps-git-268170978962.us-east5.run.app")
    sys.exit(1)

backend_url = sys.argv[1].rstrip('/')

# --- Step 1: Check health before uploading ---
print(f"Checking backend at {backend_url} ...")
try:
    resp = requests.get(f'{backend_url}/health', timeout=10)
    resp.raise_for_status()
    status = resp.json()
    print(f"  Service: OK (v{status.get('version', '?')})")
    print(f"  Board model loaded: {status.get('board_model_loaded')}")
    print(f"  Tray model loaded:  {status.get('tray_model_loaded')}")
except Exception as e:
    print(f"  ERROR: Cannot reach backend — {e}")
    print("  Is the Cloud Run service deployed and running?")
    sys.exit(1)


# --- Step 2: Upload models ---
def upload(model_type, filepath):
    """Upload a single .pkl model file."""
    if not os.path.exists(filepath):
        print(f"\n  SKIP {model_type}: file not found at {filepath}")
        return False

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"\nUploading {model_type} ({size_mb:.1f} MB) from {filepath} ...")

    try:
        with open(filepath, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')

        resp = requests.post(
            f'{backend_url}/upload-model',
            json={'model_type': model_type, 'data': b64},
            timeout=60
        )
        result = resp.json()

        if result.get('success'):
            print(f"  OK: {result['message']} ({result.get('model_class', '?')})")
            return True
        else:
            print(f"  FAIL: {result.get('error', 'Unknown error')}")
            return False
    except requests.exceptions.Timeout:
        print(f"  FAIL: Request timed out (file may be too large)")
        return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


success = True
success &= upload('board', 'board_classifier_excel_corrected.pkl')
success &= upload('tray', 'tray_classifier.pkl')


# --- Step 3: Verify final status ---
print("\n--- Final Status ---")
try:
    resp = requests.get(f'{backend_url}/health', timeout=10)
    status = resp.json()
    print(f"  Board model loaded: {status.get('board_model_loaded')}")
    print(f"  Tray model loaded:  {status.get('tray_model_loaded')}")

    if status.get('board_model_loaded') and status.get('tray_model_loaded'):
        print("\n  ALL MODELS LOADED — backend is ready for /detect")
    else:
        print("\n  WARNING: Not all models loaded. Check errors above.")
except Exception as e:
    print(f"  Could not verify: {e}")

sys.exit(0 if success else 1)
