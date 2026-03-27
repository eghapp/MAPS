#!/usr/bin/env python3
"""
Simple script to upload .pkl files to running MAPS instance
Usage: python upload_models.py <backend_url>
"""
import sys
import base64
import requests

if len(sys.argv) < 2:
    print("Usage: python upload_models.py <backend_url>")
    print("Example: python upload_models.py https://maps-git-268170978962.us-east5.run.app")
    sys.exit(1)

backend_url = sys.argv[1].rstrip('/')

def upload(model_type, filepath):
    print(f"\nUploading {model_type}...")
    try:
        with open(filepath, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        
        resp = requests.post(
            f'{backend_url}/upload-model',
            json={'model_type': model_type, 'data': b64},
            timeout=30
        )
        result = resp.json()
        
        if result.get('success'):
            print(f"  ✓ {result['message']}")
            return True
        else:
            print(f"  ✗ Error: {result['error']}")
            return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

# Upload both models
success = True
success &= upload('board', 'board_classifier_excel_corrected.pkl')
success &= upload('tray', 'tray_classifier.pkl')

# Check status
try:
    resp = requests.get(f'{backend_url}/health', timeout=5)
    status = resp.json()
    print(f"\nStatus: {status}")
except Exception as e:
    print(f"\nFailed to check status: {e}")

sys.exit(0 if success else 1)
