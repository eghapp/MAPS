#!/usr/bin/env python3
"""
MAPS Backend - Ultra simple diagnostic (text only)
"""

import os
import sys
import pickle

app_output = []
app_output.append("MAPS DIAGNOSTIC - TEXT MODE\n")
app_output.append("=" * 70 + "\n")

# Python info
app_output.append(f"Python: {sys.version}\n")
app_output.append(f"CWD: {os.getcwd()}\n\n")

# File listing
app_output.append("FILES IN DIRECTORY:\n")
for f in sorted(os.listdir('.')):
    if f.endswith(('.pkl', '.py', '.txt')):
        size = os.path.getsize(f)
        app_output.append(f"  {f:<45} {size:>12,} bytes\n")

app_output.append("\nBOARD CLASSIFIER:\n")
if os.path.exists('board_classifier_excel_corrected.pkl'):
    size = os.path.getsize('board_classifier_excel_corrected.pkl')
    app_output.append(f"  ✓ Found: {size:,} bytes\n")
    try:
        with open('board_classifier_excel_corrected.pkl', 'rb') as f:
            data = pickle.load(f)
        app_output.append(f"  ✓ Loaded: {type(data).__name__}\n")
        if isinstance(data, dict):
            app_output.append(f"  ✓ Keys: {list(data.keys())}\n")
    except Exception as e:
        app_output.append(f"  ✗ Error: {str(e)}\n")
else:
    app_output.append("  ✗ NOT FOUND\n")

app_output.append("\nTRAY CLASSIFIER:\n")
if os.path.exists('tray_classifier.pkl'):
    size = os.path.getsize('tray_classifier.pkl')
    app_output.append(f"  ✓ Found: {size:,} bytes\n")
    try:
        with open('tray_classifier.pkl', 'rb') as f:
            data = pickle.load(f)
        app_output.append(f"  ✓ Loaded: {type(data).__name__}\n")
        if isinstance(data, dict):
            app_output.append(f"  ✓ Keys: {list(data.keys())}\n")
    except Exception as e:
        app_output.append(f"  ✗ Error: {str(e)}\n")
else:
    app_output.append("  ✗ NOT FOUND\n")

app_output.append("\n" + "=" * 70 + "\n")
diagnostic_text = "".join(app_output)

# Print to stdout so Cloud Run logs it
print(diagnostic_text)

# Now create Flask app
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return f"<pre>{diagnostic_text}</pre>", 200

@app.route('/health')
def health():
    return "OK", 200

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=PORT, debug=False)
