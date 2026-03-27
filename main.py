#!/usr/bin/env python3
"""
MAPS Backend - Diagnostic version to debug model loading
"""

import os
import sys
import pickle
import traceback
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

DIAGNOSTIC_HTML = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MAPS Diagnostic</title>
    <style>
        body { font-family: monospace; background: #f0f0f0; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        h1 { color: #333; }
        .section { margin: 20px 0; padding: 15px; background: #f9f9f9; border-left: 4px solid #667eea; }
        .success { border-left-color: #4CAF50; color: #2e7d32; }
        .error { border-left-color: #f44336; color: #c62828; }
        .info { border-left-color: #2196F3; color: #1565c0; }
        pre { background: #f5f5f5; padding: 10px; overflow-x: auto; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>MAPS Diagnostic Report</h1>
        %s
    </div>
</body>
</html>'''

def check_file_exists(filename):
    """Check if a file exists and get its size"""
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        return True, size
    return False, 0

def test_pickle_load(filepath):
    """Try to load a pickle file and report what's inside"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            keys = list(data.keys())
            summary = f"Dictionary with keys: {keys}"
        else:
            summary = f"Type: {type(data).__name__}"
        
        return True, summary, None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {str(e)}"

@app.route('/')
def index():
    """Main diagnostic page"""
    html_content = ""
    
    # Check Python environment
    html_content += f'<div class="section info"><strong>Python Version:</strong> {sys.version}</div>'
    
    # Check working directory
    html_content += f'<div class="section info"><strong>Working Directory:</strong> {os.getcwd()}</div>'
    
    # Check file listing
    html_content += '<div class="section info"><strong>Files in current directory:</strong><pre>'
    for f in os.listdir('.'):
        if f.endswith('.pkl') or f.endswith('.py') or f.endswith('.txt'):
            size = os.path.getsize(f)
            html_content += f'{f:<50} {size:>12} bytes\n'
    html_content += '</pre></div>'
    
    # Check board classifier
    html_content += '<div class="section">'
    exists, size = check_file_exists('board_classifier_excel_corrected.pkl')
    if exists:
        html_content += f'<div class="success">✓ board_classifier_excel_corrected.pkl found ({size:,} bytes)</div>'
        success, summary, error = test_pickle_load('board_classifier_excel_corrected.pkl')
        if success:
            html_content += f'<div class="success">✓ Pickle loads successfully: {summary}</div>'
        else:
            html_content += f'<div class="error">✗ Pickle load error: {error}</div>'
    else:
        html_content += '<div class="error">✗ board_classifier_excel_corrected.pkl NOT FOUND</div>'
    html_content += '</div>'
    
    # Check tray classifier
    html_content += '<div class="section">'
    exists, size = check_file_exists('tray_classifier.pkl')
    if exists:
        html_content += f'<div class="success">✓ tray_classifier.pkl found ({size:,} bytes)</div>'
        success, summary, error = test_pickle_load('tray_classifier.pkl')
        if success:
            html_content += f'<div class="success">✓ Pickle loads successfully: {summary}</div>'
        else:
            html_content += f'<div class="error">✗ Pickle load error: {error}</div>'
    else:
        html_content += '<div class="error">✗ tray_classifier.pkl NOT FOUND</div>'
    html_content += '</div>'
    
    # Check required libraries
    html_content += '<div class="section"><strong>Checking Python libraries:</strong><pre>'
    libs = ['pickle', 'flask', 'numpy', 'pandas', 'sklearn', 'PIL']
    for lib in libs:
        try:
            __import__(lib)
            html_content += f'{lib:<20} ✓ available\n'
        except ImportError:
            html_content += f'{lib:<20} ✗ NOT AVAILABLE\n'
    html_content += '</pre></div>'
    
    return render_template_string(DIAGNOSTIC_HTML, html_content)

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 8080))
    print("Starting MAPS diagnostic server on port", PORT)
    app.run(host='0.0.0.0', port=PORT, debug=False)
