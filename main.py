#!/usr/bin/env python3
"""
MAPS Cloud Backend v2 — Minimal Flask app for board/tray model hosting.
Accepts .pkl model uploads at runtime via /upload-model endpoint.
Deployed on Google Cloud Run.

Version: 2.0
Updated: 2026-03-30
"""
import os
import sys
import pickle
import base64
from flask import Flask, request, jsonify

VERSION = "2.0"

app = Flask(__name__)
models = {'board': None, 'tray': None}


@app.route('/')
def index():
    """Root endpoint — confirms service is running."""
    return jsonify({
        'service': 'MAPS Cloud Backend',
        'version': VERSION,
        'endpoints': ['/health', '/upload-model', '/detect'],
        'status': 'ok'
    })


@app.route('/health')
def health():
    """Health check — shows model load status and environment info."""
    return jsonify({
        'status': 'ok',
        'version': VERSION,
        'board_model_loaded': models['board'] is not None,
        'tray_model_loaded': models['tray'] is not None,
        'python_version': sys.version,
        'message': 'Upload .pkl files to /upload-model endpoint'
    })


@app.route('/upload-model', methods=['POST'])
def upload_model():
    """Upload .pkl file as base64-encoded JSON payload.

    JSON body:
        model_type: 'board' or 'tray'
        data: base64-encoded pickle bytes
    """
    try:
        data = request.json
        if data is None:
            return jsonify({'success': False, 'error': 'Request body must be JSON'}), 400

        model_type = data.get('model_type')
        b64_data = data.get('data')

        if not model_type or not b64_data:
            return jsonify({'success': False, 'error': 'Missing model_type or data'}), 400

        if model_type not in ('board', 'tray'):
            return jsonify({'success': False, 'error': f'Unknown model_type: {model_type}. Use "board" or "tray".'}), 400

        pkl_bytes = base64.b64decode(b64_data)
        model_obj = pickle.loads(pkl_bytes)

        models[model_type] = model_obj

        # Report what was loaded
        model_info = str(type(model_obj).__name__)
        return jsonify({
            'success': True,
            'message': f'{model_type.capitalize()} model loaded',
            'model_class': model_info
        })

    except base64.binascii.Error as e:
        return jsonify({'success': False, 'error': f'Base64 decode failed: {e}'}), 400
    except pickle.UnpicklingError as e:
        return jsonify({'success': False, 'error': f'Pickle load failed: {e}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/detect', methods=['POST'])
def detect():
    """Detect board/tray from image (once models are loaded).

    Future: accept image upload and return detected board state.
    Currently returns readiness status.
    """
    board_ready = models['board'] is not None
    tray_ready = models['tray'] is not None

    if not board_ready or not tray_ready:
        missing = []
        if not board_ready:
            missing.append('board')
        if not tray_ready:
            missing.append('tray')
        return jsonify({
            'error': f'Models not loaded: {", ".join(missing)}. Upload via /upload-model first.',
            'board_loaded': board_ready,
            'tray_loaded': tray_ready
        }), 503

    return jsonify({
        'message': 'Detection ready — send image to process',
        'board_loaded': True,
        'tray_loaded': True,
        'status': 'ok'
    })


if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 8080))
    print(f"MAPS Cloud Backend v{VERSION} starting on port {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
