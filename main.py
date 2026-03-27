#!/usr/bin/env python3
import os
import pickle
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)
models = {'board': None, 'tray': None}

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'board_model_loaded': models['board'] is not None,
        'tray_model_loaded': models['tray'] is not None,
        'message': 'Upload .pkl files to /upload-model endpoint'
    })

@app.route('/upload-model', methods=['POST'])
def upload_model():
    """Upload .pkl file as base64"""
    try:
        data = request.json
        model_type = data.get('model_type')  # 'board' or 'tray'
        b64_data = data.get('data')  # base64 encoded pickle file
        
        if not model_type or not b64_data:
            return jsonify({'success': False, 'error': 'Missing model_type or data'}), 400
        
        pkl_bytes = base64.b64decode(b64_data)
        model_dict = pickle.loads(pkl_bytes)
        
        if model_type == 'board':
            models['board'] = model_dict
            return jsonify({'success': True, 'message': 'Board model loaded'})
        elif model_type == 'tray':
            models['tray'] = model_dict
            return jsonify({'success': True, 'message': 'Tray model loaded'})
        else:
            return jsonify({'success': False, 'error': 'Unknown model_type'}), 400
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect():
    """Detect board/tray from image (once models are loaded)"""
    if models['board'] is None or models['tray'] is None:
        return jsonify({'error': 'Models not loaded. Upload via /upload-model first.'}), 503
    
    return jsonify({'message': 'Detection ready', 'status': 'ok'})

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=PORT, debug=False)
