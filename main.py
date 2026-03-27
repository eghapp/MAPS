#!/usr/bin/env python3
"""
MAPS Backend Server for Google Cloud Run
ML-based Scrabble board detection and move generation
"""

import os
import io
import base64
import json
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import pickle
import pandas as pd

# ============================================================================
# SETUP
# ============================================================================

app = Flask(__name__)
CORS(app)

PORT = int(os.environ.get('PORT', 8080))

# Global model storage
board_classifier = None
board_scaler = None
tray_classifier = None
tray_scaler = None

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load ML models from Cloud Storage or local files"""
    global board_classifier, board_scaler, tray_classifier, tray_scaler
    
    try:
        # Try to load from local files first (for local testing)
        if os.path.exists('board_classifier_excel_corrected.pkl'):
            with open('board_classifier_excel_corrected.pkl', 'rb') as f:
                data = pickle.load(f)
                board_classifier = data.get('classifier')
                board_scaler = data.get('scaler')
            print("✓ Loaded board classifier from local file")
        
        if os.path.exists('tray_classifier.pkl'):
            with open('tray_classifier.pkl', 'rb') as f:
                data = pickle.load(f)
                tray_classifier = data.get('classifier')
                tray_scaler = data.get('scaler')
            print("✓ Loaded tray classifier from local file")
    
    except Exception as e:
        print(f"⚠ Warning loading models: {e}")

# Load models on startup
load_models()

# ============================================================================
# ML INFERENCE FUNCTIONS
# ============================================================================

def extract_cell_features(cell_image):
    """Extract features from a cell image"""
    cell_array = np.array(cell_image)
    
    # Handle different image modes
    if len(cell_array.shape) == 2:
        cell_array = np.stack([cell_array]*3, axis=2)
    elif cell_array.shape[2] == 4:  # RGBA
        cell_array = cell_array[:,:,:3]
    
    features = {}
    
    # Convert to grayscale
    r = cell_array[:,:,0].astype(float)
    g = cell_array[:,:,1].astype(float)
    b = cell_array[:,:,2].astype(float)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Core features
    features['gray_mean'] = float(gray.mean())
    features['gray_std'] = float(gray.std())
    features['gray_min'] = float(gray.min())
    features['gray_max'] = float(gray.max())
    features['dark_pixels'] = float((gray < 128).mean() * 100)
    features['very_dark_pixels'] = float((gray < 64).mean() * 100)
    features['r_mean'] = float(r.mean())
    features['g_mean'] = float(g.mean())
    features['b_mean'] = float(b.mean())
    features['contrast'] = float(gray.max() - gray.min())
    
    # Edge detection
    if gray.shape[0] > 2 and gray.shape[1] > 2:
        h_edges = np.abs(np.diff(gray, axis=0))
        v_edges = np.abs(np.diff(gray, axis=1))
        features['edge_strength'] = float((h_edges.mean() + v_edges.mean()) / 2)
    else:
        features['edge_strength'] = 0.0
    
    # Uniformity
    if gray.shape[0] > 1 and gray.shape[1] > 1:
        h_unif = np.sum(np.abs(np.diff(gray, axis=0)) < 10)
        v_unif = np.sum(np.abs(np.diff(gray, axis=1)) < 10)
        total_edges = (gray.shape[0]-1)*gray.shape[1] + gray.shape[0]*(gray.shape[1]-1)
        features['uniformity'] = float((h_unif + v_unif) / total_edges) if total_edges > 0 else 0.0
    else:
        features['uniformity'] = 0.0
    
    return features

def detect_board_bounds(image_array):
    """Auto-detect board region"""
    h, w = image_array.shape[:2]
    
    # Heuristic: board is typically in middle 75% of image
    board_start_y = int(h * 0.1)
    board_end_y = int(h * 0.85)
    
    return {
        'x_min': int(w * 0.05),
        'y_min': board_start_y,
        'x_max': int(w * 0.95),
        'y_max': board_end_y,
        'width': int(w * 0.9),
        'height': board_end_y - board_start_y
    }

def predict_board(image):
    """Predict board state using ML model"""
    if board_classifier is None or board_scaler is None:
        return None, "Board classifier not available"
    
    try:
        img_array = np.array(image)
        bounds = detect_board_bounds(img_array)
        
        cell_width = bounds['width'] / 15.0
        cell_height = bounds['height'] / 15.0
        
        board = []
        
        for r in range(15):
            row = []
            for c in range(15):
                # Extract cell region
                x_start = int(bounds['x_min'] + c * cell_width)
                x_end = int(bounds['x_min'] + (c + 1) * cell_width)
                y_start = int(bounds['y_min'] + r * cell_height)
                y_end = int(bounds['y_min'] + (r + 1) * cell_height)
                
                cell_img = image.crop((x_start, y_start, x_end, y_end))
                
                # Extract features
                features = extract_cell_features(cell_img)
                features_df = pd.DataFrame([features])
                features_scaled = board_scaler.transform(features_df)
                
                # Predict
                prediction = board_classifier.predict(features_scaled)[0]
                row.append(prediction)
            
            board.append(row)
        
        return board, None
    
    except Exception as e:
        return None, f"Board detection error: {str(e)}"

def predict_tray(image):
    """Predict tray letters using ML model"""
    if tray_classifier is None or tray_scaler is None:
        return None, "Tray classifier not available"
    
    try:
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Extract tray region (bottom 10-12%)
        tray_height = int(h * 0.12)
        tray_img = image.crop((0, h - tray_height, w, h))
        
        # Extract 7 character regions
        char_width = w / 8.0
        rack = ""
        
        for i in range(7):
            x_start = int(i * char_width + char_width * 0.25)
            x_end = int((i + 1) * char_width - char_width * 0.25)
            
            char_img = tray_img.crop((max(0, x_start), 0, min(w, x_end), tray_height))
            
            # Extract features
            features = extract_cell_features(char_img)
            features_df = pd.DataFrame([features])
            features_scaled = tray_scaler.transform(features_df)
            
            # Predict
            prediction = tray_classifier.predict(features_scaled)[0]
            rack += prediction
        
        return rack, None
    
    except Exception as e:
        return None, f"Tray detection error: {str(e)}"

# ============================================================================
# MOVE GENERATION (PLACEHOLDER)
# ============================================================================

def generate_moves(board, rack):
    """Generate recommended moves (simplified)"""
    # Placeholder - in production would use full MAPS solver
    moves = [
        {'word': 'PLAYED', 'score': 72, 'leave': 'AEIOU?'},
        {'word': 'BOARD', 'score': 68, 'leave': 'AEIOU?'},
        {'word': 'WORDS', 'score': 64, 'leave': 'AEIOU?'},
        {'word': 'SCRABBLE', 'score': 85, 'leave': 'AI?'},
        {'word': 'GAMES', 'score': 58, 'leave': 'AEIOU?'},
        {'word': 'TILES', 'score': 55, 'leave': 'AEIOU?'},
        {'word': 'PLAYS', 'score': 62, 'leave': 'AEIOU?'},
        {'word': 'MOVES', 'score': 71, 'leave': 'AEI?'},
        {'word': 'SCORE', 'score': 59, 'leave': 'AEIOU?'},
        {'word': 'STRATEGY', 'score': 73, 'leave': 'AEI?'},
    ]
    return moves

# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    """Serve web app HTML"""
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MAPS - Scrabble Solver</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; padding: 10px; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.1); overflow: hidden; }
        header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px 20px; text-align: center; }
        header h1 { font-size: 32px; margin-bottom: 8px; }
        header p { opacity: 0.95; font-size: 15px; }
        .main-content { padding: 20px; }
        .upload-section { background: #f9f9f9; border: 2px dashed #667eea; border-radius: 12px; padding: 30px; text-align: center; margin-bottom: 20px; }
        .upload-section.has-image { border-style: solid; background: #f0f4ff; }
        .btn-upload { background: #667eea; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; width: 100%; margin-bottom: 10px; }
        .btn-upload:hover { background: #5568d3; }
        #fileInput { display: none; }
        .image-preview { margin-top: 15px; display: none; text-align: center; }
        .image-preview.active { display: block; }
        .image-preview img { max-width: 300px; max-height: 400px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
        .content-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        @media (max-width: 900px) { .content-grid { grid-template-columns: 1fr; } }
        .section { border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; background: #fafafa; }
        .section h2 { font-size: 16px; margin-bottom: 12px; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 8px; }
        .btn-analyze { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; width: 100%; margin-bottom: 10px; }
        .btn-analyze:hover { background: #45a049; }
        .btn-analyze:disabled { background: #ccc; cursor: not-allowed; }
        .status { padding: 10px; border-radius: 6px; margin-bottom: 10px; font-size: 13px; display: none; border-left: 4px solid; }
        .status.show { display: block; }
        .status.info { background: #e3f2fd; color: #1976d2; border-left-color: #1976d2; }
        .status.success { background: #e8f5e9; color: #388e3c; border-left-color: #388e3c; }
        .status.error { background: #ffebee; color: #c62828; border-left-color: #c62828; }
        .board-preview { display: grid; grid-template-columns: repeat(15, 1fr); gap: 2px; margin-bottom: 15px; background: #ddd; padding: 5px; border-radius: 6px; }
        .cell { aspect-ratio: 1; display: flex; align-items: center; justify-content: center; font-size: 9px; font-weight: bold; border-radius: 2px; background: white; }
        .cell.tile { background: #667eea; color: white; }
        .cell.bonus { background: #ffd700; color: #333; }
        .cell.empty { background: #f0f0f0; }
        .rack-display { background: white; padding: 10px; border-radius: 6px; font-size: 18px; font-weight: bold; text-align: center; letter-spacing: 4px; color: #667eea; margin-bottom: 10px; }
        .moves-list { max-height: 500px; overflow-y: auto; border: 1px solid #ddd; border-radius: 6px; background: white; }
        .move-item { padding: 12px; border-bottom: 1px solid #eee; font-size: 13px; }
        .move-item:last-child { border-bottom: none; }
        .move-rank { display: inline-block; background: #667eea; color: white; width: 26px; height: 26px; line-height: 26px; text-align: center; border-radius: 50%; margin-right: 10px; font-weight: bold; }
        .move-word { font-weight: bold; color: #333; }
        .move-score { color: #4CAF50; font-weight: bold; float: right; }
        .move-details { font-size: 12px; color: #666; margin-top: 4px; }
        .board-stats { font-size: 12px; color: #666; text-align: center; padding: 8px; background: white; border-radius: 4px; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>♠️ MAPS Scrabble Solver</h1>
            <p>Automatic board detection and move recommendation</p>
        </header>
        
        <div class="main-content">
            <div class="upload-section" id="uploadSection">
                <h3 style="margin-bottom: 15px; color: #333;">📸 Upload Your Game Board</h3>
                <button class="btn-upload" onclick="document.getElementById('fileInput').click()">
                    📁 Choose Image
                </button>
                <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
                <div class="image-preview" id="imagePreview">
                    <img id="previewImg" src="" alt="Board preview">
                </div>
            </div>
            
            <div id="analysisSection" style="display: none;">
                <div style="margin-bottom: 10px;">
                    <button class="btn-analyze" onclick="analyzeImage()">🔍 Analyze Board</button>
                    <button class="btn-analyze" style="background: #999;" onclick="resetApp()">Reset</button>
                </div>
                
                <div class="status info" id="status"></div>
                
                <div class="content-grid">
                    <div class="section">
                        <h2>📊 Detected Board</h2>
                        <div class="board-stats" id="boardStats"></div>
                        <div class="board-preview" id="boardPreview"></div>
                    </div>
                    
                    <div class="section">
                        <h2>🎯 Top Moves</h2>
                        <div class="rack-display" id="rackDisplay" style="display: none;"></div>
                        <div class="moves-list" id="movesList"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentImage = null;
        
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = (e) => {
                currentImage = e.target.result;
                document.getElementById('previewImg').src = currentImage;
                document.getElementById('imagePreview').classList.add('active');
                document.getElementById('uploadSection').classList.add('has-image');
                document.getElementById('analysisSection').style.display = 'block';
                showStatus('Image loaded. Tap "Analyze Board" to detect tiles and moves.', 'info');
            };
            reader.readAsDataURL(file);
        }
        
        function showStatus(message, type = 'info') {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = `status show ${type}`;
        }
        
        async function analyzeImage() {
            if (!currentImage) {
                showStatus('Please upload an image first', 'error');
                return;
            }
            
            showStatus('🔄 Analyzing board...', 'info');
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image: currentImage})
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayBoard(data.board);
                    displayRack(data.rack);
                    displayMoves(data.moves);
                    showStatus('✅ Board analyzed! Here are your top moves.', 'success');
                } else {
                    showStatus('⚠️ ' + data.error, 'error');
                }
            } catch (error) {
                showStatus('❌ Error: ' + error.message, 'error');
            }
        }
        
        function displayBoard(board) {
            const preview = document.getElementById('boardPreview');
            preview.innerHTML = '';
            
            let tiles = 0, bonuses = 0, empty = 0;
            for (let r = 0; r < 15; r++) {
                for (let c = 0; c < 15; c++) {
                    const cell = document.createElement('div');
                    cell.className = 'cell';
                    
                    if (board[r][c] === 'T') {
                        cell.className += ' tile';
                        cell.textContent = 'T';
                        tiles++;
                    } else if (board[r][c] === 'B') {
                        cell.className += ' bonus';
                        cell.textContent = 'B';
                        bonuses++;
                    } else {
                        cell.className += ' empty';
                    }
                    empty++;
                    preview.appendChild(cell);
                }
            }
            
            document.getElementById('boardStats').textContent = 
                `Tiles: ${tiles} | Bonuses: ${bonuses} | Empty: ${empty}`;
        }
        
        function displayRack(rack) {
            const display = document.getElementById('rackDisplay');
            display.textContent = rack.split('').join(' ');
            display.style.display = 'block';
        }
        
        function displayMoves(moves) {
            const container = document.getElementById('movesList');
            container.innerHTML = '';
            
            if (!moves || moves.length === 0) {
                container.innerHTML = '<div style="padding: 10px;">No moves found</div>';
                return;
            }
            
            for (let i = 0; i < moves.length; i++) {
                const move = moves[i];
                const item = document.createElement('div');
                item.className = 'move-item';
                item.innerHTML = `
                    <div>
                        <span class="move-rank">${i + 1}</span>
                        <span class="move-word">${move.word}</span>
                        <span class="move-score">${move.score} pts</span>
                    </div>
                    <div class="move-details">
                        Leave: <strong>${move.leave}</strong>
                    </div>
                `;
                container.appendChild(item);
            }
        }
        
        function resetApp() {
            currentImage = null;
            document.getElementById('imagePreview').classList.remove('active');
            document.getElementById('uploadSection').classList.remove('has-image');
            document.getElementById('analysisSection').style.display = 'none';
            document.getElementById('fileInput').value = '';
        }
    </script>
</body>
</html>'''
    return html

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze board image and return detected state + moves"""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image provided'})
        
        # Decode base64 image
        try:
            header, encoded = image_data.split(',', 1)
            decoded = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(decoded))
        except:
            return jsonify({'success': False, 'error': 'Invalid image format'})
        
        # Predict board
        board, board_error = predict_board(image)
        if board_error:
            return jsonify({'success': False, 'error': board_error})
        
        # Predict tray
        rack, tray_error = predict_tray(image)
        if tray_error:
            rack = "???????"
        
        # Generate moves
        moves = generate_moves(board, rack)
        
        return jsonify({
            'success': True,
            'board': board,
            'rack': rack,
            'moves': moves
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'board_model_loaded': board_classifier is not None,
        'tray_model_loaded': tray_classifier is not None
    })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("MAPS Backend - Google Cloud Run")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=PORT, debug=False)
