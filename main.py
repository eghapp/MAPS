#!/usr/bin/env python3
"""
MAPS Cloud Run Backend v12
===========================
Module 1: Screenshot → ML Board/Tray OCR → Editable Confirmation Grid

Endpoints:
  GET  /              - Web UI (iPad-friendly: upload → detect → confirm)
  GET  /health        - Health check + model status
  POST /upload_models - Upload .pkl model files (board + tray classifiers)
  POST /upload_wordlist - Upload NWL23.txt word list for solver
  POST /detect        - Screenshot → board grid + tray (ML pipeline)

Models are in-memory only. After cold starts or redeployments,
re-upload via upload_models.py.

Author: Claude (for Ed)
Date: 2026-04-04
Version: 12
"""

import io
import json
import os
import pickle
import traceback
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, ImageFilter, ImageOps

app = Flask(__name__)

# In-memory storage
MODELS = {"board_classifier": None, "tray_classifier": None}
WORDLIST = None

# Crossplay tile values — Protocol v34 Appendix F
TILE_VALUES = {
    "A":1,"B":4,"C":3,"D":2,"E":1,"F":4,"G":4,"H":3,"I":1,"J":10,
    "K":6,"L":2,"M":3,"N":1,"O":1,"P":3,"Q":10,"R":1,"S":1,"T":1,
    "U":2,"V":6,"W":5,"X":8,"Y":4,"Z":10,"?":0
}

# Crossplay bonus layout — Protocol v34 Appendix B (verified)
BONUS_LAYOUT = [
    ["3L","","","3W","","","","2L","","","","3W","","","3L"],
    ["","2W","","","","","3L","","3L","","","","","2W",""],
    ["","","","","2L","","","","","","2L","","","",""],
    ["3W","","","2L","","","","2W","","","","2L","","","3W"],
    ["","","2L","","","3L","","","","3L","","","2L","",""],
    ["","","","","3L","","","2L","","","3L","","","",""],
    ["","3L","","","","","","","","","","","","3L",""],
    ["2L","","","2W","","2L","","","","2L","","2W","","","2L"],
    ["","3L","","","","","","","","","","","","3L",""],
    ["","","","","3L","","","2L","","","3L","","","",""],
    ["","","2L","","","3L","","","","3L","","","2L","",""],
    ["3W","","","2L","","","","2W","","","","2L","","","3W"],
    ["","","","","2L","","","","","","2L","","","",""],
    ["","2W","","","","","3L","","3L","","","","","2W",""],
    ["3L","","3W","","","","","2L","","","","","3W","","3L"],
]

# ============================================================
# Feature extraction (must match training exactly — 14 features)
# ============================================================
def extract_features(cell_img):
    gray = ImageOps.grayscale(cell_img)
    ga = np.array(gray, dtype=float)
    rgb = np.array(cell_img.convert("RGB"), dtype=float)
    h, w = ga.shape
    if h == 0 or w == 0: return [0]*14
    edges = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=float)
    ch, cw = max(1,h//4), max(1,w//4)
    center = ga[ch:h-ch, cw:w-cw]
    if center.size == 0: center = ga
    tp = ga.size
    hist, _ = np.histogram(ga.flatten(), bins=256, range=(0,256))
    p = hist / tp
    return [ga.mean(), ga.std(), (ga<128).sum()/tp*100, (ga<64).sum()/tp*100,
            edges.mean(), float((edges>128).sum()), float(ga.max()-ga.min()),
            rgb[:,:,0].mean(), rgb[:,:,1].mean(), rgb[:,:,2].mean(),
            rgb.var(), float(np.sum(p*p)), center.mean(), float((center<128).sum())]

# ============================================================
# Board and tray detection
# ============================================================
def find_board(img):
    arr = np.array(img.convert("RGB")); h, w = arr.shape[:2]
    gray = np.mean(arr, axis=2)
    board_w = int(w * 0.95); x1 = (w-board_w)//2; x2 = x1+board_w; side = board_w
    row_frac = [(gray[y, x1:x2] < 240).mean() for y in range(h)]
    board_top = int(h * 0.25)
    for y in range(int(h*0.10), int(h*0.40)):
        if row_frac[y] > 0.50:
            if all(row_frac[min(y+i,h-1)] > 0.40 for i in range(10)):
                board_top = y; break
    if board_top + side > h: side = h - board_top
    cx = w // 2; x1 = max(0, cx-side//2); x2 = x1+side
    return x1, board_top, x2, board_top + side

def find_tray(img):
    arr = np.array(img.convert("RGB")); h, w = arr.shape[:2]
    blue_frac = [((arr[y,:,0]>30)&(arr[y,:,0]<120)&(arr[y,:,2]>140)).mean() for y in range(h)]
    bands = []; in_band = False; bs = 0
    for y in range(h):
        if blue_frac[y] > 0.30:
            if not in_band: bs = y; in_band = True
        else:
            if in_band and y-bs > 15: bands.append((bs, y))
            in_band = False
    if in_band and h-bs > 15: bands.append((bs, h))
    if not bands: return None
    tt, tb = bands[-1]
    if tt < h*0.70 or (tb-tt) > 200: return None
    return tt, tb

def ocr_tile(cell_img):
    try:
        import pytesseract
    except ImportError:
        return "?"
    arr = np.array(cell_img.convert("RGB")); h, w = arr.shape[:2]
    mx, my = int(w*0.18), int(h*0.12)
    inner = arr[my:h-my, mx:w-mx]
    white_mask = (inner[:,:,0]>180)&(inner[:,:,1]>180)&(inner[:,:,2]>180)
    binary = np.where(white_mask, 0, 255).astype(np.uint8)
    bin_img = Image.fromarray(binary)
    big = ImageOps.expand(bin_img.resize((bin_img.width*5, bin_img.height*5), Image.NEAREST), border=20, fill=255)
    text = pytesseract.image_to_string(big, config="--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ").strip()
    if len(text)==1 and text.isalpha(): return text.upper()
    text = pytesseract.image_to_string(big, config="--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ").strip()
    if len(text)>=1 and text[0].isalpha(): return text[0].upper()
    return "?"

# ============================================================
# Detection pipeline
# ============================================================
def detect_board_and_tray(img):
    board_model = MODELS.get("board_classifier")
    tray_model = MODELS.get("tray_classifier")
    if not board_model:
        return {"error": "Board classifier not loaded. Upload via /upload_models first."}
    clf = board_model["classifier"]; scaler = board_model["scaler"]
    x1,y1,x2,y2 = find_board(img)
    cw = (x2-x1)/15; ch_px = (y2-y1)/15
    grid = [["."]*15 for _ in range(15)]
    cell_types = [["E"]*15 for _ in range(15)]
    tile_count = ocr_ok = 0
    for r in range(15):
        for c in range(15):
            cx1 = int(x1+c*cw); cy1 = int(y1+r*ch_px)
            cell = img.crop((cx1, cy1, int(cx1+cw), int(cy1+ch_px)))
            f = np.array([extract_features(cell)])
            pred = clf.predict(scaler.transform(f))[0]
            cell_types[r][c] = pred
            if pred == "T":
                tile_count += 1; letter = ocr_tile(cell); grid[r][c] = letter
                if letter != "?": ocr_ok += 1
            elif pred == "B":
                grid[r][c] = BONUS_LAYOUT[r][c] if BONUS_LAYOUT[r][c] else "."
            else:
                grid[r][c] = "."
    tray = []
    tray_result = find_tray(img)
    if tray_result and tray_model:
        tt, tb = tray_result; tcf = tray_model["classifier"]; tsc = tray_model["scaler"]
        tw = img.size[0]/7
        for i in range(7):
            tile = img.crop((int(i*tw), tt, int((i+1)*tw), tb)).resize((80,80))
            f = np.array([extract_features(tile)])
            tray.append(tcf.predict(tsc.transform(f))[0])
    elif tray_result:
        tray = ["?"]*7
    return {"grid": grid, "tray": tray, "cell_types": cell_types, "stats": {
        "board_bounds": [x1,y1,x2,y2], "cell_size": [int(cw),int(ch_px)],
        "tiles_detected": tile_count, "ocr_success": ocr_ok,
        "ocr_failed": tile_count-ocr_ok, "tray_detected": tray_result is not None}}

def find_words_on_board(grid):
    words = []
    for r in range(15):
        c = 0
        while c < 15:
            ch = grid[r][c]
            if len(ch)==1 and ch.isalpha():
                start = c; word = ""
                while c<15 and len(grid[r][c])==1 and grid[r][c].isalpha(): word += grid[r][c]; c += 1
                if len(word) >= 2:
                    words.append({"word":word,"direction":"across","row":r+1,"col":chr(65+start),"start":f"{chr(65+start)}{r+1}"})
            else: c += 1
    for c in range(15):
        r = 0
        while r < 15:
            ch = grid[r][c]
            if len(ch)==1 and ch.isalpha():
                start = r; word = ""
                while r<15 and len(grid[r][c])==1 and grid[r][c].isalpha(): word += grid[r][c]; r += 1
                if len(word) >= 2:
                    words.append({"word":word,"direction":"down","row":start+1,"col":chr(65+c),"start":f"{chr(65+c)}{start+1}"})
            else: r += 1
    return words

# ============================================================
# Flask routes
# ============================================================
@app.route("/health")
def health():
    return jsonify({"status":"ok","version":"v12",
        "board_classifier": MODELS["board_classifier"] is not None,
        "tray_classifier": MODELS["tray_classifier"] is not None,
        "wordlist_loaded": WORDLIST is not None,
        "wordlist_size": len(WORDLIST) if WORDLIST else 0})

@app.route("/upload_models", methods=["POST"])
def upload_models():
    results = {}
    for name in ["board_classifier", "tray_classifier"]:
        if name in request.files:
            try:
                model = pickle.loads(request.files[name].read())
                MODELS[name] = model
                results[name] = {"status":"loaded","version":model.get("version","unknown")}
            except Exception as e:
                results[name] = {"status":"error","error":str(e)}
    if not results:
        return jsonify({"error":"No model files. Use field names: board_classifier, tray_classifier"}), 400
    return jsonify(results)

@app.route("/upload_wordlist", methods=["POST"])
def upload_wordlist():
    global WORDLIST
    if "wordlist" not in request.files:
        return jsonify({"error":"No file. Use field name: wordlist"}), 400
    try:
        content = request.files["wordlist"].read().decode("utf-8", errors="ignore")
        WORDLIST = {line.strip().upper() for line in content.splitlines()
                    if line.strip().isalpha() and 2 <= len(line.strip()) <= 15}
        return jsonify({"status":"loaded","words":len(WORDLIST)})
    except Exception as e:
        return jsonify({"status":"error","error":str(e)}), 500

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error":"No image. Use field name: image"}), 400
    try:
        img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
        result = detect_board_and_tray(img)
        if "error" in result: return jsonify(result), 503
        result["words"] = find_words_on_board(result["grid"])
        result["bonus_layout"] = BONUS_LAYOUT
        return jsonify(result)
    except Exception as e:
        return jsonify({"error":str(e),"traceback":traceback.format_exc()}), 500

# ============================================================
# Web UI — iPad-friendly, with editable confirmation grid
# ============================================================
@app.route("/")
def index():
    return """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no">
<title>MAPS Analyzer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,system-ui,sans-serif;background:#f5f5f7;color:#1d1d1f;padding:10px;max-width:600px;margin:0 auto}
h1{font-size:20px;text-align:center;margin-bottom:6px}
.status{font-size:12px;color:#86868b;text-align:center;margin-bottom:12px}
.ok{color:#34c759}.no{color:#ff3b30}
.upload-area{background:#fff;border-radius:12px;padding:16px;text-align:center;border:2px dashed #d1d1d6;margin-bottom:12px}
input[type=file]{display:none}
.btn{display:inline-block;background:#007aff;color:#fff;padding:10px 20px;border-radius:10px;font-size:15px;font-weight:600;border:none;cursor:pointer;margin:4px}
.btn:disabled{background:#d1d1d6}
.btn-green{background:#34c759}
.preview{max-width:100%;max-height:200px;border-radius:8px;margin:8px auto;display:block}
.section{background:#fff;border-radius:12px;padding:12px;margin-bottom:10px}
.section h2{font-size:15px;margin-bottom:6px}

/* Editable Board Grid */
.board-grid{border-collapse:collapse;margin:0 auto;user-select:none}
.board-grid td{width:28px;height:28px;text-align:center;font-size:12px;font-weight:700;
  border:1px solid #ccc;padding:0;position:relative;cursor:pointer}
.board-grid td input{width:100%;height:100%;border:none;text-align:center;font-size:12px;
  font-weight:700;background:transparent;color:inherit;padding:0;text-transform:uppercase}
.board-grid td input:focus{outline:2px solid #007aff;background:#e8f0fe}
.cell-tile{background:#4a7fc4;color:#fff}
.cell-3W{background:#e85d3a;color:#fff;font-size:9px}
.cell-3L{background:#3aa5e8;color:#fff;font-size:9px}
.cell-2W{background:#e8a03a;color:#fff;font-size:9px}
.cell-2L{background:#7bc8a4;color:#fff;font-size:9px}
.cell-empty{background:#f0efe8;color:#999}
.cell-star{background:#e8a03a;color:#fff;font-size:14px}
.cell-error{background:#ff6b6b;color:#fff}
.col-hdr{background:none;border:none;font-size:10px;color:#86868b;font-weight:400;height:18px}
.row-hdr{background:none;border:none;font-size:10px;color:#86868b;font-weight:400;width:20px}

/* Tray */
.tray{display:flex;gap:4px;justify-content:center;margin:6px 0;flex-wrap:wrap}
.tray-tile{width:36px;height:36px;background:#3a6fc9;color:#fff;font-size:18px;font-weight:700;
  display:flex;align-items:center;justify-content:center;border-radius:6px;position:relative}
.tray-tile input{width:100%;height:100%;border:none;text-align:center;font-size:18px;
  font-weight:700;background:transparent;color:#fff;padding:0;text-transform:uppercase;border-radius:6px}
.tray-tile input:focus{outline:2px solid #fff;background:#2a5fb9}

/* Words list */
.word-tag{display:inline-block;padding:2px 6px;border-radius:4px;margin:2px;font-family:monospace;font-size:12px}
.word-ok{background:#d4edda;color:#155724}
.word-bad{background:#f8d7da;color:#721c24}
.word-unk{background:#e8f0fe;color:#004085}

.spin{display:none;margin:12px auto;width:28px;height:28px;border:3px solid #d1d1d6;
  border-top-color:#007aff;border-radius:50%;animation:sp .8s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.err{background:#fff0f0;color:#ff3b30;padding:10px;border-radius:8px;margin:8px 0;font-size:13px}
.stats{font-size:11px;color:#86868b;margin-top:4px}
.help{font-size:11px;color:#86868b;text-align:center;margin:4px 0}
</style></head><body>

<h1>MAPS Crossplay Analyzer</h1>
<div class="status" id="st">Checking...</div>

<!-- Step 1: Upload -->
<div id="step1">
<div class="upload-area">
  <p style="margin-bottom:8px;color:#86868b;font-size:13px">Upload a Crossplay screenshot</p>
  <input type="file" id="fi" accept="image/*">
  <button class="btn" onclick="document.getElementById('fi').click()">Choose Screenshot</button>
  <img id="pv" class="preview" style="display:none">
</div>
<button class="btn" id="ab" onclick="doDetect()" disabled style="width:100%">Analyze Screenshot</button>
</div>
<div class="spin" id="sp"></div>
<div id="ed"></div>

<!-- Step 2: Confirm -->
<div id="step2" style="display:none">

<div class="section">
  <h2>Your Tray <span style="font-weight:400;font-size:11px;color:#86868b">(tap to edit)</span></h2>
  <div class="tray" id="tray-area"></div>
</div>

<div class="section">
  <h2>Board <span style="font-weight:400;font-size:11px;color:#86868b">(tap any cell to correct)</span></h2>
  <div id="board-area" style="overflow-x:auto"></div>
  <div class="stats" id="sd"></div>
</div>

<div class="section">
  <h2>Words Found</h2>
  <div id="wd"></div>
</div>

<div style="text-align:center;margin:12px 0">
  <button class="btn" onclick="revalidate()">Re-validate</button>
  <button class="btn" onclick="resetAll()">New Screenshot</button>
</div>
<p class="help">Correct any ? or wrong letters above, then tap Re-validate. Solver module coming in v13.</p>
</div>

<script>
let sf=null, boardData=null, trayData=null;

// Health check
fetch('/health').then(r=>r.json()).then(d=>{
  const s=document.getElementById('st');
  const b=d.board_classifier?'<span class="ok">\\u2713</span>':'<span class="no">\\u2717 upload models</span>';
  const t=d.tray_classifier?'<span class="ok">\\u2713</span>':'<span class="no">\\u2717</span>';
  const w=d.wordlist_loaded?'<span class="ok">\\u2713 '+d.wordlist_size+'</span>':'<span class="no">\\u2717</span>';
  s.innerHTML='Board '+b+' | Tray '+t+' | Dict '+w;
}).catch(()=>{document.getElementById('st').innerHTML='<span class="no">Backend unreachable</span>'});

// File selection
document.getElementById('fi').addEventListener('change',e=>{
  sf=e.target.files[0];
  if(sf){document.getElementById('pv').src=URL.createObjectURL(sf);
  document.getElementById('pv').style.display='block';
  document.getElementById('ab').disabled=false}
});

// Step 1: Detect
async function doDetect(){
  if(!sf)return;
  document.getElementById('ab').disabled=true;
  document.getElementById('sp').style.display='block';
  document.getElementById('ed').innerHTML='';
  const fm=new FormData();fm.append('image',sf);
  try{
    const r=await fetch('/detect',{method:'POST',body:fm});
    const d=await r.json();
    if(d.error){document.getElementById('ed').innerHTML='<div class="err">'+d.error+'</div>';return}
    boardData=d.grid;
    trayData=d.tray||[];
    renderBoard(boardData, d.bonus_layout);
    renderTray(trayData);
    renderWords(boardData);
    const s=d.stats||{};
    document.getElementById('sd').textContent=
      'Tiles: '+s.tiles_detected+' | OCR OK: '+s.ocr_success+' | Failed: '+s.ocr_failed;
    document.getElementById('step1').style.display='none';
    document.getElementById('step2').style.display='block';
  }catch(e){document.getElementById('ed').innerHTML='<div class="err">'+e.message+'</div>'}
  finally{document.getElementById('ab').disabled=false;document.getElementById('sp').style.display='none'}
}

// Render editable board
function renderBoard(grid, bonus){
  const cols='ABCDEFGHIJKLMNO';
  let html='<table class="board-grid"><tr><td class="col-hdr"></td>';
  for(let c=0;c<15;c++) html+='<td class="col-hdr">'+cols[c]+'</td>';
  html+='</tr>';
  for(let r=0;r<15;r++){
    html+='<tr><td class="row-hdr">'+(r+1)+'</td>';
    for(let c=0;c<15;c++){
      const v=grid[r][c];
      const bon=bonus[r][c];
      let cls='cell-empty', display='', editable=false;
      if(v.length===1 && v.match(/[A-Z?]/i)){
        cls=v==='?'?'cell-error':'cell-tile';
        display=v; editable=true;
      } else if(r===7&&c===7&&v==='.'){
        cls='cell-star'; display='\\u2605';
      } else if(bon){
        cls='cell-'+bon.replace(' ',''); display=bon;
      }
      if(editable){
        html+='<td class="'+cls+'"><input type="text" maxlength="1" value="'+display+
          '" data-r="'+r+'" data-c="'+c+'" onfocus="this.select()" '+
          'oninput="cellEdit(this)"></td>';
      } else {
        html+='<td class="'+cls+'" onclick="cellClick(this,'+r+','+c+')">'+display+'</td>';
      }
    }
    html+='</tr>';
  }
  html+='</table>';
  document.getElementById('board-area').innerHTML=html;
}

// Click empty/bonus cell to add a letter
function cellClick(td,r,c){
  const inp=document.createElement('input');
  inp.type='text'; inp.maxLength=1; inp.dataset.r=r; inp.dataset.c=c;
  inp.onfocus=function(){this.select()};
  inp.oninput=function(){cellEdit(this)};
  td.textContent=''; td.appendChild(inp); td.className='cell-tile';
  inp.focus();
}

// Edit a cell
function cellEdit(inp){
  const v=inp.value.toUpperCase().replace(/[^A-Z]/g,'');
  inp.value=v;
  const r=parseInt(inp.dataset.r), c=parseInt(inp.dataset.c);
  if(v){
    boardData[r][c]=v;
    inp.parentElement.className='cell-tile';
  } else {
    boardData[r][c]='.';
    inp.parentElement.className='cell-empty';
  }
}

// Render editable tray
function renderTray(tray){
  let html='';
  for(let i=0;i<tray.length;i++){
    html+='<div class="tray-tile"><input type="text" maxlength="1" value="'+tray[i]+
      '" data-i="'+i+'" onfocus="this.select()" oninput="trayEdit(this)"></div>';
  }
  // Add empty slots up to 7
  for(let i=tray.length;i<7;i++){
    html+='<div class="tray-tile"><input type="text" maxlength="1" value="" '+
      'data-i="'+i+'" onfocus="this.select()" oninput="trayEdit(this)"></div>';
  }
  document.getElementById('tray-area').innerHTML=html;
}

function trayEdit(inp){
  const v=inp.value.toUpperCase().replace(/[^A-Z]/g,'');
  inp.value=v;
  const i=parseInt(inp.dataset.i);
  while(trayData.length<=i) trayData.push('');
  trayData[i]=v||'?';
}

// Find words and validate
function findWords(grid){
  const words=[];
  for(let r=0;r<15;r++){
    let c=0;
    while(c<15){
      const ch=grid[r][c];
      if(ch.length===1&&ch.match(/[A-Z]/)){
        let start=c, word='';
        while(c<15&&grid[r][c].length===1&&grid[r][c].match(/[A-Z]/)){word+=grid[r][c];c++}
        if(word.length>=2) words.push({word:word,dir:'across',start:String.fromCharCode(65+start)+(r+1)});
      } else c++;
    }
  }
  for(let c=0;c<15;c++){
    let r=0;
    while(r<15){
      const ch=grid[r][c];
      if(ch.length===1&&ch.match(/[A-Z]/)){
        let start=r, word='';
        while(r<15&&grid[r][c].length===1&&grid[r][c].match(/[A-Z]/)){word+=grid[r][c];r++}
        if(word.length>=2) words.push({word:word,dir:'down',start:String.fromCharCode(65+c)+(start+1)});
      } else r++;
    }
  }
  return words;
}

function renderWords(grid){
  const words=findWords(grid);
  let html='';
  const hasErrors=words.some(w=>w.word.includes('?'));
  words.forEach(w=>{
    const cls=w.word.includes('?')?'word-bad':'word-unk';
    html+='<span class="word-tag '+cls+'">'+w.word+' ('+w.start+' '+w.dir+')</span> ';
  });
  if(!words.length) html='<span style="color:#86868b">No words detected</span>';
  document.getElementById('wd').innerHTML=html;
}

function revalidate(){
  renderWords(boardData);
  renderBoard(boardData, """+json.dumps(BONUS_LAYOUT)+""");
  renderTray(trayData);
}

function resetAll(){
  document.getElementById('step1').style.display='block';
  document.getElementById('step2').style.display='none';
  document.getElementById('pv').style.display='none';
  document.getElementById('ab').disabled=true;
  sf=null; boardData=null; trayData=null;
}
</script></body></html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
