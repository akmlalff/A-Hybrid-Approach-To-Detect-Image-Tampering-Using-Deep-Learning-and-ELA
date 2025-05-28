# backend/inference.py

import os
import io
import json
import argparse
import base64
import numpy as np
# silence TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance, ExifTags
from datetime import datetime
import sys

if getattr(sys, 'frozen', False):
    # Running as a bundled exe
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(base_path, 'models', 'cnn_tamper.h5')
TRAIN_SIZE    = (128, 128)
TRAIN_QUALITY = 90

_model = None
def load_model():
    global _model
    if _model is None:
        try:
            # First try loading with custom_objects to handle version differences
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False, 
                custom_objects={'RMSprop': tf.keras.optimizers.RMSprop})
            
            # Ensure model is not compiled for inference
            _model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # Test model with a dummy input to ensure it works
            dummy_input = np.zeros((1, *TRAIN_SIZE, 3), dtype=np.float32)
            _model.predict(dummy_input, verbose=0)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}", file=sys.stderr)
            raise
    return _model

def make_ela(path):
    img = Image.open(path).convert('RGB')
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=TRAIN_QUALITY)
    buf.seek(0)
    rec = Image.open(buf)
    diff = ImageChops.difference(img, rec)
    maxd = max(e[1] for e in diff.getextrema()) or 1
    ela = ImageEnhance.Brightness(diff).enhance(255.0 / maxd)
    return ela.resize(TRAIN_SIZE)

def extract_metadata(path):
    img = Image.open(path)
    raw = {}
    # JPEG EXIF
    if hasattr(img, '_getexif'):
        for tag_id, val in (img._getexif() or {}).items():
            name = ExifTags.TAGS.get(tag_id, tag_id)
            raw[name] = val
    # other info (PNG tEXt, TIFF tags, etc.)
    for k, v in img.info.items():
        if k not in raw:
            raw[k] = v

    # Extract file information
    file_stats = os.stat(path)
    
    # Create simplified metadata with specified fields
    simplified = {
        "file_type": os.path.splitext(path)[1].lstrip('.').upper() or "-",
        "size": f"{file_stats.st_size} bytes" if file_stats.st_size else "-",
        "file_created": str(datetime.fromtimestamp(file_stats.st_ctime)) if file_stats.st_ctime else "-",
        "file_modified": str(datetime.fromtimestamp(file_stats.st_mtime)) if file_stats.st_mtime else "-",
        "image_dimension": f"{img.width}x{img.height}" if img.width and img.height else "-",
        "authors": raw.get("Artist", "-"),
        "date_taken": raw.get("DateTime", raw.get("DateTimeOriginal", "-")),
        "program_name": raw.get("Software", raw.get("ProcessingSoftware", "-"))
    }
    
    # now coerce to JSON‐safe
    safe = {}
    for k, v in raw.items():
        if isinstance(v, bytes):
            try:
                safe[k] = v.decode('utf-8')
            except:
                safe[k] = base64.b64encode(v).decode('ascii')
        else:
            try:
                json.dumps(v)
                safe[k] = v
            except TypeError:
                safe[k] = str(v)
                
    return {"simplified": simplified, "full": safe}

def preprocess(path):
    ela_img = make_ela(path)
    arr     = np.array(ela_img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0), ela_img

def predict(path):
    model   = load_model()
    x, ela  = preprocess(path)
    preds   = model.predict(x, verbose=0)[0]
    label   = 'tampered' if preds[1] > preds[0] else 'authentic'
    confidence = float(np.max(preds))

    # save ELA in data/route directory
    base = os.path.basename(path)
    base, _ = os.path.splitext(base)
    ela_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'route')
    os.makedirs(ela_dir, exist_ok=True)
    ela_path = os.path.join(ela_dir, base + '_ela.jpg')
    ela.save(ela_path, format='JPEG')

    # extract metadata
    metadata = extract_metadata(path)

    return {
        'label':      label,
        'confidence': confidence,
        'ela_path':   'file://' + os.path.abspath(ela_path),
        'metadata':   metadata
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image file')
    args = parser.parse_args()
    # **only** JSON on stdout
    print(json.dumps(predict(args.image)))
