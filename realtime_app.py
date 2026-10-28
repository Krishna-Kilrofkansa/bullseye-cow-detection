from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import os
import logging
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cow_detection_secret'

# Prefer eventlet if available (optional)
try:
    import eventlet  # noqa: F401
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")
except Exception:
    socketio = SocketIO(app, cors_allowed_origins="*")

# Globals for models (will be loaded lazily)
yolo_model = None
breed_model = None
lime_explainer = None
shap_explainer = None
breedtrace_engine = None

breed_classes = [
    'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari',
    'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar',
    'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam',
    'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari',
    'Krishna_Valley', 'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori',
    'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi',
    'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar', 'Toda',
    'Umblachery', 'Vechur'
]

# Basic logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)


@app.route('/')
def index():
    return render_template('realtime.html')


@app.route('/load_model', methods=['POST'])
def load_model():
    global yolo_model, breed_model
    force = request.is_json and request.json.get('force', False)
    if yolo_model is not None and breed_model is not None and not force:
        return jsonify({'status': 'success', 'message': 'Models already loaded and ready'})

    try:
        from ultralytics import YOLO
        import tensorflow as tf

        yolo_weights = request.json.get('yolo_weights', 'yolov8n.pt') if request.is_json else 'yolov8n.pt'
        breed_weights = request.json.get('breed_weights', 'classifier.h5') if request.is_json else 'classifier.h5'

        app.logger.info(f"Loading YOLO weights from: {yolo_weights}")
        yolo_model = YOLO(yolo_weights)

        app.logger.info(f"Loading breed classifier from: {breed_weights}")
        breed_model = tf.keras.models.load_model(breed_weights)

        return jsonify({'status': 'success', 'message': 'Models loaded successfully'})
    except Exception as e:
        app.logger.exception("Error loading models")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def classify_breed(cow_crop):
    """
    Classify cropped cow image using breed_model and return:
    - best breed name
    - best confidence
    - best class index
    - top 5 candidates
    - all 41 prediction probabilities sorted descending
    """
    global breed_model
    try:
        # Crucial fix: VGG16 was trained on RGB images; OpenCV frames are BGR!
        # Swapped channels caused red/brown cattle to look dark blue-grey, heavily biasing toward Jaffrabadi / Surti buffaloes!
        if len(cow_crop.shape) == 3 and cow_crop.shape[2] == 3:
            rgb_crop = cv2.cvtColor(cow_crop, cv2.COLOR_BGR2RGB)
        else:
            rgb_crop = cow_crop

        img = cv2.resize(rgb_crop, (150, 150), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32') / 255.0

        prediction = breed_model.predict(img, verbose=0)[0]
        sorted_indices = np.argsort(prediction)[::-1]

        all_predictions = []
        for idx in sorted_indices:
            all_predictions.append({
                'index': int(idx),
                'breed': breed_classes[idx],
                'confidence': float(prediction[idx])
            })

        best = all_predictions[0]
        top_candidates = all_predictions[:5]
        return best['breed'], best['confidence'], best['index'], top_candidates, all_predictions
    except Exception as e:
        app.logger.exception("Classification error")
        return "Unknown", 0.0, 0, [], []


def process_image_and_detect(frame, is_snapshot=False):
    """
    Robust cow detection and breed classification pipeline:
    1. Runs YOLOv8 with flexible threshold and quadruped animal classes (cow, horse, sheep).
    2. Applies anatomical padding margin so horns, ears, and dewlap are not clipped by tight boxes.
    3. If no box is found on a snapshot/upload, runs direct full-image classification with confidence threshold.
    """
    global yolo_model
    results = []
    h_frame, w_frame = frame.shape[:2]

    # Run YOLO with conf=0.20 and classes: 19 (cow), 17 (horse), 18 (sheep)
    try:
        detections = yolo_model(frame, conf=0.20, classes=[19, 17, 18], verbose=False)[0]
        if detections.boxes is not None and len(detections.boxes) > 0:
            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # Ensure valid box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frame, x2), min(h_frame, y2)
                
                # Contextual padding (+12%) so horns, ears, and neck folds are not clipped
                pad_x = int((x2 - x1) * 0.12)
                pad_y = int((y2 - y1) * 0.12)
                x1_pad = max(0, x1 - pad_x)
                y1_pad = max(0, y1 - pad_y)
                x2_pad = min(w_frame, x2 + pad_x)
                y2_pad = min(h_frame, y2 + pad_y)

                cow_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                if cow_crop.size > 0:
                    breed, breed_conf, breed_idx, top_candidates, all_predictions = classify_breed(cow_crop)

                    # Green bounding box for detected cow
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"{breed}: {breed_conf*100:.1f}%"
                    cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                    # Crop image preview
                    _, crop_buf = cv2.imencode('.jpg', cow_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    crop_b64 = base64.b64encode(crop_buf).decode('utf-8')

                    results.append({
                        'breed': breed,
                        'breed_conf': float(breed_conf),
                        'breed_idx': int(breed_idx),
                        'detection_conf': confidence,
                        'top_candidates': top_candidates,
                        'all_predictions': all_predictions,
                        'crop_image': f"data:image/jpeg;base64,{crop_b64}"
                    })
    except Exception as e:
        app.logger.exception("YOLO inference error")

    # Fallback: if no box found on snapshot or manual upload, test if the frame has a clear bovine
    if len(results) == 0 and is_snapshot:
        breed, breed_conf, breed_idx, top_candidates, all_predictions = classify_breed(frame)
        # Require reasonable confidence (>12%) so arbitrary rooms/walls are not reported as cows
        if breed_conf >= 0.12:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (8, 8), (w - 8, h - 8), (255, 200, 0), 2)
            label = f"{breed}: {breed_conf*100:.1f}% (Close-up)"
            cv2.putText(frame, label, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

            _, crop_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            crop_b64 = base64.b64encode(crop_buf).decode('utf-8')

            results.append({
                'breed': breed,
                'breed_conf': float(breed_conf),
                'breed_idx': int(breed_idx),
                'detection_conf': 0.75,
                'top_candidates': top_candidates,
                'all_predictions': all_predictions,
                'crop_image': f"data:image/jpeg;base64,{crop_b64}",
                'is_closeup': True
            })
        else:
            app.logger.info(f"Frame rejected: low breed confidence ({breed_conf*100:.1f}%) and no YOLO box.")

    return frame, results


# =========================================================================
# Explainable AI (XAI) Algorithms: LIME, SHAP, and Grad-CAM
# =========================================================================

def ensure_models_loaded():
    global yolo_model, breed_model
    if yolo_model is None or breed_model is None:
        from ultralytics import YOLO
        import tensorflow as tf
        yolo_model = YOLO('yolov8n.pt')
        breed_model = tf.keras.models.load_model('classifier.h5')


def compute_gradcam(model, rgb_150, class_idx):
    """Grad-CAM for VGG16 block5_conv3 layer with logit-gradient loss"""
    import tensorflow as tf
    inp = np.expand_dims(rgb_150.astype('float32') / 255.0, axis=0)
    conv_layer = model.get_layer('block5_conv3')
    grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[conv_layer.output, model.outputs[0]])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inp)
        loss = tf.math.log(predictions[:, class_idx] + 1e-10)
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    cam = conv_outputs @ pooled_grads[..., tf.newaxis]
    cam = tf.squeeze(cam).numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    
    cam_resized = cv2.resize(cam, (rgb_150.shape[1], rgb_150.shape[0]))
    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    bgr = cv2.cvtColor(rgb_150, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(bgr, 0.55, heatmap_colored, 0.45, 0)
    return overlay


def compute_lime(model, rgb_150, class_idx, num_samples=35):
    """LIME Superpixel Explanation highlighting top 5 positive feature segments"""
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    global lime_explainer
    if lime_explainer is None:
        lime_explainer = lime_image.LimeImageExplainer()

    def predict_fn(imgs):
        inp = imgs.astype('float32')
        if inp.max() > 1.0:
            inp /= 255.0
        return model.predict(inp, verbose=0)

    exp = lime_explainer.explain_instance(
        rgb_150,
        predict_fn,
        top_labels=None,
        labels=(class_idx,),
        num_samples=num_samples,
        random_seed=42
    )
    temp, mask = exp.get_image_and_mask(class_idx, positive_only=True, num_features=5, hide_rest=False)
    boundary = mark_boundaries(temp, mask, color=(0, 1, 0), mode='thick')
    lime_bgr = cv2.cvtColor((boundary * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return lime_bgr


def compute_shap(model, rgb_150, class_idx, max_evals=28):
    """SHAP Image Explanation using PartitionExplainer with blur masker"""
    import shap

    global shap_explainer
    def predict_fn(imgs):
        inp = imgs.astype('float32')
        if inp.max() > 1.0:
            inp /= 255.0
        return model.predict(inp, verbose=0)

    if shap_explainer is None:
        masker = shap.maskers.Image('blur(16, 16)', (150, 150, 3))
        shap_explainer = shap.Explainer(predict_fn, masker)

    img_batch = np.expand_dims(rgb_150, axis=0)
    shap_vals = shap_explainer(img_batch, max_evals=max_evals, batch_size=max_evals)
    
    # shap_vals.values shape: (1, 150, 150, 3, 41)
    s_map = np.mean(shap_vals.values[0, :, :, :, class_idx], axis=-1)
    s_map = np.maximum(s_map, 0)
    if s_map.max() > 0:
        s_map = s_map / s_map.max()

    s_map_resized = cv2.resize(s_map, (rgb_150.shape[1], rgb_150.shape[0]))
    heatmap = (s_map_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)
    bgr = cv2.cvtColor(rgb_150, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(bgr, 0.55, heatmap_colored, 0.45, 0)
    return overlay


@app.route('/explain_xai', methods=['POST'])
def explain_xai():
    """
    On-demand XAI explanation endpoint:
    Accepts: { image: base64, method: 'lime' | 'shap' | 'gradcam', class_idx: int }
    Returns: { status: 'success', explanation_image: base64, target_breed, target_conf, elapsed_sec, summary }
    """
    ensure_models_loaded()
    try:
        data = request.get_json(force=True)
        img_b64 = data.get('image', '')
        method = data.get('method', 'lime').lower().strip()
        class_idx = int(data.get('class_idx', 0))

        if not img_b64:
            return jsonify({'status': 'error', 'message': 'No image provided for XAI analysis'}), 400

        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]

        raw_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return jsonify({'status': 'error', 'message': 'Could not decode input image'}), 400

        # Resize to standard 150x150 required by VGG16 classifier
        crop_150 = cv2.resize(bgr, (150, 150))
        rgb_150 = cv2.cvtColor(crop_150, cv2.COLOR_BGR2RGB)

        # Get direct model prediction on this crop to verify target confidence
        inp = np.expand_dims(rgb_150.astype('float32') / 255.0, axis=0)
        preds = breed_model.predict(inp, verbose=0)[0]
        
        if class_idx < 0 or class_idx >= len(breed_classes):
            class_idx = int(np.argmax(preds))

        target_breed = breed_classes[class_idx]
        target_conf = float(preds[class_idx])

        t0 = time.time()
        if method == 'breedtrace':
            global breedtrace_engine
            if breedtrace_engine is None:
                from breedtrace_engine import BreedTraceEngine
                breedtrace_engine = BreedTraceEngine(breed_model, breed_classes)
            
            comp_idx = data.get('comp_idx', None)
            if comp_idx is not None:
                comp_idx = int(comp_idx)

            bt_res = breedtrace_engine.explain(rgb_150, target_breed_idx=class_idx, comp_breed_idx=comp_idx)
            return jsonify(bt_res)
        elif method == 'lime':
            vis = compute_lime(breed_model, rgb_150, class_idx, num_samples=35)
            summary = f"LIME identified the 5 most decisive superpixel anatomical segments supporting '{target_breed}'."
        elif method == 'shap':
            vis = compute_shap(breed_model, rgb_150, class_idx, max_evals=25)
            summary = f"SHAP computed Shapley feature contributions across image partitions for '{target_breed}'."
        elif method == 'gradcam':
            vis = compute_gradcam(breed_model, rgb_150, class_idx)
            summary = f"Grad-CAM mapped high-gradient convolutional activations in VGG16 (block5_conv3) for '{target_breed}'."
        else:
            return jsonify({'status': 'error', 'message': f"Unsupported XAI method '{method}'. Choose 'breedtrace', 'lime', 'shap', or 'gradcam'."}), 400

        elapsed = round(time.time() - t0, 2)

        # Upscale vis slightly (e.g. 360x360) for high visual fidelity in UI modal
        vis_upscaled = cv2.resize(vis, (360, 360), interpolation=cv2.INTER_LANCZOS4)
        _, buf = cv2.imencode('.jpg', vis_upscaled, [cv2.IMWRITE_JPEG_QUALITY, 90])
        vis_b64 = base64.b64encode(buf).decode('utf-8')

        return jsonify({
            'status': 'success',
            'method': method,
            'target_breed': target_breed,
            'target_conf': target_conf,
            'class_idx': class_idx,
            'elapsed_sec': elapsed,
            'summary': summary,
            'explanation_image': f"data:image/jpeg;base64,{vis_b64}"
        })
    except Exception as e:
        app.logger.exception("XAI explanation failed")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/detect_image', methods=['POST'])
def detect_image():
    """Process uploaded image file for cow detection and breed classification"""
    ensure_models_loaded()
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image uploaded'}), 400
        
        file = request.files['image']
        np_img = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Invalid image format'}), 400

        # Resize if overly large
        h, w = frame.shape[:2]
        if w > 1024:
            scale = 1024 / w
            frame = cv2.resize(frame, (1024, int(h * scale)))

        annotated_frame, results = process_image_and_detect(frame, is_snapshot=True)

        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'status': 'success',
            'image': f"data:image/jpeg;base64,{img_base64}",
            'results': results
        })
    except Exception as e:
        app.logger.exception("Error processing uploaded image")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@socketio.on('video_frame')
def handle_video_frame(data):
    ensure_models_loaded()
    is_snapshot = data.get('is_snapshot', False)

    try:
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit('detection_result', {'status': 'error', 'message': 'Frame decoding failed', 'is_snapshot': is_snapshot})
            return

        # Resize frame
        height, width = frame.shape[:2]
        target_width = 800 if is_snapshot else 640
        if width > target_width:
            scale = target_width / width
            frame = cv2.resize(frame, (target_width, int(height * scale)))

        annotated_frame, results = process_image_and_detect(frame, is_snapshot=is_snapshot)

        quality = 85 if is_snapshot else 70
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        emit('detection_result', {
            'status': 'success',
            'image': f"data:image/jpeg;base64,{img_base64}",
            'results': results,
            'is_snapshot': is_snapshot
        })
    except Exception as e:
        app.logger.exception("Error during frame handling")
        emit('detection_result', {'status': 'error', 'message': str(e), 'is_snapshot': is_snapshot})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.logger.info(f"Server starting up and ready to accept requests on port {port}")
    socketio.run(app, debug=False, host="0.0.0.0", port=port)
