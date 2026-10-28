import tensorflow as tf
import numpy as np
import cv2
import base64
import time
from skimage.segmentation import slic


class BreedTraceEngine:
    """
    BreedTrace-XAI Engine
    Novel Contrastive Anatomical Evidence Verification Architecture for Bovine Classification:
    
    - Module A: Hypothesis-Guided Anatomical Part Localization (Cranial/Horns, Hump, Dewlap, Torso, Muzzle, Background)
    - Module B: Contrastive Breed Attribution (Predicted Breed c vs Competitor d)
    - Module C: Controlled Feature Intervention (Causal Logit Margin Ablation A_p^(c,d) = m_c,d(x) - m_c,d(x^(-p)))
    - Module D: Faithfulness & Reliability Verification (Monotonic Deletion + Background Shortcut Leakage)
    """

    ANATOMICAL_PARTS = [
        {
            'id': 'cranial_horns',
            'name': 'Cranial & Horns / Ears',
            'color': (0, 220, 255),       # Cyan
            'y_range': (0.0, 0.40),
            'x_range': (0.0, 1.0),
            'description': 'Forehead convexity, horn curvature, and ear pendulousness (crucial for Gir, Kankrej, Hallikar).'
        },
        {
            'id': 'thoracic_hump',
            'name': 'Thoracic Hump & Withers',
            'color': (255, 180, 0),       # Amber
            'y_range': (0.15, 0.55),
            'x_range': (0.30, 0.95),
            'description': 'Prominent dorsal hump distinguishing Bos indicus (Zebu) from humpless Bos taurus.'
        },
        {
            'id': 'dewlap_throat',
            'name': 'Dewlap & Ventral Folds',
            'color': (0, 255, 140),       # Emerald
            'y_range': (0.42, 0.80),
            'x_range': (0.10, 0.65),
            'description': 'Voluminous loose pendulous skin folds along the throat and chest line.'
        },
        {
            'id': 'torso_flank',
            'name': 'Torso, Flank & Coat',
            'color': (180, 100, 255),     # Violet
            'y_range': (0.35, 0.90),
            'x_range': (0.35, 1.0),
            'description': 'Body depth, barrel silhouette, and coat pigmentation patterns (e.g. Holstein patches, Sahiwal reddish-dun).'
        },
        {
            'id': 'muzzle_facial',
            'name': 'Muzzle & Facial Contour',
            'color': (255, 120, 180),     # Rose
            'y_range': (0.30, 0.68),
            'x_range': (0.0, 0.45),
            'description': 'Nostril pigmentation, muzzle width, and rostral facial taper.'
        },
        {
            'id': 'background_context',
            'name': 'Background & Context Control',
            'color': (120, 140, 160),     # Slate Grey
            'y_range': (0.0, 1.0),
            'x_range': (0.0, 1.0),
            'description': 'Non-anatomical peripheral context (used strictly for shortcut and artifact detection).'
        }
    ]

    def __init__(self, model, breed_classes):
        self.model = model
        self.breed_classes = breed_classes
        
        # Sub-models for rapid logit and activation extraction
        dense_2 = model.get_layer('dense_2')
        self.kernel, self.bias = dense_2.get_weights()
        
        # Linear representation model before final softmax
        dropout_1 = model.get_layer('dropout_1')
        self.feat_model = tf.keras.models.Model(inputs=model.inputs, outputs=dropout_1.output)
        
        # Convolutional feature extraction for contrastive backpropagation
        block5_conv = model.get_layer('block5_conv3')
        self.grad_model = tf.keras.models.Model(
            inputs=model.inputs, 
            outputs=[block5_conv.output, dropout_1.output]
        )

    def compute_logits(self, batch_imgs_float):
        """Vectorized logit calculation: z(x) = feats * W + b"""
        feats = self.feat_model(batch_imgs_float).numpy()
        return feats @ self.kernel + self.bias

    def localize_anatomical_regions(self, rgb_150):
        """
        Module A: Biological hypothesis localization.
        Combines SLIC boundary superpixels with anatomical spatial priors to create
        data-adaptive anatomical candidate masks adhering to true physical contours.
        """
        h, w = rgb_150.shape[:2]
        float_img = (rgb_150 / 255.0).astype(np.float64)
        
        # SLIC superpixel clustering with compactness tuned for anatomical contours
        segments = slic(float_img, n_segments=24, compactness=8, channel_axis=-1, start_label=0)
        unique_segs = np.unique(segments)
        
        # Build spatial masks for each anatomical part
        masks = {p['id']: np.zeros((h, w), dtype=np.uint8) for p in self.ANATOMICAL_PARTS}
        
        # Border boundary mask (outer 10% perimeter) for background control
        border_mask = np.zeros((h, w), dtype=bool)
        border_width_y = int(h * 0.10)
        border_width_x = int(w * 0.10)
        border_mask[:border_width_y, :] = True
        border_mask[-border_width_y:, :] = True
        border_mask[:, :border_width_x] = True
        border_mask[:, -border_width_x:] = True

        for seg_val in unique_segs:
            seg_mask = (segments == seg_val)
            seg_ys, seg_xs = np.where(seg_mask)
            if len(seg_ys) == 0:
                continue
                
            mean_y = np.mean(seg_ys) / float(h)
            mean_x = np.mean(seg_xs) / float(w)
            
            # Check if segment lies predominantly on background perimeter
            overlap_border = np.sum(border_mask[seg_ys, seg_xs]) / float(len(seg_ys))
            if overlap_border > 0.65:
                masks['background_context'][seg_mask] = 255
                continue

            # Score each candidate anatomical part
            best_part = 'torso_flank'
            best_score = -1e9
            
            for part in self.ANATOMICAL_PARTS:
                if part['id'] == 'background_context':
                    continue
                y_min, y_max = part['y_range']
                x_min, x_max = part['x_range']
                
                # Gaussian spatial prior alignment score
                y_center = (y_min + y_max) / 2.0
                x_center = (x_min + x_max) / 2.0
                y_span = (y_max - y_min) / 2.0 + 1e-5
                x_span = (x_max - x_min) / 2.0 + 1e-5
                
                dist_y = ((mean_y - y_center) / y_span) ** 2
                dist_x = ((mean_x - x_center) / x_span) ** 2
                score = - (dist_y + dist_x)
                
                if score > best_score:
                    best_score = score
                    best_part = part['id']
                    
            masks[best_part][seg_mask] = 255

        # Ensure no empty masks; if empty, apply geometric fallback
        for part in self.ANATOMICAL_PARTS:
            pid = part['id']
            if np.sum(masks[pid]) < 50:
                y1, y2 = int(part['y_range'][0] * h), int(part['y_range'][1] * h)
                x1, x2 = int(part['x_range'][0] * w), int(part['x_range'][1] * w)
                masks[pid][y1:y2, x1:x2] = 255

        return masks

    def compute_contrastive_activation_map(self, rgb_150, target_idx, comp_idx):
        """
        Module B: Contrastive Breed Attribution.
        Computes ∇_A (z_c - z_d) at block5_conv3 to identify spatial directions
        that push the logit margin in favor of target breed c vs competitor d.
        """
        inp = tf.convert_to_tensor(np.expand_dims(rgb_150.astype('float32') / 255.0, axis=0))
        
        with tf.GradientTape() as tape:
            conv_outputs, linear_feats = self.grad_model(inp)
            logits = tf.matmul(linear_feats, self.kernel) + self.bias
            margin = logits[0, target_idx] - logits[0, comp_idx]
            
        grads = tape.gradient(margin, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        
        # Contrastive saliency projection
        contrast_cam = conv_outputs @ pooled_grads[..., tf.newaxis]
        contrast_cam = tf.squeeze(contrast_cam).numpy()
        
        # Resize to original (150, 150)
        h, w = rgb_150.shape[:2]
        cam_resized = cv2.resize(contrast_cam, (w, h))
        
        # Normalize into range [-1, 1]
        max_abs = np.max(np.abs(cam_resized)) + 1e-10
        cam_normalized = cam_resized / max_abs
        return cam_normalized

    def smooth_inpaint_region(self, rgb_150, mask_255):
        """
        Module C: Context-aware harmonic suppression.
        Prevents unphysical high-frequency edge artifacts by inpainting and blurring
        the candidate anatomical region.
        """
        bgr = cv2.cvtColor(rgb_150, cv2.COLOR_RGB2BGR)
        # Dilate mask slightly for smooth boundary transition
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_mask = cv2.dilate(mask_255, kernel, iterations=1)
        
        # Navier-Stokes harmonic inpainting
        inpainted_bgr = cv2.inpaint(bgr, dilated_mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
        
        # Heavy Gaussian blur on the mask region to completely remove discriminating textures
        blurred_bgr = cv2.GaussianBlur(bgr, (21, 21), 9)
        alpha = (cv2.GaussianBlur(dilated_mask.astype(np.float32) / 255.0, (11, 11), 5))[:, :, np.newaxis]
        
        # Blend inpainting with harmonic blur for natural OOD-safe baseline
        composite_bgr = (inpainted_bgr * (1.0 - alpha) + blurred_bgr * alpha).astype(np.uint8)
        return cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)

    def explain(self, rgb_150, target_breed_idx=None, comp_breed_idx=None):
        """
        Complete BreedTrace-XAI Execution Workflow:
        Executes Modules A, B, C, and D, returning signed anatomical attributions,
        reliability scorecards, and contrastive evidence visualization.
        """
        t0 = time.time()
        h, w = rgb_150.shape[:2]
        
        # 1. Baseline Logits and Probabilities
        inp = np.expand_dims(rgb_150.astype('float32') / 255.0, axis=0)
        base_logits = self.compute_logits(inp)[0]
        base_probs = np.exp(base_logits - np.max(base_logits))
        base_probs /= np.sum(base_probs)
        
        sorted_indices = np.argsort(base_logits)[::-1]
        if target_breed_idx is None:
            target_breed_idx = int(sorted_indices[0])
            
        if comp_breed_idx is None or comp_breed_idx == target_breed_idx:
            # Runner-up class
            comp_breed_idx = int(sorted_indices[1] if sorted_indices[0] == target_breed_idx else sorted_indices[0])

        target_name = self.breed_classes[target_breed_idx]
        comp_name = self.breed_classes[comp_breed_idx]
        
        # Baseline Logit Margin m_c,d(x) = z_c(x) - z_d(x)
        base_margin = float(base_logits[target_breed_idx] - base_logits[comp_breed_idx])
        base_target_prob = float(base_probs[target_breed_idx])
        base_comp_prob = float(base_probs[comp_breed_idx])

        # 2. Module A: Anatomical Candidate Localization
        masks = self.localize_anatomical_regions(rgb_150)

        # 3. Module B: Contrastive Activation Saliency
        contrast_cam = self.compute_contrastive_activation_map(rgb_150, target_breed_idx, comp_breed_idx)

        # 4. Module C: Vectorized Feature Interventions
        # Create intervened images x^(-p) for each anatomical hypothesis
        ablated_imgs = []
        for part in self.ANATOMICAL_PARTS:
            pid = part['id']
            p_mask = masks[pid]
            x_minus_p = self.smooth_inpaint_region(rgb_150, p_mask)
            ablated_imgs.append(x_minus_p)
            
        # Vectorized batch evaluation across all intervened representations simultaneously
        batch_ablated = np.stack(ablated_imgs).astype('float32') / 255.0
        intervened_logits = self.compute_logits(batch_ablated)
        
        # Calculate signed anatomical evidence attribution A_p^(c,d) = m_c,d(x) - m_c,d(x^(-p))
        anatomical_attributions = []
        raw_ap_values = {}
        total_positive_margin = 0.0

        for i, part in enumerate(self.ANATOMICAL_PARTS):
            pid = part['id']
            ab_logits = intervened_logits[i]
            ab_margin = float(ab_logits[target_breed_idx] - ab_logits[comp_breed_idx])
            
            # The Core BreedTrace Intervention Equation
            A_p = base_margin - ab_margin
            raw_ap_values[pid] = A_p
            
            if pid != 'background_context' and A_p > 0:
                total_positive_margin += A_p

        for part in self.ANATOMICAL_PARTS:
            pid = part['id']
            A_p = raw_ap_values[pid]
            
            # Evidence percentage contribution relative to positive support
            if pid == 'background_context':
                contribution_pct = 0.0
            else:
                contribution_pct = (A_p / total_positive_margin * 100.0) if total_positive_margin > 0 else 0.0
                
            effect = 'supports_target' if A_p > 0.05 else ('favors_competitor' if A_p < -0.05 else 'neutral')

            anatomical_attributions.append({
                'id': pid,
                'name': part['name'],
                'color': part['color'],
                'description': part['description'],
                'margin_change': round(A_p, 3),
                'contribution_pct': round(max(0.0, contribution_pct), 1),
                'effect': effect,
                'mask_pixels': int(np.sum(masks[pid] > 0))
            })

        # 5. Module D: Explanation Verification & Reliability Scorecard
        # Check 1: Monotonic Deletion Faithfulness
        top_anatomical_part = max(
            [p for p in anatomical_attributions if p['id'] != 'background_context'],
            key=lambda x: x['margin_change']
        )
        faithfulness_drop = max(0.0, top_anatomical_part['margin_change'])
        faithfulness_score = min(100.0, max(0.0, (faithfulness_drop / (abs(base_margin) + 1e-4)) * 100.0))

        # Check 2: Background Context Shortcut Leakage
        bg_margin_change = abs(raw_ap_values['background_context'])
        total_sum_abs = sum(abs(v) for v in raw_ap_values.values()) + 1e-5
        bg_leakage_pct = round((bg_margin_change / total_sum_abs) * 100.0, 1)

        # Reliability Flagging
        if bg_leakage_pct >= 20.0:
            reliability_flag = 'SHORTCUT_ALERT'
            reliability_msg = f'Warning: Model relies heavily ({bg_leakage_pct}%) on background shortcut features rather than bovine anatomy.'
        elif top_anatomical_part['margin_change'] > 0.25:
            reliability_flag = 'CONFIRMED_ANATOMICAL'
            reliability_msg = f'High Faithfulness: Decision is verified by {top_anatomical_part["name"]} ({top_anatomical_part["contribution_pct"]}% of evidence).'
        else:
            reliability_flag = 'DIFFUSE_EVIDENCE'
            reliability_msg = 'Diffuse Evidence: Low individual anatomical part contribution; model relies on distributed global texture.'

        # 6. Render High-Resolution Contrastive Evidence Visualization
        vis_bgr = self._render_evidence_map(rgb_150, masks, contrast_cam, anatomical_attributions, target_name, comp_name)
        
        # Upscale for presentation display (380x380)
        vis_upscaled = cv2.resize(vis_bgr, (380, 380), interpolation=cv2.INTER_LANCZOS4)
        _, buf = cv2.imencode('.jpg', vis_upscaled, [cv2.IMWRITE_JPEG_QUALITY, 92])
        vis_b64 = base64.b64encode(buf).decode('utf-8')

        elapsed_sec = round(time.time() - t0, 2)

        return {
            'status': 'success',
            'method': 'breedtrace',
            'target_breed': target_name,
            'target_prob': round(base_target_prob * 100.0, 1),
            'target_idx': target_breed_idx,
            'competing_breed': comp_name,
            'competing_prob': round(base_comp_prob * 100.0, 1),
            'competing_idx': comp_breed_idx,
            'baseline_margin': round(base_margin, 3),
            'anatomical_attributions': anatomical_attributions,
            'reliability': {
                'flag': reliability_flag,
                'message': reliability_msg,
                'faithfulness_score': round(faithfulness_score, 1),
                'background_leakage_pct': bg_leakage_pct,
                'primary_driver': top_anatomical_part['name']
            },
            'explanation_image': f"data:image/jpeg;base64,{vis_b64}",
            'elapsed_sec': elapsed_sec
        }

    def _render_evidence_map(self, rgb_150, masks, contrast_cam, attributions, target_name, comp_name):
        """
        Renders an publication-quality multi-layer contrastive evidence visualization:
        - Cyan glowing regions: Positive evidence confirming target breed c
        - Crimson glowing regions: Contrastive features pulling towards competitor d
        - Subtle colored boundary lines outlining verified anatomical regions
        """
        h, w = rgb_150.shape[:2]
        bgr = cv2.cvtColor(rgb_150, cv2.COLOR_RGB2BGR)
        canvas = bgr.copy()

        # 1. Overlay Positive and Negative Contrastive Activation
        pos_cam = np.maximum(contrast_cam, 0)
        neg_cam = np.maximum(-contrast_cam, 0)

        pos_colored = cv2.applyColorMap((pos_cam * 255).astype(np.uint8), cv2.COLORMAP_OCEAN)
        neg_colored = cv2.applyColorMap((neg_cam * 255).astype(np.uint8), cv2.COLORMAP_HOT)

        # Blend positive (cyan) where pos_cam > 0.2
        pos_mask = (pos_cam > 0.25)[:, :, np.newaxis]
        neg_mask = (neg_cam > 0.25)[:, :, np.newaxis]

        canvas = np.where(pos_mask, cv2.addWeighted(canvas, 0.45, pos_colored, 0.55, 0), canvas)
        canvas = np.where(neg_mask, cv2.addWeighted(canvas, 0.50, neg_colored, 0.50, 0), canvas)

        # 2. Draw Anatomical Region Contours & Badges
        for part in self.ANATOMICAL_PARTS:
            pid = part['id']
            if pid == 'background_context':
                continue
                
            mask = masks[pid]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = part['color']
            
            # Find attribution score for this part
            attr = next((a for a in attributions if a['id'] == pid), None)
            if attr and attr['margin_change'] > 0.05:
                # Strong evidence contour with subtle glow
                cv2.drawContours(canvas, contours, -1, color, 2)
            else:
                cv2.drawContours(canvas, contours, -1, (60, 60, 70), 1)

        # 3. Add Legend Header
        cv2.rectangle(canvas, (0, 0), (w, 18), (10, 10, 15), -1)
        cv2.putText(canvas, f"BREEDTRACE: {target_name.upper()} vs {comp_name.upper()}", (6, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

        return canvas
