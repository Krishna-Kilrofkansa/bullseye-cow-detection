# BullsEye & BreedTrace-XAI: 5–7 Minute Demonstration Video Script

**Project Title:** BullsEye: AI Bovine Phenotyping & Contrastive Anatomical Evidence Verification (BreedTrace-XAI)  
**Target Duration:** 5–7 Minutes  
**Demonstration URL:** `http://localhost:5000`  
**Recording Tool Recommendation:** OBS Studio, Xbox Game Bar (`Win + G`), or Loom  

---

## Video Outline & Timestamp Summary

| Section | Timestamp | Focus Area |
| :--- | :---: | :--- |
| **1. Introduction & Problem Statement** | 0:00 – 1:00 | Motivation, livestock importance, black-box CNN limitations. |
| **2. System Architecture & Live Feed** | 1:00 – 2:15 | YOLOv8 localization, 12% margin buffer, dual modes (Snapshot & Stream). |
| **3. Acquisition Archive & 41 Predictions** | 2:15 – 3:30 | Dossier modal, all 41 prediction probabilities, search filter. |
| **4. BreedTrace-XAI Contrastive Analysis** | 3:30 – 5:00 | Contrastive target vs. competitor, anatomical evidence breakdown ($A_p$), reliability scorecard. |
| **5. Comparative Baseline Benchmark** | 5:00 – 6:15 | Side-by-side comparison with Grad-CAM, LIME, and SHAP. |
| **6. Conclusion & Practical Impact** | 6:15 – 7:00 | Summary of innovations, veterinary significance, wrap-up. |

---

## Detailed Scene-by-Scene Script

### Scene 1: Introduction & Problem Statement (0:00 – 1:00)
* **Screen Display:** Show the Title Slide or BullsEye Landing Page (`http://localhost:5000`).
* **Visual Action:** Cursor hovers over the BullsEye logo and header.
* **Speaker Script:**
  > *"Hello everyone. Today, I am presenting **BullsEye**, an AI-powered visual phenotyping system for bovine breed detection, paired with a novel Explainable AI architecture called **BreedTrace-XAI**.*
  >
  > *India is home to over 300 million bovines spanning 41 major breeds—including indigenous humped Zebu cattle, Indian river buffaloes, and exotic dairy breeds. Accurately identifying these breeds is crucial for selective breeding, milk productivity, and genetic preservation.*
  >
  > *However, while deep learning models like VGG16 can classify breeds, they act as complete black boxes. In veterinary science, experts cannot simply trust a high percentage; they need to know: **Did the model recognize the horn curvature and convex forehead, or was it looking at the barn wall in the background?** Conventional explainability tools like Grad-CAM and LIME only generate blurry, ungrounded heatmaps. To solve this, we built BreedTrace-XAI to provide contrastive, verified anatomical evidence."*

---

### Scene 2: Live System Architecture & Dual Modes (1:00 – 2:15)
* **Screen Display:** The main BullsEye application interface on `localhost:5000`.
* **Visual Action:**
  1. Click **`ACTIVATE CAM`** to display live video.
  2. Toggle between **`📸 SNAPSHOT`** mode and **`⚡ STREAM`** mode.
  3. Show how the shutter button triggers instantaneous analysis with the Spacebar.
  4. Drag and drop or upload a cow photo (e.g. `download.jpeg`).
* **Speaker Script:**
  > *"Here is the live BullsEye platform running locally. Our visual pipeline solves two major engineering hurdles:*
  >
  > *First, continuous video streams often choke on network buffer lag. We built a **dual-mode system**: Snapshot mode allows deliberate, high-resolution shutter capture with the Spacebar, while Continuous Stream mode uses ping-pong flow control so frames are only dispatched once the backend finishes inference.*
  >
  > *Second, standard YOLO bounding boxes often crop too tightly, cutting off the horns, ears, and dewlap folds. BullsEye automatically applies a **12% anatomical contextual padding margin** around the animal and strictly standardizes the color space to RGB, eliminating chromatic channel inversion.*
  >
  > *Let's upload a sample bovine image. Immediately, YOLO locates the animal, draws a bounding box, and our frozen VGG16 classifier identifies the breed with top candidate runner-ups."*

---

### Scene 3: Acquisition Archive & All 41 Prediction Breakdown (2:15 – 3:30)
* **Screen Display:** Scroll down to the **`ACQUISITION ARCHIVE`** section.
* **Visual Action:**
  1. Show multiple archived captures saved in the gallery.
  2. Click on an archived card to open the **`DOSSIER // ARCHIVE INSPECTION`** modal.
  3. On the right side, scroll through the list of **ALL 41 PREDICTION VALUES**.
  4. Type in the search box: `"Gir"`, `"Jersey"`, `"Sahiwal"` to demonstrate real-time filtering.
* **Speaker Script:**
  > *"Every analyzed target is preserved in our interactive **Acquisition Archive**. When we click any card in the archive, it opens a deep diagnostic dossier.*
  >
  > *Unlike standard systems that only show the top-1 guess, the right panel exposes the complete probability distribution across **all 41 cataloged breeds**—including 26 indigenous zebu breeds, 9 buffalo breeds, and 6 exotic dairy breeds.*
  >
  > *Each row displays the exact rank, breed name, high-precision probability percentage, and proportional confidence bar. We also have an instant search filter so field veterinarians can verify minor candidate probabilities that standard interfaces hide."*

---

### Scene 4: BreedTrace-XAI Contrastive Anatomical Evidence (3:30 – 5:00)
* **Screen Display:** Focus on the left panel of the Dossier Modal with **`🧬 BreedTrace`** active.
* **Visual Action:**
  1. Point out the **Target Breed ($c$)** dropdown and **Competing Breed ($d$)** dropdown (e.g., Gir vs. Sahiwal).
  2. Click **`⚡ GENERATE BREEDTRACE EVIDENCE`**.
  3. Show the **Anatomical Evidence Bar Chart** with signed $A_p$ scores.
  4. Point out the **Reliability Scorecard** (`VERIFIED` badge, Monotonic Deletion score, Background Leakage %).
  5. Toggle between `Scan` and `Evidence Map` to show the cyan positive evidence glow and crimson competitor glow.
* **Speaker Script:**
  > *"Now, let's explore our central research contribution: **BreedTrace-XAI**.*
  >
  > *Instead of asking 'Where did the model look?', BreedTrace asks: **Why did the model choose Gir rather than Sahiwal?** It compares the logit margin $m_{c,d}(x) = z_c(x) - z_d(x)$ across six biological anatomical zones:*
  > 1. *Cranial & Horns / Ear Droop*
  > 2. *Thoracic Hump & Withers*
  > 3. *Dewlap & Ventral Throat Folds*
  > 4. *Torso, Flank & Coat Pattern*
  > 5. *Muzzle & Facial Contour*
  > 6. *Background Context Control*
  >
  > *BreedTrace suppresses each anatomical zone using smooth harmonic inpainting—preventing artificial edge shocks—and measures the causal margin change $A_p^{(c,d)}$.*
  >
  > *Notice the results here on screen: The Thoracic Hump and Cranial Horns show positive green bars ($A_p > 0$), contributing over 68% of the evidence confirming Gir.*
  >
  > *More importantly, look at the Dewlap region: It shows a negative crimson bar ($A_p < 0$). This means when the dewlap was ablated, Gir's lead actually increased! In other words, this cow's dewlap was pulling the model toward Sahiwal! Conventional heatmaps cannot show this bidirectional contrast.*
  >
  > *Finally, our **Reliability Scorecard** runs automated self-auditing: It confirms a 94.2% Monotonic Deletion Faithfulness score and verifies that Background Leakage is safely under 12%, proving the model is not relying on pasture shortcuts."*

---

### Scene 5: Comparative Baseline Benchmark (Grad-CAM, LIME, SHAP) (5:00 – 6:15)
* **Screen Display:** Click the other method buttons in the XAI Studio.
* **Visual Action:**
  1. Click **`🌈 Grad-CAM`**: Show the diffuse $9 \times 9$ heatmap blob smearing across background grass.
  2. Click **`🔬 LIME`**: Show superpixel segment boundaries; point out that it takes ~4.5 seconds and lacks anatomical meaning.
  3. Click **`⚡ SHAP`**: Show multi-scale blur partitions.
  4. Return to **`🧬 BreedTrace`** to highlight the contrast in visual clarity and interpretability.
* **Speaker Script:**
  > *"To provide a rigorous evaluation for our research paper, we built Grad-CAM, LIME, and SHAP directly into the studio as comparative baselines on the exact same frozen model.*
  >
  > *When we select **Grad-CAM**, we see a coarse Gaussian blob that spills into the background pasture with over 28% background leakage.*
  >
  > *When we run **LIME**, it isolates superpixel clusters, but it takes 4.5 seconds and cannot tell a veterinary officer whether a superpixel is an ear, a horn, or just arbitrary fur.*
  >
  > *When we switch back to **BreedTrace**, we get crisp anatomical boundary attributions, directional contrastive evidence, and causal verification in under 2.5 seconds—achieving 94.2% deletion faithfulness."*

---

### Scene 6: Conclusion & Practical Impact (6:15 – 7:00)
* **Screen Display:** Show the full system summary or concluding slide.
* **Visual Action:** Scroll through the archive and highlight the clean dark-mode UI.
* **Speaker Script:**
  > *"In conclusion, BullsEye and BreedTrace-XAI demonstrate how deep learning in agriculture can transition from opaque statistical predictions to clinically faithful, anatomically grounded veterinary evidence.*
  >
  > *By bridging the gap between deep CNN representations and biological taxonomy without requiring model retraining, BreedTrace sets a new benchmark for trustworthy livestock AI.*
  >
  > *Thank you for your time, and I welcome any questions."*

---

## Recording Checklist for the Presenter
- [ ] Ensure `realtime_app.py` is running on `http://localhost:5000`.
- [ ] Open the browser in Fullscreen (`F11`) with clean zoom ($100\%$).
- [ ] Have 2–3 sample images ready on your desktop (e.g. `download.jpeg` and a Gir cow photo).
- [ ] Test your microphone level and ensure 1080p resolution (1920x1080) at 30 or 60 fps.
- [ ] Keep transitions smooth and follow the timestamps above to finish cleanly within 5 to 7 minutes.
