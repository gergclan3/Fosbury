# High-jump form review

A local Streamlit app for reviewing a single high-jump attempt with pretrained Ultralytics YOLO11n-pose. It samples the clip, tracks the largest detected person, estimates takeoff and bar-clearance timing, overlays pose landmarks, and exports a CSV of observations. Video and inference stay on the laptop; the first launch downloads the model weights.

## Run locally

Use Python 3.10–3.12. From this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload one attempt per video with the athlete visible throughout. The first model download needs internet access. CPU inference works but can be slow; a supported NVIDIA GPU can speed it up.

## What the analysis means

- **Takeoff** is estimated at the start of a sustained interval where both ankles are elevated relative to the clip's inferred ground level. Running strides, camera movement, occlusion, or an incomplete clip can cause errors.
- **Clearance** uses a persistent near-horizontal line as a possible bar, then finds the tracked hip center nearest that line during flight. If no bar is detected, the app reports the flight apex as a **low-confidence proxy**, not a verified clearance time.
- **Knee-drive hold** is a 2D observation of how long either knee appears above the hip shortly before estimated takeoff. The 0.15-second comparison is a provisional heuristic and is not a validated coaching standard.
- The standard 17-point pose model does not locate toes, so plant-foot direction is not measured. Single-view 2D footage also cannot establish true 3D lean or hip extension. Use camera views where the full athlete and crossbar remain visible, and review estimates with a coach.

This is an early screening tool, not a validated coaching or injury-prevention system. Automatic event estimates are best-effort and should be checked against the source footage.

## Tests

```powershell
python -m unittest discover -s tests -v
```
