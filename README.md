# AI Fitness & Yoga Pose Analyzer

A computer-vision-based fitness and yoga analysis system that uses **MediaPipe Pose Landmarker** to detect human body landmarks, extract biomechanical features, analyze movement and posture, and provide exercise/pose-specific feedback.

The project is currently a **prototype under active development**, with individual analysis modules for strength exercises, yoga poses, and the Surya Namaskar sequence.

---

## Features

### Strength Exercises

The current prototype includes individual analysis modules for:

* Squats
* Lunges
* Push-ups
* Sit-ups
* Crunches
* Planks

Each exercise uses MediaPipe body landmarks and exercise-specific geometric conditions to analyze movement, identify movement states, count repetitions where applicable, and evaluate form.

### Yoga & Mobility

The current prototype includes:

* Tadasana
* Downward-Facing Dog
* Warrior I
* Warrior II
* Tree Pose
* Triangle Pose

Yoga analysis uses pose-specific landmark relationships, joint angles, distances, alignment and symmetry features.

### Surya Namaskar

Surya Namaskar is implemented as a **multi-pose sequence consisting of 12 individual asanas**.

The system analyzes the sequence as a collection of individual movements/poses rather than treating Surya Namaskar as a single static posture.

---

## How It Works

```text
Camera / Video
      │
      ▼
MediaPipe Pose Landmarker
      │
      ▼
33 Body Landmarks
      │
      ▼
Feature Extraction
      │
      ├── Joint Angles
      ├── Distances
      ├── Ratios
      ├── Symmetry
      └── Alignment
      │
      ▼
Exercise / Yoga-specific Analysis
      │
      ▼
Movement State / Pose Detection
      │
      ▼
Form Evaluation & Feedback
```

The project separates reusable computer-vision and geometry functionality from exercise- and yoga-specific analysis logic.

---

## Project Structure

```text
ai-fitness-cv/
│
├── core/
│   ├── drawing.py
│   ├── geometry.py
│   ├── landmarks.py
│   ├── landmarker.py
│   └── smoothing.py
│
├── yoga/
│   ├── __init__.py
│   ├── base_pose.py
│   ├── classifier.py
│   ├── hud.py
│   ├── logger.py
│   ├── timer.py
│   │
│   └── poses/
│       ├── __init__.py
│       ├── tadasana.py
│       ├── downward_dog.py
│       ├── warrior_i.py
│       ├── warrior_ii.py
│       ├── tree_pose.py
│       └── triangle.py
│
├── squat.py
├── lunge.py
├── pushup.py
├── situp.py
├── crunch.py
├── plank.py
│
├── tadasana.py
├── downward_dog.py
├── warrior_1.py
├── warrior_2.py
├── tree_pose.py
├── triangle_pose.py
├── surya_namaskar.py
│
├── main.py
├── yoga_pose_analyzer.py
├── extract_landmarks.py
├── test_exercises.py
├── test_yoga.py
│
├── squat_dataset.csv
├── lunge_dataset.csv
├── pushup_dataset.csv
├── situp_dataset.csv
├── crunch_dataset.csv
├── surya_namaskar_dataset.csv
├── yoga_dataset.csv
├── yoga_pose_dataset.csv
├── landmarks.csv
│
├── pose_landmarker_lite.task
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Core Architecture

The `core` package contains reusable functionality shared across the analysis modules.

### Landmark Detection

`core/landmarker.py` handles MediaPipe Pose Landmarker initialization and configuration.

### Landmark Definitions

`core/landmarks.py` contains landmark-related definitions and pose connections used by the project.

### Geometry & Feature Extraction

`core/geometry.py` provides numerical feature calculations based on detected landmarks.

Examples include:

* Joint angles
* Distances between landmarks
* Relative distances
* Ratios
* Body alignment
* Symmetry-related measurements

### Drawing

`core/drawing.py` provides utilities for visualizing the detected pose skeleton and landmarks.

### Smoothing

`core/smoothing.py` provides temporal smoothing functionality to reduce frame-to-frame fluctuations in detected features.

---

## Yoga Architecture

The yoga implementation is organized separately from the lower-level computer-vision functionality.

```text
YogaPoseAnalyzer
       │
       ▼
YogaPoseClassifier
       │
       ├── Tadasana
       ├── Downward Dog
       ├── Warrior I
       ├── Warrior II
       ├── Tree Pose
       └── Triangle Pose
       │
       ▼
Form Evaluation
       │
       ▼
Hold Timer / HUD / Dataset Logger
```

The yoga system contains reusable components for:

* Pose classification
* Form evaluation
* Hold timing
* Visual HUD
* Dataset logging
* Pose-specific form rules

Each yoga pose has its own analysis class derived from the common pose interface.

---

## Requirements

The project currently uses:

* Python 3.12
* MediaPipe
* OpenCV
* NumPy
* Pandas

Dependencies are pinned in:

```text
requirements.txt
```

Current requirements:

```text
mediapipe==1.0.1
opencv-contrib-python==5.0.0.93
numpy==2.5.2
pandas==3.0.5
```

The dependency configuration has been tested in a clean Python virtual environment.

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

### 2. Create a virtual environment

Linux/macOS:

```bash
python -m venv .venv
```

Windows:

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

#### Bash / Zsh

```bash
source .venv/bin/activate
```

#### Fish shell

```fish
source .venv/bin/activate.fish
```

#### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Verify the installation

```bash
python -c "import cv2, mediapipe, numpy, pandas; print('All dependencies OK')"
```

Expected output:

```text
All dependencies OK
```

---

## MediaPipe Model

The project uses the **MediaPipe Pose Landmarker Lite** model.

The required model file is already included in the repository:

```text
pose_landmarker_lite.task
```

No separate model download is required when using the repository as provided.

The Python modules expect the model file to be available at the path configured by the respective scripts.

---

## Running the Exercises

Activate the virtual environment first.

### Squat

```bash
python squat.py
```

### Lunge

```bash
python lunge.py
```

### Push-up

```bash
python pushup.py
```

### Sit-up

```bash
python situp.py
```

### Crunch

```bash
python crunch.py
```

### Plank

```bash
python plank.py
```

---

## Running Yoga Modules

### Tadasana

```bash
python tadasana.py
```

### Downward-Facing Dog

```bash
python downward_dog.py
```

### Warrior I

```bash
python warrior_1.py
```

### Warrior II

```bash
python warrior_2.py
```

### Tree Pose

```bash
python tree_pose.py
```

### Triangle Pose

```bash
python triangle_pose.py
```

### Surya Namaskar

```bash
python surya_namaskar.py
```

The application opens the camera, detects body landmarks and performs the corresponding exercise or pose analysis.

Press:

```text
q
```

to exit the camera window.

---

## Landmark & Dataset Extraction

The repository also contains a landmark extraction utility:

```bash
python extract_landmarks.py
```

This can be used to generate landmark-based numerical data for further analysis and dataset development.

The repository contains CSV datasets generated during development, including exercise and yoga-related landmark/feature data.

---

## Testing

Exercise-related tests can be run using:

```bash
python -m unittest test_exercises.py
```

Yoga-related tests can be run using:

```bash
python -m unittest test_yoga.py
```

The tests cover reusable geometry, landmark, smoothing, pose-analysis and exercise-related functionality implemented in the project.

---

## Current Development Status

### Strength Exercises

**Squats** are currently the most developed and tested exercise module.

The remaining exercise modules are implemented but may require additional testing and threshold tuning across different users and camera setups.

### Yoga

The listed yoga poses are implemented using pose-specific rules and feature conditions.

Further real-world testing is required to evaluate performance across:

* Different users
* Different body proportions
* Different camera distances
* Different camera heights
* Different camera orientations
* Lighting conditions
* Partial landmark visibility
* Different execution styles

### Threshold Calibration

The current system primarily uses geometric features and rule-based thresholds.

As more testing data is collected, feature selection and detection thresholds may be refined to improve robustness and reduce false detections.

---

## Limitations

The current prototype has several practical limitations.

* Detection quality depends on the visibility of the user's body.
* Camera placement can affect landmark accuracy.
* Occlusion may cause incorrect landmark detection.
* Threshold-based analysis may not generalize equally well to every user.
* Different body proportions can produce different feature distributions.
* Lighting and background conditions may influence pose detection.
* The current system has not yet been extensively validated across a large and diverse user dataset.

These limitations are part of the current prototype and are areas for future improvement.

---

## Future Improvements

Potential future development includes:

* Larger and more diverse training datasets
* Automatic threshold calibration
* More robust pose classification
* Improved temporal movement analysis
* Personalized form thresholds
* Additional exercises and yoga poses
* More detailed corrective feedback
* Rep-quality scoring
* Exercise session history
* Performance analytics
* Integration into a unified fitness application
* Improved robustness under different camera orientations and environments

---

## Development Philosophy

The project follows a modular approach where reusable landmark processing, geometry, smoothing and visualization components are separated from exercise- and pose-specific analysis.

This allows new exercises and yoga poses to be added without rebuilding the entire computer-vision pipeline.

---

## Prototype Disclaimer

This project is an experimental computer-vision prototype intended for fitness and yoga analysis.

The generated form feedback should not be considered a substitute for professional fitness, physiotherapy or medical guidance.
