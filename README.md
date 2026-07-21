---
![](pumpkin_gif.gif)
# PumpKin: A machine learning package for automatically tracking pharyngeal pumping kinematics in freely moving *C. elegans*
**Read the manuscript at [PLoS Computational Biology]([https://doi.org/10.1101/2025.06.19.660198](https://doi.org/10.1371/journal.pcbi.1014489)).**

PumpKin is designed to automatically track the pharyngeal pumping rate in individual freely moving *C. elegans*, but may be adapted to track the motion of structures in other organisms as well! This package is built using [EZ-FRCNN](https://github.com/lu-lab/ez-frcnn): a user-friendly implementation of the popular Faster Region-based Convolutional Neural Network (Faster R-CNN) originally developed by [Ren et al](https://ieeexplore.ieee.org/document/7485869). To get started, continue reading below.

## Features
- Jupyter Notebook-based for ease-of-use
- Simple in-house annotation tools
- **Fast FRCNN training**: around 1 hour on a single GPU for a standard dataset
- **Fast FRCNN inferencing**: around 15 FPS on a single GPU
- Detailed documentation and tutorials for use

## Contents
1. [Features](#features)
2. [Contents](#contents)
3. [Requirements](#requirements)
    - [Installation (Windows)](#installation-windows)
4. [Getting Started](#getting-started)
5. [Documentation](#documentation)
6. [Data Availability](#data-availability)
7. [References](#references)

## Requirements
We provide instructions for installing PumpKin (with EZ-FRCNN) on Windows below. While a GPU is **highly recommended** to use PumpKin and EZ-FRCNN, it is not required.

## Installation (Windows only)
1. Install [Docker for Windows](https://docs.docker.com/desktop/setup/install/windows-install/).
2. Launch Docker Desktop.
3. [Download](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/lu-lab/PumpKin) or clone [this repository](https://github.com/lu-lab/PumpKin).
4. Extract the contents of the ZIP file downloaded in the last step to a folder of your choice (SKIP if you used `git clone`).
5. Open the EZ-FRCNN folder and double-click `ez-frcnn.bat` to launch EZ-FRCNN. OR for a **more user-friendly experience**, double-click `ez-frcnnPane.bat` to launch the GUI.

## Getting Started
### Step 1: Follow the directions on [ez-frcnn.com](www.ez-frcnn.com) to train a model to track the *pharyngeal bulb* and inference your videos.
1. Be sure to use high-quality (at minimum 1080p) images so that the pharyngeal bulb (and grinder) are clearly labelable!
2. We recommend using 100-200 images for training, depending on the variation of your recordings. You can also train separate models for different recording conditions if one model is struggling to generalize.
3. We provide an example model at [OSF](https://osf.io/79hfv) under `/models/bulb_tracking_EXAMPLE.pth`, but we **highly recommend** either training your own model or retraining the example model as a starting point.

### Step 2: Use the `cropROI.ipynb` Jupyter Notebook to crop each of your videos to a 250x250p area centered on the pharyngeal bulb.
1. If using a different folder structure, update `VID_DIR` and `OUT_DIR` accordingly.
2. Change `video_type` to the suffix that matches your videos (e.g., `.wmv`, `.mp4`, `.avi`, etc.).
3. This script is designed to locate all videos and crop them, but if you only wish to crop a selection of them, edit `videos` to be a list of the specific videos you would like cropped:
```py
videos = ['example_video_name_1', 'example_video_name_2', ...]
```
4. All cropped videos are saved in `./videos/cropped` by default and use the same suffix as the original videos.

### Step 3: Follow the directions on [ez-frcnn.com](www.ez-frcnn.com) to train a model to track the *grinder* and inference your videos.
1. We recommend using 100-150 images for training, depending on the variation of your recordings. You can also train separate models for different recording conditions if one model is struggling to generalize.
2. We provide an example model at [OSF](https://osf.io/79hfv) under `/models/grinder_tracking_EXAMPLE.pth`, but we **highly recommend** either training your own model or retraining the example model as a starting point.

### Step 4: Use the `saveCoMs.ipynb` Jupyter Notebook to calculate and save the center of mass (CoM) of the tracked grinders for each of your videos.
These are used in PumpKin to sample a uniform number of points in the grinder for motion tracking.

1. If using a different folder structure, update `VID_DIR` and `OUT_DIR` accordingly.
2. Change `video_type` to the suffix that matches your videos (e.g., `.wmv`, `.mp4`, `.avi`, etc.).
3. This script is designed to locate all videos and calculate the CoM, but if you only wish to track a selection of them, edit `videos` to be a list of the specific videos you would like tracked.
4. All CoMs are saved in `./outputs/coms` in CSV format by default.

### Step 5: Use the `pumpkin.ipynb` Jupyter Notebook to run PumpKin on your videos and obtain pumping rate estimates.
1. If using a different folder structure, update `VID_DIR` and `OUT_DIR` accordingly.
2. Change `video_type` to the suffix that matches your videos (e.g., `.wmv`, `.mp4`, `.avi`, etc.).
3. This script is designed to locate all videos and process them, but if you only wish to process a selection of them, edit `videos` to be a list of the specific videos you would like processed.
4. PumpKin saves the detected pump times and continuous pumping rate estimate to `./outputs/pumpkin` in CSV format by default.

## Documentation
Full documentation for PumpKin is available at [https://erinshappell.github.io/pumpkin-docs/](https://erinshappell.github.io/pumpkin-docs/)

## Data Availability
Example Faster R-CNN models and all videos used for validation may be found on [OSF](https://osf.io/79hfv).

## References
EZ-FRCNN is an implementation of Faster R-CNN, an algorithm developed by [Ren et al](https://ieeexplore.ieee.org/document/7485869). 

Motion compensation and optical flow code is adapted from code written by [Isaac Berrios](https://github.com/itberrios/CV_projects/tree/main).

EZ-FRCNN was written by Jacob Wheelock and Erin Shappell for Lu Lab, 2025.

PumpKin was written by Erin Shappell for Lu Lab, 2025.

