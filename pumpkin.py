################################################################################
# pumpkin.py
# Written by Erin Shappell for Lu Lab
# 
# This module contains all functions used to generate a continuous pumping rate
# estimate from an FRCNN-tracked grinder in freely moving C. elegans.
#
################################################################################
# Imports
import re
import os
import cv2
import numpy as np
from scipy.signal import iirfilter, find_peaks, filtfilt
from scipy.ndimage import gaussian_filter1d

################################################################################
def sample_dark_pts(frame, point, box_size=(15, 15), grid_size=(3, 3)):
    """
    Sample the darkest pixels from a grid of subregions within a box centered at a given point.

    Inputs:
        frame (ndarray):     Input video frame (color image as a NumPy array).
        point (tuple):       (x, y) coordinates representing the center of the sampling box.
        box_size (tuple):    (width, height) of the box around the center within which to sample.
                             Default is (15, 15).
        grid_size (tuple):   (rows, cols) specifying how the box is divided into subregions.
                             Default is (3, 3).

    Output:
        list: A list of (x, y) tuples corresponding to the darkest pixel in each subregion,
              adjusted to the coordinates of the original frame.
              Returns an empty list if the region is invalid.
    """
    cx, cy = map(int, point)  # Ensure integer coordinates
    w, h = box_size  # Width and height of the bounding box

    # Compute bounding box coordinates
    x1, y1 = max(cx - w // 2, 0), max(cy - h // 2, 0)
    x2, y2 = min(cx + w // 2, frame.shape[1]), min(cy + h // 2, frame.shape[0])

    # Crop the region of interest (ROI)
    roi = frame[y1:y2, x1:x2]  

    # Convert to grayscale if the image is color--if invalid, return empty list
    if roi is not None: roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else: return []

    # Grid division
    rows, cols     = grid_size
    step_y, step_x = (y2 - y1) // rows, (x2 - x1) // cols  # Compute step size
    sampled_points = []
    
    # Loop over grid cells
    for i in range(rows):
        for j in range(cols):
            # Define cell boundaries
            y_start, y_end = y1 + i * step_y, y1 + (i + 1) * step_y
            x_start, x_end = x1 + j * step_x, x1 + (j + 1) * step_x

            # Extract the sub-region
            cell = roi[(i * step_y):(i + 1) * step_y, (j * step_x):(j + 1) * step_x]

            # Ensure cell has valid size before processing
            if cell.size == 0: continue

            # Find the darkest point in this cell
            min_val, _, min_loc, _ = cv2.minMaxLoc(cell)

            # Adjust coordinates to global frame
            dark_x = x_start + min_loc[0]
            dark_y = y_start + min_loc[1]

            sampled_points.append((dark_x, dark_y))

    return sampled_points

################################################################################
def calc_motion_vecs(dark_points_t1, dark_points_t2, threshold=200):
    """
    Calculate motion vectors by matching dark points between two consecutive frames.

    Inputs:
        dark_points_t1 (list): List of (x, y) coordinates from frame t-1.
        dark_points_t2 (list): List of (x, y) coordinates from frame t.
        threshold (float):     Maximum distance allowed to consider two points as a match. Default is 200.

    Output:
        ndarray: Array of motion vectors [(dx, dy), ...] representing the displacement
                 of matched points from t-1 to t. Only vectors below the threshold are included.
    """
    # Initialize array of motion vectors
    motion_vectors = []

    for pt1 in dark_points_t1:
        # Calculate distances from pt1 to all points in t2
        distances = np.linalg.norm(np.array(dark_points_t2) - np.array(pt1), axis=1)
        
        # Find the closest point in t2
        min_distance = np.min(distances)
        if min_distance <= threshold:
            # Get the index of the closest point
            closest_idx = np.argmin(distances)
            pt2 = dark_points_t2[closest_idx]
            
            # Calculate the motion (difference in coordinates)
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            motion_vectors.append((dx, dy))

    return np.array(motion_vectors)

################################################################################
def find_troughs_and_peaks(signal, distance=10, height=[0.1,5]):
    """
    Detect peaks (local maxima) that are followed by troughs (local minima) within a specified distance.

    Inputs:
        signal (ndarray):     1D array of signal values.
        distance (int):       Maximum number of frames allowed between a peak and its corresponding trough.
                              Default is 10.
        height (list):        Two-element list specifying the minimum and maximum height of peaks/troughs.
                              Default is [0.1, 5].

    Outputs:
        tuple:
            - troughs (ndarray): Indices of troughs that follow valid peaks within the specified distance.
            - peaks (ndarray):   Indices of peaks that are followed by a trough within the specified distance.
    """
    # Find troughs (invert signal to detect minima)
    troughs, _ = find_peaks(-signal, distance=distance/2, height=(height[0], height[1]))
    
    # Find peaks
    peaks, _ = find_peaks(signal, distance=distance/2, height=(height[0], height[1]))

    # Keep only peaks that come after a trough
    valid_peaks   = []
    valid_troughs = []
    
    for peak in peaks:
        # Find the first trough after the peak
        following_troughs = troughs[troughs > peak]
        
        # Remove any already matched troughs to ensure exclusivity
        following_troughs = np.array([t for t in following_troughs if t not in valid_troughs])
            
        if len(following_troughs) > 0 and np.abs(following_troughs[0] - peak) < distance:
            valid_troughs.append(following_troughs[0])
            valid_peaks.append(peak)
            
    return np.array(valid_troughs), np.array(valid_peaks)

################################################################################
def make_bin_edges(bin_width, dt, total_time):
    """
    Generate timestamp edges to segment a time series into bins of fixed width.

    Inputs:
        bin_width (float): Desired width of each bin in seconds.
        dt (float):        Time step between consecutive samples in seconds.
        total_time (float): Total duration of the signal in seconds.

    Outputs:
        ndarray: Array of timestamps representing the edges of each bin, starting at 0 and
                 covering the total_time in increments of bin_width.
    """
    # Compute the timesteps (same code as previous)
    ts = np.arange(0, total_time, dt)
    
    # Num timesteps
    n_timesteps = np.prod(ts.shape)
    
    # Warn if binsize doesn't divide the timestep evenly
    if (bin_width % dt) >= dt :
        print("Warning: bin_width doesn't evenly divide the timestep")
        print(bin_width % dt)
    
    # How many timesteps per bin?
    steps_per_bin = np.around(bin_width / dt)
    
    # Compute the bin edges using np.arange
    #   remember that arange is an open interval at the end
    #   so add one to the number of timesteps
    bin_edges = np.arange(0, n_timesteps+1, steps_per_bin) * dt
    return bin_edges

################################################################################
def motion_comp(prev_frame, curr_frame, num_points=500, points_to_use=500):
    """
    Estimate the motion transformation matrix between two sequential image frames.
    Contains code adapted from [https://github.com/itberrios/CV_projects/tree/main]

    Inputs:
        prev_frame (ndarray):   The first image frame (RGB).
        curr_frame (ndarray):   The second sequential image frame (RGB).
        num_points (int):       Number of feature points to detect in the first frame.
        points_to_use (int):    Number of matched points to use for motion estimation.

    Outputs:
        A (ndarray or None):    Estimated affine transformation matrix (2x3) or None if estimation fails.
        prev_points (ndarray):  Feature points detected in the previous frame.
        curr_points (ndarray):  Corresponding matched points in the current frame.
    """
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

    # Get features for first frame
    features = cv2.goodFeaturesToTrack(prev_gray, num_points, qualityLevel=0.01, minDistance=10)
    
    # If no feature points exist, exit
    if features is None: return None, [], []

    # Get matching features in next frame with Sparse Optical Flow Estimation
    matched_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None)

    # Reformat previous and current feature points
    prev_points = features[status==1]
    curr_points = matched_features[status==1]

    # Subsample number of points so we don't overfit
    if points_to_use > prev_points.shape[0]: points_to_use = prev_points.shape[0]

    index = np.random.choice(prev_points.shape[0], size=points_to_use, replace=False)
    prev_points_used = prev_points[index]
    curr_points_used = curr_points[index]

    # If no feature points exist, exit
    if len(prev_points_used) == 0 or len(curr_points_used) == 0: return None, [], []

    # Find transformation matrix from frame 1 to frame 2
    A, _ = cv2.estimateAffine2D(prev_points_used, curr_points_used, method=cv2.RANSAC)

    return A, prev_points, curr_points

################################################################################
def get_grinder_motion(prev_frame, curr_frame, prev_com, curr_com):
    """
    Detects matched grinder keypoints and computes motion vectors between two frames.
    Contains code adapted from [https://github.com/itberrios/CV_projects/tree/main]

    Inputs:
        prev_frame (ndarray): Previous video frame.
        curr_frame (ndarray): Current video frame.
        prev_com (tuple):     Centroid (x, y) of the grinder in the previous frame.
        curr_com (tuple):     Centroid (x, y) of the grinder in the current frame.

    Outputs:
        prev_grinder_pts (ndarray): Transformed grinder points from previous frame.
        curr_grinder_pts (ndarray): Grinder points sampled in current frame.
        motion (ndarray):           Motion vectors between matched points.
        magnitude (ndarray):        Magnitude of motion vectors.
        angle (ndarray):            Angles (radians) of motion vectors.
    """

    ### Get affine transformation for frame alignment
    # Get frame info
    h, w, _ = prev_frame.shape

    # Get affine transformation matrix for motion compensation between frames
    A, prev_pts, curr_pts = motion_comp(prev_frame, curr_frame, num_points=10000, points_to_use=5000)
    
    ### Transform previous frame's points using affine transformation
    # First, check that A was obtained correctly
    if A is None: return [],[],[],[],[]
        
    # Get transformed grinder points from the previous frame using A
    A = np.vstack((A, np.zeros((3,)))) # get 3x3 matrix to transform points
    prev_grinder_pts = sample_dark_pts(prev_frame, prev_com)
    curr_grinder_pts = sample_dark_pts(curr_frame, curr_com)
    
    # Check if points aren't found
    if not prev_grinder_pts or not curr_grinder_pts: return [],[],[],[],[] 
    
    # Compensate the previous frame's points for motion
    comp_pts = np.hstack((prev_grinder_pts, np.ones((len(prev_grinder_pts), 1)))) @ A.T
    comp_pts = comp_pts[:, :2]

    ### Obtain motion information about the grinder points
    motion    = calc_motion_vecs(comp_pts, curr_grinder_pts)
    if len(motion) < 2: return [],[],[],[],[]
    
    magnitude = np.linalg.norm(motion, ord=2, axis=1)
    angle     = np.arctan2(motion[:, 0], motion[:, 1]) 
    
    return comp_pts, curr_grinder_pts, motion, magnitude, angle
    
################################################################################
def process_video(video_path=None, grinder_coms=None):
    """
    Processes a video to compute motion vectors between grinder positions frame-by-frame.

    Inputs:
        video_path (str):       Path to the input video file.
        grinder_coms (ndarray): Array of grinder center-of-mass positions per frame, shape (n_frames, 2).

    Outputs:
        mags (list of ndarray): List of arrays containing motion vector magnitudes per frame.
        angs (list of ndarray): List of arrays containing motion vector angles (radians) per frame.
    """
    ### Check that a video path and grinder CoMs were both provided
    if video_path is None:   raise ValueError("No video path was provided.")
    if grinder_coms is None: raise ValueError("No grinder CoMs were provided.")
    
    ### Load video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    vid        = cv2.VideoCapture(video_path)
    width      = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps        = vid.get(cv2.CAP_PROP_FPS)
    tot_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    ### Initialize lists for motion information and labeled frames
    mags, angs = [], []
    saved_pts, frames = [], []

    # Initialize frames and coms
    success, prev_frame = vid.read()
    prev_com = grinder_coms[0]
    i = 1 
    while success and i < grinder_coms.shape[0]:
        success, curr_frame = vid.read()
        curr_com   = grinder_coms[i] if success else None
        curr_pts   = []

        if success:
            # If a grinder was detected, calculate the motion
            if any(prev_com) and any(curr_com):
                # Get the distances between the grinder points in the previous and current frames
                prev_pts, curr_pts, motion, mag, ang = get_grinder_motion(prev_frame, curr_frame, prev_com, curr_com)

                # Save the motion information from the current frame
                mags.append(mag)
                angs.append(ang)

                # Draw detected motion vectors
                if curr_pts and prev_pts.shape == motion.shape and save: 
                    plot_frame = plot_vecs(prev_frame.copy(), np.hstack([prev_pts,prev_pts+motion]))

            # If a grinder was NOT detected, fill with blanks and continue
            else:
                mags.append([])
                angs.append([])

            # Save previous frame and grinder CoM for next iteration
            prev_frame = curr_frame.copy()
            prev_com   = curr_com

        # Update iterator
        i += 1

        # Save the cluster locations from each frame
        saved_pts.append(curr_pts)
        
    return mags, angs

################################################################################
def get_strain_cond(video_name):
    """
    Parses a video filename to extract the strain and condition identifiers.

    Inputs:
        video_name (str): Full name of the video file (e.g., 'n2_ff_00001.wmv' or 'n2_ff_00001_cropped.wmv').

    Outputs:
        strain (str):    Extracted strain identifier from the filename (e.g., 'n2').
        condition (str): Extracted condition identifier from the filename (e.g., 'ff').

    Raises:
        ValueError: If the filename does not match the expected format 'STRAIN_CONDITION_#####.ext'.
    """
    base = os.path.splitext(video_name)[0]  # Remove extension, e.g., '.wmv'
    
    # Match STRAIN_CONDITION_##### optionally followed by '_cropped'
    match = re.match(r'^([^_]+)_([^_]+)_\d+(?:_cropped)?$', base)
    
    if not match:
        raise ValueError("Input string must be in format 'STRAIN_CONDITION_#####.ext'")
    
    strain, condition = match.group(1), match.group(2)
    return strain, condition

################################################################################
def get_filtered_motion(strain_name=None, cond_name=None, mags=None, angs=None, fps=20):
    """
    Calculates the signed and filtered motion magnitude of grinder points across frames.

    Inputs:
        strain_name (str):      Name of the genetic strain.
        cond_name (str):        Name of the experimental condition.
        mags (list of ndarray): List of motion magnitude arrays per frame.
        angs (list of ndarray): List of motion angle arrays per frame (radians).
        fps (float):            Sampling frequency of the video frames (default is 20).

    Outputs:
        signed_mags_filt (ndarray): Low-pass filtered signed average motion magnitude signal.
        fc (float):                 Cutoff frequency used for filtering based on the strain and condition.
    """
    ### Check that a strain name, condition name, and motion information were all provided
    if strain_name is None: raise ValueError("No strain name was provided.")
    if cond_name is None:   raise ValueError("No condition name was provided.")
    if mags is None:        raise ValueError("No magnitudes were provided.")
    if angs is None:        raise ValueError("No angles were provided.")
        
    ### Determine if any of the grinder points have moved significantly
    # Store magnitudes
    max_mag  = 5 # ignore motion vectors exceeding this threshold
    n_frames = len(mags)
    avg_mags = np.zeros(n_frames) # unsigned magnitude of motion
    avg_angs = np.zeros(n_frames) # average angle of motion
    signs    = np.zeros(n_frames) # negative/positive motion
    avg_signed_mags = np.zeros(n_frames) # signed magnitude of motion

    # Main loop for calculating signed motion magnitude
    for i in range(n_frames):
        # Save the average angle values (used to determine sign)
        if len(angs[i]) > 0: avg_angs[i] = np.mean(angs[i])
        else: avg_angs[i] = 0

        # Determine the sign based on the angle value
        if avg_angs[i] > 0:   signs[i] = 1
        elif avg_angs[i] < 0: signs[i] = -1

        # Save the largest magnitude + average magnitude from each frame
        if len(mags[i]) > 0 and len([x for x in mags[i] if x < max_mag]) > 0: 
            avg_mags[i] = np.mean([x for x in mags[i] if x < max_mag])
        else: avg_mags[i] = 0

        # Save the signed magnitude
        avg_signed_mags[i] = avg_mags[i]*signs[i]

    ### Build low-pass filter and apply to signal
    ### NOTE: if using a new strain, you will need to add it as follows:
    ### elif strain_name == 'new_strain' and (cond_name == 'ff' or cond_name == 'sf'): 
    ###     fc = 2*expected_max_rate_for_new_strain
    # Use a cutoff frequency (fc) equal to 2*max_expected_pumping_rate
    if strain_name == 'n2' and (cond_name == 'ff' or cond_name == 'sf'):
        fc = 10 # N2 on food has a higher expected pumping rate
    elif strain_name == 'eat2' and (cond_name == 'ff' or cond_name == 'sf'):
        fc = 4 # eat-2 on food have lower expected pumping rate
    elif cond_name == 'fs' or cond_name == 'ss':
        fc = 2 # worms off food have signficantly lower expected pumping rate
        
    Wn    = fc / (fps/2) # fps is the sampling frequency
    b, a  = iirfilter(N=2, Wn=Wn, btype='low', ftype='butter') # low-pass Butterworth filter
    signed_mags_filt = filtfilt(b, a, avg_signed_mags) # filtered signed magnitude
    
    return signed_mags_filt, fc

################################################################################
def get_pumping_rate(video_path=None, grinder_motion=None, fps=20, fc=None, 
                     min_height=0.5, max_height=5, sigma=1, save=False, save_path=None):
    """
    Estimates the continuous and discrete pumping rate of a grinder based on motion signals.

    Inputs:
        video_path (str):         Path to the input video file.
        grinder_motion (ndarray): Array of grinder motion magnitudes per frame.
        fps (float):              Frame rate of the video.
        fc (float):               Cutoff frequency used for filtering based on the strain.
        min_height (float):       Minimum height threshold for peak detection.
        max_height (float):       Maximum height threshold for peak detection.
        sigma (float):            Standard deviation for Gaussian smoothing of discrete pumping rate.
        save (bool):              Whether to save the pumping rate and peak times to disk.
        save_path (str):          Directory path to save output files (required if save=True).

    Outputs:
        pr_cont (ndarray):      Continuous pumping rate signal (Gaussian smoothed).
        pr_disc (ndarray):      Discrete pump counts per time bin.
        peaks (ndarray):        Frame indices where valid pump peaks occur.
        troughs (ndarray):      Frame indices where valid pump troughs occur.
    """
    ### Check that a video path, grinder motion, and grinder CoMs were both provided
    if video_path is None:     raise ValueError("No video path was provided.")
    if grinder_motion is None: raise ValueError("No grinder motion signal was provided.")
        
    ### Detect peaks only if they are followed by a corresponding trough
    dist = int(fps/(fc/2)) # minimum number of frames it takes to complete a pump
    height_range   = [min_height, max_height]
    troughs, peaks = find_troughs_and_peaks(grinder_motion, distance=dist, height=height_range)
    
    ### Bin the detected pumps (i.e., peaks followed by troughs)
    bins = make_bin_edges(fps,1/fps,grinder_motion.shape[0])
    pr_disc, bin_edges = np.histogram(peaks, bins=bins)
    
    ### Filter the binned counts to obtain continuous estimate of pumping rate
    pr_cont = gaussian_filter1d(pr_disc.astype('float'), sigma, mode='nearest')
    
    ### OPTIONAL: save motion and pumping rate estimates
    if save:
        if save_path is None: raise ValueError("No save path was provided.")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_rate_path = save_path + video_name.replace('_cropped', '') + '_PumpKin_rate.csv'
        np.savetxt(save_rate_path, pr_cont)
        print('Continuous pumping rate saved to ' + save_rate_path) 
        
        save_times_path = save_path + video_name.replace('_cropped', '') + '_PumpKin_times.csv'
        np.savetxt(save_times_path, peaks/fps) # saving pump times in seconds, not frames
        print('Pumping times saved to ' + save_times_path)
    
    return pr_cont, pr_disc, peaks, troughs