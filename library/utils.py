################################################################################
# training.py
# Written by Jacob Wheelock & Erin Shappell for Lu Lab
# 
# This module defines a custom `Dataset` class for loading images and corresponding 
# bounding box annotations in Pascal VOC XML format. It also includes utility functions 
# for batching data and creating PyTorch `DataLoader` objects for training and validation.
#
################################################################################
# Imports
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import glob as glob
import xml.etree.ElementTree as et
import csv
import numpy as np
import os
from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

################################################################################   
def collate_fn(batch):
    """
    Custom collate function to merge a list of samples into a batch.

    Inputs:
        batch (list): List of samples, where each sample is a tuple of data elements.

    Output:
        tuple: Tuple of tuples, where each inner tuple contains all elements
               of a given type from the batch (e.g., images, targets).

    """
    return tuple(zip(*batch))

################################################################################   
class getDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images and corresponding bounding box annotations
    from a directory containing image files and Pascal VOC-style XML annotation files.

    Attributes:
        dir_path (str):                  Directory path containing images and XML annotation files.
        width (int):                     Desired image width after resizing.
        height (int):                    Desired image height after resizing.
        transforms (callable, optional): Optional transformations to be applied on the images and bounding boxes.
        classes (list):                  List of unique class names parsed from annotation XML files, with 'background' as the first class.
        all_images (list):               Sorted list of image filenames in the dataset directory.

    Methods:
        get_classes_from_annotations():
            Parses XML annotation files to extract all unique classes.

        __getitem__(idx):
            Loads and processes the image and its annotations at index `idx`.
            Applies resizing and optional transformations.
            Returns the processed image tensor and target dictionary with bounding boxes and labels.

        __len__():
            Returns the total number of images in the dataset.

    Usage:
        dataset = getDataset(dir_path='path/to/data', width=224, height=224, transforms=transform_function)
        image, target = dataset[0]

    """
    def __init__(self, dir_path, width, height, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = self.get_classes_from_annotations()
        
        
        image_extensions = ['jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'tif']
        all_extensions = image_extensions + [ext.upper() for ext in image_extensions]  # Add uppercase versions
        self.image_paths = glob.glob(f"{self.dir_path}/*.png")
        for extension in all_extensions:
            self.image_paths.extend(glob.glob(f"{self.dir_path}/*.{extension}"))
        # Extract just the filenames
        self.all_images = [os.path.basename(image_path) for image_path in self.image_paths]
        
        self.all_images = sorted(self.all_images)
        
    def get_classes_from_annotations(self):
        """
        Parse all XML files in the dataset directory to build a list of unique classes.
        """
        classes = set()
        xml_files = glob.glob(f"{self.dir_path}/*.xml")
        for xml_file in xml_files:
            tree = et.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                try:
                    class_name = member.find('class').text
                except:
                    class_name = member.find('label').text
                classes.add(class_name)
        
        # Add 'background' as the first class and sort the rest alphabetically
        return ['background'] + sorted(classes)
    
    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        #print(image_name)
        image_path = os.path.join(self.dir_path, image_name)
        #print(image_path)
        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        af = image_name.split('.')
        # capture the corresponding XML file for getting the annotations
        annot_filename = af[0] + '.xml'
        
        annot_file_path = self.dir_path + '/' + annot_filename
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        
        root = tree.getroot()
        
        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            try:
                labels.append(self.classes.index(member.find('class').text))
            except:
                labels.append(self.classes.index(member.find('label').text))
            try:
                # xmin = left corner x-coordinates
                xmin = int(member.find('xmin').text)
            except:
                # xmin = left corner x-coordinates
                xmin = int(member.find('x').text)    
            try:
                # xmax = right corner x-coordinates
                xmax = int(member.find('xmax').text)
            except:
                # xmax = right corner x-coordinates
                xmax = xmin + int(member.find('width').text)  
            try:
                # ymin = left corner y-coordinates
                ymin = int(member.find('ymin').text)
            except:
                # xmin = left corner y-coordinates
                ymin = int(member.find('y').text)   
            try:
                # ymax = right corner x-coordinates
                ymax = int(member.find('ymax').text)
            except:
                # xmin = left corner y-coordinates
                ymax = ymin + int(member.find('height').text)   
            
            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # bounding box to tensor
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return image_resized, target
    def __len__(self):
        return len(self.all_images)

################################################################################   
def get_loaders(train_dataset, valid_dataset, BATCH_SIZE, collate_fn):
    """
    Create DataLoader objects for training and validation datasets.

    Inputs:
        train_dataset (Dataset): PyTorch Dataset object for training data.
        valid_dataset (Dataset): PyTorch Dataset object for validation data.
        BATCH_SIZE (int):        Number of samples per batch to load.
        collate_fn (callable):   Function to merge a list of samples into a mini-batch, used for handling variable-size inputs.

    Output:
        list: A list containing two DataLoader objects:
              - train_loader: DataLoader for the training dataset with shuffling enabled.
              - valid_loader: DataLoader for the validation dataset without shuffling.

    """
    train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
    )
    valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
    )
    return [train_loader, valid_loader]

################################################################################   
### PUMPKIN-SPECIFIC FUNCTIONS ###
################################################################################   
def getROI(roi_width, coords, frame, f_width, f_height):
    """
    Extract a fixed-size square region of interest (ROI) from a video frame, centered at specified coordinates,
    while ensuring the ROI remains within the frame boundaries.

    Inputs:
        roi_width (int):    Half-width of the square ROI to extract (final size will be 2*roi_width).
        coords (tuple):     (x, y) coordinates representing the center of the ROI in the frame.
        frame (ndarray):    Input video frame as a NumPy array of shape (H, W, C).
        f_width (int):      Width of the input frame.
        f_height (int):     Height of the input frame.

    Output:
        ndarray: The extracted ROI as a NumPy array of shape (2*roi_width, 2*roi_width, 3),
                 clipped and converted to uint8 format for compatibility with video processing.
    """
    # Get area centered at bounding box center
    # cropping a [2*roi_width x 2*roi_width] px area from the frame
    x, y = coords
    size = 2 * roi_width

    # Calculate initial bounds
    x_min = x - roi_width
    x_max = x + roi_width
    y_min = y - roi_width
    y_max = y + roi_width
    
    # Adjust bounds to ensure they remain within frame limits
    if x_min < 0:
        x_max += abs(x_min)
        x_min = 0
    if x_max > f_width:
        x_min -= (x_max - f_width)
        x_max = f_width
    if y_min < 0:
        y_max += abs(y_min)
        y_min = 0
    if y_max > f_height:
        y_min -= (y_max - f_height)
        y_max = f_height
    
    # Ensure the adjusted bounds are still valid
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(f_width, x_max)
    y_max = min(f_height, y_max)

    # Check final size and adjust if necessary to maintain the ROI size
    # Prioritize shifting the ROI within bounds without resizing
    if (x_max - x_min) < size:
        shift = size - (x_max - x_min)
        if x_min - shift >= 0:
            x_min -= shift
        else:
            x_max += shift
        x_min = max(0, x_min)
        x_max = min(f_width, x_max)

    if (y_max - y_min) < size:
        shift = size - (y_max - y_min)
        if y_min - shift >= 0:
            y_min -= shift
        else:
            y_max += shift
        y_min = max(0, y_min)
        y_max = min(f_height, y_max)

    # Extract ROI
    roi = frame[y_min:y_max, x_min:x_max, :]

    # Ensure it has 3 channels (BGR format)
    if len(roi.shape) == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    # Ensure dtype is uint8 for video compatibility
    roi = np.clip(roi, 0, 255).astype(np.uint8)

    return roi

################################################################################   
def getIOU(box_1, box_2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    Inputs:
        box_1 (array-like): Bounding box in the format [x_min, y_min, x_max, y_max].
        box_2 (array-like): Second bounding box in the same format.

    Output:
        float: IoU value, defined as the area of intersection divided by the area of union
               between the two bounding boxes. Returns 0 if there is no overlap.
    """
    # Coordinates of the area of intersection
    ix1 = np.maximum(box_1[0], box_2[0])
    iy1 = np.maximum(box_1[1], box_2[1])
    ix2 = np.minimum(box_1[2], box_2[2])
    iy2 = np.minimum(box_1[3], box_2[3])
     
    # Intersection height and width
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width  = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
    intersect_area = i_height * i_width
     
    # Box 1 dimensions
    b1_height = box_1[3] - box_1[1] + 1
    b1_width  = box_1[2] - box_1[0] + 1
     
    # Box 2 dimensions
    b2_height = box_2[3] - box_2[1] + 1
    b2_width  = box_2[2] - box_2[0] + 1
     
    union_area = b1_height * b1_width + b2_height * b2_width - intersect_area
     
    IOU = intersect_area / union_area
    return IOU

################################################################################   
def smooth(data, sigma):
    """
    Apply 1D Gaussian smoothing to input data.

    Inputs:
        data (array-like): 1D array of numerical values to be smoothed.
        sigma (float):     Standard deviation of the Gaussian kernel used for smoothing.

    Output:
        ndarray: Smoothed version of the input data, with the same shape.
    """
    data_smoothed = gaussian_filter1d(data,sigma,mode='nearest')
    return data_smoothed

################################################################################   
def get_dist(ROI1, ROI2):
    """
    Compute the slope between the centers of two rectangular regions of interest (ROIs).

    Inputs:
        ROI1 (array-like): Coordinates of the first ROI in the format [x_min, y_min, x_max, y_max].
        ROI2 (array-like): Coordinates of the second ROI in the same format.

    Output:
        float: Slope between the centers of the two ROIs, calculated as
               (Δy / Δx) = (center2_y - center1_y) / (center2_x - center1_x).
               May raise a divide-by-zero error if centers have the same x-coordinate.
    """
    center1 = [np.mean([ROI1[0], ROI1[2]]).astype('int'), np.mean([ROI1[1], ROI1[3]]).astype('int')]
    center2 = [np.mean([ROI2[0], ROI2[2]]).astype('int'), np.mean([ROI2[1], ROI2[3]]).astype('int')]
    return (center2[1]-center1[1])/(center2[0]-center1[0])

################################################################################   
def interpolate(data, kind='cubic'):
    """
    Interpolate missing or zero values in a 1D array using a specified interpolation method.

    Inputs:
        data (array-like): 1D array of numerical values, where zeros are treated as missing data.
        kind (str):        Type of interpolation to use (e.g., 'linear', 'cubic', 'quadratic').
                           Default is 'cubic'.

    Output:
        ndarray: Array of the same shape as input, with zero values replaced by interpolated values.
                 If fewer than two non-zero values are present, the original array is returned unchanged.
    """
    x = np.arange(len(data))
    mask = data != 0
    
    # Ensure at least two non-zero points for interpolation
    if np.sum(mask) < 2: return data
    
    f = interp1d(x[mask], data[mask], kind=kind, fill_value="extrapolate")
    return f(x)

################################################################################   
def filter_ROIs(ROIs, threshold=200, sigma=3, max_values=(250,250), verbose=False):
    """
    Filter, interpolate, and smooth a sequence of ROI center coordinates to correct outliers and missing values.

    Inputs:
        ROIs (ndarray):     2D NumPy array of shape (n, 2) containing ROI center coordinates over time.
        threshold (float):  Maximum allowed distance between consecutive points before treating as an outlier.
                            Default is 200.
        sigma (float):      Standard deviation for Gaussian smoothing. Default is 3.
        max_values (tuple): Maximum (x, y) values to clip the ROI coordinates to. Default is (250, 250).
        verbose (bool):     If True, enables verbose output (not used in current implementation). Default is False.

    Output:
        tuple: A tuple containing two arrays:
            - ROIs_filt (ndarray):   Interpolated and clipped ROI coordinates.
            - ROIs_smooth (ndarray): Smoothed ROI coordinates, rounded to integers.
    """
    # Check array dimensions
    if ROIs.shape[1] != 2:
        raise ValueError("Input ROIs must be a 2D NumPy array with shape (n, 2).")
    
    # If the first value is dropped, replace it with the next valid value
    if np.all(ROIs[0] == 0):
        ROIs[0] = ROIs[np.where(np.all(ROIs != 0, axis=1))[0][0]]

    # Replace all outlier errors with the last valid value
    last_valid = ROIs[0]  
    for i in range(1, len(ROIs)):
        if np.linalg.norm(np.array(ROIs[i]) - np.array(last_valid)) > threshold:
            ROIs[i] = last_valid
        else:
            last_valid = ROIs[i]

    # Convert lists to np arrays for interpolation
    ROIs_filt      = np.zeros_like(ROIs, dtype=float)
    ROIs_filt[:,0] = interpolate(ROIs[:,0])
    ROIs_filt[:,1] = interpolate(ROIs[:,1])
    
    # Remove negative values and values that exceed the dimensions of the video
    ROIs_filt[:,0] = np.clip(ROIs_filt[:,0], a_min=0, a_max=max_values[0]) 
    ROIs_filt[:,1] = np.clip(ROIs_filt[:,1], a_min=0, a_max=max_values[1]) 
    
    ### Step 3: Smooth interpolated ROI centers
    ROIs_smooth      = np.zeros_like(ROIs, dtype=float)
    ROIs_smooth[:,0] = smooth(ROIs_filt[:,0], sigma)
    ROIs_smooth[:,1] = smooth(ROIs_filt[:,1], sigma)
    ROIs_smooth      = ROIs_smooth.astype('int')    
    
    return ROIs_filt, ROIs_smooth