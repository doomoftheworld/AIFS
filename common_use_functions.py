"""
Module with the defined common use functions (normally file system operation functions and some pratical use functions)
"""
import csv
import json
import shutil
import requests
import base64
import io
import cv2
from common_imports import np, pd, os, Image

"""
Functions of file systems operations
"""
def write_csv(data, path):
    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerows(data)
        
        f.close()

def content_existence(path):
    return os.path.exists(path)

def contents_of_folder(folder_path):
    return os.listdir(folder_path)

def create_directory(dir, display=True):
    """
    display: Boolean indicating the folder creation information ("success" or "already exists")
    """
    try:
        # Create target Directory
        os.mkdir(dir)
        if display:
            print("Directory " , dir ,  " Created ") 
    except FileExistsError:
        if display:
            print("Directory " , dir ,  " already exists")

def erase_files_from_folder(folder_path):
    shutil.rmtree(folder_path)

def erase_one_file(file_path):
    # If file exists, delete it.
    if os.path.isfile(file_path):
        os.remove(file_path)
    else:
        # If it fails, inform the user.
        print("Error: %s file not found" % file_path)

def read_content_path(path):
    contents = {}
    for file in os.listdir(path):
        contents[file] = os.path.join(path, file)
    return contents

def path_join(*path_parts):
    return os.path.join(*path_parts)

def store_list_as_json(path, list_to_save):
    with open(path, 'w') as fp:
        json.dump(list_to_save, fp)
        fp.close()

def store_dict_as_json(path, dict_to_save):
    with open(path, 'w') as fp:
        json.dump(dict_to_save, fp)
        fp.close()

def load_json(fp):
    return json.load(fp)

def load_json_by_path(path):
    loaded_json = None
    with open(path) as fp:
        loaded_json = json.load(fp)
        fp.close()
    return loaded_json

def read_csv_to_pd_df(csv_path, first_line_as_head=True, display=False):
    df = None
    if first_line_as_head:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None)

    if display:
        print(df.to_string())
    return df

def save_list_to_csv(csv_path, list, index=None, headers=None, save_index=False, save_headers=True, sep=','):
    if index is not None:
        if headers is not None:
            pd.DataFrame(np.array(list), columns=headers, index=index).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')
        else:
            pd.DataFrame(np.array(list), index=index).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')
    else:
        if headers is not None:
            pd.DataFrame(np.array(list), columns=headers).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')
        else:
            pd.DataFrame(np.array(list)).to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+', encoding='UTF-8')

def save_df_to_csv(csv_path, df, save_index=False, save_headers=True, sep=','):
    df.to_csv(csv_path, index=save_index, header=save_headers, sep=sep, mode='w+')

def save_content_to_file(data, file_path, with_newline=True):
    # Data should be a list of string
    with open(file_path, 'w+', encoding='UTF-8') as f:  
        if with_newline:
            f.write('\n'.join(data) + '\n')
        else:
            f.writelines(data)

def download_image_from_url(img_url, save_path=None):
    """
    The save_path should contains the filename of the image
    """
    img = Image.open(requests.get(img_url, stream = True).raw)
    if save_path is not None:
        img.save(save_path)
    return img

def encode_image(image_path):
    """
    This function encodes the image into a base64 format.

    image_path: The path to the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def encode_and_resize_image(image_path, max_size=1024):
    """
    This function resizes the image under the max size and then return the base64 encoding.

    image_path: The path to the image.
    max_size: The max size of the image.
    """
    with Image.open(image_path) as img:
        # Compute the scale and resize the image
        width, height = img.size
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.LANCZOS)
        
        # Change it to jpeg
        with io.BytesIO() as buffered:
            img.save(buffered, format="JPEG", quality=85)  
            
            # return the encoding
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
def anonymize_fashion_image(image_path, output_path="temp.jpg"):
    """
    Blur faces and body features while preserving clothing details
    
    image_path: Path to input image
    output_path: Path to save processed image (default: "temp.jpg")
    """
    # Load the model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # The function to make all human-related part blurry
    def super_blur(roi):
        # Gaussian blur
        blurred = cv2.GaussianBlur(roi, (199,199), 50)
        
        # Pixelation
        pixel_size = 20
        h, w = roi.shape[:2]
        temp = cv2.resize(blurred, (w//pixel_size, h//pixel_size), 
                         interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (w,h), interpolation=cv2.INTER_NEAREST)
        
        # Mosaic
        mosaic_size = 15
        for y in range(0, h, mosaic_size):
            for x in range(0, w, mosaic_size):
                pixelated[y:y+mosaic_size, x:x+mosaic_size] = np.mean(
                    pixelated[y:y+mosaic_size, x:x+mosaic_size], axis=(0,1))
        
        return pixelated

    # Detect the sensitive areas
    detections = []
    detections.extend(face_cascade.detectMultiScale(gray, 1.1, 5))
    detections.extend(profile_cascade.detectMultiScale(gray, 1.1, 5))
    detections.extend(upper_body_cascade.detectMultiScale(gray, 1.1, 5))

    # Augment the skin detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, (0, 48, 80), (20, 255, 255))
    skin_mask = cv2.dilate(skin_mask, np.ones((11,11)), iterations=2)
    
    # Combine the area
    mask = np.zeros((h,w), dtype=np.uint8)
    for (x,y,w,h) in detections:
        cv2.rectangle(mask, (x,y), (x+w,y+h), 255, -1)
    mask = cv2.bitwise_or(mask, skin_mask)

    # Border blur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x,y,w,h = cv2.boundingRect(cnt)
            x,y,w,h = max(0,x-30), max(0,y-30), min(w,w+60), min(h,h+60)
            img[y:y+h, x:x+w] = super_blur(img[y:y+h, x:x+w])
    
    cv2.imwrite(output_path, img)
    return output_path

def whiten_fashion_image(image_path, output_path="temp.jpg"):
    """
    Replace detected human regions with pure white

    image_path: Path to input image
    output_path: Output path (default: "temp.jpg")
    """
    # Load all detection models
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # Create pure white canvas
    white_mask = np.zeros_like(img)
    white_mask[:] = (255, 255, 255)  # BGR white

    # Combined detection from all models
    detections = []
    detections.extend(face_cascade.detectMultiScale(gray, 1.1, 5))
    detections.extend(profile_cascade.detectMultiScale(gray, 1.1, 5))
    detections.extend(upper_body_cascade.detectMultiScale(gray, 1.1, 5))
    detections.extend(body_cascade.detectMultiScale(gray, 1.05, 5))

    # Skin tone detection (HSV color space)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, (0, 48, 80), (20, 255, 255))
    skin_mask = cv2.dilate(skin_mask, np.ones((11,11)), iterations=2)
    
    # Apply white mask to all detected regions
    for (x, y, w, h) in detections:
        # Expand detection area by 15 pixels
        x, y = max(0, x-15), max(0, y-15)
        w, h = min(w+30, img.shape[1]-x), min(h+30, img.shape[0]-y)
        img[y:y+h, x:x+w] = white_mask[y:y+h, x:x+w]
    
    # Whiten all skin areas
    img[skin_mask == 255] = (255, 255, 255)

    cv2.imwrite(output_path, img)
    return output_path