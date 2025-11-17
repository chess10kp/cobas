import numpy as np
import cv2

def crop_cell_from_frames(image_path: str,
                          output_dir: str = None,
                          chunk: int = None,
                          overwrite_file: bool = False):

    # Checks for output_dir or overwrite_file
    if not output_dir and not overwrite_file:
        print("[ERROR]: OUTPUT DIR MUST BE PROVIDED OR ORIGINAL FILE OVERWRITTEN")
        return None

    # Reads image file
    img = cv2.imread(image_path)

    # Converts from RGB/BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper blue HSV --> Hue, Saturation, Value
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Applies binary, color-based mask on the HSV image
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Removes further image noise using median blur, 5x5 grid
    mask = cv2.medianBlur(mask, 5)

    # Finds contours: EXTERNAL and COMPRESSED
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"[ERROR]: NO BLUE CONTOUR FOUND\nFRAME: {image_path}")
        return None

    # Finds largest contour
    contour = max(contours, key=cv2.contourArea)

    # Gets contour bounding box
    x, y, w, h = cv2.boundingRect(contour)

    if chunk:
        xy = chunk // 2

        # Increases the contour
        x, y, w, h = x-xy, y-xy, w+chunk, h+chunk

    # Crops original image
    crop = img[y:y+h, x:x+w]

    # Saves the cropped file
    if overwrite_file: # Overwrites original image file

        # Saves crop
        cv2.imwrite(image_path, crop)
        print(f"{image_path} Overwritten Successfully!")
    else: # Creates new file

        # Gets crop file name
        crop_output_path = output_dir + "/cropped_" + (image_path.split('/')[-1])

        # Saves crop
        cv2.imwrite(crop_output_path, crop)
        print(f"CROP SAVED AT {crop_output_path}")

if __name__ == "__main__":
    ORIGINAL_FRAMES_DIR = "/Users/pedropaiva/Documents/Dev/Research/CoBas E-Energy/cobasFork/Visual/extracted_frames"
    CROPPED_FRAMES_DIR = "/Users/pedropaiva/Documents/Dev/Research/CoBas E-Energy/cobasFork/Visual/cropped_frames"

    crop_cell_from_frames(image_path='/Users/pedropaiva/Documents/Dev/Research/CoBas E-Energy/cobasFork/Visual/extracted_frames/100p_5m_frame007.jpg',
                          chunk=20,
                          output_dir=CROPPED_FRAMES_DIR)