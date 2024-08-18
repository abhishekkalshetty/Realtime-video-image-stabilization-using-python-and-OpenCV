import os
import cv2
import numpy as np

def scan(file_path):
    if not os.path.isfile(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return False
    return True

def resize_frame(frame, scale_percent=50):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height))

def stabilize_video_from_camera(scale_percent=50):
    cap = cv2.VideoCapture(0)  # 0 is typically the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Get the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    prev_frame = resize_frame(prev_frame, scale_percent)
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Feature detector
    orb = cv2.ORB_create()

    # Feature matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_frame = resize_frame(curr_frame, scale_percent)
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ORB keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)

        # Match descriptors
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched points
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # Find homography
        H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        if H is not None:
            # Warp current frame
            height, width, _ = prev_frame.shape
            stabilized_frame = cv2.warpPerspective(curr_frame, H, (width, height))
        else:
            stabilized_frame = curr_frame

        # Show the stabilized frame
        cv2.imshow('Stabilized Frame', stabilized_frame)

        # Update previous frame
        prev_gray = curr_gray
        prev_frame = curr_frame

        # Check for 'q' or 'Esc' key to quit
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def stabilize_video_from_file(video_path, scale_percent=50):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    prev_frame = resize_frame(prev_frame, scale_percent)

    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Feature detector
    orb = cv2.ORB_create()

    # Feature matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_frame = resize_frame(curr_frame, scale_percent)
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ORB keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)

        # Match descriptors
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched points
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # Find homography
        H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        if H is not None:
            # Warp current frame
            height, width, _ = prev_frame.shape
            stabilized_frame = cv2.warpPerspective(curr_frame, H, (width, height))
        else:
            stabilized_frame = curr_frame

        # Show the stabilized frame
        cv2.imshow('Stabilized Frame', stabilized_frame)

        # Update previous frame
        prev_gray = curr_gray
        prev_frame = curr_frame

        # Check for 'q' or 'Esc' key to quit
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def stabilize_image(image_path, scale_percent=50):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not open image file {image_path}.")
        return
    
    image = resize_frame(image, scale_percent)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(gray, None)

    # Example: Simply using keypoints to align the image (not typical for single image stabilization)
    if len(kp) > 1:
        points1 = np.array([kp[i].pt for i in range(len(kp))], dtype=np.float32)
        points2 = points1.copy()  # This is just for structure; no actual transformation

        H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)
        if H is not None:
            height, width, _ = image.shape
            stabilized_image = cv2.warpPerspective(image, H, (width, height))
        else:
            stabilized_image = image
    else:
        stabilized_image = image

    # Show the stabilized image
    cv2.imshow('Stabilized Image', stabilized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
source = input("Enter source ('camera', 'file', or 'image'): ").strip().lower()
match source:
    case 'camera':
        stabilize_video_from_camera()
    case 'file':
        video_path = input("Enter path to video file: ").strip()
        if scan(video_path):
            stabilize_video_from_file(video_path)
    case 'image':
        image_path = input("Enter path to image file: ").strip()
        if scan(image_path):
            stabilize_image(image_path)
    case _:
        print("Invalid source. Please enter 'camera', 'file', or 'image'.")
