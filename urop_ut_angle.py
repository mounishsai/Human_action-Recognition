import cv2
import mediapipe as mp
import time
import argparse
import math
import numpy as np
import os

class PoseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        lmList = []  # Initialize lmList here
        if self.results.pose_landmarks and draw:
            for lm_id, lm in enumerate(self.results.pose_landmarks.landmark):
                if lm_id > 10:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([lm_id, cx, cy])
            lmList = sorted(lmList, key=lambda x: x[0])

            connections = self.mpPose.POSE_CONNECTIONS
            for connection in connections:
                startingindex, endingindex = connection
                if startingindex > 10 and endingindex > 10:
                    start_point = tuple(lmList[startingindex-11][1:])
                    end_point = tuple(lmList[endingindex-11][1:])
                    cv2.line(img, start_point, end_point, (255, 255, 255), 2)
            for lm in lmList:
                cv2.circle(img, (lm[1], lm[2]), 3, (0, 0, 255), cv2.FILLED)

        return lmList

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                if id > 10:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
            lmList = sorted(lmList, key=lambda x: x[0])

            if draw:
                # Draw a circle on joint 14 (left wrist) for demonstration purposes
                cv2.circle(img, (lmList[14][1], lmList[14][2]), 3, (255, 0, 0), cv2.FILLED)

        return lmList

    def calculateAngles(self, lmList):
        angles = []
        num_joints = len(lmList)
        for i in range(num_joints):
            for j in range(i + 1, num_joints):
                for k in range(j + 1, num_joints):
                    joint1 = np.array(lmList[i][1:])
                    joint2 = np.array(lmList[j][1:])
                    joint3 = np.array(lmList[k][1:])

                    # Check for zero vectors
                    if np.all(joint1 == joint2) or np.all(joint2 == joint3):
                        continue

                    vector1 = joint1 - joint2
                    vector2 = joint3 - joint2

                    # Check for zero vectors
                    if np.all(vector1 == 0) or np.all(vector2 == 0):
                        continue

                    # Check for valid range of dot product
                    dot_product = np.dot(vector1, vector2)
                    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
                    if -1 <= dot_product / norm_product <= 1:
                        angle_rad = np.arccos(dot_product / norm_product)
                        angle_deg = np.degrees(angle_rad)

                        angles.append([lmList[i][0], lmList[j][0], lmList[k][0], int(angle_deg)])

        return angles

def process_single_video(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return None

    frame_counter = 0
    angles_list = []

    # Extract label from the video file name
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    label = get_label_from_filename(video_filename)

    while True:
        success, img = cap.read()
        if not success:
            break
        #if frame_counter % 3 == 0:  # Process every 4th frame
        img = cv2.resize(img, (640, 512))
        detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
                angles = detector.calculateAngles(lmList)
                angles_list.append([angle[3] for angle in angles])

        frame_counter += 1

        if frame_counter== 24:
            break

    cap.release()

    angles_array = np.array(angles_list).T
    feature_vector = angles_array[0]

    for i in range(1, angles_array.shape[0]):
        feature_vector = np.concatenate((feature_vector, angles_array[i]))
    print(len(feature_vector), label)
    return feature_vector, label

def get_label_from_filename(filename):
    # Modify this function based on your naming convention
    if 'walk' in filename:
        return 'walk'
    elif 'clap' in filename:
        return 'clap'
    elif 'sit' in filename:
        return 'sitDown'
    elif 'stand' in filename:
        return 'standUp'
    elif 'wave' in filename:
        return 'waveHands'
    elif 'pick' in filename:
        return 'pickUp'
    elif 'carry' in filename:
        return 'carry'
    elif 'throw' in filename:
        return 'throw'
    elif 'push' in filename:
        return 'push'
    elif 'pull' in filename:
        return 'pull'
    # Add more conditions for other labels if needed
    else:
        return filename

def process_videos_folder(folder_path, detector):
    feature_matrix = []
    labels_list = []

    for video_file in os.listdir(folder_path):
        if video_file.endswith((".mp4", ".mov")):
            video_path = os.path.join(folder_path, video_file)
            print(f"Processing video: {video_path}")

            feature_vector, label = process_single_video(video_path, detector)
            if feature_vector is not None:
                feature_matrix.append(feature_vector)
                labels_list.append(label)

    feature_matrix = np.array(feature_matrix)
    return feature_matrix, labels_list

def main(folder_path):
    detector = PoseDetector(detectionCon=True, trackCon=True)
    feature_matrix, labels_list = process_videos_folder(folder_path, detector)

    # Optionally, you can save the feature matrix and labels list to a file
    np.save('feature_matrix_UT_angle.npy', feature_matrix)
    np.save('labels_list_UT_angle.npy', labels_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose Estimation')
    parser.add_argument('--folder', type=str, default='./posevideos2', help='Path to the folder containing videos')
    args = parser.parse_args()
    main(args.folder)
