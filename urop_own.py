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

    def calculateDistances(self, lmList):
        distances = []
        num_joints = len(lmList)
        for i in range(num_joints):
            for j in range(i+1, num_joints):
                joint_1 = lmList[i]
                joint_2 = lmList[j]
                dist = math.sqrt((joint_2[1] - joint_1[1])**2 + (joint_2[2] - joint_1[2])**2)
                distances.append([joint_1[0], joint_2[0], int(dist)])
        return distances

def process_single_video(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return None

    frame_counter = 0
    distances_list = []

    # Extract label from the video file name
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    label = get_label_from_filename(video_filename)

    while True:
        success, img = cap.read()
        if not success:
            break
        if frame_counter % 3 == 0:  # Process every 4th frame
            img = cv2.resize(img, (640, 512))
            detector.findPose(img)
            lmList = detector.findPosition(img, draw=False)
            if len(lmList) != 0:
              distances = detector.calculateDistances(lmList)
              distances_list.append([dist[2] for dist in distances])

        frame_counter += 1

        if frame_counter //3 == 40:
            
            break

    cap.release()

    distances_array = np.array(distances_list).T
    feature_vector = distances_array[0]

    for i in range(1, distances_array.shape[0]):
        feature_vector = np.concatenate((feature_vector, distances_array[i]))
    print(len(feature_vector),label)
    return feature_vector, label

def get_label_from_filename(filename):
    # Modify this function based on your naming convention
    if 'draw_a_triangle_action' in filename:
        return 'a1'
    elif 'claps_action' in filename:
        return 'a2'
    elif 'circle_clockwise_action' in filename:
        return 'a3'
    elif 'circle_anticlockwise_action' in filename:
        return 'a4'
    elif 'catch_action' in filename:
        return 'a5'
    elif 'boxing_action' in filename:
        return 'a6'
    elif 'bowling_action' in filename:
        return 'a7'
    elif 'basketballshoot_action' in filename:
        return 'a8'
    elif 'baseball_action' in filename:
        return 'a9'
    elif 'arm_curl_action' in filename:
        return 'a10'
    elif 'draw_x_action' in filename:
        return 'a11'
    elif 'jog_action' in filename:
        return 'a12'
    elif 'jump_action' in filename:
        return 'a13'
    elif 'knock_action' in filename:
        return 'a14'
    elif 'lunge_action' in filename:
        return 'a15'
    elif 'pickup_and_throw_action' in filename:
        return 'a16'
    elif 'push_action' in filename:
        return 'a17'
    elif 'running_action' in filename:
        return 'a18'
    elif 'sit_to_stand_action' in filename:
        return 'a19'
    elif 'squat_action' in filename:
        return 'a20'
    elif 'stand_to_sit_action' in filename:
        return 'a21'
    elif 'swipe_left_action' in filename:
        return 'a22'
    elif 'swipe_right_action' in filename:
        return 'a23'
    elif 'tennis_serve_action' in filename:
        return 'a24'
    elif 'throw_action' in filename:
        return 'a25'
    elif 'walk_action' in filename:
        return 'a26'
    elif 'wave_action' in filename:
        return 'a27'
    
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
    np.save('feature_matrix_own_distance_interleave.npy', feature_matrix)
    np.save('labels_list_own_distance_interleave.npy', labels_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose Estimation')
    parser.add_argument('--folder', type=str, default='./posevideos3', help='Path to the folder containing videos')
    args = parser.parse_args()
    main(args.folder)
