import cv2
from ultralytics import YOLO
from deep_sort_realtime import deepsort_tracker

model = YOLO('yolov8n-pose.pt')

tracker = deepsort_tracker.DeepSort()

POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 머가리
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 팔
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # 다리
    (5, 11), (6, 12)  # 몸통
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def draw_keypoints_and_connections(frame, keypoints):
    for person_kpts in keypoints:
        # 키포인트 그리기
        for x, y in person_kpts:
            if x > 0 and y > 0:  # 유효한 키포인트만
                cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)

        # 연결선 그리기
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = tuple(map(int, person_kpts[start_idx]))
            end_point = tuple(map(int, person_kpts[end_idx]))
            if all(p > 0 for p in start_point + end_point):  # 키포인트 둘 다 유효할 때만 선 연결
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)


def print_keypoints(track_id, keypoints):
    print(f"\nPerson ID: {track_id}")
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            print(f"  {KEYPOINT_NAMES[i]:<15}: ({x:.2f}, {y:.2f})")


def process_frame(frame):
    results = model(frame, conf=0.5)

    detections = []
    keypoints_list = []
    for r in results[0]:  # results[0]에 한 프레임 모든 detection 결과 담김
        box = r.boxes.xyxy[0].cpu().numpy().astype(int)
        conf = r.boxes.conf[0].cpu().numpy()
        kpts = r.keypoints.xy[0].cpu().numpy() if r.keypoints else None

        detections.append(([box[0], box[1], box[2] - box[0], box[3] - box[1]], conf, 0))
        if kpts is not None:
            keypoints_list.append(kpts)

    tracks = tracker.update_tracks(detections, frame=frame)

    for track, kpts in zip(tracks, keypoints_list):
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()

        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        print_keypoints(track_id, kpts)

    if keypoints_list:
        draw_keypoints_and_connections(frame, keypoints_list)

    return frame



def main():
    source = input("input 'webcam' or mp4 path: ")

    if source.lower() == 'webcam':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)

        cv2.imshow('Output', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
