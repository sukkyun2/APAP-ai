import cv2

from app.main import handle_estimate_distance


def test_calculate_distance():
    cap = cv2.VideoCapture("resources/person.mp4")
    assert cap.isOpened(), "Error reading video file"

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        pattern_detected, result = handle_estimate_distance(im0)

        cv2.imshow("Object Tracking", result.plot_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
