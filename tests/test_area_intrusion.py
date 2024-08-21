import cv2

from model.operations import handle_area_intrusion


def test_area_intrusion():
    cap = cv2.VideoCapture("resources/person.mp4")
    assert cap.isOpened(), "Error reading video file"

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        pattern_detected, result = handle_area_intrusion(im0)

        cv2.imshow("Object Tracking", result.plot_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
