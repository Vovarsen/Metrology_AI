import numpy as np
import cv2
import YOLO_detector
import csv
import time

MODEL_NAME = 'best2.pt'
VIDEO_PATH = "C:/Users/arsen/Downloads/video_2025-11-14_20-03-22.mp4"
OUTPUT_VIDEO_PATH = 'tracked_video_final.mp4'
STATS_FILENAME = 'performance_log.csv'

DISPLAY_WINDOW_WIDTH = 1024
RE_DETECTION_INTERVAL = 90

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print('ошибка открытия файла')
    exit()

ret, frame = cap.read()
if not ret:
    print('Не читается 1 кадр.')
    exit()

frame_h, frame_w = frame.shape[:2]

try:
    track_window = YOLO_detector.f(MODEL_NAME, frame)
except Exception as e:
    print("Ошибка при f")
    track_window = None

if track_window is None:
    print("YOLO не нашла объект на 1 кадре.")
    exit()

x, y, w, h = track_window
if w <= 0 or h <= 0:
    print("некорректные размеры.")
    exit()

shrink_factor = 0.8
new_w = int(w * shrink_factor)
new_h = int(h * shrink_factor)
new_x = x + (w - new_w) // 2
new_y = y + (h - new_h) // 2
x, y, w, h = new_x, new_y, new_w, new_h

x_end = x + w
y_end = y + h
if x >= frame_w or y >= frame_h or x_end <= 0 or y_end <= 0:
    print("ROI выходит за пределы кадра.")
    exit()

x = max(0, x)
y = max(0, y)
w = min(x_end, frame_w) - x
h = min(y_end, frame_h) - y

roi = frame[y:y+h, x:x+w]
if roi.size == 0:
    print('не вырезается')
    exit()

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask, [180, 256], [0, 180, 0, 256])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
track_window = (x, y, w, h)

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

cent_x = x + w / 2
cent_y = y + h / 2
kalman.statePre = np.array([[cent_x], [cent_y], [0], [0]], np.float32)
kalman.statePost = np.array([[cent_x], [cent_y], [0], [0]], np.float32)

fps = cap.get(cv2.CAP_PROP_FPS)
original_size = (frame_w, frame_h)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, original_size)

csv_file = open(STATS_FILENAME, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['detector_ms', 'tracker_ms', 'kalman_ms'])

WINDOW_NAME = 'Tracking'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    time_detector = 0
    time_tracker = 0
    time_kalman = 0

    start_kalman = time.perf_counter()
    prediction = kalman.predict()
    end_kalman = time.perf_counter()
    time_kalman += (end_kalman - start_kalman) * 1000

    if frame_count % RE_DETECTION_INTERVAL == 0:
        start_detector = time.perf_counter()
        det_box = YOLO_detector.f(MODEL_NAME, frame)
        end_detector = time.perf_counter()
        time_detector = (end_detector - start_detector) * 1000

        if det_box is not None:
            x_det, y_det, w_det, h_det = det_box
            track_window = (x_det, y_det, w_det, h_det)
            center_x = x_det + w_det / 2
            center_y = y_det + h_det / 2
            kalman.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
            kalman.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    start_tracker = time.perf_counter()
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    end_tracker = time.perf_counter()
    time_tracker = (end_tracker - start_tracker) * 1000

    x_t, y_t, w_t, h_t = track_window


    start_kalman = time.perf_counter()
    measurement = np.array([[x_t + w_t / 2], [y_t + h_t / 2]], np.float32)
    corrt = kalman.correct(measurement)
    end_kalman = time.perf_counter()
    time_kalman += (end_kalman - start_kalman) * 1000

    csv_writer.writerow([f"{time_detector:.2f}", f"{time_tracker:.2f}", f"{time_kalman:.2f}"])

    kalman_x, kalman_y = int(corrt[0]), int(corrt[1])
    log_x = kalman_x - w_t // 2
    log_y = kalman_y - h_t // 2

    img2 = cv2.rectangle(frame, (log_x, log_y), (log_x + w_t, log_y + h_t), (0, 255, 0), 3)
    video_writer.write(img2)

    scale_factor = DISPLAY_WINDOW_WIDTH / frame_w
    display_frame = cv2.resize(img2, None, fx=scale_factor, fy=scale_factor)
    cv2.imshow(WINDOW_NAME, display_frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

csv_file.close()
video_writer.release()
print('сохранено в файл')

cap.release()
cv2.destroyAllWindows()
