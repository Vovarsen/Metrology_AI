import numpy as np
import cv2
import YOLO_detector
import csv 

MODEL_NAME = 'best.pt'
VIDEO_PATH = "C:/Users/arsen/Downloads/100см.mp4"
IMAGE_NAME = 'frame_for_yolo.jpg'
OUTPUT_VIDEO_PATH = 'tracked_video_kalman2.mp4'
#STATS_FILENAME = 'tracking_log_kalman2.csv'

DISPLAY_WINDOW_WIDTH = 1024 
 
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print('ошибка открытия файла')
 

ret,frame = cap.read() 
cv2.imwrite(IMAGE_NAME, frame)
frame_h, frame_w = frame.shape[:2]

try:
    track_window = YOLO_detector.f(MODEL_NAME, frame)
except Exception as e:
    print("Ошибка при вызове f")
    track_window = None

if track_window is None:
    exit()



x, y, w, h = track_window

if w<=0 or h <= 0:
    exit()



shrink_factor = 0.7
new_w = int(w * shrink_factor)
new_h = int(h * shrink_factor)

new_x = x + (w - new_w) // 2
new_y = y + (h - new_h) // 2

x,y,w,h = new_x,new_y,new_w,new_h

x_end = x+w
y_end = y+h

if x >= frame_w or y >= frame_h or x_end <= 0 or y_end <= 0:
    exit()

x = max(0,x)
y = max(0,y)
w = min(x_end, frame_w) - x
h = min(y_end, frame_h) - y

roi = frame[y:y+h, x:x+w]
if roi.size == 0:
        print('не удалось вырезать')
        exit()

roi_display = cv2.resize(roi, (200, int(200 * roi.shape[0] / roi.shape[1])))
cv2.imshow('Initial ROI', roi_display)
cv2.waitKey(0)
cv2.destroyWindow('Initial ROI')

hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
   
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
track_window = (x,y,w,h)

kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype = np.float32) * 0.03

cent_x = x + w/2
cent_y = y + w/2
kalman.statePre = np.array([[cent_x], [cent_y], [0], [0]], np.float32)
kalman.statePost = np.array([[cent_x], [cent_y], [0], [0]], np.float32)


fps = cap.get(cv2.CAP_PROP_FPS)
original_size = (frame_w, frame_h)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, original_size)

#csv_file = open(STATS_FILENAME, 'w', newline='')
#csv_writer = csv.writer(csv_file)
#csv_writer.writerow(['x', 'y', 'width', 'height'])



WINDOW_NAME = 'Tracking'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

while True:
    ret ,frame = cap.read()
    
    if ret == True:

        prediction = kalman.predict()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
 

        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
 

        x_t,y_t,w_t,h_t = track_window

        measurement = np.array([[x_t + w_t /2], [y_t + h_t/2]], np.float32)
        corrt = kalman.correct(measurement)

        kalman_x, kalman_y = int(corrt[0]), int(corrt[1])
        log_x = kalman_x - w_t//2
        log_y = kalman_y - h_t//2

        #csv_writer.writerow([ log_x,log_y, w_t, h_t])
        

        img2 = cv2.rectangle(frame, (log_x,log_y), (log_x+w_t,log_y+h_t), (0,255,0), 3)
        video_writer.write(img2)

        scale_factor = DISPLAY_WINDOW_WIDTH/frame_w
        display_frame = cv2.resize(img2, None, fx = scale_factor, fy=scale_factor)
        cv2.imshow(WINDOW_NAME, display_frame)
    
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        #elif k == ord('s'):
            #cv2.imwrite('screenshot_car.jpg', img2)
   
    else:
        break

#csv_file.close()
#print(f"Статистика сохранена в файл: {STATS_FILENAME}")


video_writer.release()
print('сохранено в файл')

cap.release()
cv2.destroyAllWindows()