import cv2
import argparse
import glob
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description = 'Create a video from images')

parser.add_argument('output', type = str, help = 'Output path for video file')
parser.add_argument('input', nargs = '+', type = str, help = 'Glob pattern for input images')
parser.add_argument('-fps', type = int, help = 'FPS for video file', default = 24)

args = parser.parse_args()

FILES = []
for i in args.input:
    FILES += glob.glob(i)


filename = Path(args.output).name


FILES.sort() 

frame = cv2.imread(FILES[0])
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter(filename=filename, fourcc=fourcc, fps=args.fps, frameSize=(width, height))

for image_path in FILES:
    print(f'Добавляем кадр: {image_path}')
    img = cv2.imread(image_path)
    video.write(img)

cv2.destroyAllWindows()
video.release()

print(f"Видео '{filename}' успешно создано.")


shutil.move(filename, args.output)
print(f"Видео перемещено в '{args.output}'")