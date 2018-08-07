import time
import cv2
import os

def run():

    cap = cv2.VideoCapture(args.filename)

    count = 0
    print("Entering main Loop.")
    while True:
        try:
            ret, frame = cap.read()
            rgb = frame
        except Exception as e:
            print("Failed to grab", e)
            break

        count += 1
        print(count)
        if count%15 != 0:
            continue
        rgb_resize = cv2.resize(rgb, (256, 256))
        cv2.imwrite("original/{}.png".format(count), rgb_resize)

        key = cv2.waitKey(1)
        if key & 255 == ord('p'):
            paused = not paused

        if key & 255 == ord('q'):
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='filename', type=str, help='Name of the video file.')
    args = parser.parse_args()
    if not os.path.exists(os.path.join('./', 'original')):
        os.makedirs(os.path.join('./', 'original'))
    # os.makedirs('original', exist_ok=True)
    run()

