import time
import cv2
from moviepy.editor import VideoFileClip


def mp4_to_gif(mp4_path: str, gif_path: str):
    videoClip = VideoFileClip(mp4_path)
    videoClip.write_gif(gif_path, fps=120)

def play_mp4(mp4_path: str, sleep_per_frame: float=.2):
    cap = cv2.VideoCapture(mp4_path)
    ret, frame = cap.read()
    while(1):
        time.sleep(sleep_per_frame)
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
            cap.release()
            cv2.destroyAllWindows()
            break
        cv2.imshow('frame',frame)

if __name__=="__main__":
#    play_mp4("videos/ppo_train.mp4")
    # for i in range(1, 5):
    #     mp4_to_gif("videos/ppo-{}.mp4".format(i), "videos/gifs/ppo-{}.gif".format(i))
    mp4_to_gif("videos/ppo-7.mp4", "videos/gifs/ppo-7.gif")