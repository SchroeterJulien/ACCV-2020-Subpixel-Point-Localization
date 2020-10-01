## Allows to generate the small .mp4 videos, using the image creates during the optimization.
import cv2

version_name = "noMassConvergence_"
video_name = version_name+ '_video.mp4'

# Other parameters
n_iter = 100000 + 1
visualization_step = 200

# Get shape of frame
frame = cv2.imread("img/" + version_name + str(0) + ".png")
height, width, layers = frame.shape

## Write frames to video
_fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, _fourcc, 40.0, (width,height))
for step in range(n_iter):

    if step<0 \
            or step % int(visualization_step/4) == 0 and step<2000 \
            or step % int(2*visualization_step/2) == 0:
        print(step)

        video.write(cv2.imread("img/" + version_name + str(step) + ".png"))

cv2.destroyAllWindows()
video.release()







