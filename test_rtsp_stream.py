import time
import cv2

fps = 52
frame_width = 640
frame_height = 480
flip = 0
camSet='libcamerasrc ! video/x-raw, width=640, height=480, framerate=25/1 ! videoconvert ! appsink drop=true'
cam=cv2.VideoCapture(camSet,cv2.CAP_GSTREAMER)

gst_str_rtp = ' appsrc ! videoconvert ! video/x-raw,format=I420,width=640,height=480,framerate=25/1 !  videoconvert !\
    v4l2h264enc extra-controls="controls,repeat_sequence_header=1,h264_profile=1,h264_level=11,video_bitrate=5000000,h264_i_frame_period=26,h264_minimum_qp_value=1" ! video/x-h264,level=(string)4 ! h264parse ! rtph264pay ! \
udpsink host=0.0.0.0 port=8000'

if cam.isOpened() is not True:
    print('Cannot open camera. Exiting.')
    quit()

fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter(gst_str_rtp, fourcc, 25, (frame_width, frame_height), True)

while True:
    ret, frame = cam.read()

    cv2.imshow('webcam',frame)
    out.write(frame)
    cv2.moveWindow('webcam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
out.release()
cv2.destroyAllWindows()
