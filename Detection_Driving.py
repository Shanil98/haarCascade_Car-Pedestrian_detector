import cv2

# our image or video
#img_file = 'dashCam.jpeg'
vid = 'dashcam1.mp4'

# our pre-trained car classifier and pre-trained pedestrian classififer
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

# create opencv image, it reads the image to read it correctly
#img = cv2.imread(img_file)
# create car and pedestrian classifiers
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)



video = cv2.VideoCapture(vid)

while True:
    successful_frame_read, frame = video.read()

    if successful_frame_read:
        # convert the vid to grayscale
        black_n_white = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(black_n_white)
    pedestrians = pedestrian_tracker.detectMultiScale(black_n_white)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # display the square when the car is spotted in the frame
    cv2.imshow('heres the video', frame)

    # waitKey is necessary to display the image, till you hit a key
    key = cv2.waitKey(1)
    # checking key agains ASCII for Q or q
    if key == 81 or key == 113:
        break

print('code completed')
