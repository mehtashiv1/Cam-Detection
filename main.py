import glob
import cv2
import os
from emailing import send_email
from threading import Thread

video = cv2.VideoCapture(0)   # CAPTURE VIDEO FROM WEBCAM

opened = video.isOpened()

first_frame = None      # VARIABLE TO STORE FIRST FRAME TO COMPARE W A MOVEMENT IN NEXT FRAMES
status_list = []   # VARIABLE TO STORE THE STATUS OF OBJECT IN FRAME, IF OBJECT IN FRAME THEN STATUS = 1, ELSE 0
count = 1   # VARIABLE TO STORE THE IMAGES TAKEN WHEN THE OBJECT IS IN FRAME


def clean_folder():
    images = glob.glob("images/*.png")
    for image in images:
        os.remove(image)


if opened:
    while opened:
        status = 0
        check, frame = video.read()

        # CONVERTING FIRST FRAME TO GRAY SCALE TO REDUCE MATRIX COMPLICATIONS
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        # USING GAUSSIAN BLUR TO EASE THE COMPARISON BETWEEN FRAMES
        gau_blur = cv2.GaussianBlur(gray_frame, (13, 13), 0)

        # HOLDS FIRST FRAME VALUE AND COMPARE IT WITH NEXT FRAMES
        if first_frame is None:
            first_frame = gau_blur

        # DIFFERENTIATING FIRST FRAME WITH UPCOMING FRAMES
        delta_frame = cv2.absdiff(first_frame, gau_blur)

        # THRESHOLDING THE VIDEO, VALUE ABOUT 60 CHANGES TO 255
        threshold_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]

        # REMOVING NOISE FROM FRAME USING DILATE FUNCTION
        dil_frame = cv2.dilate(threshold_frame, None, iterations=2)

        # FINDING CONTOURS FOR DETECTING OBJECT MOVEMENT IN THE VIDEO
        # RETR_EXTERNAL FINDS THE OUTERMOST CONTOUR(BOUNDARY) OF THE OBJECT IN FRAME
        # CHAIN_APPROX SIMPLE STORES ALL THE CONTOURS(EXTERNAL)
        contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ADDING A RECTANGLE AROUND THE OBJECT IN FRAME
        for contour in contours:
            if cv2.contourArea(contour) < 3000:
                continue
            x, y, w, h = cv2.boundingRect(contour)  # FORMS A
            rectangle = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)  # GIVING THE RECTANGLE SOME DIMENSIONS
            if rectangle.any():
                status = 1           # PUTTING THE STATUS 1 WHEN THE OBJECT IS DETECTED AND THE RECTANGLE IS FORMED
                cv2.imwrite(f'images/{count}.png', frame)  # STORING THE IMAGES IN THE FOLDER WHEN OBJ IN FRAME
                count = count + 1  # INCREMENTING COUNT TO CAPTURE IMAGES IN EVERY FRAME
                all_images = glob.glob("images/*.png")  # SELECTING ALL IMAGES USING GLOB
                index = int(len(all_images) / 2)   # TAKING THE MIDDLE IMAGE TO SEND AS ATTACHMENT THROUGH MAIL
                image_with_object = all_images[index]

        status_list.append(status)   # APPENDING STATUS VALUE TO THE STATUS LIST WHEN OBJECT ENTERS
        status_list = status_list[-2:]  # TAKING LAST TWO VALUES OF THE LIST

        # WE WILL SEND THE EMAIL WHEN THE OBJECT WILL EXIT THE FRAME I.E WHEN STATUS CHANGES FROM 1 TO 0
        if status_list[0] == 1 and status_list[1] == 0:
            email_thread = Thread(target=send_email, args=(image_with_object, ))
            email_thread.daemon = True
            clean_folder_thread = Thread(target=clean_folder)
            email_thread.daemon = True

            email_thread.start()

        print(status_list)
        cv2.imshow('video', frame)
        if cv2.waitKey(2) == 27:
            break


video.release()
clean_folder_thread.start()

