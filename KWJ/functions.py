import face_recognition_KWJ as fr
import numpy as np
import cv2

import os
def frame_to_recognize_KHAN(frame,kfes,kfns):
    fls = fr.face_locations(frame)
    rflmks=fr.raw_face_landmarks(frame,face_locations=fls, model="small")
    fes = fr.face_encodings (frame,fls,rflmks)
    fns=[]
    if fes:
        flmks = fr.face_landmarks(frame, rflmks, fls, "small")

        assert(len(fes)==len(flmks)) 
        tot=len(fes)

        for i in range(tot):
            fe=fes[i]
            flmk=flmks[i]
            matches=fr.compare_faces(kfes,fe)
            name="Unknown"
            fds = fr.face_distance(kfes,fe)
            bmi = np.argmin(fds)
            if matches[bmi]:
                if kfns[bmi]=="KHAN":
                    KHAN_L=[]
                    KHAN_L.append(np.array(flmk['nose_tip'][0]))
                    KHAN_L.append(np.array((np.array(flmk['left_eye'][0])+np.array(flmk['left_eye'][1]))*0.5,dtype=np.int32))
                    KHAN_L.append(np.array((np.array(flmk['right_eye'][0])+np.array(flmk['right_eye'][1]))*0.5,dtype=np.int32))
                    return np.array(KHAN_L)
            fns.append(name)
    return False

def draw_frame_flmk(frame, flmk, im_const):
    print(flmk)
    flmk *= im_const
    for i in flmk:
        cv2.circle(frame,i,30,(255,0,0),cv2.FILLED)
    return frame

def draw_battery(frame,battery,height):
    text = "Battery: {}%\tHeight: {}".format(battery,height)
    cv2.putText(frame, text, (5, 720 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

def frame_mirror(frame):
    frame = np.rot90(frame)#whole screen rotate
    frame = np.flipud(frame)#'Battery ##%' will flip L-R
    # frame_PIL = np.array(frame)
    return frame

def KHAN_encoding(KHAN_PATH):
    My_Image=fr.load_image_file(KHAN_PATH)
    return [fr.face_encodings(My_Image)[0]], ['KHAN']

def load_faces_by_folder(path):
    lists=os.listdir(path)
    return lists