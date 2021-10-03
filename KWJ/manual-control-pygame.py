import face_recognition_KWJ as fr
# import face_recognition_models as frmodels
# import dlib
from PIL import Image, ImageDraw
from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time
import os
dir_path=os.path.dirname(os.path.realpath(__file__))
print(dir_path)

# Speed of the drone
S = 60
# Frames per second of the pygame window display
# A low number also results in input lag, as input information is processed once per frame.
FPS = 120


class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations (yaw)
            - W and S: Up and down.
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.im_const=4#>=1

        self.send_rc_control = False
        self.process_this_frame=True

        self.model="small"

        # self.frmodel=frmodels.face_recognition_model_location()
        # self.fe=dlib.face_recognition_model_v1(self.frmodel)

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

    # def face_recog(self):
    #     face_locations=fr.face_locations(self.frame_PIL)
    #     face_encodings = fr.face_encodings(self.frame_PIL, face_locations)
    #     pil_image=Image.fromarray(self.frame_PIL)
    #     draw=ImageDraw.Draw(pil_image)

    #     for top,right,bottom,left in face_locations:
    #         name="Unknown"
    #         draw.rectangle(((left,top),(right,bottom)),outline=(0,0,255))
    #         text_width,text_height=draw.textsize(name)
    #         draw.rectangle(((left,bottom-text_height-10),(right,bottom)), fill=(0,0,255),outline=(0,0,255))
    #         draw.text((left+6,bottom-text_height-5),name,fill=(255,255,255,255))
    #     return draw


    def run(self):

        self.tello.connect(False)
        print("Battery: ",self.tello.get_battery(),"%")
        self.tello.set_speed(self.speed)

        # In case streaming is on. This happens when we quit this program without the escape key.
        self.tello.streamoff()
        self.tello.streamon()

        frame_read = self.tello.get_frame_read()

        should_stop = False
        My_Image=fr.load_image_file("DJITelloPy\examples\My_Face.png")
        My_Image_encoding=fr.face_encodings(My_Image)[0]
        kfes=[My_Image_encoding]
        kfns=["KHAN"]
        while not should_stop:
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])

            frame = frame_read.frame
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            if self.process_this_frame:
                rgb_small_frame = cv2.resize(frame,(0,0), fx=1/self.im_const,fy=1/self.im_const)
                fls = fr.face_locations(rgb_small_frame)
                rflmks=fr.raw_face_landmarks(rgb_small_frame,face_locations=fls, model=self.model)
                fes = fr.face_encodings (rgb_small_frame,fls,rflmks)
                fns=[]
                if fes:
                    flmks = fr.face_landmarks(rgb_small_frame, rflmks, fls, self.model)

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
                            name=kfns[bmi]
                            if name=="KHAN":
                                noseloc=list(map(lambda x: x*self.im_const, flmk['nose_tip'][0])) 
                                print(noseloc)
                                cv2.circle(frame,noseloc,30,(255,0,0),cv2.FILLED)
                        fns.append(name)
            self.process_this_frame = not self.process_this_frame



            for (top,right,btm,left), fn  in zip(fls,fns):
                top *=self.im_const
                right *=self.im_const
                btm*=self.im_const
                left *=self.im_const
                cv2.rectangle(frame,(left,top),(right,btm),(0,0,255),2)
                cv2.rectangle(frame, (left, btm-35), (right,btm), (0,0,255), cv2.FILLED)
                font=cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame,fn,(left+6,btm-6),font,1.0,(255,255,255),1)

            text = "Battery: {}%\tHeight: {}".format(self.tello.get_battery(),self.tello.get_height())
            cv2.putText(frame, text, (5, 720 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame = np.rot90(frame)#whole screen rotate
            frame = np.flipud(frame)#'Battery ##%' will flip L-R
            self.frame_PIL = np.array(frame)
            # frame = self.face_recog()
            # print(type(frame))
            #frame = fr.load_image_file(frame)
            #face_locations=face

            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        # Call it always before finishing. To deallocate resources.
        self.tello.end()

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == 27: # ESC
            exit()
        elif key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = S
        elif key == pygame.K_c:
            cur_t=int(time.time())
            cv2.imwrite("./capture/C_{}.png".format(cur_t),self.frame_CV)
            print('capture')
            #cv2.imshow("./capture/{cur_t} image",self.frame)
            time.sleep(0.2)

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            not self.tello.land()
            self.send_rc_control = False

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
