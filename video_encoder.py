#!/usr/bin/env python3

'''

Class for encoding composite video of two consistent images

    The MIT License (MIT)

    Copyright (c) 2015-2022 David Conner (david.conner@cnu.edu)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
'''

import cv2
import os.path

class VideoEncoder:
    def __init__(self,target_path, img, frame_rate = 30.0, fps_factor=1.0, comp_height=1.0, comp_width=2.0):
        # Define the codec and create VideoWriter object

        self.fps_factor = fps_factor
        print("base image shape = ",img.shape)

        comp_frame = cv2.resize(img[:,:,:],(0,0),fx=comp_width,fy=comp_height)
        self.shape = comp_frame.shape
        print("Composite frame size=",comp_frame.shape)

        if len(self.shape) < 3 or self.shape[2] == None or self.shape[2] == 1:
            self.comp_frame = cv2.cvtColor(comp_frame, cv2.COLOR_GRAY2RGB)
        else:
            # Assume color image
            print("Using color image in comp frame",comp_frame.shape)
            self.comp_frame = comp_frame

        print("composite video shape=",self.comp_frame.shape)

        self.video_out = None
        print("Video speed factor = ",self.fps_factor)
        if self.fps_factor > 0:
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v') # 'I', 'Y', 'U', 'V') # 'D', 'I', 'V', 'X') #  cv2.VideoWriter_fourcc(*'XVID')
            print("Create video file at ",target_path+'.mp4')

            self.video_out  = cv2.VideoWriter(filename=target_path+'.mp4',##'/output.avi',
                                              fourcc = cv2.VideoWriter_fourcc('m','p','4','v'), #'I', 'Y', 'U', 'V'),
                                              fps=frame_rate,
                                              frameSize=(self.comp_frame.shape[1], self.comp_frame.shape[0]) ) # width,height
        else:
            print("No video processing !")

    def add_dual_frame(self, expanded_img, cost_img):

        h1 = expanded_img.shape[0]
        w1 = expanded_img.shape[1]
        d1 = expanded_img.shape[2]

        hc = cost_img.shape[0]
        wc = cost_img.shape[1]
        dc = cost_img.shape[2]

        h2 = self.comp_frame.shape[0]
        w2 = self.comp_frame.shape[1]
        d2 = self.comp_frame.shape[2]

        #print "i(h,w.d)=",h1,", ",w1,". ",d1,"c(h,w.d)=",hc,", ",wc,". ",dc,"  2 h2=",h2,", ",w2,",  d2=",d2
        #print "comp_frame=",self.comp_frame.shape
        #print "expanded=",expanded_img.shape
        #print "cost_img=",cost_img.shape

        # Build a composite image
        try:
            self.comp_frame[  :h1,  :w1,:d2] = expanded_img[:,:,:3]
        except Exception as msg:
            print("Invalid composite frame with expanded!")
            print(msg)
            return False
        try:
            self.comp_frame[  :h1,w1:w2,:d2] = cost_img[:,:,:3]
        except Exception as msg:
            print("Invalid composite frame with cost!")
            print(msg)
            return False

        if self.fps_factor > 0:
            try:
                #print "write frame ..."
                ret=self.video_out.write(self.comp_frame)
                if ret:
                    print("     video write ret=",ret)
            except:
                print("Failed to write a video frame!")
                return None

        return self.comp_frame

    def release(self):
        print(" fps factor = ",self.fps_factor)
        if self.fps_factor > 0:
            print("Releasing the video capture ...")
            print(" Use\n   avconv -i output.avi -b 1440k test.avi\n to compress output video")
            self.video_out.release()
