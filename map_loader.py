#!/usr/bin/python3

'''
 Search demonstrations using Python code from AI:MA by Russell and Norvig

    This code loads a map from an image file and creates a discretized map based on the image.

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

import os
import cv2
import numpy as np
from copy import deepcopy

class MapLoader:

    def __init__(self,safe_mode=True):
        self.current_image = -1
        self.image = None
        self.map   = None
        self.color = False
        self.scale = 1.0
        self.x_axis = (0.0, 1.0)
        self.y_axis=(0.0, 1.0)
        self.x_range = 1.0
        self.y_range = 1.0
        self.safe_mode = safe_mode

    def add_frame(self, src_path, file):
        sourcepath = os.path.join(src_path, file)
        try:
            print("Add frame from "+sourcepath+" ...")
            self.image=cv2.imread(sourcepath,cv2.IMREAD_UNCHANGED)
            #print self.image
            print("done!")

        except Exception as msg:
            print(msg)
            return False

        return True

    # Create discretized grid map from the input image with axis limits in Cartesian form
    def create_map(self, scale = 1.0, x_axis = (0.0, 1.0), y_axis=(0.0, 1.0)):
        print("Create map from existing black (obstacle) and white image ...")

        self.map = np.zeros((self.image.shape[0],self.image.shape[1], 3) ,dtype=np.uint8)
        print("  Create map = ",self.map.shape)

        self.x_axis  = x_axis
        self.y_axis  = y_axis
        self.x_range = x_axis[1] - x_axis[0]
        self.y_range = y_axis[1] - y_axis[0]
        self.scale   = scale

        if self.safe_mode:
            # Conservative map creation
            if (scale != 1.0):
                self.map = cv2.resize(self.map, (0,0),fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    if self.image[i][j][0] > 120: # white background is OK  (cv2 is BGR ordering)
                        ix = int(i*scale);
                        iy = int(j*scale);
                        if (self.map[ix][iy][2] ==0):
                            self.map[ix][iy][1] = 192  # green is safe
                    else:
                        ix = int(i*scale);
                        iy = int(j*scale);
                        self.map[ix][iy][1] =   0  # clear safe
                        self.map[ix][iy][2] = 192  # red is obstacle


        else:
            # Simple image based map creation
            # Image based map creation
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    self.map[i][j][0] =   0 #self.image[i][j][0]
                    self.map[i][j][1] =   0 #self.image[i][j][1]
                    self.map[i][j][2] =   0 #self.image[i][j][2]
                    if self.image[i][j][0] > 120: # white background is OK  (cv2 is BGR ordering)
                        self.map[i][j][1] = 192  # green is safe
                    else:
                        self.map[i][j][2] = 192  # red is obstacle

            if scale != 1.0:
                self.map = cv2.resize(self.map, (0,0),fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        print("Done creating map!")

    # Convert world reference point into an opencv index given map limits
    def grid_point(self, world_pt):
        grid_pt = list(world_pt)
        grid_pt[0] = int( ((world_pt[0] - self.x_axis[0])/self.x_range)*self.map.shape[1]);
        grid_pt[1] = int(self.map.shape[0]*(1.0 - ((world_pt[1] - self.y_axis[0])/self.y_range)));
        return (grid_pt[1], grid_pt[0]) # OpenCV uses (row, column) referencing from upper left

    def plot_path(self, path, scale):
        self.path = deepcopy(self.map)

        print(path[0].state)
        prior_point = list(path[0].state)
        prior_point[0] *= scale
        prior_point[1] *= scale
        prior_point = ( int(prior_point[0]), int(prior_point[1]) )
        for ndx, point in enumerate(path):
            if ndx:
                scaled_point = list(point.state)
                scaled_point[0] *= scale
                scaled_point[1] *= scale
                scaled_point = (int(scaled_point[0]), int(scaled_point[1]))

                #print "plot path from",prior_point, " to ",scaled_point
                diff = np.abs(prior_point[0] - scaled_point[0]) + np.abs(prior_point[1] - scaled_point[1])
                if diff < 10:
                    # line is x,y  while point is (j,i) row major
                    cv2.line(self.path,(prior_point[1], prior_point[0]),
                                       (scaled_point[1], scaled_point[0]),
                                       color=(255,255,0),thickness=1)
                prior_point = deepcopy(scaled_point)
            else:
                # Start
                self.path[prior_point[0]][prior_point[1]][0] = 255
                self.path[prior_point[0]][prior_point[1]][1] = 255

        # Start
        self.path[prior_point[0]][prior_point[1]][1] = 255
        self.path[prior_point[0]][prior_point[1]][2] = 255
