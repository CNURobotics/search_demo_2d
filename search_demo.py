'''

     Search demonstrations using Python code from AI:MA by Russell and Norvig
           https://code.google.com/p/aima-python/



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

import sys
import os.path
from copy import deepcopy

import cv2
import numpy as np

from russell_and_norvig_search import depth_first_graph_search, breadth_first_graph_search
from russell_and_norvig_search import greedy_best_first_graph_search, uniform_cost_search
from russell_and_norvig_search import UndirectedGraph, Problem
from russell_and_norvig_search import distance, astar_search

from color_map import color_map
from video_encoder import VideoEncoder
from map_loader import MapLoader

INFINITY=sys.maxsize


class GridProblem(Problem):
    "The problem of searching a grid from one node to another."
    def __init__(self, initial, goal, graph, map, scale, video_encoder = None):
        Problem.__init__(self, initial, goal)
        self.graph = graph
        self.map   = deepcopy(map)
        self.costs = np.ones(map.shape,dtype=np.uint8)*255
        self.cost_values = np.ones(map.shape[:2],dtype=np.uint8)*255

        self.expansion = 0

        # Define a video encoder to generate videos during the search process
        self.video_encoder =  video_encoder

    def reset_encoder(self, video_encoder, map):
        self.video_encoder =  video_encoder
        self.map   = deepcopy(map)
        self.costs = np.ones(map.shape,dtype=np.uint8)*255
        self.cost_values = np.ones(map.shape[:2],dtype=np.uint8)*255

    # Define what actions are permissible in this world
    def actions(self, parent_posn):
        "The actions at a graph node are just its neighbors."
        self.expansion += 1
        x=parent_posn[0]
        y=parent_posn[1]
        #print "Node (",x,", ",y,")"
        if self.map[x][y][0] < 200:
            self.map[x][y][0] = 255
            #self.map[x][y][1] = 255
            #self.map[x][y][2] = 255
        #actions = []

        # 8 connected grid world
        for i in range(-1,2):
            for j in range(-1,2):
                if i == 0 and j == 0:
                    continue # self reference

                # Convert to global grid coordinates
                ix = x + i
                iy = y + j

                # Keep it in bounds (assumes bounded world)
                if ix < 0 or iy < 0:
                    continue

                if ix >= self.map.shape[0]:
                    continue

                if iy >= self.map.shape[1]:
                    continue

                # Connect this node in the graph
                child_posn = (ix, iy)

                # cost is 2 times distance (3 ~ 2*sqrt(2))
                dist = min(abs(2*i)+abs(2*j), 3)

                #print "Blocked at ",ix,", ",iy, " = ",self.map[ix][iy]

                if self.map[ix][iy][2] > 190:
                    dist = INFINITY
                    #print "Blocked at ",ix,", ",iy
                    #cv2.waitKey()
                else:
                    if self.map[ix][iy][2] > 0:
                        # Multply distance by a cost function based on red channel color in map (for varying costs)
                        dist *= (5.01 + (float(self.map[ix][iy][2])/8.))
                        dist = int(dist)

                    if self.map[ix][iy][0] < 100:
                        self.map[ix][iy][0] = 92 # mark the cells we have visted during search

                    #print "   Adding ", child_posn," at d=",dist
                    self.graph.connect(parent_posn, child_posn, dist)  # add valid edge to the graph according to search criteria
                    #actions.append((ix,iy))
                    #print " Keys for parent_posn=",parent_posn," = ", self.graph.get(parent_posn).keys()


        if self.video_encoder is not None:
            # Add video to encode the search behavior
            #print "actions=",actions
            scale=0.1
            big_map   = cv2.resize(self.map, (0,0),fx=(1.0/scale),
                                   fy=(1.0/scale), interpolation=cv2.INTER_NEAREST)
            big_costs = cv2.resize(self.costs, (0,0),fx=(1.0/scale),
                                   fy=(1.0/scale), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Search ",big_map)
            cv2.imshow("Costs ", big_costs)
            cv2.waitKey(25)
            self.video_encoder.add_dual_frame(big_map, big_costs)

        return list(self.graph.get(parent_posn).keys())

    def result(self, state, action):
        "The result of going to a neighbor is just that neighbor."
        return action

    # Update the path cost and color the map image
    def path_cost(self, cost_so_far, parent_posn, action, child_posn):
        #print "parent_posn",parent_posn," to child_posn",child_posn," = ",
        # self.graph.get(parent_posn,child_posn)," + ", cost_so_far
        total_cost = cost_so_far + (self.graph.get(parent_posn,child_posn) or INFINITY)
        if self.cost_values[child_posn[0]][child_posn[1]] > total_cost:
            value = min(total_cost, 255)
            color = color_map(value)
            #print color
            self.costs[child_posn[0]][child_posn[1]][0] = color[0]*255
            self.costs[child_posn[0]][child_posn[1]][1] = color[1]*255
            self.costs[child_posn[0]][child_posn[1]][2] = color[2]*255

        return total_cost

    def total_path_cost(self,path):
        prior_node = path[0]
        cost_so_far = 0
        for next_node in path[1:]:
            #print "parent_posn",prior_node.state, " to child_posn",
            # next_node.state," = ", cost_so_far, " + ",
            # self.graph.get(prior_node.state, next_node.state)

            cost_so_far += self.graph.get(prior_node.state, next_node.state)
            prior_node = next_node
        return cost_so_far

    # Now define the different heuristics we will use during this search
    #
    def h(self, node):
        "h function is straight-line distance from a node's state to goal."
        locs = getattr(self.graph, 'locations', None)
        if locs:
            return int(distance(locs[node.state], locs[self.goal]))

        return INFINITY

    def h_x_distance(self,node):
        return abs(node.state[0] - self.goal[0])

    def h_y_distance(self,node):
        return abs(node.state[1] - self.goal[1])

    def h_manhattan(self,node):
        return self.h_x_distance(node) + self.h_y_distance(node)

    def h_euclid(self,node):
        dx = self.h_x_distance(node)
        dy = self.h_y_distance(node)
        dist = np.sqrt(dx*dx + dy*dy)
        return int(dist)

    def h_euclid2(self,node):
        dx = self.h_x_distance(node)
        dy = self.h_y_distance(node)
        dist = 2*np.sqrt(dx*dx + dy*dy)
        return int(dist)

    def h_euclid3(self,node):
        dx = self.h_x_distance(node)
        dy = self.h_y_distance(node)
        dist = 3*np.sqrt(dx*dx + dy*dy)
        return int(dist)

    def h_euclid025(self,node):
        dx = self.h_x_distance(node)
        dy = self.h_y_distance(node)
        dist = 0.25*np.sqrt(dx*dx + dy*dy)
        return int(dist)

    def h_euclid05(self,node):
        dx = self.h_x_distance(node)
        dy = self.h_y_distance(node)
        dist = 0.5*np.sqrt(dx*dx + dy*dy)
        return int(dist)


# This is the main part of the demo program
map_loader = MapLoader() # Create a MapLoader to load the world map from a simple image

base_image = "simple"  # This is the base file name of the input image for map generation
map_loader.add_frame(".",base_image+".png")

scale = 0.1
map_loader.create_map(scale) # Discretize the map based on the the scaling factor

# Create a big version of discretized map for better visualization
big_map = cv2.resize(map_loader.map, (0,0),fx=(1.0/scale),
                     fy=(1.0/scale), interpolation=cv2.INTER_NEAREST)

cv2.imshow("Image",map_loader.image)
cv2.imshow("Map",  map_loader.map)
cv2.imshow("Big",  big_map)

TARGET_DIR = "output"
if not os.path.exists(TARGET_DIR):
    print("Creating target directory <",TARGET_DIR,"> ...")
    try:
        os.makedirs(TARGET_DIR)
    except OSError:
        print("Failed to create target path!")
        sys.exit()

print("Writing the base images ...")
cv2.imwrite(TARGET_DIR+"/"+base_image+"_img.png",map_loader.image)
cv2.imwrite(TARGET_DIR+"/"+base_image+"_map.png",map_loader.map)
cv2.imwrite(TARGET_DIR+"/"+base_image+"_big_map.png",big_map)

print("Wait for key input...")
#cv2.waitKey()


print("Doing the search ...")
grid = UndirectedGraph()  # Using Russell and Norvig code

start=(4, 4)
goal=(28, 24)

# Set up the planning problem instance that defines allowable transitions
problem2 = GridProblem(start, goal, grid, map_loader.map,scale)


# Define the test cases we want to run

tests = {    "depth_first":   (depth_first_graph_search, None),
             "breadth_first": (breadth_first_graph_search, None),
             "uniform_cost":  (uniform_cost_search, None),
             "astar_search_euclid":    (astar_search, problem2.h_euclid),
             "astar_search_euclid2":    (astar_search, problem2.h_euclid2),
             "astar_search_euclid3":   (astar_search,problem2.h_euclid3),
             "astar_search_euclid025": (astar_search,problem2.h_euclid025),
             "astar_search_euclid05":  (astar_search, problem2.h_euclid05),
             "astar_search_dx":        (astar_search, problem2.h_x_distance),
             "astar_search_dy":        (astar_search, problem2.h_y_distance),
             "astar_search_manhattan": (astar_search, problem2.h_manhattan),
             "greedy_search_euclid":   (greedy_best_first_graph_search, problem2.h_euclid2),
             "greedy_search_dx":       (greedy_best_first_graph_search, problem2.h_x_distance),
             "greedy_search_dy":       (greedy_best_first_graph_search, problem2.h_y_distance),
             "greedy_search_manhattan": (greedy_best_first_graph_search, problem2.h_manhattan),
         }
for test, planner_setup in tests.items():

    print("Set up the "+test+" ...")
    file_name = os.path.join(TARGET_DIR, f"{test}_{base_image}")

    print("     output to ", file_name)

    video_encoder = VideoEncoder(file_name, map_loader.map, frame_rate = 30.0,
                                 fps_factor=1.0, comp_height=1.0/scale, comp_width=2.0/scale)

    # Update the video coder for new files
    problem2.reset_encoder( video_encoder, map_loader.map)

    # Load the correct grid search algorithm and heuristics
    print("------------- call planner ---------------------")
    planner, heuristic = planner_setup
    if heuristic is None:
        result, max_frontier_size = planner(problem2)
    else:
        result, max_frontier_size = planner(problem2, heuristic)
    print("------------- done! ---------------------")


    #result = depth_first_graph_search(problem2)
    #result = breadth_first_search(problem2)
    #result = uniform_cost_search(problem2)
    #@result = astar_search(problem2, h=problem2.h_euclid)#manhattan)#y_distance)
    with open(file_name+'.txt','w') as ftxt:
        print("     Result=",result)
        print("     expansions = ",problem2.expansion)
        ftxt.write("expansions = "+str(problem2.expansion)+"\n")
        ftxt.write("max frontier = "+str(max_frontier_size)+"\n")
        if result is not None:
            path = result.path()
            ftxt.write("path cost="+str(problem2.total_path_cost(path))+"\n")
            ftxt.write("Path="+str(path)+"\n")
            print("path cost=",problem2.total_path_cost(path))
            print("Path=",path)
            print("Plotting path ...")
            map_loader.plot_path(path, 1.0)# scale)
            big_path = cv2.resize(map_loader.path, (0,0),fx=(1.0/scale), fy=(1.0/scale), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Path",big_path)
            cv2.imwrite(file_name+"_path.png",big_path)
        else:
            ftxt.write('no path!')

    print("     Close the video ...")
    problem2.video_encoder.release()


    cv2.waitKey(500)
