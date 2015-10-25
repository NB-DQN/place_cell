import numpy as np
import math
import random
import sys

class DatasetGenerator:
    def __init__(self, size):
        self.size = size
        self.current_coordinate = (0, 0)

    def coordinate_id(self, coordinate=None):
        if coordinate is None:
            coordinate = self.current_coordinate
        return coordinate[0] + int(coordinate[1]) * self.size[0]

    def get_coordinate_from_id(self, cid):
        x = cid % self.size[0]
        y = (cid - x) / self.size[0]
        return (x, y)

    def visual_targets(self):
        return [ \
            (self.size[0], self.size[1]), \
            (          -1, self.size[1]), \
            (          -1,           -1), \
            (self.size[0],           -1)]

    def visual_image(self, coordinate=None):
        if coordinate == None:
            coordinate = self.current_coordinate
        #DEGREE_PER_DOT = 3
        DEGREE_PER_DOT = 6

        image = np.zeros(360 / DEGREE_PER_DOT)
        for target in self.visual_targets():
            distance = math.sqrt( \
                (coordinate[0] - target[0]) ** 2 + \
                (coordinate[1] - target[1]) ** 2)
            visual_width = math.degrees(math.atan(0.5 / distance))
            angle = math.degrees(math.atan2(target[1] - coordinate[1], target[0] - coordinate[0]))
            if angle < 0:
                angle += 360

            visual_range = [round(i / DEGREE_PER_DOT) for i in [angle - visual_width, angle + visual_width]]
            image[visual_range[0]:(visual_range[1] + 1)] = 1
        return image
    
    def generate_seq(self, seq_length):
        image = []
        directions = []
        velocitys =[]
        coordinates = []
        #ang_option = 60
        ang_option = 90
        vel_option = 1
        maze_size = (9,9)

        image.append(self.visual_image())
        coordinates.append((0, 0))

        for i in range(0, seq_length):
                        
            #  movement  -> ang -> velocity
            #print(self.current_coordinate)

            #vel_choice = [x*vel_option for x in range(0, 5)]
            vel_choice = [x*vel_option for x in range(1, 2)]
            vel = random.choice(vel_choice)

            ang_choice = [x for x in range(0, 360, ang_option) ]
            ang = random.choice(ang_choice)
            
           
            #print(ang)
            #print(math.maze_size[1] - self.current_coordinate[1])cos(float(ang)/180*math.pi))

            # move
            #print(self.current_coordinate)
            if  0 < ang < 90:
                X_dis = (maze_size[0] -1 - self.current_coordinate[0])/math.cos(float(ang)/180*math.pi)
                Y_dis = (maze_size[1] -1 - self.current_coordinate[1])/math.sin(float(ang)/180*math.pi)
            if 90 < ang < 180:
                X_dis = (   0            + self.current_coordinate[0])/math.cos(float(180-ang)/180*math.pi)
                Y_dis = (maze_size[1] -1 - self.current_coordinate[1])/math.sin(float(180-ang)/180*math.pi)
            if 180 < ang < 270:
                X_dis = (   0            + self.current_coordinate[0])/math.cos(float(ang-180)/180*math.pi)
                Y_dis = (   0            + self.current_coordinate[1])/math.sin(float(ang-180)/180*math.pi)
            if 270 < ang < 360:
                X_dis = (maze_size[0] -1 - self.current_coordinate[0])/math.cos(float(360-ang)/180*math.pi)
                Y_dis = (   0            + self.current_coordinate[1])/math.sin(float(360-ang)/180*math.pi)
            if ang == 0:
                X_dis = (maze_size[0] -1 - self.current_coordinate[0])
                Y_dis = 100
            if ang == 90:
                X_dis = 100
                Y_dis = (maze_size[1] -1 - self.current_coordinate[1])
            if ang == 180:
                X_dis = (   0            + self.current_coordinate[0])
                Y_dis = 100
            if ang == 270:
                X_dis = 100
                Y_dis = (   0            + self.current_coordinate[1])

            if vel > min(math.fabs(X_dis),math.fabs(Y_dis)):
                #print(min(math.fabs(X_dis),math.fabs(Y_dis)))
                #vel = min(math.fabs(X_dis),math.fabs(Y_dis)) - min(math.fabs(X_dis),math.fabs(Y_dis))%0.25
                ang = ang + 180
                if ang >= 360:
                    ang = ang - 360
                
            #print(ang)
            #print(vel)

            self.current_coordinate = (self.current_coordinate[0] + vel * round(math.cos(float(ang)/180*math.pi)),
                                       self.current_coordinate[1] + vel * round(math.sin(float(ang)/180*math.pi)))

            #print(self.current_coordinate)
            
            if self.current_coordinate[0] < -0.1 or self.current_coordinate[1] < -0.1 or self.current_coordinate[0] > 8 or  self.current_coordinate[1] > 8:
              sys.exit("over_maze")
              
            direction = [0] * (360/ang_option) # This direction needs modified later
            direction[int(ang/ang_option)] = 1
            #print(direction)
            #velocity  = [0] * 2
            #velocity[int(vel/vel_option)-1]  = 1
            velocity =[1] * 1
            #print(velocity)

            directions.append(direction)
            #print(directions)
            velocitys.append(velocity)
            image.append(self.visual_image())
            coordinates.append(self.current_coordinate)

        input = []
        for i in range(len(directions)):
        #for i in range(0, seq_length):
            #input.append(directions[i] + velocitys[i] + image[i].tolist())
            #print(input)
            input.append(directions[i] + image[i].tolist())

        return { 'input': input, 'output': image[1:], 'coordinates': coordinates[1:] }

