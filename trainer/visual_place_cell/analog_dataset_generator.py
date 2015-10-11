import numpy as np
import math
import random

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

    def visual_image(self, cid=None):
        if cid is None:
            cid = self.coordinate_id()
        coordinate = self.get_coordinate_from_id(cid)

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
	ang_option = 60
	vel_option = 1

        image.append(self.visual_image())
        coordinates.append((0, 0))

        for i in range(0, seq_length):

            vel_choice = [x for x in range(0, 2, vel_option)]
            
	          # limit movement
	    if self.current_coordinate[0] >= self.size[0] - 1:
                ang_choice = [x for x in range(180, 360, ang_option)]
            if self.current_coordinate[0] < 1:
                ang_choice = [x for x in range(0, 181, ang_option)]
            if self.current_coordinate[1] >= self.size[1] - 1:
                ang_choice = [x for x in range(90, 271, ang_option)]
            if self.current_coordinate[1] < 1:
                ang_choice = [x for x in range(0, 91, ang_option)] + [x for x in range(270, 360, ang_option)]
            else:
		ang_choice = [x for x in range(0, 360, ang_option) ]
	
	    ang = random.choice(ang_choice) 
	    vel = random.choice(vel_choice)
	    
	    # move
	    self.current_coordinate = (self.current_coordinate[0] + vel * math.cos(ang),
				       self.current_coordinate[1] + vel * math.sin(ang))
          
      	    direction = [0] * (360/ang_option)
	    direction[ang/ang_option] = 1
            velocity  = [0] * (2/vel_option)
            velocity[vel/vel_option]  = 1

            directions.append(direction)
	    velocitys.append(velocity)
            image.append(self.visual_image())
            coordinates.append(self.current_coordinate)

        input = []
        #for i in range(len(directions)):
        for i in range(0, seq_length):
            input.append(directions[i] + velocitys[i] + image[i].tolist())

        return { 'input': input, 'output': image[1:], 'coordinates': coordinates[1:] }

