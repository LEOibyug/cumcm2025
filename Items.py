import numpy as np
import math

class Item:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.display = True
    def update(self, time_step):
        pass

class Plot(Item):
    def __init__(self, pos):
        super().__init__(pos)

class Missile(Item):
    Missile_SPEED = 300

    def __init__(self, pos, target,id):
        super().__init__(pos)
        self.target = target
        self.speed = Missile.Missile_SPEED
        self.id = id
    
    def update(self, time_step):
        direction_vector = self.target.pos - self.pos
        distance = np.linalg.norm(direction_vector)
        normalized_direction = direction_vector / distance
        self.pos = self.pos + normalized_direction * self.speed * time_step

class Smoke(Item):
    def __init__(self,pos,clock):
        super().__init__(pos)
        self.clock = clock
        self.display = False
        self.sustain = 20
    
    def update(self, time_step):
        if self.display:
            self.sustain -= time_step
            if self.sustain <= 0:
                self.display = False
            self.pos[2] -= 3 * time_step
        elif self.sustain <= 0:
            pass
        else:
            self.clock -= time_step
            if self.clock < 0:
                self.display = True
                print("Smoke is now active")
    
    
class Drone(Item):
    def __init__(self,pos, direction, speed,id):
        super().__init__(pos) 
        self.direction = np.array(direction, dtype=float)
        self.direction[2] = 0 #等高飞行
        self.direction = self.direction / np.linalg.norm(self.direction)
        self.speed = speed
        self.id = id
        
    def update(self, time_step):
        self.pos = self.pos + self.direction * self.speed * time_step
    
    def drop(self,clock):
        smoke_pos = self.pos + self.direction * self.speed * clock - np.array([0, 0, 0.5*9.8*clock**2])
        return Smoke(smoke_pos,clock)
    
class Volume(Item):
    # pos 为圆柱中心
    def __init__(self, pos, radius, height):
        super().__init__(pos)
        self.radius = radius
        self.height = height
    
    def get_sample(self):
        samples = []
        z_top = self.pos[2] + self.height / 2
        # 1. 圆柱侧面采样
        for z_offset in np.arange(-self.height / 2, self.height / 2 + 1,1): 
            current_z = self.pos[2] + z_offset
            if current_z > z_top:
                current_z = z_top 
            for i in range(20):
                angle = 2 * np.pi * i / 20
                x = self.pos[0] + self.radius * np.cos(angle)
                y = self.pos[1] + self.radius * np.sin(angle)
                samples.append(Plot(np.array([x, y, current_z])))
        
        # 2. 上表面采样
        x_coords = np.arange(self.pos[0] - self.radius, self.pos[0] + self.radius + 1,1)
        y_coords = np.arange(self.pos[1] - self.radius, self.pos[1] + self.radius + 1,1)
        for x_current in x_coords:
            for y_current in y_coords:
                distance_from_center_sq = (x_current - self.pos[0])**2 + (y_current - self.pos[1])**2
                if distance_from_center_sq <= self.radius**2:
                    samples.append(Plot(np.array([x_current, y_current, z_top])))
        return samples