import numpy as np
import matplotlib.pyplot as plt

class Checker:
    
    def __init__(self, resolution, tile_size):
        self.tile_size = tile_size
        self.resolution = resolution
        
    def draw(self):
        amount = int(self.tile_size * 2)
    
        tile = np.zeros((amount, amount), dtype=int)
        tile[self.tile_size:, :self.tile_size] = 1
        tile[:self.tile_size, self.tile_size:] = 1

        num_tiles = self.resolution // amount
        full_pattern = np.tile(tile, (num_tiles, num_tiles))

        self.output = full_pattern.copy()
        return full_pattern
           
    def show(self):
        output = self.draw()
        plt.imshow(output, cmap = "gray")
        plt.show()

class Circle:
    
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        
    def draw(self):
        res = self.resolution
        cx, cy = self.position
        # x = np.linspace(0,res-1,res).reshape(1,res)
        # y = np.linspace(0,res-1,res).reshape(res,1)
        # distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        x = np.arange(res)
        y = np.arange(res)
        xx, yy = np.meshgrid(x, y)
        distance = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        
        circ = distance <= self.radius
        self.output = circ.copy()
        return circ
    
    def show(self):
        output = self.draw()
        plt.imshow(output, cmap = "gray")
        plt.show()

class Spectrum:
    
    def __init__(self, resolution):
        self.resolution = resolution
        
    def draw(self):
        size = self.resolution 
        color_image = np.zeros([size, size, 3])
        color_image[:, :, 0] = np.linspace(0, 1, size)
        color_image[:, :, 1] = np.linspace(0, 1, size).reshape(size, 1)
        color_image[:, :, 2] = np.linspace(1, 0, size)

        self.output = color_image.copy()
        return color_image
    
    def show(self):
        output = self.draw()
        plt.imshow(output)
        plt.show()

