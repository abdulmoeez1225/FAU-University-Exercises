from cProfile import label
import json
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
            
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)
        self.__nsamples = len(self.labels)
        self.__epoch = 0                            
        self.__batch_num = 0                        

        if self.batch_size <= 0 or self.batch_size > self.__nsamples:
            self.batch_size = self.__nsamples

        self.__map = np.arange(self.__nsamples)     
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        if self.__batch_num * self.batch_size >= self.__nsamples:
            self.__epoch += 1
            self.__batch_num = 0

        if self.__batch_num == 0 and self.shuffle:
            np.random.shuffle(self.__map)

        images = np.zeros((self.batch_size, *self.image_size), dtype=np.float32)
        labels = np.zeros(self.batch_size, dtype=int)

        start_index = self.__batch_num * self.batch_size
        end_index = start_index + self.batch_size

        if end_index <= self.__nsamples:
            for i in range(self.batch_size):
                img_index = self.__map[start_index + i]
                image_path = f"{self.file_path}/{img_index}.npy"
                images[i] = self.augment(np.load(image_path))
                labels[i] = self.labels[str(img_index)]
            self.__batch_num += 1

        else:
            remaining = self.__nsamples - start_index
            for i in range(remaining):
                img_index = self.__map[start_index + i]
                image_path = f"{self.file_path}/{img_index}.npy"
                images[i] = self.augment(np.load(image_path))
                labels[i] = self.labels[str(img_index)]
            for i in range(self.batch_size - remaining):
                img_index = self.__map[i]
                image_path = f"{self.file_path}/{img_index}.npy"
                images[remaining + i] = self.augment(np.load(image_path))
                labels[remaining + i] = self.labels[str(img_index)]
            self.__batch_num = 1  
        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        if img.shape != tuple(self.image_size):
            img = np.resize(img, self.image_size)
        if self.mirroring:
            if np.random.rand() > 0.5:
                img = np.fliplr(img)  
            if np.random.rand() > 0.5:
                img = np.flipud(img)

        if self.rotation:
            rotations = np.random.randint(0, 4) 
            img = np.rot90(img, k=rotations)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.__epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.labels[str(x)]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        imgs, labs = self.next()
        fig = plt.figure(figsize=(10,10))
        cols = 3
        rows = self.batch_size // 3 + (1 if self.batch_size % 3 else 0)

        for i in range(1, self.batch_size+1):
            img = imgs[i-1]
            lab = self.class_dict[labs[i-1]]
            fig.add_subplot(rows, cols, i)
            plt.imshow(img.astype('uint8'))
            plt.xticks([])
            plt.yticks([])
            plt.title(lab)
        plt.show()

