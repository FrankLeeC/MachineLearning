import numpy as np
from PIL import Image 

def convolute():
    a = np.array([[[1,2], [2,3]], [[4,2], [2,3]]])
    b = np.array([[[2,2], [3,3]], [[2,2], [3,3]]])
    c = a * b
    print(a.shape)
    for _, k in enumerate(a):
        print(k)
        print('---')
    print('---------------')
    print(c)
    print(np.sum(c))

def image():
    img = np.array(Image.open('/Users/frank/Pictures/sinnosuke.jpeg'))
    print(img.shape)

def list_join():
    l = [[i+2 for i in range(j)] for j in range(10)]
    print(l)

if __name__ == "__main__":
    # convolute()
    # image()
    list_join()