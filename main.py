from numpy import ceil
from numpy.lib.function_base import quantile
import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
Kuwahara filter implementation using HSV color space
"""

# Read the image - Notice that OpenCV reads the images as BRG instead of RGB
img = cv2.imread('image.jpg')
print(img.shape)
""" plt.imshow(img)
plt.show() """
""" img = cv2.resize(img, (img.shape[:2]/2))
plt.imshow(img)
plt.show() """

# Convert the BRG image to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
""" plt.imshow(img)
plt.show()
 """
# Convert the RGB image to HSV
img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
""" plt.imshow(img)
plt.show()

print(img.shape)
print(img[0]) """



"""
IMPLEMENTATION

"""


def kuwahara_filter(image, win_size):
    quadrant_size = int(ceil(win_size/2))
    print(quadrant_size)
    img = image.copy()


    clamp_y = lambda y1, y2: (max(0,y1), min(y2,img.shape[0]-1))
    clamp_x = lambda x1, x2: (max(0,x1), min(x2,img.shape[1]-1))

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # find the top left corner of the filter on image
            tl_x = x - int(win_size/2)
            tl_y = y - int(win_size/2)
            #print(clamp_y(tl_y, tl_y + quadrant_size))
            # find the quarters of the image
            q1 = [clamp_y(tl_y, tl_y + quadrant_size), clamp_x(tl_x, tl_x + quadrant_size)]
            q2 = [clamp_y(tl_y, tl_y + quadrant_size), clamp_x(tl_x, tl_x + quadrant_size)]
            q3 = [clamp_y(tl_y + quadrant_size, tl_y + win_size), clamp_x(tl_x, tl_x + quadrant_size)]
            q4 = [clamp_y(tl_y+ quadrant_size, tl_y + win_size), clamp_x(tl_x + quadrant_size, tl_x + win_size)]

            std_1 = image[q1[0][0]:q1[0][1], q1[1][0]:q1[1][1],2].std()
            std_2 = image[q2[0][0]:q2[0][1], q2[1][0]:q2[1][1],2].std()
            std_3 = image[q3[0][0]:q3[0][1], q3[1][0]:q3[1][1],2].std()
            std_4 = image[q4[0][0]:q4[0][1], q4[1][0]:q4[1][1],2].std()

            quads = [q1, q2, q3, q4]
            min_std = np.argmin([std_1, std_2, std_3, std_4])

            selected_quad = quads[min_std]
            #print(image[selected_quad[0][0]:selected_quad[0][1], selected_quad[1][0]:selected_quad[1][1],2])
            #if image[selected_quad[0][0]:selected_quad[0][1], selected_quad[1][0]:selected_quad[1][1],2].size != np.array([]):
            new_pixel_val = image[selected_quad[0][0]:selected_quad[0][1], selected_quad[1][0]:selected_quad[1][1],2].mean()
            img[y,x,2] = new_pixel_val
            #print(x,y)
            

    return img


filtered_img = kuwahara_filter(img, 9)
plt.imshow(filtered_img)
plt.show()
filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_HSV2RGB)
plt.imshow(filtered_img)
plt.show()
