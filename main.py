from numpy import ceil, float16
from numpy.lib.function_base import quantile
import cv2
import numpy as np






def kuwahara_filter(image, win_size):
    """
    Kuwahara filter implementation using HSV color space
    """
    quadrant_size = int(ceil(win_size/2))
    img = image.copy()


    clamp_y = lambda y1, y2: (max(0, min(y1, img.shape[0]-1)), min(y2,img.shape[0]-1))
    clamp_x = lambda x1, x2: (max(0, min(x1, img.shape[1]-1)), min(x2,img.shape[1]-1))

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # find the top left corner of the filter on image
            tl_x = x - int(win_size/2)
            tl_y = y - int(win_size/2)

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
            if selected_quad[0][0] != selected_quad[0][1] and selected_quad[1][0] != selected_quad[1][1]:
                new_pixel_val = image[selected_quad[0][0]:selected_quad[0][1], selected_quad[1][0]:selected_quad[1][1],2].mean()
                img[y,x,2] = new_pixel_val


    return img


def gaussian_formula(x,y,sigma):
    from sklearn.preprocessing import normalize
    result = (1/(2*np.pi*(sigma**2)))*np.exp(-(x**2+y**2)/(2*(sigma**2)))
    # normalize the kernel to avoid cut off with
    # large sigma values
    return result/result.sum()


def gaussian_filter(image, filter_size, sigma=1):
    """
    Gaussian filter implementation using HSV color space
    """
    from itertools import product
    img = image[:,:,2]
    center = filter_size//2
    x,y = np.mgrid[0-center:filter_size-center, 0-center:filter_size-center]
    filter_ = gaussian_formula(x,y,sigma)

    # calculate the resulting image size
    # after applying gaussian filter, y coordinate
    new_img_height = image.shape[0] - filter_size + 1
    new_img_width = image.shape[1] - filter_size + 1

    # stack all possible windows in the image vertically
    # to apply the filter later on
    new_image = np.empty((new_img_height*new_img_width, filter_size**2))

    row = 0
    for i,j in product(range(new_img_height), range(new_img_width)):
        new_image[row,:] = np.ravel(img[i:i+filter_size, j:j+filter_size])
        row += 1

    filter_ = np.ravel(filter_)
    filtered_image = np.dot(new_image, filter_).reshape(new_img_height, new_img_width).astype(np.uint8)
    image[center:new_img_height+center,center:new_img_width+center,2] = filtered_image


    return image



def mean_filter(image, filter_size):
    from itertools import product

    # new image's dimensions since
    # the border values are discarded
    new_img_height = image.shape[0] - filter_size + 1
    new_img_width = image.shape[1] - filter_size + 1  

    # store each window in numpy array to
    # average them at once to obtain the
    # new pixel values
    # preserve color channels
    windows = np.empty((new_img_height*new_img_width, filter_size**2, 3))


    row = 0
    # store each window in "windows" as the filter
    # traces the image
    for i,j in product(range(new_img_height), range(new_img_width)):
        # simply, all pixels under the filter side to side
        # with their RGB channels
        windows[row,:,:] = np.ravel(image[i:i+filter_size, j:j+filter_size]).reshape(-1,3)
        row += 1

    filtered_image =np.empty((new_img_height*new_img_width,3))

    # calculate the mean for RGB values seperately for each window
    for r in range(new_img_height*new_img_width):
        filtered_image[r,:] = windows[r,:,:].mean(axis=0)
    
    # after calculating the means for each channel and row
    # in windows, reshape the array to obtain resultant
    # image with RGB channels
    filtered_image = filtered_image.reshape(new_img_height, new_img_width, 3).astype(np.uint8)

    return filtered_image




def filter_(image, filter_name, filter_size, sigma):
    import matplotlib.pyplot as plt
    filtered_image = np.empty((1,1))
    if filter_name == "mean":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filtered_image = mean_filter(image, filter_size)
    elif filter_name == "gaussian":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        filtered_image = gaussian_filter(image, filter_size, sigma)
        filtered_image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    elif filter_name == "kuwahara":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        filtered_image = kuwahara_filter(image, filter_size)
        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_HSV2RGB)
    
    # plot the image
    plt.imshow(filtered_image)
    plt.show()



def get_image(image_path):
    # Read the image - Notice that OpenCV reads the images as BRG instead of RGB
    return cv2.imread(image_path)


if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Apply filters to images')
    parser.add_argument("--image", default="sample_image.jpg",
                        metavar="path/to/image",
                        help="'image path'")
    parser.add_argument("--filter",required=True,
                        help="'Filter name: mean, gaussian or kuwahara'")
    parser.add_argument("--size",
                        default=5,
                        metavar="5",
                        help="'Filter size. Must be an odd number. Default value is 5")
    parser.add_argument("--sigma",
                        default=1,
                        help="'Sigma value for Gaussian filter. Default is 1")

    # parse arguments
    args = parser.parse_args()
    # get filter name
    filter_name = args.filter
    if filter_name != "mean" and filter_name != "gaussian" and filter_name != "kuwahara":
        raise ValueError("Please enter a valid filter name")
    if int(args.size)%2 == 0:
        raise ValueError("Please enter an odd number")
    
    filter_(get_image(args.image), filter_name, int(args.size), int(args.sigma))