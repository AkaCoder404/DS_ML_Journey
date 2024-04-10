# Practice Convolutions on Image
import numpy as np
import cv2

image = cv2.imread('images/lena_256.jpg')

# Color Image
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2RGB)
print(image.shape) 


# 3x3 Convolutional Kernel Vertical Edge Detection
conv_kernel_3x3_edge_vertical = np.array([[1, 0, -1],
                                          [1, 0, -1],
                                          [1, 0, -1]], np.float32)

# 3x3 Convolutional Kernel Horizontal Edge Detection
conv_kernel_3x3_edge_horizontal = np.array([[1, 1, 1],
                                            [0, 0, 0],
                                            [-1, -1, -1]], np.float32)

# 3x3 Convolutional Kernel Sharpen
conv_kernel_3x3_sharpen = np.array([[0, -1, 0],
                                     [-1, 5, -1],
                                     [0, -1, 0]], np.float32)

# 3x3 Convolutional Kernel Box Blur
conv_kernel_3x3_box_blur = np.array([[1, 1, 1],
                                     [1, 1, 1],
                                     [1, 1, 1]], np.float32) / 9

# 3x3 Convolutional Kernel Gaussian Blur
conv_kernel_3x3_gaussian_blur = np.array([[1, 2, 1],
                                          [2, 4, 2],
                                          [1, 2, 1]], np.float32) / 16

# 3x3 Convolutional Kernel Emboss
conv_kernel_3x3_emboss = np.array([[-2, -1, 0],
                                   [-1, 1, 1],
                                   [0, 1, 2]], np.float32)

# 3x3 Convolutional Kernel Outline
conv_kernel_3x3_outline = np.array([[-1, -1, -1],
                                    [-1, 8, -1],
                                    [-1, -1, -1]], np.float32)

# 3x3 Convolutional Kernel Identity
conv_kernel_3x3_identity = np.array([[0, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]], np.float32)

# 4x4 Convolutional Kernel 
conv_kernel_4x4 = np.array([[0, 1, -1, 0],
                            [1, 3, -3, -1],
                            [1, 3, -3, -1],
                            [0, 1, -1, 0]], np.float32) / 3

# 3x3 Convolutional Kernel 
conv_kernel_3x3_new = np.array([[-1, -1, 2],
                             [-1, 2, 1],
                             [2, 1, 1]], np.float32)

image4x4 = cv2.filter2D(image, -1, conv_kernel_4x4)
cv2.imwrite('images/lena_256_4x4.jpg', image4x4)




# Apply all filters and stitch them to one big image
image1 = cv2.filter2D(image, -1, conv_kernel_3x3_edge_vertical)
image2 = cv2.filter2D(image, -1, conv_kernel_3x3_edge_horizontal)
image3 = cv2.filter2D(image, -1, conv_kernel_3x3_sharpen)
image4 = cv2.filter2D(image, -1, conv_kernel_3x3_box_blur)
image5 = cv2.filter2D(image, -1, conv_kernel_3x3_gaussian_blur)
image6 = cv2.filter2D(image, -1, conv_kernel_3x3_emboss)
image7 = cv2.filter2D(image, -1, conv_kernel_3x3_outline)
image8 = cv2.filter2D(image, -1, conv_kernel_3x3_identity)
image9 = cv2.filter2D(image, -1, conv_kernel_3x3_new)

# Add a label to each image
image1 = cv2.putText(image1, 'Vertical Edge Detection', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
image2 = cv2.putText(image2, 'Horizontal Edge Detection', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
image3 = cv2.putText(image3, 'Sharpen', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
image4 = cv2.putText(image4, 'Box Blur', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
image5 = cv2.putText(image5, 'Gaussian Blur', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
image6 = cv2.putText(image6, 'Emboss', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
image7 = cv2.putText(image7, 'Outline', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
image8 = cv2.putText(image8, 'Identity', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
image9 = cv2.putText(image9, 'New', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# One big image with 4 images per row
imageA = np.concatenate((image1, image2, image3, image4), axis=1)
imageB = np.concatenate((image5, image6, image7, image8), axis=1)
imageC = np.concatenate((image9, image9, image9, image9), axis=1)
image = np.concatenate((imageA, imageB, imageC), axis=0)

# Save the output image
cv2.imwrite('images/lena_output.jpg', image)