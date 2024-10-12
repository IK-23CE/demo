import streamlit as st
from skimage.transform import (hough_line, hough_line_peaks)
import numpy as np
import cv2
from matplotlib import pyplot as plt


# waitKey() waits for a key press to close the window and 0 specifies indefinite loop
# cv2.destroyAllWindows() simply destroys all the windows we created.
image1 = cv2.imread('truss.jpg', 0) 
image1 = ~image1
plt.imshow(image1, cmap='gray')
plt.savefig('s1.png')
is1 = cv2.imread("s1.png", cv2.IMREAD_COLOR)
st.image(is1)
# convert image to gray scale image 
if len(image1.shape) == 3 and image1.shape[2] == 3:
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
elif len(image1.shape) == 2:
    gray = image1  # Already grayscale
else:
    raise ValueError("Unexpected number of channels in the image.")


tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)

hspace, theta, dist = hough_line(image1, tested_angles)

plt.figure(figsize=(10,10))
plt.imshow(hspace) 
plt.savefig('s2.png')
is2 = cv2.imread("s2.png", cv2.IMREAD_COLOR)
st.image(is2)

h, q, d = hough_line_peaks(hspace, theta, dist)

angle_list=[]  

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image1, cmap='gray')
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(np.log(1 + hspace),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), dist[-1], dist[0]],
             cmap='gray', aspect=1/1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(image1, cmap='gray')

origin = np.array((0, image1.shape[1]))

for _, angle, dist in zip(*hough_line_peaks(hspace, theta, dist)):
    angle_list.append(angle) 
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    ax[2].plot(origin, (y0, y1), '-r')
ax[2].set_xlim(origin)
ax[2].set_ylim((image1.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

plt.tight_layout()
plt.show()
plt.savefig('s3.png')
is3 = cv2.imread("s3.png", cv2.IMREAD_COLOR)
st.image(is3)
plt.close()

# detect corners with the goodFeaturesToTrack function. 
corners = cv2.goodFeaturesToTrack(gray,100, 0.5, 10) 
corners = np.intp(corners)
corn=corners
Alst=['a','b','c','d','e','f','g','h']
ij=0
# we iterate through each corner, 
# making a circle at each point that we think is a corner. 
for i in corners:
    x, y = i.ravel()
    cv2.circle(image1, (x, y), 3, 255, -1)

corn.sort()
kpil_txt=st.markdown("0")
kpil_txt.write(f'<h1>{corn}</h1>')
for ki in corn:
    xi,yi=ki.ravel()
    plt.text(xi,yi,Alst[ij])
    ij+=1
    
# count=0

# Initialize a dictionary to count occurrences of the second element within range
second_element_counts = {}

# Traverse the array and count occurrences within the range of +/- 2 from each other
for i in range(corners.shape[0]):
    for j in range(i + 1, corners.shape[0]):
        y1 = corners[i, 0, 1]
        y2 = corners[j, 0, 1]
        if abs(y1 - y2) <= 2:
            if y1 in second_element_counts:
                second_element_counts[y1] += 1
            else:
                second_element_counts[y1] = 1

            if y2 in second_element_counts:
                second_element_counts[y2] += 1
            else:
                second_element_counts[y2] = 1

# Initialize count variable for elements occurring more than two times within range
count = 0

# Check which elements occur more than two times and increase count
for key, value in second_element_counts.items():
    if value > 2:
        count += 1

# Print the count of second elements occurring more than two times within range


# Print the count of second elements occurring more than two times
print("Count of second elements occurring more than two times:", count)

plt.imshow(image1)
plt.show()
plt.savefig('s4.png')
is4 = cv2.imread("s4.png", cv2.IMREAD_COLOR)
st.image(is4)
angles = [a*180/np.pi for a in angle_list]

angle_difference = np.max(angles) - np.min(angles)
print(180 - angle_difference)
