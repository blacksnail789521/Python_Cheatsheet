import numpy as np
from PIL import Image, ImageDraw


# Create defect_map.
defect_map = np.zeros((301, 301))
X_VALUE = (( np.asarray([0.9, -8.7, -5.9]) + 15 ) * 10).astype(int)
Y_VALUE = (( np.asarray([-0.1, 0, 10.0]) + 15 ) * 10).astype(int)
for i in range(X_VALUE.shape[0]):
    defect_map[X_VALUE[i]][Y_VALUE[i]] = 1



# Initialization.
img = Image.new("RGB", (defect_map.shape[0], defect_map.shape[1]))
pix = img.load()
for x in range(defect_map.shape[0]):
    for y in range(defect_map.shape[1]):
        pix[x,y] = (255, 255, 255)

# Draw circle.
draw = ImageDraw.Draw(img)
draw.ellipse((0, 0, 301, 301), outline = "gray")

# Draw defect.
for x in range(defect_map.shape[0]):
    for y in range(defect_map.shape[1]):
        if defect_map[x][y] == 1:
            pix[x,y] = (255, 0, 0)

img.show()