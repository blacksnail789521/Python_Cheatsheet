import numpy as np
from PIL import Image, ImageDraw
from skimage.transform import rotate


def draw_wafer(defect_map):
    
    # Initialization.
    img = Image.new("RGB", (defect_map.shape[0], defect_map.shape[1]))
    pix = img.load()
    for x in range(defect_map.shape[0]):
        for y in range(defect_map.shape[1]):
            pix[x, y] = (255, 255, 255)
    
    # Draw circle.
    draw = ImageDraw.Draw(img)
    draw.ellipse((0, 0, 301, 301), outline = "gray")
    
    # Draw defect.
    for x in range(defect_map.shape[0]):
        for y in range(defect_map.shape[1]):
            if defect_map[x][y] == 1:
                pix[x, y] = (255, 0, 0)
    
    img.show()


def create_defect_map_with_a_check_mark():
    
    defect_map = np.zeros((310, 301))
    # -15 ~ +15 (for being inside the circle, it should be -10 ~ +10)
    x_array, y_array = np.array([]), np.array([])
    x_array = np.append( x_array, ((np.linspace(-10,  0, num = 200) + 15) * 10) )
    y_array = np.append( y_array, ((np.linspace(  2, 10, num = 200) + 15) * 10) )
    x_array = np.append( x_array, ((np.linspace(  0, 10, num = 200) + 15) * 10) )
    y_array = np.append( y_array, ((np.linspace(-10, 10, num = 200)[::-1] + 15) * 10) )
    
    for i in range(x_array.shape[0]):
        defect_map[ int(x_array[i]) ][ int(y_array[i]) ] = 1
    
    return defect_map

defect_map = create_defect_map_with_a_check_mark()

# Draw defect_map.
draw_wafer(defect_map)

# Flip.
flipped_defect_map = np.flipud(defect_map)
draw_wafer(flipped_defect_map)

# Rotate.
rotated_defect_map = rotate(defect_map, 180, preserve_range = True)
draw_wafer(rotated_defect_map)