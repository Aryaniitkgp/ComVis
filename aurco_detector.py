import cv2 as cv

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
marker_id = 21
size = 200

img = cv.aruco.generateImageMarker(aruco_dict, marker_id, size)
output_path = f"aruco_marker_{marker_id}.png"
cv.imwrite(output_path, img)
print(f"Saved {output_path}")