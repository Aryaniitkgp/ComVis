import cv2 as cv
import numpy as np

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
marker_id = 21
size = 200
border = 50

img = cv.aruco.generateImageMarker(aruco_dict, marker_id, size)
img = cv.copyMakeBorder(img, border, border, border, border, cv.BORDER_CONSTANT, value=255)
output_path = f"aruco_marker_{marker_id}.png"
cv.imwrite(output_path, img)
print(f"Saved {output_path}")

image = cv.imread(output_path)
if image is None:
    raise FileNotFoundError(f"Could not read {output_path}")

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
corners, ids, rejected = cv.aruco.detectMarkers(gray, aruco_dict)
if ids is not None and marker_id in ids:
    marker_index = np.where(ids == marker_id)[0][0]
    detected_corners = corners[marker_index:marker_index + 1]
    detected_ids = ids[marker_index:marker_index + 1]

    cv.aruco.drawDetectedMarkers(image, detected_corners, detected_ids)

    camera_matrix = np.eye(3, 3, dtype=np.float32)
    camera_matrix[0, 2] = image.shape[1] / 2.0
    camera_matrix[1, 2] = image.shape[0] / 2.0
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    marker_length = 0.05

    rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
        detected_corners, marker_length, camera_matrix, dist_coeffs
    )

    axis_length = 0.05
    axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)
    imgpts, _ = cv.projectPoints(axis_points, rvecs[0], tvecs[0], camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    origin = tuple(imgpts[0].ravel())
    x_axis = tuple(imgpts[1].ravel())
    y_axis = tuple(imgpts[2].ravel())
    z_axis = tuple(imgpts[3].ravel())

    cv.line(image, origin, x_axis, (0, 0, 255), 2)
    cv.line(image, origin, y_axis, (0, 255, 0), 2)
    cv.line(image, origin, z_axis, (255, 0, 0), 2)

    print(f"Detected marker ID: {marker_id}")
    print(f"Rotation vector: {rvecs[0].flatten()}")
    print(f"Translation vector: {tvecs[0].flatten()}")

    pose_output_path = f"aruco_pose_{marker_id}.png"
    cv.imwrite(pose_output_path, image)
    print(f"Saved pose image: {pose_output_path}")

    try:
        cv.imshow("Detected Marker Pose", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    except cv.error:
        print("Display not available; saved image instead.")
else:
    print("Marker not detected.")