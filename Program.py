import cv2
from cv2 import aruco
import numpy as np
import os

# Get the directory where THIS script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print('BASE_DIR= ', BASE_DIR)

# Example: build a relative path
image_path = os.path.join(BASE_DIR, "ArucoMarker")
print('IMG_PATH= ', image_path)


def findArucoMarkers(wall_img, markerSize=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(wall_img, cv2.COLOR_BGR2GRAY)
    Key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDictionary = aruco.getPredefinedDictionary(Key)
    arucoParameters = aruco.DetectorParameters()
    boundbox, ids, rejected_marker = aruco.detectMarkers(imgGray, arucoDictionary, parameters=arucoParameters)
    if draw:
        aruco.drawDetectedMarkers(wall_img,boundbox)
    return [boundbox, ids]

def MarkerAugment(boundbox, id, wall_img, scenary, drawId=True):
    
    wall_width = wall_img.shape[0]
    wall_height = wall_img.shape[1]
    
    tli = int(boundbox[0][0][0]), int(boundbox[0][0][1])
    tri = int(boundbox[0][1][0]), int(boundbox[0][1][1])
    bri = int(boundbox[0][2][0]), int(boundbox[0][2][1])
    bli = int(boundbox[0][3][0]), int(boundbox[0][3][1])
    tl = boundbox[0][0][0],boundbox[0][0][1]
    tr = boundbox[0][1][0],boundbox[0][1][1]
    br = boundbox[0][2][0],boundbox[0][2][1]
    bl = boundbox[0][3][0],boundbox[0][3][1]
    # print(tl,tr,bl,br)
    
    cv2.circle(wall_img, tli, 5, (255,0,0), 2)
    cv2.circle(wall_img, tri, 5, (255,0,0), 2)
    cv2.circle(wall_img, bri, 5, (255,0,0), 2)
    cv2.circle(wall_img, bli, 5, (255,0,0), 2)
    cv2.imwrite(os.path.join(image_path, "wall_img.jpg"), wall_img)
    aruco_corners =  np.array([tl, tr, br, bl], dtype=np.float32)
    
    s=3
    center = np.mean(aruco_corners, axis=0)
    translated_corners=aruco_corners - center
    scaled_corners = translated_corners * s
    enlarged_corners = scaled_corners + center

    
    stl = [0 ,578]
    str = [3468 , 578]
    sbr = [3468 , 4046]
    sbl = [0 , 4046] 

    scenary_corners = np.float32([stl,str,sbr,sbl])
    matrix = cv2.getPerspectiveTransform(scenary_corners, enlarged_corners)
    warp_output = cv2.warpPerspective(scenary, matrix, (scenary.shape[1], scenary.shape[0]))
    
    final_output = place_scenary(warp_output, wall_img)
    cv2.imwrite(os.path.join(image_path, "final_output.jpg"), final_output)
    return final_output

def place_scenary(source_img,dest_img):
    cv2.imwrite(os.path.join(image_path, "source_img.jpg"), source_img)
    # Create masks for each channel of the source image
    b, g, r = cv2.split(source_img)
    ret, b_mask = cv2.threshold(b, 10, 255, cv2.THRESH_BINARY)
    ret, g_mask = cv2.threshold(g, 10, 255, cv2.THRESH_BINARY)
    ret, r_mask = cv2.threshold(r, 10, 255, cv2.THRESH_BINARY)

    # Combine the masks
    mask = cv2.bitwise_and(b_mask, cv2.bitwise_and(g_mask, r_mask))
    cv2.imwrite(os.path.join(image_path, "masked_img.jpg"), mask)
    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)
    cv2.imwrite(os.path.join(image_path, "inv_masked_img.jpg"), mask_inv)
    # Apply the mask to the source image for each channel
    b_masked_source = cv2.bitwise_and(b, b, mask=mask)
    g_masked_source = cv2.bitwise_and(g, g, mask=mask)
    r_masked_source = cv2.bitwise_and(r, r, mask=mask)

    # Apply the inverted mask to the destination image for each channel
    b_masked_dest = cv2.bitwise_and(dest_img[:,:,0], dest_img[:,:,0], mask=mask_inv)
    g_masked_dest = cv2.bitwise_and(dest_img[:,:,1], dest_img[:,:,1], mask=mask_inv)
    r_masked_dest = cv2.bitwise_and(dest_img[:,:,2], dest_img[:,:,2], mask=mask_inv)
    
    masked_dest = cv2.merge((b_masked_dest, g_masked_dest, r_masked_dest))
    cv2.imwrite(os.path.join(image_path, "masked_dest.jpg"), masked_dest)

    # Combine the masked source and masked destination images for each channel
    b_result = cv2.bitwise_or(b_masked_source, b_masked_dest)
    g_result = cv2.bitwise_or(g_masked_source, g_masked_dest)
    r_result = cv2.bitwise_or(r_masked_source, r_masked_dest)

    # Merge the channels back together
    result = cv2.merge((b_result, g_result, r_result))

    # Save the resulting image
    return result

def main():
    wall_img = cv2.imread(os.path.join(image_path, "9.jpg"))
    scenary = cv2.imread(os.path.join(BASE_DIR, "1.jpg"))
    aruco_got = findArucoMarkers(wall_img)
        # Creation of Loop for One Image for different aruco position
    if len(aruco_got[0]) != 0:
        for boundbox, ids in zip(aruco_got[0],aruco_got[1]):
            output = MarkerAugment(boundbox, id, wall_img, scenary)
    cv2.waitKey(0)
        
if  __name__ == "__main__":
    main()