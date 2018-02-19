# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:57:53 2018

@author: Rahul Vishwakarma

@Purpose: Image registration

"""

import sys, os, getopt
import re
import numpy as np
import cv2


'''
    Global Variables
'''
MIN_MATCH_COUNT = 3
LOWE_RATIO_TEST_FACTOR = 0.6
np.set_printoptions(precision=3,suppress=True)
execution_stage_index = 1

# Homographies
Hlist = []
Hid = 0


''' Read argument list '''
def readParams(argv):
    try:
        opts, args = getopt.getopt(argv, "hp:i:l:", ["ipath=", "imgx=", "loratio="])
        print('CV Version  : ', cv2.__version__)
        print('Args Count  : ', len(args))
    except getopt.GetoptError:
        print('Make sure you have sourced the right environment:  C:\Anaconda3\Scripts\activate.bat C:\Anaconda3 && conda activate py27')
        print('panstitch.py -p <ipath> -i <imgx> -l <loratio>')
        print('python panstitch.py -p "assets/GrandCanyon1" -i "(PIC_00[0-9][0-9].JPG)"')
        sys.exit(2)
    global LOWE_RATIO_TEST_FACTOR
    global imgpath_in
    global imgregx_in
    imgpath_in = 'assets/GrandCanyon1'
    imgregx_in = '*'
    for opt, arg in opts:
        if opt == '-h':
            print('panstitch.py -p <ipath> -i <imgx> -l <loratio>')
            sys.exit()
        elif opt in ("-p", "--ipath"):
            imgpath_in = arg
        elif opt in ("-i", "--imgx"):
            imgregx_in = arg
        elif opt in ("-l", "--loratio"):
            LOWE_RATIO_TEST_FACTOR = float(arg)
    print('Input Path  : ', imgpath_in)
    print('Image Regex : ', imgregx_in)
    print('Lowes Ratio : ', LOWE_RATIO_TEST_FACTOR)
    


''' Get the output file name from current name '''
def getOutputFilePath(filepath, prefix):
    basepath=os.path.dirname(os.path.abspath(filepath))
    basename=os.path.basename(filepath)
    name = os.path.splitext(basename)[0]
    name = prefix+"_out.png"
    new_fullpath = os.path.join(basepath,"output",name)
    return new_fullpath


'''
    Compute Keypoints using SIFT
'''
def computeKeypoints(image_filepath):
    # SIFT in OpenCV https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
    img = cv2.imread(image_filepath)
    # Working with a grayscale image
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Create SIFT object # Not available without opencv-contib
    # sift = cv2.xfeatures2d.SIFT_create()
    # kaze = cv2.KAZE_create()
    # akaze = cv2.AKAZE_create()
    brisk = cv2.BRISK_create()

    # Get the keypoints and descriptors
    kp, des = brisk.detectAndCompute(gray,None)
    return img, kp, des


'''
    Compute distance for good corresponding point matches
    FLANN is more precise when compared to Brute force matcher
'''
def computeMatches(des1, des2):
    # FLANN(Fast library for Approximate Nearest Neighbor)
    FLANN_INDEX_KDTREE = 2
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    # Convert flann based matcher
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    # KNN matcher using parameters
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    # Used Prof David Lowe's suggested ratio test
    for m,n in matches:
        if m.distance < int(LOWE_RATIO_TEST_FACTOR * float(n.distance)):
            good.append(m) # Move all the good points in source image
            #print(m,n)
    print("Length of good points in Image-1 using ratio test: ",len(good))
    return good


''' Get projected points on target image space '''
def getProjectedPointsTarget(im_src, H):
    h,w,c = im_src.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # Find the points projection in target
    dst = cv2.perspectiveTransform(pts,H)
    return dst

'''
    Compare images and draw polylines between keypoints
'''
def drawMatches(im_src, kp1, im_dst, kp2, good, H):
    mmask = np.empty((len(good),1,))
    mmask.fill(1) #print(mmask)
    matchesMask = mmask.ravel().tolist()
    dst = getProjectedPointsTarget(im_src, H)
    #img2 = cv2.polylines(im_dst,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask,  # draw only inliers
                   flags = 2)
    img3 = cv2.drawMatches(im_src,kp1,im_dst,kp2,good,None,**draw_params)
    return img3


'''
    Show images inm a fixed window
'''
def showImage(img, name):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1280,1024)
    cv2.imshow(name, img)
        


'''
    Warp/Transformed image using H
    Perspective transform on image 2
    Then overlay the image 1 on it
'''
def transformWarpImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    ''' Get projected points on target image space '''
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    # the dot product with give a matrix that will translate and warp this image
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result

''' Get Homography matrix between two images 
    as a mask with inliers 
    project threshold for RANSAC is 5 
    Solve errors related to matching using RANSAC algorithm
'''
def computeHomographyMatrixOcv(kp1, kp2, good):
    dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # print(src_pts), #print(dst_pts)
    # Get transform between two images as a mask with inliers project threshold for RANSAC is 5
    return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

''' Compute Homography using the DLT approach
    Approach is using DLT on the good points 
'''
def computeHomographyMatrixDlt(kp1, kp2, good):
    dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #print(src_pts)
    #print(dst_pts)
    # Get homography transform between two images
    A = []
    for i in range(0, len(src_pts)):
        x, y = dst_pts[i][0][0], dst_pts[i][0][1]
        u, v = src_pts[i][0][0], src_pts[i][0][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)
    return H

'''
    --------------------------------------------------------
    Used to find homography matrix H using Normalized RANSAC
    between two images with the same object.
'''
def computeTransformImagePair(im_src, kp1, des1, im_dst, kp2, des2, good):
    # If enough matches (>4) are found
    if len(good)>MIN_MATCH_COUNT:
        # Get transform between two images as a mask with inliers project threshold for RANSAC is 5
        M, mask = computeHomographyMatrixOcv(kp1, kp2, good)
        Hlist.insert(Hid,M)
        if Hid > 0 :
            M.dot(Hlist[Hid-1])
        #H = computeHomographyMatrixDlt(kp1, kp2, good)
        #print("Normalized RANSAC, Homography H: \n")
        #print (H) # 3x3 transformation matrix
        im_lined = drawMatches(im_src, kp1, im_dst, kp2, good, M)
        # Transform images based on the H matrix
        im_tfm = transformWarpImages(im_src, im_dst, M)
        return im_tfm
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        print("Tweak the values of LOWE_RATIO_TEST_FACTOR - %f" % (LOWE_RATIO_TEST_FACTOR))
    return 0

'''
    Establish Correspondence between images pairwise
'''
def imageRegistration(imgFileA, imgFileB):
    #print("Compute keypoints for the images: ",imgFileA," & ", imgFileB)
    im_src, kp1, des1 = computeKeypoints(imgFileA)
    img2, kp2, des2 = computeKeypoints(imgFileB)
    #print("Compute the distance and obtain good points")
    good = computeMatches(des1,des2)
    return computeTransformImagePair(im_src, kp1, des1, img2, kp2, des2, good)


'''
    ----------------------------------- MAIN ---------------------------------
    Main entrypoint of the code
    Reads the arguments to the program and run
'''
if __name__ == '__main__':
    print("-----------------START---------------------")
    readParams(sys.argv[1:])
    global imgFiles 
    imgFiles = [f for f in os.listdir(imgpath_in) if re.match(imgregx_in, f)]
    imgFiles.sort()
    count_images = len(imgFiles)
    print(imgFiles)

    if count_images <= 1:
        print("No images found")
        exit(0)
    elif count_images == 2:
        iterations = 1
    elif count_images % 2 == 0:
        iterations = int(np.log2(count_images)) + 1
    else: 
        iterations = int(np.log2(count_images+1)) + 1

    run_count = 0
    #iterations = 2
    print("Iterations: ", iterations)
    if count_images >= 2:
        for ex in xrange(0, iterations):
            print("")
            print("------------- Pass:",ex+1," Started ------------------")
            if ex == 0: # First Pass    
                run_count = count_images
            else: 
                if run_count % 2 == 0:
                    run_count = int(run_count/2)
                else: 
                    run_count = int((run_count+1)/2)
            print("Run count: ", run_count)
            run_id = 0
            for i in xrange(0, run_count, 2):
                # Setup current execution output parameters
                outStitchId = "%s-%s-%s" % (execution_stage_index,i,i+1)
                imgOutpt =  getOutputFilePath(imgpath_in,outStitchId) 
                if ex == 0: # First Pass
                    if i >= (run_count - 1): 
                        print("Drop Last call in first pass")
                        imgFileA = imgpath_in+os.sep+imgFiles[i-1]
                        imgFileB = imgpath_in+os.sep+imgFiles[i]
                    else:
                        imgFileA = imgpath_in+os.sep+imgFiles[i]
                        imgFileB = imgpath_in+os.sep+imgFiles[i+1]
                else: 
                    if i >= (run_count - 1): 
                        #print("Drop Last call")
                        # Operate on the last file and a previously generated file
                        inStitchId = "%s-%s-%s" % (execution_stage_index-1,4*run_id,4*run_id+1)
                        imgFileA = getOutputFilePath(imgpath_in, inStitchId)
                        inStitchId = "%s-%s-%s" % (execution_stage_index,i-2,i-1)
                        imgFileB = getOutputFilePath(imgpath_in, inStitchId)
                    else:     
                        # Setup input image parameters
                        inStitchId = "%s-%s-%s" % (execution_stage_index-1,4*run_id,4*run_id+1)
                        imgFileA = getOutputFilePath(imgpath_in, inStitchId)
                        inStitchId = "%s-%s-%s" % (execution_stage_index-1,4*run_id+2,4*run_id+3)
                        imgFileB = getOutputFilePath(imgpath_in, inStitchId)
                print("Operate: ", os.path.basename(imgFileA)," ~ ", os.path.basename(imgFileB))
                print("------------- Stitch ",outStitchId,"-----------------")
                warped_img = imageRegistration(imgFileA,imgFileB)
                Hid = Hid + 1 # Id of the homography in the dict
                # Write output to file
                cv2.imwrite(imgOutpt, warped_img )
                run_id = run_id + 1
                del warped_img
            #Pass done
            execution_stage_index = execution_stage_index + 1
            Hlist = []
            Hid = 0
            print("------------- Pass:",ex+1," Complete ----------------")
    
    print("-----------------EXIT----------------------")
    #cv2.waitKey(0)

