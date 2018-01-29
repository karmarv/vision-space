# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:57:53 2018

@author: Rahul Vishwakarma

@Purpose: Image registration project

"""

import sys, os, getopt
import re
import numpy as np
import cv2


'''
    Global Variables
'''
MIN_MATCH_COUNT = 3
LOWE_RATIO_TEST_FACTOR = 0.7
np.set_printoptions(precision=3,suppress=True)
xavg=0 # x avg for normalization
yavg=0 # y avg for normalization



''' Read argument list '''
def readParams(argv):
    try:
        opts, args = getopt.getopt(argv, "hp:i:l:", ["ipath=", "imgx=", "loratio="])
        print('CV Version  : ', cv2.__version__)
        print('Args Count  : ', len(args))
    except getopt.GetoptError:
        print('panstitch.py -p <ipath> -i <imgx> -l <loratio>')
        print('python panstitch.py -p "assets/GrandCanyon1" -i "(PIC_00[0-9][0-9].JPG)"')
        sys.exit(2)
    global LOWE_RATIO_TEST_FACTOR
    global imgpath_in
    global imgregx_in
    imgpath_in = 'assets/GrandCanyon1'
    imgregx_in = '(PIC_00[0-9][0-9].jpg)'
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
    name = prefix+"_out_kp_"+name+".png"
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
    # Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    # Get the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(gray,None)
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

''' Get projected points on target image space '''
def getProjectedPointsTarget(im_src, H):
    h,w,c = im_src.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # Find the points projection in target
    dst = cv2.perspectiveTransform(pts,H)
    return dst

'''
    Warp/Transformed image using H
'''
def transformImages(im_src, im_dst, H):
    # Warp source image to destination based on homography matrix H
    im_out = cv2.warpPerspective(im_src, H, (im_dst.shape[1],im_dst.shape[0]))
    return im_out

'''
    Superimpose the source image on the target image
'''
def superImposeImagesWarp(im_src, im_dst, H):
    print("Superimpose Images \n")
    height, width = im_src.shape[:2]
    mod_photo = cv2.warpPerspective(im_src, H, (width, height))
    new_image = cv2.addWeighted(mod_photo, 0.5, im_dst, 0.5, 1)
    print("Dest shape: \n",new_image.shape)
    return new_image
def superImposeImages(im_src, im_dst, H):

     # Image points on destination plane
    h1, w1 = im_src.shape[:2]
    pts = np.float32([ [0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0] ]).reshape(-1,1,2)
    pers_pts = cv2.perspectiveTransform(pts,H)
    print("Size of warped: ",pers_pts)
    h2, w2 = im_dst.shape[:2]
    dest_pts = np.float32([ [0,0],[0,h2-1],[w2-1,h2-1],[w2-1,0] ]).reshape(-1,1,2)
    print("Size of warped: ",dest_pts)
    # get the bounding rectangle for the four corner points (and thus, the transformed image)
    dx, dy, dwidth, dheight = cv2.boundingRect(dest_pts)
    sx, sy, swidth, sheight = cv2.boundingRect(pers_pts)
    rheight = sheight+ dheight #dheight if (dheight>sheight) else sheight
    rwidth = swidth+dwidth #dwidth if (dwidth>swidth) else swidth

    # translation matrix
    t1 = np.array([1, 0, dx, 0, 1, dy, 0, 0,1]).reshape((3,3))
    t2 = np.array([1, 0, sx, 0, 1, sy, 0, 0,1]).reshape((3,3))

    print("Proj Source Bounding Images ",sx, sy, swidth, sheight,)
    print("Destination Bounding Images ",dx, dy, dwidth, dheight)
    print("Result size: ", rwidth, rheight)
    print("Superimpose Images \n S: ",im_src.shape[0],im_src.shape[1]," \n D: ",im_dst.shape[0],im_dst.shape[1] )
    result = cv2.warpPerspective(im_src, H,(rwidth,rheight))
    result[dx:im_dst.shape[0]+dx, dy:im_dst.shape[1]+dy] = im_dst
    return result


'''
    Show images
'''
def showImages(im_src, im_dst, im_lined, im_tfm, im_sup, H):
    cv2.imwrite( getOutputFilePath(imgpath_in,"l"), im_lined );
    cv2.imwrite( getOutputFilePath(imgpath_in,"t"), im_tfm );
    cv2.imwrite( getOutputFilePath(imgpath_in,"s"), im_sup );

    cv2.imshow("Src->Dst Keypoints Matched", im_lined)
    cv2.imshow("Warped Source Image using Homography", im_tfm)
    #cv2.namedWindow('SuperImposed Images',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('SuperImposed Images', 1280,1024)
    cv2.imshow("SuperImposed Images", im_sup)
    cv2.waitKey(0)



'''
    Compute the SVD for Homography matrix
'''
def computeSVD(kp1,kp2,good):
    npoints = len(good)
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    d = getNormPoints(src_pts)
    s = getNormPoints(dst_pts)
    #print("Source Points:\n ",src_pts), print("Destination Points:\n ",dst_pts)
    A = np.empty((2*npoints,9,))
    A.fill(0)
    # Create the (2n,9) matrix for Singular Value decomposition
    for i in range(0, npoints):
        x, y = src_pts[i][0][0],src_pts[i][0][1]     # X1, Y1
        u, v = dst_pts[i][0][0],dst_pts[i][0][1]     # X2, Y2
        # Add rows to the A matrix
        A[(2*i),:]   = np.array([x, y, 1, 0, 0, 0, -x*u, -u*y, -u])
        A[(2*i)+1,:] = np.array([0, 0, 0, x, y, 1, -x*v, -v*y, -v])
    U, D, V = np.linalg.svd(np.asarray(A))
    # Take out the last row which will be 8th col upon transpose
    H = np.reshape(V[-1,:] / V[-1,-1],(3,3), order='F').transpose()
    return H

'''
    TODO: Figure the Normalization logic by scaling 1.414 from the centroid
'''
def getNormPoints(pts):
    # http://www.maths.lth.se/matematiklth/personal/calle/datorseende13/notes/forelas3.pdf
    b = np.empty((len(pts),2,))
    x = 0
    y = 0
    s = 1.414
    for i in range(len(pts)):
        #print(i,") ",pts[i])
        x+=pts[i][0][0]
        y+=pts[i][0][1]
        b[i,:] = pts[i]
    xavg=x/(len(pts))
    yavg=y/(len(pts))
    P = np.zeros((len(pts),3))
    P[:,:-1] = b
    #print("P Matrix: \n", P)
    N= np.array([[s,0,-s*xavg],[0,s,-s*yavg],[0,0,1]])
    #print("N Matrix: \n", N)
    c = np.dot(P,N)
    d = c[:,[0,1]]
    #print("Dot product\n ",d)
    #print("Avg: ",xavg,yavg)
    return d


'''
    ------------------------------------- C -----------------------------------
    Used to find homography matrix H using Normalized RANSAC
    between two images with the same object.
    Solve errors related to matching using RANSAC algorithm
'''
def computeHomographyNormRansac(im_src, kp1, des1, im_dst, kp2, des2, good):
    # If enough matches (>4) are found
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        # print(src_pts), #print(dst_pts)
        # Get transform between two images as a mask with inliers project threshold for RANSAC is 5
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print("Normalized RANSAC, Homography H: \n")
        print (M) # 3x3 transformation matrix
        im_lined = drawMatches(im_src,kp1,im_dst,kp2,good,M)
        # Transform images based on the H matrix
        im_tfm = transformImages(im_src, im_dst, M)
        im_sup = superImposeImages(im_src, im_dst, M)
        showImages(im_src, im_dst, im_lined, im_tfm,im_sup, M)
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        print("Tweak the values of LOWE_RATIO_TEST_FACTOR - %f" % (LOWE_RATIO_TEST_FACTOR))


'''
    ------------------------------------- B -----------------------------------
    Used to find homography Matrix H using Normalized DLT(without RANSAC)
    between two images with the same object using inherent opencv normalization
'''
def computeHomographyNormDlt(im_src, kp1, des1, im_dst, kp2, des2, good):
    # If enough matches (>4) are found
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        # print(src_pts), #print(dst_pts)
        M, mask = cv2.findHomography(src_pts, dst_pts, 0)
        print("Normalized DLT, Homography H: \n")
        print (M) # 3x3 transformation matrix
        im_lined = drawMatches(im_src,kp1,im_dst,kp2,good,M)
        # Transform images based on the H matrix
        im_tfm = transformImages(im_src, im_dst, M)
        im_sup = superImposeImages(im_src, im_dst, M)
        showImages(im_src, im_dst, im_lined, im_tfm,im_sup, M)
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        print("Tweak the values of LOWE_RATIO_TEST_FACTOR - %f" % (LOWE_RATIO_TEST_FACTOR))


'''
    ------------------------------------- A -----------------------------------
    Used to find homography Matrix H using Standard DLT
    Implemented the homography logic using SVD computation in computeSVD
'''
def computeHomographyStdDlt(im_src, kp1, des1, im_dst, kp2, des2, good):
    npoints = len(good)
    # If enough matches (>10) are found
    if npoints>MIN_MATCH_COUNT:
        H = computeSVD(kp1,kp2,good)
        H = H/H[2,2] # H using DLT
        print("Standard DLT, Homography H: \n ",H)
        im_lined = drawMatches(im_src,kp1,im_dst,kp2,good,H)
        # Transform images based on the H matrix
        im_tfm = transformImages(im_src, im_dst, H)
        im_sup = superImposeImages(im_src, im_dst, H)
        showImages(im_src,im_dst, im_lined, im_tfm,im_sup, H)
    else:
        print("Not enough matches are found - %d/%d" % (npoints,MIN_MATCH_COUNT))
        print("Tweak the values of LOWE_RATIO_TEST_FACTOR - %f" % (LOWE_RATIO_TEST_FACTOR))


'''
    Establish Correspondence between images pairwise
'''
def imageRegistration(imgFileA, imgFileB):
    print("Compute keypoints for the images: ",imgFileA," & ", imgFileB)
    im_src, kp1, des1 = computeKeypoints(imgFileA)
    img2, kp2, des2 = computeKeypoints(imgFileB)
    print("Compute the distance and obtain good points")
    good = computeMatches(des1,des2)

    print("------------------ A -------------------------")
    #computeHomographyStdDlt(im_src, kp1, des1, img2, kp2, des2, good)
    print("------------------ B ------------------------")
    #computeHomographyNormDlt(im_src, kp1, des1, img2, kp2, des2, good)
    print("------------------ C ----------------------")
    computeHomographyNormRansac(im_src, kp1, des1, img2, kp2, des2, good)

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
    # print(imgFiles);

    imgFileA = imgpath_in+os.sep+imgFiles[0]
    imgFileB = imgpath_in+os.sep+imgFiles[1]
    imageRegistration(imgFileA,imgFileB)
    print("-----------------EXIT----------------------")
    pass

