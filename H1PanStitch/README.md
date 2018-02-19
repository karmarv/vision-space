------------------------------------------------------------
- Image stitching using SIFT and RANSAC technique in OpenCV.
- Using SIFT/SURF and for speed, BRISK which is in OpenCV
- Keeping track of all the homographies at each iteration
------------------------------------------------------------

- Run as:  
```
    $ python panstitch2.py -p <ipath> -i <regex for list of images>
    $ python panstitch2.py -p "assets/GrandCanyonSml" -i "(PIC_00[0-9][0-9].JPG)"
```
- For the images provided in Desk folder
```
    $ python panstitch2.py -p "assets/Desk" -i "(IMG_[0-9]+_[0-9]+.jpg)"
```
- Program Dumps the intermediate/final image in the assets/output folder
- The file that is generated last is the final panorama file, others are intermediate results