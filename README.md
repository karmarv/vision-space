# vision-space
Computer Vision Project Repository

----------------------------------------------------

### Setup Reference 
Link: https://rahulvishwakarma.wordpress.com/2018/01/26/anaconda-for-your-image-processing-machine-learning-neural-networks-computer-vision-development-environment-using-vs-code/

In case you use Cmder console
- Open up Cmder and execute the activation as below
- C:\Anaconda3\Scripts\activate.bat C:\Anaconda3

----------------------------------------------------

1. H1PanStitch
- Image stitching using SIFT and RANSAC technique in OpenCV.
- Run as:  
```
    $ panstitch.py -p <ipath> -i <imgx> -l <loratio>
    $ python panstitch.py -p "assets/GrandCanyon1" -i "(PIC_00[0-9][0-9].JPG)"
```
- This stitches the images pairwise at each stage
- Dump the results in assets/output folder

2. H2RecogTF
- Image recognition/classification task using Tensorflow
-
```
    $ python recog.py
```
