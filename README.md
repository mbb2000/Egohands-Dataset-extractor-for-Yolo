# Egohands-Dataset-extractor-for-Yolo
Dataset extraction and preparation of egohands to use with YoloV3

*You need to place the ".py" folder in the same with the ".zip" file of egohands data set.
*if you are using zipfile, it has to be the default name from dataset original source. For own extraction, name the head folder to "egohands".

*Usage : python ego_yolov3_gen.py --extract (whether to extract the zip or not) --train (train size) --val (validation size) --test (test size)

*The txt files will be in format of "absolute_path_of_image xmin,ymin,xmax,ymax,class_id xmin,ymin,..."

This will create "ego_txt" folder in the current working directory which contains class.names, train, val, test text files for input to YoloV3.
