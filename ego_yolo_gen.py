import scipy.io as sio
import numpy as np
import time
import random
import argparse
import os
import zipfile
import shutil
from tqdm import tqdm 
from google.colab.patches import cv2_imshow

ap = argparse.ArgumentParser()
ap.add_argument("--extract",default = True)
ap.add_argument("--train",required=True)
ap.add_argument("--val",required=True)
ap.add_argument("--test",required=True)
args = vars(ap.parse_args())

def main():
  extract = args['extract']
  train_size = args['train']
  val_size = args['val']
  test_size = args['test']

  if extract == "True":
    if os.path.exists("egohands"):
      print("Found egohands folder. Deleting...")
      shutil.rmtree("egohands")
    print("extracting zip...")
    zip_file = zipfile.ZipFile("egohands_data.zip")
    zip_file.extractall("egohands")
    print("Extraction completed")
  
  if os.path.exists("egohands"):
    time.sleep(0.5)
    rename_files("egohands/_LABELLED_SAMPLES")

    time.sleep(0.5)
    generate_yolo_txt("egohands/_LABELLED_SAMPLES")

    time.sleep(0.5)
    collect_txt("egohands/_LABELLED_SAMPLES")

    time.sleep(0.5)
    split_data("egohands/_LABELLED_SAMPLES/txt_files",train_size,val_size,test_size)

  else:
    print("There is no folder named egohands. Please create one or set extract to TRUE")

#helper functions
def rename_files(image_dir):
  for root, dirs, filenames in os.walk(image_dir):
    for dir in dirs:
      for f in os.listdir(os.path.join(image_dir, dir)):
        if (f.split('.')[1] == 'jpg'):
          from_rn = os.path.join(image_dir,dir,f)
          to_rn = os.path.join(image_dir,dir)+'/' + dir+'_'+f
          os.rename(from_rn, to_rn)
  print("Renaming process is done!")

def list_to_str(list):
  ls_str = str()
  for lis in list:
    for i,element in enumerate(lis):
      content = str(element)
      if i < (len(lis)-1):
        ls_str += content + ","
      elif i == (len(lis)-1):
        ls_str += content + " "
  return ls_str  

def create_yolo_txt(image_head_dir, dir):
  image_path_array = []
  for root, dirs, filenames in os.walk(os.path.join(image_head_dir, dir)):
    for f in filenames:
      if f.split(".")[1] == 'jpg':
        abs_f = os.path.join(os.path.abspath(os.getcwd()),image_head_dir,dir) +"/"+f
        image_path_array.append(abs_f)
  image_path_array.sort()

  boxes = sio.loadmat( os.path.join(image_head_dir,dir)+'/'+"polygons.mat")
  polygons = boxes['polygons'][0]

  hands = ['myleft', 'myright', 'yourleft','yourright']
  photo_num = 0
  
  for each_photo in polygons:
    box_array = []
    txtholder = []
    hand_index = 0
    image_path , image_name = os.path.split(image_path_array[photo_num])
    for each_point_gp in each_photo:
      xmin,ymin,xmax,ymax = 0,0,0,0
      findex = 0
      
      for point in each_point_gp:
        if (len(point) == 2):
          x = int(point[0])
          y = int(point[1])
          if findex == 0:
            xmin = x
            ymin = y
          findex += 1
          xmax = x if (x>xmax) else xmax
          ymax = y if (y>ymax) else ymax
          xmin = x if (x<xmin) else xmin
          ymin = y if (y<ymin) else ymin

      hold = {}
      hold['xmin'] = xmin
      hold['ymin'] = ymin
      hold['xmax'] = xmax
      hold['ymax'] = ymax
      #hold['findex'] = findex
      if xmin >0 and ymin>0 and xmax>0 and ymax>0:
        box_array.append(hold)

        bbox_content = [xmin,ymin,xmax,ymax,hand_index]
        txtholder.append(bbox_content)

      hand_index += 1

    txt_name =image_head_dir+ "/"+ dir +"/"+ image_name.split('.')[0] + ".txt"
    
    if list_to_str(txtholder) == " " or list_to_str(txtholder) == "":
      continue
    else:
      txt_file = open(txt_name , "w+")
      txt_file.write(image_path_array[photo_num]+" "+list_to_str(txtholder))
      txt_file.close()

    photo_num += 1

def generate_yolo_txt(image_head_dir):

  with tqdm(total = len(os.listdir(image_head_dir)),desc="YoloTxT generating :", bar_format="{desc}{percentage:3.0f}% {bar} [{n_fmt}/{total_fmt}]  ", ncols=64) as pbar:
    for root, dirs, filenames in os.walk(image_head_dir):
      for dir in dirs:
        create_yolo_txt(image_head_dir,dir)
        pbar.update(1)

  print("Yolo.txt generation is completed!")

def collect_txt(image_head_path):
  if os.path.exists("egohands/_LABELLED_SAMPLES/txt_files"):
    print('File already exists!')
  else:
    os.mkdir("egohands/_LABELLED_SAMPLES/txt_files")

  txt_dir = os.path.join(image_head_path,"txt_files")

  with tqdm(total = len(os.listdir(image_head_path))-1,desc="Txt Collection :",bar_format="{desc}{percentage:3.0f}% {bar} [{n_fmt}/{total_fmt}]  ", ncols=64 )as pbar:
    for root,dirs,filenames in os.walk(image_head_path):
      for dir in dirs:
        if dir == "txt_files":
          continue
        else:
          image_folder_dir = os.path.join(image_head_path, dir)
          for files in os.listdir(image_folder_dir):
            if files.split('.')[1] == "txt":
              shutil.copy(os.path.join(image_folder_dir,files), txt_dir+"/"+ files)
            else: continue
        pbar.update(1)
  print("Txt file collection is done!")
  
def split_data(txt_files_path, train_size, val_size, test_size):
  
  train_size = int(train_size)
  val_size = int(val_size)
  test_size = int(test_size)
  total_size = train_size + val_size + test_size 

  if total_size > len(os.listdir(txt_files_path)):
    print("Size Limit Error!")
  else:
    if os.path.exists("ego_txt"):
      print("One folder named ego_txt exists!\nDeleting...") 
      shutil.rmtree("ego_txt")
    print("Creating ego_txt folder in cwd...\n")
    os.mkdir("ego_txt")

    time.sleep(0.5)
    with tqdm(total = 7, desc = "randomizer :", bar_format="{desc}{percentage:3.0f}% {bar} [{n_fmt}/{total_fmt}]  ",ncols = 50)as pbarrdm:
      pbarrdm.update(1)
      random_num_all = random.sample(range(len(os.listdir(txt_files_path))), k = total_size)
      pbarrdm.update(1)
      random_num_train= random.sample(range(total_size), k = train_size)
      pbarrdm.update(1)
      random_num_val= random.sample(range(val_size + test_size), k = val_size)
      pbarrdm.update(1)

      all_txt = [os.listdir(txt_files_path)[i] for i in random_num_all]
      train_txt = [all_txt[i] for i in random_num_train]
      pbarrdm.update(1)
      val_test_txt = [i for i in all_txt if i not in train_txt]
      pbarrdm.update(1)
      val_txt = [val_test_txt[i] for i in random_num_val]
      test_txt = [i for i in val_test_txt if i not in val_txt]
      pbarrdm.update(1)
    print("Completed randomizing!\n")
    pbarrdm.close()
    
    time.sleep(0.5)
    with tqdm(total = len(all_txt), desc = "writer :", bar_format="{desc}{percentage:3.0f}% {bar} [{n_fmt}/{total_fmt}]  ",ncols = 50)as pbar:

      class_file = open("ego_txt/class_file.names", "w")
      class_file.write("myleft\nmyright\nyourleft\nyourright\n")
      class_file.close()

      train_file = open("ego_txt/train_ego.txt", "w") 
      for txt_name in train_txt:
        with open(txt_files_path+"/"+txt_name, "r") as file:
          txt = file.read()
          train_file.write(txt + "\n")
        file.close()
        pbar.update(1)
      train_file.close()

      val_file = open("ego_txt/val_ego.txt", "w")
      for txt_name in val_txt:
        with open(txt_files_path+"/"+txt_name, "r") as file:
          txt = file.read()
          val_file.write(txt + "\n")
        file.close()
        pbar.update(1)
      val_file.close()

      test_file = open("ego_txt/test_ego.txt", "w")
      for txt_name in test_txt:
        with open(txt_files_path+"/"+txt_name, "r") as file:
          txt = file.read()
          test_file.write(txt + "\n")
        file.close()
        pbar.update(1)
      test_file.close()
      pbar.close()

  print("Data splitting completed!")

if __name__ == "__main__":
  main()
