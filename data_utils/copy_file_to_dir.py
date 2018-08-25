import os
import shutil

input_dir = '/home/leon/Disk/dataset/ShapeNetCar/02958343'
output_dir = '/home/leon/Disk/dataset/ShapeNetCarObj'

for dir_name in os.listdir(input_dir):
    obj_path = os.path.join(input_dir, dir_name, 'models/model_normalized.obj')
    obj_path_new = os.path.join(output_dir, dir_name + ".obj")
    if os.path.exists(obj_path):
        shutil.copyfile(obj_path, obj_path_new)
        print(obj_path_new)