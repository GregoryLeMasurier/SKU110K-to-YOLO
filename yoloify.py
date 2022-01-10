from __future__ import division

import os
import sys
import csv
import pandas as pd
from pathlib import Path

# Main
dataset_path = ""
if len(sys.argv) > 1 :
    dataset_path = sys.argv[1]
    if(dataset_path.endswith(os.sep)):
        dataset_path = dataset_path[:-1]
    print("Using path provided via command line: " + dataset_path)
else:
    dataset_path = os.getcwd()
dataset_name = dataset_path.split(os.sep)[-1]
print("Saving dataset with name: " + dataset_name)

# Create dataset yaml file
full_yaml_path = os.path.join(dataset_path, dataset_name + ".yaml")
f = open(full_yaml_path, "w")
with open(full_yaml_path, 'w') as f:
    f.write("path: ../datasets/" + dataset_name + os.linesep)
    f.write("train: images/train" + os.linesep)
    f.write("val: images/val" + os.linesep)
    f.write("test: images/test" + os.linesep + os.linesep)
    f.write("nc: 1" + os.linesep)
    f.write("names: [ 'object' ]" + os.linesep)

# Create necessary directories
full_image_path = os.path.join(dataset_path, "images")
full_image_train_path = os.path.join(full_image_path, "train")
if not os.path.isdir(full_image_train_path):
    os.makedirs(full_image_train_path)
full_image_val_path = os.path.join(full_image_path, "val")
if not os.path.isdir(full_image_val_path):
    os.makedirs(full_image_val_path)
full_image_test_path = os.path.join(full_image_path, "test")
if not os.path.isdir(full_image_test_path):
    os.makedirs(full_image_test_path)

full_label_path = os.path.join(dataset_path, "labels")
if not os.path.isdir(full_label_path):
    os.makedirs(full_label_path)
full_label_train_path = os.path.join(full_label_path, "train")
if not os.path.isdir(full_label_train_path):
    os.makedirs(full_label_train_path)
full_label_val_path = os.path.join(full_label_path, "val")
if not os.path.isdir(full_label_val_path):
    os.makedirs(full_label_val_path)
full_label_test_path = os.path.join(full_label_path, "test")
if not os.path.isdir(full_label_test_path):
    os.makedirs(full_label_test_path)

# Yoloify dataset
full_annotation_path = os.path.join(dataset_path, "annotations")
header = ['image_name','x1','y1','x2','y2','class','image_width','image_height']
for filename in os.listdir(full_annotation_path):
    label = []
    x_center = []
    y_center = []
    width = []
    height = []
    if filename.endswith(".csv"):
        label_path_name = full_label_path
        image_path_name = full_image_path
        full_csv_file_path = os.path.join(full_annotation_path, filename)
        csv_base_filename = Path(full_csv_file_path).stem
        if(csv_base_filename.endswith("train")):
            label_path_name = full_label_train_path
            image_path_name = full_image_train_path
            print("TRAIN: " + full_image_train_path)
        elif(csv_base_filename.endswith("val")):
            label_path_name = full_label_val_path
            image_path_name = full_image_val_path
            print("VAL: " + full_image_val_path)
        elif(csv_base_filename.endswith("test")):
            label_path_name = full_label_test_path
            image_path_name = full_image_test_path
            print("TEST: " + full_image_test_path)
        df = pd.read_csv(full_csv_file_path, names=header)
        df['x1'] = df['x1'].astype(int)
        df['x2'] = df['x2'].astype(int)
        df['y1'] = df['y1'].astype(int)
        df['y2'] = df['y2'].astype(int)
        df['image_width'] = df['image_width'].astype(int)
        df['image_height'] = df['image_height'].astype(int)
        prev_id = -1
        prev_image_name = ""
        for index, row in df.iterrows():
	    #Get image number
            id = Path(row['image_name']).stem.split('_')[1]
            if id != prev_id and prev_id != -1:
                #save txt file
                new_df = pd.DataFrame({'class': label, 'x_center': x_center, 'y_center': y_center, 'width': width, 'height': height})
                #print(new_df)
                label_file_name = os.path.splitext(prev_image_name)[0]+'.txt'
                current_label_path = os.path.join(label_path_name, label_file_name)
                old_img_path = os.path.join(full_image_path, prev_image_name)
                current_img_path = os.path.join(image_path_name, prev_image_name)
                if os.path.exists(old_img_path):
                    os.rename(old_img_path, current_img_path)
                #print("LABEL FILE NAME: " + current_label_path)
                with open(current_label_path, 'w') as f:
                    df_contents = new_df.to_string(header=False, index=False)
                    f.write(df_contents)
                label = []
                x_center = []
                y_center = []
                width = []
                height = []
            label.append(0)
            n_width = row['x2'] - row['x1']
            width.append(n_width / row['image_width'])
            x_ctr = (row['x1'] + (n_width / 2)) / row['image_width']
            x_center.append(x_ctr)
            n_height = row['y2'] - row['y1']
            height.append(n_height / row['image_height'])
            y_ctr = (row['y1'] + (n_height / 2)) / row['image_height']
            y_center.append(y_ctr)
            prev_image_name = row['image_name']
            prev_id = id
        new_df = pd.DataFrame({'class': label, 'x_center': x_center, 'y_center': y_center, 'width': width, 'height': height})
        #print(new_df)
        label_file_name = os.path.splitext(prev_image_name)[0]+'.txt'
        current_label_path = os.path.join(label_path_name, label_file_name)
        old_img_path = os.path.join(full_image_path, prev_image_name)
        current_img_path = os.path.join(image_path_name, prev_image_name)
        if os.path.exists(old_img_path):
            os.rename(old_img_path, current_img_path)
        #print("LABEL FILE NAME: " + current_label_path)
        with open(current_label_path, 'w') as f:
            df_contents = new_df.to_string(header=False, index=False)
            f.write(df_contents)
    else:
        continue
    #print(len(rows))

#Move files


