import os 
import random

# Prepare a random input image from the training set
train_horse_dir = './horse-or-human/horses'
train_horse_names = os.listdir(train_horse_dir)
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]

train_human_dir = './horse-or-human/humans'
train_human_names = os.listdir(train_human_dir)
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]

img_path = random.choice(horse_img_files + human_img_files)
print(img_path)