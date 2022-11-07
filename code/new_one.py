import os
directory = 'bhoj'
parent_dir = 'd:/AI_CLASS/attendance system using ML&DL/code/dataset'
path = os.path.join(parent_dir, directory)
os.mkdir(path)
print("directory '%s'is created" %directory)