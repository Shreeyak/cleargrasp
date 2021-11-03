import glob
import os

import imageio
import numpy as np

rgb_path = '/home/gani/deeplearning/brain/datasets/matterport3d/matterport_rgb/v1/scans/'
normal_path = '/home/gani/deeplearning/brain/datasets/matterport3d/matterport_render_normal'

house_list = [os.path.join(rgb_path, name) for name in os.listdir(rgb_path) if os.path.isdir(os.path.join(rgb_path, name))]
rgb_folder = 'matterport_color_images'
normal_folder = 'mesh_images'

rgb_list = []
for house_id in range(len(house_list)):
    rgb_dir = os.path.join(house_list[house_id], os.path.basename(house_list[house_id]), rgb_folder)
    # print(rgb_dir)
    for (dirpath, dirnames, filenames) in os.walk(rgb_dir):
        for filename in filenames:
            rgb_list.append(os.path.join(rgb_dir, filename))

# print(rgb_list)

for i in range(len(rgb_list)):
    house_id = rgb_list[i].split('/')[-4]
    normal_file_folder = os.path.join(normal_path, house_id, normal_folder)
    normals = []
    for (dirpath, dirnames, filenames) in os.walk(normal_file_folder):
        for filename in filenames:
            if (filename.endswith(".png")):
                # print(filename.split('_')[0] + '_' + 'i' + filename.split('_')[1][1] + '_' + filename.split('_')[2])
                # print((rgb_list[i].split('/')[-1]).split('.')[0])
                if ((filename.split('_')[0] + '_' + 'i' + filename.split('_')[1][1] + '_' + filename.split('_')[2])  == (rgb_list[i].split('/')[-1]).split('.')[0]):
                    normals.append(os.path.join(dirpath, filename))
    if len(normals) < 3:
        raise ValueError('labels are not present for {}'.format(rgb_list[i]))
    normals.sort()
    print(rgb_list[i])
    print(normals)
    x = imageio.imread(normals[0])
    np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint16)
    x = x / 32768
    x = x - 1
    y = imageio.imread(normals[1])
    y = y / 32768
    y = y - 1
    z = imageio.imread(normals[2])
    z = z / 32768
    z = z - 1
    xyz = np.stack((x, y, z), axis=2)
    # print(xyz)
    print(xyz.shape)

