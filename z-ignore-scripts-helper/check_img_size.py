import os
import imageio
import glob

images_dir = '/home/shrek/greppy/brain/transparent_detection/models/greppy-gbrain-transparent-detection/data/datasets/test/studio_pics_sorted/selected_test/d435'

list_img = sorted(glob.glob(os.path.join(images_dir, '*' + 'jpg')))

for img in list_img:
    image = imageio.imread(img)

    height = image.shape[0]
    width = image.shape[1]

    if height != 720:
        print('HEIGHT NOT 720: {}'.format(os.path.basename(img)))
    if width != 1280:
        print('WIDTH NOT 1280: {}'.format(os.path.basename(img)))
    else:
        print('{}'.format(os.path.basename(img)))