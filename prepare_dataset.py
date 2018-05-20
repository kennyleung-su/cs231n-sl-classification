# This script will create the dataset/ directory and populate it with data.
import os
import shutil

folders = ['train', 'valid', 'test']

ROOT_DIR, _ = os.path.split(os.getcwd())
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')

def main():
    if os.path.exists(DATASET_DIR):
        print('Dataset has already been created. Exiting program now...')
        return

    for folder in folders:
        print('Moving {0} set according to its labels'.format(folder))
        label_filename = os.path.join(ROOT_DIR, folder + '.txt')
        label_file = open(label_filename, 'r')

        counter = 0
        # loop through the lines
        for line in label_file:
            print("Processing line: {}".format(counter))
            rgb, kinect, label = line.split(' ')
            label = label.replace('\n', '')
            new_data_dir = os.path.join(DATASET_DIR, folder, label)
            rgb_vid_full = os.path.join(ROOT_DIR, rgb)
            rgb_frame_full, _ = os.path.splitext(rgb_vid_full)
            kinect_vid_full = os.path.join(ROOT_DIR, kinect)
            kinect_frame_full, _ = os.path.splitext(kinect_vid_full)

            if os.path.exists(rgb_vid_full):
                _, filename = os.path.split(rgb_vid_full)
                new_path = os.path.join(new_data_dir, filename)
                if not os.path.exists(new_path):
                    os.makedir(new_path)
                shutil.move(rgb_vid_full, new_path)

            if os.path.exists(rgb_frame_full):
                _, filename = os.path.split(rgb_frame_full)
                new_path = os.path.join(new_data_dir, filename)
                if not os.path.exists(new_path):
                    os.makedir(new_path)
                shutil.move(rgb_frame_full, new_path)

            if os.path.exists(kinect_vid_full):
                _, filename = os.path.split(kinect_vid_full)
                new_path = os.path.join(new_data_dir, filename)
                if not os.path.exists(new_path):
                    os.makedir(new_path)
                shutil.move(kinect_vid_full, new_path)

            if os.path.exists(kinect_frame_full):
                _, filename = os.path.split(kinect_frame_full)
                new_path = os.path.join(new_data_dir, filename)
                if not os.path.exists(new_path):
                    os.makedir(new_path)
                shutil.move(kinect_frame_full, new_path)

            counter += 1

        label_file.close()

main()