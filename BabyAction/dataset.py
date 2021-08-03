import torch.utils.data as data
import cv2
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd
import torch
import torchvision.transforms.functional as tF


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])


    @property
    def min_frame(self):
        return int(self._data[2])

    @property
    def max_frame(self):
        return int(self._data[3])


class TSNDataSet(data.Dataset):
    def __init__(self, df,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self.db_path = "/gpu-data/babyrobot/data/children/actions/kinect1/videos_orig"
        self.frames_path = "/gpu-data/babyrobot/data/children/actions/kinect1/videos_orig_frames"
        
        self.df = df



        self.video_list = self.df["video"]
    
    def get_bounding_box(self, image, joints, format="cv2"):
        # print(joints.shape)
        joints = joints.reshape((25+21+21,3))
        # print(joints.shape)
        joints[joints[:,2]<0.1] = np.nan
        joints[np.isnan(joints[:,2])] = np.nan

        joint_min_x = int(round(np.nanmin(joints[:,0])))
        joint_min_y = int(round(np.nanmin(joints[:,1])))

        joint_max_x = int(round(np.nanmax(joints[:,0])))
        joint_max_y = int(round(np.nanmax(joints[:,1])))

        # print(joint_max_x)

        expand_x = int(round(30/100 * (joint_max_x-joint_min_x)))
        expand_y = int(round(30/100 * (joint_max_y-joint_min_y)))

        if format == "cv2":
            return image[max(0,joint_min_y-expand_y):min(joint_max_y+expand_y, image.shape[0]), max(0,joint_min_x-expand_x):min(joint_max_x+expand_x,image.shape[1])]
        elif format == "PIL":
            bottom = min(joint_max_y+expand_y, image.height)
            right = min(joint_max_x+expand_x,image.width)
            top = max(0,joint_min_y-expand_y)
            left = max(0,joint_min_x-expand_x)
            # print(top, left, bottom, right)
            return tF.crop(image, top, left, bottom-top ,right-left)


    def joints(self, index):
        sample = self.df.iloc[index]

        joints_path = os.path.join('BabyAction/openpose/%s.npy'%sample['video'].replace(".avi","").replace("AggelosX","Aggelos_X"))

        joints = np.load(joints_path)
       

        return joints

    

    def _load_image(self, directory, idx, index, mode="body"):
        keypoints = self.joints(index)

        keypoints = keypoints[idx]



        sample = self.df.iloc[index]

        if self.modality == 'RGB' or self.modality == 'RGBDiff':

            frame = Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert("RGB")
            # print(idx)
            if keypoints.size == 0:
                face = frame
                pass #just do the whole frame
            elif pd.DataFrame(keypoints).isnull().values.any():
                face = frame
            else:
                face = self.get_bounding_box(frame, keypoints, format="PIL")
                # face=frame
                if face.size == 0:
                    print(keypoints)
                    face = frame

            return [face]

        elif self.modality == 'Flow':
            frame_x = Image.open(os.path.join(directory, self.image_tmpl.format('flow_x', idx))).convert('L')
            frame_y = Image.open(os.path.join(directory, self.image_tmpl.format('flow_y', idx))).convert('L')


            if keypoints.size == 0:
                body_x = frame_x
                body_y = frame_y
                pass #just do the whole frame
            elif pd.DataFrame(keypoints).isnull().values.any():
                body_x = frame_x
                body_y = frame_y
            else:
                body_x = self.get_bounding_box(frame_x, keypoints, format="PIL")
                body_y = self.get_bounding_box(frame_y, keypoints, format="PIL")

                if body_x.size == 0:
                    body_x = frame_x
                    body_y = frame_y


            return [body_x, body_y]


    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments) # + (record.min_frame+1)
            # print(record.num_frames, record.min_frame, record.max_frame)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        sample = self.df.iloc[index]


        fname = os.path.join(self.db_path,self.df.iloc[index]["video"]).replace("AggelosX","Aggelos_X")

        capture = cv2.VideoCapture(fname)
        # print(frame_count)
        capture.release()


        record_path = os.path.join(self.frames_path,sample["video"].replace(".avi","")).replace("AggelosX","Aggelos_X")

        record = VideoRecord([record_path, frame_count])

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices, index)

    def get(self, record, indices, index):

        images = list()


        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p, index, mode="face")

                images.extend(seg_imgs)

                if p < record.num_frames:
                    p += 1

        categorical = int(self.df.iloc[index]['label']) - 1


        if self.transform is None:
            process_data = images
        else:
            process_data = self.transform(images)


        return process_data, torch.tensor(categorical)

    def __len__(self):
        return len(self.df)
