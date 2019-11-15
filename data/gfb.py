"""Generated flying boxes dataset
"""

import os.path as ospath
import os
import cv2
import json
import torch
import torch.utils.data as data

GFB_CLASSES = ("box")
GFB_ROOT = ospath.join("data", "gfb")

def bb_percent_to_pixel(frame, target):
    xmin, ymin, xmax, ymax, label = target

    xmin = int(xmin * frame.shape[1])
    xmax = int(xmax * frame.shape[1])
    ymin = frame.shape[0] - int(ymin * frame.shape[0])
    ymax = frame.shape[0] - int(ymax * frame.shape[0])
    
    return [ xmin, ymin, xmax, ymax, label_id ]
    

def transform_annotation(frame, targets):
    label_dict = {"sphere": 0, "box": 1 }
    
    res = []
    for target in targets:
        xmin, ymin, xmax, ymax, label = target
        label_id = label_dict[label]
        res.append([ xmin, ymin, xmax, ymax, label_id ])
    return res

class GFBDetection(data.Dataset):
    def __init__(self, root, name='GFB'):
        self.root = root
        self.name = name
        
        video_path = ospath.join(self.root, "video")
        annotation_path = ospath.join(self.root, "annotation")

        self.num_samples = len([ f for f in os.listdir(video_path) if ospath.isfile(ospath.join(video_path, f)) ])
        self._imgpath = ospath.join(video_path, "frame_{}.bmp")
        self._annotationpath = ospath.join(annotation_path, "frame_{}.json")

    def __getitem__(self, index):
        img, target, h, w = self.pull_item(index)
        return img, target

    def __len__(self):
        return self.num_samples

    def pull_item(self, index):
        img = self.pull_image(index)
        target = transform_annotation(img, self.pull_anno(index))

        height, width, _channels = img.shape
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, target, height, width

    def pull_image(self, index):
        return cv2.imread(self._imgpath.format((index)))

    def pull_anno(self, index):
        with open(self._annotationpath.format(index), "r") as f:
            # Assumes array of bounding boxes with classes. I.e. [ [ xmin, ymin, xmax, ymax, label ], ... ]
            return json.load(f)

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


if __name__ == '__main__':
    print("Using data directory: {}".format(GFB_ROOT))
    dataset = GFBDetection(GFB_ROOT)
    print("Number of samples: {}".format(len(dataset)))

    frame_count = 10
    frame = dataset.pull_image(frame_count).copy()

    # Draw target bounding boxes
    for target in transform_annotation(frame, dataset.pull_anno(frame_count)):
        xmin, ymin, xmax, ymax, label = bb_percent_to_pixel(target)

        color = (255, 255, 255)
        if label == 0:
            color = (0, 100, 255)
        elif label == 1:
            color = (255, 100, 0)
        
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
    
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    
    
