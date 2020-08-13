import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import albumentations as albu



class NeckDataset(Dataset):
    def __init__(self, root_dir, set, transform, file_list):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform
        self.file_list = file_list

        albu_transform = albu.Compose([
        albu.HorizontalFlip(p=0.5),
        #albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0.2, shift_limit=0.1, p=1, border_mode=0),
        albu.OneOf(
            [
                #albu.CLAHE(p=1),
                albu.RandomGamma(p=0.2),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=0.2),
            ],
            p=0.9,
        ),
    ], bbox_params=albu.BboxParams(format='coco', label_fields=['category_ids']),)
        self.albu_transform = albu_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        img, h,w = self.load_image(idx)
        annot = self.load_annotations(idx, h, w)
######################plot GT#########################
        # img_copy = img.copy()
        # count, _ = annot.shape
        # for i in range(count):
        #     cv2.rectangle(img_copy, (int(annot[i, 0]), int(annot[i, 1])), (int(annot[i, 2]+ annot[i, 0]), int(annot[i, 1]+ annot[i, 3])), (0, 255, 0), 5)
        # cv2.imwrite("1.png", img_copy * 255)
        # count = 0
######################################################################
        #sample = {'img': img, 'annot': annot}
        annot_list = annot[:,0:5].tolist()
        category_ids = annot[:,4].tolist()
        sample = self.albu_transform(image=img, bboxes = annot_list, category_ids=category_ids)
        bboxes = sample['bboxes']
        bboxes_np = np.array(bboxes)


        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        bboxes_np[:, 2] = bboxes_np[:, 0] + bboxes_np[:, 2]
        bboxes_np[:, 3] = bboxes_np[:, 1] + bboxes_np[:, 3]

# ############plot GT########
#         img_copy_2 = sample['image'].copy()
#         count_2, _ = bboxes_np.shape
#         for i in range(count_2):
#             #if bboxes[i, 4] >= 0:
#             cv2.rectangle(img_copy_2, (int(bboxes_np[i, 0]), int(bboxes_np[i, 1])),
#                        (int(bboxes_np[i,2]), int(bboxes_np[i,3])), (255, 0, 0), 1)
#         cv2.imwrite("2.png", img_copy_2)
###################################

        sample_out = {'img': sample['image'], 'annot': bboxes_np}


        if self.transform:
            sample_final = self.transform(sample_out)


        return sample_final

    def load_image(self, image_index):
        path = self.root_dir + '/' + self.set_name + '/image/' + self.file_list[image_index] + '.png'
        img = cv2.imread(path)
        print(path)
        h, w, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)/255, h,w

        #return img.astype(np.float32) / 255. , w,h

    def load_annotations(self, image_index, h, w):
        # get ground truth annotations
        path_anno = self.root_dir + '/' + self.set_name + '/annotation/' + self.file_list[image_index] + '.txt'
        annotations = np.loadtxt(path_anno)
        annotations_copy = annotations.copy()
        anno_shape = annotations.shape

        # [c_x, c_y, w, h] -> [x1,y1, x2,y2]############ Prevent GT out of range
        if len(anno_shape) == 1: # GT 1개
            annotations_copy[0] = (annotations[1] - annotations[3]/2)*w
            if annotations_copy[0] <0:
                #annotations_copy[0] = 0
                annotations_copy[0] = 0

            annotations_copy[1] = (annotations[2] - annotations[4]/2)*h
            if annotations_copy[1]  < 0 :
                #annotations_copy[1] = 0
                annotations_copy[1] = 0

            annotations_copy[2] =  annotations[3]*w

            if annotations_copy[2] + annotations_copy[0] > w:
                annotations_copy[2] = w- annotations_copy[0]
                #annotations_copy[2] = (w*0.9 - annotations_copy[0])

            annotations_copy[3] =  annotations[4]*h

            if annotations_copy[3] + annotations_copy[1] > h:
                annotations_copy[3] = (h - annotations_copy[1])

            annotations_copy[4] = 0
            annotations_copy = annotations_copy[np.newaxis,:]

        else : #GT 여러개
            for idx in range(anno_shape[0]):
                annotations_copy[idx,0] = (annotations[idx,1] - annotations[idx,3] / 2) * w
                if annotations_copy[idx,0] < 0:
                    annotations_copy[idx,0] = 0

                annotations_copy[idx,1] = (annotations[idx,2] - annotations[idx,4] / 2) * h
                if annotations_copy[idx,1] < 0:
                    annotations_copy[idx,1] = 0

                annotations_copy[idx,2] = annotations[idx,3] * w
                if annotations_copy[idx,2] + annotations_copy[idx,0] > w:
                    annotations_copy[idx,2] = w - annotations_copy[idx,0]

                annotations_copy[idx,3] = annotations[idx,4] * h
                if annotations_copy[idx,3] + annotations_copy[idx,1] > h:
                    annotations_copy[idx,3] = h - annotations_copy[idx,1]

                annotations_copy[idx,4] = 0

        ### prevent out of GT range
        return annotations_copy





class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
