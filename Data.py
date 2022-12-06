import torch
import glob
import numpy as np
from PIL import Image
import pandas as pd
import copy


dir = '/home/maihaoxiang/dataset/HC18/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Loads to the GPU 
x_size = (512, 512)
y_size = (512, 512)
# x_size = (572, 572)
# y_size = (572, 572)

class Data(torch.utils.data.Dataset):

    def __init__(self, data, transforms=None):
        self.transforms = transforms
        self.data = data
        self.x_train = sorted(glob.glob(dir+'training_set/*HC.png'))
        self.x_val = sorted(glob.glob(dir+'val_set/*HC.png'))
        df_sl_x = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/training_set_pixel_size_and_HC.xlsx",usecols=['短轴左x']))
        df_sl_y = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/training_set_pixel_size_and_HC.xlsx",usecols=['短轴左y(pixel)']))
        df_sr_x = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/training_set_pixel_size_and_HC.xlsx",usecols=['短轴右x']))
        df_sr_y = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/training_set_pixel_size_and_HC.xlsx",usecols=['短轴右y']))
        df_lu_x = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/training_set_pixel_size_and_HC.xlsx",usecols=['长轴上x']))
        df_lu_y = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/training_set_pixel_size_and_HC.xlsx",usecols=['长轴上y']))
        df_ld_x = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/training_set_pixel_size_and_HC.xlsx",usecols=['长轴下x']))
        df_ld_y = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/training_set_pixel_size_and_HC.xlsx",usecols=['长轴下y']))
        #self.x_test = sorted(glob.glob(dir+'test_set/*HC.png'))
        self.valid_person_list = []#用来存储合法的人体数据信息

        for i in range(len(self.x_train)):
            ann_keypoint = []
            a1 = df_sl_x[i]
            b1 = df_sl_y[i]
            p1 = np.array((a1,b1))
            ann_keypoint.append(p1)
            a2 = df_sr_x[i]
            b2 = df_sr_y[i]
            p2 = np.array((a2,b2))
            ann_keypoint.append(p2)
            a3 = df_lu_x[i]
            b3 = df_lu_y[i]
            p3 = np.array((a3,b3))
            ann_keypoint.append(p3)
            a4 = df_ld_x[i]
            b4 = df_ld_y[i]
            p4 = np.array((a4,b4))
            ann_keypoint.append(p4)
            vis = np.ones(4)
            info = {
                "id":i,
                "box": [0, 0, 512, 512],
            }
            info["keypoints"] = np.array(ann_keypoint)
            info["visible"] = vis
            self.valid_person_list.append(info)
        # self.y_train = sorted(glob.glob(dir+'training_set/*_Annotation.png'))
        # self.y_val = sorted(glob.glob(dir+'val_set/*_Annotation.png'))
        #self.y_test = sorted(glob.glob(dir+'test_set/*HC_Annotation.png'))
        val_df_sl_x = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/val_set_pixel_and_HC.xlsx",usecols=['短轴左x']))
        val_df_sl_y = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/val_set_pixel_and_HC.xlsx",usecols=['短轴左y']))
        val_df_sr_x = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/val_set_pixel_and_HC.xlsx",usecols=['短轴右x']))
        val_df_sr_y = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/val_set_pixel_and_HC.xlsx",usecols=['短轴右y']))
        val_df_lu_x = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/val_set_pixel_and_HC.xlsx",usecols=['长轴上x']))
        val_df_lu_y = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/val_set_pixel_and_HC.xlsx",usecols=['长轴上y']))
        val_df_ld_x = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/val_set_pixel_and_HC.xlsx",usecols=['长轴下x']))
        val_df_ld_y = np.array(pd.read_excel("/home/maihaoxiang/dataset/HC18/val_set_pixel_and_HC.xlsx",usecols=['长轴下y']))
        #self.x_test = sorted(glob.glob(dir+'test_set/*HC.png'))
        self.val_valid_person_list = []#用来存储合法的人体数据信息

        for j in range(len(self.x_val)-1):
            ann_keypoint = []
            a1 = val_df_sl_x[j]
            b1 = val_df_sl_y[j]
            p1 = np.array((a1,b1))
            ann_keypoint.append(p1)
            a2 = val_df_sr_x[j]
            b2 = val_df_sr_y[j]
            p2 = np.array((a2,b2))
            ann_keypoint.append(p2)
            a3 = val_df_lu_x[j]
            b3 = val_df_lu_y[j]
            p3 = np.array((a3,b3))
            ann_keypoint.append(p3)
            a4 = val_df_ld_x[j]
            b4 = val_df_ld_y[j]
            p4 = np.array((a4,b4))
            ann_keypoint.append(p4)
            vis = np.ones(4)
            info = {
                "id":j,
            }
            info["keypoints"] = np.array(ann_keypoint)
            info["visible"] = vis
            self.val_valid_person_list.append(info)


    def __len__(self):

        if(self.data == 'train'):
            return len(self.x_train)

        elif(self.data == 'validate'):
            return len(self.x_val)

        # elif(self.data == 'test'):
        #     return len(self.x_test)

        else:
            return ValueError("No data")



    def __getitem__(self, idx):

        if(self.data == 'train'):
            target = copy.deepcopy(self.valid_person_list[idx])
            image = np.array(Image.open(self.x_train[idx]).convert("RGB").resize(x_size))
            #image = image.transpose(2,1,0)
            image, person_info = self.transforms(image, target)
            return image,target
            #return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

        elif(self.data == 'validate'):
            target = copy.deepcopy(self.val_valid_person_list[idx])
            image = np.array(Image.open(self.x_val[idx]).convert("RGB").resize(x_size))
            #image = image.transpose(2,1,0)
            image, person_info = self.transforms(image, target)
            return image,target
        elif(self.data == 'test'):
            # x = np.array(Image.open(self.x_test[idx]).convert("L").resize(x_size)).reshape(1, 572, 572)
            # y = np.array(Image.open(self.y_test[idx]).convert("L").resize(y_size)).reshape(1, 572, 572) # Convert to gray scale
            x = np.array(Image.open(self.x_test[idx]).convert("RGB").resize(x_size)).reshape(512, 512,3)
            y = np.array(Image.open(self.y_test[idx]).convert("RGB").resize(y_size)).reshape(512, 512,3) # Convert to gray scale
            return torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device)
            #return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

        else:
            return ValueError("No data")
