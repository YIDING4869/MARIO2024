import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class OCTData(Dataset):
    def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name_ti = os.path.join(self.root_dir, self.data_frame.iloc[idx, 3])
        img_name_ti_1 = os.path.join(self.root_dir, self.data_frame.iloc[idx, 4])

        image_ti = Image.open(img_name_ti).convert('L')  # 灰度图
        image_ti_1 = Image.open(img_name_ti_1).convert('L')  # 灰度图

        if self.transform:
            image_ti = self.transform(image_ti)
            image_ti_1 = self.transform(image_ti_1)

        label = int(self.data_frame.iloc[idx, 5])  # 确保标签是整数

        if label not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid label found: {label}")

        sample = {
            'image_ti': image_ti,
            'image_ti_1': image_ti_1,
            'label': label,
            'case_id': self.data_frame.iloc[idx, 0]
        }

        return sample


# class OCTData(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.data_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data_frame)
#
#     def __getitem__(self, idx):
#         img_name_ti = os.path.join(self.root_dir, self.data_frame.iloc[idx, 3])
#         img_name_ti_1 = os.path.join(self.root_dir, self.data_frame.iloc[idx, 4])
#
#         image_ti = Image.open(img_name_ti).convert('L')  # 灰度图
#         image_ti_1 = Image.open(img_name_ti_1).convert('L')  # 灰度图
#
#         if self.transform:
#             image_ti = self.transform(image_ti)
#             image_ti_1 = self.transform(image_ti_1)
#         label = int(self.data_frame.iloc[idx, 5])  # 确保标签是整数
#
#         if label not in [0, 1, 2, 3]:
#             raise ValueError(f"Invalid label found: {label}")
#         sample = {
#             'image_ti': image_ti,
#             'image_ti_1': image_ti_1,
#             'label': int(self.data_frame.iloc[idx, 5]),
#             'case_id': self.data_frame.iloc[idx, 0]
#         }
#
#         return sample




### combine with 2 images and the patient information

# import os
# import pandas as pd
# import torch
# from torchvision import transforms
# from torch.utils.data import Dataset
# from PIL import Image
#
#
# class OCTData(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.data_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#         # 指定不需要的列
#         self.exclude_columns = ['label', 'split_type', 'image_at', 'localize_at','']
#         # 提取所有列
#         self.patient_info_columns = [col for col in self.data_frame.columns if col not in self.exclude_columns]
#
#     def __len__(self):
#         return len(self.data_frame)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image_at'])
#         loc_img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['localize_at'])
#         image = Image.open(img_name).convert('L')  # Convert to grayscale
#         localized_image = Image.open(loc_img_name).convert('L')  # Convert to grayscale
#
#         if self.transform:
#             image = self.transform(image)
#             localized_image = self.transform(localized_image)
#
#         # Extract patient information excluding specified columns
#         patient_info = self.data_frame.iloc[idx][self.patient_info_columns].values.astype('float32')
#         patient_info = torch.tensor(patient_info)
#
#         # Print extracted patient information for verification
#         print(f"Extracted patient information for index {idx}: {self.patient_info_columns}")
#         print(f"Values: {patient_info}")
#
#         label = int(self.data_frame.iloc[idx]['label'])
#
#         return image, localized_image, patient_info, label

# import os
# import pandas as pd
# import torch
# from torchvision import transforms
# from torch.utils.data import Dataset
# from PIL import Image
#
#
# class OCTData(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.data_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#         # 指定不需要的列
#         self.exclude_columns = ['label', 'split_type', 'image_at_ti', 'image_at_ti+1', 'LOCALIZER_at_ti+1',
#                                 'LOCALIZER_at_ti', 'case']
#
#
#         # 提取所有列
#         self.patient_info_columns = [col for col in self.data_frame.columns if col not in self.exclude_columns]
#
#         # 打印列名进行检查
#         print("所有列名:", self.data_frame.columns)
#         print("患者信息列:", self.patient_info_columns)
#         self.data_frame = pd.get_dummies(self.data_frame, columns=self.patient_info_columns, drop_first=True)
#         self.patient_info_columns = [col for col in self.data_frame.columns if col not in self.exclude_columns]
#
#
#     def __len__(self):
#         return len(self.data_frame)
#
#     def __getitem__(self, idx):
#         img_name_ti = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image_at_ti'])
#         img_name_ti_1 = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image_at_ti+1'])
#         loc_img_name_ti = os.path.join(self.root_dir, self.data_frame.iloc[idx]['LOCALIZER_at_ti'])
#         loc_img_name_ti_1 = os.path.join(self.root_dir, self.data_frame.iloc[idx]['LOCALIZER_at_ti+1'])
#
#         image_ti = Image.open(img_name_ti).convert('L')  # Convert to grayscale
#         image_ti_1 = Image.open(img_name_ti_1).convert('L')  # Convert to grayscale
#         loc_image_ti = Image.open(loc_img_name_ti).convert('L')  # Convert to grayscale
#         loc_image_ti_1 = Image.open(loc_img_name_ti_1).convert('L')  # Convert to grayscale
#
#         if self.transform:
#             image_ti = self.transform(image_ti)
#             image_ti_1 = self.transform(image_ti_1)
#             loc_image_ti = self.transform(loc_image_ti)
#             loc_image_ti_1 = self.transform(loc_image_ti_1)
#
#         # Extract patient information excluding specified columns
#         patient_info = self.data_frame.iloc[idx][self.patient_info_columns].values.astype('float32')
#         patient_info = torch.tensor(patient_info)
#
#         label = int(self.data_frame.iloc[idx]['label'])
#
#         # return image_ti, image_ti_1, loc_image_ti, loc_image_ti_1, patient_info, label
#
#         sample = {'image_ti': image_ti,
#                   'label': label, 'patient_info': patient_info, 'image_ti_1': image_ti_1,
#                   'loc_image_ti': loc_image_ti,
#                   'loc_image_ti_1': loc_image_ti_1}
#
#         return sample


class TestData(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name_ti = os.path.join(self.root_dir, self.data_frame.iloc[idx, 3])
        img_name_ti_1 = os.path.join(self.root_dir, self.data_frame.iloc[idx, 4])

        image_ti = Image.open(img_name_ti).convert('L')  # 灰度图
        image_ti_1 = Image.open(img_name_ti_1).convert('L')  # 灰度图

        if self.transform:
            image_ti = self.transform(image_ti)
            image_ti_1 = self.transform(image_ti_1)

        sample = {
            'image_ti': image_ti,
            'image_ti_1': image_ti_1,
            'case_id': self.data_frame.iloc[idx, 14]
        }

        return sample

# class TestData(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.data_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#         # 指定不需要的列
#         self.exclude_columns = ['split_type', 'image_at_ti', 'image_at_ti+1', 'LOCALIZER_at_ti+1',
#                                 'LOCALIZER_at_ti', 'case']
#         # 提取所有列
#         self.patient_info_columns = [col for col in self.data_frame.columns if col not in self.exclude_columns]
#         self.data_frame = pd.get_dummies(self.data_frame, columns=self.patient_info_columns, drop_first=True)
#         self.patient_info_columns = [col for col in self.data_frame.columns if col not in self.exclude_columns]
#     def __len__(self):
#         return len(self.data_frame)
#
#     def __getitem__(self, idx):
#         img_name_ti = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image_at_ti'])
#         img_name_ti_1 = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image_at_ti+1'])
#         loc_img_name_ti = os.path.join(self.root_dir, self.data_frame.iloc[idx]['LOCALIZER_at_ti'])
#         loc_img_name_ti_1 = os.path.join(self.root_dir, self.data_frame.iloc[idx]['LOCALIZER_at_ti+1'])
#
#         image_ti = Image.open(img_name_ti).convert('L')  # Convert to grayscale
#         image_ti_1 = Image.open(img_name_ti_1).convert('L')  # Convert to grayscale
#         loc_image_ti = Image.open(loc_img_name_ti).convert('L')  # Convert to grayscale
#         loc_image_ti_1 = Image.open(loc_img_name_ti_1).convert('L')  # Convert to grayscale
#
#         if self.transform:
#             image_ti = self.transform(image_ti)
#             image_ti_1 = self.transform(image_ti_1)
#             loc_image_ti = self.transform(loc_image_ti)
#             loc_image_ti_1 = self.transform(loc_image_ti_1)
#
#
#         patient_info = self.data_frame.iloc[idx][self.patient_info_columns].values.astype('float32')
#         patient_info = torch.tensor(patient_info)
#
#
#         # return image_ti, image_ti_1, loc_image_ti, loc_image_ti_1, patient_info, label
#
#         sample = {'image_ti': image_ti,
#                     'patient_info': patient_info, 'image_ti_1': image_ti_1,
#                   'loc_image_ti': loc_image_ti,
#                   'loc_image_ti_1': loc_image_ti_1,
#                   'case': self.data_frame.iloc[idx]['case']}
#
#         return sample
