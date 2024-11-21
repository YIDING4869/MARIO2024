import torch
import torchvision
from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
from torchvision.models import EfficientNet_V2_L_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
import timm


# resnet50
# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         self.model = models.resnet50(pretrained=True)
#         self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改第一层以接受单通道输入
#         self.model.fc = nn.Identity()  # 移除最后一层全连接层，因为我们要提取特征
#         self.fc = nn.Linear(2048, 4)  # 添加自己的分类层，根据ResNet50的输出特征维度
#
#     def forward(self, image_ti, image_ti_1):
#         # 计算图像的绝对差异
#         image_diff = torch.abs(image_ti - image_ti_1)
#         # 提取特征
#         x = self.model(image_diff)
#         # 分类
#         x = self.fc(x)
#         return x


##### efficient v2


# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         # 使用 EfficientNet V2
#         self.model = torchvision.models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
#
#         # 修改第一层以接受单通道输入
#         self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
#
#         # 替换最后一层分类器
#         num_features = self.model.classifier[1].in_features
#         self.model.classifier[1] = nn.Linear(num_features, 4)  # Replace with your desired output size
#
#
#     def forward(self, image_ti, image_ti_1):
#         # 计算图像的绝对差异
#         image_diff = torch.abs(image_ti - image_ti_1)
#         # 提取特征并分类
#         x = self.model(image_diff)
#         return x
#
#


### VIT

# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         # Load Vision Transformer (ViT) model
#         self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
#
#         # Modify the first convolutional layer to accept 1 channel instead of 3
#         self.model.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
#
#         # Modify the classifier layer for the desired output size
#         num_features = self.model.heads.head.in_features
#         self.model.heads.head = nn.Linear(num_features, 4)
#
#     def forward(self, image_ti, image_ti_1):
#         # Calculate absolute difference between images
#         image_diff = torch.abs(image_ti - image_ti_1)
#
#         # Resize to match ViT input size if necessary (224x224)
#         image_diff = nn.functional.interpolate(image_diff, size=(224, 224), mode='bilinear', align_corners=False)
#
#         # Extract features and classify
#         x = self.model(image_diff)
#         return x

## convnext2_large


# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         # Load the ConvNeXt V2 model from timm
#         self.model = timm.create_model('convnextv2_large', pretrained=True, in_chans=1, num_classes=4)
#
#     def forward(self, image_ti, image_ti_1):
#         # Calculate the absolute difference between the two images
#         image_diff = torch.abs(image_ti - image_ti_1)
#         # Extract features and classify
#         x = self.model(image_diff)
#         return x



## convnext2_large_v2
# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         # Load the ConvNeXt V2 model from timm
#         self.model = timm.create_model('convnextv2_large', pretrained=True, in_chans=2, num_classes=4)
#
#     def forward(self, image_ti, image_ti_1):
#         # Calculate the absolute difference between the two images
#         image_diff = torch.abs(image_ti - image_ti_1)
#         # Extract features and classify
#         x3 = torch.cat((image_ti,image_diff), dim=1)
#         x = self.model(x3)
#         return x




# convnext2_large_v3
# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         # Load the ConvNeXt V2 model from timm
#         self.model = timm.create_model('convnextv2_large', pretrained=True, in_chans=3, num_classes=4)
#
#     def forward(self, image_ti, image_ti_1):
#         # Calculate the absolute difference between the two images
#         image_diff = torch.abs(image_ti - image_ti_1)
#         # Extract features and classify
#         x3 = torch.cat((image_ti,image_ti_1,image_diff), dim=1)
#         x = self.model(x3)
#         return x
#
# #
# # import torch
# # import torch.nn as nn
# # import timm



# # v1
# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         # 使用 timm 导入预训练的 ConvNeXt V2-L 模型
#         self.model = timm.create_model('convnextv2_large', pretrained=True)
#         self.model.stem[0] = nn.Conv2d(1, self.model.stem[0].out_channels, kernel_size=4, stride=4)
#         self.model.head.fc = nn.Identity()  # 移除最后一层分类器
#
#         # 定义用于比较的全连接层
#         self.fc = nn.Sequential(
#             nn.Linear(1536, 512),
#             nn.ReLU(),
#             nn.Linear(512, 4),  # 输出类别数
#             # nn.Softmax(dim=1)
#         )
#
#     def forward_one(self, x):
#         return self.model(x)
#
#     def forward(self, input1, input2):
#         output1 = self.forward_one(input1)
#         output2 = self.forward_one(input2)
#         diff = torch.abs(output1 - output2)  # 计算差异
#         out = self.fc(diff)
#         return out

# v0
class OCTClassifier0(nn.Module):
    def __init__(self, num_classes=4):
        super(OCTClassifier0, self).__init__()
        # 使用 timm 导入预训练的 ConvNeXt V2-L 模型
        self.model = timm.create_model('convnextv2_large', pretrained=True)

        # 调整第一个卷积层以接受单通道图像
        self.model.stem[0] = nn.Conv2d(1, self.model.stem[0].out_channels, kernel_size=4, stride=4)

        self.model.head.fc = nn.Identity()  # 移除最后一层分类器

        # 定义用于处理每个特征的全连接层
        self.fc_single = nn.Sequential(
            nn.Linear(1536, 512),  # 输入大小取决于模型输出大小
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # 定义用于处理差异特征的全连接层
        self.fc_diff = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # 定义最终分类的全连接层
        self.fc_final = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),  # 输出类别数
            # nn.Softmax(dim=1)  # 使用Softmax进行多类别分类
        )

    def forward_one(self, x):
        return self.model(x)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)

        diff = torch.abs(output1 - output2)  # 计算差异

        # 处理每个特征和差异特征
        feature1 = self.fc_single(output1)
        feature2 = self.fc_single(output2)
        feature_diff = self.fc_diff(diff)

        # 连接特征并通过最终分类层
        combined_features = torch.cat((feature1, feature2, feature_diff), dim=1)
        out = self.fc_final(combined_features)

        return out

class OCTClassifier4(nn.Module):
    def __init__(self, num_classes=4):
        super(OCTClassifier4, self).__init__()
        # 使用 timm 导入预训练的 ConvNeXt V2-L 模型
        self.model = timm.create_model('convnextv2_large', pretrained=True)

        # 调整第一个卷积层以接受单通道图像
        self.model.stem[0] = nn.Conv2d(1, self.model.stem[0].out_channels, kernel_size=4, stride=4)

        self.model.head.fc = nn.Identity()  # 移除最后一层分类器

        # 定义用于处理每个特征的全连接层
        self.fc_single = nn.Sequential(
            nn.Linear(1536, 512),  # 输入大小取决于模型输出大小
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # 定义用于处理差异特征的全连接层
        self.fc_diff = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # 定义最终分类的全连接层
        self.fc_final = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),  # 输出类别数
            # nn.Softmax(dim=1)  # 使用Softmax进行多类别分类
        )

    def forward_one(self, x):
        return self.model(x)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)

        diff = output1 - output2  # 计算差异

        # 处理每个特征和差异特征
        feature1 = self.fc_single(output1)
        feature2 = self.fc_single(output2)
        feature_diff = self.fc_diff(diff)

        # 连接特征并通过最终分类层
        combined_features = torch.cat((feature1, feature2, feature_diff), dim=1)
        out = self.fc_final(combined_features)

        return out



# v2

class OCTClassifierv2(nn.Module):
    def __init__(self, num_classes=4):
        super(OCTClassifierv2, self).__init__()
        # 使用 timm 导入预训练的 ConvNeXt V2-L 模型
        self.model = timm.create_model('convnextv2_large', pretrained=True)

        # 调整第一个卷积层以接受单通道图像
        self.model.stem[0] = nn.Conv2d(1, self.model.stem[0].out_channels, kernel_size=4, stride=4)

        self.model.head.fc = nn.Identity()  # 移除最后一层分类器

        # 定义用于处理每个特征的全连接层
        self.fc_single = nn.Sequential(
            nn.Linear(1536, 512),  # 输入大小取决于模型输出大小
            nn.ReLU(),
            nn.Linear(512, 256)
        )


        # 定义最终分类的全连接层
        self.fc_final = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),  # 输出类别数
            # nn.Softmax(dim=1)  # 使用Softmax进行多类别分类
        )

    def forward_one(self, x):
        return self.model(x)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)


        # 处理每个特征和差异特征
        feature1 = self.fc_single(output1)
        feature2 = self.fc_single(output2)

        # 连接特征并通过最终分类层
        combined_features = torch.cat((feature1, feature2), dim=1)
        out = self.fc_final(combined_features)

        return out


class OCTClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(OCTClassifier, self).__init__()
        # 使用 timm 导入预训练的 ResNet50 模型
        self.model = timm.create_model('resnet50', pretrained=True)

        # 调整第一个卷积层以接受单通道图像
        self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)

        # 移除最后的分类器层
        self.model.fc = nn.Identity()

        # 获取特征输出大小（这里ResNet50的最后一个卷积层的输出通道数通常是2048）
        num_features = self.model.num_features

        # 定义用于处理每个特征的全连接层
        self.fc_single = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # 定义用于处理差异特征的全连接层
        self.fc_diff = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # 定义最终分类的全连接层
        self.fc_final = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward_one(self, x):
        return self.model(x)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)

        diff = torch.abs(output1 - output2)  # 计算差异

        # 处理每个特征和差异特征
        feature1 = self.fc_single(output1)
        feature2 = self.fc_single(output2)
        feature_diff = self.fc_diff(diff)

        # 连接特征并通过最终分类层
        combined_features = torch.cat((feature1, feature2, feature_diff), dim=1)
        out = self.fc_final(combined_features)

        return out



class OCTClassifier2(nn.Module):
    def __init__(self, num_classes=4):
        super(OCTClassifier2, self).__init__()
        # 使用 timm 导入预训练的 EfficientNet 模型
        self.model = timm.create_model('efficientnet_b0', pretrained=True)

        # 调整第一个卷积层以接受单通道图像
        self.model.conv_stem = nn.Conv2d(1, self.model.conv_stem.out_channels, kernel_size=3, stride=2, padding=1, bias=False)

        # 移除最后的分类器层
        self.model.classifier = nn.Identity()

        # 获取特征输出大小
        num_features = self.model.num_features

        # 定义用于处理每个特征的全连接层
        self.fc_single = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # 定义用于处理差异特征的全连接层
        self.fc_diff = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # 定义最终分类的全连接层
        self.fc_final = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward_one(self, x):
        return self.model(x)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)

        diff = torch.abs(output1 - output2)  # 计算差异

        # 处理每个特征和差异特征
        feature1 = self.fc_single(output1)
        feature2 = self.fc_single(output2)
        feature_diff = self.fc_diff(diff)

        # 连接特征并通过最终分类层
        combined_features = torch.cat((feature1, feature2, feature_diff), dim=1)
        out = self.fc_final(combined_features)

        return out




class OCTClassifier3(nn.Module):
    def __init__(self, num_classes=4):
        super(OCTClassifier3, self).__init__()
        # 使用 timm 导入预训练的 Vision Transformer 模型
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)

        # 调整第一个卷积层以接受单通道图像
        self.model.patch_embed.proj = nn.Conv2d(1, self.model.patch_embed.proj.out_channels, kernel_size=16, stride=16)

        # 移除最后的分类器层
        self.model.head = nn.Identity()

        # 获取特征输出大小
        num_features = self.model.num_features

        # 定义用于处理每个特征的全连接层
        self.fc_single = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # 定义用于处理差异特征的全连接层
        self.fc_diff = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # 定义最终分类的全连接层
        self.fc_final = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward_one(self, x):
        return self.model(x)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)

        diff = torch.abs(output1 - output2)  # 计算差异

        # 处理每个特征和差异特征
        feature1 = self.fc_single(output1)
        feature2 = self.fc_single(output2)
        feature_diff = self.fc_diff(diff)

        # 连接特征并通过最终分类层
        combined_features = torch.cat((feature1, feature2, feature_diff), dim=1)
        out = self.fc_final(combined_features)

        return out









# # v2
#
# class OCTClassifier(nn.Module):
#     def __init__(self, num_classes=4):
#         super(OCTClassifier, self).__init__()
#         # 使用 timm 导入预训练的 ConvNeXt V2-L 模型
#         self.model = timm.create_model('convnextv2_large', pretrained=True)
#
#         # 调整第一个卷积层以接受单通道图像
#         self.model.stem[0] = nn.Conv2d(1, self.model.stem[0].out_channels, kernel_size=4, stride=4)
#
#         self.model.head.fc = nn.Identity()  # 移除最后一层分类器
#
#         # 定义用于处理每个特征的全连接层
#         self.fc_single = nn.Sequential(
#             nn.Linear(1536, 512),  # 输入大小取决于模型输出大小
#             nn.ReLU(),
#             nn.Linear(512, 256)
#         )
#
#
#         # 定义最终分类的全连接层
#         self.fc_final = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_classes),  # 输出类别数
#             # nn.Softmax(dim=1)  # 使用Softmax进行多类别分类
#         )
#
#     def forward_one(self, x):
#         return self.model(x)
#
#     def forward(self, input1, input2):
#         output1 = self.forward_one(input1)
#         output2 = self.forward_one(input2)
#
#
#         # 处理每个特征和差异特征
#         feature1 = self.fc_single(output1)
#         feature2 = self.fc_single(output2)
#
#         # 连接特征并通过最终分类层
#         combined_features = torch.cat((feature1, feature2), dim=1)
#         out = self.fc_final(combined_features)
#
#         return out




### convnext2_large_test
# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         # 使用 timm 导入 ConvNeXt V2-L 模型
#         self.model = timm.create_model('convnextv2_large', pretrained=True)
#
#         # 修改第一层以接受单通道输入
#         self.model.stem[0] = nn.Conv2d(1, self.model.stem[0].out_channels, kernel_size=4, stride=4, bias=False)
#
#         # 替换最后一层分类器
#         num_features = self.model.head.fc.in_features
#         self.model.head.fc = nn.Linear(num_features, 128)  # 中间层输出128维特征
#
#         # 定义用于融合病人信息的全连接层
#         self.fc_patient = nn.Linear(len(self.patient_info), 128)
#         self.fc_combined = nn.Linear(128 + 128, 4)  # Assuming 4 classes for classification
#
#     def forward(self, image_ti, image_ti_1, loc_image_ti, loc_image_ti_1, patient_info):
#         # 计算图像的绝对差异
#         image_diff = torch.abs(image_ti - image_ti_1)
#         loc_image_diff = torch.abs(loc_image_ti - loc_image_ti_1)
#
#         # Concatenate the image differences and pass through the model
#         x = torch.cat((image_diff, loc_image_diff), dim=1)  # Concatenate along the channel dimension
#         features = self.model(x)
#
#         # 处理病人信息
#         patient_info = self.fc_patient(patient_info)
#         patient_info = torch.relu(patient_info)
#
#         # Concatenate the extracted features with patient information
#         combined = torch.cat((features, patient_info), dim=1)
#         combined = self.fc_combined(combined)
#
#         return combined

# class OCTClassifier(nn.Module):
#     def __init__(self, num_patient_info_features):
#         super(OCTClassifier, self).__init__()
#         # 使用 timm 导入 ConvNeXt V2-L 模型
#         self.model = timm.create_model('convnextv2_large', pretrained=True)
#
#         # 修改第一层以接受单通道输入
#         self.model.stem[0] = nn.Conv2d(1, self.model.stem[0].out_channels, kernel_size=4, stride=4, bias=False)
#
#         # 替换最后一层分类器
#         num_features = self.model.head.fc.in_features
#         self.model.head.fc = nn.Linear(num_features, 128)  # 中间层输出128维特征
#
#         # 定义用于融合病人信息的全连接层
#         self.fc_patient = nn.Linear(num_patient_info_features, 128)
#         self.fc_combined = nn.Linear(256, 4)  # Assuming 4 classes for classification
#
#     def forward(self, image_ti, image_ti_1, loc_image_ti, loc_image_ti_1, patient_info):
#         # 计算图像的绝对差异
#         image_diff = torch.abs(image_ti - image_ti_1)
#         loc_image_diff = torch.abs(loc_image_ti - loc_image_ti_1)
#
#         # Concatenate the image differences and pass through the model
#         x = torch.cat((image_diff, loc_image_diff), dim=1)  # Concatenate along the channel dimension
#         features = self.model(x)
#
#         # 处理病人信息
#         patient_info = self.fc_patient(patient_info)
#         patient_info = torch.relu(patient_info)
#
#         # Concatenate the extracted features with patient information
#         combined = torch.cat((features, patient_info), dim=1)
#         combined = self.fc_combined(combined)
#
#         return combined
#
# import torch
# import torch.nn as nn
# import timm
#
# class OCTClassifier(nn.Module):
#     def __init__(self, num_patient_info_features):
#         super(OCTClassifier, self).__init__()
#         # 使用 timm 导入 ConvNeXt V2-L 模型
#         self.model = timm.create_model('convnextv2_large', pretrained=True)
#
#         # 修改第一层以接受多通道输入
#         self.model.stem[0] = nn.Conv2d(2, self.model.stem[0].out_channels, kernel_size=4, stride=4, bias=False)
#
#         # 替换最后一层分类器
#         num_features = self.model.head.fc.in_features
#         self.model.head.fc = nn.Linear(num_features, 128)  # 中间层输出128维特征
#
#         # 定义用于融合病人信息的全连接层
#         self.fc_patient = nn.Linear(num_patient_info_features, 128)
#         self.fc_combined = nn.Linear(256, 4)  # Assuming 4 classes for classification
#
#     def forward(self, image_ti, image_ti_1, loc_image_ti, loc_image_ti_1, patient_info):
#         # 计算图像的绝对差异
#         image_diff = torch.abs(image_ti - image_ti_1)
#         loc_image_diff = torch.abs(loc_image_ti - loc_image_ti_1)
#
#         # Concatenate the image differences and pass through the model
#         x = torch.cat((image_diff, loc_image_diff), dim=1)  # Concatenate along the channel dimension
#         features = self.model(x)
#
#         # 处理病人信息
#         patient_info = self.fc_patient(patient_info)
#         patient_info = torch.relu(patient_info)
#
#         # Concatenate the extracted features with patient information
#         combined = torch.cat((features, patient_info), dim=1)
#         combined = self.fc_combined(combined)
#
#         return combined