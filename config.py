import torch


class Config:
    def __init__(self):

        self.seed = 1
        self.fold = 5
        self.split = 0.8

        # 模型参数
        self.model_name = 'convnextv2_large_v4_final'
        self.pretrained = True
        self.input_channels = 1
        self.num_classes = 1


        # 训练参数

        self.num_epochs = 61
        self.lr = 0.00002  # 0.0005
        self.step_size = 2
        self.gamma = 0.8
        self.batch_size = 16

        # 数据增强和预处理
        self.image_size = (224, 224)
        self.normalize_mean = [0.5]
        self.normalize_std = [0.5]

        self.data_number = f'{self.seed}_{self.fold}'

        # 其他
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_file = f'data/submission/{self.model_name}_split_0.8_bs_16_lr_{self.lr}_seed_{self.seed}_aug3_grouploc_final.csv'
        # self.train_csv_file = 'data/data_1/df_task1_train_challenge.csv'
        # self.train_root_dir = 'data/data_1/train'
        # self.test_csv_file = 'data/data_1/df_task1_val_challenge.csv'
        # self.test_root_dir = 'data/data_1/val'
        self.final_csv_file = 'data/data_1/combined_dataset.csv'
        self.final_root_dir = 'data/data_1/final'
        self.final_train_csv_file = f'data/data_1/train_fold_{self.data_number}.csv'
        self.final_val_csv_file = f'data/data_1/val_fold_{self.data_number}.csv'


# 实例化配置类
config = Config()
