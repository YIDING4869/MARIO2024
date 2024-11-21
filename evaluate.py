import pandas as pd
import torch
from torch import device
from config import config

def evaluate_model(model, test_loader, output_file=config.output_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for samples in test_loader:
            inputs_ti = samples['image_ti'].to(device)
            inputs_ti_1 = samples['image_ti_1'].to(device)
            case_ids = samples['case_id']

            outputs = model(inputs_ti, inputs_ti_1)

            # 假设模型输出是一个包含4个类别概率的张量
            _, predicted = torch.max(outputs, 1)

            # 将预测结果和case_id保存
            for case_id, prediction in zip(case_ids, predicted):
                predictions.append({'case': case_id.item(), 'prediction': prediction.item()})

            # 生成提交文件
        df = pd.DataFrame(predictions)
        df.to_csv(output_file, index=False)

        print(f'Submission file saved as {output_file}')


# def evaluate_model(model, test_loader, output_file=config.output_file):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     predictions = []
#     correct_predictions = 0
#     total_predictions = 0
#
#     with torch.no_grad():
#         for samples in test_loader:
#             inputs_ti = samples['image_ti'].to(device)
#             inputs_ti_1 = samples['image_ti_1'].to(device)
#             case_ids = samples['case_id']
#             loc_image_ti = samples['loc_image_ti'].to(device, dtype=torch.float)
#             loc_image_ti_1 = samples['loc_image_ti_1'].to(device, dtype=torch.float)
#             patient_info = samples['patient_info'].to(device, dtype=torch.float)
#             outputs = model(inputs_ti, inputs_ti_1, loc_image_ti, loc_image_ti_1, patient_info)
#
#             # 假设模型输出是一个包含4个类别概率的张量
#             _, predicted = torch.max(outputs, 1)
#
#             # 将预测结果和case_id保存
#             for case_id, prediction in zip(case_ids, predicted):
#                 case_id_num = case_id.item()
#                 prediction_num = prediction.item()
#
#                 # 检查转换后的类型
#                 assert isinstance(case_id_num, int), f"case_id is not an integer: {case_id_num}"
#                 assert isinstance(prediction_num, int), f"prediction is not an integer: {prediction_num}"
#
#                 predictions.append({'case': case_id_num, 'prediction': prediction_num})
#
#     # 生成提交文件
#     df = pd.DataFrame(predictions)
#     df.to_csv(output_file, index=False)
#
#     print(f'Submission file saved as {output_file}')



