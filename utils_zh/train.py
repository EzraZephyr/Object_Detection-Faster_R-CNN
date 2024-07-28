import csv
import time

import torch

from utils_zh.data_process import data_process
from utils_zh.faster_rcnn import load_model



def train():

    train_loader = data_process(
        image_dir='../data/VOC2012/animal_images_train',
        annotation_dir='../data/VOC2012/animal_annotations_train',
    )
    model = load_model()
    print("Completed loading model")
    # 加载模型

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)


    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    # 使用SGD优化器 节省内存

    train_log = '../model/training.log'
    train_csv = '../model/training_metrics.csv'

    with open(train_log, 'w') as log_file, open(train_csv, 'w', newline='') as csv_file:
        fieldnames = ['epoch', 'loss']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        # 定义训练日志和csv文件 方便后续构图

        num_epochs = 10
        print("Start training")
        print("Number of train_loader: ", len(train_loader))

        for epoch in range(num_epochs):
            model.train()
            # 将模型调整为训练模式

            total_loss = 0.0
            start_time = time.time()

            for images, targets in train_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                # 将每个待处理数据移动到定义好的设备上

                optimizer.zero_grad()
                loss_dict = model(images, targets)
                # 梯度清零和通过内置的__call__方法实现分类损失和边界框回归损失

                losses = torch.tensor(0.0, device=device)
                # 初始化损失数为张量 因为下文values提取的是整型不是张量

                for loss in loss_dict.values():
                    losses += loss
                    # 将分类损和边界框狂损失相加一起处理

                losses.backward()
                optimizer.step()
                # 反向传播和更新参数

                total_loss += losses.item()
                avg_loss = total_loss / len(train_loader.dataset)
                elapsed_time = time.time() - start_time
                log_message = f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s'
                print(log_message)
                log_file.write(log_message + '\n')
                writer.writerow({'epoch': epoch+1, 'loss': avg_loss})
                # 将平均损失计算出来后 输出并存入日志和csv文件


    torch.save(model.state_dict(), '../model/model.pt')
    # 保存模型

if __name__ == '__main__':
    train()