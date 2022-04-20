import os

import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms, datasets

from utils.global_settings import CHECKPOINT_PATH, RESULT_DIR
from model import simplenet


def test(net, use_gpu, test_loader, result_file_name):
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    net.eval()
    with torch.no_grad():
        test_acc = 0
        label_list = []
        probability_predicted_list = []
        label_predicted_list = []
        for batch_images, batch_labels in test_loader:
            if use_gpu:
                batch_images = Variable(batch_images.cuda())
                batch_labels = Variable(batch_labels.cuda())
            else:
                batch_images = Variable(batch_images)
                batch_labels = Variable(batch_labels)

            output = net(batch_images)
            softmax = nn.Softmax(dim=1)
            output = softmax(output)
            _, pred_label = output.max(1)
            num_correct = pred_label.eq(batch_labels).sum()
            test_acc += num_correct.item()

            label_list.extend(batch_labels.cpu().numpy().tolist())
            probability_predicted_list.extend(output.cpu().numpy().tolist())
            label_predicted_list.extend(pred_label.cpu().numpy().tolist())

        test_acc = test_acc / len(test_loader.dataset)
        print("Test Acc: %f" % test_acc)
        df = pd.DataFrame(probability_predicted_list,
                          columns=['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9'])
        df.insert(df.shape[1], 'label-pre', label_predicted_list)
        df.insert(df.shape[1], 'label_gt', label_list)
        df.to_csv(os.path.join(RESULT_DIR, result_file_name + str(round(test_acc, 3))) + '.csv')


if __name__ == '__main__':
    data_root_path = '/data/zengnanrong/Architectural_Heritage_Elements_Dataset/'
    assert os.path.exists(data_root_path), "{} path does not exist.".format(data_root_path)

    test_dataset = datasets.ImageFolder(root=os.path.join(data_root_path, "test"), transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=64,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=32)
    net = simplenet.generate_model(use_gpu=True, gpu_id=['1'], in_channels=3, num_classes=10)
    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, 'simplenet.pkl'))
    net.load_state_dict(checkpoint['state_dict'])
    test(net, True, test_loader, 'simplenet_test_')
