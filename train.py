import os
from datetime import datetime

from torch import nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from utils.global_settings import LOG_DIR, TIME_NOW, CHECKPOINT_PATH
from model import simplenet

import torch
from torchvision import transforms, datasets


def train(net, use_gpu, train_loader, num_epochs, optimizer, scheduler, criterion, save_model_name):
    prev_time = datetime.now()

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, TIME_NOW))

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0
        net.train()
        for batch_images, batch_labels in train_loader:
            if use_gpu:
                batch_images = Variable(batch_images.cuda())
                batch_labels = Variable(batch_labels.cuda())
            else:
                batch_images = Variable(batch_images)
                batch_labels = Variable(batch_labels)

            optimizer.zero_grad()
            output = net(batch_images)
            loss = criterion(output, batch_labels)
            loss = loss.requires_grad_()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred_label = output.max(1)
            num_correct = pred_label.eq(batch_labels).sum()
            train_acc += num_correct.item()

        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader.dataset)

        epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " % (epoch + 1, train_loss, train_acc))

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time
        print(epoch_str + time_str)

        scheduler.step()

        writer.add_scalars('Train', {'Accuracy': train_acc, 'Loss': train_loss}, epoch + 1)
        writer.add_scalars('Lr', {'Learning rate': scheduler.get_last_lr()[0]}, epoch + 1)

    torch.save({'ecpoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
                },
               os.path.join(CHECKPOINT_PATH, save_model_name))
    writer.close()


if __name__ == '__main__':
    data_root_path = '/data/zengnanrong/Architectural_Heritage_Elements_Dataset/'
    assert os.path.exists(data_root_path), "{} path does not exist.".format(data_root_path)

    data_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root_path, "train"), transform=data_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=64,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=32)
    net = simplenet.generate_model(use_gpu=True, gpu_id=['1'], in_channels=3, num_classes=10)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-4, weight_decay=1e-3)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.CrossEntropyLoss()

    train(net, True, train_loader, 300, optimizer, scheduler, criterion, 'simplenet.pkl')
