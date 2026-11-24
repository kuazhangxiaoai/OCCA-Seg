from torch.utils.data import DataLoader
from network.losses import *
from network.datasets.potsdam_dataset import *
from network.models.occaseg import OCCASeg
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 45
ignore_index = len(CLASSES)
train_batch_size = 4
val_batch_size = 4
lr = 1e-4
weight_decay = 0.01
backbone_lr = 1e-3
backbone_weight_decay = 1e-2
num_classes = len(CLASSES)
token_length=num_classes
classes = CLASSES

test_time_aug = 'd4'
output_mask_dir, output_mask_rgb_dir = None, None
weights_name = "occaseg"
weights_path = "checkpoints/potsdam/{}".format(weights_name)
test_weights_name = weights_name
log_name = 'potsdam/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = [0]  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None #"checkpoints/potsdam/delta-0817l0.8lr/delta-0817l0.8lr.ckpt"  # whether continue training with the checkpoint, default None
strategy = None

#  define the network
net = OCCASeg(
    model="convnext_base",
    nc=num_classes,
    ns=4,
    ch=(128, 256, 512, 1024)
)
#net = OCCASeg(
#    model="convnext_large",
#    nc = num_classes,
#    ns = 4,
#    ch = (192, 384, 768, 1536)
#)

# define the loss
loss = DenseLoss(ignore_index=ignore_index)
use_aux_loss = True
has_contrastive_loss = False

# define the dataloader

train_dataset = PotsdamDataset(data_root='/home/yanggang/workspace/D2LS/data/Potsdam1024/train', mode='train',
                               mosaic_ratio=0.25, transform=train_aug)

val_dataset = PotsdamDataset(data_root='/home/yanggang/workspace/D2LS/data/Potsdam1024/val', transform=val_aug)
test_dataset = PotsdamDataset(data_root='/home/yanggang/workspace/D2LS/data/Potsdam1024/val',
                              transform=test_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
base_optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

