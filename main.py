import torch
import argparse
import time
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from dataset import MyDataset, collate_fn
from model import MLP, resnet50x1, resnet50x2, resnet50x4

parser = argparse.ArgumentParser()
parser.add_argument('--encoder',type=str,default='resnet50x1',help='type of encoder: resnet50x1, resnet50x2 or resnet50x4')
parser.add_argument('--epochs',type=int, default=100, help='train epochs')
parser.add_argument('--batch_size',type=int, default=8, help='batch size')
parser.add_argument('--num_hid',type=int, default=1500, help='num of downstream MLP hidden nodes')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda',default=False,action='store_true',help='use cuda')

args = parser.parse_args()


def train(encoder, DSmlp, train_loader, optimizer, log_file, device):
    for batch_id, (data,target,target_lengths) in enumerate(train_loader):
        input = data.to(device)
        feature = encoder(input.float())
        optimizer.zero_grad()
        output = DSmlp(feature)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_id%20 == 0:
            line = f"loss = {loss.item()}       [{batch_id*len(data)}/{len(train_loader.dataset)}]"
            print(line)
            log_file.write(line + '\n')



# choose CPU or CUDA
if args.cuda:
    device = 'cuda'
else:
    device = 'cpu'

dataset = MyDataset()
train_loader = torch.utils.data.DataLoader(dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 sampler=None, 
                                 batch_sampler=None, 
                                 collate_fn=collate_fn)

if args.encoder == 'resnet50x1':
    encoder = resnet50x1().to(device)
    checkpoint = torch.load("resnet50-1x.pth")
elif args.encoder == 'resnet50x2':
    encoder = resnet50x2().to(device)
    checkpoint = torch.load("resnet50-2x.pth")
elif args.encoder == 'resnet50x4':
    encoder = resnet50x4().to(device)
    checkpoint = torch.load("resnet50-4x.pth")
else:
    print(f"Unknown model: {args.encoder}")
    exit()

encoder.load_state_dict(checkpoint['state_dict'])

current_time = time.localtime(time.time())
time_name = f"{current_time.tm_year}-{current_time.tm_mon}-{current_time.tm_mday}-{current_time.tm_hour}:{current_time.tm_min}:{current_time.tm_sec}"
log_file = open('training_log/' + time_name + '.txt', 'w')
log_file.write(f"encoder: {args.encoder}\nepochs: {args.epochs}\nbatch_size: {args.batch_size}\nnum_hid: {args.num_hid}\nlr: {args.lr}\n")
encoder.eval()
dsMLP = MLP(args.num_hid).to(device)
optimizer = optim.AdamW(dsMLP.parameters(), lr=args.lr)
epoch = 0
while epoch < args.epochs:
    epoch = epoch + 1
    print(f"epoch: {epoch}")
    log_file.write(f"epoch: {epoch}\n")
    train(encoder, dsMLP, train_loader, optimizer, log_file, device)


log_file.close()
torch.save(dsMLP.state_dict(), 'training_model/' + time_name + '.pth')