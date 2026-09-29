import argparse
import copy
import datetime
import os
import random

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from imageDatastore import imageDatastore
from model_UNET_EDSR import UNET, UNET_EDSR
from utils import (
    Timer,
    adjust_learning_rate,
    compute_psnr,
    compute_ssim,
    load_checkpoint,
    print_network,
    save_checkpoint,
    setup_seed,
    tensor2np,
    time_text,
)


path = r"C:\Users\onue687i\Documents\AirDent\TW Source Material\720"

parser = argparse.ArgumentParser(description="net")
# Training
parser.add_argument("--num_epochs", default=50, type=int)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--val_batch_size", default=4, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--lr_DropFactor",  default=0.5, type=float)
parser.add_argument("--lr_DropPeriod", default=10, type=int)
parser.add_argument("--save_everyEpoch", default=1, type=int)
parser.add_argument("--save_path", default="./folder/", type=str)
parser.add_argument("--valid_everyEpoch", default=1, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--num_GPUs", default=1, type=int)

# Dataset
parser.add_argument("--dir_ZTrain", default=os.path.join(path, "HR_Train"), type=str)
parser.add_argument("--dir_XTrain", default=os.path.join(path, "sim_MCF_Train"), type=str)
parser.add_argument("--dir_ZValid", default=os.path.join(path, "HR_Valid"), type=str)
parser.add_argument("--dir_XValid", default=os.path.join(path, "sim_MCF_Valid"), type=str)
parser.add_argument("--n_colors", default=1, type=int, help="number of color channels to use")

# Pre-train
parser.add_argument("--pretrain", action="store_true", help="load pre-trained model")
parser.add_argument("--dir_pretrain", default="", type=str, help="path1 to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number")

parser.add_argument("--save_model_arch",action="store_true", help='save model architecture')
parser.add_argument("--rgb_range", default=1, type=int, help="maxium value of RGB")
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--ext", default=".png", type=str)
args = parser.parse_args()


# Needed on Windows: each DataLoader worker re-runs this file from the top.
# The guard makes the workers skip everything below, so only the main program trains.
if __name__ == "__main__":

    setup_seed(args.seed)

    pespective = False

    # Create the checkpoint folder if it does not exist yet
    os.makedirs(args.save_path, exist_ok=True)

    # Dataset
    dsTrain = imageDatastore(
        args.dir_ZTrain,
        args.dir_XTrain,
        n_colors=args.n_colors,
        rgb_range=args.rgb_range,
        ext=args.ext
    )

    dsValid = imageDatastore(
        args.dir_ZValid,
        args.dir_XValid,
        n_colors=args.n_colors,
        rgb_range=args.rgb_range,
        ext=args.ext
    )


    train_loader = DataLoader(
        dataset=dsTrain,
        num_workers=args.num_workers,
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=True,
        drop_last=True, 
        persistent_workers=False
    )

    valid_loader = DataLoader(
        dataset=dsValid,
        num_workers=args.num_workers,
        batch_size=args.val_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )


    if args.num_GPUs > 0:
        main_device = 0
        device = f"cuda:{main_device}"  
        device_ids = list(range(args.num_GPUs)) if args.num_GPUs > 1 else None
    else:
        device = "cpu"


    model = UNET(in_channels=args.n_colors, out_channels=args.n_colors, inplace=False)

    model_arch = copy.deepcopy(model)  
    model = model.to(device)     

    # Load checkpoint
    if args.pretrain:
        # weights_only=False: the file also stores 'args'. Only load checkpoint files you trust.
        checkpoint = torch.load(args.dir_pretrain, map_location=device, weights_only=False)
        load_checkpoint(checkpoint, model)

    l1_criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.pretrain:
        optimizer.load_state_dict(checkpoint["optimizer"])


    # Data Parallel
    model = nn.DataParallel(model, device_ids=device_ids, output_device=main_device).to(device) if args.num_GPUs > 1 else model.to(device)     

    #%%
    def train(epoch):
        model.train()
        loss = 0.0
    
        adjust_learning_rate(optimizer, epoch-1, args.lr, args.lr_DropFactor, args.lr_DropPeriod)
    
        # Pespective: one-liner stage management
        if pespective:
            if scheduler.step(epoch): print(f">>> {scheduler.stage_name}")
        
        
        print("\n------- Epoch {}/{} \t\t lr = {}".format(epoch, args.num_epochs, optimizer.param_groups[0]['lr']))
    
        tqdm_iterator = tqdm(train_loader, desc="In training", leave=True, ncols=80) # 
    
        for ii, (dlX, dlZ) in enumerate(tqdm_iterator):
            dlX, dlZ = dlX.to(device), dlZ.float().to(device)

            dlY = model(dlX)
            loss_iter = l1_criterion(dlY, dlZ)
            
            if pespective: optimizer_align.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            loss_iter.backward()

            optimizer.step()
            if pespective: optimizer_align.step()
        
            # tqdm loop: postfix --> loss
            tqdm_iterator.set_postfix(loss=loss_iter.item())  
        
            loss += loss_iter.detach().cpu().item()
        
        loss /= len(train_loader)
        print("\nTrain loss: {:.4f}".format(loss))
        return loss

    def valid():
        model.eval()
        loss = 0.0
        peaksnr, ssimval = 0, 0
        count = 0

        for (dlVX, dlVZ) in valid_loader:
            dlVX = dlVX.to(device)
            dlVZ = dlVZ.to(device)

            with torch.no_grad():
                dlVY = model(dlVX)
                loss += l1_criterion(dlVY, dlVZ).item() * dlVZ.shape[0]

            # PSNR, SSIM
            batch_val = dlVZ.shape[0]
            count += batch_val
            for jj in range(batch_val):
                VY = tensor2np(dlVY.detach()[jj])
                VZ = tensor2np(dlVZ.detach()[jj])
            
                peaksnr += compute_psnr(VZ, VY)
                ssimval += compute_ssim(VY, VZ)
    
        loss /= count
        peaksnr /= count
        ssimval /= count
        print("Valid loss: {:.4f} \t\t PSNR: {:.4f} \t\t SSIM: {:.4f}".format(loss, peaksnr, ssimval))

        return loss


    print("===> Training")

    # Losses of epochs
    if args.pretrain:
        loss_stats = {k: v for k, v in checkpoint['loss_stats'].items()}
    else:
        loss_stats = {'train': [], 'valid': [], 'valid_expt': []}
    loss_stats.setdefault('valid_epoch', list(range(1, len(loss_stats['valid']) + 1)))

    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = args.start_epoch
    best_validLoss = min(loss_stats['valid']) if loss_stats['valid'] else float('inf')


    code_start = datetime.datetime.now()
    timer = Timer()

    for epoch in range(args.start_epoch, args.num_epochs+1):
        t_epochStart = timer.t()
    
        # Train
        loss_train = train(epoch)
        loss_stats['train'].append(loss_train)
    
        # valid
        do_valid = (epoch % args.valid_everyEpoch == 0) or (epoch == args.num_epochs)
    
        if do_valid:
            loss_valid = valid()
            loss_stats['valid'].append(loss_valid)
            loss_stats['valid_epoch'].append(epoch)

            # Save the model with the best valid
            if loss_valid < best_validLoss:
                best_validLoss = loss_valid
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch

                # Save the best model now, only when it gets better
                save_checkpoint(model, model_arch, optimizer, epoch, loss_stats, args, text='_best')
    
        # Save model
        if epoch % args.save_everyEpoch == 0 or epoch == args.num_epochs:
            save_checkpoint(model, model_arch, optimizer, epoch, loss_stats, args)
        
        
        # Timing
        t_epochEnd = timer.t()
        prog = (epoch - args.start_epoch + 1) / (args.num_epochs - args.start_epoch + 1)
        t_epoch = time_text(t_epochEnd - t_epochStart)
        t_elapsed, t_all = time_text(t_epochEnd), time_text(t_epochEnd / prog)
        print('Time cost: {}, {}/{}'.format(t_epoch, t_elapsed, t_all))


    code_end = datetime.datetime.now()
    print('Total cost times: %s' % str(code_end - code_start).split('.', 2)[0])


    # Plot TRAINING PROGRESS
    plt.figure('Training progress',figsize=(10,5))
    plt.title("Training progress")
    plt.plot(range(1, len(loss_stats['train']) + 1), loss_stats['train'], label="Train")
    plt.plot(loss_stats['valid_epoch'], loss_stats['valid'], label="Valid")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.show()


    torch.cuda.empty_cache()