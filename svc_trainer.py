import os
import sys
import matplotlib.pyplot as plt # Import matplotlib
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import argparse
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from vits_extend.train import train

torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
    args = parser.parse_args()

    hp = OmegaConf.load(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    assert hp.data.hop_length == 320, \
        'hp.data.hop_length must be equal to 320, got %d' % hp.data.hop_length

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []

    args.num_gpus = 0
    torch.manual_seed(hp.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.train.seed)
        args.num_gpus = torch.cuda.device_count()
        print('Batch size per GPU :', hp.train.batch_size)

        if args.num_gpus > 1:
            # Modify mp.spawn to return losses
            results = mp.spawn(train, nprocs=args.num_gpus,
                              args=(args, args.checkpoint_path, hp, hp_str,),
                              join=True)  # join=True to wait for processes to finish
            
            # Assuming train function returns train_losses, val_losses
            train_losses = results[0]  
            val_losses = results[1] 

        else:
            # Modify train to return losses
            train_losses, val_losses = train(0, args, args.checkpoint_path, hp, hp_str) 
    else:
        print('No GPU find!')

    # Plot the metrics after training is complete
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(f"{args.name}_loss_plot.png") # Save the plot to a file
    plt.show()