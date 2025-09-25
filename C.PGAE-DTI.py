import time
import argparse
from tqdm import tqdm

import scipy.sparse as sp
from torch_geometric.data import Data
import numpy as np
import torch
import torch_geometric.transforms as T

from maskgae.utils import Logger, set_seed, tab_printer
from maskgae.model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder, DotEdgeDecoder
from maskgae.mask import MaskEdge, MaskPath


def train_linkpred(model, splits, args, fold, device):
    global z

    def train(data):
        model.train()
        loss_total, edge_loss, degree_loss = model.train_epoch(data.to(device), optimizer,
                                                                  alpha=args.alpha, batch_size=args.batch_size)
        return loss_total, edge_loss, degree_loss

    print('Start Training...  {}-fold'.format(fold))

    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    # train......
    for epoch in tqdm(range(1, 1 + args.epochs), desc="Training Epochs"):
        t1 = time.time()
        loss_total, edge_loss, degree_loss = train(splits['train'])  
        t2 = time.time()

    # save embedding Z
    z = model.encoder.get_embedding(splits['train'].x, splits['train'].edge_index, mode="last", l2_normalize=True)
    embedding = z.data.cpu().numpy()
    np.savetxt('../Result/embedding' + str(fold) + '_' + str(args.layer) + '.txt', embedding)


def main():
    parser = argparse.ArgumentParser()   
    parser.add_argument("--mask", nargs="?", default="None",
                        help="Masking stractegy, `Path`, `Edge` or `None`")
    parser.add_argument('--seed', type=int, default=2023,
                        help='Random seed for model and dataset. (default: 2023)')
    parser.add_argument('--bn', action='store_true',
                        help='Whether to use batch normalization for GNN encoder')
    parser.add_argument("--layer", nargs="?", default="gin",
                        help="GNN encoder")
    parser.add_argument("--encoder_activation", nargs="?", default="elu",
                        help="Activation function for GNN encoder (choice:1.relu.2.elu)")
    parser.add_argument('--encoder_channels', type=int, default=128,
                        help='Channels of GNN encoder. (default: 128)')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Channels of hidden representation. (default: 64)')
    parser.add_argument('--decoder_channels', type=int, default=64,
                        help='Channels of decoder. (default: 64)')
    parser.add_argument('--encoder_layers', type=int, default=2,
                        help='Number of layers of encoder. (default: 2)')
    parser.add_argument('--decoder_layers', type=int, default=2,
                        help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.4,
                        help='Dropout probability of encoder. (default: 0.4)')
    parser.add_argument('--decoder_dropout', type=float, default=0.3,
                        help='Dropout probability of decoder. (default: 0.3)')
    parser.add_argument('--alpha', type=float, default=0.003,
                        help='loss weight for degree prediction. (default: 2e-3)')

    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate for training. (default: 1e-2)')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='weight_decay for training. (default: 5e-5)')
    parser.add_argument('--grad_norm', type=float, default=1.0,
                        help='grad_norm for training. (default: 1.0.)')
    parser.add_argument('--batch_size', type=int, default=2 ** 16,
                        help='Number of batch size. (default: 2**16)')

    # parser.add_argument("--start", nargs="?", default="edge",
    #                     help="Which Type to sample starting nodes for random walks, (default: edge)")
    parser.add_argument('--p', type=float, default=0,
                        help='Mask ratio or sample ratio for MaskEdge/MaskPath')
    parser.add_argument("--save_path", nargs="?", default="model_linkpred",
                        help="save path for model. (default: model_linkpred)")
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs. (default: 1000)')
    parser.add_argument("--device", type=int, default=0,
                        help="<0 is cpu")
    parser.add_argument('--f', dest="folds", type=int, default=5,
                        help="Cross validation folds")

    try:
        args = parser.parse_args()
        # print(tab_printer(args))
    except:
        parser.print_help()
        exit(0)

    if not args.save_path.endswith('.pth'):
        args.save_path += '.pth'

    if args.device < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.ToUndirected(),
        T.ToDevice(device),
    ])

    # Prepare data......
    folds = args.folds
    for fold in range(folds):
        set_seed(args.seed)
        print("prepare: ", fold, "-fold data!!!")
        A = np.loadtxt('../whole_data/divide_result/A' + str(fold) + '.txt')
        edge_index = sp.coo_matrix(A)
        edge_index = np.vstack((edge_index.row, edge_index.col))
        edge_index = torch.LongTensor(edge_index).to(device)
        x = np.loadtxt('../whole_data/divide_result/X.txt')
        x = torch.tensor(x)
        x = x.float()
        data = Data(x=x, edge_index=edge_index)

        splits = dict(train=data)

        if args.mask == 'Path':
            mask = MaskPath(p=args.p, num_nodes=data.num_nodes,
                            start=args.start,
                            walk_length=args.encoder_layers + 1)
        elif args.mask == 'Edge':
            mask = MaskEdge(p=args.p)
        else:
            mask = None

        encoder = GNNEncoder(data.num_features, args.encoder_channels, args.hidden_channels,
                             num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                             bn=args.bn, layer=args.layer, activation=args.encoder_activation)

        if args.decoder_layers == 0:
            edge_decoder = DotEdgeDecoder()
        else:
            edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                                       num_layers=args.decoder_layers, dropout=args.decoder_dropout)

        degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                                       num_layers=args.decoder_layers, dropout=args.decoder_dropout)

        model = MaskGAE(encoder, edge_decoder, degree_decoder, mask).to(device)
        train_linkpred(model, splits, args, fold, device=device)


if __name__ == "__main__":
    main()