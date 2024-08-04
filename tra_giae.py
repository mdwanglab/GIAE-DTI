import argparse
from tqdm import tqdm
import scipy.sparse as sp
from torch_geometric.data import Data
import numpy as np
import torch
from utils.utils import set_seed
from utils.model import GAE, DegreeDecoder, EdgeDecoder, GNNEncoder, DotEdgeDecoder
from utils.mask import MaskEdge, MaskPath


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
        loss_total, edge_loss, degree_loss = train(splits['train'])

    # save embedding Z
    z = model.encoder.get_embedding(splits['train'].x, splits['train'].edge_index, mode="last", l2_normalize=True)
    embedding = z.data.cpu().numpy()
    np.savetxt('../Result/embedding' + str(fold) + '_' + str(args.layer) + '.txt', embedding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', dest="folds", type=int, default=5, help="Cross validation folds")
    parser.add_argument('--epochs', type=int, default=1000,help='Number of training epochs')
    parser.add_argument("--device", type=int, default=0, help="<0 is cpu")
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    parser.add_argument("--layer", nargs="?", default="gin", help="GNN encoder")
    parser.add_argument("--mask", nargs="?", default="None", help="Masking stractegy, Path、Edge or None")
    parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization')
    parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function: relu or elu")
    parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder')
    parser.add_argument('--hidden_channels', type=int, default=64, help='hidden embedding')
    parser.add_argument('--decoder_channels', type=int, default=64, help='Channels of decoder')
    parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers of encoder')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders')
    parser.add_argument('--encoder_dropout', type=float, default=0.4, help='encoder dropout')
    parser.add_argument('--decoder_dropout', type=float, default=0.3, help='Dropout probability of decoder')
    parser.add_argument('--alpha', type=float, default=0.003, help='loss weight for degree prediction')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay')
    parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training')
    parser.add_argument('--batch_size', type=int, default=2 ** 16, help='Number of batch size')
    parser.add_argument('--p', type=float, default=0, help='Mask ratio or sample ratio for MaskEdge/MaskPath')
    parser.add_argument("--save_path", nargs="?", default="model_linkpred")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if not args.save_path.endswith('.pth'):
        args.save_path += '.pth'

    if args.device < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # Prepare data......
    folds = args.folds
    for fold in range(folds):
        set_seed(args.seed)
        print("prepare: ", fold, "-fold data!!!")
        A = np.loadtxt('../network data/A' + str(fold) + '.txt')
        edge_index = sp.coo_matrix(A)
        edge_index = np.vstack((edge_index.row, edge_index.col))
        edge_index = torch.LongTensor(edge_index).to(device)
        x = np.loadtxt('../network data/X.txt')
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

        model = GAE(encoder, edge_decoder, degree_decoder, mask).to(device)
        train_linkpred(model, splits, args, fold, device=device)