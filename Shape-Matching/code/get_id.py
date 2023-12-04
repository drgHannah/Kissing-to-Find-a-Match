import argparse



def get_train_type():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_type', help='Select the traintype: Marin etal (0), Ours (1), Ours Stochastic (2)', type=int, default=0, choices=[0,1,2])
    parser.add_argument('--lr',help='learning rate', type=float, default=0.001)
    parser.add_argument('--pretrain',help='Load pretrained network', type=float, default=0)
    parser.add_argument('--name',help='Select name for network. Use the same for train_basis.py and continue.py', default="base")
    parser.add_argument('--npoint',help='Number of points per object', type=int, default=1000)
    parser.add_argument('--number_extra_entries', help='For stochastic training, this number defines the amount of extra entries k are used for training in each row of the permutation matrix', type=int, default=100)
    parser.add_argument('--epoch', help='Epochs for first netowork, it just works when pretrain is not used', type=int, default=1600)
    parser.add_argument('--dataset', help='dataset for train, default [surr12k]', type=str, default='surr12k', choices=['surr12k', 'tosca'])

    args = parser.parse_args()
    return args
