import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # in/out

    parser.add_argument('--outf', default='./experiments',
                        help='trained model will be saved at here')
    parser.add_argument('--name', default='chasedb1',
                        help='trained model will be saved at here')
    parser.add_argument('--model_path', default='./best_model.pth',
                        help='Checkpoint path')

    parser.add_argument('--save', default='best',
                        help='save name of experiment in args.outf directory')
    parser.add_argument('--dataset',default='CHASEDB1',
                        help='Dataset type: DRIVE, CHASEDB1, HRF, ROSE')
    # data
    parser.add_argument('--train_data_path_list',
                        default='./prepare_dataset/data_path_list/CHASEDB1/train.txt')
    parser.add_argument('--test_data_path_list',
                        default='./prepare_dataset/data_path_list/CHASEDB1/test.txt')
    parser.add_argument('--train_patch_height', default=64)
    parser.add_argument('--train_patch_width', default=64)
    parser.add_argument('--N_patches', default=150000,
                        help='Number of training image patches')
    parser.add_argument('--inside_FOV', default='center',
                        help='Choose from [not,center,all]')
    parser.add_argument('--val_ratio', default=0.01,
                        help='The ratio of the validation set in the training set')
    parser.add_argument('--sample_visualization', default=False,
                        help='Visualization of training samples')
    # model parameters
    parser.add_argument('--in_channels', default=1,type=int,
                        help='input channels of model')
    parser.add_argument('--classes', default=2,type=int, 
                        help='output channels of model')

    # training
    parser.add_argument('--N_epochs', default=80, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--epoch_min', default=0)
    parser.add_argument('--epoch_max', default=0)
    parser.add_argument('--alpha', default=2)

    parser.add_argument('--batch_size', default=32,  #64
                        type=int, help='batch size')
    parser.add_argument('--early-stop', default=15, type=int,
                        help='early stopping')
    parser.add_argument('--lr', default=0.0008, type=float,
                        help='initial learning rate')
    parser.add_argument('--val_on_test', default=False, type=bool,
                        help='Validation on testset')

    # for pre_trained checkpoint
    parser.add_argument('--start_epoch', default=1, 
                        help='Start epoch')
    parser.add_argument('--pre_trained', default=None,
                        help='(path of trained _model)load trained model to continue train')

    # testing
    parser.add_argument('--test_patch_height', default=96)
    parser.add_argument('--test_patch_width', default=96)
    parser.add_argument('--stride_height', default=16)
    parser.add_argument('--stride_width', default=16)

    # hardware setting
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use GPU calculating')

    args = parser.parse_args()

    return args
