from train import *
from inference import *

if __name__=='__main__':
    # wandb initializing
    wandb.init(project='stage3', entity='doooom')
    
    # get config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True) # ex) original_config
    ipts = parser.parse_args()

    # get args in config file
    args = EasyDict()
    with open(f'./config/{ipts.config_name}.json', 'r') as f:
        args.update(json.load(f))

    # save hyperparameters in wandb
    wandb.config.config_name = ipts.config_name
    wandb.config.update(args)

    # training
    train(args)

    # inference
    inference(args)

    # submit
    print('\n* Start submission...')
    submit(f'./submission/{args.save_file_name}.csv', f'{args.save_file_name} / {ipts.config_name}')
    print('* End submission.')