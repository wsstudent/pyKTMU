import argparse
from wandb_train import main, unlearning_arg_parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[unlearning_arg_parser])
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="gkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    # model params
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--graph_type", type=str, default='transition',help='dense or transition')

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    args = parser.parse_args()
    args.emb_size = args.hidden_dim
    params = vars(args)
    main(params)
