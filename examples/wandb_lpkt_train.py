import argparse
from wandb_train import main, unlearning_arg_parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[unlearning_arg_parser])
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="lpkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    # model params
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--d_a", type=int, default=64)
    parser.add_argument("--d_e", type=int, default=64)
    parser.add_argument("--d_k", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.03)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    args = parser.parse_args()
    params = vars(args)
    main(params)
