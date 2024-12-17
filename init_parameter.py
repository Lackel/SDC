from argparse import ArgumentParser

def init_model():
    parser = ArgumentParser()
    
    parser.add_argument("--dataset", default='banking', type=str, help="The name of the dataset to train selected.")

    parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes.")

    parser.add_argument("--cluster_num_factor", default=1.0, type=float, help="The factor (magnification) of the number of clusters K.")

    parser.add_argument("--data_dir", default='data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--save_results_path", type=str, default='outputs', help="The path to save results.")

    parser.add_argument("--pretrain_dir", default='pretrain_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--train_dir", default='train_models', type=str,
                        help="The output directory where the final model is stored in.")

    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="The path or name for the pre-trained bert model.")

    parser.add_argument("--tokenizer", default="bert-base-uncased", type=str,
                        help="The path or name for the tokenizer")

    parser.add_argument("--max_seq_length", default=None, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")

    parser.add_argument("--warmup_proportion", default=0.1, type=float)

    parser.add_argument("--freeze_bert_parameters", action="store_true", help="Freeze the last parameters of BERT.")

    parser.add_argument("--save_model", default=False, type=str, help="Save trained model.")

    parser.add_argument("--save_results", default=False, type=str, help="Save trained model.")

    parser.add_argument("--pretrain", action="store_true", help="Pre-train the model with labeled data.")

    parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization.")

    parser.add_argument("--rtr_prob", default=0.25, type=float,
                        help="Probability for random token replacement")

    parser.add_argument("--labeled_ratio", default=0.1, type=float,
                        help="The ratio of labeled samples in the training set.")

    parser.add_argument("--gpu_id", type=str, default='3', help="Select the GPU id.")

    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Batch size for training.")
    
    parser.add_argument("--pretrain_batch_size", default=128, type=int,
                        help="Batch size for training.")

    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Batch size for evaluation.")

    parser.add_argument("--pre_wait_patient", default=20, type=int,
                        help="Patient steps for pre-training Early Stop.")

    parser.add_argument("--num_pretrain_epochs", default=100, type=float,
                        help="The pre-training epochs.")

    parser.add_argument("--num_train_epochs", default=80, type=float,
                        help="The training epochs.")

    parser.add_argument("--lr_pre", default=5e-5, type=float,
                        help="The learning rate for pre-training.")

    parser.add_argument("--lr", default=5e-5, type=float,
                        help="The learning rate for training.")

    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")

    parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")

    parser.add_argument("--imb-factor", default=1, type=float, help="imbalance factor of the data, default 1")
    
    
    return parser
