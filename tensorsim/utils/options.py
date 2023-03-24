import configargparse
import yaml

def parses_ul(config_path):
    parser = configargparse.ArgParser(
        default_config_files=[str(config_path)],
        description = "Hyper Parameters",
        config_file_parser_class = configargparse.YAMLConfigFileParser)
    
    parser.add_argument("--config", is_config_file=True, default=False, help='config file path')
    
    #set arguments that we use later. These are pretty self explanatory 
    parser.add_argument("--max_epochs", default=20, type=int) 
    parser.add_argument("--resume_from_checkpoint", type=str, help="resume from checkpoint")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--gpus", type=int, default=0, help="how many gpus")
    parser.add_argument("--use_16bit", type=bool, default=False, help="use 16 bit precision")
    parser.add_argument("--val_check_interval", type=float, default=1, help="how often within one training epoch to check the validation set")
    parser.add_argument("--profiler", action="store_true", help="use profiler")
    parser.add_argument("--test_args", action="store_true", help="print args")
    parser.add_argument("--model_name", default="fpn", type=str)
    parser.add_argument("--load_name", default="list", type=str)
    
    #arguments for the dataset
    parser.add_argument("--data_root", type=str, required=True, help="path of dataset")
    parser.add_argument("--train_list", type=str, required=True, help="path of train dataset list")
    parser.add_argument("--val_list", type=str, required=True, help="path of val dataset list")
    parser.add_argument("--test_list", type=str, required=True, help="path of test dataset list")
    parser.add_argument("--train_dir", type=str, required=True, help="subdir path of train dataset")
    parser.add_argument("--val_dir", type=str, required=True, help="subdir of val dataset list")
    parser.add_argument("--test_dir", type=str, required=True, help="subdir of test dataset list")
    
    #arguments for the neural network
    parser.add_argument("--input_size", default=200, type=int)
    parser.add_argument("--mean_layout", default=0, type=float)
    parser.add_argument("--std_layout", default=1, type=float)
    parser.add_argument("--mean_heat", default=0, type=float)
    parser.add_argument("--std_heat", default=1, type=float)
    parser.add_argument("--max_iters", default=10000, type=int)
    
    #training parameters
    parser.add_argument("--lr", default="0.01", type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=2, type=int, help="num_workers in DataLoader")
    
    #parameters in the dataset
    parser.add_argument("--nx", type=int)
    parser.add_argument("--length", type=float)
    parser.add_argument("--u_D", type=float)
    parser.add_argument("--bcs", type=yaml.safe_load, action="append", help="Dirichlet boundaries", )
    
    #needed to run in jupyter notebook
    parser.add_argument('-f')
    
    args = parser.parse_args()
    return args