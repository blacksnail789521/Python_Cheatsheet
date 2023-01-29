import argparse
from pprint import pprint


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="argparse_example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show default value
    )

    """ We don't really need positional arguments' """
    # # Positional arguments (must given)
    # parser.add_argument('x', type=int, help='the base')
    # parser.add_argument('y', type=int, help='the exponent', choices = [0, 1, 2], default = 2)

    # Optional arguments
    parser.add_argument(
        "-b", "--bias", type=int, help="the bias"
    )  # metavar = 'BIAS', default = None
    parser.add_argument("--batch_size", type=int, help="the batch_size", metavar="N")
    parser.add_argument(
        "--enable_bn", type=bool, help="enable batch normalization", default=True
    )
    parser.add_argument("--model_name", type=str, help="model name", default="DNN")
    parser.add_argument(
        "-l1",
        "--list1",
        nargs="+",
        type=int,  # list of int
        help="set all elements in a list",
        default=[0],
    )  # python argparse_example.py -l1 1 2 3 4
    parser.add_argument(
        "-l2",
        "--list2",
        action="append",
        type=str,  # list of str
        help="add element one by one",
        default=["a"],
    )  # python argparse_example.py -l2 b -l2 c -l2 d
    parser.add_argument("--foo", help="foo help")

    # Exclusive group
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v", "--verbose", action="store_true", help="increase output verbosity"
    )
    group.add_argument(
        "-q", "--quiet", action="store_true"
    )  # action --> act as a flag (Don't use 'metavar')

    """----------------------------------------------------------------------"""

    args = parser.parse_known_args(args=[])[0]  # Allow unrecognized arguments

    return args


if __name__ == "__main__":

    # Assume we use `python argparse_example.py 3 2`
    args = parse_args()
    pprint(vars(args))  # Treat the Namespace as a dictionary

    """
    {'batch_size': None,
     'bias': None,
     'enable_bn': True,
     'list1': [0],
     'list2': ['a'],
     'model_name': 'DNN',
     'quiet': False,
     'verbose': False}
    """

    # Use `python argparse_example.py -h` to get the help message.
    """
    usage: argparse_example.py [-h] [-b BIAS] [--batch_size N]
                               [--enable_bn ENABLE_BN]
                               [--model_name MODEL_NAME]
                               [-l1 LIST1 [LIST1 ...]] [-l2 LIST2]
                               [-v | -q]
    
    argparse_example
    
    optional arguments:
      -h, --help            show this help message and exit
      -b BIAS, --bias BIAS  the bias (default: None)
      --batch_size N        the batch_size (not use in this case)
                            (default: None)
      --enable_bn ENABLE_BN
                            enable batch normalization (default:
                            True)
      --model_name MODEL_NAME
                            model name (default: DNN)
      -l1 LIST1 [LIST1 ...], --list1 LIST1 [LIST1 ...]
                            set all elements in a list (default:
                            [0])
      -l2 LIST2, --list2 LIST2
                            add element one by one (default: ['a'])
      -v, --verbose         increase output verbosity (default:
                            False)
      -q, --quiet
    """
