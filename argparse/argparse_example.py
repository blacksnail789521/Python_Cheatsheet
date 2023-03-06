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
    parser.add_argument("-b", "--bias", type=int, help="the bias")  # default = None
    parser.add_argument("--batch_size", type=int, help="the batch_size", default=128)
    # parser.add_argument("--use_gpu", type=bool, help="use gpu", default=False) # Don't use this
    parser.add_argument("--use_gpu", action="store_true", help="use gpu")
    parser.add_argument("--model_name", type=str, help="model name", default="DNN")
    parser.add_argument(
        "-l1",
        "--list1",
        nargs="+",
        type=int,  # list of int
        help="set all elements in a list",
        default=[1],  # It will be overwritten
    )  # python argparse_example.py --list1 1 2 3 4
    parser.add_argument(
        "-l2",
        "--list2",
        action="append",
        type=str,  # list of str
        help="add element one by one",
        default=["a"],  # It will not be overwritten
        choices=["a", "b", "c", "d"],
    )  # python argparse_example.py --list2 b --list2 c --list2 d

    """ We don't really need exclusive group """
    # # Exclusive group
    # group = parser.add_mutually_exclusive_group()
    # group.add_argument(
    #     "-v", "--verbose", action="store_true", help="increase output verbosity"
    # )
    # group.add_argument(
    #     "-q", "--quiet", action="store_true"
    # )

    """----------------------------------------------------------------------"""

    # Allow unrecognized arguments
    args, _ = parser.parse_known_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    pprint(vars(args))  # Treat the Namespace as a dictionary

    """
    {'batch_size': 128,
    'bias': None,
    'list1': [1],
    'list2': ['a'],
    'model_name': 'DNN',
    'use_gpu': False}
    """

    # Use `python argparse_example.py -h` to get the help message.
    """
    usage: argparse_example.py [-h] [-b BIAS] [--batch_size BATCH_SIZE] [--enable_bn ENABLE_BN] [--use_gpu] [--model_name MODEL_NAME] [-l1 LIST1 [LIST1 ...]] [-l2 {a,b,c,d}]

    argparse_example

    options:
    -h, --help            show this help message and exit
    -b BIAS, --bias BIAS  the bias (default: None)
    --batch_size BATCH_SIZE
                            the batch_size (default: 128)
    --use_gpu             use gpu (default: False)
    --model_name MODEL_NAME
                            model name (default: DNN)
    -l1 LIST1 [LIST1 ...], --list1 LIST1 [LIST1 ...]
                            set all elements in a list (default: [1])
    -l2 {a,b,c,d}, --list2 {a,b,c,d}
                            add element one by one (default: ['a'])
    """
