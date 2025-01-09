import argparse


def parse():
    parser = argparse.ArgumentParser(description='Pass action with arguments')
    parser.add_argument("action", type=str, help="action")
    parser.add_argument('other_args', nargs='*', help='Other arguments')
    return parser.parse_args()
