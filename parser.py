import argparse

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train-length', type=int, default=0)
    parser.add_argument('--visual-prompt', type=str, choices=['none', 'deep', 'shallow'], default='none')
    parser.add_argument('--text-prompt', type=str, choices=['none', 'deep', 'shallow'], default='none')
    parser.add_argument('--classifier', type=str, choices=['none'], default='none')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--momenton', type=float, default=0.9)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=75)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('-o', '--output-dir', type=str, default='.')

    return parser