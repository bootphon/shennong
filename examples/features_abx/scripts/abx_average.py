#!/usr/bin/env python
"""Average an ABX score file to get a collapsed error rate in [0, 100]

Implemented from
https://github.com/bootphon/ABXpy/blob/zerospeech2015/bin/english_eval1.py#L169

"""

import argparse
import ast
import numpy as np
import os
import pandas


def average(df, task_type):
    if task_type == 'across':
        df['context'] = df['by']
    elif task_type == 'within':
        arr = np.array(map(ast.literal_eval, df['by']))
        df['talker']  = [e for e, f in arr]
        df['context'] = [f for e, f in arr]
    else:
        raise ValueError('Unknown task type: {0}'.format(task_type))

    del df['by']

    # aggregate on talkers
    groups = df.groupby(['context', 'phone_1', 'phone_2'], as_index=False)
    df = groups['score'].mean()
    # aggregate on contexts
    groups = df.groupby(['phone_1', 'phone_2'], as_index=False)
    df = groups['score'].mean()

    return (1 - df.mean()[0]) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    parser.add_argument('task_type', choices=['within', 'across'])
    args = parser.parse_args()

    avg = average(pandas.read_csv(args.csv_file, sep='\t'), task_type=args.task_type)
    print('{:.4f}'.format(avg))


if __name__ == '__main__':
    main()
