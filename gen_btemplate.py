#!/usr/bin/env python
import numpy as np
import argparse
import btemplate as bt

parser = argparse.ArgumentParser()
parser.add_argument('yaml_file' , default=None, type=str, help='file_yaml')
parser.add_argument('idx'       , default=None, type=int, help='idx')
args     = parser.parse_args()

btmp = bt.btemplate(args.yaml_file)
#btmp = bt.btemplate(args.yaml_file, bk_yaml="yaml/bk.yaml")
idx  = args.idx
print(idx)

# spec from 3g mask; no purification
auto, cross, auto_in = btmp.get_masked_spec(idx)
print("auto:", auto)
print("cross:", cross)

#clsall = btmp.get_pure_masked_spec(idx)

