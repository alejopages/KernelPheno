from ..models.model import AlexNet

import os.path as osp

TEST_DATA = '/home/apages/pysrc/KernelPheno/data/datasets/gray_normed_unsegged/data'
OUTDIR = osp.join(osp.dirname(__file__), 'alexnet')

model = AlexNet()
model.train('m1.mdl', TEST_DATA, OUTDIR, 1, 1)

print("test completed")
