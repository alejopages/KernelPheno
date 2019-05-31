from ..models.model import AlexNet

import os.path as osp

TEST_DATA = ''
OUTDIR = osp.join(osp.dirname(__file__), 'alexnet')

model = AlexNet()
model.train('m1.mdl', TEST_DATA, OUTDIR, 1, 10)

print("test completed") 
