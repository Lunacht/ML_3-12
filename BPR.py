import os
from recq.recommenders.bpr import BPR
from recq.utils import Dataset
from recq.tools.parser import parse_args
from recq.tools.io import print_seperate_line

args = parse_args("bpr")
print_seperate_line()
for key, value in vars(args).items():
    print(key + "=" + str(value))
print_seperate_line()

curr_dir = os.path.dirname(__file__)

data_dir = os.path.join(curr_dir, "data", args.dataset)
model_dir = os.path.join(curr_dir, "output", "model")

dataset = Dataset(args, data_dir)
model = BPR(args, dataset)
model.fit(args, model_dir)
