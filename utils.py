import random
import torch
import os
from datetime import datetime
import shutil
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def create_log_folder(base_folder, run_name, sys_argv):
    ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    log_dir = os.path.join(base_folder, "%s_%s" % (run_name, ts))
    os.makedirs(log_dir, exist_ok=True)

    cmd = ' '.join(sys_argv)
    with open("%s/cmd.txt" % log_dir, "w") as f:
        f.write(cmd)

    os.makedirs("%s/src" % log_dir, exist_ok=True)
    list_files = os.listdir("./")
    for fname in list_files:
        if fname[-3:] == ".py":
            shutil.copy("./%s" % fname, "%s/src/" % log_dir)

    return log_dir

