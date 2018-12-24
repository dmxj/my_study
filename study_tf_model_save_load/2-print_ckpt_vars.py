# -*- coding: utf-8 -* -
'''
打印检查点中的变量
'''
from tensorflow.python.tools import inspect_checkpoint as chkp

ckpt_path = "./models/1_save_vars/model.ckpt"

# print all tensor in ckeckpoint file
chkp.print_tensors_in_checkpoint_file(ckpt_path, tensor_name="", all_tensors=True)

# print only tensor v1 in checkpoint file
chkp.print_tensors_in_checkpoint_file(ckpt_path, tensor_name="v1", all_tensors=False)

# print only tensor v2 in checkpoint file
chkp.print_tensors_in_checkpoint_file(ckpt_path, tensor_name="v2", all_tensors=False)

