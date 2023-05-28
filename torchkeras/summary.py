import os,sys
import datetime
import numpy as np 
import pandas as pd 
import torch
from collections import OrderedDict

layer_modules = (torch.nn.MultiheadAttention, )


def summary(model,input_data = None,input_data_args = None,input_shape=None,input_dtype = torch.FloatTensor, batch_size=-1, 
            *args, **kwargs):
    """
    give example input data as least one way like below:
    ① input_data ---> model.forward(input_data) 
    ② input_data_args ---> model.forward(*input_data_args)
    ③ input_shape & input_dtype ---> model.forward(*[torch.rand(2, *size).type(input_dtype) for size in input_shape])
    """
    
    hooks = []
    summary = OrderedDict()
    
    def register_hook(module):
        def hook(module, inputs, outputs):
            
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            key = "%s-%i" % (class_name, module_idx + 1)

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                try:
                    info["out"] = [batch_size] + list(outputs[0].size())[1:]
                except AttributeError:
                    # pack_padded_seq and pad_packed_seq store feature into data attribute
                    info["out"] = [batch_size] + list(outputs[0].data.size())[1:]
            else:
                info["out"] = [batch_size] + list(outputs.size())[1:]

            info["params_nt"], info["params"] = 0, 0 
            for name, param in module.named_parameters():
                info["params"] += param.nelement() * param.requires_grad
                info["params_nt"] += param.nelement() * (not param.requires_grad)

            summary[key] = info

        # ignore Sequential and ModuleList and other containers
        if isinstance(module, layer_modules) or not module._modules:
            hooks.append(module.register_forward_hook(hook))
        

    model.apply(register_hook)
    
    # multiple inputs to the network
    if isinstance(input_shape, tuple):
        input_shape = [input_shape]
        
    if input_data is not None:
        x = [input_data]
    elif input_shape is not None:
        # batch_size of 2 for batchnorm
        x = [torch.rand(2, *size).type(input_dtype) for size in input_shape]
    elif input_data_args is not None:
        x = input_data_args
    else:
        x = []
    try:
        with torch.no_grad():
            model(*x) if not (kwargs or args) else model(*x, *args, **kwargs)
    except Exception:
        # This can be usefull for debugging
        print("Failed to run summary...")
        raise
    finally:
        for hook in hooks:
            hook.remove()
    summary_logs = []     
    summary_logs.append("--------------------------------------------------------------------------")
    line_new = "{:<30}  {:>20} {:>20}".format("Layer (type)", "Output Shape", "Param #")
    summary_logs.append(line_new)
    summary_logs.append("==========================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # layer, output_shape, params
        line_new = "{:<30}  {:>20} {:>20}".format(
            layer,
            str(summary[layer]["out"]),
            "{0:,}".format(summary[layer]["params"]+summary[layer]["params_nt"])
        )
        total_params += (summary[layer]["params"]+summary[layer]["params_nt"])
        total_output += np.prod(summary[layer]["out"])
        trainable_params += summary[layer]["params"]
        summary_logs.append(line_new)

    # assume 4 bytes/number
    if input_data is not None:
        total_input_size = abs(sys.getsizeof(input_data)/(1024 ** 2.))
    elif input_shape is not None:
        total_input_size = abs(np.prod(input_shape) * batch_size * 4. / (1024 ** 2.))
    else:
        total_input_size = 0.0
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_logs.append("==========================================================================")
    summary_logs.append("Total params: {0:,}".format(total_params))
    summary_logs.append("Trainable params: {0:,}".format(trainable_params))
    summary_logs.append("Non-trainable params: {0:,}".format(total_params - trainable_params))
    summary_logs.append("--------------------------------------------------------------------------")
    summary_logs.append("Input size (MB): %0.6f" % total_input_size)
    summary_logs.append("Forward/backward pass size (MB): %0.6f" % total_output_size)
    summary_logs.append("Params size (MB): %0.6f" % total_params_size)
    summary_logs.append("Estimated Total Size (MB): %0.6f" % total_size)
    summary_logs.append("--------------------------------------------------------------------------")
    
    summary_info = "\n".join(summary_logs)
    
    print(summary_info)
    return summary_info


def flop_summary(model,input_data):
    from fvcore import nn as fnn 
    print(fnn.flop_count_table(fnn.FlopCountAnalysis(model,input_data)))
    
    

