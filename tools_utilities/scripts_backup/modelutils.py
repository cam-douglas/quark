import torch
import torch.nn as nn


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    """find all Linear and Conv2d layers in module recursively and

    return a {name -> module} dictionary
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res
