import torch

# Filter out batch norm parameters and remove them from weight decay - gets us higher accuracy 93.2 -> 93.48
# https://arxiv.org/pdf/1807.11205.pdf
def bnwd_optim_params(model, model_params, master_params):
    bn_params, remaining_params = split_bn_params(model, model_params, master_params)
    return [{'params':bn_params,'weight_decay':0}, {'params':remaining_params}]


def split_bn_params(model, model_params, master_params):
    def get_bn_params(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): return module.parameters()
        accum = set()
        for child in module.children(): [accum.add(p) for p in get_bn_params(child)]
        return accum
    
    mod_bn_params = get_bn_params(model)
    zipped_params = list(zip(model_params, master_params))

    mas_bn_params = [p_mast for p_mod,p_mast in zipped_params if p_mod in mod_bn_params]
    mas_rem_params = [p_mast for p_mod,p_mast in zipped_params if p_mod not in mod_bn_params]
    return mas_bn_params, mas_rem_params
    