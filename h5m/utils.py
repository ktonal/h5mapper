

def depth_first_apply(dict_like, func):
    out = {}
    for k, v in dict_like.items():
        if type(v) is type(dict_like):
            out[k] = depth_first_apply(v, func)
        else:
            out[k] = func(v)
    return out