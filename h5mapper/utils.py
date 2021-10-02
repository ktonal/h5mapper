__all__ = [
    'depth_first_apply',
    'flatten_dict',
]


def depth_first_apply(dict_like, func):
    out = {}
    for k, v in dict_like.items():
        if type(v) is type(dict_like):
            out[k] = depth_first_apply(v, func)
        else:
            out[k] = func(v)
    return out


def flatten_dict(dd, separator='/', prefix=''):
    return {prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
            } if isinstance(dd, dict) else {prefix: dd}
