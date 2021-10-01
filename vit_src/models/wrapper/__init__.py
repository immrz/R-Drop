from .rdrop import RDropWrapper
from .two_aug import TwoAugWrapper
from .rdrop_da import RDropDAWrapper, RDropDAMutualWrapper
from .semi_supv import SemiSupvWrapper
from torch.nn import Module


def get_wrapper(
    wrapper: str,
    model: Module,
    alpha: float = 1.0,
    beta: float = 0.5,
    consistency: str = None,
    consist_func: str = None,
    **kwargs,
) -> Module:
    if wrapper is None or wrapper.lower() == "none":
        return model
    elif wrapper == "rdrop":
        return RDropWrapper(model=model,
                            consistency=consistency,
                            consist_func=consist_func,
                            alpha=alpha)
    elif wrapper == "twoaug":
        return TwoAugWrapper(model=model,
                             consistency=consistency is not None,
                             alpha=alpha)
    elif wrapper == "rdropDA":
        if consist_func == "mutual":
            return RDropDAMutualWrapper(model=model, alpha=alpha)
        else:
            return RDropDAWrapper(model=model,
                                  alpha=alpha,
                                  beta=beta,
                                  model_cfunc=kwargs["model_cfunc"],
                                  data_cfunc=kwargs["data_cfunc"])
    elif wrapper == "uda":
        return SemiSupvWrapper(model=model,
                               alpha=alpha,
                               beta=beta,
                               rdrop=consistency is not None)
    else:
        raise NotImplementedError
