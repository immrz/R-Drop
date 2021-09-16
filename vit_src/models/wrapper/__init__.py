from .rdrop import RDropWrapper
from .two_aug import TwoAugWrapper
from .rdrop_da import RDropDAWrapper
from torch.nn import Module


def get_wrapper(
    wrapper: str,
    model: Module,
    alpha: float = 1.0,
    consistency: str = None,
    consist_func: str = None,
    stop_grad: bool = False,
) -> Module:
    if wrapper is None or wrapper.lower() == "none":
        return model
    elif wrapper == "rdrop":
        return RDropWrapper(model=model,
                            consistency=consistency,
                            consist_func=consist_func,
                            alpha=alpha,
                            stop_grad=stop_grad)
    elif wrapper == "twoaug":
        return TwoAugWrapper(model=model,
                             consistency=consistency is not None,
                             alpha=alpha)
    elif wrapper == "rdropDA":
        return RDropDAWrapper(model=model,
                              consistency=consistency,
                              alpha=alpha)
    else:
        raise NotImplementedError
