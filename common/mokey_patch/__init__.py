def apply_monkey_patch():
    from .torch_patch import _apply_monkey_patch4torch
    _apply_monkey_patch4torch()
