"""Injection utilities."""
from .policy import get_activate_policies


def find_match_policy(module, policies):
    """Iterate through all policies and find the first match."""
    for policy in policies:
        if policy.match(module):
            return policy
    return None


def inject_module(model, policy_clses=None):
    """Inject modules in the model."""
    policies = []
    for policy_cls in policy_clses if policy_clses is not None else get_activate_policies():
        policies.append(policy_cls())
    record = {}

    def _helper(model):
        """Traverse model's children recursively and apply any transformations in ``policies``."""
        for name, child in model.named_children():
            policy = find_match_policy(child, policies)
            if policy is not None:
                msg = f"Apply {policy.__class__.__name__} to {name}"
                if msg not in record:
                    record[msg] = 0
                record[msg] += 1
                new_child = policy.init_from_object(child)
                setattr(model, name, new_child)
            else:
                # If no match, continue to traverse the children.
                _helper(child)

    _helper(model)
    if record:
        for msg, count in record.items():
            print(f"{msg} for {count} times")
    else:
        print("No module replaced")


class InjectModuleContext:
    """Context manager to inject modules in the model.
    This is modified from DeepSpeed ZeRO-3 Init.
    """

    def __init__(self, policies=None):
        self.policies = []
        for policy_cls in policies if policies is not None else get_activate_policies():
            self.policies.append(policy_cls())
        self.record = {}
        self.load_state_dict_backup = None

    def __enter__(self):
        for policy in self.policies:
            policy.hook(self)
        self.inject_load_state_dict()

    def inject_load_state_dict(self):
        import transformers.modeling_utils
        self.load_state_dict_backup = transformers.modeling_utils.load_state_dict

        def wrap_load_state_dict(orig_load_func, policies):
            def wrap_load_func(*args, **kwargs):
                state_dict = orig_load_func(*args, **kwargs)
                for policy in policies:
                    state_dict = policy.load_state_dict_post_hook(state_dict)
                return state_dict
            return wrap_load_func

        # Different policies may conflict with each other in the way they modify the state dict.
        # HF will give detailed warnings later if there is any any mismatch
        # between the state_dict and the model parameters.
        transformers.modeling_utils.load_state_dict = \
            wrap_load_state_dict(self.load_state_dict_backup, self.policies)

    def __exit__(self, exc_type, exc_value, traceback):
        for policy in self.policies:
            policy.unhook()

        if self.load_state_dict_backup is not None:
            import transformers.modeling_utils
            transformers.modeling_utils.load_state_dict = self.load_state_dict_backup

        if self.record:
            for msg, count in self.record.items():
                print(f"{msg} for {count} times")
        else:
            print("No module replaced")
