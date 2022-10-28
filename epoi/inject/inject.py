"""Injection utilities."""
from inspect import getmembers, ismodule
from .policy import encoder, decoder



def get_policy_list():
    """Scan all policies under the policy namespace."""
    policies = []
    for parent_mod in [encoder, decoder]:
        for name, mod in getmembers(parent_mod):
            if not ismodule(mod) and name.startswith("Inject") and name.endswith("Policy"):
                policies.append(mod)
    return policies


POLICIES = get_policy_list()


def find_match_policy(module, policies):
    """Iterate through all policies and find the first match."""
    for policy in policies:
        if policy.match(module):
            return policy
    return None


def inject_module(model, policy_clses=None):
    """Inject modules in the model."""
    policies = []
    for policy_cls in policy_clses if policy_clses is not None else POLICIES:
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
        for policy_cls in policies if policies is not None else POLICIES:
            self.policies.append(policy_cls())
        self.record = {}

    def __enter__(self):
        for policy in self.policies:
            policy.hook(self)

    def __exit__(self, exc_type, exc_value, traceback):
        for policy in self.policies:
            policy.unhook()

        if self.record:
            for msg, count in self.record.items():
                print(f"{msg} for {count} times")
        else:
            print("No module replaced")
