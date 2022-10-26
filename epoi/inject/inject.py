"""Injection utilities."""
from inspect import getmembers, ismodule
from .policy import encoder, decoder


def get_policy_list():
    policies = []
    for parent_mod in [encoder, decoder]:
        for name, mod in getmembers(parent_mod):
            if not ismodule(mod) and name.startswith("Inject") and name.endswith("Policy"):
                policies.append(mod)
    return policies


POLICIES = get_policy_list()


def inject_module(model, policies=None):
    """Inject modules in the model."""
    policies = policies if policies is not None else POLICIES
    record = {}

    def _helper(model):
        """Traverse model's children recursively and apply any transformations in ``policies``."""
        for name, child in model.named_children():
            for policy in policies:
                if policy.match(child):
                    msg = f"Apply {policy.__name__} to {name}"
                    if msg not in record:
                        record[msg] = 0
                    record[msg] += 1
                    new_child = policy.init(child)
                    policy.assign_params(new_child, child)
                    setattr(model, name, new_child)
                    replaced = True
                    break
            else:
                _helper(child)

    _helper(model)
    if record:
        for msg, count in record.items():
            print(f"{msg} for {count} times")
    else:
        print("No module replaced")
