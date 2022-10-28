"""Injection Policies."""
from inspect import getmembers, ismodule
from . import encoder, decoder

def get_policy_list():
    """Scan all policies under the policy namespace."""
    policies = []
    for parent_mod in [encoder, decoder]:
        for name, mod in getmembers(parent_mod):
            if not ismodule(mod) and name.startswith("Inject") and name.endswith("Policy"):
                policies.append(mod)
    return policies

POLICIES = get_policy_list()

def register_policy():
    """Register a custom policy."""
    def _do_reg(policy):
        POLICIES.append(policy)

    return _do_reg
