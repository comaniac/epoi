"""Injection Policies."""
from inspect import getmembers, ismodule
from . import bert, gpt, t5, bloom


def init_policy_list():
    """Initialize policy list with builtin policies under the namespace."""
    policies = {}  # mapping from policy class to status (bool)
    for parent_mod in [bert, gpt, t5, bloom]:
        for name, mod in getmembers(parent_mod):
            if not ismodule(mod) and name.startswith("Inject") and name.endswith("Policy"):
                policies[mod] = True
    return policies


POLICIES = init_policy_list()


def get_activate_policies():
    """Get all activated policies."""
    return [policy for policy, activate in POLICIES.items() if activate]


def get_all_policies():
    """Get all policies and their status."""
    return POLICIES


def disable_policy(policy):
    """Disable a policy."""
    if policy not in POLICIES:
        raise ValueError(f"Policy {policy} not found")
    POLICIES[policy] = False


def disable_all_policies():
    """Disable all policies."""
    for policy in POLICIES:
        POLICIES[policy] = False


def enable_policy(policy):
    """Enable a policy."""
    if policy not in POLICIES:
        raise ValueError(f"Policy {policy} not found")
    POLICIES[policy] = True


def enable_all_policies():
    """Enable all policies."""
    for policy in POLICIES:
        POLICIES[policy] = True


def register_policy():
    """Register a custom policy."""

    def _do_reg(policy):
        POLICIES[policy] = True
        return policy

    return _do_reg
