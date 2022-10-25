"""Inject policy utilities."""


def get_arg(name, position, args, kwargs, default=None):
    """Get the argument from args or kwargs."""
    if name in kwargs:
        return kwargs[name]
    elif len(args) > position:
        return args[position]
    return default


def check_unsupported_arg(name, position, args, kwargs, expected_value=None):
    """Check if the argument is supported."""
    val = get_arg(name, position, args, kwargs)
    assert val == expected_value, f"Unsupported {name}={val}"
