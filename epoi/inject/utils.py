"""Inject policy utilities."""


def get_arg(name, position, args, kwargs, default=None):
    """Get the argument from args or kwargs."""
    if name in kwargs:
        return kwargs[name]
    elif len(args) > position:
        return args[position]
    return default


def check_unsupported_arg(name, position, args, kwargs, expected_values=None):
    """Check if the argument is supported."""
    expected_values = expected_values if isinstance(expected_values, list) else [expected_values]
    val = get_arg(name, position, args, kwargs, expected_values[0])
    assert val in expected_values, f"Unsupported {name}={val}"
