"""Base class of model injection policy."""


class ModuleInjectPolicy:
    @classmethod
    def init(cls, orig, **kwargs):
        """Initialize an instance to inject."""
        ret = cls.init_impl(orig, **kwargs)
        cls.wrap_forward(ret)
        return ret

    @staticmethod
    def match(other):
        """Check if the other module matches the module that could be replaced."""
        return False

    @staticmethod
    def init_impl(orig, **kwargs):
        """Initialize an instance to inject."""
        raise NotImplementedError()

    @staticmethod
    def assign_params(this, orig):
        """Assign the parameters in the original module to the injected module."""
        raise NotImplementedError()

    @staticmethod
    def wrap_forward(this):
        """Wrap the original module's forward method to deal with inconsistent I/O layouts."""
