"""Base class of model injection policy."""


class ModuleInjectPolicy:
    def __init__(self):
        self.backup = {}

    @classmethod
    def init_from_object(cls, orig, **kwargs):
        """Initialize an instance from a created object."""
        args = cls.gen_init_config_from_object(orig, **kwargs)
        ret = cls.inject_module(**kwargs)(**args)
        cls.assign_params(ret, orig, **kwargs)
        cls.wrap_forward(cls, orig.__class__, ret)
        return ret

    @staticmethod
    def target_modules():
        """A list of target modules to be injected."""
        return []

    @staticmethod
    def wrap_forward(policy_cls, orig_cls, this):
        """Wrap the original module's forward method to deal with inconsistent I/O layouts."""
        this.forward = policy_cls.gen_wrap_forward(orig_cls, this.forward)

    def match(self, mod):
        """Check if the other module matches the module that could be replaced."""
        for target_module, target_cls in self.target_modules():
            if isinstance(mod, getattr(target_module, target_cls)):
                return True
        return False

    def hook(self, ctx):
        """Hook the target modules to be injected with the custom one.
        With the hook, all target modules will be replaced with the custom one during
        the construction.
        """
        if not self.target_modules():
            return

        policy = self
        module_cls = self.inject_module()

        # Mock the target modules to be the inject modules.
        for target_module, target_cls in self.target_modules():
            self.backup[(target_module, target_cls)] = target_cls

            # Explicitly create a wrapper instead of using lambda to avoid lazy initialization.
            # In that case, target_module and target_cls may be the last values in the loop.
            def wrap_forward(self):
                policy.wrap_forward(policy, getattr(target_module, target_cls), self)

            class InjectedModule(module_cls):
                """Wrap the inject module to
                1. Re-organize the constructor arguments for original module to match the inject module.
                2. Wrap the forward method to deal with inconsistent I/O layout.
                """

                def __init__(self, *args, **kwargs):
                    msg = f"Apply {module_cls.__name__}"
                    if msg not in ctx.record:
                        ctx.record[msg] = 0
                    ctx.record[msg] += 1
                    new_args = policy.gen_init_config_from_config(*args, **kwargs)
                    module_cls.__init__(self, **new_args)
                    wrap_forward(self)

            setattr(target_module, target_cls, InjectedModule)

    def unhook(self):
        """Unhook the target modules."""
        if not self.target_modules():
            return

        # Unmock the target modules.
        for target_module, target_cls in self.target_modules():
            setattr(target_module, target_cls, self.backup[(target_module, target_cls)])
            del self.backup[(target_module, target_cls)]

    @staticmethod
    def inject_module(**kwargs):
        """The custom module to inject."""
        raise NotImplementedError()

    @staticmethod
    def gen_init_config_from_object(orig, **kwargs):
        """Generate the initialization config from an existing object."""
        raise NotImplementedError()

    @staticmethod
    def gen_init_config_from_config(*args, **kwargs):
        """Generate the initialization config from the config of the module."""
        raise NotImplementedError()

    @staticmethod
    def assign_params(this, orig, **kwargs):
        """Assign the parameters in the original module to the injected module."""
        raise NotImplementedError()

    @staticmethod
    def gen_wrap_forward(orig_cls, forward):
        """Generate a wrapper to wrap the inject module's forward function."""
        return forward

    @staticmethod
    def load_state_dict_post_hook(state_dict):
        """Rename the parameters in the state_dict for the injected modules"""
        return state_dict
