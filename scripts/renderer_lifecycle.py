"""Explicit Windows GLFW resource-lifetime correction for the pinned renderer.

An obsolete MjrContext must be freed with its owning GL context current.
This changes resource cleanup only, never observations, physics, or policy inputs.
"""
import ctypes


def context_id(context):
    return ctypes.cast(context, ctypes.c_void_p).value if context else None


def dispose_in_owner_context(context, dispose, backend, identify=context_id):
    previous = backend.get_current_context()
    owner = context.gl_ctx._context
    different = identify(previous) != identify(owner)
    context.gl_ctx.make_current()
    try:
        dispose(context)
    finally:
        if different:
            backend.make_context_current(previous)


def install():
    import glfw
    from robosuite.utils.binding_utils import MjRenderContext
    if getattr(MjRenderContext, "_embodiedops_owner_cleanup", False):
        raise RuntimeError("Renderer cleanup adapter already installed")
    original = MjRenderContext.__del__

    def release(context, dispose=original, backend=glfw):
        dispose_in_owner_context(context, dispose, backend)

    MjRenderContext.__del__ = release
    MjRenderContext._embodiedops_owner_cleanup = True
