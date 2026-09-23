"""BIGFix: Bidirectional Image Generation with Token Fixing."""

__version__ = "1.0.0"

__all__ = ["MaskGITPipeline", "MaskGITOutput"]


def __getattr__(name):
    # Lazy export: `from bigfix import MaskGITPipeline` works, but a bare `import bigfix` (or
    # `import bigfix.utils.utils`) does not pull in torch, transformers, etc.
    if name in __all__:
        from bigfix import txt2img_pipeline
        return getattr(txt2img_pipeline, name)
    raise AttributeError(f"module 'bigfix' has no attribute {name!r}")
