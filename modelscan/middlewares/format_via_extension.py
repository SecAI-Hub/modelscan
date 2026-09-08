from .middleware import MiddlewareBase
from modelscan.model import Model
from modelscan.settings import Property, SupportedModelFormats
from typing import Callable


class FormatViaExtensionMiddleware(MiddlewareBase):
    def __call__(self, model: Model, call_next: Callable[[Model], None]) -> None:
        extension = model.get_source().suffix
        formats = [
            format
            for format, extensions in self._settings["formats"].items()
            if extension in extensions
        ]
        normalized = []
        for format in formats:
            if isinstance(format, str):
                format = getattr(SupportedModelFormats, format, None)
            if not isinstance(format, Property):
                raise ValueError("Unsupported format name in scanner settings")
            normalized.append(format)
        if normalized:
            model.set_context(
                "formats", (model.get_context("formats") or []) + normalized
            )

        call_next(model)
