"""Compatibility entrypoint that re-exports the official OpenEnv server app."""

from server.app import app, main

__all__ = ["app", "main"]
