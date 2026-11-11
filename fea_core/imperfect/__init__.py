"""Geometric imperfections and nonlinear buckling (F9 C2, GNIA / GMNIA) – see fea_core.imperfect.run."""
from .model import IMPERFECTION_KINDS, ImperfectionError, NonlinearBucklingSettings

__all__ = ["IMPERFECTION_KINDS", "ImperfectionError", "NonlinearBucklingSettings"]
