"""Contacts of assemblies (F8.5): fixed connections in the conforming mesh (`bonded`) and the contact pairs of ccx (`model`, `detect`,
`inp`, `results`, `check`). Backward-compatible re-export `bonded_pairs`, `edge_only` (formerly fea_core.contacts).
"""
from .bonded import bonded_pairs, edge_only  # noqa: F401
from .model import (ContactPair, ContactSettings, AnalysisSettings, LoadStep, Motion,  # noqa: F401
                    CONTACT_TYPES, CONTACT_LABELS, NONLINEAR_TYPES)
