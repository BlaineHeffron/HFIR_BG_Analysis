"""Verified spectrum-name corrections; old releases retain their original names."""

# File 186/run 295: 6 May 2021, within Cycle 491 (13 April–8 May).
SPECTRUM_RENAMES = {"CYCLE461_DOWN_FACING_OVERNIGHT": "CYCLE491_DOWN_FACING_OVERNIGHT"}


def spectrum_name_candidates(name):
    """Canonical stem first, then historical alias; never rewrite arbitrary cycles."""
    stem = str(name).removesuffix('.txt')
    for old, new in SPECTRUM_RENAMES.items():
        if stem in (old, new):
            return (new, old)
    return (stem,)
