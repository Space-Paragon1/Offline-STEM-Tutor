from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Tuple

# Minimal periodic table (we can expand later)
ATOMIC_MASS: Dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.098,
    "Ca": 40.078,
    "Fe": 55.845,
    "Cu": 63.546,
    "Zn": 65.38,
}

TOKEN_RE = re.compile(r"([A-Z][a-z]?|\(|\)|\d+)")

def parse_formula(formula: str) -> Dict[str, int]:
    """
    Parse a chemical formula like H2O, Ca(OH)2 into element counts.
    Supports parentheses + multipliers. No hydration dots yet.
    """
    tokens = TOKEN_RE.findall(formula.strip())
    if not tokens:
        raise ValueError("Empty formula.")

    stack = [dict()]  # stack of dicts
    i = 0

    def add_elem(d: Dict[str, int], elem: str, count: int):
        d[elem] = d.get(elem, 0) + count

    while i < len(tokens):
        t = tokens[i]

        if t == "(":
            stack.append({})
            i += 1
            continue

        if t == ")":
            if len(stack) == 1:
                raise ValueError("Unmatched ')'.")
            group = stack.pop()
            i += 1
            mult = 1
            if i < len(tokens) and tokens[i].isdigit():
                mult = int(tokens[i])
                i += 1
            for elem, cnt in group.items():
                add_elem(stack[-1], elem, cnt * mult)
            continue

        if re.match(r"[A-Z][a-z]?$", t):  # element
            elem = t
            i += 1
            count = 1
            if i < len(tokens) and tokens[i].isdigit():
                count = int(tokens[i])
                i += 1
            add_elem(stack[-1], elem, count)
            continue

        if t.isdigit():
            # digits should only follow element or ')'
            raise ValueError(f"Unexpected number '{t}' in formula.")
        raise ValueError(f"Unexpected token '{t}'.")

    if len(stack) != 1:
        raise ValueError("Unmatched '('.")

    return stack[0]

def molar_mass(formula: str) -> float:
    counts = parse_formula(formula)
    total = 0.0
    unknown = [e for e in counts if e not in ATOMIC_MASS]
    if unknown:
        raise ValueError(f"Unknown element(s): {', '.join(unknown)}. Add them to ATOMIC_MASS.")
    for elem, cnt in counts.items():
        total += ATOMIC_MASS[elem] * cnt
    return total

import re
from typing import Optional

NUM_UNIT_RE = re.compile(r"([-+]?\d*\.?\d+)\s*([a-zA-Z/]+)?")

def _to_float(x: str) -> float:
    return float(x.replace(",", "").strip())

def liters_from(value: float, unit: str) -> float:
    u = (unit or "L").strip().lower()
    if u in ("l", "liter", "liters"):
        return value
    if u in ("ml", "milliliter", "milliliters"):
        return value / 1000.0
    raise ValueError(f"Unsupported volume unit '{unit}'. Use L or mL.")

def moles_from_mass(mass_g: float, formula: str) -> float:
    mm = molar_mass(formula)
    return mass_g / mm

def parse_number_and_unit(text: str) -> tuple[float, str]:
    m = NUM_UNIT_RE.search(text.strip())
    if not m:
        raise ValueError("Could not parse a number and unit.")
    val = _to_float(m.group(1))
    unit = (m.group(2) or "").strip()
    return val, unit

def calc_molarity(*, moles: Optional[float] = None, mass_g: Optional[float] = None,
                  formula: Optional[str] = None, volume_value: float, volume_unit: str) -> float:
    V = liters_from(volume_value, volume_unit)
    if V <= 0:
        raise ValueError("Volume must be > 0.")

    if moles is None:
        if mass_g is None or not formula:
            raise ValueError("Provide either moles, or (mass"
            ""
            ""
            ""
            ""
            ""
            ""
            " in g + formula).")
        moles = moles_from_mass(mass_g, formula)

    if moles <= 0:
        raise ValueError("Moles must be > 0.")
    return moles / V
