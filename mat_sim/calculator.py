"""MLIP-Calculator-Initialisierung.

Unterstützt MACE-MP (Default) und CHGNet als Fallback.
Beide liefern Kräfte & Energien ohne DFT-Aufwand.

Automatische Device-Erkennung:
  - Wenn PyTorch eine CUDA-GPU findet → device='cuda', dtype='float32'
  - Andernfalls → device='cpu', dtype='float64'
"""

from __future__ import annotations

import logging
from typing import Literal

from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)


def _detect_device(requested: str) -> str:
    """Device-String validieren bzw. automatisch erkennen.

    Wenn ``requested == "auto"`` (oder None), wird CUDA genutzt,
    falls PyTorch eine GPU findet.
    """
    if requested in ("cpu", "cuda"):
        return requested

    # Auto-Erkennung
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA-GPU erkannt → device='cuda'")
            return "cuda"
    except Exception:  # noqa: BLE001
        pass

    logger.info("Keine GPU erkannt → device='cpu'")
    return "cpu"


def get_mace_calculator(
    model: str = "medium",
    device: str = "cpu",
    dispersion: bool = False,
) -> Calculator:
    """MACE-MP Universal-Potenzial initialisieren.

    Parameters
    ----------
    model
        Modellgröße: ``"small"``, ``"medium"`` oder ``"large"``.
    device
        ``"cpu"``, ``"cuda"`` oder ``"auto"``.
    dispersion
        D3-Dispersionskorrektur aktivieren.

    Returns
    -------
    Calculator
        ASE-kompatibler MACE-Calculator.
    """
    from mace.calculators import mace_mp

    device = _detect_device(device)

    # float32 bei CUDA für Tensor-Core-Optimierung (A100/H100)
    # float64 bei CPU für numerische Präzision
    default_dtype = "float32" if device == "cuda" else "float64"

    logger.info("MACE-MP: device=%s, dtype=%s, model=%s", device, default_dtype, model)
    return mace_mp(
        model=model,
        device=device,
        dispersion=dispersion,
        default_dtype=default_dtype,
    )


def get_chgnet_calculator(
    model: str = "0.2.0",
    device: str = "cpu",
) -> Calculator:
    """CHGNet-MLIP initialisieren (Fallback)."""
    from chgnet.model.dynamics import CHGNetCalculator

    device = _detect_device(device)
    logger.info("CHGNet: device=%s, model=%s", device, model)
    return CHGNetCalculator(model=model, use_device=device)


def get_calculator(
    backend: Literal["mace", "chgnet"] = "mace",
    device: str = "cpu",
) -> Calculator:
    """Einheitliche Factory für MLIP-Calculator.

    Parameters
    ----------
    backend
        ``"mace"`` (Default, MACE-MP) oder ``"chgnet"``.
    device
        ``"cpu"``, ``"cuda"`` oder ``"auto"``.
    """
    if backend == "mace":
        return get_mace_calculator(device=device)
    elif backend == "chgnet":
        return get_chgnet_calculator(device=device)
    else:
        raise ValueError(f"Unbekannter Backend: {backend!r}")
