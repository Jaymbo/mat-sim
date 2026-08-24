"""High-Throughput Screening-Pipeline für Switchable Radiative Cooling.

Submodule
---------
acquisition   – Datenakquisition via Materials Project + ASE-Konvertierung
calculator     – MLIP-Calculator-Initialisierung (MACE-MP / CHGNet)
md             – Thermische Rampe (NPT-MD mit Nosé-Hoover)
metrics        – Echtzeit-Metriken: RDF, Q_l, MSD, T_switch / T_decay
storage        – SQLite-Export & -Import
pipeline       – End-to-End-Orchestrierung
analyze        – Visualisierung & optische Vor-Evaluierung
"""
__all__ = [
    "acquisition",
    "calculator",
    "md",
    "metrics",
    "storage",
    "pipeline",
    "analyze",
]
