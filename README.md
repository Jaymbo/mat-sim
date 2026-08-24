# mat-sim – Switchable Radiative Cooling Pipeline

High-Throughput Screening-Pipeline, die kristalline Strukturen aus dem
Materials Project virtuell „aufheizt“ (NPT-MD mit MLIP), um
Phasenwechsel-Materialien für adaptive passive Strahlungskühlung zu
entdecken.

## Architektur

```
mat_sim/
├── __init__.py        Paket-Marker
├── acquisition.py     MP-Query + pymatgen→ASE-Konvertierung
├── calculator.py      MLIP-Factory (MACE-MP / CHGNet)
├── md.py              Thermische Rampe (NPT, Nosé-Hoover)
├── metrics.py         RDF, Steinhardt Q4, MSD, T_switch/T_decay
├── storage.py         SQLite-Export
├── pipeline.py        End-to-End-Orchestrierung
└── run.py             CLI-Einstiegspunkt
```

## Quick Start

```bash
pip install pymatgen ase mace-torch chgnet scipy numpy

export MP_API_KEY="dein_materials_project_key"

python -m mat_sim.run --chemsys V-O Ti-O Sr-Ti-O --t-max 600 --delta-t 10
```

## Methodik

| Komponente | Implementierung |
|---|---|
| MLIP | MACE-MP (Universal-Potenzial, Default) oder CHGNet |
| MD-Ensemble | NPT mit Nosé-Hoover-Thermostat & -Barostat (1 atm) |
| Temperaturrampe | 0 K → T_max in Schritten von ΔT, pro Schritt N thermalization steps |
| Phasenwechsel (T_switch) | Diskontinuierliche RDF-Peak-Verschiebung (> 0.15 Å) |
| Zerfall (T_decay) | Lindemann-Kriterium: MSD > 12 % × d_NN² |

## Ergebnis-Export

SQLite-Datenbank (`results.db`) mit Tabelle `materials`:

| Spalte | Beschreibung |
|---|---|
| `material_id` | MP-ID (Primary Key) |
| `formula` | Chemische Formel |
| `t_switch` | Phasenwechsel-Temperatur (K) oder NULL |
| `t_decay` | Zerfalls-Temperatur (K) oder NULL |
| `rdf_before_json` / `rdf_after_json` | RDF-Snapshots um T_switch (für Mie-Streuung) |
| `temperatures` / `msd_values` / `ql_values` / `volumes` / `energies` | Komplette Verläufe als JSON-Arrays |
