# mat-sim – Switchable Radiative Cooling Pipeline

High-Throughput Screening-Pipeline, die kristalline Strukturen aus dem
Materials Project virtuell „aufheizt\" (NPT-MD mit MLIP), um
Phasenwechsel-Materialien für adaptive passive Strahlungskühlung zu
entdecken. Optische Eigenschaften werden via Drude-Lorentz-Modell und
Mie-Streuung bewertet (Cooling-/Heating-Score für Smart-Textile-Anwendungen).

## Architektur

```
mat_sim/
├── acquisition.py    MP-Query + pymatgen→ASE-Konvertierung + ternäre Erweiterung
├── calculator.py     MLIP-Factory (MACE-MP / CHGNet) mit Auto-CUDA-Erkennung
├── md.py             Thermische Rampe (NPT-MD, Nosé-Hoover, Early-Stopping)
├── metrics.py        RDF, Steinhardt Q4, MSD, T_switch/T_decay-Erkennung
├── optics.py         Drude-Lorentz ε(ω) + IR-Phononen + Mie-Streuung + Scores
├── storage.py        SQLite-Queue (structures) + Ergebnis-DB (materials)
├── pipeline.py       Zwei-Phasen-Orchestrierung: Ingest → Process
├── analyze.py        2×3-Dashboard + optische Vor-Evaluierung
└── run.py            CLI-Einstiegspunkt
```

## Quick Start

```bash
# 1. Repo klonen
git clone git@github.com:Jaymbo/mat-sim.git
cd mat-sim

# 2. API-Key setzen
cp .env.example .env
# → MP_API_KEY in .env eintragen

# 3. Environment mit uv erstellen
uv sync

# 4. Strukturen herunterladen (Ingest)
uv run python -m mat_sim.run --ingest --chemsys V-O Ti-O Cr-O Mn-O Sr-Ti-O

# 5. Simulation starten (Process)
uv run python -m mat_sim.run --db results.db --device auto --duration-min 25

# 6. Ergebnisse analysieren
uv run python -m mat_sim.run --analyze --all --db results.db
```

Alternativ mit Conda: `conda env create -f environment.yml && conda activate mat-sim`.

## Workflow

### Zwei-Phasen-Architektur

```
┌─────────┐     ┌──────────────────┐     ┌──────────┐
│ Ingest  │────▶│  SQLite-Queue    │────▶│ Process  │
│ (1×)    │     │  structures      │    │ (N× par.)│
│ MP → DB │     │  pending → done  │    │ MD + Opt │
└─────────┘     └──────────────────┘     └──────────┘
```

**Phase 1 – Ingest** (einmalig): Lädt Strukturen von Materials Project
(einschließlich ternärer Verbindungen) und speichert sie in der
`structures`-Tabelle der SQLite-Datenbank (`status='pending'`).

**Phase 2 – Process** (parallelisierbar): Jeder Worker claimt atomar die
nächste `pending`-Struktur, führt NPT-MD durch, speichert das Ergebnis und
markiert sie als `done`. Bei Time-Out → Exit 88 (Dispatcher reicht neuen
Job ein).

### SLURM-Cluster-Betrieb

```bash
# 1. Auf dem Login-Knoten: uv installieren (falls nicht vorhanden)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Environment erstellen
uv sync

# 3. Einmalig: Ingest (CPU-Job)
sbatch ingest.sbatch

# 4. Dispatcher starten (lokal, tmux oder nohup)
./dispatcher.sh --max-jobs 10

# 5. Status überwachen
uv run python -m mat_sim.run --queue-stats --db results.db
```

Der Dispatcher überwacht die Queue und reicht automatisch neue Worker-Jobs
ein, bis alle Strukturen verarbeitet sind oder `MAX_JOBS` erreicht ist.

| Datei | Rolle |
|---|---|
| `ingest.sbatch` | CPU-Job: Strukturen von MP → DB |
| `run_cluster.sbatch` | GPU-Worker: claimt + simuliert bis Time-Out |
| `dispatcher.sh` | Überwacht Queue, reicht Worker-Jobs bis `--max-jobs` |

## CLI-Referenz

### Ingest
```bash
uv run python -m mat_sim.run --ingest --chemsys V-O Ti-O [--include-ternary] [--e-hull-max 0.1]
```

### Process
```bash
uv run python -m mat_sim.run --db results.db [--device auto] [--duration-min 25] [--t-max 500] [--delta-t 10]
```

### Analyse
```bash
uv run python -m mat_sim.run --analyze --material-id mp-1234 --db results.db
uv run python -m mat_sim.run --analyze --all --db results.db
```

### Queue-Management
```bash
uv run python -m mat_sim.run --queue-stats --db results.db
uv run python -m mat_sim.run --reset-stale --db results.db [--stale-minutes 30]
```

### Alle Optionen
```bash
uv run python -m mat_sim.run --help
```

## Methodik

| Komponente | Implementierung |
|---|---|
| MLIP | MACE-MP (Universal-Potenzial, Default) oder CHGNet |
| MD-Ensemble | NPT mit Nosé-Hoover-Thermostat & -Barostat (1 atm) |
| Temperaturrampe | 0 K → T_max in Schritten von ΔT, pro Schritt N thermalization steps |
| Early Stopping | Abbruch bei thermischem Gleichgewicht (rel. Std < 2 % nach 20 Schritten) |
| Phasenwechsel (T_switch) | Diskontinuierliche RDF-Peak-Verschiebung (> 0.15 Å) |
| Zerfall (T_decay) | Lindemann-Kriterium: MSD > 12 % × d_NN² |
| Optik | Drude-Lorentz ε(ω) + IR-Phononen (8–13 µm) + Mie-Streuung (PyMieScatt) |
| Cooling-Score | 50 % Solar-Reflexion (AM1.5-gewichtet) + 50 % IR-Emissivität |
| Heating-Score | 50 % Solar-Absorption + 50 % IR-Rückhaltung (1−ε) |
| Total-Score | Arithmetisches Mittel aus Cooling und Heating |

## Datenbank-Schema

### `structures` (Queue)
| Spalte | Beschreibung |
|---|---|
| `material_id` | MP-ID (Primary Key) |
| `formula` | Chemische Formel |
| `chemsys` | Chemisches System (z. B. `V-O`) |
| `structure_json` | Serialisierte pymatgen-Struktur |
| `status` | `pending` → `processing` → `done` / `error` |
| `claimed_by` | Worker-ID (`hostname-pid`) |
| `claimed_at` | Zeitstempel des Claims |

### `materials` (Ergebnisse)
| Spalte | Beschreibung |
|---|---|
| `material_id` | MP-ID (Primary Key) |
| `formula` | Chemische Formel |
| `status` | `converged` / `diverged` / `decayed` |
| `t_switch` | Phasenwechsel-Temperatur (K) oder NULL |
| `t_decay` | Zerfalls-Temperatur (K) oder NULL |
| `rdf_before_json` / `rdf_after_json` | RDF-Snapshots um T_switch |
| `temperatures` / `msd_values` / `ql_values` / `volumes` / `energies` | Verläufe als JSON-Arrays |

## Abhängigkeiten

- Python ≥ 3.11
- pymatgen, mp-api (Materials Project API)
- ase (Atomic Simulation Environment)
- mace-torch (MACE-MP Universal Potential)
- chgnet (Fallback MLIP)
- scipy, numpy, matplotlib
- PyMieScatt (Mie-Streuung)
- python-dotenv (API-Key aus `.env`)

Siehe `pyproject.toml` (uv/pip) oder `environment.yml` (Conda).

## Lizenz

MIT – siehe `LICENSE`.
