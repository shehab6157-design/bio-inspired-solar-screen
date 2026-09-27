#!/usr/bin/env bash
# Regenerate every result and chart in docs/, then run the tests.
#   ./run_all.sh          everything (about 20-30 minutes: real-light runs are long)
#   ./run_all.sh --quick  skip the long real-light and evolution runs (about 2 minutes)
set -e
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
cd src

step() { echo; echo "===== $1 ====="; }

step "M2  day_sim";          python3 day_sim.py
step "M3  stack_config";     python3 stack_config.py
step "M4  sensitivity";      python3 sensitivity.py
step "M6  chameleon";        python3 chameleon.py
step "M6  configure";        python3 configure.py ../devices/*.json
step "S1  spectral";         python3 spectral.py
step "S3  unified";          python3 unified.py
step "S7  polarizer";        python3 polarizer.py
step "S8  two_seas";         python3 two_seas.py
step "S9  light_score";      python3 light_score.py
step "S10 light_eye";        python3 light_eye.py

if [ "$1" != "--quick" ]; then
  step "S2  circadian";      python3 circadian.py
  step "S4  evolve";         python3 evolve.py
  if [ -f ../data/pvgis_2020.csv ]; then
    step "S5  real_data";    python3 real_data.py
    step "S5b placement";    python3 placement.py
    step "S6  hive";         python3 hive.py
    step "S6b climates";     python3 climates.py
  else
    echo "Skipping real-light runs: data/pvgis_2020.csv not found (see src/real_data.py)"
  fi
fi

step "ORGANISM";             python3 organism.py
cd ..
step "TESTS";                python3 -m pytest tests -q
