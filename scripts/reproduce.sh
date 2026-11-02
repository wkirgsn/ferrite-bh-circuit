#!/bin/bash
# Reproduce the experiments, tables and figures of the paper. Run from the repository root after `uv sync` and
# `uv run python src/prepare_data.py` (see README). Each step can be run alone: scripts/reproduce.sh <step> [...].
# Times: one AMD RX 9070 XT (16 GB, ROCm 6.4), 15 materials, 20000 steps per run unless noted.
# Training output goes to logs/, evaluations to results/, tables to paper/tables/, figures to paper/fig/.
set -e
cd "$(dirname "$0")/../src"
mkdir -p ../logs ../results
MATS=3C94,3C90,N49,3F4,N27,N30,3E6,77,78,N87,3C92,3C95,FEC007,FEC014,T37
train() {   # name config [extra args]: train all 15 materials, then evaluate on the test windows
  uv run python compact_m.py --variants ../configs/$2.json --mats $MATS --win 1000 --ks 100,500,900 --bs 768 --qreg 0.5 \
    ${3:-} --name $1 > ../logs/$1.txt 2>&1
  uv run python eval_m.py $1 > ../results/eval_$1.txt 2>&1
}

step_final() {     # final model (32 parameters, loss SRE + NERE) and the energy_w 0/3/10 variants: seed 0 (~1.9 h),
                   # then seeds 1-9 of energy_w 1 and 3 (~1 h each)
  train nere nere "--accum 4"
  for s in 1 2 3 4 5 6 7 8 9; do train ne_s$s nere_seeds "--seed $s --accum 2"; done
  uv run python main_stats.py nere final_e1                                # per-scenario statistics (CPU, slow)
  for v in final final_e1 final_e3 final_e10; do uv run python official_eval.py nere:$v; done
  for s in 1 2 3 4 5 6 7 8 9; do uv run python official_eval.py ne_s$s:final_e1 ne_s$s:final_e3; done
}

step_path() {      # smaller and unconstrained models (31-24 parameters, ~3 h), then seeds 1-9 of the 31- and
                   # 27-parameter variants (~1.2 h each)
  train small_ne small_ne "--accum 6"
  for s in 1 2 3 4 5 6 7 8 9; do train sne_s$s small_ne_seeds "--seed $s --accum 2"; done
}

step_ablation() {  # one element removed at a time (~3.6 h)
  train ablate ablate "--accum 8"
}

step_ood() {       # restricted training (temperature, frequency, data fraction) vs the GRU8 replica (~2 h)
  OOD=$(cat ../configs/ood_slots.txt)
  uv run python compact_m.py --variants ../configs/ood_final.json --mats "$OOD" --win 1000 --ks 100,500,900 --bs 768 \
    --accum 2 --qreg 0.5 --name ood_final > ../logs/ood_final.txt 2>&1
  uv run python train_gru_multi.py --mats "$OOD" --name gru_ood --steps 20000 > ../logs/gru_ood.txt 2>&1
  uv run python ood_eval.py ood_final final gru_ood > ../results/ood_final.txt 2>&1
}

step_transfer() {  # MC2 -> MC1 specimen transfer, refitting parameter groups (needs step_final; ~3 min per material)
  G='{"none": [], "tau": ["tau", "hl"], "amp": ["a1","a3","bs","dw","cz"], "tau_amp": ["tau","hl","a1","a3","bs","dw","cz"], "all": "all"}'
  for m in 3E6 3C94 3C90 N49 3F4 N27 N30 77 78 N87; do
    uv run python finetune_m.py --init nere:final_e1:$m --mat mc1:$m@70 --groups "$G" --win 1000 --ks 100,500,900 --bs 160 \
      --qreg 0.5 --name ftf_${m}_mc1 > ../logs/ftf_${m}_mc1.txt 2>&1
    grep "^val " ../logs/ftf_${m}_mc1.txt > ../results/ftf_${m}_mc1.txt
  done
}

step_checks() {    # claims in the text: equivalent circuit in SPICE (needs ngspice), step-size consistency, EKF estimator
  uv run python spice_check.py nere final_e1 $MATS > ../results/spice_check_nere_e1.txt 2>&1
  uv run python ode_check.py nere final_e1 > ../results/ode_check_nere_e1.txt 2>&1
  uv run python upsample_check.py nere final_e1 > ../results/upsample_check_nere_e1.txt 2>&1
  uv run python ekf_eval.py nere final_e1 --tune > ../results/ekf_nere_final_e1.txt 2>&1
}

step_export() {    # ferrite_bh: parameters as JSON, SPICE subcircuits, checked against PyTorch and ngspice (CPU)
  uv run python export_params.py nere:final_e1
  (cd .. && uv run python -m ferrite_bh spice)
  uv run python check_numpy.py 6 > ../results/check_numpy.txt 2>&1
  uv run python check_spice_lib.py all 2 > ../results/check_spice_lib.txt 2>&1
}

step_schedule() {  # gain-scheduled circuits (needs step_final): per-material MLP with 16 or 64 hidden units sets the
                   # element values at every step, trained jointly with the 32 parameters on the backbone of each
                   # classical seed, 20000 steps; seeds 0-9 (~1 h / ~1.7 h each); then the energy check of the MLP16 seed 0
  for h in 16 64; do
    for s in 0 1 2 3 4 5 6 7 8 9; do
      local init=nere:final_e1 name=hyAB${h}j_s$s; [ $s -gt 0 ] && init=ne_s$s:final_e1
      [ $h = 16 ] && [ $s = 0 ] && name=hyAB16j_long
      uv run python hyper_m.py --init $init --opt AB --hidden $h --bs 256 --accum 2 --joint --seed $s --steps 20000 \
        --name $name > ../logs/$name.txt 2>&1
      uv run python eval_m.py $name > ../results/eval_$name.txt 2>&1
      uv run python official_eval.py $name:final_e1+hyperAB${h}j
    done
  done
  uv run python hyper_analysis.py hyAB16j_long
}

step_tables() {    # paper/tables/*.typ
  uv run python main_table.py nere final_e1
  uv run python path_table.py
  uv run python ablate_table.py
  uv run python ood_table.py
  uv run python transfer_table.py
  uv run python param_table.py nere:final_e1
  uv run python tab_schedule.py
}

step_figures() {   # paper/fig/*.pdf, and the README figures paper/fig/ranking.svg and circuit.svg
  for f in fig_overview fig_examples fig_ranking; do uv run python $f.py; done
  (cd ../paper && uv run --group dev typst compile fig_circuit_readme.typ fig/circuit.pdf && uv run --group dev python svg_flatten.py fig/circuit.pdf fig/circuit.svg && rm fig/circuit.pdf)
}

step_paper() {     # paper/main.pdf and paper/supplement.pdf (typst from the dev dependency group)
  cd ../paper && uv run --group dev typst compile main.typ && uv run --group dev typst compile supplement.typ
}

for s in ${@:-final path ablation ood transfer checks schedule export tables figures paper}; do echo "== $s"; step_$s; done
