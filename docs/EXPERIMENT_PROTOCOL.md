# SG-MBRL Full Experiment Protocol

This document turns the expanded experiment outline into a concrete research execution plan. The goal is to make the paper support five separate claims:

1. **Sample efficiency:** SG-MBRL reaches good control performance with fewer real environment transitions.
2. **Long-horizon physical stability:** SG-MBRL produces physically plausible long rollouts, not just accurate one-step predictions.
3. **Graph-size and topology transfer:** SG-MBRL learns local interaction laws that transfer across graph sizes and adjacency patterns.
4. **Control improvement:** SG-MBRL improves actual task return, success rate, and terminal task error.
5. **Component necessity:** graph structure, symplectic/Hamiltonian structure, and uncertainty each contribute.

The single highest-priority experiment is the matched-planner comparison:

> **SG-MBRL vs DirectGNN-MPC under identical CEM-MPC settings.**

This is the cleanest test of whether symplectic graph Hamiltonian structure improves model-based control beyond a generic graph world model.

---

## 0. Global Experimental Standards

### 0.1 Repository Organization

Use a deterministic experiment structure:

```text
experiments/
  configs/
    envs/
    models/
    planners/
    sweeps/
  src/
    envs/
    models/
    planners/
    training/
    eval/
    metrics/
  scripts/
    run_sweep.py
    aggregate_results.py
    make_plots.py
    make_tables.py
    sanity_check_rollouts.py
  results/
    raw/
    aggregated/
    figures/
    tables/
```

Every run should save:

- config file,
- git commit hash,
- random seed,
- environment name and graph instance,
- model hyperparameters,
- planner hyperparameters,
- parameter count,
- training curves,
- evaluation trajectories,
- imagined MPC trajectories,
- model predictions,
- energy values,
- uncertainty estimates,
- wall-clock time.

### 0.2 Random Seeds

Use the same seed set across matched comparisons.

Recommended seed sets:

```text
MBRL headline seeds:        0,1,2,3,4,5,6,7,8,9
Minimum MBRL seeds:         0,1,2,3,4
Model-free seeds:           0,1,2,3,4,5,6,7,8,9
Dynamics-only seeds:        0,1,2,3,4
```

Use paired seeds for fairness: seed `s` should generate the same train/eval graph instances, initial states, target states, and action-noise streams for all methods where possible.

### 0.3 Reporting Rules

For every scalar result report:

- mean across seeds,
- standard error,
- 95% bootstrap confidence interval when possible,
- number of seeds,
- environment interaction budget,
- model parameter count.

For plots:

- line = seed mean,
- band = standard error or 95% CI,
- x-axis = real environment transitions unless otherwise stated,
- use identical axes across methods inside the same environment.

For statistical claims:

- use paired comparisons when seeds and test instances are matched,
- report effect sizes in addition to p-values,
- emphasize trends and confidence intervals over single-run wins.

### 0.4 Shared Evaluation Protocol

For each trained controller/model:

1. Freeze the model and planner settings.
2. Evaluate on a held-out set of 50 to 200 initial states per environment.
3. Use no exploration noise during evaluation unless the method is inherently stochastic.
4. Use identical evaluation initial states and targets for all methods.
5. Save full trajectories and final task metrics.

Recommended evaluation episodes:

```text
small/fast tasks:   200 eval episodes per seed
hard particle tasks: 50-100 eval episodes per seed
```

---

## 1. Primary Headline Experiment: SG-MBRL vs Strong MBRL Baselines

### Claim Tested

SG-MBRL achieves better control return with fewer real environment interactions than generic MBRL.

### Environments

Run at least three:

1. **Coupled oscillator graph stabilization**
   - Graphs: chains, grids, random geometric graphs.
   - State: `z=(q,p)`.
   - Action: bounded nodal forces.
   - Reward: negative average energy or distance to rest state.

2. **Spring-mass chain target reaching**
   - Graphs: chains with fixed or free endpoints.
   - Target: desired node positions.
   - Reward: negative target error, velocity penalty, action penalty.

3. **Lennard-Jones or sparse N-body particle steering**
   - Graphs: nearest-neighbor or radius graphs.
   - Target: steer particles to positions or formation.
   - Reward: negative target/formation error, collision penalty, action penalty.

### Methods

Compare:

| Method | Purpose |
| --- | --- |
| SG-MBRL: ensemble ResidualSympGNN + CEM-MPC | full method |
| DirectGNN + CEM-MPC | graph but not Hamiltonian/symplectic |
| PETS-style DirectGNN ensemble | strong uncertainty-aware MBRL baseline |
| MBPO-style short-rollout policy optimization | strong model-based policy-learning baseline |
| HNN/HGN + CEM-MPC | Hamiltonian baseline |
| MLP dynamics + CEM-MPC | no graph, no symplectic |
| no-control / random-action | sanity checks |

### Environment Budgets

Use:

```text
N_env = 1k, 2.5k, 5k, 10k, 25k, 50k, 100k
```

For hard particle tasks optionally extend to:

```text
N_env = 250k, 500k
```

### Step-by-Step Procedure

1. Define environment configs for all three tasks.
2. Define a shared train/eval split of graph instances, initial states, and target states.
3. For each seed and method:
   - initialize replay buffer with the same random transitions,
   - train/evaluate at each budget checkpoint,
   - save model checkpoints and evaluation trajectories.
4. For each checkpoint:
   - train or update the world model,
   - run MPC/policy optimization using the method-specific controller,
   - collect real transitions until the next budget,
   - evaluate frozen controller on the held-out set.
5. Aggregate across seeds.
6. Produce return-vs-steps plots per environment.
7. Compute samples-to-threshold using task-specific success thresholds.
8. Report final return, cumulative regret/reward, terminal error, and success rate.

### Required Plots

For each environment:

- return vs `N_env`,
- terminal task error vs `N_env`,
- success rate vs `N_env`,
- cumulative regret/reward vs `N_env`,
- samples-to-threshold bar chart.

### Acceptance Pattern

The headline is supported if SG-MBRL:

- reaches a chosen success threshold at fewer environment steps,
- has better or comparable final return,
- has lower variance than generic baselines on hard tasks,
- does not lose badly to PETS/MBPO under tuned settings.

---

## 2. Matched-Planner Experiment: SG-MBRL vs DirectGNN-MPC

### Claim Tested

Symplectic graph Hamiltonian structure helps beyond graph prediction alone.

### Compared Methods

Only compare:

```text
SG-MBRL + CEM-MPC
DirectGNN + same CEM-MPC
```

Everything must be identical except the learned dynamics parameterization.

### Matched Conditions

Match:

- same real transition data,
- same replay-buffer sampling,
- same graph input,
- same action bounds,
- same reward,
- same planner horizon,
- same CEM population,
- same CEM iterations,
- same CEM elite fraction,
- same model-training epochs,
- similar parameter count,
- same evaluation seeds.

### Planner Sweep

Use:

```text
K_plan = 5, 10, 20, 30
CEM population = 512 or 1024
CEM iterations = 5 to 10
elite fraction = 0.05 to 0.15
```

### Metrics

- MPC return,
- terminal task error,
- model exploitation gap,
- rollout divergence rate,
- energy drift inside imagined rollouts,
- planning horizon sensitivity,
- action smoothness.

### Step-by-Step Procedure

1. Train both models on exactly the same datasets at each `N_env`.
2. Verify parameter counts are within 10-20%, or report both parameter-matched and overparameterized variants.
3. For each held-out state, run CEM with identical random candidate initializations.
4. Save the selected action sequence and imagined trajectory.
5. Execute the selected first action in the true simulator.
6. For diagnostic evaluation, execute the full planned open-loop sequence in the true simulator and compute:
   - imagined return `J_hat`,
   - true open-loop return `J`,
   - exploitation gap `J_hat - J`.
7. Repeat for all horizons.
8. Plot return, energy drift, divergence rate, and exploitation gap vs planning horizon.

### Acceptance Pattern

This experiment supports the core claim if:

- SG-MBRL has better return under identical CEM settings,
- DirectGNN degrades more at long planning horizons,
- SG-MBRL has smaller imagined-rollout energy drift,
- SG-MBRL has smaller exploitation gap.

---

## 3. Dynamics and Long-Horizon Rollout Experiment

### Claim Tested

Generic models can have good one-step error but produce globally nonphysical rollouts.

### Environments

Use:

1. harmonic oscillator,
2. controlled pendulum,
3. double pendulum,
4. spring-mass chain,
5. oscillator graph,
6. Lennard-Jones particles.

### Models

Compare:

- SG-MBRL SympGNN,
- DirectGNN,
- MLP next-state,
- HNN,
- HGN,
- SympNoGraph,
- GraphNoSymp.

### Rollout Horizons

Use:

```text
K_rollout = 1, 10, 25, 50, 100, 250, 500
```

### Metrics

- one-step prediction RMSE,
- multi-step rollout RMSE,
- relative energy drift,
- maximum energy drift,
- final energy drift,
- symplecticity defect,
- rollout divergence rate.

Symplecticity defect:

```text
|| D P_hat(z)^T J D P_hat(z) - J ||_F
```

Use automatic differentiation or finite differences for `D P_hat`.

### Step-by-Step Procedure

1. Generate train/validation/test trajectory datasets for each environment.
2. Train each model with identical train/validation splits.
3. Early-stop on one-step validation loss.
4. Evaluate one-step RMSE on held-out transitions.
5. For each test initial state, roll out each model for each horizon.
6. Compare against the true simulator trajectory.
7. Compute energy along model and true trajectories.
8. Estimate symplecticity defect at sampled states.
9. Mark divergence if:
   - state norm exceeds a predefined physical bound,
   - energy error exceeds a threshold,
   - NaNs/infs occur,
   - rollout RMSE exceeds a task-specific threshold.
10. Plot one-step error, rollout RMSE, energy drift, symplecticity defect, and divergence rate vs horizon.

### Key Reviewer-Proof Plot

Create a two-panel figure:

- left: one-step validation error, showing SG-MBRL and DirectGNN approximately matched,
- right: long-horizon energy drift or divergence rate, showing SG-MBRL much better.

### Acceptance Pattern

The physics claim is supported if SG-MBRL has:

- comparable one-step error,
- lower long-horizon energy drift,
- lower divergence rate,
- lower symplecticity defect,
- more stable phase portraits.

---

## 4. Graph-Size Transfer Experiment

### Claim Tested

Graph-local node/edge energy learning transfers to larger systems better than dense or generic predictors.

### Training Sizes

Use:

```text
|V| = 4, 8, 16
```

### Test Sizes

Use:

```text
|V| = 32, 64, 128, 256
```

For grids:

```text
4x4 -> 8x8 -> 16x16
```

### Environments

Use:

1. spring-mass chains,
2. coupled oscillator graphs,
3. mass-spring grids,
4. sparse N-body graphs,
5. Lennard-Jones particle systems.

### Metrics

- rollout RMSE vs graph size,
- return vs graph size,
- energy drift vs graph size,
- success rate vs graph size,
- transfer gap:

```text
J_small - J_large
```

Also report per-node normalized metrics so larger graphs are not unfairly penalized by dimensionality:

```text
RMSE_per_node = ||z_hat - z||_2 / sqrt(|V|)
EnergyDrift_per_node = |H(z_t)-H(z_0)| / |V|
```

### Step-by-Step Procedure

1. Train models only on small graphs.
2. Use shared node/edge functions for graph-local models.
3. Test zero-shot on larger graphs without retraining.
4. Evaluate both prediction and control.
5. For control, use the same MPC settings across graph sizes, scaling action dimensions but not changing model architecture.
6. Plot metrics against `|V|` on a log-scaled x-axis.
7. Report transfer gap from training size to each test size.

### Acceptance Pattern

This supports the graph-locality claim if SG-MBRL degrades more gracefully as `|V|` increases, especially compared with MLP and nonlocal baselines.

---

## 5. Topology Transfer Experiment

### Claim Tested

SG-MBRL learns reusable local interaction laws rather than memorizing one graph template.

### Train Topologies

Use either:

```text
chains
```

or:

```text
chains + small grids
```

### Test Topologies

Test on:

- cycles,
- stars,
- trees,
- grids,
- Erdos-Renyi graphs,
- random geometric graphs,
- sparse nearest-neighbor particle graphs.

### Metrics

- rollout RMSE,
- energy drift,
- return,
- success rate,
- graph-size normalized performance,
- transfer gap by topology.

### Step-by-Step Procedure

1. Train on the selected source topology set.
2. Hold out all target topologies from training.
3. Generate target graphs with matched node counts and degree ranges where possible.
4. Evaluate zero-shot prediction rollouts.
5. Evaluate zero-shot MPC control.
6. Optionally evaluate few-shot adaptation with 100, 500, and 1000 transitions from the new topology.
7. Plot performance by topology.

### Acceptance Pattern

The claim is supported if SG-MBRL transfers better than fixed-template or generic models, especially to topologies with similar local interactions but different global adjacency.

---

## 6. Ablation Matrix

### Claim Tested

The gain comes from the combination of graph structure, symplectic/Hamiltonian structure, and uncertainty.

### Variants

| Variant | Graph | Symplectic/Hamiltonian | Uncertainty |
| --- | ---: | ---: | ---: |
| MLPNext | No | No | No |
| DirectGNN | Yes | No | No |
| HNN / SympNoGraph | No | Yes | No |
| SympGNN no uncertainty | Yes | Yes | No |
| SG-MBRL full | Yes | Yes | Yes |
| SG-MBRL no symplectic integrator | Yes | Hamiltonian only | Yes |
| SG-MBRL no graph edge terms | No | Yes | Yes |
| SG-MBRL no uncertainty bonus | Yes | Yes | No |

### Metrics

- return,
- samples to threshold,
- terminal error,
- energy drift,
- rollout RMSE,
- graph-size transfer,
- model exploitation gap.

### Step-by-Step Procedure

1. Implement each ablation with minimal changes from the full method.
2. Match parameter counts where possible.
3. Use the same training data and planner settings.
4. Run on at least:
   - oscillator stabilization,
   - spring-mass target reaching,
   - one hard particle-control task.
5. Aggregate across seeds.
6. Produce an ablation table and a component-ladder plot.

### Desired Pattern

The paper is strongest if results show:

```text
graph only < graph + symplectic < graph + symplectic + uncertainty
```

The result does not need to hold on every metric, but it should hold on the central control and rollout-stability metrics.

---

## 7. Hard RL Control Experiment

### Claim Tested

SG-MBRL is useful for control, not just toy system identification.

### Tasks

Run at least three:

1. graph oscillator stabilization,
2. target-reaching with controlled particles,
3. formation control,
4. energy-shaping task,
5. Lennard-Jones particle steering,
6. sparse N-body target steering.

The strongest version includes Lennard-Jones particle steering or sparse N-body gravitational steering.

### Metrics

- return vs environment steps,
- success rate,
- terminal task error,
- samples to threshold,
- final return,
- MPC failure rate,
- action magnitude and smoothness.

### Success Thresholds

Target reaching:

```text
(1/|V|) sum_i ||q_i - q_i^*||^2 <= tau_target
```

Formation control:

```text
d_shape(q, q^*) <= tau_shape
```

Stabilization:

```text
(1/|V|) sum_i (||q_i||^2 + ||p_i||^2) <= tau_stable
```

### Step-by-Step Procedure

1. Define each task reward and success threshold before running experiments.
2. Validate that no-control and random-action baselines fail or perform poorly.
3. Run SG-MBRL and strongest MBRL baselines under matched budgets.
4. Run model-free baselines under extended budgets.
5. Evaluate success rate on held-out initial states.
6. Report both return and physical/task metrics.

### Acceptance Pattern

This supports the control claim if SG-MBRL obtains higher success rates or reaches thresholds with fewer real transitions.

---

## 8. Model-Free Baseline Experiment

### Claim Tested

SG-MBRL is sample-efficient relative to tuned model-free RL.

### Baselines

Use standard tuned implementations:

- SAC,
- PPO,
- TD3.

Recommended sources:

- CleanRL,
- Stable-Baselines3,
- RLlib,
- existing verified internal implementations.

Do not rely on the small prototype baselines for final claims.

### Budgets

Use:

```text
N_env = 1k, 5k, 10k, 25k, 50k, 100k, 250k, 500k
```

For harder tasks:

```text
N_env >= 1M
```

### Step-by-Step Procedure

1. Tune each model-free method on validation seeds or validation tasks.
2. Freeze hyperparameters before final test seeds.
3. Use the same observation/action normalization as MBRL methods.
4. Evaluate at the same checkpoints.
5. Report return, terminal error, success rate, and variance across seeds.

### Acceptance Pattern

The desired result is not necessarily that model-free methods never catch up; the important claim is that SG-MBRL reaches useful performance at much lower `N_env`.

---

## 9. Strong PETS and MBPO Baseline Experiment

### Claim Tested

SG-MBRL competes against serious modern MBRL baselines.

### PETS-Style Baseline

Use:

- probabilistic ensemble,
- ensemble size `M=5` or `M=7`,
- trajectory sampling,
- CEM planning,
- same CEM settings as SG-MBRL where possible.

### MBPO-Style Baseline

Use:

- learned dynamics ensemble,
- model rollout horizon:

```text
K_rollout = 1, 5, 10
```

- SAC or TD3 policy optimization on real + generated model data.

### Step-by-Step Procedure

1. Implement PETS with uncertainty-aware trajectory sampling.
2. Match CEM horizon, population, iterations, and action bounds to SG-MBRL.
3. Implement MBPO with short model rollouts and a tuned off-policy learner.
4. Tune each baseline on validation tasks.
5. Freeze hyperparameters for final seeds.
6. Report results in the same headline plots as SG-MBRL.

### Acceptance Pattern

This experiment is successful if SG-MBRL is competitive with or better than PETS/MBPO in the low-data regime and has better physical diagnostics.

---

## 10. Uncertainty Calibration and Model Exploitation

### Claim Tested

Uncertainty estimates are meaningful and reduce planner exploitation.

### Calibration Metrics

Coverage:

```text
P[ ||z_next - z_hat_next|| <= u_alpha(z,a) ] ~= alpha
```

Correlation:

```text
corr(u(z,a), ||z_next - z_hat_next||)
```

Model exploitation gap:

```text
J_hat(a_0:K) - J(a_0:K)
```

Failure rate:

```text
P[ J_hat - J > tau_exploit ]
```

### Conditions

Evaluate calibration under:

1. in-distribution states,
2. larger graphs,
3. unseen topologies,
4. high-energy states,
5. longer MPC horizons.

### Step-by-Step Procedure

1. Fit uncertainty estimates using ensembles, calibrated residuals, conformal intervals, or Bayesian last layer.
2. On held-out transitions, compute prediction error and uncertainty width.
3. Build reliability diagrams for 50%, 68%, 90%, and 95% intervals.
4. Compute expected calibration error for regression intervals.
5. During MPC, save imagined returns and true open-loop returns.
6. Compare optimistic, neutral, and conservative uncertainty objectives.
7. Report exploitation gap and failure rate.

### Acceptance Pattern

The uncertainty claim is supported if calibrated uncertainty:

- has coverage close to nominal,
- correlates with prediction error,
- reduces high-gap model exploitation events,
- improves exploration or robustness.

---

## 11. Planning Horizon Sensitivity

### Claim Tested

SG-MBRL benefits from longer planning horizons more reliably than generic world models.

### Horizons

Use:

```text
K_plan = 1, 5, 10, 20, 30, 50
```

### Methods

Compare:

- SG-MBRL,
- DirectGNN-MPC,
- PETS,
- HGN-MPC,
- MLP-MPC.

### Metrics

- return,
- model exploitation gap,
- imagined rollout energy drift,
- divergence rate,
- action smoothness.

### Step-by-Step Procedure

1. Fix trained models at matched `N_env`.
2. Sweep only the planning horizon.
3. Keep CEM population and iterations fixed, or scale population transparently in a separate compute-matched variant.
4. Evaluate on the same held-out initial states.
5. Save imagined and true open-loop trajectories.
6. Plot return and exploitation gap vs horizon.

### Expected Pattern

Generic models may improve from horizon 1 to 10 and then degrade. SG-MBRL should degrade less or remain useful at longer horizons.

---

## 12. Energy Drift Under Matched One-Step Error

### Claim Tested

Low one-step error does not imply physically valid planning rollouts.

### Protocol

1. Train SG-MBRL and DirectGNN variants over a grid of training epochs, widths, and regularization settings.
2. Select model checkpoints with approximately matched one-step validation error.
3. Keep the selected one-step errors within a narrow tolerance, for example 5-10%.
4. Compare:
   - long rollout RMSE,
   - energy drift,
   - symplecticity defect,
   - phase portrait,
   - planning return.

### Required Figure

Use a matched-error figure:

- x-axis: rollout horizon,
- y-axis: energy drift or rollout RMSE,
- annotate one-step validation RMSE for each model.

### Acceptance Pattern

This directly supports the paper if SG-MBRL and DirectGNN have similar one-step error but SG-MBRL has much better long-horizon energy behavior and planning stability.

---

## 13. Parameter-Matched Model-Capacity Experiment

### Claim Tested

The improvement is due to structure, not merely model size.

### Protocol

For major comparisons run:

1. parameter-matched DirectGNN,
2. larger DirectGNN,
3. parameter-matched SG-MBRL,
4. larger SG-MBRL.

### Required Table

| Model | Parameters | Hidden width | Depth | Graph? | Symplectic? | Uncertainty? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLPNext | TBD | TBD | TBD | No | No | No |
| DirectGNN matched | TBD | TBD | TBD | Yes | No | Optional |
| DirectGNN large | TBD | TBD | TBD | Yes | No | Optional |
| ResidualSympGNN matched | TBD | TBD | TBD | Yes | Yes | Optional |
| SG-MBRL full | TBD | TBD | TBD | Yes | Yes | Yes |

### Step-by-Step Procedure

1. Count parameters programmatically for every model.
2. Choose widths/depths to create matched pairs.
3. Run dynamics and control tests under identical data and planner settings.
4. Report parameter counts in every main result table.
5. Include a larger DirectGNN baseline to test whether capacity alone closes the gap.

### Acceptance Pattern

The structure claim is supported if SG-MBRL remains better on stability, transfer, or sample efficiency when DirectGNN is parameter-matched or larger.

---

## 14. Final Experiment Matrix

| Experiment | Environments | Baselines | Main Metric | Claim Tested |
| --- | --- | --- | --- | --- |
| Dynamics rollout | oscillator, pendulum, spring-mass, LJ | MLP, DirectGNN, HNN, HGN | energy drift, rollout RMSE | physical structure |
| Matched one-step error | oscillator, spring-mass | DirectGNN, SG-MBRL | long-horizon drift | local accuracy is insufficient |
| Graph-size transfer | chains, grids, particles | DirectGNN, HGN, SG-MBRL | return/RMSE vs `|V|` | graph locality |
| Topology transfer | chain to grid/star/random | DirectGNN, SG-MBRL | transfer gap | local law learning |
| Main MBRL sample efficiency | graph oscillator, spring-mass, LJ | PETS, MBPO, DirectGNN-MPC | return vs steps | sample efficiency |
| Hard control | formation, target, LJ steering | PETS, MBPO, SAC, PPO, TD3 | success rate | actual RL improvement |
| Ablations | all main tasks | graph/symp/uncertainty variants | return, drift | component necessity |
| Planning diagnostics | all control tasks | MPC variants | exploitation gap | planning robustness |

---

## 15. Minimal Strong Version

This is enough for a credible submission:

1. dynamics and energy drift on oscillator, pendulum, and spring-mass,
2. graph-size transfer on spring-mass chains and grids,
3. main control on graph oscillator stabilization and target reaching,
4. one hard task: Lennard-Jones steering or N-body steering,
5. SG-MBRL vs DirectGNN-MPC, PETS, MBPO, SAC, PPO, TD3,
6. full ablation matrix,
7. 5 seeds for MBRL, 10 for model-free,
8. parameter-count table,
9. planning-horizon sensitivity,
10. uncertainty calibration.

---

## 16. Ideal Version

This would make the paper very strong:

1. all minimal experiments,
2. graph sizes up to `|V|=128` or `|V|=256`,
3. topology transfer across chain, grid, Erdos-Renyi, and random geometric graphs,
4. tuned PETS and MBPO implementations,
5. tuned SAC/PPO/TD3 implementations,
6. hard Lennard-Jones and sparse N-body control,
7. matched one-step-error experiment,
8. parameter-matched and overparameterized baselines,
9. phase-space visualizations,
10. model-exploitation failure analysis.

---

## 17. Execution Priority

Run experiments in this order:

1. SG-MBRL vs DirectGNN-MPC with identical planner.
2. Graph-size transfer on spring-mass chains and grids.
3. Energy drift under matched one-step error.
4. Full ablation matrix.
5. PETS and MBPO baselines.
6. Hard particle-control task.
7. SAC/PPO/TD3 with tuned implementations.
8. Uncertainty calibration and model exploitation.
9. Planning horizon sensitivity.
10. Parameter-count fairness table.

---

## 18. Paper Integration Plan

### Main Paper Figures

Use at most 5-6 main figures:

1. **Headline sample efficiency:** return vs `N_env` for 3 environments.
2. **Matched planner:** SG-MBRL vs DirectGNN-MPC under identical CEM.
3. **Long-horizon physics:** matched one-step error but lower energy drift.
4. **Graph-size/topology transfer:** performance vs graph size/topology.
5. **Ablation ladder:** graph, symplectic, uncertainty components.
6. **Hard control:** success rates and terminal errors.

### Supplement Figures

Move detailed sweeps to appendix:

- all seeds,
- all environments,
- planning horizon sensitivity,
- calibration diagrams,
- symplecticity defect plots,
- parameter-matched model tables,
- additional phase portraits,
- failure cases.

### Wording Discipline

Use scoped claims:

- Say SG-MBRL helps when the system is graph-local and approximately Hamiltonian over the planning horizon.
- Do not claim universal superiority over DirectGNN.
- Do not claim symplecticity guarantees lower pointwise error.
- Emphasize physical stability, transfer, and reduced model exploitation.

---

## 19. Pre-Submission Checklist

Before finalizing the paper, verify:

- [ ] All main experiments use at least 5 seeds.
- [ ] Headline MBRL plots use 10 seeds if compute permits.
- [ ] Model-free baselines are tuned standard implementations.
- [ ] PETS and MBPO are strong, not toy approximations.
- [ ] SG-MBRL and DirectGNN-MPC use identical CEM settings.
- [ ] Parameter counts are reported.
- [ ] One-step errors are reported alongside long-rollout metrics.
- [ ] Energy drift and divergence thresholds are defined before evaluation.
- [ ] Success thresholds are task-specific and fixed before running.
- [ ] Larger graph transfer is zero-shot unless explicitly labeled few-shot.
- [ ] Topology transfer excludes target topologies from training.
- [ ] Uncertainty calibration is tested in-distribution and out-of-distribution.
- [ ] Raw configs and seeds are saved for reproducibility.

