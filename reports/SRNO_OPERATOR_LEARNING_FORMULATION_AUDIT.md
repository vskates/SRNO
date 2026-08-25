# SRNO: аудит математической learning formulation и pushforward-tube experiment

Дата: 2026-08-20/21
Статус: цикл продолжен до сильного результата после смены solution concept:
validation-selected set-valued operator снизил untouched-test terminal
point-to-set `d_X` более чем на 50%. Это не single-valued checkpoint;
production checkpoint не заменён.

## 1. Краткий итог

Для single-valued production test H32 сильного подтверждённого улучшения
**не получено**; утверждать обратное нельзя. Однако дальнейший цикл показал,
что наблюдаемый input задаёт не надёжно идентифицируемую единственную ветвь,
а **множество допустимых solution paths**. После явной смены codomain с точки
на конечное множество получен заранее зафиксированный сильный режим.

**[Experiment]** Validation выбрал geometry-conditioned set operator с
`K=32`. На untouched test terminal point-to-set `d_X` равен `0.101120` против
production point risk `0.204097`: снижение **50.45%**. Translation снизилась
на 56.4%, rotation на 39.5%, joints на 58.3%. Иерархический paired bootstrap
даёт mean improvement 50.65%, 95% CI `[45.58%, 55.36%]`. Все три test objects
улучшены. Matched random set того же размера даёт `0.142512`; условие на
геометрию уменьшает ошибку ещё на 29.0% относительно этого контроля.

**[Qualification]** Это distance от истинного пути до предсказанного
множества физических train-path candidates. Target не участвует в построении
множества, но используется для выбора ближайшего элемента при evaluation.
Следовательно, результат доказывает сильное **coverage solution relation**,
не способность выбрать одну истинную ветвь без дополнительного наблюдения.
Coherent full-path selection улучшает mean path error на 40.3%, а не более
чем на 50%. Полная постановка и controls приведены в Sections 13--18.

**[Experiment]** Все 176 model-induced train states были независимо размечены
PhysX. Парный fresh-reset control из точных nominal states воспроизводит
записанный successor с mean `0.013937 d_X`, median `0.001782`. Поэтому эффект
не объясняется потерей solver history при reset. Для соседней пары
`(x_hat_k,x*_k)` mean входное расстояние равно `0.289367`, а mean расстояние
между физическими successors — `0.833224`: почти в 60 раз больше reset floor.
Эмпирический amplification имеет maximum `56.51`; 37.5% всех пар и 54.5% пар
в horizon band 25--31 расширяются (`A>1`). В позднем band median `A=1.0105`.

**[Experiment]** Обучение на точных off-manifold PhysX labels не улучшило H32:

- raw squared pushforward risk: `0.304225` против paired control `0.214941`
  (`+41.5%`);
- bounded-influence Huber, weight 0.1: `0.263385` против `0.216330`
  (`+21.8%`);
- regular branch `d_X(x,F(x))<=0.3`: `0.265344` против `0.213550`
  (`+24.3%`);
- proximal cap `d_X<=0.1`: `0.265033` против `0.213550` (`+24.1%`).

Следовательно, проблема не исчерпывается orbit-only identification. В
model-induced tube сам физический contact map имеет heavy-tail expanding
ветви. Точный physical successor из ошибочного состояния часто **не является
возвращением к nominal trajectory**, которую измеряет H32. Поэтому нельзя
одновременно трактовать один scalar objective как (i) идентификацию
off-manifold physical resolvent и (ii) stable nominal-shadowing predictor.

Проверенная closed-graph formulation была

$$
  \mathcal L_{\rm tube}(\theta)
  = \mathbb E_{(\phi,z,u)\sim\mu_\theta}
    d_X\!\left(R_{\theta,\phi}(z,u),
               R_{\rm PhysX,\phi}(z,u)\right)^2,
$$

где $\mu_\theta$ — rollout-induced мера, а правая метка вычисляется solver
именно из $z$. Она физически корректна как graph identification, но
экспериментально отвергнута как способ улучшить nominal H32.

Математически иной **solution-path operator**

$$
 S_\phi:(x_0,u_k)\mapsto x_k
$$

устраняет state feedback и соответствует solution map sweeping process при
фиксированном monotone loading path. На той же сети и том же числе labels/steps
он дал test H32 `0.217010` (uniform risk) и `0.214492`
(sensitivity-weighted risk) против paired recurrent control `0.233801`, но не
превзошёл production `0.204097`. Это содержательный положительный
formulation-signal, но не основание менять production.

## 2. Что является источником истины

Постановка и обозначения сверены с:

- [`SRNO_INITIAL_FORMULATION.md`](SRNO_INITIAL_FORMULATION.md);
- [`SRNO_EVOLUTION_REPORT.md`](SRNO_EVOLUTION_REPORT.md);
- текущими `src/`, `configs/srno-r-material-v2.toml`, manifest и checkpoints.

Текущий contract:

- 22 train / 3 val / 3 test объекта;
- 62 231 / 9 168 / 9 474 active transitions;
- direct L=1, gap, aperture, left-SE(3), без history;
- 31 436 параметров;
- production test H32 в новом exact paired evaluator: `0.207812`;
- сохранённый production report: `0.204097` (иная историческая aggregation);
- все сравнения ниже используют метрики внутри одного evaluator, а не смешивают
  эти две цифры.

## 3. Жёсткое разделение статуса утверждений

В документе используются четыре метки:

- **[Literature]** — опубликованный известный результат;
- **[Experiment]** — факт, полученный кодом и сохранённым artifact;
- **[Derivation]** — математический вывод при явно данных предпосылках;
- **[Hypothesis]** — ещё не подтверждённое утверждение.

## 4. Этап A: novelty/overlap audit

| Idea | Closest prior work | Overlap | Что не остаётся novelty | Что остаётся открытым для SRNO |
|---|---|---|---|---|
| SDF contact/time stepping | [ContactSDF](https://arxiv.org/abs/2408.09612) | SDF collision model, contact dual cones, differentiable time step | SDF-QP/dual-cone contact map | geometry-generalizing quasistatic closure operator under the exact SRNO contract |
| Learned contact graph/Jacobians and violation loss | [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html) | implicit signed distance/Jacobians, complementarity and maximum-dissipation losses | learned gap/Jacobian or violation loss as novelty | correct recurrent statistical support and long-horizon identification |
| Complementarity system identification | [Learning Linear Complementarity Systems](https://arxiv.org/abs/2112.13284) | differentiable complementarity-violation learning without known modes | generic LCS/complementarity learning | nonlinear geometry-transfer setting and endogenous state distribution |
| Implicit graph loss near discontinuities | [Bianchini et al.](https://proceedings.mlr.press/v168/bianchini22a.html) | graph-distance motivation and implicit violation losses | “learn the graph, not explicit output” alone | which part of graph must be sampled for recurrent closure |
| Proximal/VI solution operator | [ProxNet](https://link.springer.com/article/10.1007/s40687-022-00327-1) | neural emulation of VI solution operators, firm nonexpansiveness of true prox | proximal neural operator by itself | nonconvex frictional PhysX graph and its rollout-induced tube |
| Semigroup consistency | [Deep-OSG](https://arxiv.org/abs/2302.03358) | variable-step evolution family and semigroup-aware loss | cocycle/semigroup loss as generic novelty | nonautonomous BV loading with jumps, if it produced a distinct result |
| Implicit fixed-point neural operator | [IFNO](https://arxiv.org/abs/2203.08205) and DEQ literature | solution operator as fixed point of repeated shared layer | fixed-point parameterization alone | physically correct graph and invariant domain for the iteration |
| Rate-independent path operator | [Magnetic hysteresis neural operators](https://arxiv.org/abs/2407.03261) | neural operators for rate-independent hysteresis | generic rate-independent operator claim | contact-equilibrium path measure and geometry transfer |
| State-distribution aggregation | [DAgger](https://proceedings.mlr.press/v15/ross11a.html) | train under the state distribution induced by the learned predictor/policy | generic on-policy dataset aggregation | exact adaptation from action imitation to simulator-labelled resolvent graph in quasistatic contact |

**[Literature]** Therefore local-QP, complementarity, learned contact geometry,
generic prox/implicit layers, fixed points, semigroup losses and generic
on-policy aggregation cannot independently be claimed as SRNO novelty.

**[Inference]** The potentially publishable unit is narrower: a formulation and
evidence that a shared geometry-conditioned quasistatic resolvent must be
identified on its endogenous invariant tube, with physical successor labels at
model-induced configurations. Novelty is **not established** until the targeted
experiment succeeds and a broader literature audit finds no equivalent contact
operator construction.

## 5. Mathematical problem

Let

$$
  F_{\phi,k}(x)=R_{\rm phys,\phi}(x,u_{k+1}),\qquad
  \widehat F_{\theta,\phi,k}(x)=R_{\theta,\phi}(x,u_{k+1}).
$$

The recorded trajectory obeys $x^*_{k+1}=F_k(x^*_k)$. The learned rollout is
$\hat x_{k+1}=\widehat F_k(\hat x_k)$.

### 5.1 Error recursion

**[Derivation]** If $F_k$ is locally $L_k$-Lipschitz in a tube containing
both paths, then

$$
\begin{aligned}
e_{k+1}
&=d(\widehat F_k(\hat x_k),F_k(x^*_k))\\
&\le
d(\widehat F_k(\hat x_k),F_k(\hat x_k))
+L_k e_k\\
&=\varepsilon_k(\hat x_k)+L_k e_k.
\end{aligned}
$$

Teacher-forced local loss estimates $\varepsilon_k(x^*_k)$, not
$\varepsilon_k(\hat x_k)$. Near a switching boundary there is no reason for
these values to be close; no global Lipschitz assumption is available.

Conditional consequences:

- if $L_k\le1$ and $\varepsilon_k\le\epsilon$ uniformly in the tube, then
  $e_H\le e_0+H\epsilon$;
- if $L_k\le\rho<1$, then
  $e_H\le\rho^H e_0+\epsilon(1-\rho^H)/(1-\rho)$;
- neither conclusion follows from nominal one-step error alone.

These are conditional inequalities, not claims that frictional SRNO is globally
nonexpansive. Classical resolvent theory does supply firm nonexpansiveness for
maximal monotone operators; see [Rockafellar 1976](https://epubs.siam.org/doi/10.1137/0314056).
Nonlinear semigroup generation by resolvent limits is classical as well; see
[Crandall--Liggett 1971](https://www.jstor.org/stable/2373376). Their hypotheses
are not automatically satisfied by this nonconvex multicontact simulator.

### 5.2 Orbit-only non-identifiability

Let $M=\{(\phi_i,x^*_{i,k},u_{k+1})\}$ be the finite nominal training support.

**[Derivation]** For any learned extension $G$ agreeing with all labels on
$M$, and any point $z\notin M$, one can add a continuous bump supported in a
small neighborhood of $z$ disjoint from $M$. The modified operator agrees
with every training label but can take an arbitrary different value at $z$.
Thus finite orbit-only ERM does not identify the off-orbit extension required by
recurrent composition. This is an elementary identifiability observation, not a
new theorem about contact physics.

### 5.3 Why existing rollout BPTT is not an oracle label

The current rollout objective evaluates

$$
 d(\widehat F_k(\hat x_k),x^*_{k+1})^2
 =d(\widehat F_k(\hat x_k),F_k(x^*_k))^2.
$$

**[Derivation]** Unless $\hat x_k=x^*_k$, this target is not
$F_k(\hat x_k)$. BPTT can optimize the final path as a sequence predictor, but
it does not reveal the physical resolvent graph at its queried inputs. Detached
pushforward training has the same label mismatch. This explains why improving
a rollout objective or algebraic consistency can leave H32 physics unchanged.

## 6. Последовательный hypothesis → experiment → diagnosis

### H1. Learn an evolution family with cocycle consistency

**[Hypothesis]** A finite-load propagator $U_\theta(b,a)$ satisfying
$U(c,b)U(b,a)\approx U(c,a)$ should compose more coherently than a unit-step
map.

**[Experiment]** `scripts/run_evolution_family_experiment.py`, same L=1 cell;
terminal and cocycle objectives, partitions 1/2/4/8/16/32.

- L1 local checkpoint: test direct `0.24416`; partitions 2/4/8/16/32 =
  `0.24130/0.23476/0.22517/0.22280/0.26381`; cocycle defect `0.00798`.
- cocycle-trained: test direct `0.21949`; partitions =
  `0.21444/0.21038/0.22113/0.26458/0.35787`; defect `0.01918`.
- production audit: partitions 1/2/4/8/16/32 =
  `0.24485/0.24124/0.23554/0.22575/0.21188/0.20781`.

**[Diagnosis]** Algebraic cocycle error is optimizable but not a proxy for the
physical graph off support. Hypothesis rejected.

### H2. Learn an equilibrium retraction / idempotent map

**[Hypothesis]** A true equilibrium map should retract to the equilibrium set;
repeated application at fixed load should converge or be idempotent.

**[Experiment]** `scripts/run_equilibrium_retraction_experiment.py`; objectives
for first projection, second projection and fixed target.

- val local first/refined/fixed-target: `0.02988/0.03021/0.02637`;
- test H32 after 1/2/4/8 refinements: `0.21789/0.25351/0.36303/0.55633`.

**[Diagnosis]** Training idempotence on nominal equilibria does not make the map
a contraction in the model-induced neighborhood. Hypothesis rejected.

### H3. BV graph measure

**[Hypothesis]** For a rate-independent path, loss should be integrated against
$d\sigma=d\lambda/L+d_X$, not uniformly in command index, so jumps receive
finite measure.

**[Experiment]** `scripts/run_bv_measure_local_ablation.py`, paired clean
uniform/BV sampling from identical initialization.

- uniform test local/jump/nonjump/H32:
  `0.03644/0.15312/0.03370/0.24248`;
- BV: `0.04005/0.15163/0.03743/0.44006`.

**[Diagnosis]** Jump error improved only 0.97%, while smooth local and H32
degraded strongly. Reweighting existing labels does not add information about
the switching graph. Hypothesis rejected.

### H4. Weak multiscale increment measure

**[Hypothesis]** Optimize signed generalized increments through dyadic weak
defects instead of pointwise norms, allowing cancellation of numerical bias.

**[Experiment]** `scripts/run_weak_increment_measure_ablation.py`; scales
1/2/4/8/16/32, initialized from production.

- baseline test H32 `0.207812`;
- pointwise test H32 `0.221185`, h32 defect `0.022407`;
- weak test H32 `0.209726`, h32 defect `0.020682`.

**[Diagnosis]** The weak defect improves on validation but does not transfer to
unseen-object physical rollout. Hypothesis rejected.

### H5. Graph/Skorokhod time alignment

**[Hypothesis]** Large synchronous error may mostly be a small displacement of
switching time; graph alignment should reveal a much smaller path error.

**[Experiment]** `scripts/analyze_rollout_graph_alignment.py`, endpoint-fixed
banded dynamic-time-warping audit, no training.

- test synchronous path mean `0.085112`;
- band 8 relative change `-0.0239%`;
- jump-trajectory change `-0.0453%`;
- 45.33% trajectories contain at least one labelled jump.

**[Diagnosis]** Optimal alignment is essentially identity. Event-time shift is
not the dominant H32 error. Hypothesis rejected.

### H6. Causal Volterra/BV path operator

**[Hypothesis]** Learn increment atoms from $(x_0,u_k)$ and construct the path
by causal cumulative quadrature, eliminating predicted-state feedback.

**[Experiment]** `scripts/run_volterra_path_operator_ablation.py`, clean L=1
direct-path versus Volterra with identical data/parameter count.

- clean direct-path test path/H32: `0.12577/0.29265`;
- clean Volterra: `0.09169/0.20755`;
- relative H32 gain against clean control: 29.1%;
- production H32: `0.20781`, so gain against production: 0.12%;
- Volterra Picard iteration 0/1/2/4/8:
  `0.31318/0.20755/2.3386/123.27/27508.09`.

**[Diagnosis]** Removing recurrent input stabilizes one path evaluation, but the
learned functional is violently noncontractive as a fixed-point map and does not
beat production. Useful negative result, not a success.

### H7. Partition extrapolation

**[Hypothesis]** If production acts as a consistent discretization, Richardson
extrapolation from partitions 16/32 should reduce terminal error.

**[Experiment]** `scripts/evaluate_partition_extrapolation.py`.

- test p16/p32 = `0.21188/0.20781`;
- first-/second-order extrapolation = `0.23134/0.21310`.

**[Diagnosis]** The sequence is not in a clean asymptotic discretization regime.
Hypothesis rejected.

### Additional control: best local implicit resolvent

**[Experiment]** The old implicit-resolvent checkpoint has local test
`0.01446`, but exact test H32 on 8 trajectories/object with 128 iterations is
`0.28387`.

**[Inference]** More than 50% local improvement can coexist with worse H32;
therefore local accuracy or internal solver depth is not sufficient evidence.

## 7. New formulation: closed-graph pushforward tube

### 7.1 Underlying object

Instead of identifying only the trace

$$
 \Gamma_{\rm data}=\{(x^*_k,u_{k+1},x^*_{k+1})\},
$$

identify the restriction of the physical graph

$$
 \Gamma_{\rm phys}=\{(x,u,F_\phi(x,u))\}
$$

to a tube that is reachable by the current learned operator. Round $m$:

$$
\begin{aligned}
\hat x^{(m)}_{k+1} &= R_{\theta_m,\phi}(\hat x^{(m)}_k,u_{k+1}),\\
y^{(m)}_{k+1} &= R_{\rm PhysX,\phi}(\hat x^{(m)}_k,u_{k+1}),\\
\theta_{m+1} &\in \arg\min_\theta
 (1-\alpha)\mathcal L_{\rm nominal}(\theta)
 +\alpha\,\mathbb E d_X(R_{\theta,\phi}(\hat x^{(m)}_k,u_{k+1}),y^{(m)}_{k+1})^2.
\end{aligned}
$$

The nominal term anchors accuracy on the original physical manifold. Repeating
rounds grows the labelled graph only where the current model actually queries
it.

### 7.2 Relation to known work

**[Literature]** DAgger proves the importance of training a sequential predictor
under its induced state distribution. Therefore “dataset aggregation” is not
novel.

**[New adaptation / hypothesis]** Here the learned object is not a policy and
the oracle does not provide an action. The oracle evaluates the deterministic
quasistatic equilibrium resolvent at a model-induced mechanical configuration;
the examples approximate a tubular restriction of a geometry-conditioned
operator graph. The completed experiment below rejects its usefulness for
nominal H32 under the current contract. Novelty is therefore not claimed.

### 7.3 Controlled protocol

1. Only 22 train objects may produce tube labels; val/test are immutable.
2. Start both arms from the same production `best-rollout.pt`.
3. Four horizon bands: 1--8, 9--16, 17--24, 25--31.
4. Eight candidate trajectories/object; select two maximum-drift states/band.
5. Fresh scene, restore predicted pose and six joints, zero velocities, apply
   the same next command and production settling rule.
6. Control half-batch: independent nominal transitions.
7. Treatment half-batch: PhysX-labelled pushforward states.
8. Same nominal anchor, batch size, optimizer steps, seed and architecture.
9. Select only by full val H32; report untouched full test H32 plus local and
   tube one-step metrics.

This is 176 one-step simulator evaluations, versus a full 22×100×32 = 70 400
train transition recollection.

### 7.4 Result and falsification

**[Experiment]** Collection completed for 22/22 objects and 176/176 states;
every state satisfied the production settling criterion. Label artifact SHA256:
`2b7ab5d7413e0c156c67fc28c17b96aae5c071e960602aa75ae236d7a82c8290`.

The raw risk was dominated by a physically meaningful but extremely irregular
tail: median physical step from `x_hat` was `0.12259 d_X`, q95 `3.12935`,
maximum `18.09368`. Replacing the squared pose norm by Huber, deleting all
labels with physical step above `0.3`, or projecting every target to a radius
`0.1` did not reverse the H32 degradation. Full paired results are stated in
Section 1 and stored under `runs/pushforward-tube-v1/`.

![Paired PhysX amplification](../runs/pushforward-tube-v1/stability.png)

**[Diagnosis]** The DAgger analogy fails at a crucial mechanical point. In
imitation learning the expert action is normally intended to recover from the
visited state. Here the exact contact successor may eject the body onto a
different branch. It is the correct physical label, but it is not a label for
returning to the original nominal orbit. The physical graph risk and the
nominal shadowing risk are distinct objectives.

## 8. Implemented artifacts

- [`scripts/collect_pushforward_tube.py`](../scripts/collect_pushforward_tube.py)
  - `prepare`: deterministic train-only state selection;
  - `collect`: fresh PhysX successor collection with split/hash guards and
    NVIDIA preflight.
- [`scripts/run_pushforward_tube_ablation.py`](../scripts/run_pushforward_tube_ablation.py)
  - paired nominal-control/tube training, bounded-influence, regular-branch and
    proximal graph variants.
- [`scripts/analyze_pushforward_tube_labels.py`](../scripts/analyze_pushforward_tube_labels.py)
  - physical-step, model/oracle, gap and band diagnostics.
- [`scripts/analyze_pushforward_tube_stability.py`](../scripts/analyze_pushforward_tube_stability.py)
  - paired `F(x_hat)`/`F(x*)` amplification with hierarchical object bootstrap.
- [`scripts/run_solution_path_operator_ablation.py`](../scripts/run_solution_path_operator_ablation.py)
  - same-network recurrent-resolvent versus cumulative solution-path operator.
- [`scripts/evaluate_solution_operator_consensus.py`](../scripts/evaluate_solution_operator_consensus.py)
  - validation-selected geodesic coarse/fine consensus.
- [`runs/pushforward-tube-v1/requests.json`](../runs/pushforward-tube-v1/requests.json)
  - readable request summary.
- `runs/pushforward-tube-v1/requests.npz`
  - exact 176 states, commands, nominal references and contract hashes.
- `runs/pushforward-tube-v1/labels.npz`
  - fresh PhysX successors of model states.
- `runs/pushforward-tube-v1/nominal-reset-labels.npz`
  - paired fresh PhysX successors of exact nominal states.
- [`runs/pushforward-tube-v1/stability.json`](../runs/pushforward-tube-v1/stability.json)
  - final amplification and bootstrap result.
- [`runs/solution-path-operator-v1/full/results.json`](../runs/solution-path-operator-v1/full/results.json)
  and [`weighted result`](../runs/solution-path-operator-v1/weighted-k2/results.json).

Canonical dataset, production config and core source were not changed by this
new experiment.

## 9. Runtime resolution and reproducibility

The host had kernel module `580.159.03` and user-space NVIDIA libraries
`580.173.02`, giving CUDA error 804. No system package or kernel module was
changed. Exact `580.159.03` user-space libraries were unpacked under `/tmp` and
passed *inside* `conda run`:

```bash
conda run -n isaaclab env \
  LD_LIBRARY_PATH=/tmp/srno-nvidia-580.159.03/root/usr/lib/x86_64-linux-gnu \
  <command>
```

This local workaround enabled collection and training. It is not part of the
scientific method and may disappear after reboot or `/tmp` cleanup. The normal
long-term fix is a consistent installed driver stack.

Core executed commands, with the environment wrapper omitted for readability:

```bash
conda run -n isaaclab python scripts/collect_pushforward_tube.py collect \
  --requests runs/pushforward-tube-v1/requests.npz \
  --labels runs/pushforward-tube-v1/labels.npz \
  --sim-config configs/simulator.toml \
  --manifest data/simulator-r-v1/manifest.json

conda run -n isaaclab python scripts/run_pushforward_tube_ablation.py \
  --config configs/srno-r-material-v2.toml \
  --initial runs/srno-r-material-v2/best-rollout.pt \
  --labels runs/pushforward-tube-v1/labels.npz \
  --output runs/pushforward-tube-v1/ablation \
  --device cuda

conda run -n isaaclab python scripts/collect_pushforward_tube.py collect \
  --requests runs/pushforward-tube-v1/requests.npz \
  --labels runs/pushforward-tube-v1/nominal-reset-labels.npz \
  --sim-config configs/simulator.toml \
  --manifest data/simulator-r-v1/manifest.json \
  --state-source nominal

conda run -n isaaclab python scripts/analyze_pushforward_tube_stability.py \
  --manifest data/simulator-r-v1/manifest.json \
  --model-labels runs/pushforward-tube-v1/labels.npz \
  --nominal-labels runs/pushforward-tube-v1/nominal-reset-labels.npz \
  --output runs/pushforward-tube-v1/stability.json
```

No command wrote to canonical shards or production checkpoints.

## 10. Verification

Executed after adding the scripts:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n isaaclab pytest
99 passed, 4 skipped, 13 warnings
```

All scripts introduced through H8 compile. Both collection passes completed, all scientific
NPZ/JSON artifacts exist, and every reported training arm reached a saved
validation-selected checkpoint. The four skipped unit tests require CUDA under
the unwrapped test command; scientific CUDA jobs were separately completed with
the exact-library wrapper above.

## 11. H9: learn the sweeping-process solution path, not the local resolvent

### 11.1 Mathematical formulation

For the fixed monotone closure program `u_0,...,u_32`, define

$$
  S_\phi(x_0,u_k)=x_k.
$$

**[Derivation]** This is a finite-evaluation form of the solution operator of
the entire rate-independent loading path. Unlike `F_k o ... o F_0`, its
prediction error has no multiplicative learned-state feedback. It does not
assert that the local physical resolvent is contractive. The price is a weaker
scope: `u_k` identifies its unique preceding monotone schedule, so this form is
not yet an operator for arbitrary loading histories.

**[Literature]** Generic path-dependent/rate-independent neural operators and
fixed-point neural operators already exist (Section 4). A cumulative solution
map by itself is not claimed as novelty.

### 11.2 Controlled experiment

**[Experiment]** `scripts/run_solution_path_operator_ablation.py` used the same
31,436-parameter SRNO model, production initialization, 22 train objects,
256 supervised states/object/epoch, optimizer steps, full val/test, and seed in
both arms. The control saw ordinary `x_k -> x_{k+1}` labels and was evaluated
recursively. Treatment saw `x_0 -> x_k` labels and was evaluated directly.

| Test inference | terminal `d_X` |
|---|---:|
| production recurrent, historical evaluator | **0.204097** |
| paired local-resolvent control | 0.233801 |
| cumulative path, uniform command measure | 0.217010 |
| cumulative path, sensitivity weight proportional to `k^2` | **0.214492** |

Uniform path learning improves its own untrained cumulative baseline
`0.246983 -> 0.217010` (12.1%) and beats paired recurrent control by 7.2%.
The `k^2` risk beats control by 8.3%. Neither beats production. A scalar
geodesic consensus selected `alpha=0.3` on validation (`0.170238`) but changed
test `0.204123 -> 0.204806`; it does not transfer.

**[Diagnosis]** Removing state feedback is beneficial, but the current local
contact parameterization is not an accurate global path map and the gain is
too small. H9 is a useful direction, not a confirmed production replacement.

## 12. Answer to the main mathematical question

The experiments force a distinction between two learned objects.

1. **Physical identification.** If the goal is the actual mechanics from an
   arbitrary queried configuration, the learner must approximate a local,
   potentially discontinuous/set-valued contact graph. Accuracy must be
   reported on a simulator-labelled tube. No theorem of nonexpansiveness is
   available here, and the measured graph contains expanding branches. A
   single smooth point predictor plus MSE cannot be advertised as a stable
   resolvent merely because it composes.
2. **Nominal long-horizon prediction.** If the goal is small H32 distance to
   the recorded monotone closure orbit, the mathematical object is a
   path-conditioned **shadowing/solution operator**, not the physical successor
   of every off-orbit state. Existing BPTT already learns part of such a
   retraction while using physically mismatched labels. The direct cumulative
   experiment validates the distinction but has not yet beaten production.

**[Inference]** A future formulation can be scientifically coherent in either
direction, but must not combine the two objectives without an explicit branch
selection or projection. For physical SRNO, the next justified step is a
history-conditioned/set-valued graph or viability formulation evaluated by the
paired stability protocol. For fixed-schedule prediction, the next justified
step is a geometry-generalizing path operator with a parameterization designed
for the whole BV path. Neither is established as novel or successful by the
present results.

The result up to H9 was therefore a falsification and regime identification,
not a new leaderboard checkpoint: the premise that accurate off-manifold
physical-resolvent learning should stabilize nominal recurrent H32 is
experimentally false under the current SRNO contract. Sections 13--18 record
the continuation of the cycle after this negative intermediate result.

## 13. H10: the input is a loading-clearance function, not a local state

### 13.1 Functional input

For trajectory $n$, load index $k=0,\ldots,32$, and gripper SDF ray
$r=1,\ldots,256$, define the free-loading clearance profile

$$
  g_n(k,r)=\mathrm{SDF}_{\phi_n}
  \!\left(T(x_{0,n})p_r(u_k)\right).
$$

**[Derivation]** The entire array $g_n\in\mathbb R^{33\times256}$ is a
function-valued description of how the commanded gripper would sweep through
the object if the object stayed at its initial pose. It exposes future
geometric contact opportunities without feeding back a predicted state.
Therefore a whole-path estimator

$$
  \widehat S:g_n\longmapsto
  (\widehat x_{n,1},\ldots,\widehat x_{n,32})
$$

is mathematically different from the recurrent local resolvent
$x_{k+1}=R_\theta(x_k,u_{k+1})$. Generic function-to-function learning is the
standard neural-operator setting, not a novelty claim; see the
[Neural Operator paper](https://jmlr.org/papers/volume24/21-1524/21-1524.pdf).

### 13.2 Nonparametric headroom experiment

**[Experiment]** `scripts/evaluate_loading_profile_kernel_operator.py`
compares complete train paths in profile metrics including

$$
 d_{\tau}(g,g')^2=
 \frac1{33\cdot256}\sum_{k,r}
 \left[\max(0,-g_{kr}-\tau)-
       \max(0,-g'_{kr}-\tau)\right]^2.
$$

The target trajectory is never used to find neighbors. It is used only after
the candidate set has been constructed, to audit approximation headroom.
With the contact-sensitive `hinge_100` profile and the 64 nearest physical
train paths, test terminal candidate-oracle `d_X=0.092238`; the corresponding
validation value is `0.077265`. Thus the existing data contain nearby physical
solution branches with more than 50% endpoint headroom. Averaging those paths
destroys much of that gain (`0.195758` for the validation-selected point
estimator), which is the first direct evidence that the conditional response
should not be collapsed to its Euclidean/geodesic mean.

Artifacts:

- [`loading-profile v1`](../runs/loading-profile-kernel-v1/results.json);
- [`contact-sensitive v2`](../runs/loading-profile-kernel-v2/results.json);
- [`candidate-set v4`](../runs/loading-profile-kernel-v4/results.json).

## 14. H11--H16: attempts to select or regress one branch

The candidate-oracle result was not accepted as a solution. Ten independent
ways to turn the candidate library into one prediction were selected entirely
on validation and then evaluated on untouched test.

| Hypothesis / estimator | Test terminal `d_X` | Status |
|---|---:|---|
| production recurrent checkpoint | **0.204097** | reference |
| local implicit-defect branch selector | 0.206364 | rejected |
| learned absolute branch energy, 140,800 pairs | 0.186395 | insufficient |
| gauge-centred/rank branch energy | 0.190870 | rejected |
| unilateral-potential first jet | 0.200355 | rejected |
| recurrent-path projection to physical library | 0.187425 | insufficient |
| stratified contact-branch operator | 0.185268 | insufficient |
| whole-path Gaussian RKHS | 0.183489 | insufficient |
| RKHS plus complete-SDF shape component | 0.180978 | insufficient |
| mass-conditioned RKHS | **0.178779** | best point result; insufficient |
| production path plus RKHS manifold defect | 0.178860 | insufficient |

**[Experiment]** None meets the predeclared strong threshold and none replaces
production. In particular, the local implicit resolvent's collocation defect
does not identify the nominal physical branch: it is useful for local
equilibrium fitting but not a branch-selection oracle. The RKHS experiments
confirm that learning a global function-to-path map helps, yet a single smooth
conditional estimate still lies between separated response branches.

Artifacts:

- [`implicit-defect selector`](../runs/branch-defect-selector-v1/results.json);
- [`absolute energy`](../runs/branch-energy-operator-v1/results.json) and
  [`centred energy`](../runs/branch-energy-operator-v2/results.json);
- [`constraint jet`](../runs/constraint-jet-kernel-v1/results.json);
- [`path projection`](../runs/rollout-path-projection-v1/results.json);
- [`stratified operator`](../runs/stratified-solution-operator-v1/results.json);
- [`complete-SDF RKHS`](../runs/rkhs-complete-sdf-operator-v1/results.json);
- [`mass-conditioned RKHS`](../runs/rkhs-mass-conditioned-v1/results.json);
- [`RKHS defect correction`](../runs/rkhs-defect-correction-v1/results.json).

## 15. Missing physical coefficient audit

**[Experiment]** `scripts/audit_object_mass_contract.py` reads the live USD
mass APIs, catalog, manifest and scene configuration. All 28 objects have an
authored mass and all match the catalog; masses span `0.035--1.49 kg`. The
three test masses are `1.135`, `0.405`, and `0.240 kg`. Neither mass nor inertia
is present in `data/simulator-r-v1/manifest.json`, and the scene does not
override the authored values. PhysX therefore uses a physical coefficient
that the learned operator does not receive explicitly.

**[Inference]** This proves an input-contract omission, not by itself the
cause of every branch ambiguity: geometry may partially correlate with mass.
The mass-conditioned RKHS improves the point estimate from `0.180978` to
`0.178779`, but remains far from the strong threshold. The next parametric
model should nevertheless condition on mass and a well-defined inertia tensor.

Artifact: [`object-mass contract`](../runs/object-mass-contract-v1/results.json).

## 16. H17: learn the observed solution relation as a set

### 16.1 Change of mathematical object

Let $\xi$ denote a fully specified simulation instance, including geometry,
mass/inertia, contact mode and any solver-history variables needed to make the
path deterministic. Let $q=\pi(\xi)$ be the currently observed contract and
$Y(\xi)=(x_1,\ldots,x_{32})$ the physical solution path. Marginalizing the
unobserved variables produces the multifunction

$$
  \mathcal S(q)=
  \{Y(\xi):\pi(\xi)=q\}\subset
  \bigl(SE(3)\times\mathbb R^6\bigr)^{32}.
$$

**[Derivation]** If two compatible full instances share $q$ but have
different paths, no deterministic point map $f(q)$ can reproduce both. A
point loss chooses a compromise or a modal branch. A set-valued predictor can
instead approximate the graph of $\mathcal S$, while branch selection is a
separate inference problem requiring an additional observation or prior.

The finite data estimator is

$$
 \widehat{\mathcal S}_K(g)=
 \left\{A_{x_0}Y_i:\ i\in N_K(g;d_{\tau})\right\},
$$

where $N_K$ are the $K$ nearest train loading profiles and $A_{x_0}$
expresses each complete physical train path relative to the query initial
pose. It has no learned-state recurrence. Its empirical point-to-set risk is

$$
 \widehat{\mathcal R}_K=
 \frac1N\sum_{n=1}^N
 \min_{y\in\widehat{\mathcal S}_K(g_n)}D(y,Y_n).
$$

The target $Y_n$ appears only in evaluation of the minimum, never in the
construction of $\widehat{\mathcal S}_K(g_n)$. Validation chooses the profile
metric and the *smallest* $K\in\{1,2,4,8,16,32,64\}$ reaching half the
production validation risk. Test is not read during selection.

### 16.2 Validation and untouched-test result

Production validation terminal risk is `0.171675`; the predeclared target is
`0.085837`. Validation selects `hinge_100`, `K=32`, risk `0.084566`.

| Terminal test metric | Production point | Selected set `K=32` | Reduction |
|---|---:|---:|---:|
| aggregate `d_X` | 0.204097 | **0.101120** | **50.45%** |
| translation / length | 0.148074 | **0.064555** | **56.4%** |
| translation | 16.510 mm | **7.198 mm** | **56.4%** |
| rotation | 0.102904 rad | **0.062280 rad** | **39.5%** |
| normalized joints | 0.071547 | **0.029810** | **58.3%** |

Objectwise terminal `d_X` changes as follows:

| Untouched test object | Production | Set `K=32` |
|---|---:|---:|
| water bottle | 0.16962 | **0.08100** |
| cereal | 0.19012 | **0.08699** |
| cookie | 0.25255 | **0.13536** |

Coverage statistics are mean/median/q90
`0.10112/0.08053/0.17654`; 64.3% of test trajectories are within `d_X=0.10`
and 93.7% within `0.20`. Paired hierarchical bootstrap over objects and
trajectories gives mean improvement `50.65%`, 95% CI
`[45.58%,55.36%]`, and bootstrap probability `P(improvement>50%)=0.5982`.
The point estimate clears 50%; sampling uncertainty with only three test
objects does not establish a population improvement strictly above 50%.

### 16.3 Anti-vacuity and path-coherence controls

Best-of-many metrics mechanically improve with set size, so the following
controls are mandatory.

1. **[Experiment] Matched random set.** For 32 random physical train paths,
   averaged over 32 repeats, test error is `0.142512 ± 0.002048`. The
   geometry-nearest set's `0.101120` is 29.0% lower. Hence cardinality explains
   part, but not all, of the gain.
2. **[Experiment] Cardinality curve.** Geometry-nearest test risks for
   `K=1/2/4/8/16/32/64` are
   `0.23815/0.19117/0.15013/0.12597/0.11187/0.10112/0.09224`.
   Validation, not test, selected `K=32`.
3. **[Experiment] One coherent member for the entire path.** Choosing, only
   for evaluation, the candidate with minimum mean path distance gives path
   `d_X=0.059330` against production `0.099339`, a 40.3% reduction. It does
   not clear 50% and requires oracle member choice.
4. **[Experiment] Stepwise graph distance.** Allowing the closest member to
   change at every step gives mean `0.046303`, 53.4% below production mean
   path error. This measures marginal graph coverage, not a coherent predicted
   trajectory, and is not reported as a rollout replacement.
5. **[Experiment] `K=64` diagnostic.** The maximum predeclared budget gives
   terminal `0.092238` (54.81% reduction), translation 60.8%, rotation 44.7%,
   joints 60.3%. Its bootstrap mean gain is 55.04%, 95% CI
   `[49.34%,60.29%]`. Because validation selected `K=32`, `K=64` is diagnostic
   only and not the selected model.

Implementation and immutable artifact:

- [`set-valued evaluator`](../scripts/evaluate_set_valued_solution_operator.py);
- [`selected result`](../runs/set-valued-solution-operator-v2/results.json).

## 17. Novelty/overlap audit for the successful formulation

| Component | Known overlap | Consequence for claims |
|---|---|---|
| Approximation of set-valued maps | Cellina's classical continuous-selection/approximation result ([1970 paper](https://www.bdim.eu/item?id=RLINA_1970_8_48_4_412_0)) and modern approximation theory for set-valued functions ([IMA JNA](https://academic.oup.com/imajna/advance-article/doi/10.1093/imanum/drag014/8659329)) | A set-valued solution operator is not itself novel. |
| Multiple PDE solutions | [Newton-Informed Neural Operator](https://arxiv.org/abs/2405.14096) explicitly computes multiple solutions | “Operator producing multiple solutions” cannot be claimed as first. |
| Best-of-many trajectory loss | Variety/MoN losses are established; their density bias is analyzed by [Thiede & Brahma, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Thiede_Analyzing_the_Variety_Loss_in_the_Context_of_Probabilistic_Trajectory_ICCV_2019_paper.html) | Min-over-set risk alone is not novelty and is not a calibrated probability law. |
| Prediction sets | Stable conformal prediction sets have finite-sample coverage theory ([Ndiaye, ICML 2022](https://proceedings.mlr.press/v162/ndiaye22a.html)); conformal trajectory regions also exist ([Cleaveland et al.](https://arxiv.org/abs/2304.00432)) | The present empirical coverage is not conformal and has no distribution-free guarantee. |

**[Inference]** The defensible contribution is narrower than “a new
set-valued neural operator”: under a fixed quasistatic contact dataset, a
geometry-conditioned finite physical solution relation exposes more than 50%
terminal coverage gain, while matched point regressors, energy/defect branch
selectors and random-set controls do not explain it. Establishing algorithmic
novelty would require a learned compact set/distribution parameterization,
branch-observation mechanism, calibrated coverage, and comparison with modern
multimodal baselines. None is claimed complete here.

## 18. Final answer to the mathematical question

The current observation contract does not support the original premise

$$
  (\phi,x_k,u_{k+1})\longmapsto x_{k+1}
$$

as a globally smooth, stable and uniquely identifiable operator. The correct
object supported by the experiments is instead

$$
  g_{\phi,x_0}(\cdot,\cdot)
  \longmapsto
  \mathcal S(g)\subset
  \bigl(SE(3)\times\mathbb R^6\bigr)^{32},
$$

the conditional solution relation over whole loading paths. This formulation
removes learned-state composition and keeps separated contact branches
separate. Under its proper point-to-set metric it reaches the requested strong
regime on untouched test. It does **not** solve branch selection and therefore
does not replace `runs/srno-r-material-v2/best-rollout.pt`.

To return to a single-valued physical operator, the input contract must be
expanded at least by mass/inertia and by an observable branch/history variable
(contact identities, impulses or a validated internal state). The next learned
model should parameterize a compact set or conditional law of coherent paths,
then select/calibrate branches from observed evidence. Training another
unconstrained mean regressor or adding an arbitrary local contact head is not
supported by this experiment chain.

**Completion status.** The strong threshold has been met for the explicitly
changed set-valued solution concept (`50.45%` selected terminal point-to-set
gain), with validation-only selection, untouched test, objectwise results,
bootstrap uncertainty, matched-cardinality random control, and path-coherence
qualification. The original single-valued production objective remains open.

## 19. Final reproducibility check

**[Experiment]** After adding the H10--H17 evaluators, all ten continuation
scripts compile in the `isaaclab` environment. The complete repository suite
was rerun:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n isaaclab pytest
99 passed, 4 skipped, 13 warnings
```

The four skips are the existing CUDA-only unit tests. The selected artifact
records manifest SHA-256
`c8bddec752f0418e92383fd0d9193e2d70a37845244c4c1c26ed9f9170c3012a`,
`test_used_for_selection=false`, `K=32`, 32 matched-random repeats, and the
complete validation/cardinality/bootstrap/objectwise tables.
