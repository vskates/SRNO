# MintyGrasp: conditional proximal-envelope learning for grasping through occlusion

Дата research pass: 2026-08-25.

Статус: новая независимая идея; **не является заявлением о достигнутом SOTA**.

Целевой уровень: general ML / structured prediction / scientific ML с robotic grasping как физически проверяемой инстанциацией.

## 0. Executive verdict

Предлагается учить не скрытую форму, не grasp-success probability, не posterior над outcome-функцией, не feasible-action set и не геометрический line transform. Новый объект — **условное математическое ожидание Moreau envelope малой convex terminal-contact problem**, индуцированной полной скрытой геометрией для query grasp.

Рабочее название общей постановки:

> **Coarsened Proximal-Envelope Learning (CPEL):** по частичному наблюдению и query предсказывать Bayes-average value и response локальной скрытой variational problem, сохраняя convexity, smoothness и proximal consistency по конструкции.

Роботическая модель называется **MintyGrasp**. Для полного объекта `S` и parallel-jaw candidate `g` offline contact oracle строит не mesh target для сети, а низкоразмерную convex contact-residual energy

\[
\Phi_{S,g}:\mathbb R^d\rightarrow\mathbb R_+\cup\{+\infty\},
\qquad d\approx 7\text{--}10,
\]

на локальных terminal perturbations. Для virtual probe `u` определяются

\[
P^\lambda_{S,g}(u)
=\operatorname{prox}_{\lambda\Phi_{S,g}}(u),
\]

\[
M^\lambda_{S,g}(u)
=\min_z\left\{
\Phi_{S,g}(z)+\frac{1}{2\lambda}\lVert z-u\rVert^2
\right\}.
\]

Новая статистическая цель:

\[
\bar M^\lambda(x,g,u)
=\mathbb E\!\left[M^\lambda_{S,g}(u)\mid X=x\right].
\]

Это ожидаемая минимальная local correction/work energy для данного grasp под всеми hidden geometries, согласующимися с одним noisy occluded RGB-D observation через обучающее распределение.

Новый learning objective — **Coarsened Proximal Envelope Score (CoPES)**: random-probe Sobolev score, одновременно сопоставляющий value envelope и его gradient/proximal response. Архитектура — **Conditional Proximal Envelope Operator (CPEO)**: sparse observation encoder, gripper-local query encoder, conditional input-convex energy и маленький differentiable prox layer. Для каждого `(x,g)` output по `u` является корректным Moreau envelope по конструкции.

Центральное новое знание не в том, что существуют Moreau envelopes, proximal averages или ICNN. Эти компоненты известны. Новая проверяемая гипотеза и paper unit таковы:

1. conditional expected envelope является компактным Bayes target для coarsened convex-response problems;
2. его value-gradient supervision строго сильнее response-only proximal regression;
3. **value anchor сохраняет decision-relevant hidden ambiguity, которую средний prox response теряет**;
4. exact envelope architecture не допускает non-monotone, non-integrable и physically impossible local response fields;
5. этот target может давать более высокую sample/compute efficiency, чем full completion, и более надёжный severe-occlusion selection, чем unconstrained scalar critic.

Идея заслуживает `go` только после дешёвого oracle Gate 0. Если integrated envelope risk на полной геометрии не предсказывает physical grasp outcome лучше обычных antipodal/force-closure margins, дальнейшее обучение MintyGrasp не оправдано.

---

## 1. Exact task contract

### 1.1 Included

- один rigid target object на полке;
- один noisy wrist RGB-D frame;
- supplied target mask/ID или общий upstream segmenter;
- обычная self-occlusion и не более одного foreground blocker / shelf lip;
- hidden target geometry известна только через training shape distribution;
- fixed parallel-jaw gripper;
- terminal 6-DoF grasp pose и commanded width;
- short jaw closure и millimetric/centimetric lift только как physical readout;
- небольшие terminal perturbations pose, width и calibration;
- full mesh и contact oracle доступны offline во время synthetic training;
- candidate ranking сначала проверяется при frozen high-recall proposal generator.

### 1.2 Excluded

- RL;
- VLA;
- active view, pushing, obstacle removal или tactile exploration;
- generic clutter;
- long-horizon approach-to-lift feasibility;
- causal taxonomy failure modes;
- full mesh, point completion, global occupancy, SDF, TSDF или NeRF как output;
- posterior samples скрытых shapes;
- random feasible-action set;
- posterior над целой grasp-outcome function;
- dense field в полном `SE(3) × width`, вычисляемый до query.

Observed shelf/blocker geometry проходит одинаковый deterministic collision gate для всех методов. Learned object касается только query-local terminal target interaction.

### 1.3 Два режима оценки

1. **Information-only:** foreground blocker присутствует в RGB-D, но убирается перед grasp execution без движения target/camera. Этот режим изолирует hidden-target inference.
2. **Combined shelf:** blocker остаётся; его observed geometry обрабатывается общим deterministic gate.

Главный ML claim должен сначала пройти information-only regime. Иначе gain может оказаться улучшенным obstacle collision filtering.

---

## 2. Explicit non-intersection with today's Markdown ideas

Проверены все occlusion Markdown-файлы, изменённые 25 августа 2026. Два файла — `MetaContact.md` и `MetaContact-2.md` — побитно идентичны и считаются одной веткой.

| Сегодняшнее направление | Его основной estimand / objective / architecture | Почему MintyGrasp не пересекается |
|---|---|---|
| FiGO / OC-GOP | posterior над grasp-outcome function; Blackwell/tower KCM | MintyGrasp детерминированно учит conditional expected convex envelope; нет posterior, shared latent, CVaR или filtration loss |
| FiberGrasp | necessary/possible action sets над observation fiber | нет lower/upper grasp sets, rough approximation или worst-case membership |
| Grasp Metamers / MetaContact | sensor-equivalent grouped shapes и joint bi-contact mixture | нет metamer groups, contact mixture, likelihood-weighted repeated outcomes или contact posterior |
| DQPL / CRFSP | posterior над feasible-action margin field | нет random action field и functional flow |
| FELLAS / CEN | random closed feasible set, Choquet hit/inclusion queries | CoPES сопоставляет values/gradients convex programs, а не вероятности set events |
| Grasp-Certificate Process / RJPN | stochastic certificate process и Energy–Variogram score | нет certificate posterior, variogram, tail score или ray–jaw incidence attention |
| CapGrasp | capacity operator для Boolean gripper-region events | нет Boolean hit signatures, inclusion–exclusion или probability circuit |
| AvoGrasp | avoidance functional failure set для pose packets | нет avoidance probability и robust packet event |
| FiRe | response polytope, support matching и information-contractive witnesses | нет uncertainty polytope, support function, max-min selector или contraction по evidence levels |
| JILT | positive jaw-line measures, Fourier moments и moment-cone projection | нет line/ray transform, positive-measure moments или reconstruction в transform domain |

Главное различие по научному объекту:

> Сегодняшние идеи представляют uncertainty, action sets, outcomes, event laws или скрытую interaction geometry. MintyGrasp представляет **Bayes value function семейства скрытых convex terminal problems** вместе с её integrable response gradient.

### 2.1 Ближайшая внутренняя угроза

FiRe и ранняя отвергнутая ветка `posterior over wrench-space convex body` являются наиболее опасной semantic collision. В MintyGrasp нельзя:

- материализовать grasp wrench set;
- предсказывать его vertices/support function;
- делать max-min по consistency set;
- заявлять uncertainty-set learning.

Contact oracle может внутри использовать convex cones или a small QP, но network output — **expected regularized optimal value as a function of local probe**, а не set of possible responses. Reviewer-facing ablation обязана сравнить MintyGrasp с learned distance/support-to-set baseline, чтобы показать, что value-gradient conditional envelope, а не просто новый язык для convex grasp metric, несёт выигрыш.

---

## 3. What literature already occupies

### 3.1 External occlusion is a real measured gap, not a new claim

[TARGO / TARGO-Net](https://targo-benchmark.github.io/) создаёт controlled target-driven grasping benchmark с single RGB-D и уровнями occlusion до `0.9`. Project page сообщает падение примерно `20 pp` или больше для нескольких strong systems при extreme occlusion; completion-based TARGO-Net устойчивее, но также деградирует. Следовательно, нельзя заявлять «first occluded grasping». Можно заявлять новый response-learning principle на уже подтверждённой проблеме.

### 3.2 Shape completion and uncertainty propagation are occupied

- TARGO-Net использует target completion и target-scene fusion.
- [ZeroGrasp, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.html) совместно предсказывает reconstruction и grasps.
- [PSSNet, CoRL 2020](https://proceedings.mlr.press/v155/saund21a.html) генерирует diverse plausible completions из ambiguous depth и демонстрирует grasping under occlusion.
- [Robust Grasp Planning over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645) оценивает grasps на MC-dropout completion samples и сообщает statistically significant gains over a point estimate.
- [Measuring Uncertainty in Shape Completion to Improve Grasp Quality](https://arxiv.org/abs/2504.16183) добавляет completion uncertainty к grasp ranking и сообщает улучшение real rank-5 success.

Эти работы дают indirect evidence, что hidden-shape ambiguity важна, но закрывают очевидный путь `posterior shapes -> planning`.

### 3.3 Direct grasp/contact prediction is occupied

[Contact-GraspNet](https://arxiv.org/abs/2103.14127) напрямую генерирует 6-DoF parallel-jaw grasps из depth recording и сокращает representation, привязывая contact к observed points. Direct scalar evaluator, contact prediction или another implicit grasp score сами по себе не являются вкладом.

### 3.4 Proximal/convex network ingredients are occupied

- [Input Convex Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/amos17b.html) вводит neural architectures, convex по выбранным inputs, и differentiable optimization inference.
- [What's in a Prior? Learned Proximal Networks, ICLR 2024](https://zhenghanfang.github.io/learned-proximal-networks/) параметризует exact proximal operators и предлагает proximal matching для inverse problems.
- [Learning with Fitzpatrick Losses, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/90caeb952bd4b03c7d8e7a0e31fc9a8b-Abstract-Conference.html) уже использует maximal-monotone/Fitzpatrick theory для новых supervised losses.
- [Learning Maximally Monotone Operators for Image Recovery](https://arxiv.org/abs/2012.13247) учит resolvents/maximally monotone regularizers для inverse imaging.

Поэтому нельзя заявлять novelty для ICNN, prox layer, monotone operator или Fitzpatrick/Fenchel gap по отдельности.

### 3.5 Proximal averaging is established mathematics

[The Proximal Average: Basic Theory](https://doi.org/10.1137/070687542) доказывает, что convex combination proximal mappings соответствует proximal mapping специального proximal average. Более новая работа по [integral proximal mixtures and proximal expectation](https://doi.org/10.3934/eect.2025017) изучает integral families. Это математическая опора, не наш theorem claim.

### 3.6 Contact solvers already use proximal operators

[Rigid Body Contact Problems using Proximal Operators](https://diglib.eg.org/items/fd65c61f-54e9-4e10-80b5-07e9bd6c2a26) даёт proximal formalism для rigid contact and friction. [Learning Contact Dynamics using Physically Structured Neural Networks, AISTATS 2021](https://proceedings.mlr.press/v130/hochlehnert21a.html) показывает, что contact-aware structural biases могут улучшать data efficiency и поведение на noisy nonsmooth dynamics относительно black-box models.

Поэтому novelty не в фразе `contact + prox`. Gap после поиска на 25.08.2026:

> Не найдена работа, которая по coarsened visual observation учит conditional expected Moreau envelope query-local hidden contact programs, использует совместный value-gradient proper score и применяет additive envelope level для сохранения decision-relevant ambiguity under occlusion.

Это evidence from search, а не доказательство отсутствия всех unpublished manuscripts.

---

## 4. Sequential search and rejected alternatives

### Branch A — grasp-space conservative gradient / Sobolev critic

Идея: учить gradient grasp-quality field и интегрировать neural ODE к local optima.

Отклонение: сводится к Sobolev training обычного implicit critic; flow/diffusion grasp generation и action fields уже заняты. Нет нового identifiable target, а absolute offsets и hidden ambiguity теряются.

### Branch B — pairwise preference / tournament field

Идея: учить probability, что `g_i` лучше `g_j`, через Hodge-decomposed tournament.

Отклонение: learning-to-rank, random utility и decision-focused ranking уже существуют; ветка также прямо отмечена как слабая в сегодняшнем DQPL-файле. Она нарушила бы non-intersection.

### Branch C — learned dual KKT certificate

Идея: предсказывать primal/dual solution и KKT gap contact program.

Отклонение: один dual certificate снова становится grasp certificate/scalar feasibility head. При ошибочных скрытых coefficients малый learned KKT residual не удостоверяет true geometry. Ветка слишком близка RJPN/certificate direction.

### Branch D — monotone contact operator only

Идея: учить `u -> prox(u)` squared loss с exact monotone architecture.

Отклонение как финальный objective: LPN и monotone-operator learning уже занимают общий механизм. Кроме того, response-only map теряет additive cost of hidden incompatibility; ниже дан точный counterexample.

### Branch E — conditional expected Moreau envelope

Исправление Branch D:

- учить **value и gradient вместе**;
- сделать target ожидаемым envelope, а не произвольным operator;
- использовать proximal-average closure, чтобы conditional averaging не выходило из structured hypothesis class;
- выбирать grasp по интегралу expected contact correction energy, для которого target Bayes-sufficient;
- отделить exact structural prior от uncertainty posterior.

Эта ветка выжила как MintyGrasp.

---

## 5. General ML formulation: learning hidden variational problems under coarsening

Пусть:

- `S ~ P` — hidden state;
- `X ~ O(.|S)` — partial/noisy/coarsened observation;
- `q ∈ Q` — downstream query;
- `Phi_{S,q}` — proper closed convex function на малом Hilbert space `Z`;
- `u ~ nu_q` — virtual perturbation/probe.

Обычные варианты решают одну из трёх задач:

1. восстановить `S`;
2. предсказать minimizer одной фиксированной problem;
3. предсказать downstream label/scalar utility.

CPEL ставит другой вопрос:

> Можно ли амортизированно учить **conditional expected regularized value function** семейства hidden variational problems по random queries, сохраняя exact operator geometry, и тем самым принимать решения без reconstruction hidden state?

Для `lambda > 0`:

\[
M^\lambda_{S,q}(u)
=\inf_z\left[
\Phi_{S,q}(z)+\frac{1}{2\lambda}\lVert z-u\rVert^2
\right],
\]

\[
P^\lambda_{S,q}(u)
=\arg\min_z\left[
\Phi_{S,q}(z)+\frac{1}{2\lambda}\lVert z-u\rVert^2
\right].
\]

Bayes target:

\[
\bar M^\lambda(x,q,u)
=\mathbb E[M^\lambda_{S,q}(u)\mid X=x].
\]

При стандартных integrability/differentiation assumptions:

\[
\nabla_u\bar M^\lambda(x,q,u)
=\frac{1}{\lambda}
\left(u-
\mathbb E[P^\lambda_{S,q}(u)\mid X=x]
\right).
\]

`bar M` одновременно содержит:

- absolute minimal correction/work cost;
- direction of the correction;
- local sensitivity across probes;
- a smooth, convex and integrable response field;
- hidden-world averaging без shape posterior at inference.

Это не generic neural operator learning: output class имеет exact variational semantics и downstream decision is an integral/functional of that value.

---

## 6. Grasp specialization: a local terminal-contact program

### 6.1 Query and local coordinates

Grasp candidate:

\[
g=(t,R,w)\in (SE(3)\times W)/C_2,
\]

где `C2` — symmetry of jaw swap.

Используется dimensionless local variable

\[
z=(\delta t_x,\delta t_y,\delta t_z,
\delta\omega_x,\delta\omega_y,\delta\omega_z,
\delta w)\in\mathbb R^7.
\]

Scaling задаётся measured calibration tolerances, чтобы translation, rotation и width имели сопоставимый metric. Это не arm trajectory; `z` описывает только terminal neighborhood.

### 6.2 Full-geometry oracle

Для training shape `S` и query `g` full mesh используется offline для:

1. pad-footprint ray casting;
2. first-contact linearization;
3. local signed gaps;
4. known-friction or fixed conservative friction-cone residuals;
5. bilateral balance/antipodality residuals;
6. gripper-body target collision residual within the terminal neighborhood.

Неизвестные mass, center of mass, material dynamics и long lift trajectory не вводятся. Friction coefficient в первой версии фиксирован или берётся из малого calibrated set; causal failure modes не моделируются.

### 6.3 Convex contact-residual energy

Один реализуемый oracle template:

\[
\Phi_{S,g}(z)
=c_{S,g}
+\frac12
\operatorname{dist}_{W_c}^2
\left(A_{S,g}z-b_{S,g},\;\mathcal K_\mu\right)
+\frac{\gamma}{2}\lVert B_{S,g}z-r_{S,g}\rVert^2
+\iota_{\mathcal Z}(z).
\]

Здесь:

- `A z - b` — affine local contact/gap response from the full-mesh linearization;
- `K_mu` — product of closed convex unilateral/friction cones in the chosen maximum-dissipation approximation;
- `B z - r` — bilateral symmetry and force-balance residual;
- `Z` — calibrated compact convex terminal perturbation set;
- `c_{S,g} >= 0` — constant miss/non-antipodal/body-collision penalty that does not alter the local response gradient.

Squared distance to a closed convex cone after an affine map is convex; quadratic balance term and convex indicator preserve convexity. `Phi` is query-local and low-dimensional. It is neither predicted geometry nor a causal model of a whole grasp cycle.

### 6.4 Why the constant term is scientifically important

If a hidden shape has no valid opposing contact, its local correction directions can look similar to another shape after regularization, while its absolute feasibility cost must remain higher. A response-only prox learner is blind to any additive `c_{S,g}` because

\[
\operatorname{prox}_{\lambda(\Phi+c)}
=\operatorname{prox}_{\lambda\Phi}.
\]

The envelope value changes by `c`. CoPES therefore learns the anchored value and the response; removing the value term is not a mild ablation but deletes one class of occlusion evidence.

### 6.5 Probe family

`u ~ nu_g` is a small, hardware-fixed set/distribution of dimensionless virtual terminal challenges:

- symmetric closing offsets;
- differential left/right jaw offsets;
- lateral pad micro-slip probes;
- small pitch/yaw/roll errors;
- width/calibration errors;
- sparse random combinations.

Use `B=8--24` probes per grasp during training and deterministic quadrature at inference. No dense 3-D sampling is required.

---

## 7. New learning objective: Coarsened Proximal Envelope Score

### 7.1 Oracle targets

For a training tuple `(S,x,g,u)`, small convex solve returns

\[
z^*=P^\lambda_{S,g}(u)
\]

and

\[
m^*=M^\lambda_{S,g}(u).
\]

The oracle gradient is free from the same solve:

\[
r^*=\nabla_uM^\lambda_{S,g}(u)
=\lambda^{-1}(u-z^*).
\]

### 7.2 Value-gradient score

MintyGrasp predicts `hat M_theta(x,g,u)`. Define

\[
\widehat r_\theta(x,g,u)=\nabla_u\widehat M_\theta(x,g,u).
\]

Основной objective:

\[
\mathcal L_{\mathrm{CoPES}}
=\mathbb E_{S,x,g,u}
\left[
w_v\,\rho_v(\widehat M_\theta-m^*)
+w_r\,\rho_r
\left(\lambda\widehat r_\theta-(u-z^*)\right)
\right].
\]

Для теории `rho_v(a)=a^2`, `rho_r(v)=||v||^2`. В реальном noisy oracle Huber versions допустимы, но strict propriety claim тогда надо формулировать аккуратно.

### 7.3 Population target

При squared terms population minimizer удовлетворяет

\[
\widehat M^*(x,g,u)=
\mathbb E[M^\lambda_{S,g}(u)\mid x,g,u],
\]

и

\[
\nabla_u\widehat M^*(x,g,u)
=\mathbb E[\nabla_uM^\lambda_{S,g}(u)\mid x,g,u].
\]

Value term фиксирует additive constant; gradient term даёт dense local response supervision и снижает число oracle probes, нужное для восстановления формы envelope.

### 7.4 Why this is not ordinary multi-task MSE

Сеть не имеет двух независимых heads. Архитектура принуждает value и response быть одной функцией:

\[
\widehat r_\theta=\nabla_u\widehat M_\theta,
\]

а Hessian удовлетворять envelope geometry. Невозможны циклические/non-integrable response predictions, которые дают низкую pointwise ошибку, но не соответствуют ни одной variational problem.

### 7.5 Optional second-order term

Если oracle permits stable implicit differentiation, можно добавить Hutchinson directional curvature matching:

\[
\mathcal L_{\mathrm{curv}}
=\mathbb E_{v}
\left\|
v^\top\nabla_u^2\widehat M_\theta
-v^\top\nabla_u^2 M^\lambda_{S,g}
\right\|^2.
\]

Это extension, не core. Paper должен сначала показать independent gain value+gradient над value-only и response-only.

### 7.6 Grasp decision

Для calibrated probe distribution `nu_hw`:

\[
R^*(x,g)=
\mathbb E_{u\sim\nu_{\rm hw}}
[\bar M^\lambda(x,g,u)].
\]

Выбор:

\[
\hat g(x)=
\arg\min_{g\in\mathcal C(x)}
\sum_{b=1}^{B_q}\omega_b
\widehat M_\theta(x,g,u_b),
\]

после observed collision gate. Это Bayes rule **для expected local correction/work risk**, а не для binary success probability. Binary grasp success остаётся внешним physical evaluation. Нельзя в abstract незаметно подменять одно другим.

---

## 8. New architecture: MintyGrasp / CPEO

```text
noisy RGB-D + target/obstacle/shelf roles
                    |
        sparse point/ray encoder
                    |
     query grasp g + two pad frames
                    |
 gripper-local bilateral cross-attention
                    |
       context h_theta(x, g)
                    |
 conditional input-convex energy Phi_theta(h, z)
                    |
 probes u_1 ... u_B -> differentiable ProxLayer
                    |
 (M_hat(u_b), z_hat(u_b), grad_u M_hat(u_b))
                    |
 calibrated quadrature -> expected contact risk -> rank
```

### 8.1 Sparse evidence encoder

Input tokens:

- target point position/RGB/normal/depth confidence;
- camera ray direction;
- observed free-space segment;
- shadow indicator after foreground obstacle;
- semantic role target/obstacle/shelf.

Encoder can be `SE(3)`-equivariant or use a strictly gripper-relative coordinate system. It must not densify a global voxel/SDF volume.

### 8.2 Bilateral query encoder

For candidate `g`, transform relevant evidence into left-pad and right-pad frames. Shared weights process both pads; swap is handled by exact symmetrization:

\[
h^{\rm sym}(x,g)=
\operatorname{Pool}
\{h_L(x,g),h_R(x,g)\}.
\]

Cross-attention is from pad footprint/query tokens to observed point tokens, not camera rays to hypothetical jaw-line measures. This distinction matters relative to RJPN/JILT.

### 8.3 Conditional input-convex energy

`Phi_theta(h,z)` is arbitrary in context `h` but convex in the small variable `z`. One implementation is a partially input-convex network:

\[
a_{k+1}=\sigma_k(W_k^+a_k+U_k z+V_k h+b_k),
\qquad W_k^+\ge 0,
\]

with convex non-decreasing activations. A nonnegative context-dependent offset `c_theta(h)` is explicit. It is essential for ambiguity, not a disposable bias.

### 8.4 Differentiable ProxLayer

For each probe:

\[
\hat z(u)=
\arg\min_z
\left[
\Phi_\theta(h,z)
+\frac{1}{2\lambda}\lVert z-u\rVert^2
\right].
\]

Strong convexity from the quadratic makes the solution unique. Because `d` is small, batched unrolled proximal Newton/FISTA or an implicit differentiable solver is feasible. Core experiments should use a solver tolerance tight enough that structural claims are numerical facts, not wishful regularization.

Value and response:

\[
\widehat M_\theta(u)=
\Phi_\theta(h,\hat z)
+\frac{1}{2\lambda}\lVert\hat z-u\rVert^2,
\]

\[
\nabla_u\widehat M_\theta(u)
=\lambda^{-1}(u-\hat z).
\]

### 8.5 Faster LPN variant

After validating the exact ProxLayer, a learned-proximal-network parameterization may amortize the inner solve. It must preserve:

- firm non-expansiveness of `u -> hat z(u)`;
- integrability/conservative Jacobian;
- a learned absolute envelope anchor.

Response-only LPN без anchor запрещён как full method: exact counterexample below shows why.

### 8.6 Candidate generation

First paper version:

- frozen high-recall generator shared by every ranker;
- fixed 64/128 candidates;
- same observed collision filter;
- report candidate oracle recall separately;
- only then compare ranking.

Secondary end-to-end version may add a lightweight proposal head. Proposal improvements cannot be counted as evidence for CoPES.

### 8.7 Complexity target

With `N_g=64`, `B_q=12`, `d=7`, 6--10 small solver iterations and shared observation encoding, target is below `100 ms` ranking on the lab GPU and memory well below full 3-D completion. This is a design target, not a measured result.

Report:

- wall-clock including encoder and ProxLayer;
- solver residual;
- peak GPU memory;
- number of oracle labels/probes;
- energy if practical;
- scaling in `N_g`, `B_q`, `d`, and iterations.

---

## 9. Theory package

### Proposition 1 — conditional envelope closure

For fixed `(x,g)`, assume `Phi_{S,g}` are proper closed convex and envelopes are integrable. Then

\[
\bar M^\lambda(x,g,\cdot)
=\mathbb E[M^\lambda_{S,g}(\cdot)\mid X=x]
\]

is convex and `1/lambda`-smooth. It is the Moreau envelope of an integral proximal average/proximal expectation under the appropriate regularity conditions.

This is an application/corollary of known proximal-average mathematics, not claimed as a wholly new convex-analysis theorem.

### Proposition 2 — response identity

Under dominated differentiation:

\[
\nabla\bar M^\lambda(u)
=\lambda^{-1}
\left(u-\mathbb E[P^\lambda_{S,g}(u)\mid X=x]\right).
\]

Thus the conditional mean proximal response is consistent with one smooth convex value function.

### Proposition 3 — CoPES propriety

Let probe measure `nu` have support on the compact query domain and use squared value/gradient terms with positive weights. Over `H^1(nu)` the unique population minimizer up to `nu`-a.e. equality is `bar M`. Gradient-only score identifies only an equivalence class modulo additive constants; any positive value weight removes this non-identifiability.

### Proposition 4 — architectural validity

For every context, exact ProxLayer produces:

- a convex differentiable envelope;
- `1/lambda`-Lipschitz gradient;
- a firmly non-expansive prox response;
- a symmetric PSD Jacobian almost everywhere when the potential is smooth.

An unconstrained MLP response head has none of these properties.

### Proposition 5 — selection regret

For finite candidate pool and normalized probe weights, if

\[
\sup_{g,u}
|\widehat M(x,g,u)-\bar M(x,g,u)|\le\epsilon
\]

and quadrature error is at most `delta`, then

\[
R^*(x,\hat g)-\min_g R^*(x,g)
\le 2(\epsilon+\delta).
\]

This connects the learned value error to the exact decision objective. It does **not** upper-bound binary physical failure without an additional calibration theorem.

### Proposition 6 — deliberate non-reconstruction

Training/evaluation probes occupy only a bounded `d`-dimensional terminal domain for queried grasps. Any two shapes inducing the same family of local convex energies on this domain are indistinguishable to MintyGrasp. Surface changes outside all pad/body terminal interaction regions lie in the representation null space. Therefore the learned object is non-injective with respect to full shape by design.

### Proposition 7 — stability to sensor perturbation

If context encoder is `L_h`-Lipschitz and conditional energy parameters vary Lipschitzly with context, strong convexity of the prox subproblem yields a bound on `||hat z_x(u)-hat z_{x'}(u)||` and hence on envelope risk. The exact constant depends on the chosen conditional ICNN parameterization. This theorem should be proved only for the implemented architecture, not stated generically.

---

## 10. Exact ambiguity counterexample: why response-only learning fails

Consider scalar hidden worlds with equal posterior probability and `lambda=1`:

\[
\Phi_-(z)=\iota_{\{-a\}}(z),
\qquad
\Phi_+(z)=\iota_{\{+a\}}(z).
\]

Their prox maps are constants:

\[
P_-(u)=-a,
\qquad P_+(u)=a.
\]

Conditional mean response:

\[
\bar P(u)=0.
\]

Это тот же response, что у certain world `Phi_0=iota_{\{0\}}`. Любой response-only learner полностью смешивает:

- определённый contact state at `0`;
- 50/50 hidden ambiguity between `-a` and `+a`.

Но envelopes различаются:

\[
M_-(u)=\tfrac12(u+a)^2,
\qquad
M_+(u)=\tfrac12(u-a)^2,
\]

\[
\bar M_{\pm}(u)=\tfrac12u^2+\tfrac12a^2,
\]

тогда как

\[
M_0(u)=\tfrac12u^2.
\]

Gradients одинаковы; absolute values отличаются на `a^2/2`. Это даёт точный тезис:

> Conditional mean response can create a physically plausible “ghost” response and erase ambiguity; the anchored expected envelope retains the Bayes correction cost even when all first-order responses cancel.

Для grasping аналог — две скрытые задние поверхности требуют противоположных terminal corrections. Средний correction равен нулю и выглядит уверенно, но expected minimal work/correction остаётся положительным. Именно эту величину учит CoPES.

Это не posterior uncertainty model: сеть не восстанавливает два modes. Она сохраняет их decision cost в scalar potential level.

---

## 11. Why MintyGrasp could outperform — indirect evidence only

### 11.1 Against deterministic completion

TARGO подтверждает severe-occlusion degradation и пользу occlusion-aware training, но его inference строит completed target representation. MintyGrasp оптимизирует только low-dimensional downstream contact program. Potential advantage:

- меньше output dimension;
- нет reconstruction loss на grasp-irrelevant surface;
- no mesh extraction/voxel decoding;
- direct supervision by local mechanics value and response.

Это делает sample/compute advantage правдоподобным, но не доказанным.

### 11.2 Against stochastic completion

Lundell et al. и PSSNet показывают пользу multiple shape hypotheses. Их cost включает shape samples и grasp evaluation per sample. MintyGrasp амортизирует ожидаемую regularized value напрямую и использует fixed probe quadrature. Potential advantage — similar expected-risk decision at lower inference cost.

Ограничение: MintyGrasp не даёт arbitrary posterior risk functionals после обучения. Если lab нужен CVaR или mode inspection, stochastic completion может быть принципиально сильнее.

### 11.3 Against direct scalar critic

Direct BCE/regression получает один label per `(x,g)` и может учить non-smooth decision boundary. CoPES получает bundle of values and gradients from one small oracle family and imposes exact convex/smooth geometry across probes. Structured contact-dynamics work даёт косвенное evidence, что physical inductive bias помогает на noisy nonsmooth systems.

Решающий experiment: same encoder, same candidates, same number of full shapes, scalar critic vs value-only vs response-only vs CoPES.

### 11.4 Against unconstrained operator regression

ICNN/LPN literature демонстрирует, что exact convex/proximal architectures practically trainable and дают interpretable/convergent operator behavior. Projection into a valid hypothesis class cannot worsen Euclidean error to a valid target when the projection itself is exact. Но conditional ICNN capacity может быть ниже unconstrained MLP; equal-parameter comparison обязателен.

### 11.5 Why value+gradient may be more data-efficient

Один full-mesh local program cheaply answers many virtual probes. Observation encoding — дорогая часть; probe labels после local linearization/QP относительно дёшевы. Shared-context bundle training может дать больше mechanically coherent supervision на один rendered RGB-D. Это аналогично amortized operator learning, но реальный `cost(render/contact solve)` надо профилировать, а не предполагать.

### 11.6 Exact limit of the superiority hypothesis

Защищаемая гипотеза:

> At matched candidate recall and compute, a valid conditional expected contact envelope will improve severe-occlusion ranking/calibration of expected local contact risk over full-shape completion and unconstrained scalar/operator baselines.

Нельзя до experiments утверждать:

- SOTA physical success;
- conditional safety guarantee;
- correct binary success probability;
- full Coulomb/contact realism;
- universal superiority over posterior shape models.

---

## 12. Experimental programme

### Gate 0 — oracle sufficiency before any visual model

Использовать full meshes only.

1. Generate a fixed candidate bank per object.
2. Build local convex contact programs.
3. Compute integrated envelope risk.
4. Execute short closure/lift in a higher-fidelity simulator not identical to the convex oracle.
5. Compare ranking with antipodal margin, Ferrari–Canny/force-closure proxy, tolerance score and a direct simulation success estimator.

Go only if envelope risk:

- materially improves top-1 regret or success prediction;
- remains useful under held-out friction/calibration perturbations;
- is not equivalent to one scalar oracle feature;
- benefits from multi-probe structure.

Kill if a one-number analytic metric matches it within noise.

### Gate 0b — value-anchor necessity

Construct hidden-shape pairs whose local prox responses nearly cancel while absolute feasibility costs differ. Compare:

- response-only LPN;
- envelope-gradient without value anchor;
- value-only;
- full CoPES.

The full method must win specifically on these pairs. Otherwise the central ambiguity theorem is decorative.

### Gate 1 — general ML benchmark: MaskedQP

Generate random low-dimensional convex programs

\[
\Phi_S(z)=\tfrac12z^TQ_Sz+c_S^Tz+\iota_{A_Sz\le b_S}(z),
\]

where observation reveals a noisy subset/sketch of coefficients and query gives `u`. Evaluate:

- envelope value error;
- gradient/prox error;
- structural violations;
- argmin/query-selection regret;
- OOD mask severity;
- ambiguity pairs with equal mean response and different expected value.

Baselines:

- unconstrained MLP value head;
- unconstrained value+gradient head;
- conditional ICNN value-only;
- LPN response-only;
- deep set/neural operator;
- MintyGrasp/CPEO with CoPES.

This second domain is essential for general-ML breadth. Without it the paper risks being viewed as contact-specific engineering.

### Gate 2 — controlled RGB-D occlusion benchmark

Use TARGO-compatible data generation but isolate one target and at most one blocker. Create paired renderings:

- full target visibility;
- self-occlusion only;
- foreground blocker severity bins;
- information-only execution;
- blocker-retained execution.

Train on ordinary per-shape examples, not explicit metamer/fiber groups. Ambiguity twins are evaluation stress tests only, preserving non-intersection with MetaContact/FiberGrasp.

### Gate 3 — matched candidate ranking

All methods receive identical 64/128 candidates and deterministic observed-collision gate. Report:

- candidate oracle recall;
- conditional ranking regret;
- top-1 simulator success;
- success vs visibility;
- expected-envelope calibration;
- selected extreme calibration;
- runtime/memory;
- solver residual and structural violation.

### Gate 4 — real shelf robot

Protocol:

- 20--30 unseen objects;
- controlled easy/medium/severe foreground occlusion;
- blocker-removed information-only subset;
- blocker-retained subset;
- repeated trials with randomized target pose;
- pre-registered exclusion and failure rules;
- Wilson or bootstrap confidence intervals;
- paired statistical test where scene seeds permit.

Do not mix segmentation failures with grasp-ranking failures; report both end-to-end and conditional-on-correct-mask results.

---

## 13. Required robotics baselines

1. Direct scalar critic with identical encoder/query features.
2. Direct critic + pose/noise augmentation.
3. Conditional mean contact-residual regressor without convexity.
4. Response-only LPN.
5. Value-only conditional ICNN.
6. Full CoPES/CPEO.
7. TARGO-Net or closest released completion-based target-grasp system.
8. Strong shape completion + fixed grasp detector.
9. MC-dropout / diverse completion + expected or conservative re-ranking.
10. Contact-GraspNet/GSNet-style direct detector where compatible.
11. Full-mesh analytic oracle and full-mesh higher-fidelity simulation oracle as upper bounds.

If exact external systems cannot share candidate pools, separate native end-to-end results from controlled reranking results.

---

## 14. Critical ablations

### Objective

- value-only;
- gradient/response-only;
- value + gradient CoPES;
- independent value and response heads;
- exact integrable joint head;
- Huber vs squared score;
- with/without curvature extension.

### Architecture

- unconstrained MLP energy;
- soft convexity penalty;
- exact conditional ICNN;
- exact ProxLayer;
- amortized LPN;
- with/without explicit offset `c_theta`;
- with/without jaw-swap symmetry;
- global encoder vs gripper-local bilateral cross-attention.

### Physics target

- binary success only;
- one scalar contact margin;
- single symmetric probe;
- structured probe bank;
- frictionless vs convex friction cone;
- local linearization radius;
- oracle mismatch against higher-fidelity simulator.

### Efficiency

- probes `B in {1,4,8,16,32}`;
- prox iterations;
- context reuse across candidates;
- cached oracle linearization;
- equal wall-clock and equal number of full meshes.

---

## 15. Falsification and kill criteria

### Kill 1 — local convex oracle is not decision-sufficient

If full-mesh envelope risk does not predict high-fidelity closure/lift better than standard analytic metrics, stop.

### Kill 2 — convexification erases the relevant contact modes

If local nonconvex contact behavior dominates ranking and convex oracle systematically prefers failures, method's physical premise is wrong.

### Kill 3 — value anchor is unnecessary

If response-only LPN matches CoPES on constructed cancellation pairs and severe occlusion, the new objective contribution collapses.

### Kill 4 — scalar critic is enough

If same-encoder BCE/regression matches MintyGrasp at equal labels/compute, structured target is not practically justified.

### Kill 5 — completion remains better at equal compute

If stochastic or deterministic completion wins success, calibration and runtime, no superiority claim survives.

### Kill 6 — solver cost destroys efficiency

If exact ProxLayer exceeds the ranking budget and amortized LPN loses the gain, architecture is not lab-usable.

### Kill 7 — simulation-oracle leakage

If performance disappears under different contact solver, friction or depth-noise model, the method learned oracle artifacts.

### Kill 8 — only the candidate generator improves

If equal-candidate ranking does not improve, remove all end-to-end claims.

### Kill 9 — no general ML result

If MaskedQP shows no benefit or theory reduces to a trivial restatement of supervised ICNN regression, target CoRL/RSS rather than ICLR.

### Kill 10 — novelty collision

If a prior work is found that already combines conditional expected Moreau-envelope regression, value-gradient scoring under coarsening and a structured proximal architecture, headline novelty must be replaced or project stopped.

---

## 16. Adversarial novelty audit

### 16.1 Hostile reviewer summary

> “This is an ICNN plus a differentiable convex contact solver trained with Sobolev loss. Proximal averages and learned proximal networks are old. The robot application merely learns another analytic grasp metric.”

Это сильная и частично справедливая критика. Paper survives только если покажет, что:

1. conditional expected envelope is the precise estimand, not decoration;
2. value anchor solves a real response-averaging ambiguity demonstrated analytically and empirically;
3. CoPES independently beats value-only/response-only/unconstrained heads;
4. the structured target transfers to a second coarsened convex-program domain;
5. equal-candidate severe-occlusion grasping materially improves;
6. output remains much smaller/faster than reconstruction.

### 16.2 Exact novelty sentence

Защищаемая формулировка:

> We introduce conditional proximal-envelope learning for coarsened decision problems: a random-query value-gradient score elicits the expected Moreau envelope of hidden convex response programs, while a conditional proximal architecture realizes the target exactly. In occluded grasping, this learns local contact-work risk from one RGB-D view without reconstructing shape or predicting an outcome/set posterior; the anchored envelope retains ambiguity that conditional mean proximal responses provably erase.

### 16.3 What must not be claimed

- first ICNN;
- first learned proximal operator;
- first use of Moreau envelope in ML;
- first proximal contact solver;
- first physics-informed grasp network;
- first reconstruction-free grasping;
- posterior uncertainty recovery;
- safety certificate;
- exact real Coulomb mechanics;
- guaranteed SOTA.

### 16.4 Nearest prior matrix

| Prior | Already provides | Missing relative to proposed paper |
|---|---|---|
| ICNN | convex network in selected inputs | no coarsened hidden-program estimand, envelope score or occluded grasping |
| Learned Proximal Networks | exact proximal maps; proximal matching for priors | no conditional expected envelope, value anchor ambiguity or query-local contact decision |
| Fitzpatrick Losses | monotone-operator-derived supervised losses | classification/link losses, not hidden convex-program envelope regression |
| Proximal average / proximal expectation | closure of averages/integral mixtures | mathematics only; no learned conditional observation/query operator or decision benchmark |
| Rigid contact prox solvers | proximal numerical contact mechanics | no visual amortization, conditional learning or grasp selection under hidden geometry |
| Structured contact dynamics | physics-aware contact nets and data-efficiency evidence | dynamics from observations/touch, not conditional envelope of hidden terminal problems |
| TARGO-Net | direct benchmark and strong occlusion-aware completion system | reconstructs target; no variational envelope/operator target |
| Uncertain completions | ambiguity propagation improves grasp planning | full shapes sampled at inference; no direct expected contact program |
| Direct grasp detectors | efficient partial-PCD grasp prediction | no anchored variational operator or structural validity |

### 16.5 Current novelty verdict

As of 25 August 2026, exact combination not found. Estimated novelty is **promising but not secure** because every mathematical component has strong prior art. Novelty must reside in the statistical target, response-only impossibility result, CoPES objective, architecture-target closure and empirical phenomenon together. If experiments validate only the architecture, likely outcome is a specialized robotics paper rather than ICLR.

---

## 17. ICLR 2027 audit

Official [ICLR 2027 reviewer guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines) asks four central questions: specific problem, motivation/placement, support for claims and significance/new knowledge. It explicitly says SOTA is not required, but claims must be rigorous and valuable. The [Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers) includes general ML, structured prediction, uncertainty and robotics, and asks for ambitious, complete work.

### Specific problem

Clear:

> Learn Bayes values and responses of hidden query-indexed convex programs from coarsened observations without reconstructing hidden states.

Grasping is a concrete hard instance with measurable physical consequences.

### Motivation and placement

Strong if paper clearly separates:

- hidden-state reconstruction;
- outcome prediction;
- operator response-only learning;
- conditional expected envelope learning.

Weak if framed as “use ICNN for grasping.”

### Support for claims

Minimum credible package:

1. formal closure/propriety/regret statements;
2. exact cancellation counterexample;
3. MaskedQP general benchmark;
4. oracle sufficiency study;
5. equal-candidate severe-occlusion comparisons;
6. strong completion and scalar/operator baselines;
7. real robot or exceptionally convincing sim-to-real evidence;
8. runtime/memory/solver residual;
9. negative results disclosed.

### Significant new knowledge

Potentially yes, if experiments establish all three:

1. response-only conditional operator averaging loses an empirically important ambiguity cost;
2. value-gradient conditional envelope restores it with a valid structured model;
3. this changes decisions in both a generic masked-program domain and physical grasping.

Robotic success improvement alone is insufficient for general-ML significance.

### Submission timing reality

ICLR 2027 full-paper deadline is 25 September 2026 AOE. From 25 August there is roughly one month. Unless datasets, simulator, robot stack and baselines are already operational, completing the required evidence package by that deadline is unrealistic. Scientific target should remain ICLR-level; practical submission may need ICLR 2028 rather than a rushed under-supported ICLR 2027 paper.

### Honest pre-experiment verdict

- **Current state:** interesting formulation, not yet an accept-level paper.
- **If only MaskedQP + simulation succeeds:** borderline; likely reviewer concern about robotics proxy and obvious combination.
- **If theory + second domain + equal-compute severe-occlusion gain + real robot succeed:** credible ICLR submission.
- **If only architecture improves:** redirect to CoRL/RSS/ICRA.

---

## 18. Preregistered claims and thresholds

Thresholds below are suggested go/no-go criteria, not results.

### Gate A — phenomenon

- response-cancellation subset must have at least `10 pp` larger ranking error for response-only than value+gradient at matched model/compute;
- full-mesh envelope oracle must beat the best single analytic metric with statistically clear improvement.

### Gate B — method

- CoPES must improve severe-bin candidate ranking regret by at least `10%` relative or `3 pp` physical top-1 success over the strongest same-encoder scalar/operator baseline;
- gain must survive equal-wall-clock and equal-number-of-full-shapes controls;
- exact structural violation near numerical tolerance; unconstrained baselines' violations reported.

### Gate C — efficiency

- at least `2x` lower peak memory or `2x` lower ranking latency than the strongest completion-based uncertainty pipeline, unless physical success gain is substantially larger;
- no more than `100 ms` ranking for 64 candidates on the lab target GPU is the desired deployment threshold.

### Gate D — real robot

- severe-occlusion improvement with confidence interval excluding zero;
- no degradation larger than `2 pp` on easy visibility;
- benefit exists in information-only regime, not only blocker-retained collision cases.

Exact thresholds should be adjusted once baseline variance and feasible trial count are measured, before viewing final test results.

---

## 19. Minimum implementation roadmap

### Phase 0 — 3--5 days: contact-program audit

- implement one `d=7` convex local oracle;
- validate convexity numerically and analytically;
- test against higher-fidelity simulator;
- profile oracle cost;
- run Gate 0 on a small mesh subset.

### Phase 1 — 3--5 days: synthetic CPEL

- build MaskedQP;
- implement value-only ICNN, response-only LPN and CPEO;
- verify cancellation example;
- test CoPES propriety/structure.

### Phase 2 — 1--2 weeks: frozen-feature grasp reranking

- use precomputed RGB-D features or a small point encoder;
- fixed candidates;
- train on oracle probe bundles;
- complete core objective/architecture ablations.

### Phase 3 — 2--4 weeks: full visual model and TARGO-compatible protocol

- sparse encoder;
- realistic noise/occlusion;
- strong baselines;
- efficiency profile;
- held-out shape families.

### Phase 4 — 2--4 weeks: real robot

- calibrated terminal perturbations;
- information-only and combined regimes;
- preregistered trial matrix;
- failure audit.

### Phase 5 — paper hardening

- formal proofs;
- second-domain completeness;
- newest literature audit;
- reproducibility package;
- limitations and negative results.

---

## 20. Scientific unit tests

1. `Phi_theta(h,z)` passes random Jensen convexity tests in `z`.
2. Prox solve residual below fixed tolerance for every reported query.
3. Firm non-expansiveness:
   \[
   \|P(u)-P(v)\|^2\le\langle P(u)-P(v),u-v\rangle.
   \]
4. Envelope gradient identity matches autodiff and finite differences.
5. Jaw-swap gives identical risk after probe permutation.
6. Rigid transform of scene/grasp preserves risk within numerical tolerance.
7. Adding a constant to oracle `Phi` changes value but not prox response.
8. Response-only model fails the `{-a,+a}` ambiguity unit test; CoPES distinguishes it.
9. Quadrature converges as probe count increases.
10. Candidate ordering is stable to solver tolerance once tolerance is below the declared level.
11. Observed collision gate is identical across baselines.
12. Candidate recall and ranking error are logged separately.
13. No full-shape information leaks into the test input or cached context.
14. Model output dimension/compute does not scale with global scene voxel resolution.

---

## 21. Draft paper pitch

### Possible title

**Learning the Value of Hidden Contact Problems: Conditional Proximal Envelopes for Grasping through Occlusion**

Alternative:

**MintyGrasp: Value–Response Learning of Occluded Contact without Shape Completion**

### Draft abstract

Single-view grasping is a coarsened inverse decision problem: many hidden object geometries agree with the same RGB-D observation but induce different local contact problems. Existing methods reconstruct one or more shapes or directly regress grasp outcomes. We introduce conditional proximal-envelope learning, which instead predicts the expected Moreau envelope of a query-indexed hidden convex response problem. The envelope jointly represents an absolute correction cost and an integrable proximal response. We show that conditional mean proximal responses can erase decision-relevant ambiguity even when they remain physically valid, whereas the anchored expected envelope retains its Bayes cost. We propose CoPES, a random-probe value-gradient score, and CPEO, a conditional input-convex architecture with a differentiable proximal layer that realizes a valid envelope by construction. MintyGrasp instantiates this principle for query-local parallel-jaw terminal contact under foreground occlusion, without predicting meshes, occupancy fields, feasible-action sets, or outcome posteriors. The empirical claims must be filled only after MaskedQP, matched-candidate simulation and real-shelf experiments.

### One-sentence contribution

> Learn the conditional value function of the hidden contact optimization problem—not the hidden object and not merely its mean optimizer.

---

## 22. Final decision

MintyGrasp is a genuinely different direction from all occlusion ideas written today at the levels that matter: estimand, objective and architecture.

Its strongest aspect is the exact separation between **mean response** and **expected value** under hidden ambiguity. The counterexample is simple, general and directly dictates the objective: response-only learned prox maps are insufficient; a value anchor is necessary. Proximal-average closure then makes the conditional target representable without abandoning physical structure.

Its largest risk is equally clear: the local convex contact energy may be a weak proxy for real parallel-jaw success. This cannot be repaired by a larger network. Therefore the correct next action is Gate 0, not end-to-end training.

Conditional recommendation:

- pursue the oracle and MaskedQP pilots;
- continue only if value-anchor necessity and contact-program sufficiency both appear;
- target ICLR only with a second-domain result, exact theory, equal-compute robotics comparisons and real validation;
- otherwise retain the structured contact scorer as a smaller robotics contribution without overstating generality.

---

## 23. Primary sources

### Occlusion and grasping

- Xia et al., *TARGO and TARGO-Net: Benchmarking Target-Driven Object Grasping Under Occlusions*: https://targo-benchmark.github.io/ and https://arxiv.org/abs/2407.06168
- Iwase et al., *ZeroGrasp*, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.html
- Saund and Berenson, *Diverse Plausible Shape Completions from Ambiguous Depth Images*, CoRL: https://proceedings.mlr.press/v155/saund21a.html
- Lundell et al., *Robust Grasp Planning Over Uncertain Shape Completions*: https://arxiv.org/abs/1903.00645
- Duarte et al., *Measuring Uncertainty in Shape Completion to Improve Grasp Quality*: https://arxiv.org/abs/2504.16183
- Sundermeyer et al., *Contact-GraspNet*: https://arxiv.org/abs/2103.14127
- Fang et al., *GraspNet-1Billion*: https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html

### Convex/proximal learning

- Amos, Xu and Kolter, *Input Convex Neural Networks*, ICML 2017: https://proceedings.mlr.press/v70/amos17b.html
- Fang, Buchanan and Sulam, *What's in a Prior? Learned Proximal Networks for Inverse Problems*, ICLR 2024: https://zhenghanfang.github.io/learned-proximal-networks/
- Rakotomandimby et al., *Learning with Fitzpatrick Losses*, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/90caeb952bd4b03c7d8e7a0e31fc9a8b-Abstract-Conference.html
- Pesquet et al., *Learning Maximally Monotone Operators for Image Recovery*: https://arxiv.org/abs/2012.13247
- Parikh and Boyd, *Proximal Algorithms*: https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf

### Proximal averages and contact structure

- Bauschke et al., *The Proximal Average: Basic Theory*: https://doi.org/10.1137/070687542
- Combettes and Cornejo, *Variational analysis of proximal compositions and integral proximal mixtures*: https://doi.org/10.3934/eect.2025017
- Erleben, *Rigid Body Contact Problems using Proximal Operators*: https://diglib.eg.org/items/fd65c61f-54e9-4e10-80b5-07e9bd6c2a26
- Hochlehnert et al., *Learning Contact Dynamics using Physically Structured Neural Networks*, AISTATS 2021: https://proceedings.mlr.press/v130/hochlehnert21a.html
- Han, Trinkle and Li, *Grasp Analysis as Linear Matrix Inequality Problems*: https://www.cs.cmu.edu/~lihan/Research/LMI_icra.html

### Venue

- ICLR 2027 Call for Papers: https://www.iclr.cc/Conferences/2027/CallForPapers
- ICLR 2027 Reviewer Guidelines: https://iclr.cc/Conferences/2027/ReviewerGuidelines
