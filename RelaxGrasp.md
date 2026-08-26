# RelaxGrasp: conditional mixtures of mechanical relaxation for grasping through occlusion

**Research pass:** 25 августа 2026.  
**Статус:** новая, независимо сформулированная исследовательская гипотеза; не заявление о достигнутом SOTA.  
**Целевой уровень:** ICLR / general ML, где robotic grasping — физически проверяемая инстанциация, а не единственный смысл метода.  
**General framework:** **Coarsened Relaxation-Spectrum Learning (CRSL)**.  
**Новый learning objective:** **Conditional Spectral Laplace Score (CSLS)**.  
**Новая architecture:** **Positive Spectral Relaxation Network (PSRN)**.  
**Robotic instantiation:** **RelaxGrasp**.

---

## 0. Executive verdict

Предлагается учить не скрытую форму, не вероятность успеха grasp-а, не posterior над outcome field, не feasible-action set, не contact distribution, не response polytope, не line transform и не value/proximal map скрытой convex problem.

Новый объект обучения — **условная средняя матричная spectral measure локальной механической релаксации grasp-а**.

Для полного объекта $S$ и query grasp-а $g$ offline oracle строит малую, нормированную terminal-stability matrix

$$
K_{S,g}\in\mathbb S_+^p,
\qquad p=6,
$$

где нулевые направления означают, что standardized compliant grasp не восстанавливает соответствующее малое смещение/вращение объекта, а большие eigenvalues означают быстро восстанавливаемые направления. Эта матрица не является reconstruction target: она имеет размер $6\times6$ и относится только к одному query grasp после локального замыкания jaws.

Вместо регрессии $K$, её среднего или одного eigenvalue используется виртуальная overdamped relaxation

$$
H_{S,g}(t)=\exp(-tK_{S,g}),\qquad t\ge0.
$$

Если направление хорошо удерживается, его response быстро затухает. Если оно не удерживается, eigenvalue равен нулю и response не затухает. При occlusion скрытая геометрия случайна относительно наблюдения $X=x$, поэтому Bayes target

$$
\bar H(x,g,t)
{}={}
\mathbb E\!\left[\exp(-tK_{S,g})\mid X=x\right]
$$

обычно **не является semigroup одного среднего оператора**. В общем случае не существует матрицы $\widehat K(x,g)$, для которой одновременно

$$
\bar H(x,g,t)=\exp(-t\widehat K(x,g))
\quad\forall t.
$$

Это не косметическая тонкость. Например, две неразличимые hidden geometries могут иметь

$$
K_1=\begin{bmatrix}1&0\\0&0\end{bmatrix},
\qquad
K_2=\begin{bmatrix}0&0\\0&1\end{bmatrix}.
$$

Их средняя stiffness равна $\frac12I$ и ложно выглядит устойчивой во всех направлениях. Но правильная средняя relaxation

$$
\tfrac12e^{-tK_1}+\tfrac12e^{-tK_2}
{}={}
\tfrac12(1+e^{-t})I
$$

не стремится к нулю: при $t\to\infty$ остаётся $\frac12I$. Тем самым representation сохраняет ambiguity-weighted weak modes, не создавая ни одной complete shape hypothesis.

Главная научная ставка:

> Под coarsened observation правильным компактным Bayes object для local physical stability может быть не средний hidden operator и не posterior над hidden state, а положительная смесь его relaxation modes. Conditional averaging превращает semigroup в matrix-completely-monotone curve; архитектура должна представлять именно эту более широкую семью.

Для обучения вводится **Conditional Spectral Laplace Score**. Model предсказывает положительную matrix-valued measure $\widehat\Sigma_{x,g}$ на спектре стабильности, а score сравнивает её Laplace transform с $e^{-tK_{S,g}}$ на случайных временах и virtual disturbance directions. При полном support probe distribution этот score строго elicites conditional mean spectral measure.

PSRN представляет её конечной положительной quadrature:

$$
\widehat H_\theta(t\mid x,g)
{}={}
\sum_{k=0}^{r-1}
A_k(x,g)e^{-t\alpha_k(x,g)},
$$

$$
A_k\succeq0,
\qquad
\sum_k A_k=I,
\qquad
\alpha_0=0,
\quad
\alpha_k>0\ (k>0).
$$

По конструкции output:

- symmetric positive semidefinite;
- равен (I) при $t=0$;
- монотонно уменьшается в Loewner order;
- имеет alternating-sign derivatives;
- допускает ненулевой long-time residue $A_0$, соответствующий hidden weak modes;
- не обязан выполнять semigroup law, что принципиально необходимо при ambiguity.

Это **conditional go**, а не готовый ICLR paper. Метод стоит развивать только если два ранних результата подтвердятся:

1. full-geometry relaxation risk лучше обычного scalar grasp margin предсказывает standardized short-lift outcome;
2. в ambiguity experiment средняя relaxation и PSRN существенно превосходят `predict mean K -> exponentiate`, direct scalar critic и unconstrained time-conditioned head при одинаковых данных и encoder-е.

Если хотя бы один пункт не выполняется, математическая конструкция не оправдывает robotics paper.

---

## 1. Exact task contract

### 1.1 Included

- один rigid target object на полке;
- один noisy RGB-D frame с wrist camera;
- supplied target mask/ID или одинаковый upstream segmenter для всех методов;
- обычная self-occlusion и не более одного foreground blocker / shelf lip;
- hidden grasp-relevant geometry известна только через training distribution форм;
- fixed parallel-jaw gripper;
- terminal 6-DoF grasp pose и commanded opening width;
- fixed compliant jaw-closing controller;
- удержание при малом, заранее фиксированном подъёме;
- небольшие terminal perturbations pose/calibration;
- full mesh, contact simulation и oracle labels доступны offline при synthetic training;
- frozen high-recall candidate generator на первом этапе.

### 1.2 Excluded

- reinforcement learning;
- VLA/VLM как основной метод;
- active view, pushing, obstacle removal, tactile exploration;
- generic clutter и rearrangement;
- full approach-to-lift trajectory feasibility;
- long-horizon manipulation;
- causal failure-mode decomposition;
- full mesh, point completion, SDF, TSDF, occupancy или NeRF как network output;
- posterior samples shapes;
- posterior над grasp-outcome function;
- random feasible-action set;
- response uncertainty polytope;
- line/ray transform скрытой геометрии;
- learned convex contact energy, Moreau envelope или differentiable proximal layer.

Observed shelf/blocker geometry проходит один и тот же deterministic terminal-collision gate для всех методов. Learned part касается только того, как скрытая target geometry меняет local closure stability.

### 1.3 Два обязательных evaluation regime

**Information-only.** Blocker присутствует при съёмке и удаляется до grasp execution без движения target/camera. Этот режим изолирует inference hidden target mechanics.

**Combined shelf.** Blocker остаётся, а его observed geometry обрабатывается одинаковым deterministic collision filter.

Главный ML claim сначала должен быть доказан в information-only regime. Иначе gain может возникнуть из obstacle filtering, а не из нового learning object.

---

## 2. Explicit non-intersection with today's Markdown ideas

Проверены markdown-файлы с occlusion ideas, изменённые 25 августа 2026. `MetaContact.md` и `MetaContact-2.md` идентичны и считаются одной веткой.

| Сегодняшняя идея | Её estimand / objective / architecture | Почему RelaxGrasp не совпадает |
|---|---|---|
| FiGO / OC-GOP | posterior над grasp-outcome function; Blackwell/tower KCM | RelaxGrasp детерминированно учит conditional mean matrix relaxation curve; нет function posterior, shared latent, CVaR или filtration loss |
| FiberGrasp | necessary/possible grasp sets на observation fiber | нет lower/upper action sets, support infimum/supremum или rough-set membership |
| Grasp Metamers / MetaContact | exact sensor-equivalent groups и joint bi-contact mixture posterior | нет metamer grouping, contact mixture, contact likelihood или paired-contact decoder |
| DQPL / CRFSP | posterior над random feasible-margin field в action space | output живёт в малом mechanical-mode space одного query, а не в stochastic action field |
| FELLAS / Choquet Excursion Network | random closed feasible set; hit/inclusion probabilities и Choquet score | CSLS сопоставляет matrix relaxation transforms, не вероятности событий множества |
| Grasp-Certificate Process / RJPN | stochastic certificate process; Energy–Variogram score; ray–jaw incidence attention | нет stochastic process samples, variogram, tail transform или ray–jaw attention |
| CapGrasp | conditional capacity operator Boolean gripper-region events | нет Boolean event circuit, inclusion–exclusion или hit signature |
| AvoGrasp | avoidance probability failure set для pose packets | нет failure-set avoidance functional и all-poses-success event |
| FiRe | response polytope; action-directed support matching; evidence-contractive witnesses | нет uncertainty polytope, support function, max-min response set или contraction across evidence levels |
| JILT | positive jaw-line measures, Fourier moments и moment-cone projection | нет geometric line measure, Fourier moment, ray bundle или transform-domain completion |
| MintyGrasp | expected Moreau envelope hidden convex contact problem; value-gradient CoPES; ICNN + prox layer | нет learned convex energy, virtual-probe minimization, proximal mapping или Sobolev value-gradient objective; PSRN parameterizes a non-semigroup mixture of exponential relaxation modes |

Главное различие:

> Сегодняшние методы представляют uncertainty над outcomes/sets/contacts, response support, interaction geometry или expected variational value. RelaxGrasp представляет **conditional mean spectral measure terminal relaxation**, для которой скрытая ambiguity проявляется как нарушение single-semigroup assumption и ненулевые non-decaying modes.

### 2.1 Опасная близость к MintyGrasp и уже выполненный pivot

Первоначально рассматривался target

$$
\mathbb E[(K+\lambda I)^{-1}\mid x,g].
$$

Он отброшен. Для quadratic energy такой matrix resolvent является масштабированным родственником proximal response / Moreau gradient и слишком близок к MintyGrasp. Простая замена `prox` на слово `resolvent` не удовлетворяет non-intersection requirement.

Выбранный target использует другую структуру:

$$
\mathbb E[e^{-tK}\mid x,g],
$$

то есть transient relaxation и matrix Laplace spectral measure. PSRN намеренно **не** является semigroup одного $K$, тогда как quadratic prox/resolvent head задаётся одним operator/energy. Conditional mixture, long-time null residue, CSLS и positive matrix-measure quadrature — независимое ядро.

Тем не менее reviewer может назвать RelaxGrasp «линейной relaxation-версией MintyGrasp». Поэтому обязательны:

- direct comparison с quadratic Minty/prox head;
- эксперимент на semigroup defect;
- ablation `mean K -> exp`;
- доказательство/демонстрация, что mixture head даёт независимый gain именно при hidden mechanical ambiguity.

Если gain исчезает, идею следует закрыть как внутреннее пересечение, а не переименовывать.

### 2.2 Отличие от старой rejected heat-flow idea

В `reports/EdgeFlux.md` была отклонена heat-flow smoothing binary grasp field **по grasp-pose manifold**. Там heat equation сглаживает success function в $SE(3)$ и фактически обобщает pose-noise convolution Johns et al.

RelaxGrasp не диффундирует по action space. Время $t$ индексирует виртуальную релаксацию **object-twist modes после конкретного terminal contact**, а generator — query-specific $6\times6$ stability operator. Нет convolution соседних grasp poses и нет semigroup consistency по $SE(3)$.

### 2.3 Отличие от conditional Laplace functional в FELLAS log

В FELLAS-файле как rejected branch рассматривался scalar Laplace functional бинарного feasible vector для восстановления joint random-set law. RelaxGrasp:

- не кодирует закон feasible vector;
- не отвечает на set-event queries;
- использует matrix exponential spectrum физического stability operator;
- elicites conditional mean positive matrix measure, а не joint distribution действий.

Общее слово `Laplace` относится к стандартному integral transform и не является novelty claim.

---

## 3. What the literature already occupies

### 3.1 External occlusion is already a measured problem

[TARGO / TARGO-Net](https://targo-benchmark.github.io/) напрямую оценивает target-driven grasping из одного RGB-D при visibility levels до 0.9. Balanced test содержит 1000 scenes на уровень. Project page сообщает около 20 percentage points degradation или больше у ряда strong baselines при extreme occlusion; completion-based TARGO-Net падает примерно на 7 points и остаётся сильным обязательным baseline. Следовательно, нельзя заявлять:

- first grasping under occlusion;
- first controlled occlusion benchmark;
- first target/scene occlusion reasoning;
- novelty простого occlusion augmentation.

### 3.2 Full and local completion are occupied

- [ZeroGrasp, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.pdf) jointly reconstructs geometry and predicts grasps; reports SOTA GraspNet metrics and a real pick-rate increase from 56.25% to 75% over its selected baseline.
- [NeuGraspNet, RSS 2024](https://www.roboticsproceedings.org/rss20/p046.pdf) uses single-view implicit geometry and global/local neural surface rendering for grasp quality.
- [Local Occupancy-Enhanced Grasping, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09354.pdf) completes grasp-local occupancy and shows that missing local geometry matters.
- [TOSC, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/download/38053/42015) explicitly completes potential contact regions instead of the whole shape and reports improvements in grasp displacement and Chamfer distance.

Therefore `complete less geometry`, query-local occupancy and task-oriented completion are not new. RelaxGrasp must win because of a different statistical/mechanical target, not because its hidden representation is merely smaller.

### 3.3 Uncertainty-aware completion and direct uncertainty are occupied

- [Robust Grasp Planning over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645) samples MC-dropout shapes and reports statistically significant simulated and real grasp gains over a point estimate.
- [Measuring Uncertainty in Shape Completion to Improve Grasp Quality](https://arxiv.org/abs/2504.16183) uses completion uncertainty for real parallel-jaw grasp ranking.
- [UNCLE-Grasp](https://arxiv.org/abs/2601.14492) samples leaf-occluded strawberry completions, aggregates force-closure feasibility and uses LCB/abstention.
- [FFHFlow, CoRL 2025](https://proceedings.mlr.press/v305/feng25a.html) learns flow-based diverse dexterous grasps and uses likelihood/latent features for uncertainty-aware ranking.

Thus posterior shapes, ensembles, LCB, CVaR, flow likelihood, deep-ensemble uncertainty and abstention cannot be headline novelty.

### 3.4 Grasp stiffness and spectral stability are established mechanics

RelaxGrasp does **not** invent stiffness-based grasp analysis.

- [Enabling Grasp Action: Generalized Evaluation of Grasp Stability via Contact Stiffness](https://arxiv.org/abs/1810.08317) builds grasp stiffness from contact mechanics and reports that its minimum-eigenvalue criterion follows the tendency of a standard grasp index.
- [On-Orbit Robotic Grasping: Grasp Stability Analysis and Experimental Results](https://pmc.ncbi.nlm.nih.gov/articles/PMC8247653/) uses a generalized stiffness/mass eigenproblem; positive eigenvalues indicate stability, zero eigenvalues marginal stability and negative eigenvalues instability under its model.
- [Grasp quality measures: review and performance](https://link.springer.com/article/10.1007/s10514-014-9402-3) reviews spectral/singular-value grasp measures and their limitations.
- Classical parallel-jaw compliance effects were analyzed long before deep learning; see [Compliance effects in a parallel jaw gripper](https://doi.org/10.1016/S0094-114X(03)00100-9).

Therefore claims such as `first spectral grasp metric`, `first learned stiffness`, `first matrix grasp quality` или `first compliance-aware grasping` запрещены.

### 3.5 Structured physical learning is established

- [Compositional Port-Hamiltonian Neural Networks](https://proceedings.mlr.press/v211/neary23a.html) hard-code port-Hamiltonian structure and observe compositional accuracy/passivity properties.
- [Invariant Neural Operators](https://proceedings.mlr.press/v206/liu23f.html) encode invariance/conservation and report improved accuracy and efficiency over neural-operator baselines on physical systems.
- [Extending Lagrangian and Hamiltonian Neural Networks with Differentiable Contact Models, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/b7a8486459730bea9569414ef76cf03f-Abstract.html) shows the value of explicit contact structure for learning nonsmooth physical systems.
- [Neural Operators with Localized Integral and Differential Kernels, ICML 2024](https://proceedings.mlr.press/v235/liu-schiaffini24a.html) reports large error reductions from locality-aware operator structure.

These works support the general mechanism `valid structure can reduce hypothesis space`, but they prevent claiming novelty for physics-informed networks, passivity or neural operators in general.

### 3.6 Completely monotone / spectral-mixture mathematics is established

For $K\succeq0$, spectral calculus gives

$$
e^{-tK}=\int_0^\infty e^{-t\kappa}\,dE_K(\kappa).
$$

Matrix-valued completely monotone kernels and their positive-measure representations are established mathematics; see, for example, [Hanyga, matrix-valued completely monotone kernels](https://arxiv.org/abs/2106.07946). Positive matrix-measure / rational-function structure is also studied in [Milton and Putinar](https://arxiv.org/abs/2206.02926). Scalar completely monotone neural parameterizations already occur in [Deep Archimedean Copulas, NeurIPS 2020](https://papers.nips.cc/paper_files/paper/2020/file/10eb6500bd1e4a3704818012a1593cc3-Paper.pdf).

Finite positive exponential sums themselves are classical Prony/generalized-Maxwell models. They have been used to approximate stretched-exponential relaxation [Mauro and Mauro, 2018](https://arxiv.org/abs/1803.07706), fitted with a custom neural network [He, Li and Du, 2019](https://www.ijpe-online.com/article/2019/0973-1318/0973-1318-15-1-107.shtml), and generalized to continuous relaxation-spectrum identification from measured time curves [Honerkamp-style direct spectrum identification, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11478369/). Deep inverse-Laplace spectrum reconstruction also exists in NMR [Wang et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12383262/).

Hence novelty is **not** Bernstein's theorem, matrix exponentials, positive residues, Prony sums, inverse-Laplace recovery or learning a relaxation curve separately. The proposed delta must remain the conditional, query-indexed, matrix-valued Bayes object under coarsening; its transform-proper objective; the identity-partition PSD head; and the theorem that conditional averaging leaves the single-semigroup class.

### 3.7 Gap after targeted search

На дату среза не найдена работа, которая одновременно:

1. starts from one foreground-occluded RGB-D target observation;
2. maps each parallel-jaw query to a terminal stability operator computed only offline from full geometry;
3. treats hidden-world averaging as a **mixture of relaxation semigroups rather than a single mean operator**;
4. directly elicits the conditional mean matrix spectral measure with a random-time transform score;
5. uses a positive-residue, identity-partition architecture guaranteeing complete monotonicity and allowing a non-decaying zero-mode residue;
6. shows a general-ML result on coarsened physical systems plus controlled grasping and real-robot validation;
7. does this without reconstruction, RL, VLA, stochastic outcome field, action set, response polytope, geometric line transform or proximal envelope.

Это evidence from a targeted search, а не доказательство отсутствия unpublished/contemporaneous work. Перед submission нужны Google Scholar/Semantic Scholar citation chasing по stiffness papers, matrix completely monotone models и recent ICLR 2027 submissions.

---

## 4. Sequential search and rejected alternatives

### Branch A — another posterior over grasp success

Идея: distribution/quantiles of one scalar success margin.

Отклонение: FiGO, Grasp-Certificate Process, CRFSP и uncertainty-aware grasping уже занимают posterior/tail space. Для one-shot expected utility marginal scalar probability теоретически достаточна; новый decoder не создаёт substantial knowledge.

### Branch B — robust action sets or support images

Идея: support, necessary/possible set, response set, polytope, max-min policy.

Отклонение: пересекается с FiberGrasp, FiRe, FELLAS/AvoGrasp и prior robust-DFL literature.

### Branch C — compact contact geometry transform

Идея: moments/rays/local projections of hidden contact geometry.

Отклонение: JILT, CapGrasp, MetaContact, TOSC и local occupancy уже занимают geometric/event/contact representations. Даже новый basis выглядел бы заменой transform.

### Branch D — conditional convex response / monotone operator

Идея: learn value, gradient, prox или resolvent локального contact program.

Отклонение: это ядро MintyGrasp. Переход к quadratic operator не создаёт независимую идею.

### Branch E — stiffness matrix regression

Идея: предсказывать $\widehat K(x,g)$ и ранжировать по minimum eigenvalue.

Отклонение: слишком близко к известным grasp stiffness metrics и обычной structured regression. Более того, $\mathbb E[K\mid x]$ может быть full-rank, когда каждый compatible world имеет different zero mode. Средняя матрица создаёт `Frankenstein stability`.

### Branch F — conditional matrix resolvent

Идея: learn $\mathbb E[(K+\lambda I)^{-1}\mid x,g]$.

Отклонение: математически интересно и risk-sensitive, но в quadratic case является близким родственником Moreau/proximal response. Это нарушает non-intersection с MintyGrasp.

### Branch G — selected: non-semigroup relaxation mixture

Замена inverse response на time relaxation дала независимое ядро:

- conditional expectation сохраняет complete monotonicity;
- conditional expectation обычно разрушает semigroup law;
- long-time residue точно сохраняет expected projector weak modes;
- output parameterized by positive matrix spectral measure, а не одним operator/energy;
- objective является transform score на measures, а не value-gradient regression.

Эта ветка выживает только условно: reviewer collision с `quadratic Minty` должен быть эмпирически снят.

---

## 5. General ML problem: coarsened relaxation-spectrum learning

### 5.1 Latent system, coarsened observation and query

Пусть:

- $Z\in\mathcal Z$ — полный hidden physical state;
- $X=\mathcal O_\omega(Z)+\epsilon$ — coarsened/noisy observation;
- $a\in\mathcal A$ — query/action/port configuration;
- $K(Z,a)\in\mathbb S_+^p$ — normalized local stability operator.

Full state induces relaxation

$$
H_{Z,a}(t)=e^{-tK(Z,a)}.
$$

Spectral theorem даёт unique projection-valued measure $E_{K}$:

$$
K=\int \kappa\,dE_K(\kappa),
\qquad
H(t)=\int e^{-t\kappa}\,dE_K(\kappa).
$$

При port projection $B_a\in\mathbb R^{p\times q}$, $B_a^TB_a=I_q$, используется positive matrix measure

$$
d\Sigma_{Z,a}(\kappa)
{}={}
B_a^T dE_{K(Z,a)}(\kappa)B_a.
$$

В grasping $q=p=6$ и $B=I$ — dimension уже мала. В general benchmark $B$ позволяет query-local port reduction большого hidden system.

### 5.2 Target under coarsening

Определим conditional mean spectral measure

$$
\bar\Sigma_{x,a}(A)
{}={}
\mathbb E[\Sigma_{Z,a}(A)\mid X=x]
$$

для measurable spectral sets $A\subseteq[0,\kappa_{max}]$.

Её transform:

$$
\bar H(x,a,t)
{}={}
\int e^{-t\kappa}\,d\bar\Sigma_{x,a}(\kappa)
{}={}
\mathbb E[B_a^Te^{-tK(Z,a)}B_a\mid x].
$$

Это deterministic Bayes statistic. Она не является posterior над shapes, contacts или outcome functions. Она также не сохраняет coupling между разными actions. Это намеренное сжатие: downstream loss должен зависеть только от expected relaxation cost данного query.

### 5.3 Почему mean operator — неверный model class

Naive structured model предсказывает

$$
\bar K(x,a)=\mathbb E[K(Z,a)\mid x]
$$

и использует $e^{-t\bar K}$.

Но generally

$$
e^{-t\mathbb E[K\mid x]}
\ne
\mathbb E[e^{-tK}\mid x].
$$

Правая часть — mixture of semigroups. Она выполняет normalization и complete monotonicity, но обычно не выполняет

$$
\bar H(t+s)=\bar H(t)\bar H(s).
$$

Следовательно, forcing one generator $\widehat K$ — structural misspecification именно там, где coarsening оставляет несколько mechanically different worlds.

### 5.4 Decision-relevant long-time residue

Для каждого PSD $K$:

$$
\lim_{t\to\infty}e^{-tK}=P_{\ker K}.
$$

По dominated convergence:

$$
A_0^\star(x,a)
:=
\lim_{t\to\infty}\bar H(x,a,t)
{}={}
\mathbb E[P_{\ker K(Z,a)}\mid x].
$$

Это expected projector non-restoring modes. В отличие от scalar failure probability он сохраняет **direction** слабости; в отличие от posterior он не разделяет hidden worlds.

Для normalized disturbance covariance $Q\succeq0$:

$$
r_0(x,a;Q)
{}={}
\mathrm{tr}(QA_0^\star)
$$

измеряет ambiguity-weighted mass нереставрируемых directions, релевантных задаче.

### 5.5 Multiscale relaxation cost

Zero-mode mass различает hard failure, но не positive near-zero modes. Поэтому основной cost:

$$
C^\star(x,a)
{}={}
\beta_0\mathrm{tr}(Q_0A_0^\star)
{}+
\int_{t_{min}}^{t_{max}}
w(t)\mathrm{tr}(Q(t)\bar H(x,a,t))dt.
$$

Большие $t$ чувствительны к near-zero eigenvalues, малые $t$ — к общей stiffness scale. $Q_0,Q(t),w(t)$ фиксируются из hardware disturbance model до test evaluation.

Decision:

$$
a^*(x)=\arg\min_{a\in\mathcal C(x)}C^\star(x,a).
$$

Если все costs выше preregistered threshold, допускается `abstain`; forced-choice result всё равно сообщается отдельно.

### 5.6 Exact boundary of sufficiency

$\bar\Sigma$ sufficient только для losses линейных по expected relaxation curve:

$$
L(a,Z)
{}={}
\beta_0\mathrm{tr}(Q_0P_{\ker K(Z,a)})
{}+
\int w(t)\mathrm{tr}(Q(t)e^{-tK(Z,a)})dt.
$$

Она не sufficient для:

- arbitrary tail probability binary success;
- joint portfolio utility нескольких grasпов;
- collision-free full approach;
- nonlocal lift dynamics;
- reconstruction-dependent tasks.

Paper не должен объявлять universal decision sufficiency.

---

## 6. Grasp specialization: terminal stability operator

### 6.1 Grasp query and coordinates

Grasp

$$
g=(R,t,w)
\in
\mathcal G=(SE(3)/C_2)\times[w_{min},w_{max}],
$$

где $C_2$ учитывает exchange identical jaws.

Все terminal twists выражаются в gripper frame. Translation/rotation приводятся к совместимым units через fixed characteristic length $\ell_g$:

$$
\xi=
\begin{bmatrix}
\Delta p/\ell_g\\
\Delta\theta
\end{bmatrix}\in\mathbb R^6.
$$

Это устраняет произвольность смешанных units в eigenvalue comparison.

### 6.2 Full-geometry compliant-contact oracle

Для full mesh $S$ и $g$:

1. simulate/solve fixed short jaw closure with calibrated pad compliance and preload;
2. identify terminal contacts and active stick/slide regime;
3. linearize local object restoring wrench around terminal equilibrium;
4. obtain symmetric tangent energy Hessian $H_{S,g}\in\mathbb S^6$ in normalized twist coordinates;
5. mark invalid closure/no bilateral support explicitly.

Если high-fidelity simulator не даёт trustworthy Hessian, finite-difference virtual perturbations $\pm\epsilon e_j$ around equilibrium могут оценить symmetric work matrix. Symmetrization должна быть physical, а не просто post-hoc `(H+H.T)/2`: reciprocity error до symmetrization сообщается как oracle diagnostic.

### 6.3 Rectified stability operator

Tangent stiffness может иметь negative modes. Для stable-relaxation target вводится preregistered physical threshold $\tau$:

$$
\widetilde H_{S,g}=M_g^{-1/2}H_{S,g}M_g^{-1/2},
$$

$$
K_{S,g}
{}={}
(\widetilde H_{S,g}-\tau I)_+.
$$

Здесь $(A)_+=V\mathrm{diag}(\max(\lambda_i,0))V^T$. Для invalid closure устанавливается $K=0$.

Интерпретация:

- modes ниже threshold, включая negative, становятся non-decaying;
- weak positive modes decay slowly;
- strong stable modes decay rapidly.

$\tau$ определяется из controller repeatability / minimum restoring work, а не подбирается на test success.

### 6.4 Why not regress raw stiffness

Raw $H$ имеет три проблемы:

1. units и reference-frame dependence;
2. negative eigenvalues делают heat response grow unbounded;
3. conditional mean может cancel/rotate incompatible weak modes.

Normalized rectification переводит local stability в bounded relaxation target

$$
0\preceq e^{-tK}\preceq I,
$$

который стабильно обучается и допускает exact long-time interpretation.

### 6.5 What oracle does not include

- arm IK and global reachability;
- full collision-free approach;
- obstacle removal;
- transport after lift;
- causal label of why failure happened;
- unknown deformable-object dynamics beyond fixed compliant-contact model.

Так scope остаётся local and falsifiable.

---

## 7. New learning objective: Conditional Spectral Laplace Score

### 7.1 Prediction space

Model predicts positive normalized matrix measure

$$
\widehat\Sigma_\theta(\cdot\mid x,g)
\quad\text{on}\quad[0,\kappa_{max}],
$$

$$
\widehat\Sigma_\theta([0,\kappa_{max}])=I_p.
$$

Its transform:

$$
\widehat H_\theta(t\mid x,g)
{}={}
\int e^{-t\kappa}d\widehat\Sigma_\theta(\kappa\mid x,g).
$$

### 7.2 Per-sample oracle measure

For eigendecomposition

$$
K_{S,g}=\sum_{j=1}^{p}\kappa_jv_jv_j^T,
$$

oracle spectral measure is

$$
\Sigma_{S,g}
{}={}
\sum_{j=1}^{p}v_jv_j^T\delta_{\kappa_j},
$$

and

$$
H_{S,g}(t)=\sum_jv_jv_j^Te^{-t\kappa_j}.
$$

Labels are therefore only eigenpairs of a $6\times6$ operator, not meshes/voxels.

### 7.3 Population score

Let $\nu_t$ have positive density on $[t_{min},t_{max}]$, preferably log-uniform, and let $\nu_u$ be a disturbance-probe distribution with full-rank covariance. Define

$$
\mathcal L_{\mathrm{CSLS}}(\theta)
{}={}
\mathbb E_{X,S,g}
\mathbb E_{t\sim\nu_t,u\sim\nu_u}
\left[
\left\|
\left(
\widehat H_\theta(t\mid X,g)
-e^{-tK_{S,g}}
\right)u
\right\|_2^2
\right].
$$

Optionally add a quadratic-work term

$$
\gamma
\left(
u^T\widehat H_\theta(t)u
-u^Te^{-tK}u
\right)^2,
$$

but it is not required for identifiability if vector probes have full covariance.

### 7.4 Why this is a measure score, not arbitrary time MSE

For scalar measures $\mu,\eta$, integrated transform distance expands into an MMD with kernel

$$
k_\nu(\kappa,\kappa')
{}={}
\int e^{-t(\kappa+\kappa')}d\nu_t(t).
$$

The matrix version applies this kernel to every bilinear projection $u^Td\Sigma v$. Because Laplace transforms of finite measures on a compact interval are injective, a sufficiently rich time interval distinguishes measures.

Thus CSLS learns an entire conditional spectral statistic while evaluating only random (t,u) queries.

### 7.5 Properness statement

Conditioned on $X=x,g$, squared-loss decomposition gives

$$
\mathbb E\|\widehat H(t)u-H_{S,g}(t)u\|^2
{}={}
\|\widehat H(t)u-\bar H(t)u\|^2
{}+\mathrm{Var}(H_{S,g}(t)u\mid x,g).
$$

Therefore the unique population minimizer in transform space is

$$
\widehat H^\star(t\mid x,g)=\bar H(x,g,t)
$$

almost everywhere in $t$. Analytic continuation/injectivity of the Laplace transform then identifies $\bar\Sigma_{x,g}$.

Claims should say `strictly proper for the conditional mean spectral measure under stated support and integrability assumptions`, not `proper for the full posterior of K`.

### 7.6 Long-time emphasis without numerical explosion

Use $t\sim\mathrm{LogUniform}(t_{min},t_{max})$ with normalized spectrum $\kappa\in[0,\kappa_{max}]$. Add explicit residue supervision

$$
\mathcal L_0
{}={}
\left\|A_0-P_{\ker K_{S,g}}\right\|_F^2
$$

only if exact zero modes are stable under oracle tolerances. Otherwise supervise a finite large-$t$ target and call it `slow-mode mass`, not exact null probability.

Total objective:

$$
\mathcal L
{}={}
\mathcal L_{\mathrm{CSLS}}
{}+\lambda_0\mathcal L_0
{}+\lambda_{rank}\mathcal L_{decision},
$$

where $\mathcal L_{decision}$ is optional pairwise ranking by oracle relaxation cost. The headline contribution must survive with $\lambda_{rank}=0$; otherwise CSLS is only an auxiliary regularizer for ordinary ranking.

### 7.7 Efficient stochastic estimator

Per `(x,g)` training step:

1. sample 2–4 log-times $t$;
2. sample 2–4 hardware-weighted directions $u$;
3. compute $e^{-tK}u$ from cached six-dimensional eigenpairs;
4. evaluate PSRN analytical sum of exponentials;
5. backpropagate CSLS.

No ODE solve, shape sample, convex optimization or matrix exponential is required inside the learned model.

---

## 8. New architecture: Positive Spectral Relaxation Network

### 8.1 Interface

$$
(x,g,t)
\mapsto
\widehat H_\theta(t\mid x,g)\in\mathbb S_+^6.
$$

Internally the network outputs $r$ spectral atoms and PSD residues:

$$
\{\alpha_k(x,g),A_k(x,g)\}_{k=0}^{r-1}.
$$

Recommended $r=8$ initially; ablate $r\in\{1,2,4,8,16\}$.

### 8.2 Observation encoder

Use a sparse point/voxel encoder already validated in the lab; architecture novelty should not depend on inventing another backbone.

Input token types:

- visible target points with RGB/depth confidence;
- visible blocker/shelf points;
- compact 2-D target/occlusion-mask context;
- camera calibration.

Do not materialize completed occupancy. Do not use RJPN-style camera-ray–jaw-ray incidence attention. A generic sparse equivariant encoder plus gripper-frame pooling is sufficient.

### 8.3 Query-local gripper feature

Transform visible points into gripper coordinates and pool:

- signed coordinate along closing axis;
- distance to two pad boxes and palm box;
- approach-axis coordinate;
- visibility/type embedding;
- global target token.

These are deterministic query coordinates, not a predicted line transform. The head never outputs hidden occupancy along those coordinates.

### 8.4 Conditional hypernetwork

Concatenate observation context and grasp token:

$$
h_{x,g}=\mathrm{MLP}([h_x,h_g,h_{local}]).
$$

Small heads output:

- raw pole logits $b_k$;
- raw residue factors $B_k\in\mathbb R^{6\times r_k}$;
- optional scale $s(x,g)$ bounded around a global physical normalization.

Poles:

$$
\alpha_0=0,
\qquad
\alpha_k
{}={}
\alpha_{min}
{}+(\alpha_{max}-\alpha_{min})\sigma(b_k),\quad k>0.
$$

For interpretability poles can be sorted, but ordering is not required for validity.

### 8.5 Exact PSD residue partition

Raw residues:

$$
C_k=B_kB_k^T\succeq0.
$$

Let

$$
S=\sum_{k=0}^{r-1}C_k+\epsilon I.
$$

Define

$$
A_k=S^{-1/2}C_kS^{-1/2},
$$

and an additional fixed residual atom

$$
A_{res}=\epsilon S^{-1}
$$

at a fixed high pole $\alpha_{res}=\alpha_{max}$. Then exactly

$$
\sum_kA_k+A_{res}=I.
$$

For $6\times6$ matrices eigendecomposition/Cholesky of $S$ is cheap and stable.

### 8.6 Relaxation decoder

$$
\widehat H_\theta(t\mid x,g)
{}={}
A_0
{}+\sum_{k=1}^{r-1}A_ke^{-t\alpha_k}
{}+A_{res}e^{-t\alpha_{max}}.
$$

This gives exact identities:

$$
\widehat H(0)=I,
$$

$$
(-1)^n\frac{d^n}{dt^n}\widehat H(t)
{}={}
\sum_{k>0}A_k\alpha_k^ne^{-t\alpha_k}
\succeq0,
$$

$$
\lim_{t\to\infty}\widehat H(t)=A_0.
$$

No soft regularizer is needed for PSD, normalization or complete monotonicity.

### 8.7 Deliberate non-semigroup design

Generally

$$
\widehat H(t+s)\ne\widehat H(t)\widehat H(s).
$$

This is a feature. Enforcing equality would collapse all atoms into the spectral projectors of one operator and remove precisely the conditional mixture required by occlusion ambiguity.

The model should report **semigroup defect**

$$
D_{sg}(x,g)
{}={}
\mathbb E_{t,s}
\left\|
\widehat H(t+s)-\widehat H(t)\widehat H(s)
\right\|_F^2
$$

as a diagnostic, not as a loss to minimize.

### 8.8 Decision head

Compute analytically

$$
\widehat C(x,g)
{}={}
\beta_0\mathrm{tr}(Q_0A_0)
{}+\sum_k
\mathrm{tr}(Q_kA_k)
\int_{t_{min}}^{t_{max}}w(t)e^{-t\alpha_k}dt.
$$

For log-uniform $w$, integral can be tabulated or evaluated with 8–16 fixed quadrature nodes. Candidate with minimum cost is selected after the common collision gate.

No learned scalar quality head is used in the main model. A monotone one-dimensional calibration from cost to success probability may be fitted on held-out data for reporting, but it is not the representation.

### 8.9 Complexity target

For $B$ candidates and $r$ atoms:

- encoder: once per scene;
- local query pooling: $O(Bm_qd)$ over a small neighbor set;
- spectral head: $O(Brp^2)$, $p=6$;
- no 3-D decoder;
- no shape sampling;
- no ODE/prox solver at inference.

Target: rerank 256 candidates in under 50 ms after encoder on a current GPU, with peak-memory well below completion baselines. This is a target, not an existing result.

---

## 9. Theory package

### Proposition 1 — conditional closure

For every (Z,a), $H_{Z,a}(t)=B^Te^{-tK(Z,a)}B$ is matrix completely monotone and $H(0)=I$. If entries are integrable, conditional expectation preserves these properties:

$$
(-1)^n\bar H^{(n)}(t)\succeq0.
$$

Therefore the Bayes target stays inside the class represented by a positive matrix measure, even though it generally leaves the smaller class of single semigroups.

### Proposition 2 — semigroup misspecification

There exist conditional distributions of $K\mid x,a$ for which no deterministic $\widehat K$ satisfies

$$
e^{-t\widehat K}=\mathbb E[e^{-tK}\mid x,a]
\quad\forall t\ge0.
$$

The diagonal twin example in Section 10 is an exact proof. More generally, equality for all $t$ imposes moment identities such as

$$
\mathbb E[K^2\mid x]=\mathbb E[K\mid x]^2,
$$

which fail under nonzero operator variance.

### Proposition 3 — zero-mode preservation

$$
\lim_{t\to\infty}\bar H(x,a,t)
{}={}
\mathbb E[B^TP_{\ker K}B\mid x,a].
$$

Thus long-time residue does not vanish merely because different hidden worlds fail in different directions.

### Proposition 4 — CSLS propriety

If $\mathrm{Cov}(u)\succ0$, $\nu_t$ has support with an accumulation point and measures have bounded mass/support, the unique population minimizer of CSLS is $\bar H$, and injectivity of the Laplace transform identifies $\bar\Sigma$.

### Proposition 5 — finite-atom approximation

Every bounded positive matrix measure on compact $[0,\kappa_{max}]$ can be approximated weakly by finite atomic positive matrix measures with the same total mass. Their Laplace transforms converge uniformly on compact time intervals. Hence PSRN can approximate the target as $r\to\infty$.

This is an approximation result, not a claim that $r=8$ always suffices.

### Proposition 6 — decision regret

Suppose for all candidates

$$
|\widehat C(x,g)-C^\star(x,g)|\le\epsilon.
$$

Then for $\hat g=\arg\min_g\widehat C(x,g)$:

$$
C^\star(x,\hat g)-\min_gC^\star(x,g)\le2\epsilon.
$$

A corresponding bound follows from integrated operator error and norms of (Q,w).

### Proposition 7 — deliberate non-reconstruction

Full geometry enters the model only through

$$
S\mapsto K_{S,g}\mapsto\Sigma_{S,g}
$$

for sampled queries. Infinitely many shapes/contact arrangements can induce the same local $6\times6$ stability operator. Therefore exact shape recovery from this target is impossible without additional information.

This does not prove statistical sample-efficiency; it only establishes quotienting.

### What must not be claimed without further proof

- that $A_0$ equals binary failure probability;
- that complete monotonicity alone ensures physical realizability by one grasp;
- that conditional mean relaxation is sufficient for arbitrary risk measures;
- that finite $r$ recovers the exact spectral measure;
- that full-rank stiffness is equivalent to nonlinear force closure in every contact model;
- distribution-free safety guarantees;
- superiority over scalar critic or completion before experiments.

---

## 10. Exact ambiguity example

### 10.1 Two hidden worlds

Observation $x$ is compatible with two equiprobable worlds. A given query grasp has two normalized stability operators:

$$
K_1=\mathrm{diag}(1,0),
\qquad
K_2=\mathrm{diag}(0,1).
$$

Each world has one non-restoring direction.

### 10.2 Mean-operator failure

$$
\bar K=\tfrac12(K_1+K_2)=\tfrac12I.
$$

Single-operator relaxation predicts

$$
e^{-t\bar K}=e^{-t/2}I\to0.
$$

It declares that every direction eventually relaxes.

### 10.3 Correct conditional relaxation

$$
\bar H(t)
{}={}
\tfrac12e^{-tK_1}+\tfrac12e^{-tK_2}
{}={}
\tfrac12(1+e^{-t})I.
$$

$$
\lim_{t\to\infty}\bar H(t)=\tfrac12I.
$$

PSRN represents this exactly with poles $0,1$ and residues $\frac12I,\frac12I$.

### 10.4 Semigroup defect

For $t,s>0$:

$$
\bar H(t+s)
-\bar H(t)\bar H(s)
{}={}
\tfrac14(1-e^{-t})(1-e^{-s})I\succ0.
$$

Thus the failure of semigroup composition is not numerical error; it is an observable signature of unresolved hidden modes.

### 10.5 What direct scalar critic can still do

A well-trained binary critic may learn 50% success directly and beat a badly chosen relaxation cost. The example proves only that **mean mechanical operator** is misspecified, not that a scalar critic is impossible.

Hence equal-encoder scalar BCE/ranking is the most important baseline and a kill test.

---

## 11. Why RelaxGrasp could outperform: indirect evidence only

### 11.1 Hidden geometry materially affects grasping

TARGO shows strong degradation with occlusion. NeuGraspNet, Local Occupancy and ZeroGrasp show that adding hidden/local geometry reasoning improves grasp metrics. Therefore the missing geometry is not harmless noise.

### 11.2 Uncertainty-aware decisions can beat point completion

Lundell et al., UNCLE-Grasp and completion-uncertainty ranking report gains from preserving ambiguity rather than using one shape estimate. RelaxGrasp preserves only its mechanical spectral image, making the same qualitative benefit plausible at lower inference cost.

### 11.3 Stiffness eigenstructure is mechanically relevant

Classical and modern stiffness analyses use eigenvalue signs/minima to diagnose local grasp stability. RelaxGrasp does not ask a network to discover this structure from a binary label alone; it exposes directional, multiscale supervision.

### 11.4 Structured physical models often improve data efficiency/generalization

INO, port-Hamiltonian networks and differentiable contact models show that exact structural constraints can improve physical learning or preserve validity. PSRN similarly removes impossible outputs: increasing relaxation, non-PSD response, wrong $H(0)$, negative residues and arbitrary cross-time inconsistency.

### 11.5 The main advantage has an exact counterexample

The twin example proves a specific failure of `predict average operator`. This is stronger than a generic intuition that uncertainty matters. It does not prove advantage over direct scalar or posterior completion, but motivates precisely the ablation that can.

### 11.6 Computation is plausibly smaller

PSRN outputs $r$ six-dimensional PSD residues/poles per query. Completion methods output thousands to millions of geometric variables, sometimes multiple samples. At matched encoder/candidate bank, lower head latency and memory are plausible by construction.

### 11.7 Exact limit of the superiority hypothesis

RelaxGrasp should win only when:

- local terminal mechanics predicts short-lift success;
- hidden worlds differ in weak-mode directions/scales;
- finite relaxation spectrum is learnable from observation priors;
- structured cross-time supervision helps more than it restricts;
- candidate recall is not the bottleneck.

If labels are nearly binary and one scalar is sufficient, direct critic should win. The paper must allow that outcome.

---

## 12. Experimental programme

### Gate 0 — full-geometry oracle validity before visual learning

Use complete meshes and 20k–100k candidate grasps across diverse objects.

For every candidate compute:

- standard antipodal/force-closure score;
- Ferrari–Canny-style epsilon where available;
- minimum normalized tangent-stiffness eigenvalue;
- binary simulated short-lift outcome under small perturbations;
- RelaxGrasp oracle cost from $K$.

Questions:

1. Does relaxation cost rank physical success better than minimum eigenvalue and standard analytic scalar?
2. Is long-time slow-mode mass informative beyond one eigenvalue?
3. Is result stable across pad stiffness, friction and threshold $\tau$?

**Go:** relaxation cost improves top-1 success/AUROC or worst-decile failure materially (preregister, e.g. at least 3 pp top-1 or 0.03 AUROC) over the best simple stiffness scalar on held-out object families.

**Kill:** no independent predictive value, extreme sensitivity to contact solver, or two-contact parallel-jaw oracle is almost always rank-deficient in a non-discriminative way.

### Gate 1 — general ML benchmark: MaskedPort

Create latent spring/truss/graph systems with PSD generator $L_Z$, port query $B_a$, and connected structured masking of internal elements.

Target:

$$
B_a^Te^{-tL_Z}B_a.
$$

Construct ambiguity pairs where the same visible subgraph has different hidden weak modes.

Baselines:

1. reconstruct hidden graph, then simulate;
2. predict $\mathbb E[L\mid x]$, then exponentiate;
3. unconstrained $f(x,a,t)$ MLP;
4. monotone scalar exponential mixture independently per matrix entry;
5. conditional diffusion over hidden $L$;
6. PSRN + CSLS.

Metrics:

- integrated transform error;
- zero/slow-mode residue error;
- semigroup-defect recovery;
- decision regret under held-out port costs;
- PSD/monotonicity/normalization violations;
- 1%, 5%, 10%, 100% training-data curves;
- OOD mask topology, graph size and stiffness scale.

**Kill:** PSRN does not beat unconstrained time head or mean-operator baseline in ambiguity strata and low-data/OOD regimes.

### Gate 2 — controlled RGB-D occlusion reranking

Freeze one high-recall candidate generator. For each full object:

- render same target/camera with connected foreground blockers;
- use visibility bins 0–0.2, 0.2–0.4, 0.4–0.6, 0.6–0.8, 0.8–0.9;
- store complete-geometry $K_{S,g}$ and lift outcomes for a shared candidate bank;
- include unseen instances, category-held-out and shape-family-shift splits;
- separate information-only and combined regimes.

Main hypotheses:

- H1: `mean K -> exp` becomes optimistically biased in high ambiguity;
- H2: PSRN reduces relaxation error and top-one regret;
- H3: structure advantage grows as labels shrink or occlusion increases;
- H4: benefit remains with identical candidates and collision filter.

### Gate 3 — end-to-end candidate generation

Only after fixed-bank success:

- keep PSRN selector;
- add a lightweight proposal head or use an existing generator;
- report proposal recall separately;
- do not attribute proposal improvements to spectral objective.

### Gate 4 — real shelf robot

Recommended paired protocol:

- 15–25 unseen household objects;
- 3–4 blocker configurations plus clean observation;
- randomized method/order;
- at least 250–400 total executed grasps if resources allow;
- record depth, mask, candidate bank, selected grasp, cost curve, collision outcome, closure, slip and short-lift success;
- bootstrap CIs by object, not only by grasp attempt.

Information-only trials should remove blocker after imaging. Combined trials keep it and use the common collision gate.

Primary real metric:

- paired top-1 short-lift success in severe-occlusion scenes.

Secondary:

- forced-choice success;
- selective risk/coverage if abstention is used;
- failures per object;
- latency and peak memory;
- calibration of slow-mode risk.

---

## 13. Required baselines

### Robotics baselines

1. same candidate generator + direct BCE success critic;
2. same encoder + signed margin regression;
3. same encoder + minimum-eigenvalue regression;
4. same encoder + full $K$ regression followed by $e^{-tK}$;
5. same encoder + unconstrained time-conditioned matrix head;
6. TARGO-Net or closest reproducible completion-based external-occlusion model;
7. ZeroGrasp / NeuGraspNet / Local Occupancy where protocol permits;
8. deterministic completion + identical mechanics;
9. stochastic completion + identical relaxation evaluation / LCB;
10. deep ensemble scalar critic;
11. quadratic Minty/prox-style response head as internal collision baseline;
12. JILT/FiRe/other internal ideas only if implementations exist before experiment freeze.

### Fairness constraints

- identical target masks;
- identical candidate bank for reranking;
- identical collision filter;
- identical training objects/occlusion renders;
- matched encoder and parameter budget for direct heads;
- equal label-generation wall clock reported;
- equal inference hardware and batch size;
- no completion metric substituted for grasp success.

---

## 14. Critical ablations

### Estimand

- $K$ regression vs $e^{-tK}$ transform supervision;
- one scalar eigenvalue vs full directional matrix measure;
- short-time only vs log-time range;
- no explicit zero atom;
- exact null residue vs finite large-time slow residue;
- binary auxiliary loss on/off.

### Architecture

- unconstrained residues;
- PSD residues without identity partition;
- single-generator semigroup $e^{-t\widehat K}$;
- PSRN mixture;
- $r=1,2,4,8,16$;
- fixed vs learned poles;
- diagonal vs full matrix residues;
- generic MLP over $t$ with matched parameters.

### Physics

- different pad stiffness;
- friction randomization;
- threshold $\tau$;
- stiffness oracle vs contact-wrench metric;
- finite-difference vs analytic Hessian;
- gravity-only $Q$ vs isotropic perturbations;
- terminal collision included/excluded consistently.

### Observation

- depth only;
- RGB + depth;
- without occluder points/mask;
- class-agnostic vs class token;
- clean-only training vs physical blocker augmentation.

### Efficiency

- labels per object;
- training wall clock;
- inference latency vs number of candidates;
- peak GPU memory;
- sensitivity to candidate count.

---

## 15. Falsification and kill criteria

### Kill 1 — terminal stiffness is not a useful oracle

If full-geometry relaxation cost does not predict short-lift stability beyond existing analytic scores, stop before visual model training.

### Kill 2 — rank deficiency is trivial

If almost every parallel-jaw grasp has the same nullspace because the chosen contact model cannot represent stabilizing friction/compliance, the spectral target is non-discriminative.

### Kill 3 — mean operator is enough

If `predict K -> exp` matches PSRN on high-ambiguity twins, the main non-semigroup claim is empirically irrelevant.

### Kill 4 — direct scalar critic is enough

If matched BCE/margin models equal or beat PSRN across success, regret, low-data and OOD tests, spectral supervision is ornamental.

### Kill 5 — structure gives no independent gain

If unconstrained time-conditioned matrix prediction matches PSRN without validity failures, positive spectral architecture lacks practical value.

### Kill 6 — quadratic Minty baseline matches exactly

If a quadratic proximal/Moreau head reproduces all gains, RelaxGrasp is insufficiently distinct from today's MintyGrasp and should not proceed as a separate method.

### Kill 7 — completion remains better at matched resources

If strong completion pipelines win materially at equal data/compute and PSRN has no latency/sample advantage, the SOTA hypothesis fails.

### Kill 8 — oracle sim-to-real mismatch

If predicted weak modes correlate in simulation but not with physical slips/lifts, contact stiffness assumptions are wrong for the hardware.

### Kill 9 — threshold fragility

If rankings change radically under small defensible changes of $\tau$, the rectified operator is not a stable scientific target.

### Kill 10 — candidate recall dominates

If no successful candidate exists in the common bank, selector comparison cannot support the core claim.

### Kill 11 — method secretly needs category recognition

If gains disappear in depth-only/category-held-out splits, claims must be narrowed to category-prior inference.

### Kill 12 — novelty collision

If citation chasing finds prior conditional matrix-relaxation learning with an equivalent score/head under coarsened observations, reposition as application or stop.

---

## 16. Adversarial novelty audit

### 16.1 Strongest hostile reviewer summary

> “This is a stiffness-based grasp metric fed through a mixture of exponentials. Grasp stiffness eigenvalues are classical, completely monotone mixtures are classical, physics-informed networks are classical, and MintyGrasp already learns structured contact response. A scalar grasp critic is simpler.”

This criticism is valid unless the paper demonstrates all of:

1. exact failure of mean/single-generator models on controlled ambiguity;
2. a theorem package centered on conditional mixtures leaving the semigroup class;
3. strict transform elicitation and valid matrix-measure architecture;
4. independent low-data/OOD/decision gain over scalar and quadratic-prox baselines;
5. real occluded parallel-jaw benefit at lower inference cost than completion.

### 16.2 Defensible novelty sentence

> We introduce conditional relaxation-spectrum learning: rather than reconstructing a coarsened physical state or fitting one average stability operator, we elicit the conditional mean matrix spectral measure through random-time relaxation scores and represent it with an identity-normalized positive-residue network that preserves non-decaying hidden modes while deliberately allowing the Bayes response to violate single-semigroup composition.

### 16.3 Claims that must not be made

- first occlusion-aware grasping;
- first use of stiffness/eigenvalues in grasping;
- first spectral grasping method;
- first completely monotone neural network;
- first physics-informed grasp network;
- guaranteed safe grasp;
- calibrated posterior over hidden shapes;
- exact force-closure probability;
- SOTA before controlled comparisons.

### 16.4 Nearest-work matrix

| Work family | Output | Core structure | Remaining delta |
|---|---|---|---|
| Classical grasp stiffness | one full-state stiffness/eigenvalue | contact mechanics | no learned conditional spectral mixture from occluded RGB-D |
| TARGO/ZeroGrasp/NeuGraspNet | completed/implicit geometry + grasp | reconstruction/rendering | no conditional mechanical relaxation measure |
| Uncertain completion/UNCLE | shape samples + robust aggregation | posterior shape sampling | RelaxGrasp amortizes only mean port spectrum; no shape samples |
| Direct grasp critic | scalar quality/probability | discriminative regression | no directional multiscale relaxation or structural validity |
| Port-Hamiltonian/physical NNs | dynamics/energy model | conservation/passivity | no coarsened conditional mixture or grasp selector |
| Completely monotone networks | scalar valid transforms/copulas | positive exponential mixtures | no query-conditioned matrix measure or terminal grasp mechanics |
| Prony / inverse-relaxation learning | spectrum fitted or inverted from an observed temporal response | positive exponential modes or inverse Laplace transform | input here is a coarsened static observation and action query; target is a conditional matrix Bayes measure, not recovery of the spectrum that generated a supplied time curve |
| MintyGrasp | expected Moreau value + prox response | convexity/integrability | RelaxGrasp target is non-semigroup matrix relaxation mixture; no optimization layer |

### 16.5 Current novelty verdict

- **Estimand novelty:** plausibly strong.
- **Objective novelty:** medium-to-strong; built from classical transform/MSE ingredients, but targets a new conditional matrix measure.
- **Architecture novelty:** strong if exact residue partition and non-semigroup argument are central.
- **Robotics novelty:** medium until search/experiments confirm no learned stiffness-spectrum grasp selector.
- **Collision risk with MintyGrasp:** material and must be experimentally audited.

---

## 17. ICLR 2027 audit

Official [ICLR 2027 Reviewer Guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines) asks four central questions and explicitly states that SOTA is not mandatory if a submission contributes convincing new, relevant and impactful knowledge.

### 17.1 Specific question

> How should a model represent local physical response when a coarsened observation leaves several hidden stability operators plausible and their conditional average is not itself a valid single relaxation semigroup?

This is more general than grasping and more specific than generic uncertainty learning.

### 17.2 Motivation and placement

Strong if paper clearly separates:

- reconstruction;
- one mean operator;
- posterior sampling;
- conditional mixture transform.

The diagonal twin must appear in the introduction, and classical stiffness/completely-monotone prior art must be acknowledged early.

### 17.3 Support for claims

A defensible submission needs:

1. proofs of conditional closure, semigroup misspecification, zero-mode limit, score propriety, approximation and regret;
2. MaskedPort general benchmark;
3. fixed-candidate RGB-D occlusion experiments;
4. strongest direct/reconstruction/quadratic-prox baselines;
5. physical robot validation or an unusually strong sim-to-real protocol;
6. compute/sample-efficiency curves.

Without items 1–4 it is not an ICLR general-ML paper.

### 17.4 Significant new knowledge

Potential substantial knowledge is not `PSRN improves grasping by N%`, but:

1. conditional averaging of latent relaxation semigroups leaves the semigroup model class;
2. a mean hidden operator can erase mutually exclusive weak directions;
3. the correct Bayes response remains inside a tractable matrix-completely-monotone class;
4. a positive spectral-measure head can learn it efficiently and expose a long-time weak-mode residue;
5. this representation can improve decisions under structured coarsening without hidden-state reconstruction.

If experiments validate only robotics gain but not 1–5, venue fit shifts toward CoRL/RSS.

### 17.5 Honest pre-experiment rating

- novelty: **7.2/10** after targeted search;
- technical depth: **8/10** if theorem package is rigorous, **5/10** without it;
- general-ML breadth: **8/10** with MaskedPort, **5.5/10** with grasping only;
- lab fit: **7.5/10**, limited by contact-stiffness oracle fidelity;
- empirical risk: **high**;
- current ICLR acceptance guess: **35–50% conditional on all gates**, not a calibrated probability.

### 17.6 Submission timing reality

ICLR 2027 full-paper deadline is [25 September 2026](https://iclr.cc/Conferences/2027/Dates), about one month after this research pass. A credible submission is realistic only if:

- synthetic scene/candidate infrastructure already exists;
- Gate 0 can be completed in days;
- MaskedPort and core proofs are parallelized immediately;
- robot experiments are already schedulable.

Otherwise forcing this deadline will likely produce a weak paper. The scientific idea is better preserved for the next suitable cycle than submitted without decisive baselines.

---

## 18. Preregistered success conditions

Exact thresholds should be frozen after a pilot but before final benchmark evaluation.

### Phenomenon

- high-ambiguity observations have materially larger oracle semigroup defect than clean observations;
- mean-operator baseline underestimates large-time weak-mode mass;
- defect/slow-mode mass predicts excess failure beyond visibility percentage alone.

### Objective

- CSLS improves integrated relaxation error by at least 15% relative to unconstrained time MLP or mean-K baseline on ambiguous/OOD strata;
- random-time learning matches dense-time supervision within 2% while using fewer oracle queries.

### Architecture

- PSRN produces zero structural violations by construction;
- positive-mixture structure improves low-data/OOD relaxation error by at least 10% over parameter-matched unconstrained head;
- $r\le8$ reaches within 3% of $r=16$ on decision regret.

### Grasping

- at fixed candidate bank, severe-occlusion top-1 short-lift success improves by at least 5 pp over the strongest direct scalar/stiffness baseline with 95% object-bootstrap CI excluding zero;
- result remains after controlling for candidate recall and collision filter;
- inference memory or latency is at most one third of stochastic completion pipeline, or another preregistered material efficiency gain;
- real-robot paired improvement is directionally consistent and statistically credible.

These are project gates, not fabricated expected results.

---

## 19. Minimum implementation roadmap

### Phase 0 — 3–5 days: oracle audit

1. implement/validate normalized tangent stiffness extraction;
2. create $K=(\widetilde H-\tau I)_+$;
3. compute relaxation cost analytically;
4. compare with existing metrics and lift outcomes;
5. stop if Gate 0 fails.

### Phase 1 — 4–7 days: MaskedPort

1. generate ambiguous spring/truss systems;
2. implement mean-K, unconstrained-time and PSRN heads;
3. verify exact structural identities numerically;
4. run low-data/OOD curves;
5. draft theory alongside experiments.

### Phase 2 — 1–2 weeks: frozen-feature grasp reranking

1. cache scene encoder features/candidate banks;
2. precompute $6\times6$ oracle spectra;
3. train matched heads;
4. evaluate severity and ambiguity strata;
5. decide whether full visual training is justified.

### Phase 3 — 2–4 weeks: full visual model

1. train sparse RGB-D encoder + PSRN;
2. compare completion and direct baselines;
3. profile compute;
4. finalize sim-to-real randomization.

### Phase 4 — 2–4 weeks: real robot

1. preregister object/blocker protocol;
2. randomize execution order;
3. collect paired trials;
4. analyze by-object bootstrap CIs;
5. audit physical failure videos against predicted weak directions.

### Phase 5 — paper hardening

1. author-level citation chase;
2. proof checking;
3. release MaskedPort and oracle code;
4. document negative results and threshold sensitivity;
5. remove every unsupported safety/SOTA claim.

---

## 20. Scientific unit tests

Before long training, PSRN must pass:

1. `H(0) == I` to numerical tolerance;
2. every $H(t)$ symmetric PSD;
3. $H(t_2)\preceq H(t_1)$ for $t_2>t_1$;
4. alternating derivative signs through at least order 4;
5. long-time limit equals $A_0$;
6. residue sum equals identity;
7. exact recovery of the 2-D ambiguity example with two atoms;
8. inability of one-generator baseline to fit that example;
9. equivariance/invariance under global scene/gripper transform after coordinate normalization;
10. jaw-exchange $C_2$ symmetry;
11. decision integral agrees with dense numerical quadrature;
12. gradients remain finite for $t_{max}$ and near-zero poles;
13. no performance change from candidate ordering;
14. collision gate identical across all methods.

---

## 21. Draft paper pitch

### Possible title

> **Hidden Mechanics Do Not Average: Conditional Spectral Relaxation Learning from Occluded Observations**

Robotics-facing alternative:

> **RelaxGrasp: Learning Mixtures of Mechanical Relaxation for Grasping through Occlusion**

### One-sentence contribution

> We show that coarsening-induced averages of local physical semigroups are generally not semigroups, and introduce a strictly elicited positive spectral-mixture representation that preserves hidden non-decaying modes and enables completion-free grasp selection from one occluded RGB-D view.

### Abstract skeleton

Partial observations can leave several hidden physical operators compatible with the same input. Reconstructing one state or predicting one average operator is then structurally misspecified: the conditional average of their relaxation semigroups is generally not a semigroup, and averaging operators can erase mutually exclusive weak modes. We formulate coarsened relaxation-spectrum learning, whose target is the conditional mean matrix spectral measure of a query-local stability operator. We introduce the Conditional Spectral Laplace Score, a random-time proper score for this measure, and the Positive Spectral Relaxation Network, which represents its transform as an identity-normalized sum of positive matrix residues and exponential modes. The architecture guarantees complete monotonicity while allowing a non-decaying residue that equals the expected projector onto unresolved weak modes. We prove target closure, strict elicitation, finite-atom approximation and decision-regret results. On a general masked physical-system benchmark and single-view parallel-jaw grasping under foreground occlusion, the method must be tested against hidden-state reconstruction, direct critics and single-operator structured models. No performance numbers should be written until those experiments exist.

---

## 22. Final decision

**Recommendation: pursue only through Gate 0 and Gate 1 immediately.**

RelaxGrasp is genuinely separated from today's selected occlusion methods by estimand, objective and architecture:

- estimand: conditional mean matrix relaxation spectrum;
- objective: random-time matrix Laplace transform score;
- architecture: positive-residue non-semigroup relaxation mixture;
- new knowledge: hidden operators do not average into one valid Bayes semigroup, and their distinct weak modes survive as a long-time residue.

Its strongest qualities are:

- exact ambiguity counterexample;
- compact $6\times6$ query-local target;
- hard structural validity;
- plausible compute advantage over completion;
- broad coarsened-physical-learning interpretation.

Its strongest risks are:

- terminal stiffness may be a poor proxy for real short-lift success;
- two-finger contact model may make weak modes trivial;
- scalar critic may already learn everything needed;
- reviewer may collapse the idea into quadratic Minty/contact-response learning;
- ICLR 2027 timing is extremely tight.

The project becomes an ICLR-level contribution only if it establishes **new knowledge about conditional mixtures of physical relaxation**, not merely a new grasp-quality head.

---

## 23. Primary sources checked

### Occlusion and robotic grasping

1. TARGO / TARGO-Net project and benchmark: https://targo-benchmark.github.io/
2. TARGO paper: https://arxiv.org/abs/2407.06168
3. ZeroGrasp, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/papers/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.pdf
4. NeuGraspNet, RSS 2024: https://www.roboticsproceedings.org/rss20/p046.pdf
5. Local Occupancy-Enhanced Grasping, ECCV 2024: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09354.pdf
6. TOSC, AAAI 2026: https://ojs.aaai.org/index.php/AAAI/article/download/38053/42015
7. Robust Grasp Planning over Uncertain Shape Completions: https://arxiv.org/abs/1903.00645
8. Measuring Uncertainty in Shape Completion to Improve Grasp Quality: https://arxiv.org/abs/2504.16183
9. UNCLE-Grasp: https://arxiv.org/abs/2601.14492
10. FFHFlow, CoRL 2025: https://proceedings.mlr.press/v305/feng25a.html

### Grasp mechanics

11. Enabling Grasp Action via Contact Stiffness: https://arxiv.org/abs/1810.08317
12. On-Orbit Robotic Grasping Stability Analysis: https://pmc.ncbi.nlm.nih.gov/articles/PMC8247653/
13. Grasp quality measures review: https://doi.org/10.1007/s10514-014-9402-3
14. Compliance effects in a parallel jaw gripper: https://doi.org/10.1016/S0094-114X(03)00100-9
15. Passive Reaction Analysis for Grasp Stability: https://arxiv.org/abs/1801.06558

### Structured physical learning

16. Compositional Port-Hamiltonian Neural Networks: https://proceedings.mlr.press/v211/neary23a.html
17. Invariant Neural Operators: https://proceedings.mlr.press/v206/liu23f.html
18. Differentiable Contact Models for Lagrangian/Hamiltonian NNs: https://proceedings.neurips.cc/paper_files/paper/2021/hash/b7a8486459730bea9569414ef76cf03f-Abstract.html
19. Localized Neural Operators, ICML 2024: https://proceedings.mlr.press/v235/liu-schiaffini24a.html

### Matrix transforms and completely monotone structure

20. Matrix-valued completely monotone kernels / viscoelasticity: https://arxiv.org/abs/2106.07946
21. Matrix-valued Stieltjes functions: https://arxiv.org/abs/2206.02926
22. Deep Archimedean Copulas, NeurIPS 2020: https://papers.nips.cc/paper_files/paper/2020/file/10eb6500bd1e4a3704818012a1593cc3-Paper.pdf
23. Rational Krylov for Stieltjes matrix functions: https://arxiv.org/abs/1908.02032

### Venue

24. ICLR 2027 Reviewer Guide: https://iclr.cc/Conferences/2027/ReviewerGuidelines
25. ICLR 2027 dates: https://iclr.cc/Conferences/2027/Dates

### Prior art on Prony and inverse relaxation spectra

26. Prony representation of stretched-exponential relaxation: https://arxiv.org/abs/1803.07706
27. Neural fitting of Prony relaxation curves: https://www.ijpe-online.com/article/2019/0973-1318/0973-1318-15-1-107.shtml
28. Direct identification of continuous relaxation spectra: https://pmc.ncbi.nlm.nih.gov/articles/PMC11478369/
29. Deep inverse-Laplace reconstruction of NMR relaxation spectra: https://pmc.ncbi.nlm.nih.gov/articles/PMC12383262/
