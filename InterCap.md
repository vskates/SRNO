# Learn the interaction, not the shape

## Conditional capacity learning for occlusion-robust parallel-jaw grasp selection

**Статус документа:** независимый research memo, 25 августа 2026.  
**Предлагаемое имя метода:** **InterCap** (Interaction-Capacity Network).  
**Основная ставка:** новый general-ML objective, а не новая комбинация grasping-модулей.  
**Ограничения:** без RL, без VLA, без реконструкции полного объекта, без оценки всего цикла reach–grasp–lift.  
**Честная формулировка novelty:** exhaustive novelty невозможно доказать поиском; ниже указано, что именно было найдено и какие claims всё ещё требуют дополнительного librarian-style поиска перед submission.

---

## 1. Итоговый вердикт

Предлагается не восстанавливать скрытую форму и не предсказывать один scalar grasp-success score. Вместо этого следует учить **условный закон минимальных геометрических событий взаимодействия**, которые gripper может проверить своим телом и губками: попадание скрытой поверхности в запрещённый swept volume, существование доступной пары противоположных контактов в заданной margin-ячейке, несовместимость ширины и т. п.

Полная скрытая геометрия при фиксированном grasp (g) детерминированно порождает конечное множество таких **interaction witnesses**. Из-за окклюзии это множество случайно условно на единственном RGB-D observation. Его закон предлагается представлять не mesh, SDF, occupancy grid или point samples, а **capacity functional** — вероятностями hit/miss для небольших gripper-shaped query sets.

Ключевой learning objective — **joint capacity log score**:

\[
\mathcal L_{\mathrm{cap}}
=
-\mathbb E\log p_\theta
\!\left(h_g(S)\mid X,g\right),
\]

где $h_g(S)\in\{0,1\}^m$ — совместный hit pattern на $m$ непересекающихся ячейках interaction space. В отличие от $m$ независимых BCE, objective является strictly proper для **совместного** закона событий и тем самым сохраняет корреляции между скрытыми контактами и collision. Эти корреляции принципиальны для формулы вида «body misses **и** существует согласованная левая–правая contact pair».

Эффективная параметризация — conditional nonnegative-rank factorization вероятностного тензора:

\[
p_\theta(h\mid X,g)
=
\sum_{k=1}^{K}\pi_k(X,g)
\prod_{j=1}^{m}
a_{kj}(X,g)^{h_j}
\bigl(1-a_{kj}(X,g)\bigr)^{1-h_j}.
\]

Это гарантированно задаёт корректный joint distribution и, следовательно, корректную completely-alternating capacity на конечной query algebra; стоимость — $O(Km)$, без $2^m$-head, shape samples и 3-D decoding. $K=1$ ровно даёт слабый independent-events baseline, а при $K\le 2^m$ семейство способно точно представить любой закон на $\{0,1\}^m$.

**Почему это кандидат на ICLR, а не ещё один grasp pipeline:** центральный объект обучения — условная capacity неизвестного random set по partial observation; роботизированный grasp является строгим и полезным instantiation. Из постановки следуют properness, capacity coherence, decision sufficiency и plug-in decision-regret bound. Архитектура — лишь эффективный conditional estimator этого нового объекта.

**Почему это может быть сильнее существующих методов:** reconstruction-based методы тратят capacity на геометрию, которая никогда не пересекается с gripper queries; direct scalar scorers не моделируют joint event law и не могут менять collision/contact criterion после обучения. InterCap учит ровно тот quotient скрытой формы, который достаточен для выбора grasp, и аналитически marginalizes (K) interaction modes за (O(Km)).

---

## 2. Точная задача и границы claim

### 2.1 Setup

Имеется:

- humanoid robot с wrist RGB-D camera;
- rigid target object на полке;
- один foreground obstacle, частично закрывающий target; это **не clutter**;
- parallel-jaw gripper;
- одно noisy RGB-D observation;
- короткое действие: подвести уже выбранный terminal grasp, закрыть губки и поднять объект на несколько сантиметров.

Вход метода:

\[
X=(P, V),
\]

где (P) — сегментированное noisy partial point cloud target/foreground/shelf, а (V) — компактное описание known-free / occluded rays из depth image. Маска target считается данным upstream input; open-vocabulary grounding не является вкладом.

Кандидат:

\[
g\in \mathrm{SE}(3)\times [w_{\min},w_{\max}].
\]

Candidate generator намеренно не является вкладом. Для чистого сравнения все rerankers получают один и тот же set (G(X)), сгенерированный сильной существующей моделью и дополненный локальными perturbations.

Выход InterCap — не grasp trajectory и не shape. Для каждого $g$ это условный закон небольшого hit pattern и derived score:

\[
\operatorname{score}_\alpha(g)
=
\sup\bigl\{\tau:
\Pr_\theta(C_g=0,\ M_g\ge \tau\mid X,g)\ge 1-\alpha
\bigr\}.
\]

Здесь (C_g) означает terminal/small-sweep collision, а (M_g) — дискретизированный antipodal-contact margin. Это **не** исполнимость всего approach/lift cycle.

### 2.2 Что paper утверждает

1. При скрытой геометрии правильный uncertainty object для grasp selection — не posterior over shapes, а posterior over a task-induced random set of interaction witnesses.
2. Conditional capacity на grasp-query algebra является достаточной статистикой для любого utility, измеримого относительно этих witnesses.
3. Joint capacity log score учит этот объект непосредственно и корректно; independent per-event BCE и scalar success BCE теряют нужную структуру.
4. Low-rank conditional capacity можно вычислять достаточно быстро для reranking десятков или сотен grasпов.

### 2.3 Что paper не утверждает

- абсолютную safety guarantee вне training distribution;
- работу с deformable, transparent или articulated objects;
- неизвестный friction/material без дополнительной random variable;
- решение target segmentation;
- collision-free motion planning всего манипулятора;
- SOTA до проведения заявленного experiment suite.

---

## 3. Карта литературы: что уже сделано

Поиск охватывал работы до 25.08.2026 по single-view/partial-point-cloud grasping, occlusion, uncertain shape, learned collision checking, task-oriented completion, probabilistic grasp generation, decision-focused learning, conformal prediction и random closed sets. Ниже только направления, реально ограничивающие novelty.

### 3.1 Direct grasp fields и candidate generation

- [GraspNet-1Billion](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html) дал единый benchmark, более миллиарда аналитически размеченных grasps и point-cloud network, но его стандартная метрика не изолирует severe foreground occlusion целевого объекта.
- [Contact-GraspNet](https://arxiv.org/abs/2103.14127) генерирует distribution of 6-DoF parallel-jaw grasps прямо из depth, привязывая grasp к наблюдаемому contact point; на structured clutter сообщено более 90% success. Это сильный generator, но скрытый контакт, отсутствующий в point cloud, не получает отдельного probabilistic event law.
- [Graspness/GSNet](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.html) показывает, что ранняя фильтрация graspable regions даёт более 30 AP улучшения и высокую скорость. Это аргумент за общий сильный candidate set, но не решение ambiguity скрытой формы.
- [AnyGrasp](https://arxiv.org/abs/2212.08333) сообщает 93.3% bin-clearing success и устойчивость к depth noise; его цель — dense spatial-temporal grasp perception, не условный закон скрытых contact/collision events.
- [OrbitGrasp](https://proceedings.mlr.press/v270/hu25b.html) моделирует непрерывную grasp-quality function на (S^2) через spherical harmonics и показывает преимущество SE(3)-equivariance. Это сильное основание использовать canonical/equivariant query encoder, но output остаётся quality function, а не coherent joint uncertainty law.

**Gap:** эти методы в основном решают generation/ranking по видимой геометрии. Простое добавление ещё одного scalar success head не является новым objective.

### 3.2 Полная или локальная shape completion

- [Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645) семплирует полные voxel shapes через MC dropout и усредняет grasp metric. Реальный success вырос с 48% до 59%, что является прямым доказательством полезности shape uncertainty, но completion и evaluation заняли соответственно 86.73 s и 83.25 s в опубликованной реализации.
- [Diverse Plausible Shape Completions](https://proceedings.mlr.press/v155/saund21a.html) показывает, что single depth image допускает несколько существенно разных plausible completions, и демонстрирует пользу при grasping occluded objects. Это опровергает point-estimate completion как достаточное решение.
- [Local Occupancy-Enhanced Object Grasping](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09354.pdf) реконструирует только локальные grasp regions, а не всю сцену; реальный test превосходит baseline без occupancy примерно на 6.62%. Это ближайший аргумент, что task-local geometry полезна, но метод всё ещё supervised by voxel occupancy и декодирует локальную форму.
- [NeuGraspNet](https://openreview.net/forum?id=Fdu33eoZas) учит implicit scene geometry и локальное neural surface rendering; на hard viewpoints он заметно сильнее нескольких implicit/semi-implicit baselines. Но occupancy reconstruction является core component метода.
- [PCF-Grasp](https://arxiv.org/abs/2504.16320) использует completion как features и сообщает +17.8 percentage points в real-world success относительно SOTA baseline. Это косвенно подтверждает ценность shape prior, но сохраняет completion stage.
- [TOSC](https://arxiv.org/abs/2601.05499) уже формулирует task-oriented shape completion и восстанавливает потенциальные contact regions вместо всего объекта; сообщено улучшение grasp displacement на 16.17% и Chamfer distance на 55.26%. Поэтому «давайте восстанавливать только contact surface» уже **не novelty**.

**Gap:** даже наиболее task-oriented варианты возвращают geometry. Они оптимизируют voxel/point reconstruction loss, хотя downstream decision использует лишь конечный набор intersection/contact predicates.

### 3.3 Compact contact/collision representations

- [SceneCollisionNet](https://arxiv.org/abs/2011.10726) учит collision query между point clouds без object models; сообщено +9.8% collision accuracy и 75× speedup против лучшего baseline. Это сильное косвенное доказательство, что query-conditioned interaction prediction может заменить explicit geometry для downstream reasoning.
- [CADGrasp](https://arxiv.org/abs/2601.15039) — важнейший closest work. Он предсказывает sparse interaction bisector surface (IBS), содержащую contact/collision information, затем оптимизирует dexterous grasp. Однако представление остаётся $40^3$ voxel tensor, генерируется occupancy diffusion, требует grasp optimization и занимает 6.51 s; реальный success 93.8% против 83.9% baseline. Его ablation также показывает падение 86.5% → 56.1% без decomposition contact representation. Это подтверждает полезность structured contact information, но не предлагает conditional capacity, proper joint hit objective или reconstruction-free event algebra.
- [SpringGrasp](https://arxiv.org/abs/2404.13532) учитывает GPIS shape uncertainty в differentiable compliant-grasp metric и сообщает как минимум +18% success против force-closure planner. Он ориентирован на dexterous compliant hand и всё равно строит uncertain surface representation.

**Gap:** learned collision query обычно даёт отдельный point estimate; contact/collision intermediate representations остаются геометрическими и высокоразмерными. Не найден метод, учащий совместный закон Boolean interaction algebra без geometry output.

### 3.4 Probabilistic grasp generation не равно probabilistic grasp evaluation

- [6-DOF GraspNet](https://openaccess.thecvf.com/content_ICCV_2019/html/Mousavian_6-DOF_GraspNet_Variational_Grasp_Generation_for_Object_Manipulation_ICCV_2019_paper.html), [Grasp Diffusion Network](https://arxiv.org/abs/2412.08398) и FFHFlow моделируют multimodal $p(g\mid X)$. Это полезно для diversity, но likelihood grasp pose не является вероятностью физического успеха и не задаёт совместный закон hidden contact events для заданного $g$.
- [Deep Learning a Grasp Function](https://arxiv.org/abs/1608.02239) сглаживает grasp score по pose uncertainty. Это uncertainty по исполнению pose, а не ambiguity скрытой формы.

**Gap:** генеративная неопределённость по действиям и epistemic/aleatoric uncertainty взаимодействия — разные объекты.

### 3.5 General ML foundations

- [Decision-Focused Learning through Learning to Rank](https://proceedings.mlr.press/v162/mandi22a.html) показывает, что downstream decision quality можно оптимизировать как ranking, а ограниченный solution set контролирует runtime с небольшим влиянием на regret.
- [Decision Trees for Predict-then-Optimize](https://proceedings.mlr.press/v119/elmachtoub20a.html) показывает, что decision loss способен дать лучшие решения и меньшую complexity, чем training на parameter prediction error. Это поддерживает отказ от geometry loss, но обычный decision-focused loss сам по себе уже не нов.
- [Risk-Controlling Prediction Sets](https://arxiv.org/abs/2101.02703) дают distribution-free expected-risk control. Conformal calibration полезна как post-hoc extension, но «conformal grasp score» недостаточен как основной ICLR contribution.
- [Deep Sets](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) и [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) дают permutation-invariant universal/attention architectures; induced attention снижает self-attention от quadratic к linear по числу элементов.
- Neural Processes моделируют distributions over functions conditioned on partial observations. Они вдохновляют shared latent interaction modes, но generic neural process over scalar grasp function не обеспечивает hit/union logic и capacity coherence.

### 3.6 Random closed sets и capacity

Для random closed set (W) capacity functional

\[
T(K)=\Pr(W\cap K\ne\varnothing)
\]

играет роль CDF. Choquet–Kendall–Matheron theorem утверждает, что при стандартных topological assumptions закон random closed set однозначно определяется normalized, upper-semicontinuous, completely-alternating capacity. Краткая формулировка условий доступна в [Algorithmic Randomness and Capacity of Closed Sets](https://arxiv.org/abs/1106.2993); современная работа также прямо формулирует уникальность закона по capacity [здесь](https://www.sciencedirect.com/science/article/abs/pii/S0165011421003699).

Random closed sets применялись в старой робототехнике для statistical environment maps и localization, например [Rolfes & Rendas](https://www.sciencedirect.com/science/article/abs/pii/S0921889002002609). Следовательно, claim «random sets впервые в robotics» был бы ложным. Но в проведённом поиске не найдены:

1. conditional neural capacity estimator для grasp-induced contact/collision query algebra;
2. proper joint hit objective для partial RGB-D grasp selection;
3. task-sufficiency quotient скрытых форм через interaction-witness capacity;
4. low-rank conditional capacity tensor, используемый для exact Boolean grasp marginalization.

Именно их сочетание, а не сам термин capacity, составляет novelty claim.

---

## 4. Итерации, которые следует сознательно отвергнуть

### Вариант A: posterior shape samples + CVaR grasp metric

**Почему казался разумным:** естественно marginalize hidden geometry и брать lower tail.  
**Почему отвергнут:** уже есть uncertain shape completion и robust planning; CVaR стандартен; inference требует shape generation/evaluation; нарушает запрет на reconstruction.

### Вариант B: direct scalar success probability + conformal lower bound

**Почему казался разумным:** прост, быстр, легко калибруется.  
**Почему отвергнут:** $p(\text{success}\mid X,g)$ — обычная supervised classification; conformal layer не создаёт substantial new knowledge. Scalar не отвечает, *почему* hidden geometry делает grasp рискованным, и не переносится на другой friction/width criterion.

### Вариант C: neural process над целой grasp-quality function (q_S(g))

**Почему казался разумным:** coherent function samples вместо независимых scores.  
**Почему отвергнут:** generic conditional neural process уже известен. Для risk-neutral выбора одного grasp expected regret относительно shape-specific oracle имеет тот же argmax, что и expected score; поэтому заявленная joint-function сложность часто не меняет решение. Minimax regret добавляет известный robust-decision objective, но не новую grasp-relevant structure.

### Вариант D: intervals, которые обязаны сужаться при раскрытии новых points

**Почему казался разумным:** больше visibility должно уменьшать uncertainty.  
**Почему отвергнут математически:** conditional quantile interval не обязан быть pointwise nested при conditioning; новое evidence может сдвинуть весь posterior. Правильна tower property в ожидании, а не произвольное nestedness для каждого observation. Такое ограничение могло бы обучать неверный posterior.

### Вариант E: восстановить только contact regions / local occupancy / IBS

**Почему казался разумным:** меньше полной сцены и ближе к задаче.  
**Почему отвергнут:** TOSC, local occupancy-enhanced grasping и CADGrasp уже занимают этот участок. Новый voxel/contact decoder выглядел бы модификацией существующей robotics line, именно чего следует избежать.

### Вариант F: независимые first-contact hazards для двух губок

**Почему казался разумным:** survival analysis вдоль jaw closing trajectory дёшев и физически интерпретируем.  
**Почему отвергнут как финальный objective:** независимые hazards не сохраняют correlation, вызванную одной и той же hidden shape. Stable grasp — conjunction согласованных левого/правого контактов и collision miss; произведение независимо откалиброванных marginals систематически искажает её вероятность.

### Выживший вариант: conditional capacity of interaction witnesses

Он сохраняет физически нужную joint structure, не реконструирует geometry, имеет proper objective и допускает теоретически чистую task-sufficiency формулировку.

---

## 5. Broad open problem

### 5.1 Общая постановка

Пусть latent world (S) наблюдается через many-to-one sensing map:

\[
X\sim p(X\mid S).
\]

Action (g) не требует знать весь (S); он взаимодействует с ним через детерминированный **witness operator**

\[
\mathcal W_g:S\mapsto W_g(S)\subseteq \mathcal E_g,
\]

где (mathcal E_g) — компактное canonical interaction space. При partial observation (W_g(S)) становится conditional random closed set.

**Open ML question:** можно ли выучить $\mathcal L(W_g\mid X)$ только через action-shaped set queries и proper scoring, не восстанавливая $S$, причём так, чтобы:

- ответы на разные queries были probabilistically coherent;
- произвольные Boolean downstream constraints вычислялись аналитически;
- representation была sufficient для decision family;
- вычисления зависели от числа queries, а не от resolution latent world?

Grasping — наглядный special case, но тот же objective применим к occluded collision checking, inspection, tool insertion, medical probe placement и visibility-limited design verification.

### 5.2 Почему это не просто «predict task labels»

Один label фиксирует один downstream utility. Capacity на query algebra даёт **перенастраиваемый закон взаимодействия**: после обучения можно менять allowable friction cone, gripper width tolerance, collision dilation и risk level, выбирая другие unions ячеек без повторного shape learning. Это промежуточная точка между дорогим world model и негибким scalar predictor.

---

## 6. Минимальная математическая формализация для parallel-jaw gripper

### 6.1 Witness space

Для каждого grasp (g) переводим gripper и локальную observed geometry в canonical gripper frame. Full mesh нужен **только offline** для вычисления witness set.

Определим disjoint-union space

\[
\mathcal E_g
=
\mathcal E_g^{\mathrm{pair}}
\;\sqcup\;
\mathcal E_g^{\mathrm{forbid}}.
\]

- $\mathcal E_g^{\mathrm{pair}}$ содержит только пары first-reachable jaw contacts, представленные compact contact-pair descriptor: closure coordinate, pad coordinate и antipodal/friction margin.
- $\mathcal E_g^{\mathrm{forbid}}$ содержит intersections hidden solid с terminal gripper body и очень коротким final insertion/closing sweep.

Не нужно предсказывать эти coordinates как dense field. Пространство заранее разбивается на $m$ непересекающихся, физически определённых cells:

\[
\mathcal P_g=\{K_1(g),\ldots,K_m(g)\}.
\]

Пример для $m=12$: одна collision cell; восемь contact-pair cells по closure/antipodal-margin; три cells для pad-edge/width slack. Конкретная partition — hyperparameter, а не набор новых latent scene variables.

### 6.2 Hit vector

\[
h_{g,j}(S)
=
\mathbf 1\!\left[W_g(S)\cap K_j(g)\ne\varnothing\right],
\qquad j=1,\dots,m.
\]

Поскольку $S$ неизвестна после observation, target объекта обучения:

\[
p^*(h\mid X,g),\qquad h\in\{0,1\}^{m}.
\]

Для любого subset (A\subseteq[m]) avoidance functional на union cells равен

\[
Q^*_{X,g}(A)
=
\Pr\!\left(
W_g\cap\bigcup_{j\in A}K_j=\varnothing
\mid X,g
\right)
=
\sum_{h:\,h_j=0\ \forall j\in A}p^*(h\mid X,g),
\]

а capacity (T^*=1-Q^*). Таким образом joint hit law и capacity на конечной algebra эквивалентны через Möbius/inclusion–exclusion relations.

### 6.3 Grasp event без длинной state formulation

Пусть cell $K_0$ — forbidden collision, а $R_\tau\subseteq\{1,\dots,m-1\}$ — contact-pair cells с analytic margin не меньше $\tau$. Тогда

\[
\Phi_{g,\tau}(h)
=
\neg h_0
\;\wedge\;
\bigvee_{j\in R_\tau}h_j.
\]

Это ровно terminal geometric viability: gripper body не пересекает скрытую geometry и существует reachable antipodal pair требуемого качества. Ни trajectory state, ни full-scene SDF, ни causal failure taxonomy не вводятся.

### 6.4 Risk-aware selection

\[
P_{\theta,\tau}(g\mid X)
=
\Pr_\theta(\Phi_{g,\tau}=1\mid X,g).
\]

Выбираем не максимальный mean analytic score, а максимальный posterior lower margin:

\[
g^*
=
\arg\max_{g\in G(X)}
\sup\{\tau:P_{\theta,\tau}(g\mid X)\ge1-\alpha\}.
\]

$\alpha$ задаёт reliability level. Tie-breaker — larger $P_{\theta,0}$, затем observed clearance.

---

## 7. Новый learning objective

### 7.1 Conditional low-rank capacity tensor

Joint probability tensor $p(h\mid X,g)$ имеет $2^m$ cells. InterCap использует conditional mixture of product Bernoullis:

\[
p_\theta(h\mid X,g)
=
\sum_{k=1}^{K}\pi_{\theta k}(X,g)
\prod_{j=1}^{m}
a_{\theta kj}(X,g)^{h_j}
\bigl(1-a_{\theta kj}(X,g)\bigr)^{1-h_j}.
\]

$\pi_k\ge0$, $\sum_k\pi_k=1$, $a_{kj}\in(0,1)$. Shared component $k$ — не shape sample; это low-dimensional **interaction mode**, определённый только через hit statistics.

Вероятность grasp event вычисляется точно:

\[
P_{\theta,\tau}(g\mid X)
=
\sum_{k=1}^{K}\pi_k
(1-a_{k0})
\left[1-\prod_{j\in R_\tau}(1-a_{kj})\right].
\]

Никакого Monte Carlo at inference не требуется.

### 7.2 Capacity Log Score (CapLog)

Основной loss:

\[
\boxed{
\mathcal L_{\mathrm{CapLog}}(\theta)
=
-\mathbb E_{S,X,g}
\log\left[
\sum_{k=1}^{K}\pi_{\theta k}(X,g)
\prod_{j=1}^{m}
a_{\theta kj}^{h_{g,j}(S)}
(1-a_{\theta kj})^{1-h_{g,j}(S)}
\right]
}
\]

Важно обучать log probability **целого pattern**, а не сумму независимых BCE после marginalization. Сумма BCE правильна только для $K=1$ / условной независимости hit cells и не штрафует неверные correlations.

### 7.3 Atomic query algebra и criterion randomization

В основной, теоретически чистой версии $K_1,\ldots,K_m$ образуют фиксированную **fine atomic partition** canonical interaction space. Friction margin, width slack и collision clearance дискретизируются один раз; nested collision dilations представлены непересекающимися shells. Любой допустимый downstream criterion затем является Boolean formula над unions этих atoms.

На training и validation случайно выбираются:

- friction-cone threshold;
- width/pad tolerance;
- collision dilation;
- risk level $\alpha$.

Эти choices меняют только analytic event formula, но не target hit pattern. Поэтому один и тот же learned joint law обслуживает много criteria, а ответы остаются логически coherent внутри конечной algebra. Это сильнее фиксированного 12-way classifier, но честно слабее exact capacity на всех compact subsets: произвольный новый threshold аппроксимируется с resolution atomic partition. Multi-resolution $m$-ablation должна показать эту погрешность.

Вариант с jittered cell boundaries допустим лишь как последующая extension с отдельным projective-consistency loss между coarse и refined partitions; он не нужен для главного claim.

### 7.4 Необязательные, но не центральные regularizers

1. **Mode usage:** слабый entropy floor на batch-average $\pi$, чтобы предотвратить преждевременный collapse.
2. **Occlusion counterfactual consistency:** один full object рендерится с разными blockers; loss остаётся обычным CapLog на каждом observation. Не следует навязывать pointwise nested uncertainty.
3. **Observed-geometry hard constraints:** если query полностью лежит в measured free space, соответствующий hit probability должен быть нулевым; если measured surface заведомо пересекает cell, единичным. Это exact sensor evidence, не learned completion.

Не следует добавлять reconstruction auxiliary loss в main model: он размоет центральный claim.

---

## 8. Теоретические результаты, которые реально можно доказать

### Proposition 1: finite capacity coherence

Любой $p_\theta(h\mid X,g)$ из mixture model является корректным probability distribution на hit patterns. Индуцированные $T_\theta(A)=1-Q_\theta(A)$ normalized, monotone и completely alternating на конечной algebra cells.

**Смысл:** модель не может выдать логически несовместимые ответы вроде (T(K_1)>T(K_1\cup K_2)), если union queries вычисляются из joint law.

### Proposition 2: strict propriety

При неограниченном model class conditional expected CapLog единственно минимизируется в $p_\theta(h\mid X,g)=p^*(h\mid X,g)$ почти наверное. Следовательно, objective одновременно elicит все hit, miss, union и Boolean-event probabilities на выбранной algebra.

### Proposition 3: representation universality

Любой distribution на $\{0,1\}^m$ можно точно представить mixture of at most $2^m$ deterministic product Bernoullis: один component на каждый pattern, $a_{kj}\in\{0,1\}$. Поэтому ограничение $K\ll2^m$ является low nonnegative-rank inductive bias, а не принципиальной потерей expressivity.

### Theorem 1: interaction-quotient sufficiency

Пусть downstream utility имеет вид

\[
u(S,g)=\tilde u(h_g(S),g).
\]

Тогда Bayes-optimal action зависит от latent-shape posterior $p(S\mid X)$ только через $p(h_g\mid X,g)$:

\[
\mathbb E[u(S,g)\mid X]
=
\sum_h \tilde u(h,g)p(h\mid X,g).
\]

Две hidden shapes, порождающие одинаковые witness patterns для всех $g\in G$, decision-equivalent. Их различия не должны занимать representation capacity. Это формализует «не реконструировать task-irrelevant geometry» без нестрогой information-bottleneck риторики.

### Corollary: plug-in decision regret

Если $u\in[0,1]$ и для всех candidates

\[
\operatorname{TV}
\bigl(p_\theta(h\mid X,g),p^*(h\mid X,g)\bigr)
\le\varepsilon,
\]

то regret grasp, выбранного plug-in maximization, не превосходит $2\varepsilon$. Это напрямую связывает качество conditional capacity с downstream selection и даёт осмысленную calibration metric.

### Что не надо обещать теоретически

- convergence реальной deep network к true posterior;
- safety под arbitrary distribution shift;
- точность rigid/contact simulator для реального friction;
- полную Choquet uniqueness при конечном наборе query cells. Uniqueness здесь только для restricted finite interaction algebra; это сознательная task quotient.

---

## 9. Архитектура InterCapNet

### 9.1 Observation encoder

Один sparse point encoder обрабатывает (P) с тремя token types: target surface, foreground/shelf surface, free/occluded ray evidence. Можно использовать sparse point transformer; архитектурный claim не должен зависеть от конкретного fashionable backbone.

Выход:

- global context token (z_X);
- point/ray tokens с positions и local features.

### 9.2 Canonical grasp-query encoder

Для каждого (g):

1. локальные tokens переводятся в gripper frame;
2. (m) query cells кодируются своими границами и mark type;
3. cell tokens cross-attend только к (k_{nn}) nearby observation tokens;
4. permutation-invariant pooling формирует (z_{X,g,j}).

Canonicalization даёт exact invariance к совместному rigid transform input+grasp; полноценный equivariant backbone можно проверить ablation, но не делать обязательным.

### 9.3 Low-rank capacity head

- mode head: $\pi_{1:K}=\operatorname{softmax}(f_\pi(z_X,z_g))$;
- hit head: (a_{kj}=\sigma(f_a(z_X,z_g,z_{X,g,j},e_j,k))).

Один observation encoding переиспользуется для всех (M) candidates. При (M=128,m=12,K=8) head вычисляет всего (12{,}288) Bernoulli parameters до batching overhead.

### 9.4 Analytic event layer

Event layer не имеет trainable parameters. Она получает требуемые $(\alpha,\tau)$, строит $R_\tau$ и вычисляет exact formula для $P_{\theta,\tau}$. Это отделяет learned uncertainty law от пользовательского risk preference.

### 9.5 Почему это не implicit reconstruction

- нет function (o(x):\mathbb R^3\to[0,1]);
- нет SDF/mesh/point-cloud decoder;
- нельзя визуализировать complete object без решения новой inverse problem;
- supervision существует только на gripper-induced interaction cells;
- compute растёт с (M m K), а не с (N_{voxels}) или числом shape samples.

Если reviewer всё равно назовёт representation «task-space implicit geometry», ответ должен быть: да, она хранит ровно task-equivalence class, но не идентифицирует world geometry; Theorem 1 показывает, почему эта потеря информации сознательна и достаточна.

---

## 10. Data engine и efficient learnability

### 10.1 Offline labels

Для каждого full CAD object и shelf scene:

1. sample target pose и один foreground blocker;
2. render wrist RGB-D с RealSense-like missing depth, quantization, axial/lateral noise;
3. vary visible fraction target, например 15–80%;
4. получить общий candidate set;
5. для каждого (g) exact mesh queries дают (W_g(S)) и hit pattern (h_g(S)).

Label generation — parallel ray casting/collision checking. Physics rollout не нужен для основного label; короткий lift simulation используется только как secondary label/evaluation.

### 10.2 Shape splits

Нужны минимум четыре axes generalization:

- held-out instances;
- held-out categories;
- held-out foreground-obstacle geometries;
- held-out sensor-noise severity.

Обязательно убрать near-duplicate meshes между train/test. Иначе hidden-shape prior будет выглядеть сильнее из-за retrieval leakage.

### 10.3 Counterfactual occlusion groups

Один и тот же full object следует рендерить с несколькими blockers/viewpoints. Это позволяет измерить:

- как posterior capacity меняется при сохранении shape и изменении visibility;
- уменьшается ли NLL в среднем с ростом visible fraction;
- не использует ли сеть shortcut по blocker identity.

Важно: это evaluation of information behavior, а не ложный pointwise nesting constraint.

### 10.4 Candidate coverage

Training queries должны включать:

- candidates сильного visible-only generator;
- локальные SE(3) perturbations вокруг них;
- hard negatives, которые проходят observed collision filter, но проваливаются из-за hidden geometry;
- oracle-good hidden-shape grasps только для диагностики candidate recall, не как test-time input.

Если oracle-good grasp отсутствует в (G(X)), reranker не может помочь. Поэтому candidate recall@full-mesh-success должен сообщаться отдельно.

### 10.5 Масштаб

Реалистичный первый stage:

- 10–30k meshes после deduplication;
- 20–50 occlusion renders/mesh;
- 64–128 candidates/render;
- $m=12$–16, $K=4,8,16$ ablation.

Hit labels намного дешевле полного dynamics rollout и естественно batchable на GPU/Embree/OptiX.

---

## 11. Эксперимент, способный подтвердить SOTA claim

### 11.1 Controlled benchmark

Создать **ShelfOcc-Grasp** protocol: target + один foreground obstacle + shelf, one wrist RGB-D, parallel jaw, no clutter. Occlusion severity задаётся долей hidden target surface и отдельно тем, закрыта ли потенциальная second contact side.

Primary metrics:

1. top-1 short-lift success;
2. collision-free antipodal success по full mesh;
3. selected-grasp regret к full-mesh oracle внутри общего (G(X));
4. negative log-likelihood полного hit pattern;
5. ECE/Brier для (Phi_{g,0});
6. risk–coverage curve: success среди grasps с predicted lower margin $\ge\tau$;
7. latency и peak memory.

### 11.2 Baselines

С одинаковым candidate set:

1. original generator score (GSNet/OrbitGrasp-class baseline);
2. scalar success BCE reranker, тот же observation backbone;
3. independent per-cell BCE, тот же InterCap encoder, (K=1);
4. deep ensemble / MC-dropout scalar reranker;
5. local occupancy completion + analytic evaluator;
6. stochastic shape completion + Monte Carlo robust score;
7. implicit collision query + scalar contact score;
8. InterCap (K>1).

Отдельно, где code/data позволяют, сравнить с Local Occupancy-Enhanced, NeuGraspNet, PCF-Grasp и adapted TOSC. Не следует выдавать dexterous-only CADGrasp/TOSC за perfectly matched baselines; они важны для related-work and representation comparison.

### 11.3 Critical ablations

- joint CapLog vs sum of marginal BCE;
- (K=1,2,4,8,16);
- coarse vs fine atomic partitions;
- без occlusion-ray tokens;
- без hard negatives hidden by blocker;
- canonical frame vs ordinary concatenation of pose;
- fixed risk-neutral (P(success)) vs lower-margin selection;
- reconstruction auxiliary loss added: гипотеза — он не помогает или ухудшает compute/decision efficiency.

### 11.4 Real robot

Не менее 30 held-out physical objects, 3–5 blocker placements на объект, random reset, минимум 3 repeated trials per condition. Outcome:

- gripper reaches prescribed terminal pose;
- closes;
- object is lifted 2–5 cm and held 2 s.

Не включать navigation, base motion, long approach planner или downstream placement. Failure taxonomy нужна только для measurement: terminal collision, no enclosure, slip/drop, perception/segmentation; не превращать её в causal modeling contribution.

### 11.5 Статистика

- paired trials по object/pose между methods;
- bootstrap 95% CI по objects, а не по correlated frames;
- mixed-effects logistic regression как secondary analysis;
- заранее зафиксированный primary endpoint: top-1 short-lift success under severe occlusion;
- report absolute percentage points и denominator, не только relative improvement.

---

## 12. Фальсифицируемые гипотезы и stop/go criteria

### H1: joint law действительно нужен

InterCap $K\ge4$ должен заметно превосходить $K=1$ independent model по pattern NLL и conjunction calibration. Если gain есть только по NLL, но отсутствует по grasp selection, центральный claim ослабевает.

### H2: task quotient лучше reconstruction при равном compute

При matched backbone/latency InterCap должен иметь меньший selected-grasp regret, чем local occupancy completion. Если completion стабильно лучше при малой разнице compute, paper превращается в отрицательный результат.

### H3: advantage растёт с ambiguity

Gain должен быть мал при 0–20% occlusion и расти при hidden second-contact geometry. Если gain одинаков везде, вероятен shortcut или просто stronger backbone.

### H4: atomic capacity даёт criterion transfer

После обучения модель должна без finetuning сохранять ranking при изменении friction threshold, gripper-width tolerance и collision dilation в пределах training support. Fixed scalar scorer этого не умеет.

### H5: real-world effect

Предлагаемый practical go criterion: не менее +8 absolute percentage points top-1 success над strongest visible-only/scalar reranker в severe-occlusion subset, с CI, не пересекающим ноль. Это не прогноз результата, а порог, ниже которого SOTA/ICLR framing следует пересмотреть.

---

## 13. Косвенные свидетельства, что идея может сработать

Это не доказательства InterCap, а независимые empirical links:

1. **Uncertainty hidden shape полезна:** uncertain completion подняла real grasp success 48% → 59%, хотя была чрезвычайно медленной ([Lundell et al.](https://arxiv.org/abs/1903.00645)).
2. **Task-local geometry полезна:** local occupancy дала +6.62% real grasping против того же baseline без completion ([Ma et al.](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09354.pdf)).
3. **Completion features могут дать большой practical gain:** PCF-Grasp сообщает +17.8 points real success ([paper](https://arxiv.org/abs/2504.16320)).
4. **Contact/collision structured representation полезна:** CADGrasp 93.8% vs 83.9% baseline in real clutter; removing contact decomposition in simulation снижает 86.5% до 56.1% ([paper](https://arxiv.org/abs/2601.15039)).
5. **Interaction queries эффективны:** SceneCollisionNet +9.8% collision accuracy и 75× faster than best baseline ([paper](https://arxiv.org/abs/2011.10726)).
6. **Equivariance и continuous action structure полезны:** OrbitGrasp существенно превосходит baselines в simulation и hardware ([paper](https://proceedings.mlr.press/v270/hu25b.html)).
7. **Decision-targeted training может быть лучше parameter reconstruction:** predict-then-optimize literature демонстрирует более качественные решения при меньшей model complexity ([Elmachtoub et al.](https://proceedings.mlr.press/v119/elmachtoub20a.html)).

InterCap соединяет эти факты не как robotics pipeline, а как mathematical response: uncertainty нужна; contact/collision events достаточны; query learning быстро; следовательно следует учить coherent conditional event law напрямую.

---

## 14. Adversarial novelty audit

### «Это просто mixture density network над hand-crafted labels»

Частично верно на уровне реализации. Защита возможна только если paper докажет и экспериментально покажет три вещи:

1. labels образуют finite restriction random-set capacity, а не произвольный vector;
2. joint proper score и capacity coherence улучшают conjunction calibration/decision против identical-backbone BCE;
3. atomic query algebra переносится на новые physical criteria без retraining.

Если третий пункт убрать, novelty действительно может показаться недостаточной.

### «Это local occupancy с другим названием»

Нет: local occupancy отвечает pointwise $p(o(x)=1\mid X)$ и позволяет реконструировать grid. InterCap отвечает на gripper-induced interaction cells, включает joint contact-pair events и не идентифицирует pointwise shape. Нужно показать orders-of-magnitude меньший output size и невозможность mesh recovery из outputs.

### «Capacity давно известна»

Да; novelty не в Choquet theory. Novelty — conditional neural capacity learning на action-query algebra, low-rank joint CapLog objective и decision-sufficiency construction. Related work должен честно цитировать random-set maps in robotics.

### «Любой scalar success predictor уже Bayes sufficient»

Для одного фиксированного utility — да. Поэтому paper обязан демонстрировать family of criteria/risk levels и transfer без retraining. Capacity representation ценна как minimal reusable interaction law между world model и single-task score.

### «Mixture components не соответствуют shapes»

И не должны. Они non-identifiable low-rank factors joint event tensor. Нельзя интерпретировать component (k) как конкретную completion. Это преимущество по compute и ограничение по explainability.

### «Simulator contacts не переносятся в reality»

Риск высокий. Снизить его можно через analytic rigid geometry labels, aggressive sensor randomization, small real calibration set и reporting sim-to-real calibration. Не следует прятать этот риск за foundation model.

### «Candidate generator не предлагает grasp на скрытой стороне»

Это ограничение recall. Paper про **selection**, не generation. Нужно измерять oracle recall общего (G(X)); при необходимости добавить generic uniform/perturbed candidates, не новую learned generator architecture.

---

## 15. ICLR acceptance audit по официальным критериям

[ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide) сводит решение к новым знаниям и достаточной ценности для community; reviewer должен проверить motivation/literature placement, поддержку claims, rigor и significance, причём SOTA не является обязательным.

### Originality: потенциально high, но условно

Сильная часть — новый predicted object и proper objective. Слабая — известные mixture Bernoulli, set attention и random-set theory. Originality будет высокой только при ясной theorem–architecture–experiment связке и criterion-transfer experiment.

### Quality: достижимо

Есть точные propositions, clean same-candidate controlled comparison и чёткий falsification protocol. Нельзя заменять hardware statistics несколькими qualitative videos.

### Clarity: достижимо

Нужно держать main notation ровно на трёх объектах: (X,g,h). Random-set background — одна figure и одна theorem box; не перегружать статью topology.

### Significance: от medium до high

High, если метод одновременно:

- улучшает severe-occlusion grasp success;
- заметно быстрее probabilistic reconstruction;
- калиброван на Boolean grasp events;
- переносится между query criteria;
- показывает general synthetic random-set benchmark вне robotics.

Без последнего или criterion transfer reviewer может классифицировать работу как robotics application и отправить на CoRL/RSS.

### Рекомендуемый paper package

1. **General ML section:** conditional capacity learning, CapLog, propositions.
2. **Toy benchmark:** partially observed 2-D/3-D random shapes; arbitrary hit-query prediction, union consistency и decision regret.
3. **Robotics instantiation:** ShelfOcc-Grasp simulation.
4. **Hardware:** строго ограниченная target+blocker scene.
5. **Open resources:** renderer, witness labeler, benchmark splits, calibration code.

---

## 16. Минимальный план реализации

### Phase 0: kill test до большой модели

На 2-D procedural shapes построить (m=8) query cells и сравнить:

- scalar success BCE;
- independent hits;
- InterCap (K=4);
- occupancy reconstruction.

Проверить joint NLL, union consistency, criterion transfer, selection regret. Если InterCap не выигрывает при сильной multimodal ambiguity, не масштабировать.

### Phase 1: geometry-only 3-D

Full CAD meshes, synthetic shelf/blocker, exact ray/collision labels; frozen candidate generator; PointNet++/sparse transformer backbone. Цель — доказать H1–H4 без physics/domain gap.

### Phase 2: RGB-D noise and real scans

Подмешать real backgrounds, sensor missingness, calibration error. Использовать небольшой real validation set только для temperature calibration, не для скрытого test tuning.

### Phase 3: hardware

Paired randomized trials, predefined protocol, fixed checkpoint.

### Phase 4: submission-strength theory/benchmark

Формально доказать propositions, опубликовать code/data, провести независимый nearest-neighbor and leakage audit.

---

## 17. Что должно попасть в abstract будущей статьи

Черновая claim skeleton, без выдуманных результатов:

> Learning to act from partial geometry is commonly approached either by reconstructing the latent world or by predicting a task-specific scalar. We introduce conditional interaction capacities, a middle representation that models the joint law of action-induced hit/miss events without reconstructing geometry. Our Capacity Log Score is strictly proper for the finite hit algebra, and a low-rank conditional capacity tensor yields coherent Boolean event probabilities in linear time in the number of queries. We instantiate the framework for parallel-jaw grasp selection from a single occluded RGB-D view, where hidden shape induces correlated contact and collision events. [После экспериментов: реальные quantitative claims.] 

Нельзя писать «first uncertainty-aware occlusion grasping» или «guaranteed safe». Оба claim неверны/неподдержаны.

---

## 18. Главные риски проекта

1. **Insufficient novelty perception:** mitigated general random-set benchmark + criterion transfer + formal objective.
2. **Witness discretization loses decisive geometry:** multi-resolution partition and adaptive cell boundaries; report convergence with (m).
3. **Low-rank underfit:** compare (K), autoregressive oracle head и full categorical head при малом (m).
4. **Mode collapse:** batch-level mode-usage monitoring, multiple restarts; не обещать semantic modes.
5. **Sim-to-real gap:** geometry-derived labels, noise randomization, real calibration, paired tests.
6. **Candidate recall bottleneck:** report oracle recall, keep generator common across methods.
7. **Segmentation shortcut:** mask-quality stress test and oracle-mask vs noisy-mask reporting.
8. **Foreground obstacle dominates:** balanced dataset, где одинаковый blocker скрывает разные shapes и один shape скрывается разными blockers.
9. **Scalar baseline unexpectedly wins:** тогда честный вывод — reusable capacity не окупает complexity для fixed gripper criterion; ICLR claim нужно снять.

---

## 19. Финальная оценка

**Научная novelty:** 8/10 как proposal; 5/10 без arbitrary-query/criterion-transfer demonstration.  
**Математическая чистота:** 8/10; конечная algebra избегает тяжёлой topology и даёт точные proofs.  
**Efficient learnability:** 8/10; $O(Km)$ head, one-shot observation encoder, labels дешевле dynamics.  
**Соответствие лабораторному setup:** 9/10; single wrist RGB-D, target+foreground obstacle, parallel jaw, tiny lift.  
**Риск пересечения с closest robotics work:** medium; особенно CADGrasp/TOSC/local occupancy, но различие defensible.  
**ICLR potential:** high только при joint-calibration gain, criterion transfer, controlled benchmark и real paired trials.  
**Потенциал SOTA:** правдоподобный для severe-occlusion **selection under a shared candidate set**, но до экспериментов это гипотеза, не факт.

Главная формула всей идеи:

\[
\boxed{
\text{partial RGB-D}
\longrightarrow
p_\theta(\text{interaction-hit algebra}\mid X,g)
\longrightarrow
\text{exact risk-aware grasp score},
\quad
\text{without }\hat S
}
\]

Именно переход от «predict hidden shape» к «properly learn the conditional law of action-induced set events» является broad, open и потенциально ICLR-level вкладом.

---

## 20. Selected primary references

### Grasping and partial geometry

1. Fang et al. [GraspNet-1Billion](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html), CVPR 2020.
2. Sundermeyer et al. [Contact-GraspNet](https://arxiv.org/abs/2103.14127), ICRA 2021.
3. Wang et al. [Graspness Discovery / GSNet](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.html), ICCV 2021.
4. Fang et al. [AnyGrasp](https://arxiv.org/abs/2212.08333), IEEE T-RO 2023.
5. Hu et al. [OrbitGrasp](https://proceedings.mlr.press/v270/hu25b.html), CoRL 2024 proceedings, 2025.
6. Jauhri et al. [NeuGraspNet](https://openreview.net/forum?id=Fdu33eoZas), 2024.
7. Ma et al. [Local Occupancy-Enhanced Object Grasping](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09354.pdf), ECCV 2024.
8. Lundell et al. [Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645), IROS 2019.
9. Saund & Berenson. [Diverse Plausible Shape Completions](https://proceedings.mlr.press/v155/saund21a.html), CoRL 2020 proceedings, 2021.
10. Cheng et al. [PCF-Grasp](https://arxiv.org/abs/2504.16320), 2025 preprint.
11. Wu et al. [TOSC](https://arxiv.org/abs/2601.05499), 2026 preprint.
12. Zhang et al. [CADGrasp](https://arxiv.org/abs/2601.15039), 2026 preprint.
13. Chen et al. [SpringGrasp](https://arxiv.org/abs/2404.13532), 2024.
14. Danielczuk et al. [Object Rearrangement Using Learned Implicit Collision Functions](https://arxiv.org/abs/2011.10726), ICRA 2021.
15. Carvalho et al. [Grasp Diffusion Network](https://arxiv.org/abs/2412.08398), 2024.

### General ML and mathematics

16. Mandi et al. [Decision-Focused Learning Through the Lens of Learning to Rank](https://proceedings.mlr.press/v162/mandi22a.html), ICML 2022.
17. Elmachtoub et al. [Decision Trees for Predict-then-Optimize](https://proceedings.mlr.press/v119/elmachtoub20a.html), ICML 2020.
18. Bates et al. [Distribution-Free, Risk-Controlling Prediction Sets](https://arxiv.org/abs/2101.02703), JACM 2024 / preprint 2021.
19. Zaheer et al. [Deep Sets](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html), NeurIPS 2017.
20. Lee et al. [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html), ICML 2019.
21. Cenzer et al. [Algorithmic Randomness and Capacity of Closed Sets](https://arxiv.org/abs/1106.2993), 2011.
22. Gallego et al. [On the connectedness of a random closed set](https://www.sciencedirect.com/science/article/abs/pii/S0165011421003699), Fuzzy Sets and Systems 2022.
23. Rolfes & Rendas. [Statistical environment representation for navigation in natural environments](https://www.sciencedirect.com/science/article/abs/pii/S0921889002002609), Robotics and Autonomous Systems 2002.
24. [ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide).

---

## 21. Независимость данного memo

Этот документ сформирован как самостоятельное исследование по внешней литературе и математической разработке. Содержимое существующих repository-файлов и прежних markdown-идей не использовалось. Поэтому документ не делает утверждений о текстуальном различии с недоступными для проверки локальными идеями; его защита от тематического пересечения основана на выбранном специфическом objective: **conditional low-rank capacity of a grasp-induced hit algebra**.
