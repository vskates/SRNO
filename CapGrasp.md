# Do Not Complete What You Cannot See

## Conditional Capacity Learning for reliable parallel-jaw grasp selection under foreground occlusion

**Дата исследования:** 25 августа 2026  
**Статус:** сформулированная исследовательская гипотеза и программа её фальсификации; не заявление о достигнутом SOTA  
**Рабочие названия:** Action-Query Capacity Learning (AQCL) — общий framework; CAPGrasp / Choquet Grasp Operator (CGrO) — grasping-инстанциация; Conditional Coverage Mixture Network (CCMN) — архитектура головы.

---

## 1. Итоговый вердикт

Наиболее сильная найденная формулировка — **не восстанавливать скрытую форму и не предсказывать одним scalar head вероятность успеха конкретного grasp**, а обучать условный **capacity functional** скрытого случайного геометрического множества на пространстве task-запросов. Такой оператор отвечает на вопрос вида:

> «Какова вероятность, что неизвестная полная геометрия пересечёт заданное компактное множество допустимых контактов, запрещённых коллизий или их объединение, при данном частичном RGB-D наблюдении?»

Для parallel-jaw grasp `g` определяются два запроса:

- `A_g`: существует допустимая пара контактов внутри двух pad-толерантностей и с заданным friction/antipodal margin;
- `B_g`: скрытая геометрия объекта пересекает запрещённое swept/body-множество gripper, исключая разрешённые contact patches.

Если `T_O(K)` — conditional hitting capacity, то вероятность геометрического grasp-сертификата получается **точным тождеством**, а не эвристическим произведением независимых вероятностей:

$$
p_{\mathrm{geo}}(g\mid O)
=\Pr[H(A_g)\land \neg H(B_g)\mid O]
=T_O(A_g\cup B_g)-T_O(B_g).
$$

Это сохраняет зависимость между скрытыми контактами и скрытой коллизией, потому что оба события порождены одной неизвестной формой. Полная mesh/voxel/SDF-реконструкция не нужна ни на обучении в качестве output, ни на inference. Модель учит лишь decision-relevant quotient распределения форм: какие формы неразличимы для заданного семейства запросов, считаются эквивалентными.

Главная general-ML гипотеза работы:

> **Conditional capacity learning** является новым классом structured probabilistic prediction для частично наблюдаемых случайных множеств: proper objective элицитирует task-restricted hitting law, а архитектура valid-capacity-by-construction даёт логически согласуемую композицию новых Boolean/set queries.

Это сильнее узкого «ещё одного grasp scorer». На grasping можно показать физический эффект; на синтетических random-set задачах — общую статистическую и compositional ценность. Именно второй компонент нужен, чтобы работа имела правдоподобный ICLR-level вклад.

Моя оценка на текущем этапе:

- **идея без серьёзных экспериментов:** примерно 3/10 на ICLR — reviewer легко назовёт её сложно записанным success classifier;
- **идея с theorem package, synthetic random-set benchmark, controlled simulator и real-robot evidence:** примерно 7/10 — правдоподобная main-conference submission;
- **SOTA:** не доказан. Есть хороший путь к SOTA-потенциалу именно при сильной окклюзии, но утверждать SOTA до реализации и paired evaluation нельзя.

---

## 2. Точная постановка и границы работы

### 2.1. Сцена

- humanoid robot;
- wrist-mounted RGB-D camera;
- один известный target на полке;
- один foreground obstacle может частично скрывать target;
- не cluttered scene;
- единственное шумное RGB-D наблюдение;
- parallel-jaw gripper;
- решение: выбрать grasp из конечного candidate set, выполнить grasp и слегка поднять предмет.

### 2.2. Что моделируется

Моделируется неопределённость **скрытой grasp-релевантной геометрии target**, обусловленная partial observation и обучающим shape distribution. Наблюдаемые shelf/obstacle и видимая поверхность используются в deterministic/inflated collision checks. Pose/depth noise можно интегрировать как nuisance-переменную при генерации меток.

### 2.3. Что намеренно не моделируется

- RL и trial-and-error policy learning;
- VLA и semantic planning;
- obstacle removal или active view selection;
- clutter reasoning;
- полная траектория approach-to-lift;
- dynamics, controller failure, suction, deformables;
- causal taxonomy всех failure modes;
- full-scene SDF, NeRF или giant variable stack;
- full object reconstruction как обязательный промежуточный output.

Модель выдаёт вероятность **локального геометрического/quasi-static сертификата**, а не вероятность успеха всего manipulation episode. Реальный small-lift success служит downstream evaluation и проверкой валидности сертификата.

---

## 3. Что уже сделано: карта ближайшей robotics-литературы

Robotics-работы здесь использованы только для установления ближайших методов, gaps и востребованности задачи. Источник самой формулировки — random-set theory, Choquet capacities, proper scoring и structured set functions.

| Направление | Что делает | Почему не решает выбранную задачу |
|---|---|---|
| [TARGO / TARGO-Net](https://arxiv.org/abs/2407.06168), IJCV 2026 | Exact-nearest benchmark: target-aware grasping under occlusion. Сегментирует target, completion восстанавливает скрытую геометрию, transformer fusion предсказывает grasp. | Использует deterministic completion и deterministic grasp output; не представляет posterior hidden geometry и не даёт согласуемой contact/collision probability algebra. Сцены в основном cluttered, тогда как здесь один occluder. |
| [Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645), Lundell et al. | MC-dropout генерирует несколько voxel completions; analytical grasp quality усредняется по samples. Показывает пользу shape uncertainty. | Реконструирует полные shapes и делает online Monte Carlo. Не учит query operator, capacity law или compositional Boolean events. |
| [Local Occupancy-Enhanced Object Grasping](https://arxiv.org/abs/2407.15771), ECCV 2024 | Предсказывает local occupancy около grasp points и улучшает GraspNet AP при occlusion. | Deterministic local completion/occupancy; нет posterior зависимости между paired contacts и collision. |
| [NeuGraspNet](https://arxiv.org/abs/2306.07392), RSS 2024 | TSDF и implicit local surface rendering для geometry-aware grasp quality. | Геометрическое implicit representation, но не conditional random-set law и не capacity composition. |
| [CenterGrasp](https://arxiv.org/abs/2312.08240) | Joint holistic latent shape reconstruction и grasp estimation; демонстрирует пользу shape completion. | Восстанавливает цельную форму; output не является calibrated uncertainty over task events. |
| [GIGA](https://arxiv.org/abs/2104.01542) | Implicit geometry and affordance для 6-DoF grasping. | Детерминированное implicit field; не моделирует hidden-shape distribution. |
| [Contact-GraspNet](https://arxiv.org/abs/2103.14127) | Anchor grasp на наблюдаемой поверхности, редуцируя 6D search. | Visible-surface-centric; скрытый second contact/collision posterior явно не представлен. |
| [Object Pose and Shape Estimation for Grasping: Does it Work?](https://arxiv.org/abs/2605.26944), 2026 | Модульная pose/shape reconstruction плюс antipodal grasp sampling может выигрывать у end-to-end baseline. | Сильный baseline и аргумент, что geometry полезна, но качество зависит от reconstruction accuracy. |
| [TOSC](https://arxiv.org/abs/2601.05499), 2026 | Task-oriented completion только contact-relevant областей для dexterous grasping. | Всё ещё reconstruct-then-grasp и другая hand regime. |
| [FFHFlow](https://proceedings.mlr.press/v305/feng25a.html), CoRL 2025 | Flow distribution dexterous grasps из partial clouds, likelihood introspection и uncertainty-aware ranker. | Распределение grasps, а не latent geometry hit law; нет query algebra для contact/collision. |
| [Variational Neural Belief Parameterizations](https://arxiv.org/abs/2604.25897), 2026 | Gaussian-mixture belief over contact/pose и differentiable CVaR-POMDP/MPC для dexterous execution. | Показывает, что «просто CVaR» уже не нов; моделирует execution/contact belief и control, а не single-view hidden-set querying. |
| [AnyDexGrasp](https://graspnet.net/anydexgrasp/assets/files/AnyDexGrasp.pdf), ICLR 2025 workshop | Contact-centric hidden surface positions/normals из partial view; hand-agnostic dexterous representation. | Ближайший contact-centric prior, но не joint conditional capacity и не согласуемая логика contact-without-collision. |
| [Learning contact representations in real-world clutter for universal robotic grasping](https://www.nature.com/articles/s42256-026-01292-y), Nature Machine Intelligence 2026 | Hardware-agnostic spatial contact features и differentiable optimizer; сильные real-world результаты. | Deterministic contact representation/optimizer; clutter focus; не random-set posterior. |
| [Deep Collision Probability Fields](https://arxiv.org/abs/2409.04306), RA-L 2024 | Amortized collision probability queries для shapes с unimodal pose uncertainty. | Scalar collision field для параметризованной pose uncertainty, не conditional law unseen shape и не contact-pair/collision composition. |
| [CATNIPS](https://arxiv.org/abs/2302.12931), RSS 2023 | NeRF преобразуется в Poisson point process для chance-constrained collision planning. | Требует explicit NeRF scene и stochastic geometry для navigation, не learned conditional capacity hidden shape. |

### 3.1. Выявленный gap

Литература в основном выбирает одну из четырёх стратегий:

1. восстановить полную или локальную геометрию;
2. предсказать scalar grasp quality/success;
3. предсказать distribution непосредственно над grasps/contacts;
4. sample shapes и усреднить analytical score.

Не найдено метода, который бы одновременно:

- учил **условный hitting functional** неизвестной формы;
- был reconstruction-free на inference;
- сохранял статистическую зависимость contact и collision events;
- гарантировал valid-capacity constraints by construction;
- позволял вычислять новые union/intersection/complement task predicates без переобучения отдельного success head.

Целенаправленный поиск по сочетаниям `neural capacity functional`, `conditional random set learning`, `Choquet grasping`, `random closed set robot grasp`, `hitting functional collision/contact` до 25.08.2026 не обнаружил прямого аналога. Это **search-based novelty evidence, не доказательство отсутствия**; перед submission нужен повторный systematic search по Scholar, Semantic Scholar, arXiv, OpenReview и citation graph.

---

## 4. Последовательный журнал кандидатов и причины отказа

### Кандидат A: stochastic completion + CVaR grasp score

Идея: генерировать posterior samples скрытой формы и выбирать grasp по lower-tail analytical quality.

**Отказ.** Lundell et al. уже используют uncertain shape completions для robust planning; FFHFlow и Variational Neural Belief methods занимают distributional/risk-aware пространство, а CVaR давно используется и в risk-aware grasping. Это хороший baseline, но слабая general-ML novelty. Кроме того, online samples нарушают требование efficient reconstruction-free inference.

### Кандидат B: conditional quantile force-closure margin

Идея: предсказывать quantile/CDF analytical grasp margin для каждого candidate.

**Отказ.** Это стандартная distributional regression над scalar label. Она не раскрывает структуру скрытой геометрии, плохо переносит изменения contact tolerances/gripper geometry и не создаёт substantial new knowledge сверх calibrated grasp scoring.

### Кандидат C: latent-oracle regret / minimax regret

Идея: учить regret grasp относительно лучшего grasp для неизвестной истинной формы.

**Отказ.** Для ожидаемого regret член oracle quality не зависит от выбранного grasp и сокращается:

$$
\arg\min_g \mathbb E[Q(g^*(S),S)-Q(g,S)\mid O]
=\arg\max_g \mathbb E[Q(g,S)\mid O].
$$

То есть objective маскирует обычный expected quality. Tail regret не даёт абсолютной надёжности и снова уходит в generic risk measure.

### Кандидат D: first-contact survival operator

Идея: вдоль jaw closure предсказывать distribution первого столкновения, используя nested swept volumes как survival time.

**Частичный отказ.** Survival formulation красива и даёт monotonicity по времени закрытия, но одна survival curve плохо выражает совместное существование двух допустимых контактов и отсутствие body collision. Она остаётся полезным **следствием** capacity operator:

$$
\Pr(\tau_g\le t\mid O)=T_O(K_g(t)),
$$

где `K_g(t)` — вложенные swept volumes. Основной framework должен быть богаче survival prediction.

### Кандидат E: conditional random-set capacity learning

**Выбран.** Он удовлетворяет всем ключевым ограничениям:

- не требует full reconstruction;
- имеет broad general-ML постановку;
- даёт proper learning objective;
- имеет структурную архитектуру и theorem package;
- естественно выражает paired contact + no collision;
- после единственного observation encoding оценивает много candidates батчированно;
- допускает прямую проверку, превосходит ли compositional value обычный success classifier.

---

## 5. Математическая формализация

### 5.1. Скрытая форма как условное случайное замкнутое множество

Пусть `S` — полная поверхность target в camera/world frame. В теории предполагается компактная `C¹`-поверхность; в реализации — triangle mesh. Для nonsmooth объектов можно использовать generalized normal bundle.

Наблюдение:

$$
O=\mathrm{Render}(S,\xi_{cam},S_{occ})+\epsilon,
$$

где `S_occ` — foreground occluder, `ε` — depth noise/dropout. После наблюдения `S` остаётся случайным замкнутым множеством с conditional law `P(S|O)`.

Ориентированный normal bundle:

$$
\widetilde S=\{(x,n(x)):x\in S\}.
$$

Для paired contacts нужен lifted pair space:

$$
P(S)=\widetilde S\times\widetilde S.
$$

Пара может нести marks: расстояние, opposing-normal angle, conservative friction-cone margin, analytical wrench/force-closure margin и устойчивость к малому pose perturbation.

Вводим tagged latent random set:

$$
Z(S)=\big(\{+\}\times P(S)\big)\;\cup\;
      \big(\{-\}\times\widetilde S\big).
$$

Тег `+` относится к существованию допустимой пары контактов; тег `−` — к одиночной поверхности, которая может вызвать collision. Оба компонента детерминированы одной формой `S`, поэтому их зависимость сохраняется.

### 5.2. Hitting capacity

Для компактного query set `K`:

$$
H_Z(K)=\mathbf 1[Z\cap K\ne\varnothing],
\qquad
T_O(K)=\Pr[Z\cap K\ne\varnothing\mid O].
$$

`T_O` — условный capacity/hitting functional случайного замкнутого множества. В классической random-set theory распределение random closed set определяется его capacity functional; Choquet characterization связывает такие функции с normalized, upper-semicontinuous, completely alternating capacities. Практическая модель работает на task-restricted finite query algebra, поэтому не пытается идентифицировать закон на всех компактах пространства.

### 5.3. Grasp-запросы

Для candidate `g=(R,t,w)` строятся:

**Положительный pair-query `A_g`.** Это компакт в `+`-компоненте, содержащий пары `(x_1,n_1,x_2,n_2)`, которые:

- попадают в две opposing pad tolerance slabs;
- согласованы с выбранной final jaw width `w` и stroke;
- удовлетворяют conservative antipodal/friction condition при фиксированном `μ₀`;
- при необходимости имеют analytic `ε`-quality выше порога `γ`;
- устойчивы к небольшому pose/noise perturbation.

**Отрицательный query `B_g`.** Это компакт в `−`-компоненте, соответствующий запрещённому body/swept volume fingers и palm до final width, за вычетом разрешённых pad contact zones. Он проверяет collision hidden target geometry. Collision с наблюдаемыми obstacle/shelf проверяется hard filter по inflated point cloud.

Если candidate generator не предсказывает final width надёжно, допустим небольшой discrete set `w_1,…,w_k`; каждый `(g,w_i)` становится отдельным candidate. Это устраняет неоднозначность, когда fingers встречают объект асимметрично в разные моменты closure.

### 5.4. Геометрический сертификат

$$
C_g=H_Z(A_g)\land\neg H_Z(B_g).
$$

Тогда:

$$
\begin{aligned}
p_{geo}(g\mid O)
&=\Pr[C_g\mid O]\\
&=\Pr[H(A_g)\lor H(B_g)\mid O]-\Pr[H(B_g)\mid O]\\
&=T_O(A_g\cup B_g)-T_O(B_g).
\end{aligned}
$$

Это ключевой результат. Нельзя заменять его на `P(H(A))·(1−P(H(B)))`: contact existence и collision обычно сильно зависимы через thickness, curvature и hidden backside geometry.

Более сложные Boolean certificates выражаются через inclusion–exclusion/Möbius inversion на конечной query algebra. Например, две обязательные contact conditions и один forbidden region можно получить из capacities соответствующих unions.

### 5.5. Decision-relevant quotient вместо shape reconstruction

Пусть `𝒬` — семейство используемых task queries. Определим эквивалентность форм:

$$
S\sim_{\mathcal Q}S'
\iff
\forall K\in\mathcal Q:\ H_{Z(S)}(K)=H_{Z(S')}(K).
$$

Модель должна учить posterior не в огромном пространстве meshes, а на quotient `𝒮/∼_𝒬`: различия формы, которые не меняют ни одного task predicate, намеренно забываются.

**Proposition 1 — decision sufficiency.** Для любой decision loss, измеримой относительно конечного набора hit predicates из `𝒬`, условное распределение equivalence class `[S]_𝒬` является достаточной статистикой для Bayes-optimal action; полный posterior `P(S|O)` не даёт дополнительной decision value.

**Набросок доказательства.** Такая loss факторизуется через vector hit pattern `h_𝒬(S)`. Conditional risk каждой action есть expectation функции только от `h_𝒬`. Следовательно, две conditional shape laws с одинаковым pushforward по `h_𝒬` порождают одинаковые risks и одинаковое множество Bayes actions.

Это формализует тезис «не реконструируй то, что решение никогда не спрашивает».

---

## 6. Новый learning objective: elicitation условной capacity

### 6.1. Training tuple

Из полного training mesh строятся разные partial observations:

$$
(S,O,\delta),
$$

где `δ` содержит sampled camera/gripper pose noise, depth corruption и при необходимости conservative nuisance friction. Для каждого observation выбирается batch compact queries `K_1,…,K_m` из task distribution `μ_Q`.

Ground-truth hit label вычисляется offline exact/robust mesh predicates:

$$
y_K=\mathbf 1[Z(S,\delta)\cap K\ne\varnothing].
$$

Для union метка бесплатна:

$$
y_{K_1\cup\dots\cup K_r}=\max_i y_{K_i}.
$$

### 6.2. Capacity proper loss

$$
\mathcal L_{cap}(\theta)=
\mathbb E_{(O,S),K\sim\mu_Q}
\left[-y_K\log\widehat T_\theta(K\mid O)
-(1-y_K)\log(1-\widehat T_\theta(K\mid O))\right].
$$

Обучение должно включать:

- single positive pair queries `A`;
- single collision queries `B`;
- grasp-relevant unions `A∪B`;
- unions между queries разных grasps в одной сцене;
- nested/range queries с разными tolerances;
- random unions, чтобы идентифицировать joint hit pattern, а не только marginals.

Sampling балансируется по hits/misses, occlusion bucket, geometry type и query scale; иначе почти все случайные volumes будут misses.

**Proposition 2 — proper elicitation.** При неограниченной model class population minimizer binary log loss для каждого query равен истинному `T_O(K)` почти всюду по `P(O)μ_Q(K)`.

**Набросок доказательства.** Условно на `(O,K)` label Bernoulli с параметром `T_O(K)`. Bernoulli log score строго proper, и его единственный minimizer — conditional event probability.

Важно: гарантия относится к **task-restricted query distribution**. Обещание восстановить capacity для произвольного никогда не встречавшегося компакта было бы необоснованным.

### 6.3. Почему не обучать основной `y_good` head

Direct BCE по `y_good=C_g` — обязательный baseline, но не основная модель. Он даёт одно endpoint число и не обязан:

- согласованно отвечать на `A`, `B`, `A∪B`;
- переноситься на новый friction/tolerance threshold;
- переиспользоваться для нового gripper query;
- поддерживать set inclusion и union identities;
- объяснять, растёт риск из-за отсутствия контактов или hidden collision.

Главный experimental burden — показать, что структурная capacity действительно выигрывает в compositional/OOD regime, а не только добавляет notation.

---

## 7. Архитектура CCMN: valid capacity by construction

### 7.1. Общая схема

1. **Observation encoder.** RGB-D target crop, target-visible point tokens и obstacle/shelf tokens кодируются один раз SE(3)-aware point transformer или sparse point backbone. Никакой mesh/voxel completion head нет.
2. **Canonical query constructor.** Геометрия gripper превращается в typed query tokens: pair-contact compact `A_g`, forbidden swept compact `B_g`, их tolerance/friction marks.
3. **Query encoder.** Point/surface samples компакта и analytic marks кодируются permutation-invariant Deep Sets/Point Transformer block; cross-attention к observation tokens локализует запрос в сцене.
4. **Latent coverage modes.** Из observation извлекаются `H` mode tokens и weights `π_h(O)`. Mode — не shape sample и не декодируется в geometry; он представляет coherent pattern ответов на queries.
5. **Capacity head.** Для каждого atomic compact-query `K_j` и mode `h` модель выдаёт `r_{hj}=r_h(K_j,O)∈[0,1]`.

### 7.2. Mixture noisy-OR capacity

Сначала зафиксируем конечное семейство физических запросов `𝒦_m={K_1,…,K_m}` и hit vector `Y_i=H_Z(K_i)`. Он индуцирует случайное множество индексов

$$
R_Z=\{i:Y_i=1\}\subseteq\{1,\ldots,m\}.
$$

Для `J⊆{1,…,m}` его hitting capacity точно совпадает с ограничением физической capacity:

$$
\Pr[R_Z\cap J\ne\varnothing\mid O]
=\Pr\!\left[Z\cap\left(\bigcup_{j\in J}K_j\right)\ne\varnothing\mid O\right]
=T_O\!\left(\bigcup_{j\in J}K_j\right).
$$

Таким образом, finite head моделирует не метафорический set score, а joint law ответов физического hidden set на выбранную query family. Для union, канонически представленного списком query indices `J`, определим:

$$
\widehat T_\theta\left(\bigcup_{j\in J}K_j\mid O\right)
=1-\sum_{h=1}^{H}\pi_h(O)
\prod_{j\in J}\big(1-r_h(K_j,O)\big),
$$

где `π_h≥0` и `Σ_hπ_h=1`.

Для grasp:

$$
\widehat p_{geo}(g\mid O)
=\widehat T(A_g\cup B_g\mid O)-\widehat T(B_g\mid O)
=\sum_h\pi_h\,r_h(A_g)\,[1-r_h(B_g)].
$$

Эта factorization предполагает conditional independence атомарных hit indicators внутри mode, но смесь modes восстанавливает произвольную finite joint distribution при достаточном `H`.

### 7.3. Структурные свойства

**Proposition 3 — finite capacity validity.** На конечной алгебре query indices CCMN задаёт normalized, monotone, completely alternating hitting capacity.

**Обоснование.** Для каждого mode можно построить random subset query indices с независимыми Bernoulli inclusions `r_hj`; mixture по `π_h` задаёт корректное распределение random subset. Формула выше есть его hitting probability. Любой hitting functional random set является capacity с нужными finite-difference inequalities.

Следствия:

- `T(∅)=0`;
- `J⊆L ⇒ C(J)≤C(L)` на canonical query-index lattice;
- `T(A∪B)−T(B)≥0`, поэтому composed grasp probability не становится отрицательной;
- порядок query atoms не влияет на ответ;
- один posterior mode mixture переиспользуется для всех candidates сцены.

**Proposition 4 — finite universality.** Любое joint distribution `P(Y_1,…,Y_m)` бинарных hit events представимо CCMN не более чем `2^m` deterministic modes: один mode на каждый bit vector, `r_hj∈{0,1}`, `π_h=P(Y=h)`.

Практический `H=16` или `32` — low-rank approximation, а не точный универсальный posterior для тысяч queries. Это следует честно измерять ablation по `H`, calibration и cross-query likelihood.

### 7.4. Непрерывная геометрия и canonical query algebra

Capacity guarantee буквально действует на union конечного семейства query objects. Равенство с физической capacity точно для зафиксированного семейства и его unions, но архитектура сама по себе не знает, что два разных списка могут задавать один и тот же geometric set. Чтобы избежать такой несогласованности:

- unions всегда передаются как список canonical atoms, а не повторно rasterized единым токеном;
- nested tolerance family использует shared base cells/interval endpoints;
- при необходимости локальный gripper frame atomizes pad slabs и swept body на фиксированную multi-resolution lattice, а запрос передаётся как множество cell IDs;
- можно добавить sampled inclusion loss `max(0,T(K)−T(L))` для геометрически вложенных continuous queries;
- exact labels всё равно вычисляются на mesh, а discretization error измеряется отдельно.

Это не мелкая implementation detail, а одна из главных научных рисков: valid set-function structure нельзя заявлять в continuous space, если query encoder нарушает canonical set identity.

### 7.5. Вычислительная стоимость

Observation кодируется один раз. После этого для `G` candidates и двух основных atoms на grasp стоимость головы примерно:

$$
O(GHd),
$$

с полностью batchable tensor operations. Нет online mesh decoding, volumetric grid или Monte Carlo shape completion. Целевой engineering budget: `H≤32`, 128–512 geometric samples на query только в encoder, latency ниже deterministic local-completion baseline при одинаковом backbone.

### 7.6. Минимальный inference pseudocode

```text
tokens, mode_weights, mode_context = encode_observation(rgbd)
candidates = fixed_candidate_generator(visible_target_points)

for g in candidates (batched):
    reject if inflated_observed_scene_collides(g)
    A = build_contact_pair_query(g, pad_tol, mu0, margin_gamma)
    B = build_hidden_body_sweep_query(g, final_width)
    rA = query_modes(A, tokens, mode_context)
    rB = query_modes(B, tokens, mode_context)
    p_geo[g] = sum_h mode_weights[h] * rA[h] * (1 - rB[h])

return argmax_g p_geo[g]
```

Если robot обязан всегда grasp, выбирается top-1. Если допустимо abstention, threshold калибруется отдельно; active perception и replanning не входят в scope.

---

## 8. Теоретический пакет, достаточный для general-ML submission

### 8.1. Capacity estimation to decision regret

Пусть конечный candidate set `𝒢(O)` фиксирован и для каждого `g` выполнено:

$$
|\widehat T(A_g\cup B_g)-T(A_g\cup B_g)|\le\varepsilon,
\quad
|\widehat T(B_g)-T(B_g)|\le\varepsilon.
$$

Тогда:

$$
|\widehat p_{geo}(g)-p_{geo}(g)|\le2\varepsilon.
$$

Если `ĝ=argmax_g p̂(g)`, `g*=argmax_g p(g)`, то:

$$
p(g^*)-p(\widehat g)\le4\varepsilon.
$$

Доказательство — triangle inequality и стандартный plug-in argmax decomposition. Это простой, но полезный мост от capacity calibration к selection regret.

### 8.2. Boolean-query extension

Для finite family predicates `H(K_i)` любая monotone Boolean formula может быть представлена через probabilities unions/intersections. Hitting capacities всех unions определяют joint law hit vector через Möbius inversion. Значит, один learned operator потенциально поддерживает новые grasp certificates, если их atoms лежат в покрытой query family.

На практике exponential inversion не делается: для малых grasp formulas (`A`, `B`, иногда 2–4 contact/collision atoms) нужны лишь несколько union queries.

### 8.3. Derived survival view

Для nested swept volumes `K_g(t₁)⊆K_g(t₂)` при `t₁≤t₂`:

$$
F_{\tau_g}(t\mid O)=\Pr(\tau_g\le t\mid O)=T_O(K_g(t)).
$$

Поэтому capacity head автоматически задаёт monotone first-contact CDF на canonical nested lattice. Это связывает framework с survival analysis, но не ограничивает его одномерным временем.

### 8.4. Что theorem package не обещает

- uniform error bound не получается бесплатно из empirical BCE;
- finite lattice validity не равна upper semicontinuity на всех continuous compacts;
- правильный geometric certificate не гарантирует controller/dynamic lift success;
- universality с `2^m` modes непрактична при большом `m`;
- OOD category calibration требует эмпирической проверки или дополнительных assumptions.

Чёткое ограничение claims повысит доверие reviewer сильнее, чем формально широкий, но неверный theorem.

---

## 9. Данные и генерация supervision

### 9.1. Synthetic source

Основной источник meshes — [ACRONYM](https://arxiv.org/abs/2011.09584): 17.7M parallel-jaw grasps, 8,872 objects, 262 categories. Он даёт масштаб и existing analytic/simulation grasps, но capacity labels следует пересчитать под собственные query definitions, чтобы не зависеть от candidate-label leakage.

Дополнительно возможны ShapeNet/GraspNet objects. [Dex-Net 2.0](https://arxiv.org/abs/1703.09312) показывает, что крупномасштабные synthetic analytic labels могут переноситься на real grasping; это indirect feasibility evidence, не гарантия для данной архитектуры.

### 9.2. ShelfOcc-1 benchmark

Нужен контролируемый benchmark, совпадающий с лабораторией:

- один target на shelf;
- один foreground occluder;
- wrist-camera viewpoints;
- target identity/category splits без mesh leakage;
- occlusion ratio buckets: `0–0.2`, `0.2–0.4`, `0.4–0.6`, `0.6–0.75`, `>0.75`;
- RealSense-like quantization, missing depth, edge flying pixels, pose noise;
- variation occluder width/depth/material;
- fixed candidate generator для всех selectors.

Occlusion ratio надо считать по target-visible pixels относительно unoccluded render, а не по arbitrary point count. Помимо random split обязателен category-disjoint и shape-family-disjoint split.

### 9.3. Offline labels

Для каждого full mesh и gripper query:

1. robust mesh-pad intersection на small pose perturbation set;
2. oriented normals и conservative friction cone;
3. paired-contact/antipodal или wrench margin;
4. mesh–swept-body collision за вычетом contact patches;
5. `y_A`, `y_B`, `y_{A∪B}=max(y_A,y_B)`;
6. random cross-grasp unions для joint-mode supervision.

Рекомендуется хранить только compact query parameters и Boolean/margin labels, не voxelized completed shape. Mesh нужен offline oracle, но не learning target.

### 9.4. Real data

Минимальная убедительная real-robot часть:

- 30–50 novel household targets разных shape families;
- 3–5 occluders;
- несколько controlled occlusion levels на object;
- paired or randomized-block comparison методов;
- automatic small-lift criterion плюс blinded visual audit;
- bootstrap confidence intervals и hierarchical/per-object random effects;
- отдельно contact failure, premature body collision и slip только как evaluation annotations, не causal model.

---

## 10. Экспериментальная программа и фальсификация

### 10.1. Research questions

**RQ1.** Элицитирует ли CCMN calibrated task-restricted capacity лучше unconstrained query MLP?  
**RQ2.** Даёт ли exact capacity composition `T(A∪B)−T(B)` лучший top-1 selection, чем direct success head и independence product?  
**RQ3.** Переносится ли operator на новые Boolean compositions, tolerances, gripper dimensions и unseen categories?  
**RQ4.** Достаточен ли reconstruction-free quotient для high-occlusion performance относительно local/full completion?  
**RQ5.** Даёт ли выигрыш реальную пользу при fixed candidate generator, а не улучшение candidate proposal?

### 10.2. Обязательные baselines

Все learnable baselines получают одинаковый observation backbone и candidate set, где возможно.

1. direct scalar `P(y_good|O,g)` BCE head;
2. independent marginals `P(A)·[1−P(B)]`;
3. unconstrained joint four-class head для `(H(A),H(B))`;
4. generic query-conditioned MLP, обученный на тех же union labels;
5. DCPF-style scalar collision/query probability network;
6. deterministic local occupancy/completion, LOE-style + analytical checker;
7. full completion + single completed-shape planner;
8. stochastic completion + MC expected score и lower-tail/CVaR;
9. TARGO-Net;
10. NeuGraspNet/GIGA/Contact-GraspNet/AnyGrasp, насколько их интерфейсы допускают честный fixed-scene protocol;
11. oracle full-shape analytical selector;
12. candidate-generator oracle recall ceiling.

Unconstrained joint four-class head — особенно важный baseline: для только двух events он может быть проще CCMN. Framework обязан выиграть при shared multi-query supervision и compositional/OOD tests, а не только на in-distribution `A∧¬B`.

### 10.3. Metrics

**Task:**

- top-1 geometric certificate success;
- simulation grasp-and-small-lift success;
- real-robot small-lift success;
- performance vs occlusion/noise bucket;
- regret to full-shape oracle;
- candidate oracle recall.

**Probabilistic:**

- NLL, Brier score, reliability diagrams, adaptive ECE;
- class-balanced versions при редких events;
- compositional error для `P(A∧¬B)`;
- success–coverage curve и AURC;
- calibration under category/noise/occlusion shift.

**Structural:**

- monotonicity violations на nested queries;
- complete-alternation/union inequality violations;
- representation-consistency: один union через разные legal decompositions;
- cross-query joint likelihood;
- validity после quantization/pruning.

**System:**

- latency per scene и per 1,000 candidates;
- peak memory;
- accuracy–latency curve по `H`;
- offline labeling cost.

### 10.4. Ablations

- number of modes `H∈{1,4,8,16,32,64}`;
- structured noisy-OR vs unconstrained MLP;
- no union supervision;
- only within-grasp vs cross-grasp union supervision;
- lifted contact-pair query vs independent two-contact queries;
- no observed-scene hard collision filter;
- query lattice resolution;
- pad tolerance/friction margin curriculum;
- RGB+depth vs depth only;
- noise augmentation;
- nuisance pose integration;
- optional conformal threshold;
- latent modes decoded/not decoded only as diagnostic — main model never receives reconstruction loss.

### 10.5. General-ML synthetic benchmark

Чтобы вклад не свёлся к robotics pipeline, нужен отдельный benchmark случайных множеств с известной истинной capacity:

- occluded unions of disks/convex bodies;
- Boolean random sets;
- Cox/clustered point processes;
- random curves/surfaces с multimodal hidden continuation;
- observations как censored projections;
- train queries — singles и small unions;
- test — unseen unions, intersections через inversion, nested ranges и changed query scale.

Сравниваются unconstrained neural set functions, monotone/submodular networks, independent marginals, autoregressive hit-vector model и CCMN. Здесь можно точно измерять integrated capacity error, joint-law recovery, finite-difference violations, sample efficiency и compositional generalization.

Это центральный ICLR experiment: он проверяет общий conditional random-set learning claim без confounds candidate generation, physics или robot control.

### 10.6. Жёсткие falsification gates

Направление следует остановить или существенно переосмыслить, если выполняется хотя бы одно:

1. **Certificate gap:** geometric certificate имеет AUROC ниже примерно `0.75` относительно sim/real small-lift outcome; тогда улучшается не тот target.
2. **No compositional gain:** CCMN не превосходит matched direct/joint heads на unseen Boolean compositions или changed tolerances; тогда «capacity» — лишняя параметризация.
3. **Candidate bottleneck:** oracle recall при сильной окклюзии настолько низок, что selector не способен улучшить success; тогда сначала нужен другой candidate generator.
4. **Mode collapse:** joint/union calibration не улучшается с `H`, modes дублируются, а independent baseline равен CCMN.
5. **Sim-to-real collapse:** uncertainty оказывается калиброванной только к renderer artifacts.
6. **Continuous inconsistency:** разные decompositions одного geometric query дают materially разные probabilities.

Эти gates должны быть preregistered до полного benchmark sweep, иначе легко post-hoc выбрать удобные metrics.

---

## 11. Почему идея может работать: косвенные доказательства

Косвенные evidence chains не доказывают CAPGrasp, но уменьшают риск каждого отдельного допущения.

1. **Hidden geometry действительно полезна.** TARGO, CenterGrasp, NeuGraspNet и local occupancy methods показывают, что completion/implicit geometry смягчают degradation при occlusion.
2. **Uncertainty формы полезнее одной completion.** Lundell et al. показывают статистически лучший grasp planning при учёте нескольких uncertain completions.
3. **Task-local representation может заменить full reconstruction.** Local Occupancy-Enhanced Grasping и TOSC показывают ценность focus на contact-relevant regions.
4. **Synthetic analytical supervision переносима.** Dex-Net 2.0 и масштаб ACRONYM делают offline hit-query labels реалистичной стратегией.
5. **Amortized probability queries вычислительно жизнеспособны.** Deep Collision Probability Fields демонстрирует высокоточную и быструю оценку collision probabilities без online integration, хотя решает другую uncertainty model.
6. **Structured set functions обучаемы.** [Deep Submodular Functions](https://proceedings.neurips.cc/paper/2016/hash/7fea637fd6d02b8f0adf6f7dc36aed93-Abstract.html), [Neural Estimation of Submodular Functions](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7b76eea0c3683e440c3d362620f578cd-Abstract.html), [Learning Coverage Functions](https://proceedings.mlr.press/v35/feldman14a.html) и [Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) показывают, что permutation-invariant и validity-constrained set-function architectures возможны.
7. **Proper + structural probability models полезны.** [SurvivalMonotonic-net](https://proceedings.mlr.press/v151/rindt22a.html) демонстрирует, что monotonicity-by-construction и proper censored likelihood способны одновременно улучшать validity, speed и likelihood в другой области.

Наиболее убедительная expected advantage возникает при **мультимодальном backside ambiguity**. Например, видимая передняя часть совместима и с тонким плоским объектом, и с толстым выпуклым. У первой mode возможна пара контактов без palm collision; у второй contact pair и collision возникают совместно. Independent marginals смешивают несовместимые факты, а latent coverage modes сохраняют correlation без декодирования двух shapes.

---

## 12. Novelty matrix

| Свойство | Completion methods | Direct success head | Stochastic completion | Grasp/contact distribution | Collision probability field | AQCL / CAPGrasp |
|---|---:|---:|---:|---:|---:|---:|
| Full hidden shape output не нужен | иногда | да | нет | да | да | **да** |
| Hidden geometry distribution | нет | неявно | да | косвенно | ограниченно | **task-restricted** |
| Joint contact/collision dependence | через decoded shape | непрозрачно | через samples | обычно нет | нет contacts | **явно через mode law** |
| Valid set capacity by construction | нет | нет | empirical MC | нет | нет | **да, finite algebra** |
| New Boolean query composition | analytical after reconstruction | нет | analytical per sample | нет | collision only | **да** |
| Proper query-level objective | обычно reconstruction loss | endpoint BCE | likelihood + MC | likelihood | BCE/regression | **capacity log score** |
| No online reconstruction/MC | completion нужен | да | нет | да | да | **да** |
| General random-set ML contribution | нет | нет | слабый | слабый | узкий | **центральный claim** |

### 12.1. Самая опасная reviewer-критика

> «Это direct success classifier, разложенный на два события и записанный через Choquet terminology.»

Ответ не может быть риторическим. Нужны три empirical facts:

1. один operator отвечает на unseen union/Boolean queries без retraining;
2. structural head имеет нулевые violations и лучшую calibration/sample efficiency, чем matched unconstrained models;
3. эта compositional accuracy приводит к лучшему top-1 selection при changed tolerance/gripper/category, а не только к красивым reliability plots.

Если эти результаты не получены, ICLR novelty claim не выдерживает.

---

## 13. Calibration и abstention

После обучения можно добавить held-out [Conformal Risk Control](https://arxiv.org/abs/2208.02814) для выбора threshold/abstention policy с контролем **маргинального ожидаемого monotone loss** при exchangeability. Это не основная novelty и не per-scene guarantee.

Правильная формулировка:

- calibration set выбирает threshold;
- report coverage и empirical risk с finite-sample bound;
- при distribution shift гарантия может нарушиться;
- если robot обязан всегда действовать, conformal module не используется для selection и остаётся диагностикой.

Не следует заявлять, что conformal method «гарантирует безопасный grasp для данного изображения».

---

## 14. ICLR-аудит

[ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide) просит оценивать ясность problem/motivation, позиционирование относительно literature, корректность evidence/rigor и достаточную значимость нового знания. [ICLR 2027 Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers) явно включает probabilistic methods, uncertainty, structured prediction, learning on geometries/topologies, hybrid/physics-informed learning и robotics/autonomy.

### 14.1. Почему fit существует

- новый general-ML object: conditional capacity functional;
- новый proper task-query objective;
- architecture validity through random-subset representation;
- decision-sufficiency quotient объясняет отказ от reconstruction;
- theory связывает capacity error и action regret;
- robotics служит demanding downstream validation;
- synthetic benchmark изолирует общий learning claim.

### 14.2. Что требуется для acceptance-level package

**Минимум:**

1. строгая finite-query theory и точные boundaries claims;
2. synthetic random-set benchmark с known ground truth;
3. matched-backbone baselines, особенно direct/joint query heads;
4. controlled ShelfOcc-1 simulator;
5. real-robot study;
6. release code, queries, labels и benchmark generation;
7. systematic novelty appendix с search strings и cut-off date;
8. failure cases и falsification gate results.

### 14.3. Главные риски

| Риск | Вероятность | Ущерб | Mitigation |
|---|---:|---:|---|
| «Fancy classifier» | высокая | критический | OOD Boolean composition + new tolerance/gripper tests; matched joint head. |
| Query lattice не соответствует continuous sets | средняя | высокий | canonical atoms, exact mesh labels, decomposition tests, explicit finite-only theorem. |
| Candidate generator теряет hidden grasps | высокая при >75% occlusion | высокий | report oracle recall; visible-root and prior-proposal mixture; не приписывать selector чужой bottleneck. |
| Certificate слабо коррелирует с lift | средняя | критический | early pilot gate; добавить только минимальный quasi-static margin, не full feasibility model. |
| Exponential modes | средняя | средний | low-rank `H`, sparse/hierarchical modes, measure calibration-vs-latency. |
| Sim-to-real noise | высокая | высокий | sensor recordings, domain randomization, real calibration split, category-disjoint testing. |
| Novelty collision с новой работой | неизбежно растёт | высокий | repeat search before submission; emphasize theorems/operator, not just application. |
| Training query distribution too narrow | средняя | высокий | curriculum, random unions, nested scales, explicit support statement. |

---

## 15. Рекомендуемая последовательность реализации

### Phase 0 — двухнедельный kill test

- 2D random-set toy with exact capacity;
- compare direct, independent, four-class joint и `H=4/16` CCMN;
- train singles + limited unions, test unseen unions/nested queries;
- проверить mode collapse и calibration.

**Продолжать**, только если structured model даёт compositional gain или заметно меньшую sample complexity.

### Phase 1 — geometric certificate pilot

- 200–500 meshes;
- один simplified gripper;
- exact `A`, `B`, union labels;
- full-shape oracle vs certificate vs simulated small lift;
- измерить candidate recall и certificate AUROC.

### Phase 2 — ShelfOcc-1

- scalable rendering/noise;
- fixed candidate protocol;
- matched backbones и completion baselines;
- category/shape-family OOD;
- latency and memory.

### Phase 3 — real robot

- preregistered paired study;
- target/occluder/occlusion blocks;
- confidence intervals;
- no cherry-picked grasp videos as primary evidence.

### Phase 4 — paper-strength generalization

- changed gripper pad width/tolerance;
- unseen task certificates composed from known atoms;
- random-set benchmark release;
- systematic literature refresh and theorem polishing.

---

## 16. Возможное название, abstract claim и one-sentence contribution

### Paper title

**Do Not Complete What You Cannot See: Conditional Capacity Learning for Reliable Grasping under Occlusion**

### One-sentence contribution

> We introduce conditional capacity learning, a reconstruction-free structured prediction framework that elicits the task-restricted hitting law of a latent random set and composes calibrated contact and collision events into grasp decisions through a valid finite Choquet capacity.

### Claim, который допустим только после успешных экспериментов

> Across synthetic random-set tasks, controlled single-occluder shelf scenes, and real parallel-jaw grasping, a validity-constrained capacity operator improves compositional calibration and high-occlusion top-1 selection over matched direct classifiers, local/full completion, and Monte Carlo shape uncertainty baselines at lower inference cost.

Не следует заранее писать «first», «guaranteed safe» или «state of the art». `First` требует обновлённого exhaustive search; safety выходит за геометрический certificate; SOTA требует принятого benchmark protocol и статистической значимости.

---

## 17. Финальная оценка направления

CAPGrasp — редкий кандидат, который одновременно соответствует узкой лабораторной задаче и раскрывается в broad ML problem. Его сильная сторона не в новом backbone, а в смене prediction target:

$$
\text{hidden shape reconstruction}
\quad\longrightarrow\quad
\text{conditional law of task queries}.
$$

Capacity functional является естественным минимальным объектом для existential geometry predicates. Lifted pair space выражает два согласованных контакта; negative query — hidden collision; difference of capacities даёт joint event без independence approximation. Proper query loss и mixture-coverage head делают формулировку efficiently learnable и structurally valid на finite query algebra.

Наиболее вероятный путь к существенному performance gain — сильная occlusion с несколькими правдоподобными backside modes, где deterministic completion ошибочно «усредняет» форму, independent event heads теряют correlation, а full stochastic completion слишком дорога. На слабой occlusion CAPGrasp может лишь сравняться с direct/local methods; это нормально и должно быть показано честно.

Рекомендация: **продолжать с короткого general-ML kill test, затем geometric-certificate pilot**. Не начинать с большой robot system. Если compositional advantage не возникает уже в контролируемых random-set задачах, broad claim неверен и направление надо закрыть. Если возникает и переносится в ShelfOcc-1, это правдоподобная ICLR-level работа с реальным SOTA-потенциалом, а не очередной вариант shape completion.

---

## 18. Основные источники

### Random sets, structured set functions и calibration

1. I. Molchanov, [Theory of Random Sets](https://www.nzdr.ru/data/media/biblio/kolxoz/M/MD/Molchanov%20I.%20Theory%20of%20Random%20Sets%20%28ISBN%20185233892X%29%28Springer%2C%202005%29%28501s%29_MD_.pdf), Springer, 2005.
2. B. Figliuzzi, [Introduction to Stochastic Geometry](https://www.cmm.minesparis.psl.eu/~figliuzzi/Stochastic_Geometry.pdf), tutorial notes: capacity/hitting functional and Choquet theorem.
3. R. Iyer, J. Bilmes, [Deep Submodular Functions](https://proceedings.neurips.cc/paper/2016/hash/7fea637fd6d02b8f0adf6f7dc36aed93-Abstract.html), NeurIPS 2016.
4. M. De, D. Chakrabarti, S. Dey, [Neural Estimation of Submodular Functions with Applications to Differentiable Subset Selection](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7b76eea0c3683e440c3d362620f578cd-Abstract.html), NeurIPS 2022.
5. V. Feldman, P. Kothari, [Learning Coverage Functions and Private Release of Marginals](https://proceedings.mlr.press/v35/feldman14a.html), COLT 2014.
6. M. Zaheer et al., [Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html), NeurIPS 2017.
7. P. Rindt et al., [SurvivalMonotonic-net](https://proceedings.mlr.press/v151/rindt22a.html), AISTATS 2022.
8. A. Angelopoulos et al., [Conformal Risk Control](https://arxiv.org/abs/2208.02814), ICLR 2024.

### Grasping, occlusion и uncertainty

9. Y. Xia et al., [TARGO: Benchmarking Target-driven Object Grasping under Occlusions](https://arxiv.org/abs/2407.06168), IJCV 2026; [project/benchmark](https://targo-benchmark.github.io/).
10. J. Lundell et al., [Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645), 2019.
11. K. Ma et al., [Local Occupancy-Enhanced Object Grasping with Multiple Triplanar Projection](https://arxiv.org/abs/2407.15771), ECCV 2024.
12. [Learning Any-View 6DoF Robotic Grasping in Cluttered Scenes via Neural Surface Rendering (NeuGraspNet)](https://arxiv.org/abs/2306.07392), 2024.
13. [CenterGrasp: Object-Aware Implicit Representation Learning for Simultaneous Shape Reconstruction and 6-DoF Grasp Estimation](https://arxiv.org/abs/2312.08240), 2024.
14. Z. Jiang et al., [Synergies between Affordance and Geometry: 6-DoF Grasp Detection via Implicit Representations](https://arxiv.org/abs/2104.01542), GIGA, RSS 2021.
15. M. Sundermeyer et al., [Contact-GraspNet](https://arxiv.org/abs/2103.14127), ICRA 2021.
16. A. Karke et al., [Object Pose and Shape Estimation for Grasping: Does it Work?](https://arxiv.org/abs/2605.26944), 2026.
17. Z. Feng et al., [FFHFlow: Diverse and Uncertainty-Aware Dexterous Grasp Generation via Flow Variational Inference](https://proceedings.mlr.press/v305/feng25a.html), CoRL 2025.
18. C. Enwerem et al., [Variational Neural Belief Parameterizations for Robust Dexterous Grasping under Multimodal Uncertainty](https://arxiv.org/abs/2604.25897), IROS 2026.
19. [AnyDexGrasp](https://graspnet.net/anydexgrasp/assets/files/AnyDexGrasp.pdf), ICLR 2025 Workshop.
20. W. Wu et al., [TOSC: Task-Oriented Shape Completion for Open-World Dexterous Grasp Generation from Partial Point Clouds](https://arxiv.org/abs/2601.05499), AAAI 2026.

### Probability queries, datasets и transfer evidence

21. F. Herrmann et al., [Safe and Efficient Path Planning under Uncertainty via Deep Collision Probability Fields](https://arxiv.org/abs/2409.04306), RA-L 2024.
22. A. Adamkiewicz et al., [CATNIPS: Collision Avoidance Through Neural Implicit Probabilistic Scenes](https://arxiv.org/abs/2302.12931), RSS 2023.
23. J. Mahler et al., [Dex-Net 2.0](https://arxiv.org/abs/1703.09312), RSS 2017.
24. C. Eppner et al., [ACRONYM: A Large-Scale Grasp Dataset Based on Simulation](https://arxiv.org/abs/2011.09584), ICRA 2021.
25. H. Fang et al., [GraspNet-1Billion](https://arxiv.org/abs/1912.13470), CVPR 2020.

### Venue criteria

26. [ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide).
27. [ICLR 2027 Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers).
