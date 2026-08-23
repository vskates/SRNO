# LimitGrasp для parallel-jaw grasping

## Итоговый вердикт

Самая сильная новая постановка второго research pass — не ещё один grasp
scorer, не реконструкция скрытой формы и не оценка feasibility всего движения,
а изучение **сходимости самого множества успешных действий**, когда supervision
получается из негладкого contact simulator.

Рабочее название paper:

> **LimitGrasp: Learning Stable Action Sets from Non-Converged Contact
> Simulators**

Одно предложение, содержащее всю идею:

> Contact-rich grasp datasets обычно объявляют один numerical simulator run
> физическим label; мы предлагаем рассматривать успешные grasps как
> последовательность множеств при refinement timestep, mesh и solver tolerance,
> учить её внутренний и внешний Painlevé--Kuratowski limits непосредственно из
> partial RGB-D и выбирать grasps из stable core, не запуская simulator и не
> восстанавливая full geometry на inference.

Это **conditional go**, а не обещание accept. Постановка имеет сильную
problem-level novelty только если предварительный эксперимент обнаружит
воспроизводимую numerical ambiguity band, которая:

1. не сводится к багам или различию физических contact models;
2. предсказывается из локальной наблюдаемой геометрии;
3. объясняет реальные ошибки default simulator сверх обычного grasp score;
4. позволяет при равном simulation budget выбирать grasps лучше, чем один
   high-fidelity oracle.

Если эти условия не выполняются, nested-envelope model будет выглядеть как
обычный multi-fidelity regression head и направление следует отвергнуть.

## 1. Точная задача и сознательно исключённый scope

Дано одно RGB-D наблюдение $o$ target object на полке с wrist camera. Перед
объектом может находиться один фронтальный occluder. Сцена не cluttered. Point
cloud может содержать depth quantization, dropout, outliers и calibration noise.
Считается, что target mask или crop уже получен существующим perception stack.

Модель оценивает parallel-jaw grasp

$$
g\in\mathcal G\subset (SE(3)/C_2)\times[w_{\min},w_{\max}],
$$

где $C_2$ учитывает finger-swap symmetry. Отдельный frozen candidate generator,
получающий только observation, возвращает

$$
G(o)=\{g_1,\ldots,g_M\}.
$$

Operational test намеренно короткий:

1. внешний motion stack уже поместил открытый gripper в стандартизованную
   terminal pre-grasp pose;
2. губки закрываются одним фиксированным force/velocity controller;
3. объект удерживается и поднимается или нагружается на несколько миллиметров;
4. success определяется retention и ограниченным relative slip.

Полный approach trajectory, arm reachability, collision-free motion до
pre-grasp, длительный lift и последующее manipulation не входят в label и
claim. Paper не изучает causal failure modes. RL и VLA не используются. На вход
модели не подаются complete mesh, voxel grid или scene SDF.

Полная геометрия и contact simulator доступны только offline для генерации
paired multi-resolution supervision.

## 2. Почему один simulator label не является ground truth

Пусть $z\in\mathcal Z$ — полное физическое состояние object, gripper pad и
локальной shelf scene, а

$$
o=R(z)
$$

— noisy RGB-D observation. Пусть $\xi\sim\Xi$ — измеренная physical execution
perturbation: ошибка terminal pose, controller repeatability и sensor-aligned
initialization. Распределение $\Xi$ фиксируется во всех numerical experiments.

Нужно различать три объекта:

1. реальный binary outcome заданного terminal protocol;
2. solution выбранной continuous compliant-contact model $\mathcal M$;
3. output конкретного mesh, timestep, collision margin, tolerance и numerical
   solver, который лишь приближает $\mathcal M$.

Обычный grasp dataset неявно отождествляет их:

$$
y(z,g)=Y_{\mathcal M,\gamma,k}(z,g,\xi),
$$

где $\gamma$ — выбранная discretization path, а $k$ — единственный уровень
resolution. При smooth forward problem ошибка этого отождествления может быть
малой. При contact creation, stick--slip switching, edge contact и separation
малое numerical изменение может передвинуть decision boundary в grasp space.

В результате два dataset generation pipelines с одинаковыми images,
objects и grasps могут задавать разные Bayes targets:

$$
f_{\gamma,k}(o,g)
=
\mathbb E\!\left[
Y_{\mathcal M,\gamma,k}(Z,g,\xi)\mid o,g
\right].
$$

Большая сеть не устраняет этот bias: при infinite data она точнее воспроизводит
выбранный numerical oracle.

Практическая мотивация уже видна в литературе:

- comparative contact-model analysis показывает, что numerical relaxation
  существенно меняет downstream robotics behavior:
  https://arxiv.org/abs/2304.06372
- SimBenchmark отдельно измеряет contact-solver error и integration error:
  https://leggedrobotics.github.io/SimBenchmark/
- IPC-GraspSim улучшает physical prediction parallel-jaw grasps более точной
  compliant-contact simulation, но с большой вычислительной стоимостью:
  https://arxiv.org/abs/2111.01391
- Get a Grip сообщает, что перенос hand initialization из in-contact в
  pre-grasp устранил nonphysical Isaac Gym behavior и нестабильные labels:
  https://arxiv.org/abs/2410.23701
- ICLR 2026 работа о hard contacts показывает сильную зависимость simulator
  gradients от stiffness и integration:
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/44039e59aaf6a41b16f1fc5b27bcd409-Abstract-Conference.html

Эти работы чинят или сравнивают simulator. Они не делают asymptotic action set
объектом perception-conditioned supervised learning.

## 3. Новый объект: пределы множества успешных grasпов

### 3.1 Refinement family

До генерации данных фиксируются:

- continuous contact model $\mathcal M$;
- material/contact parameters этой модели;
- terminal controller и success criterion;
- physical perturbation distribution $\Xi$;
- семейство $\Gamma$ численных refinement paths, предназначенных приближать одну
  и ту же $\mathcal M$.

Путь $\gamma\in\Gamma$ задаёт последовательность:

$$
\left(
\Delta t_{\gamma,k},
\Delta x_{\gamma,k},
\varepsilon_{\mathrm{contact},\gamma,k},
\varepsilon_{\mathrm{solve},\gamma,k},
N_{\mathrm{iter},\gamma,k}
\right),
$$

и scale $\eta_{\gamma,k}\downarrow0$. Здесь $\Delta x$ может включать object
collision mesh и compliant-pad discretization.

Для fixed full scene и grasp определяется operationally smoothed utility

$$
q_{\gamma,k}(z,g)
=
\Pr_{\xi\sim\Xi}
\left[
Y_{\mathcal M,\gamma,k}(z,g,\xi)=1
\right].
$$

Во всех уровнях используются common random perturbation seeds. Иначе variance
Monte Carlo ошибочно выглядит как numerical non-convergence.

При threshold $\tau$ каждый уровень задаёт closed success set

$$
S_{\gamma,k}(z;\tau)
=
\overline{
\{g\in\mathcal G:q_{\gamma,k}(z,g)\ge\tau\}
}.
$$

Paper изучает convergence этих action sets, а не trajectory contact forces
во всём rollout.

### 3.2 Painlevé--Kuratowski inner и outer limits

Пусть $d_{\mathcal G}$ — metric на compact рабочей области gripper-symmetry
quotient. Для path $\gamma$ определим

$$
\operatorname{Li}S_{\gamma,k}
=
\left\{
g:
\limsup_{k\to\infty}
d_{\mathcal G}(g,S_{\gamma,k})=0
\right\},
$$

$$
\operatorname{Ls}S_{\gamma,k}
=
\left\{
g:
\liminf_{k\to\infty}
d_{\mathcal G}(g,S_{\gamma,k})=0
\right\}.
$$

Inner limit содержит actions, которые можно приблизить успешными grasps на
каждом достаточно fine уровне. Outer limit также допускает успех лишь вдоль
подпоследовательности refinements.

Объединим объявленные consistent paths:

$$
S_{\mathrm{core}}(z;\tau)
=
\bigcap_{\gamma\in\Gamma}
\operatorname{Li}S_{\gamma,k}(z;\tau),
$$

$$
S_{\mathrm{possible}}(z;\tau)
=
\overline{
\bigcup_{\gamma\in\Gamma}
\operatorname{Ls}S_{\gamma,k}(z;\tau)
}.
$$

**Stable success core** $S_{\mathrm{core}}$ содержит grasps, устойчивые ко всем
объявленным refinement paths. **Possible success envelope**
$S_{\mathrm{possible}}$ содержит всё, что может оставаться successful вдоль
какого-либо arbitrarily fine path.

Разность

$$
B_{\mathrm{num}}(z;\tau)
=
S_{\mathrm{possible}}(z;\tau)
\setminus
S_{\mathrm{core}}(z;\tau)
$$

называется **numerical ambiguity band**.

Если schemes корректно сходятся к единому regular contact solution, два
множества совпадают вне threshold boundary. Если gap имеет значимую меру, нельзя
честно объявить один scalar simulator output физическим label без дополнительной
model-selection assumption.

### 3.3 Observation-conditioned learning target

Model не получает $z$. Она учит две action fields:

$$
L(o,g)
=
\Pr_{Z\mid o}
\left[
g\in S_{\mathrm{core}}(Z;\tau)
\right],
$$

$$
U(o,g)
=
\Pr_{Z\mid o}
\left[
g\in S_{\mathrm{possible}}(Z;\tau)
\right].
$$

$L$ — вероятность принадлежности stable core, а

$$
A_{\mathrm{num}}(o,g)=U(o,g)-L(o,g)
$$

— вероятность принадлежности numerical ambiguity band. Hidden geometry и
sensor noise остаются обычной conditional input uncertainty; numerical
ambiguity определяется только сравнением *одного и того же* полного
scene/grasp pair под refinement.

Основной selector:

$$
\widehat g(o)
\in
\arg\max_{g\in G(o)}L_\theta(o,g).
$$

Можно добавить coverage condition

$$
U_\theta(o,g)-L_\theta(o,g)\le\beta,
$$

но abstention не является novelty.

### 3.4 Почему это не arbitrary simulator voting

Разные engines нельзя автоматически считать refinement paths. Rigid
point-contact, penalty contact, hydroelastic contact и finite-element
compliant pads могут описывать разные continuous models.

Primary benchmark обязан:

1. сопоставить units, friction law, restitution, pad stiffness, controller и
   success criterion;
2. строить refinement внутри одной declared model;
3. использовать второй implementation path только если можно защитить
   convergence к той же model;
4. отдельно обозначать disagreement разных physical models как
   **model-form uncertainty**, а не numerical limit.

Этот запрет защищает paper от превращения в min/max ensemble нескольких
несопоставимых engines.

## 4. Теоретическое ядро

### 4.1 Finite-prefix non-identifiability

Никакой finite набор simulator resolutions сам по себе не идентифицирует
infinite-resolution limit.

Пусть наблюдены sets $S_1,\ldots,S_K$. Существуют две continuation sequences,
совпадающие на всём prefix:

$$
S_k^{(a)}=S_k^{(b)},\qquad k\le K,
$$

но после $K$ первая sequence остаётся постоянной, а вторая чередуется между
двумя separated sets. Их inner/outer limits различны.

Следовательно:

> Limit extrapolation невозможен distribution-free; paper обязан объявить
> convergence class и проверять её на unseen finer levels.

Это важный negative result. Он запрещает скрывать extrapolation assumption за
нейросетью.

### 4.2 Локальная convergence model

Away from contact-mode intersections разумная рабочая assumption задаётся
asymptotic expansion signed distance:

$$
d_{\mathcal G}(g,S_{\gamma,k})_{\mathrm{signed}}
=
d_{\gamma,\infty}(g)
+c_\gamma(g)\eta_{\gamma,k}^{p_\gamma(g)}
+o(\eta_{\gamma,k}^{p_\gamma(g)}),
$$

где $p_\gamma(g)>0$.

Вблизи mode switches expansion может не существовать или быть nonuniform.
Именно поэтому target задаётся set limits, а не Richardson extrapolation одного
binary label. Такие regions должны проявляться как ненулевая envelope width, а
не насильно fit-иться smooth power law.

### 4.3 Set consistency

При следующих assumptions:

1. numerical contact solutions graph-converge к solution set continuous model;
2. success functional continuous вне threshold boundary;
3. рабочая grasp область compact;
4. learned signed-distance envelopes сходятся uniform;

predicted inner/outer sets сходятся в Painlevé--Kuratowski sense. При
дополнительной regularity и nonempty compact sets можно получить Hausdorff
convergence.

Claim нельзя формулировать для произвольного rigid frictional contact solver.
Assumptions и boundary exceptions должны быть указаны рядом с theorem.

### 4.4 Decision stability

Если learned stable-core probability удовлетворяет

$$
\sup_{g\in G(o)}
|\widehat L(o,g)-L(o,g)|
\le\delta,
$$

а $\widehat g$ и $g^\star$ максимизируют соответственно $\widehat L$ и $L$ на
одном candidate set, то

$$
L(o,g^\star)-L(o,\widehat g)\le2\delta.
$$

Таким образом uniform action-field error напрямую контролирует downstream
selection. Average regression error этого не гарантирует.

### 4.5 Solver-selection dependence обычного ERM

Пусть два finite oracles $\alpha$ и $\beta$ создают условные success
probabilities $f_\alpha(o,g)$ и $f_\beta(o,g)$. Binary cross-entropy ERM на
каждом dataset consistency-converges к своему Bayes function.

Если

$$
\Pr_{o,g}
\left[
f_\alpha(o,g)\ne f_\beta(o,g)
\right]>0,
$$

то infinite data не делает scorers oracle-independent. Это label-definition
problem, а не variance или model capacity problem.

### 4.6 Safe refinement stopping

Пусть validated numerical error bound на level $k$ задаёт interval

$$
q_\infty(z,g)\in
[\ell_k(z,g),u_k(z,g)].
$$

Если $u_k<\tau$, grasp останется failure; если $\ell_k>\tau$, он останется
success. Только intervals, пересекающие threshold, требуют следующего
expensive refinement. Это даёт principled offline label allocation.

## 5. Efficiently learnable formalization: Nested Limit Field

### 5.1 Представление

Назовём модель **Nested Limit Field (NLF)**. Один point/ray encoder
$E_\theta(o)$ обрабатывает только target-masked local RGB-D и небольшое число
явно наблюдаемых obstacle/shelf points около queried closing region.

Для каждого candidate grasp query decoder возвращает

$$
\begin{aligned}
a_\theta(o,g),\quad
b^-_\theta(o,g),b^+_\theta(o,g),\quad
c^-_\theta(o,g),c^+_\theta(o,g),\quad
p^-_\theta(o,g),p^+_\theta(o,g),
\end{aligned}
$$

где $b^\pm,c^\pm\ge0$, а $p^\pm$ ограничены положительным interval,
валидированным pilot study.

Для scalar resolution coordinate $\eta$ nested envelopes имеют вид

$$
\ell_\theta(o,g,\eta)
=
\sigma\left(
a_\theta-b^-_\theta-c^-_\theta\eta^{p^-_\theta}
\right),
$$

$$
u_\theta(o,g,\eta)
=
\sigma\left(
a_\theta+b^+_\theta+c^+_\theta\eta^{p^+_\theta}
\right).
$$

При $\eta\downarrow0$ lower envelope не убывает, upper envelope не возрастает.
Formal limit prediction:

$$
L_\theta(o,g)=\sigma(a_\theta-b^-_\theta),
$$

$$
U_\theta(o,g)=\sigma(a_\theta+b^+_\theta).
$$

$c^\pm$ описывают uncertainty, исчезающую при numerical refinement.
$b^\pm$ разрешают residual path disagreement или nonuniform boundary behavior.
Если declared model имеет unique well-resolved limit, regularizer и data должны
сжимать $b^\pm$ почти до нуля вдали от decision boundary.

Это не длинный state vector. После shared encoder модель выдаёт семь scalars на
grasp; inference равен $O(M)$ и не включает simulation, mesh reconstruction или
SDF queries.

### 5.2 Несравнимые refinement coordinates

Один $\eta$ допустим только для заранее заданной path, например

$$
\Delta t_k=\Delta t_0/4^k,
\qquad
\Delta x_k=\Delta x_0/2^k,
$$

с одновременно затягиваемым solver tolerance.

Если timestep, pad mesh и iteration budget меняются независимо, между settings
нет естественного total order. Нельзя произвольно кодировать их одним fidelity
index. Тогда NLF использует маленькую monotone lattice

$$
r_\theta(o,g,\eta_t,\eta_x,\eta_s)\ge0
$$

с coordinatewise monotonicity:

$$
\frac{\partial r_\theta}{\partial\eta_i}\ge0.
$$

Envelopes задаются как $\sigma(a-b^- - r^-)$ и
$\sigma(a+b^+ + r^+)$. Architecture должна отражать partial order, но
конкретный lattice layer не является novelty.

### 5.3 Finite tail labels

Для fixed $(z,g,\gamma,k)$ выполняются $N$ matched perturbation trials. Из
числа successes строятся binomial confidence bounds

$$
\operatorname{LCB}_{\gamma,k}(z,g),
\qquad
\operatorname{UCB}_{\gamma,k}(z,g).
$$

Empirical tail envelopes:

$$
\widehat\ell_k(z,g)
=
\min_{\gamma,\;j\ge k}
\operatorname{LCB}_{\gamma,j}(z,g),
$$

$$
\widehat u_k(z,g)
=
\max_{\gamma,\;j\ge k}
\operatorname{UCB}_{\gamma,j}(z,g).
$$

При удалении coarse prefix lower target может только вырасти, upper target
может только уменьшиться. Это finite-data analog inner/outer limit.

Training loss:

$$
\begin{aligned}
\mathcal L={}&
\sum_{b,m,k}
w_{bmk}
\left[
\rho\bigl(
\ell_{\theta,bmk}-\widehat\ell_{bmk}
\bigr)
+
\rho\bigl(
u_{\theta,bmk}-\widehat u_{bmk}
\bigr)
\right]\\
&+\lambda_{\mathrm{bin}}\mathcal L_{\mathrm{binomial}}
+\lambda_w
\mathbb E[U_\theta-L_\theta]
+\lambda_{\mathrm{mono}}\mathcal L_{\mathrm{order}}.
\end{aligned}
$$

$\rho$ — robust regression loss. Width penalty разрешён только при held-out
finer-level coverage constraint; иначе network научится искусственно узким
intervals. $\mathcal L_{\mathrm{order}}$ нужен только для general architecture,
не гарантирующей nesting by construction.

### 5.4 Grouped supervision

Training item должен сохранять identity полного scene/grasp:

$$
\mathcal D_{b,m}
=
\left(
o_b,g_m,
\{Y_{b,m,\gamma,k,n}\}_{\gamma,k,n}
\right).
$$

Если результаты разных levels случайно перемешать между похожими objects или
grasps, модель увидит ordinary heteroscedastic label noise и не сможет отделить
numerical flips от physical/data variation.

Candidate generator запускается один раз по $o_b$. Один и тот же $G(o_b)$
копируется во все simulations. Использовать full mesh для генерации candidates
запрещено: тогда easiest candidates могут зависеть от numerical oracle и
benchmark перестанет измерять perception-conditioned reranking.

### 5.5 Budget-aware refinement

Expensive finest labels нужны не для всех pairs.

1. Все $(z,g)$ запускаются на cheap coarse setting.
2. Random stratified subset запускается на всех levels и образует unbiased
   evaluation set.
3. Для остальных NLF оценивает ожидаемое уменьшение interval.
4. Refine pairs, у которых envelope пересекает $\tau$, нарушается ожидаемый
   decay или велико влияние на top-$K$ action selection.
5. Periodically добавляется uniform exploration batch, чтобы learned allocator
   не скрывал собственные blind spots.

Это active offline data acquisition, не RL. В сравнении с baselines считается
реальный compute: GPU-hours, solver steps и число perturbation rollouts.

### 5.6 Inference

На роботе:

1. получить RGB-D и target crop;
2. сгенерировать $M$ candidates существующим method;
3. одним encoder pass вычислить $L_\theta$, $U_\theta$ для каждого grasp;
4. исключить analytic collisions по наблюдаемым points;
5. выбрать максимальный $L_\theta$ среди grasps с допустимой ambiguity width.

Full geometry, numerical-resolution variables и simulator IDs на inference
отсутствуют. Model предсказывает limit target по observation, а не выбирает
любимый solver.

## 6. Multi-Resolution Contact Grasp benchmark

### 6.1 Objects и observations

Первый full benchmark:

- 200--500 rigid objects;
- ordinary household geometry;
- специально включённые thin rims, rounded edges, shallow chamfers,
  near-parallel faces, narrow necks и small concavities;
- single shelf support;
- один frontal occluder или его отсутствие;
- no clutter.

Для каждой scene рендерится wrist RGB-D при:

- no, mild и severe non-total occlusion;
- нескольких camera poses внутри лабораторного диапазона;
- measured depth quantization, dropout и outlier process;
- target mask и compact shelf-plane descriptor.

Split строится по object geometry, а не по camera frame. Adversarial contact
features должны встречаться только в части test objects, чтобы проверить, учит
ли модель generic numerical instability cues.

### 6.2 Candidate set

Frozen observation-only generator создаёт 256--512 candidates. В benchmark
сохраняются:

- pose и width;
- standard learned grasp score;
- analytic visible-geometry margins;
- local RGB-D/ray crop indices;
- candidate recall относительно full-mesh oracle только как diagnostic.

LimitGrasp не получает преимущества от нового generator. Главный destructive
baseline использует те же candidates и encoder.

### 6.3 Refinement grid

Минимальная grid должна включать:

- timestep levels хотя бы с $4\times$ ratio;
- object collision meshes с контролируемой Hausdorff error;
- compliant-pad mesh levels;
- contact/collision tolerance;
- nonlinear or complementarity solve tolerance;
- solver iteration budget;
- одинаковый controller sampling rate или его согласованный refinement.

Для каждой coordinate строятся one-at-a-time curves и совместные prescribed
paths. Только joint refinement может претендовать на continuum target;
one-at-a-time curves нужны для attribution.

Если используется IPC-GraspSim, его high-accuracy compliant-contact result
служит одним path/reference, но не объявляется абсолютной physical truth.
Если используется MuJoCo/Isaac, параметры должны быть переведены в одну
физическую convention настолько, насколько это возможно; несовместимые
settings остаются отдельным model-form experiment.

### 6.4 Utility labels

Для каждого selected $(z,g)$:

1. gripper стартует из одинаковой terminal pre-grasp;
2. initial state переносится между discretizations без penetration;
3. применяются common random pose/controller perturbations;
4. closure и малый load test имеют одинаковые physical durations;
5. записываются binary success, relative slip, contact-mode summary и solver
   diagnostics.

Model обучается на success probability envelopes. Contact forces и modes
используются для scientific analysis numerical flips, но не являются
дополнительным длинным input или causal failure labels.

### 6.5 Physical experiment

На actual parallel-jaw gripper выбирается balanced set:

- stable-core successes;
- stable failures;
- numerical ambiguity-band grasps;
- pairs с почти одинаковым default simulator score, но разной predicted
  ambiguity.

Каждый grasp повторяется при sampled terminal pose errors. До test внешний
planner приводит gripper в terminal pose; approach не оценивается.

Главная regression:

$$
\Pr(Y_{\mathrm{real}}=1)
\sim
s_{\mathrm{default}}
+A_{\mathrm{num}}
+m_{\mathrm{geom}}
+\text{object random effect}.
$$

$A_{\mathrm{num}}$ должен давать incremental predictive value сверх default
score и geometric margin. Ещё сильнее paired test: среди score-matched grasps
stable-core candidate должен чаще проходить real load test.

Numerical stability не равна physical correctness. Реальный test необходим,
потому что consistently wrong solver может иметь идеальную convergence.

## 7. Обязательный дешёвый falsification pilot

До обучения NLF:

- 30--50 objects;
- около 5,000 diverse grasps;
- четыре nested timestep/tolerance levels;
- два collision meshes и два pad meshes;
- 16--32 common perturbation seeds;
- минимум один carefully matched continuous contact model;
- небольшой real balanced subset.

Проект продолжать только при одновременном выполнении условий:

1. Default-to-finest label flip rate не меньше 5% overall или не меньше 10% в
   заранее объявленном low-margin stratum.
2. Flips воспроизводимы при common seeds и концентрируются около interpretable
   contact/geometric switching regions; они не похожи на random engine
   nondeterminism.
3. После исправления unit, initialization, collision-margin и controller
   mismatches остаётся существенная refinement dependence.
4. Simple nested tail extrapolator на unseen finer level достигает не меньше
   90% envelope coverage.
5. При этом mean probability width меньше 0.20 хотя бы для 70% test pairs;
   trivial interval $[0,1]$ не считается успехом.
6. Ambiguity-band membership предсказывает real error после conditioning на
   default score и analytic margin.
7. Equal-compute strategy «один finest/IPC run на меньшем числе samples» не
   достигает того же selection result.
8. Хотя бы для половины test observations имеется candidate с
   $L(o,g)>0.7$; иначе stable-core selection vacuous.

Failure пунктов 1--3 убивает problem claim. Failure пунктов 4--5 убивает
learnability claim. Failure пунктов 6--7 убивает grasping-value claim, даже если
numerical-analysis result сам по себе интересен.

Pilot обязан завершиться go/no-go memo до реализации большой perception model.

## 8. Полная экспериментальная программа

### 8.1 Splits

- unseen object instances;
- unseen object categories;
- unseen contact-feature programs;
- unseen occluder dimensions и visibility fractions;
- unseen PCD noise severity;
- unseen refinement combinations;
- **unseen finer levels**, превосходящие все training fidelities;
- synthetic-to-real objects;
- optional transfer на второй gripper pad material.

Последний split не должен использоваться для claim solver convergence одной
model; это transfer test.

### 8.2 Destructive baselines при одинаковом budget

1. Default single-fidelity grasp scorer.
2. Direct finest-observed-label head.
3. Direct worst-simulator/min-label head.
4. Deep ensemble single-fidelity scorers.
5. Domain randomization по physical friction, mass и pose.
6. Mean и worst-case arbitrary simulator ensemble.
7. Multi-fidelity Gaussian process.
8. Deep multi-fidelity regressor, предсказывающий designated highest fidelity.
9. Finest solver для каждого training item.
10. IPC-GraspSim или другой наиболее accurate available oracle на subset.
11. NLF без asymptotic term, то есть просто two-headed interval network.
12. Oracle, видящий все refinement levels.
13. Standard partial-PCD grasp scorer и analytic force-closure baseline.

Главные destructive comparisons — пункты 2, 7, 9 и 10. Если NLF не выигрывает
по unseen-level calibration и real selection при равном compute, новая
формализация не оправдана.

### 8.3 Metrics

- per-coordinate и joint-path label flip rate;
- successive-set Hausdorff or Chamfer distance на dense candidate graph;
- core/possible set precision и recall;
- envelope coverage на unseen finer levels;
- width при fixed coverage;
- calibration of $L_\theta$ и $U_\theta$;
- real grasp success и slip;
- stable-core selection regret против all-level oracle;
- abstention-risk curve;
- simulation calls, solver steps и wall-clock;
- inference latency/memory;
- performance versus occlusion и PCD noise;
- candidate recall отдельно от evaluator quality.

Aggregate accuracy недостаточна: dataset может быть dominated очевидными
stable failures и successes. Metrics обязательно stratified по default score
margin и ambiguity status.

### 8.4 Ablations

- только timestep, только mesh, только tolerance и joint refinement;
- scalar $\eta$ против monotone multi-coordinate lattice;
- без $b^\pm$, то есть forced unique limit;
- без $c^\pm$, то есть static interval;
- power-law tail против nonparametric monotone envelope;
- matched против independent perturbation seeds;
- grouped против randomly shuffled supervision;
- с и без adaptive refinement;
- random против score-threshold label allocation;
- exact RGB-D против measured noise;
- no/mild/severe occlusion;
- local crop radius;
- $M\in\{128,256,512,1024\}$;
- fixed candidate generator alternatives;
- success probability против hard one-rollout labels.

### 8.5 Positive controls

Нужны задачи, где convergence известна лучше, чем в full grasp simulation:

1. frictionless or single-contact analytic impact;
2. planar block with a known complementarity solution;
3. smooth compliant contact with verified reference integration;
4. deliberately inconsistent solver as negative control.

Они показывают, что NLF сжимает interval при genuine convergence и сохраняет gap
при path-dependent output. Один grasp benchmark недостаточен для broad ML claim.

## 9. Novelty boundary после adversarial literature audit

| Направление | Что уже существует | Чего оно не даёт |
|---|---|---|
| Direct partial-RGB-D grasping | Scores или poses из partial point cloud | Не проверяет, сходится ли simulator-defined supervised target |
| TARGO / occlusion-aware grasping | Target completion и fusion при single-view occlusion | Occlusion является input difficulty; numerical label refinement не является learning object |
| IPC-GraspSim | Более accurate compliant parallel-jaw simulator | Один improved oracle, а не learned inner/outer limits action sets |
| Contact-engine benchmarks | Physical/computational tests разных contact solvers | Нет perception-conditioned action field, multi-resolution training или real grasp selector |
| DiffMJX | Более полезные gradients hard-contact simulator | Не изучает convergence forward binary/probabilistic action labels |
| Multi-fidelity regression | Cheap prediction designated high-fidelity output | Последний finite level объявлен truth; нет asymptotic inner/outer set target |
| Discretization-invariant neural operators | Architecture работает на разных input/output grids | Discretization относится к represented physical fields, не к label oracle decision boundary |
| Domain randomization | Expected robustness к physical nuisance distribution | Numerical approximation settings ошибочно нельзя трактовать как deployment randomness |
| Multi-simulator ensemble | Voting, mean или worst-case по engines | Не требует общей continuous model и не различает model-form и discretization uncertainty |
| Robust grasp planning | Robustness к pose, shape, friction или load | Physical uncertainty меняется; в LimitGrasp она фиксирована, меняется numerical approximation |
| Conformal/interval prediction | Coverage для unknown targets | Не определяет правильный target и не создаёт set limit |
| LimitGrasp | Inner/outer action-set limits, paired refinement supervision, nested amortized field, unseen-fidelity и real tests | Новый stack, который ещё должен пройти falsification pilot |

Ключевые general ML boundaries:

- multi-fidelity active learning:
  https://proceedings.mlr.press/v202/wu23p.html
- multi-fidelity Gaussian processes:
  https://proceedings.mlr.press/v130/wang21c.html
- discretization-invariant operator learning:
  https://iclr-blogposts.github.io/2026/blog/2026/discretisation-invariance/
- learning simulation similarity metrics:
  https://proceedings.mlr.press/v119/kohl20a.html

Поэтому **не являются novelty по отдельности**:

- min/max нескольких labels;
- uncertainty head;
- monotone network;
- power-law extrapolation;
- active fidelity allocation;
- Kuratowski notation;
- local point-cloud encoder;
- robust argmax.

Защищаемая contribution stack:

1. новая supervised-learning задача — action-set limits из approximate
   nonsmooth label oracles;
2. finite-prefix non-identifiability и solver-dependent ERM result;
3. nested limit estimator с held-out finer-level evaluation;
4. paired multi-resolution contact-grasp benchmark;
5. empirical discovery и geometry numerical ambiguity band;
6. real evidence, что stable-core selection исправляет simulator-induced
   grasp ranking errors при равном compute;
7. перенос principle на одну простую non-grasp contact task.

Прямого совпадения с этим stack в проведённом поиске не найдено. Это не
разрешает писать «first» без повторного Google Scholar, Semantic Scholar, DBLP,
OpenReview и arXiv audit перед submission. Особенно нужно повторить поиск по
фразам:

- *learning from non-converged simulators*;
- *numerical-label uncertainty*;
- *action-set convergence under discretization*;
- *solver-invariant affordance learning*;
- *multi-resolution contact labels*;
- *Kuratowski limit learned feasible set*.

## 10. Mock review по официальным критериям ICLR

Актуальный ICLR 2027 reviewer guide предлагает спрашивать:

1. какая конкретная проблема решается;
2. хорошо ли мотивирован и расположен в literature подход;
3. подтверждены ли claims rigorously;
4. создаёт ли работа значимое новое знание;
5. может ли результат открыть новое направление, даже без established
   leaderboard SOTA.

Источник:
https://iclr.cc/Conferences/2027/ReviewerGuidelines

### Вероятные strengths

- Вопрос точен: какой action target должен учить predictor, если contact oracle
  доступен только через numerical refinement?
- Problem existence проверяется label-flip curves, а не общей фразой
  «simulation is imperfect».
- Set-limit definition не зависит от любого finite coarse prefix.
- Negative theorem честно показывает необходимость convergence assumption.
- Theory, data grouping, architecture и unseen-level evaluation следуют из
  одной постановки.
- Deployment model мала и не требует simulator или complete geometry.
- Physical experiment отличает numerical stability от abstract convergence.
- Result потенциально обобщается на contact-rich scientific ML за пределами
  grasping.

### Вероятные причины weak reject

1. «Это robust multi-fidelity regression с новым названием».
2. Authors смешали разные physical models и назвали их discretizations.
3. Один high-fidelity simulator дешевле и точнее всей конструкции.
4. Disagreement создан заведомо плохими coarse settings.
5. Power-law extrapolation не работает около contact switches.
6. Stable set пуст или слишком conservative.
7. Numerical stability не коррелирует с real success.
8. Model improvement вызван большим label budget.
9. Candidate generator определяет весь результат.
10. Теория применима только away from boundary, где проблема и так мала.
11. Kuratowski limits декоративны: experiments используют только min/max
    конечных levels.
12. Grasp benchmark недостаточен для broad ICLR audience.

Экспериментальная программа отвечает на эти objections:

- matched continuous model и separate model-form study отвечают на пункт 2;
- equal-compute finest-oracle baseline — на 3 и 8;
- practical settings и post-debug pilot — на 4;
- nonparametric envelope и честные failure regions — на 5;
- common-grasp coverage — на 6;
- balanced real experiment — на 7;
- frozen shared candidates и recall oracle — на 9;
- boundary-stratified metrics — на 10;
- unseen-finer set-distance evaluation — на 11;
- analytic contact positive control — на 12.

### Честная оценка acceptance potential

- **Сейчас, без pilot:** weak reject. Problem hypothesis правдоподобна, но
  central empirical phenomenon не измерен.
- **После positive simulation pilot, без real test:** borderline. Reviewer
  может считать ambiguity артефактом solver.
- **После real correlation, но без equal-compute high-fidelity baseline:**
  borderline/weak reject; practical value не доказан.
- **После всех go/no-go gates, unseen-fidelity generalization, strong
  destructive baselines и real improvement:** правдоподобный ICLR
  accept-level submission.
- **Если один calibrated IPC/high-fidelity oracle выигрывает:** proposal
  закрыть, а не добавлять architecture.

Главная potential contribution — не performance gain сам по себе, а новое
знание: contact-rich supervised action labels могут иметь геометрически
структурированную, наблюдаемую numerical ambiguity band, которую нельзя
корректно представить единственным finite-oracle label.

## 11. Ограничения, которые нужно объявить заранее

1. Limit существует только относительно declared continuous contact model и
   refinement family.
2. Different physical contact models могут иметь разные корректные limits.
3. Finite data не идентифицирует asymptotic limit без assumptions.
4. Set convergence может быть nonuniform около contact-mode intersections.
5. Stable numerical prediction не гарантирует physical correctness.
6. Model учит conditional probability по training object distribution; она не
   даёт worst-case guarantee для произвольной hidden geometry.
7. $S_{\mathrm{core}}$ может быть слишком conservative при широком path family.
8. Threshold $\tau$ влияет на set topology; нужно показывать curves по
   нескольким заранее выбранным thresholds.
9. Candidate generator ограничивает reachable action set и верхнюю границу
   результата.
10. Real pad wear, temperature и contamination меняют physical model и не
    являются numerical refinement.
11. Полный approach, collision-free reachability и long-horizon manipulation
    остаются вне scope.
12. Adaptive labeling может вводить sampling bias; uniform finest-level holdout
    обязателен.
13. Multi-resolution compliant simulation может оказаться дороже достаточного
    числа real trials.
14. Failure to outperform a direct finest-label model нельзя скрывать
    qualitative set visualizations.

## 12. Порядок исполнения проекта

1. Зафиксировать terminal close-and-small-load protocol и измерить $\Xi$.
2. Выбрать одну continuous compliant-contact model $\mathcal M$.
3. Выписать все numerical coordinates и построить две prescribed refinement
   paths.
4. Проверить unit, initialization, penetration и controller equivalence.
5. Зафиксировать один observation-only candidate generator.
6. Собрать 30--50 object pilot и 5,000 grasp pairs.
7. Построить flip-rate, set-distance и convergence plots без neural model.
8. Выполнить простой tail extrapolation на withheld finer level.
9. Провести balanced real pilot.
10. Принять go/no-go решение по разделу 7.
11. Только после go реализовать NLF и equal-budget baselines.
12. Добавить adaptive refinement с uniform holdout.
13. Масштабировать benchmark и добавить analytic non-grasp control task.
14. Повторить full novelty search.
15. Строить paper вокруг new target и empirical law; architecture оставить
    минимальным проверяемым estimator.

### Минимальный engineering pilot

Самый дешёвый executable вариант не требует сразу нескольких engines:

1. взять один simulator с настраиваемым compliant contact;
2. зафиксировать pad material и controller;
3. построить $4\times2\times2$ grid timestep, collision mesh и solver tolerance;
4. использовать Sobol/common-random pose perturbations;
5. считать empirical success sets на 512 shared candidates;
6. анализировать tail min/max и successive set distance;
7. напечатать 8--12 objects, максимизирующих predicted disagreement.

Если уже на этом этапе default labels стабильны или disagreement не переносится
в real, full benchmark не нужен.

## 13. Paper skeleton

### Introduction

- simulator labels не являются автоматически ground truth;
- contact makes decision sets nonuniform under refinement;
- learning question и stable action-set limits;
- summary contributions.

### Problem setup

- terminal grasp protocol;
- physical/numerical separation;
- refinement paths;
- success sets.

### Action-set limits

- inner/outer definitions;
- ambiguity band;
- observation-conditioned fields;
- finite-prefix impossibility.

### Method

- Nested Limit Field;
- finite tail confidence envelopes;
- grouped supervision;
- adaptive refinement.

### Theory

- set consistency under explicit assumptions;
- solver-dependent ERM;
- selection regret;
- stopping rule.

### Benchmark

- matched multi-resolution simulations;
- observation and candidate controls;
- real balanced subset;
- analytic positive controls.

### Results

- existence and structure of ambiguity band;
- unseen-finer calibration;
- equal-compute selection;
- real transfer;
- failure regimes.

### Discussion

- numerical stability versus physical accuracy;
- model-form uncertainty;
- limitations and broader contact-rich learning.

## 14. Strongest honest abstract

> Learning-based robotic grasping commonly treats the output of a physics
> simulator as ground-truth supervision. For contact-rich outcomes, however,
> the induced successful-action set can change with timestep, contact
> discretization, mesh resolution, and solver tolerance. We formulate learning
> from such approximate oracles through the inner and outer limits of action
> sets under declared numerical refinement paths. We show that finite
> single-resolution supervision cannot identify this target and that standard
> empirical risk minimization remains solver-dependent even with infinite
> data. We introduce a nested limit field that amortizes the stable-success
> core from partial RGB-D while allocating expensive refined simulations only
> near unresolved action boundaries. On a paired multi-resolution
> parallel-jaw benchmark and real close-and-lift tests, the learned core
> [must be filled only after experiments] predicts finer-oracle and physical
> success more reliably at equal simulation cost.

Square-bracket text нельзя заменять положительным claim до экспериментов.

## 15. Основные источники

- Официальные критерии ICLR 2027:
  https://iclr.cc/Conferences/2027/ReviewerGuidelines
- ICLR 2027 author guidelines:
  https://iclr.cc/Conferences/2027/AuthorGuidelines
- ICLR 2027 AI policy:
  https://iclr.cc/Conferences/2027/AIPolicyForAuthors
- Contact models in robotics, comparative analysis:
  https://arxiv.org/abs/2304.06372
- SimBenchmark:
  https://leggedrobotics.github.io/SimBenchmark/
- IPC-GraspSim:
  https://arxiv.org/abs/2111.01391
- Incremental Potential Contact:
  https://ipc-sim.github.io/
- Get a Grip:
  https://arxiv.org/abs/2410.23701
- Hard Contacts with Soft Gradients, ICLR 2026:
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/44039e59aaf6a41b16f1fc5b27bcd409-Abstract-Conference.html
- Quasi-static pushing, grasping, and jamming:
  https://arxiv.org/abs/1902.03487
- Grasp'D differentiable contact-rich grasp synthesis:
  https://arxiv.org/abs/2208.12250
- Multi-fidelity high-order Gaussian processes:
  https://proceedings.mlr.press/v130/wang21c.html
- Disentangled multi-fidelity deep Bayesian active learning:
  https://proceedings.mlr.press/v202/wu23p.html
- Discretization invariance overview:
  https://iclr-blogposts.github.io/2026/blog/2026/discretisation-invariance/
- Learning similarity metrics for numerical simulations:
  https://proceedings.mlr.press/v119/kohl20a.html
- TARGO target-oriented occluded grasping:
  https://targo-benchmark.github.io/
- ICGNet instance-centric grasping:
  https://icgraspnet.github.io/
- NeuGraspNet:
  https://openreview.net/forum?id=Fdu33eoZas
- FIRMGrasp:
  https://arxiv.org/abs/2607.25049

Полный журнал broad search, отвергнутых циклов и novelty audit находится в
reports/ICLR_GRASP_IDEA_RESEARCH.md.

## 16. Submission timing и disclosure

По состоянию на 2026-08-24 deadlines ближайшего ICLR 2027:

- abstract: 2026-09-18 AOE;
- paper: 2026-09-25 AOE.

Без уже готовой multi-resolution simulation infrastructure и real data
научно разумная цель — ICLR 2028. Rushed submission без pilot противоречит
центральному требованию proposal: сначала доказать существование phenomenon.

ICLR 2027 требует AI-use statement. Существенное использование AI в research
ideation и drafting этого proposal должно быть раскрыто точно; human authors
несут ответственность за повторную проверку literature, математических
assumptions, proofs и всех empirical claims.
