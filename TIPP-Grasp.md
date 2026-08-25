# Predict the decision, not the shape

## Task-Image Posterior Processes for reliable parallel-jaw grasp selection under occlusion

**Статус:** исследовательская гипотеза и план falsification-first проверки, 25 августа 2026 г.  
**Предлагаемое короткое имя:** **TIPP-Grasp** (*Task-Image Posterior Process for Grasping*).  
**Новый learning objective:** **DTKS** (*Decision-Topology Kernel Score*).  
**Новая архитектура:** **PCLO** (*Projectively Consistent Latent Operator*).

---

## 0. Итог в одном абзаце

Вместо восстановления скрытой формы объекта или предсказания независимого confidence для каждого grasp предлагается учить условное распределение **всей механической utility-функции на множестве grasp-кандидатов**. Полная форма (Y) во время обучения используется только как privileged source дешёвых механических labels (U_Y(g)); модель никогда не обязана выдавать mesh, occupancy или SDF. Скрытые формы, которые дают одинаковую utility-функцию для всех parallel-jaw действий, отождествляются в *task-image quotient*. Один общий латентный sample задаёт согласованный вариант utility сразу для всех кандидатов, поэтому модель видит, какие grasps становятся хорошими и плохими **совместно** при одной и той же скрытой геометрии. Новый proper objective DTKS сопоставляет finite-dimensional distributions этого stochastic process и дополнительно — не теряя строгой propriety — сопоставляет распределение top-set, signed success boundary и statewise regret. На inference выбирается grasp с малым upper-tail regret и достаточной вероятностью положительного локального grasp certificate; если такого grasp нет, разрешается abstention. Это не reconstruction pipeline, не grasp generator, не RL/VLA, не causal failure model и не оценка полного approach-to-lift цикла.

Главная проверяемая гипотеза: **при одинаковом candidate set и вычислительном бюджете joint posterior над task image даст лучший top-1 success, меньший tail regret и лучшую risk–coverage curve на visibility-equivalent, но grasp-conflicting shapes, чем deterministic scorers, independent uncertainty heads и completion-based Monte Carlo.** До эксперимента нельзя честно утверждать ни SOTA, ни принятие в ICLR.

---

## 1. Точная постановка

### 1.1. Сцена

- Один target-object на полке и один foreground occluder; общие cluttered scenes не рассматриваются.
- Одно RGB-D наблюдение (O=(I,D,K,T_{cw})) с wrist camera; depth содержит пропуски, quantization, edge noise и outliers.
- Target mask и occluder mask считаются доступными от существующего perception front-end. Их качество надо отдельно стресс-тестировать, но segmentation не является научным вкладом.
- Gripper — parallel jaw. Выход задачи — выбор одного grasp из конечного набора кандидатов (G_O=\{g_1,\ldots,g_K\}\), (g_i\in SE(3)\times\mathbb R_+), или abstention.
- Не моделируются траектория подхода, кинематика всей руки и полный цикл подъёма. До learned selector выполняется только консервативный read-only geometric filter: конечная конфигурация gripper не должна пересекать наблюдаемые obstacle/shelf points с noise dilation.

Такое ограничение намеренно отделяет научную подзадачу — **выбор по скрытой grasp-релевантной геометрии** — от motion planning и whole-cycle feasibility.

### 1.2. Privileged latent world и локальный mechanical certificate

Пусть (Y\in\mathcal Y) — полная target geometry, известная только в simulator/training data. Для grasp (g) зададим bounded signed certificate

\[
U_Y(g)\in[-1,1].
\]

Он должен измерять только состояние после локального closing parallel jaws:

1. пальцы при закрытии получают два допустимых контакта;
2. contact normals удовлетворяют frictional antipodality;
3. локальный grasp wrench space имеет положительный запас;
4. palm/finger bodies не имеют недопустимого penetration вне contact pads.

Практическая label-функция — сглаженный робастный margin

\[
U_Y(g)=\mathbb E_{\eta\sim\nu}
\big[\psi(C(Y,g\oplus\eta))\big],
\]

где (C) — signed analytic/simulation certificate, (eta) — малые pose, depth-to-world и friction perturbations, а (psi) ограничивает score. Сглаживание уменьшает разрывы при смене contact topology. Никакие SDF/mesh targets не входят в loss модели.

Важно: certificate является proxy для очень малого lift, а не обещанием предсказать успех всей траектории. Корреляцию certificate с физическим pickup надо измерить отдельно.

### 1.3. Task image вместо hidden state

Каждая полная форма индуцирует функцию

\[
T(Y)=F_Y,\qquad F_Y:g\mapsto U_Y(g).
\]

Определим task-equivalence

\[
Y\sim_TY'\quad\Longleftrightarrow\quad
U_Y(g)=U_{Y'}(g)\ \text{для почти всех }g.
\]

Нас интересует не posterior (p(Y\mid O)), а его pushforward

\[
\Pi_O=T_\#p(Y\mid O),
\]

то есть условное распределение random function (F_Y\). Это и есть **Task-Image Posterior Process (TIPP)**.

Для конечного candidate set (G_O) наблюдаемая restriction имеет вид

\[
\mathbf u_Y(G_O)=
(U_Y(g_1),\ldots,U_Y(g_K))\in[-1,1]^K.
\]

Размер learned output — (K) scalars на stochastic sample, а не (64^3) или (128^3) geometric field.

---

## 2. Почему задача не сводится к обычному per-grasp probability

Если оптимизировать только expected binary success одного действия, достаточно marginal (p(S_g=1\mid O)), и joint stochastic process был бы неоправданным усложнением. Поэтому TIPP должен проверяться на более строгой и practically relevant цели: **один и тот же grasp должен оставаться близким к лучшему при разных скрытых геометриях, совместимых с наблюдением**.

Для одной скрытой формы определим smooth oracle value и statewise regret:

\[
b_\tau(\mathbf u)=
\tau\log\sum_{j=1}^K\exp(u_j/\tau),
\qquad
r_i(\mathbf u)=b_\tau(\mathbf u)-u_i.
\]

Joint sample (mathbf u^{(s)}) отвечает одному возможному hidden world; поэтому (r_i^{(s)}) говорит, насколько (g_i) хуже лучшего доступного grasp **в том же мире**. Независимые uncertainty heads по кандидатам не задают это совместное событие корректно.

Выбор:

\[
g^*=\arg\min_{g_i\in G_O}
\operatorname{CVaR}^{\rm upper}_{\alpha}
\left[r_i(\mathbf U)\mid O\right]
\]

при ограничении

\[
\Pr[U_Y(g_i)>0\mid O]\ge 1-\delta.
\]

Если feasible set пуст, selector abstains; в forced-attempt режиме выбирается минимум CVaR без ограничения.

Почему regret полезен именно при occlusion: две скрытые формы могут иметь по одному отличному, но разному grasp. Усреднение score может предпочесть хрупкий компромисс или переуверенный mode. Upper-tail statewise regret напрямую штрафует grasp, который резко проигрывает при части plausible shapes. При этом absolute-margin constraint не позволяет назвать «надёжным» наименее плохой grasp на intrinsically ungraspable object.

Простая связь с успехом: если (B=b_\tau(\mathbf U)\), то для любого (gamma>0)

\[
\Pr(U_i\le0)
\le
\Pr(B<\gamma)+\Pr(r_i\ge\gamma).
\]

Следовательно, малый tail regret действительно ограничивает вероятность failure, если почти каждый plausible hidden world содержит grasp с запасом не меньше (gamma).

---

## 3. Новый learning objective: Decision-Topology Kernel Score

### 3.1. Зачем не BCE/MSE, Gaussian head или обычный ranking loss

- BCE по каждому grasp восстанавливает только marginals и допускает невозможные combinations между grasps.
- Diagonal Gaussian задаёт unimodal symmetric uncertainty и не представляет mutually exclusive hidden-shape modes.
- Pairwise/listwise rank loss сохраняет порядок, но не калибрует absolute success boundary и может collapse-нуть posterior.
- Diffusion over a fixed (K)-vector зависит от размера/порядка candidate set и требует многих denoising steps.
- Full likelihood в implicit stochastic-process model недоступен.

Нужен sample-based, adversarial-free, strictly proper score для joint finite restriction, который особенно чувствителен к top-set и boundary.

### 3.2. Base kernel score

Пусть модель даёт (S) joint samples

\[
\hat{\mathbf u}^{(s)}\sim P_\theta^{G}(\cdot\mid O),
\quad s=1,\ldots,S.
\]

Для characteristic kernel (k_0), например смеси IMQ/RBF kernels на нормированных margins, используем unbiased empirical kernel score

\[
\widehat{\mathcal S}_{k_0}
=
\frac{1}{S(S-1)}\sum_{s\ne t}
k_0(\hat{\mathbf u}^{(s)},\hat{\mathbf u}^{(t)})
-\frac{2}{S}\sum_s k_0(\hat{\mathbf u}^{(s)},\mathbf u).
\]

С точностью до не зависящего от модели члена это MMD squared между predictive distribution и point observation. В ожидании по ground-truth samples score строго proper: уникальный population minimizer — истинная conditional joint distribution.

### 3.3. Decision-topology map

Определим гладкую карту

\[
h_{\tau,\kappa}(\mathbf u)=
\left[
\underbrace{b_\tau(\mathbf u)-\mathbf u}_{\text{statewise regret}},
\underbrace{\operatorname{softmax}(\mathbf u/\tau)}_{\text{near-optimal set}},
\underbrace{\sigma(\mathbf u/\kappa)}_{\text{signed boundary}},
\underbrace{b_\tau(\mathbf u)}_{\text{oracle availability}}
\right].
\]

Второй characteristic kernel (k_D) применяется к pushforward distribution (h_\#P_\theta^G). Финальный objective:

\[
\boxed{
\mathcal L_{\rm DTKS}
=\mathbb E_{(O,Y),G}
\left[
\mathcal S_{k_0}(P_\theta^G,\mathbf u_Y^G)
+\lambda
\mathcal S_{k_D}
(h_\#P_\theta^G,h(\mathbf u_Y^G))
\right].}
\]

Это не произвольная сумма calibration и ranking losses:

- первая часть идентифицирует полную joint law;
- вторая часть перераспределяет finite-sample capacity к decision-relevant topology: кто near-optimal, пересекается ли zero boundary и существует ли вообще хороший кандидат;
- обе части являются proper kernel scores; наличие characteristic (k_0) делает сумму strictly proper при любом (lambda\ge0). Значит, decision emphasis не меняет population truth target, в отличие от чистого regret/ranking objective.

### 3.4. Random action sketches

На каждом SGD step берётся случайный sketch (G\subset G_O) переменного размера: uniform candidates + hard near-boundary + diverse orientations. Distribution sketches должна иметь full support над finite subsets. Это даёт:

- обучение function restrictions вместо фиксированного output tensor;
- контроль памяти;
- множество statewise comparisons из одного full-geometry label batch;
- идентификацию stochastic-process law через finite-dimensional distributions при continuity assumption.

Hard sampling не должно зависеть только от текущей модели: минимум 50% candidates выбираются независимо, иначе возможна слепая зона и нарушается аргумент идентификации.

---

## 4. Новая архитектура PCLO

### 4.1. Observation tokens без реконструкции

Входные tokens:

- видимые target points ((x,y,z,r,g,b,\sigma_D));
- foreground-obstacle points с отдельным type embedding;
- camera-ray direction, visibility/occlusion-boundary flag и distance-to-depth-discontinuity;
- shelf plane как несколько observed plane tokens, не как scene SDF.

Лёгкий Vector-Neuron или SE(3)-equivariant point encoder строит scalar/vector features. Equivariance нужна не как декоративная деталь: nuisance global pose не должна расходовать capacity, а relative gripper coordinates определяют механику.

### 4.2. Grasp-relative query operator

Для каждого (g_i):

1. видимые points переводятся в gripper frame;
2. compact support attention выбирает points в closing corridor и вокруг fingertips;
3. отдельный cross-attention читает global object token и occlusion-boundary tokens;
4. pose token содержит jaw width, approach axis, closing axis и camera-relative orientation.

Получается (q_i=Q_\theta(O,g_i)). Decoder не видит другие candidates; поэтому значение underlying function в (g_i) не меняется при добавлении/удалении (g_j).

### 4.3. Один latent world для всех grasps

Из pooled observation context условный normalizing flow получает

\[
z^{(s)}=T_\theta(\epsilon^{(s)};E_\theta(O)),
\qquad \epsilon^{(s)}\sim\mathcal N(0,I_d).
\]

Один и тот же (z^{(s)}) передаётся всем queries:

\[
\hat u_i^{(s)}=D_\theta(q_i,z^{(s)}).
\]

Таким образом, sample index (s) — это не независимый noise для score, а один coherent task-world. Нелинейный decoder позволяет одному latent mode менять целые regions grasp space совместно.

### 4.4. Projective consistency и symmetries

Для (G'\subset G) restriction samples, вычисленных с тем же (z), буквально совпадает:

\[
\hat{\mathbf u}^{(s)}(G')
=\operatorname{restrict}_{G'}
\hat{\mathbf u}^{(s)}(G).
\]

Это сильнее одной permutation equivariance: модель не переопределяет posterior при смене числа candidates. Parallel-jaw symmetry (g\equiv gR_\pi) обеспечивается weight sharing/symmetrization decoder outputs. Candidate order не влияет ни на samples, ни на decision.

### 4.5. Efficiency target

Рекомендуемые стартовые значения:

- 2–4k visible points;
- latent (d=32) или (64);
- (K=128) candidates;
- (S=8) posterior samples в online режиме, (S=32) для evaluation;
- один observation encoding, после него batched query decoding;
- target: менее 50 ms selector latency на desktop GPU и менее 1 GB дополнительной memory.

Сложность decoder (O(SKc)), DTKS training (O(S^2K)); нет voxel volume, mesh extraction, per-completion grasp generation или repeated collision checking по полным reconstructed shapes.

---

## 5. Training data и ambiguity, которую нельзя спрятать средними метриками

### 5.1. Основной corpus

[ACRONYM](https://arxiv.org/abs/2011.09584) содержит 17.7M physics-labeled parallel-jaw grasps на 8,872 objects из 262 categories и показывает пользу масштаба для grasp learning. Его meshes/grasps можно использовать как основу; новые labels должны быть пересчитаны как signed smoothed certificate, а не просто copied binary result.

Для каждого object/pose:

1. поставить один foreground occluder между камерой и target;
2. рендерить одно RGB-D с wrist-like intrinsics;
3. варьировать occlusion ratio, но гарантировать видимость target mask;
4. применять measured noise model конкретной камеры и отдельный harsher corruption split;
5. генерировать один общий candidate set аналитическим sampler без learned completion;
6. оценивать все candidates на full mesh offline.

Train/val/test splitting — по mesh identity, а основной generalization split — ещё и по category. Scene duplicates одного mesh не должны попадать в разные splits.

### 5.2. Visibility-Equivalent Grasp-Conflicting Twins (VEGCT)

Обычный random test позволяет модели угадать category/shape по видимой части и почти не проверяет posterior ambiguity. Нужен специальный stress benchmark:

- после alignment найти пары/группы shapes, чьи rendered visible depth и silhouette различаются меньше RealSense noise tolerance;
- потребовать сильное различие hidden surfaces;
- среди одинакового candidate set потребовать низкую rank correlation utility vectors или разные oracle grasps;
- скрыть mesh identity и подавать shared/noise-equivalent observation.

Это создаёт *decision-conflicting fibers*: perception evidence практически одинаково, но правильный top grasp зависит от hidden geometry. Здесь deterministic scorer обязан усреднять, independent uncertainty может знать marginals, а coherent joint posterior должен восстановить альтернативные ranking modes.

VEGCT используется как test и как небольшой controlled training subset. Отбор test twins по ground-truth disagreement надо зафиксировать до сравнения методов и публиковать весь protocol, иначе возникает selection bias.

### 5.3. Real data

Минимально убедительная hardware проверка:

- 25–40 unseen household objects, включая asymmetric hidden backs;
- 3–4 foreground occluder shapes;
- 3 occlusion bands и два depth-noise режима;
- не менее 400 randomized forced-attempt trials плюс selective trials;
- один и тот же candidate generator, robot controller и execution order для всех selectors;
- blind logging до reveal method ID;
- confidence intervals и hierarchical model/cluster bootstrap по object identity, а не treating attempts as independent.

---

## 6. Теоретические claims, которые реально можно защищать

### Claim A — task-image sufficiency

Для любой decision rule, чья conditional objective зависит от (Y) только через (U_Y(g)), posterior (p(Y\mid O)) можно заменить pushforward (Pi_O) без изменения Bayes/risk-sensitive решения. Это следует из change of variables для measurable map (T:Y\mapsto F_Y).

Более сильная минимальность должна формулироваться аккуратно: если representation позволяет вычислять ожидания всех bounded continuous functionals всех finite restrictions (F_Y(g_{1:K})), она определяет закон (Pi_O). Это minimality относительно **семейства distribution-sensitive grasp decisions**, а не относительно одного fixed expected-success action.

### Claim B — strict propriety DTKS

Kernel score с characteristic (k_0) строго proper. Kernel score после (h) proper для pushforward distribution. Неотрицательная сумма сохраняет strict propriety благодаря первой части. Поэтому в realizable infinite-data limit DTKS не жертвует calibration ради ranking.

### Claim C — posterior-to-decision regret bound

На (C(G)) используем sup norm. Map

\[
F\mapsto r_g(F)=\max_{g'\in G}F(g')-F(g)
\]

является 2-Lipschitz. Если

\[
W_1(\widehat\Pi_O,\Pi_O)\le\varepsilon
\]

и risk functional $\rho$ $L_\rho$-Lipschitz по $W_1$, то ошибка risk каждого grasp не больше $2L_\rho\varepsilon$, а excess risk действия, выбранного по $\widehat\Pi_O$, не больше $4L_\rho\varepsilon$. Для CVaR с tail mass $\alpha$ константа ухудшается как $1/\alpha$.

### Claim D — finite candidate approximation

Если (F_Y(g)) (L_g)-Lipschitz после smoothing, а candidates образуют (delta_G)-net допустимого grasp domain, добавочный oracle gap ограничен (O(L_g\delta_G)). Это отделяет ошибку selector от ошибки candidate generator.

### Что нельзя заявлять как theorem

- Что task image всегда имеет меньшую statistical complexity, чем shape posterior: deterministic pushforward удаляет nuisance information, но neural sample complexity зависит от function class и data.
- Что analytic margin гарантирует real pickup.
- Что CVaR-regret всегда лучше expected success; это preference для reliability и должна сравниваться на forced-attempt success.
- Что PCLO posterior является «истинной Bayesian uncertainty»: это learned conditional distribution, качество которой проверяется calibration и VEGCT recovery.

---

## 7. Baselines и абляции

### 7.1. Robotics baselines при общем candidate set

1. deterministic BCE/MSE scorer с тем же point encoder;
2. deep ensemble deterministic scorers;
3. independent heteroscedastic/quantile heads на каждый grasp;
4. [Contact-GraspNet](https://arxiv.org/abs/2103.14127), [GSNet/Graspness](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.html), [AnyGrasp](https://arxiv.org/abs/2212.08333), [OrbitGrasp](https://proceedings.mlr.press/v270/hu25b.html) — по возможности их scorer/ranking output;
5. [TARGO-Net](https://arxiv.org/abs/2407.06168) — наиболее прямой published baseline для target-driven grasping under occlusion; использовать его один-target/один-occluder subset и отдельно учитывать, что сам метод содержит shape-completion module;
6. [vMF-Contact](https://arxiv.org/abs/2411.03591) как uncertainty-aware contact baseline;
7. MC-dropout point completion + common analytic scorer, повторяющий принцип [Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645) и [UNCLE-Grasp](https://arxiv.org/abs/2601.14492);
8. reconstruction-aware [NeuGraspNet](https://arxiv.org/abs/2306.07392) / [ZeroGrasp](https://openaccess.thecvf.com/content/CVPR2025/html/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.html), если код и лицензия позволяют fair evaluation;
9. full-mesh oracle и visible-only oracle как верхняя/нижняя диагностические границы.

Нельзя сравнивать end-to-end numbers из чужих protocols как прямой SOTA claim. Главный controlled comparison фиксирует observations, candidates и executor, меняется только selector.

### 7.2. General-ML ablations

- (k_0) only против DTKS;
- independent latent per candidate против shared global latent;
- shared latent, но decoder видит весь candidate set, против projectively consistent decoder;
- energy score против characteristic-kernel score;
- Gaussian latent без conditional flow против flow latent;
- no equivariance / no jaw symmetry;
- no occlusion-ray tokens / no RGB / no depth-uncertainty channel;
- expected utility, lower quantile utility, CVaR utility, expected regret, CVaR regret;
- random scenes only против addition VEGCT bundles;
- (S\in\{1,4,8,16,32\}), (K\in\{32,64,128,256\}).

### 7.3. Metrics

**Primary:**

- real and simulated forced-attempt top-1 grasp success;
- upper-tail statewise regret and oracle gap;
- success–coverage/AURC при abstention;
- success на VEGCT и recovery of multimodal ranking distribution.

**Calibration:**

- per-grasp Brier/ECE только как marginal diagnostics;
- joint kernel score/MMD на held-out bundles;
- calibration of (Pr(r_i>\gamma));
- coverage of predicted near-optimal set;
- calibration by occlusion band и sensor-noise band.

**Efficiency:** latency, peak memory, training label cost, number of mesh/collision calls at inference (для TIPP должно быть zero).

### 7.4. Pre-registered success thresholds

Идея заслуживает paper-scale продолжения, если одновременно:

1. TIPP statistically improves forced-attempt success over same-backbone deterministic and independent-uncertainty models;
2. shared process materially улучшает VEGCT tail regret, а не только calibration score;
3. DTKS превосходит (k_0)-only при одинаковой model capacity;
4. gain сохраняется против хотя бы одной strong completion-based uncertainty baseline;
5. latency остаётся практически допустимой;
6. real-robot confidence interval исключает тривиальный gain.

Если improvement появляется только при abstention, только на синтетических twins или исчезает при equal-compute baseline, central claim считается опровергнутым.

---

## 8. Последовательный аудит отвергнутых направлений

### Путь 1 — stochastic full reconstruction + robust planner

**Почему выглядел разумно.** Occlusion создаёт multiple shape hypotheses; можно sampled completions прогнать через analytic grasp metric.

**Почему отвергнут.** Это уже основной сюжет [Lundell et al.](https://arxiv.org/abs/1903.00645): MC-dropout completions и joint evaluation дали статистически значимое улучшение в 90k simulated и 200 real grasps. [TARGO](https://arxiv.org/abs/2407.06168) уже создаёт специальный benchmark target-driven grasping under occlusion, а его наиболее устойчивый к росту occlusion TARGO-Net включает shape-completion module. [UNCLE-Grasp](https://arxiv.org/abs/2601.14492) в 2026 году добавляет LCB/abstention и сообщает на самом высоком simulated occlusion conditional success 0.870 против 0.780 у deterministic completed+geometry baseline. [ZeroGrasp](https://openaccess.thecvf.com/content/CVPR2025/html/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.html) и [GraspFoM](https://arxiv.org/abs/2606.08440) делают reconstruction центральной частью сильных современных систем. Это также нарушает заданное ограничение на full geometry.

### Путь 2 — uncertainty отдельных contacts/normals

**Почему выглядел разумно.** Для parallel jaws mechanical stability локальна; можно предсказывать distribution contact normals и force-closure probability.

**Почему отвергнут как core.** [PONG](https://arxiv.org/abs/2309.16930) уже выводит conservative probability-of-force-closure bound при uncertain normals. [vMF-Contact](https://arxiv.org/abs/2411.03591) уже сочетает directional von Mises–Fisher posterior, evidential objective и partial reconstruction auxiliary task и сообщает 39% improvement overall clearance rate над baselines. Marginal contact uncertainty также не моделирует mutually exclusive rankings всего candidate set.

### Путь 3 — generative distribution непосредственно над grasp poses

**Почему выглядел разумно.** Multimodal grasp set естественно моделировать VAE/diffusion/flow.

**Почему отвергнут.** Это насыщенное направление: [6-DoF GraspNet](https://openaccess.thecvf.com/content_ICCV_2019/html/Mousavian_6-DOF_GraspNet_Variational_Grasp_Generation_for_Object_Manipulation_ICCV_2019_paper.html), [Implicit Grasp Diffusion](https://proceedings.mlr.press/v270/song25b.html), [FFHFlow](https://proceedings.mlr.press/v305/feng25a.html) и [GraspGen-X](https://openaccess.thecvf.com/content/CVPR2026/html/Han_GraspGen-X_Cross-Embodiment_6-DOF_Diffusion-based_Grasping_CVPR_2026_paper.html) уже закрывают diverse/uncertainty-aware generation. Distribution good poses не обязана быть calibrated posterior того, как hidden geometry меняет quality фиксированного candidate set.

### Путь 4 — implicit hidden local occupancy

**Почему выглядел разумно.** Выводить только local occupancy около fingertip дешевле full shape.

**Почему отвергнут.** Он всё ещё предсказывает geometry, а не decision image. [TOSC](https://ojs.aaai.org/index.php/AAAI/article/view/38053) уже сужает completion до потенциальных contact regions и генерирует несколько task-oriented completions, хотя решает semantic dexterous-grasping задачу. Кроме NeuGraspNet, совсем свежий [PartialBiGrasp](https://arxiv.org/abs/2608.19188) от 19 августа 2026 уже заявляет hidden-local-geometry inference из partial views через convolutional occupancy features для bimanual force-closure grasps. Поэтому ни contact-region completion, ни parallel-jaw specialization сами по себе не образуют достаточной novelty; существенное отличие TIPP — posterior непосредственно над utility functions без геометрического output.

### Путь 5 — conformal lower bound на каждый grasp

**Почему выглядел разумно.** Даёт finite-sample reliability и естественный abstention.

**Почему не core.** [Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html) уже контролирует expected bounded monotone risk, а [Utility-Directed Conformal Prediction](https://openreview.net/forum?id=iOMnn1hSBO) уже строит prediction sets с учётом downstream utility. Их применение к grasp confidence без нового predictive object — calibration wrapper, не substantial learning architecture. CRC полезен как post-hoc add-on на held-out real calibration set, причём гарантия должна явно зависеть от exchangeability/shift assumptions.

### Путь 6 — TIPP + DTKS + PCLO

Этот путь оставлен, потому что одновременно:

- не выдаёт geometry;
- моделирует irreducible hidden-shape ambiguity, а не только epistemic weights;
- имеет новый general-ML target — posterior task image;
- требует joint stochastic-process architecture;
- даёт strictly proper, decision-topology-aware objective;
- делает falsifiable prediction именно там, где existing point scorers и completion pipelines принципиально различаются.

---

## 9. Novelty audit относительно general ML

### 9.1. Ближайшие идеи

[Bayes-Sufficient Representations in Supervised Learning](https://arxiv.org/abs/2606.04045) определяет loss-dependent Bayes quotient входов, которым соответствует один и тот же Bayes-optimal action. Это важнейшая близкая работа и её нельзя скрывать. Отличие TIPP:

- Bayes quotient там минимален для **одной fixed supervised action rule**;
- TIPP quotient расположен на latent worlds (Y), а объект предсказания — posterior distribution целой action-indexed utility function;
- TIPP сохраняет информацию для variable candidate sets, tail-risk, abstention и near-optimal sets, а не только unique Bayes action;
- DTKS и PCLO — конкретный learnable stochastic-process mechanism, отсутствующий в Bayes-sufficiency framework.

[Decision-Focused Learning through Learning to Rank](https://proceedings.mlr.press/v162/mandi22a.html) показывает, что downstream objective можно учить как ordering feasible solutions и что point/pairwise losses trade off MSE и regret. Отличие: эта линия в основном учит point cost/objective parameters, тогда как DTKS — proper conditional **distribution** над correlated utilities и statewise regret topology.

Более близкие probabilistic/robust варианты существенно сужают формулировку novelty. [End-to-End Learning for Stochastic Optimization: A Bayesian Perspective](https://proceedings.mlr.press/v202/rychener23a.html) рассматривает обучение conditional outcome distribution для последующей stochastic optimization; [DF²](https://proceedings.mlr.press/v286/kong25a.html) вместо точного forecaster напрямую учит expected optimization function; [Robust Decision-Focused Learning via Worst-Case Regret](https://proceedings.mlr.press/v337/yamao26a.html) оптимизирует worst-case regret по uncertainty/Wasserstein ambiguity sets. Поэтому TIPP не может заявлять новым «probabilistic DFL», прямое обучение objective или robust regret. Его более узкое отличие — pushforward posterior **целой action-indexed utility function**, нужен именно для correlated tail regret, variable query sets и calibrated finite-dimensional laws; DTKS сохраняет distributional truth через strict propriety, а не только минимизирует downstream regret.

[Utility-Directed Conformal Prediction](https://openreview.net/forum?id=iOMnn1hSBO) согласует prediction sets с заданной downstream utility при сохранении coverage. Это соседняя, но иная задача: utility там направляет форму outcome set; TIPP учит саму joint random utility function и может затем получить разные risk-sensitive решения. Utility-directed conformalization остаётся сильным post-hoc baseline для success–coverage, а не центральным объектом модели.

[Neural Diffusion Processes](https://proceedings.mlr.press/v202/dutordoir23a.html) моделируют distributions over functions через finite marginals и exchangeable architecture; [Continuous-Time Functional Diffusion Processes](https://proceedings.neurips.cc/paper_files/paper/2023/hash/75cd262a3fd8e76e37bb7941db141a1d-Abstract-Conference.html) формализуют diffusion в function space. Отличие: TIPP не заявляет изобретение function-space generative modeling. Novelty должна быть в task-image pushforward, exact projective query operator, DTKS и risk-sensitive regret selection.

[Probabilistic Forecasting with Generative Networks via Scoring Rule Minimization](https://www.jmlr.org/papers/v25/23-0038.html) подтверждает, что sample-based proper scoring rules могут обучать implicit generative forecasts adversarial-free и улучшать calibration. DTKS использует эту general principle, но вводит новый sum-of-proper-scores construction на finite action restrictions и их decision topology.

### 9.2. Что можно честно заявлять новым

Только после повторного literature search перед submission:

1. постановка partial-observation learning как прямого восстановления pushforward posterior (T_\#p(Y\mid O)) над task-induced utility functions без latent-state reconstruction;
2. state/task-image quotient, достаточный для семейства distribution-sensitive decisions;
3. DTKS — strictly proper joint score с дополнительным proper score на regret/top-set/boundary pushforward;
4. PCLO — observation-conditioned, symmetry-aware, projectively consistent implicit process для arbitrary grasp query sets;
5. VEGCT benchmark, который контролирует perceptual equivalence и decision conflict.

Нельзя заявлять новыми отдельно: uncertainty-aware grasping, robust regret, kernel scores, stochastic processes, equivariant point encoders, CVaR или abstention.

---

## 10. Косвенные свидетельства, что гипотеза может сработать

Это не proof of superiority, а цепочка независимых указаний.

1. **Hidden geometry действительно важна.** [NeuGraspNet](https://arxiv.org/abs/2306.07392) сообщает превосходство implicit/semi-implicit методов при single-view partial scenes; [ZeroGrasp](https://openaccess.thecvf.com/content/CVPR2025/html/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.html) достигает заявленного SOTA, совместно моделируя reconstruction и grasps. Специальный [TARGO benchmark](https://arxiv.org/abs/2407.06168) дополнительно показывает ухудшение проверенных grasp models при росте target occlusion. Значит, игнорировать invisible geometry нельзя.
2. **Distribution лучше point completion.** [Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645) показал statistically significant gains в simulation и hardware при планировании по нескольким completions, а не одной оценке.
3. **Uncertainty-aware decision даёт physical gains.** [PONG](https://arxiv.org/abs/2309.16930) связывает рост conservative force-closure probability с резким снижением failures; vMF-Contact сообщает крупный real clearance gain; UNCLE-Grasp показывает улучшение при severe occlusion.
4. **Task-focused objective может быть эффективнее reconstruction accuracy.** Decision-focused learning показывает, что errors надо оценивать по их совместному влиянию на feasible-solution ranking, а не как независимый prediction error. Это прямо поддерживает отказ от Chamfer/occupancy loss как основной цели.
5. **Joint non-Gaussian function distributions learnable.** Neural Diffusion Processes превосходят neural processes и имитируют non-Gaussian posterior behavior; это подтверждает representational feasibility, хотя PCLO использует более дешёвый implicit latent operator.
6. **Proper scoring-rule training practically viable.** JMLR 2024 scoring-rule generative forecasting сообщает более сильную calibration и меньше tuning, чем adversarial alternatives.
7. **Large-scale labels существуют.** ACRONYM показывает, что миллионы physics-labeled parallel-jaw grasps на тысячах shapes улучшают сильные grasp planners; значит, privileged task-image supervision масштабируема.

Ключевая непроверенная стрелка — что posterior task image легче и полезнее learned shape posterior при равном compute. Именно её обязан проверить controlled experiment.

---

## 11. ICLR acceptance audit

[ICLR 2027 Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers) явно включает general ML, uncertainty quantification, structured prediction и robotics. [Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines) сводят оценку к ясной проблеме, мотивации в литературе, поддержке claims, rigor и новому знанию/ценности; SOTA не обязателен.

### Потенциально сильная ICLR story

- новый объект обучения, применимый шире grasping: posterior task images under partial observation;
- формальная sufficiency и regret bounds;
- новый strictly proper objective;
- architecture с exact consistency/symmetry properties;
- ambiguity-controlled benchmark, на котором нельзя победить простым deterministic regression;
- real system как подтверждение, а не единственный вклад.

### Наиболее вероятные reviewer objections

1. **«Это просто decision-focused learning + neural process + grasp application».** Нужны theorem-level distinction, DTKS versus all components, и второй controlled non-robotics toy inverse-decision task в appendix.
2. **«Joint law не нужна для выбора одного action».** Центральный ответ — statewise tail regret/near-optimal consensus; обязательна абляция, где marginals совпадают, а joint copula различается и только TIPP выбирает правильно.
3. **«Task margin — скрытая реконструкция другим именем».** Показать dimensionality, отсутствие occupancy supervision/output и pairs разных meshes с одинаковым task image.
4. **«Synthetic posterior не calibrated в real world».** Real calibration split, hardware risk–coverage и explicit limitation under distribution shift.
5. **«Candidate generator определяет результат».** Фиксированный общий candidate set, oracle coverage metric и (delta_G)-net analysis.
6. **«Improvement — от большей модели/compute».** Equal-parameter, equal-latency и equal-number-of-samples comparisons.
7. **«Contemporary overlap».** Обязательно обсуждать Bayes-Sufficient Representations, probabilistic/robust DFL, Utility-Directed Conformal Prediction, TARGO, TOSC, GraspFoM и PartialBiGrasp; сделать ещё один search перед submission deadline.

### Текущая оценка потенциала

- **Novel question:** strong, если joint task-image posterior действительно отсутствует в concurrent work.
- **Technical depth:** potentially strong благодаря proper objective + process consistency + regret theory.
- **Broad relevance:** medium-to-strong; надо показать generic inverse-decision toy и не писать paper как robotics pipeline.
- **Empirical burden:** very high; без VEGCT, strong uncertainty baselines и hardware идея останется conceptual.
- **SOTA plausibility:** credible hypothesis на occlusion-specific top-1 selection, но не установленный факт.

---

## 12. Минимальный generic toy для general-ML claim

Чтобы работа не выглядела только новым grasp scorer, добавить synthetic inverse-decision problem:

- latent 2-D body имеет скрытую правую половину;
- observation — noisy projection левой половины;
- actions — locations/orientations двух support points;
- utility — signed static-equilibrium margin под случайной нагрузкой;
- несколько hidden bodies дают одинаковое observation, но разные optimal supports.

Сравнить full latent reconstruction, deterministic utility, independent quantiles и TIPP. Ground-truth conditional posterior известен, поэтому можно измерить exact Wasserstein/MMD, joint calibration и Bayes regret. Это physics/statics inspiration, но не второй robotics pipeline.

---

## 13. Реалистичный порядок реализации

### Phase 0 — kill test до большой модели

1. Собрать 1–2k VEGCT bundles и (K=32) candidates.
2. Обучить tiny shared-latent generator и independent quantile baseline на precomputed utility vectors без point-cloud encoder.
3. Проверить: одинаковые marginals / разные joint modes, CVaR-regret selection и DTKS gradients.

Если shared model не выигрывает в oracle-feature setting, архитектуру на RGB-D строить не следует.

### Phase 1 — controlled simulation

1. Зафиксировать certificate и проверить correlation с small-lift simulation.
2. Реализовать point encoder + projective decoder.
3. Сравнить same-backbone deterministic, independent, shared-(k_0), shared-DTKS.
4. Провести occlusion/noise/category-shift sweeps.

### Phase 2 — completion baselines и efficiency

1. MC-dropout completion + analytic ranking.
2. Доступные ZeroGrasp/NeuGraspNet-style baselines.
3. Equal-compute curves: success versus latency/memory.

### Phase 3 — robot и paper claims

1. Frozen model, held-out calibration, randomized blind trials.
2. Falsification thresholds из §7.4.
3. Отделить подтверждённые claims от hypotheses.

---

## 14. Финальная формулировка paper contribution

> We study decision learning under partial observation when the latent world is high-dimensional but affects a downstream action only through a task utility function. We propose to infer the posterior task image—the pushforward posterior over action-indexed utilities—rather than reconstruct the latent world. We introduce a projectively consistent latent operator and a strictly proper decision-topology kernel score that preserves calibration while emphasizing the posterior geometry of near-optimal actions, success boundaries, and statewise regret. Instantiated for parallel-jaw grasp selection from one occluded noisy RGB-D view, the model predicts coherent grasp-margin functions without outputting object geometry. An ambiguity-controlled benchmark and real-robot evaluation test whether this representation improves tail-regret and grasp success over deterministic, independent-uncertainty, and stochastic-completion baselines.

Эта формулировка broad, но не обещает больше, чем можно экспериментально доказать.

---

## 15. Краткая библиография первичных источников

### Robotics / grasp gap audit

- Fang et al., [GraspNet-1Billion](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html), CVPR 2020.
- Breyer et al., [Volumetric Grasping Network](https://proceedings.mlr.press/v155/breyer21a.html), CoRL 2020/PMLR 2021.
- Wang et al., [Graspness Discovery / GSNet](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.html), ICCV 2021.
- Eppner et al., [ACRONYM](https://arxiv.org/abs/2011.09584), ICRA 2021.
- Lundell et al., [Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645), ICRA 2019.
- Li et al., [PONG](https://arxiv.org/abs/2309.16930), uncertainty-aware analytic force closure.
- Jauhri et al., [NeuGraspNet](https://arxiv.org/abs/2306.07392), single-view implicit geometry and grasping.
- Ma et al., [Generalizing 6-DoF Grasp Detection via Domain Prior Knowledge](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Generalizing_6-DoF_Grasp_Detection_via_Domain_Prior_Knowledge_CVPR_2024_paper.html), CVPR 2024.
- Shi et al., [vMF-Contact](https://arxiv.org/abs/2411.03591), 2024.
- Xia et al., [TARGO: Benchmarking Target-driven Object Grasping under Occlusions](https://arxiv.org/abs/2407.06168), 2024.
- Iwase et al., [ZeroGrasp](https://openaccess.thecvf.com/content/CVPR2025/html/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.html), CVPR 2025.
- Song et al., [Implicit Grasp Diffusion](https://proceedings.mlr.press/v270/song25b.html), CoRL 2024/PMLR 2025.
- Feng et al., [FFHFlow](https://proceedings.mlr.press/v305/feng25a.html), CoRL 2025.
- Mansour et al., [UNCLE-Grasp](https://arxiv.org/abs/2601.14492), 2026.
- Wu et al., [TOSC](https://ojs.aaai.org/index.php/AAAI/article/view/38053), AAAI 2026.
- Wu et al., [GraspFoM](https://arxiv.org/abs/2606.08440), 2026.
- Kaura et al., [PartialBiGrasp](https://arxiv.org/abs/2608.19188), 19 Aug 2026.

### General ML / mathematical inspiration

- Sevetlidis, [Bayes-Sufficient Representations in Supervised Learning](https://arxiv.org/abs/2606.04045), 2026.
- Mandi et al., [Decision-Focused Learning: Through the Lens of Learning to Rank](https://proceedings.mlr.press/v162/mandi22a.html), ICML 2022.
- Rychener et al., [End-to-End Learning for Stochastic Optimization: A Bayesian Perspective](https://proceedings.mlr.press/v202/rychener23a.html), ICML 2023.
- Cortes-Gomez et al., [Utility-Directed Conformal Prediction](https://openreview.net/forum?id=iOMnn1hSBO), ICLR 2025.
- Kong et al., [DF²: Distribution-Free Decision-Focused Learning](https://proceedings.mlr.press/v286/kong25a.html), UAI 2025.
- Yamao et al., [Robust Decision-Focused Learning via Worst-Case Regret Minimization](https://proceedings.mlr.press/v337/yamao26a.html), UAI 2026.
- Garnelo et al., [Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a.html), ICML 2018.
- Dutordoir et al., [Neural Diffusion Processes](https://proceedings.mlr.press/v202/dutordoir23a.html), ICML 2023.
- Franzese et al., [Continuous-Time Functional Diffusion Processes](https://proceedings.neurips.cc/paper_files/paper/2023/hash/75cd262a3fd8e76e37bb7941db141a1d-Abstract-Conference.html), NeurIPS 2023.
- Pacchiardi et al., [Probabilistic Forecasting with Generative Networks via Scoring Rule Minimization](https://www.jmlr.org/papers/v25/23-0038.html), JMLR 2024.
- Angelopoulos et al., [Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html), ICLR 2024.
- Fuchs et al., [SE(3)-Transformers](https://proceedings.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html), NeurIPS 2020.
- Deng et al., [Vector Neurons](https://openaccess.thecvf.com/content/ICCV2021/html/Deng_Vector_Neurons_A_General_Framework_for_SO3-Equivariant_Networks_ICCV_2021_paper.html), ICCV 2021.

---

## 16. Независимость от локальных markdown-идей

Эта записка создана без чтения `reports/EdgeFlux.md`, других файлов репозитория и локальных markdown-описаний сегодняшних occlusion ideas. Поэтому она не делает ложного утверждения, что overlap с неизвестным содержимым формально проверен; она лишь соблюдает требование независимой разработки. Внешний literature novelty audit проведён по доступным первичным источникам до 25 августа 2026 года.
