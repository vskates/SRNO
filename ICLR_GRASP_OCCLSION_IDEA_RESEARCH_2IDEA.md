# Beyond Shape Completion: FiGO — Blackwell-consistent grasp-outcome fields under visual occlusion

**Research log / working proposal. Updated: 2026-08-25.**

## 0. Короткий итог

Исходная гипотеза уже имеет прямое подтверждение. Наиболее чистое найденное свидетельство — **TARGO / TARGO-Net**: это controlled benchmark target-driven grasping из одного RGB-D кадра, где видимость цели изменяется от 0 до 0.9. Авторы сообщают, что VGN, GIGA, EdgeGraspNet и ICGNet теряют примерно 20 процентных пунктов или больше при экстремальной окклюзии; TARGO-Net также ухудшается, хотя лишь примерно на 7 пунктов. В реальных экспериментах падение GIGA от easy к hard составляет примерно 40 пунктов, TARGO-Net — примерно 13. Это гораздо сильнее исходного предположения «наверняка станет хуже»: эффект уже измерен на парных сценах с тем же target и camera pose ([project page](https://targo-benchmark.github.io/), [paper](https://arxiv.org/abs/2407.06168)).

Однако gap теперь нельзя формулировать как «никто не изучал grasping under external occlusion». Это неверно. Также уже заняты следующие очевидные решения:

- completion + target/scene fusion — TARGO-Net;
- stochastic shape completion + risk aggregation/LCB — Robust Grasp Planning over Uncertain Shape Completions и свежий UNCLE-Grasp;
- diverse partial-shape hypotheses — PSSNet;
- uncertainty-aware likelihood/ranking напрямую в grasp generator — FFHFlow;
- completion только contact/task-relevant regions — TOSC;
- occupancy uncertainty, добавленная к grasp score — PUGS.

Поэтому сильная идея должна менять **не модуль**, а **объект обучения**.

Предлагаемый statistical target — не полная форма и не scalar confidence, а условное **grasp-outcome field**; в минимальной версии это семейство калиброванных marginals, а в расширенной — распределение целой функции исходов

\[
F_S:\mathcal G\to\mathbb R,\qquad g\mapsto m(S,g),
\]

индуцированное posterior’ом скрытой формы $S\mid X$. Здесь $m$ — локальный signed grasp-stability margin для parallel-jaw gripper, а $\mathcal G=(SE(3)\times W)/C_2$ учитывает диапазон ширины и симметрию перестановки губок.

Название рабочего framework: **FiGO — Filtration-consistent Grasp Outcome Processes**.

Центральная идея FiGO:

1. Учить task-sufficient pushforward posterior $\Pi_X=\mathcal L(F_S\mid X)$ напрямую, не реконструируя $S$.
2. Представлять marginals conditional distributional field; при joint-dependent utility добавлять один общий latent sample, задающий **когерентный sample всей grasp-quality landscape**. Latent neural process — подходящая реализация, но не novelty claim.
3. Использовать вложенные уровни физически корректной окклюзии как filtration и обучать posterior’ы выполнять tower property. Это measure-valued martingale constraint, реализованный как kernel conditional moment restriction, а не ошибочный pairwise KL между clean и occluded predictions.
4. Выбирать grasp по lower-tail utility (например, CVaR) и при необходимости калибровать нижнюю границу на уровне **уже выбранного** grasp’а.

После adversarial novelty search архитектурный пункт 2 понижен из core в extension: [Grasping Neural Process](http://groups.csail.mit.edu/rrg/papers/noseworthy_shaw_icra24.pdf) уже использует shared object latent для action-feasibility functions, хотя получает evidence из interaction history, а не из visual occlusion. Кроме того, [Martingale Posterior Neural Processes](https://openreview.net/forum?id=-9PVqZ-IR_) уже занимают сочетание слов *martingale + neural process*, но используют иной predictive-posterior construction. Поэтому защищаемая novelty FiGO — Blackwell/tower coherence под visual garbling и её KCM-обучение; joint latent остаётся проверяемой надстройкой.

Это не гарантированная ICLR acceptance и пока не SOTA: без экспериментов такое утверждение было бы недобросовестным. Но из рассмотренных вариантов этот имеет наиболее сильную комбинацию: потенциально новый learning constraint, общий ML-принцип, компактная математическая структура, прямое соответствие laboratory setup и проверяемые falsification criteria.

---

## 1. Точная область задачи

### 1.1 Что рассматривается

- rigid target object на полке;
- один RGB-D кадр с wrist camera;
- noisy target point cloud;
- self-occlusion присутствует всегда;
- дополнительно часть передней видимой поверхности может закрываться **одним** препятствием/объектом;
- parallel-jaw gripper;
- результат — захват и короткий подъём на несколько сантиметров;
- class label от YOLO может быть доступен, но framework обязан работать и без него;
- известен training distribution форм, но test instances могут быть unseen.

### 1.2 Что сознательно не моделируется

- RL;
- VLA;
- active view / next-best-view;
- causal failure-mode decomposition;
- вероятность успеха всей длинной цепочки approach → contact → lift → transport;
- dense SDF всей сцены;
- общий clutter reasoning.

Физическая доступность остаётся отдельным, прозрачным этапом: observed shelf/occluder geometry используется обычным collision checker для удаления явно недостижимых кандидатов. Учимая задача FiGO — **скрытая target geometry и её влияние на локальную устойчивость grasp**. Это не оценка всего motion cycle.

### 1.3 Два эффекта, которые нельзя смешивать

External occluder ухудшает grasp по двум разным причинам:

1. **Informational occlusion:** исчезают target points, posterior по форме и контактам расширяется.
2. **Physical obstruction:** часть подходов/поз gripper становится collision-infeasible.

TARGO намеренно изучает совместный эффект. Для научной чистоты новая работа должна иметь два протокола:

- **information-only:** кадр снимается через препятствие, затем препятствие убирается без движения target/camera, после чего выполняется grasp;
- **combined:** препятствие остаётся, но collision filtering не обучается и оценивается отдельно.

Именно первый протокол строго проверяет исходное предположение о вреде дополнительной потери поверхности. Второй демонстрирует применимость на полке.

---

## 2. Карта литературы и фактический gap

### 2.1 Direct grasp prediction from partial observations

Классические и современные модели уже умеют работать с single-view partial PCD:

- [GPD](https://arxiv.org/abs/1706.09911) принимает noisy, partially occluded RGB-D/PCD без CAD model;
- [S4G](https://proceedings.mlr.press/v100/qin20a.html) делает amodal single-view SE(3) grasp detection;
- [REGNet](https://arxiv.org/abs/2002.12647) прямо формулирует задачу как prediction from partial noisy observations;
- [Contact-GraspNet](https://arxiv.org/abs/2103.14127) якорит 6-DoF grasp в observed point и снижает размерность представления;
- [Graspness/GSNet](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.html) фильтрует graspable regions и даёт большой прирост на GraspNet-1Billion;
- [AnyGrasp](https://arxiv.org/abs/2212.08333) показывает 93.3% success в bin clearing и устойчивость к depth noise;
- [EdgeGraspNet](https://arxiv.org/abs/2211.00191), [OrbitGrasp](https://proceedings.mlr.press/v270/hu25b.html) и [EquiGraspFlow](https://openreview.net/forum?id=5lSkn5v4LK) используют geometric invariance/equivariance;
- [NeuGraspNet](https://sites.google.com/view/neugraspnet/home) строит implicit feature volume из одного random-view depth;
- [GraspLDM](https://arxiv.org/abs/2312.11243), [Grasp Diffusion Network](https://arxiv.org/abs/2412.08398), [Implicit Grasp Diffusion](https://proceedings.mlr.press/v270/song25b.html) и [GraspGen](https://arxiv.org/abs/2507.13097) моделируют multimodal grasp distributions.

Эти методы в основном учат (p(g\mid X)), pointwise quality (q(g,X)), либо dense grasp map. Они не представляют posterior distribution функции (g\mapsto m(S,g)), индуцированную скрытой формой.

### 2.2 Evidence: partial и externally occluded observation действительно вредят

1. **EquiGraspFlow** отдельно отмечает occlusion как limitation и показывает degradation при partial inputs в приложении, несмотря на сильную equivariance ([paper](https://openreview.net/forum?id=5lSkn5v4LK)).
2. **Generalizing 6-DoF Grasp Detection via Domain Prior Knowledge** прямо пишет, что для novel objects модель не может восстановить occluded parts только из partial point cloud, и потому использует multi-view TSDF ([CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Generalizing_6-DoF_Grasp_Detection_via_Domain_Prior_Knowledge_CVPR_2024_paper.pdf)).
3. **OK-Robot** относит значительную часть manipulation failures к single RGB-D view и ограничениям parallel-jaw gripper ([paper](https://arxiv.org/abs/2401.12202)).
4. Самое прямое evidence даёт **TARGO**: balanced test set содержит по 1000 scenes на каждый occlusion level от 0 до 0.9; performance у SOTA снижается с окклюзией. TARGO-Net снижает падение за счёт completion и target-scene reasoning, но не устраняет его ([project page](https://targo-benchmark.github.io/)).

Итог: исходная эмпирическая гипотеза подтверждена. Самостоятельная работа «измерим degradation» уже недостаточно нова, но information-only decomposition всё ещё не является центральной постановкой TARGO.

### 2.3 Shape completion и uncertainty-aware planning

- [Shape Completion Enabled Robotic Grasping](https://arxiv.org/abs/1609.08546) — ранний deterministic completion → planner.
- [Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645) включает MC-dropout shape samples, строит candidates на mean shape и усредняет analytic grasp quality по samples; uncertainty-aware planning статистически улучшает real grasp success.
- [PSSNet](https://arxiv.org/abs/2011.09390) показывает принципиальную multimodality ambiguous depth. Особенно релевантен YCB «narrow slit»: разные формы согласованы с небольшой видимой полосой; diverse completions приводят к удачному side grasp там, где single-mode completion ломается.
- [Shape Completion with Prediction of Uncertain Regions](https://arxiv.org/abs/2308.00377) показывает, что избегание predicted uncertain regions улучшает grasp quality.
- [Measuring Uncertainty in Shape Completion to Improve Grasp Quality](https://arxiv.org/abs/2504.16183) добавляет completion uncertainty в ранжирование grasp’ов и улучшает rank-5 real success.
- [PCF-Grasp](https://arxiv.org/abs/2504.16320) использует completion как geometry feature и сообщает +17.8% real success относительно выбранного SOTA.
- [TOSC](https://ojs.aaai.org/index.php/AAAI/article/view/38053) завершает не всю форму, а task/contact-relevant regions для open-world task-oriented dexterous grasping.
- [UNCLE-Grasp](https://arxiv.org/abs/2601.14492) применяет MC-dropout completions, force-closure filters и lower confidence bound под leaf occlusion; основной action при высокой ambiguity — abstention.

Следовательно, «probabilistic completion + robust score» не является новой идеей. Более того, она нарушает исходное требование эффективности: нужно генерировать несколько полных геометрий, планировать/проверять grasps на каждой и надеяться, что completion posterior калиброван именно в task-relevant regions.

### 2.4 Direct uncertainty in grasp models

[FFHFlow](https://proceedings.mlr.press/v305/feng25a.html) — важнейший сосед. Он учит flow-based latent grasp distribution для dexterous hand на partial PCD и использует exact flow likelihood вместе с discriminative evaluator. Но его view uncertainty — типичность generated grasp под (p(g\mid X)), а не posterior distribution физического margin (p(m\mid X,g)). Высокая плотность хороших training grasps не равна высокой вероятности success across shapes, совместимых с observation. Модель также не задаёт joint outcome correlations между разными grasp queries и не требует Bayesian consistency между уровнями окклюзии.

[PUGS](https://arxiv.org/abs/2502.09824) моделирует multi-view reconstruction/pose uncertainty в underwater setting и делит pre-trained grasp confidence на occupancy variance. Это полезное доказательство того, что uncertainty-aware selection помогает, но не posterior hidden-shape outcomes из single view.

**Критический близкий прецедент:** Noseworthy et al., [Amortized Inference for Efficient Grasp Model Adaptation, ICRA 2024](http://groups.csail.mit.edu/rrg/papers/noseworthy_shaw_icra24.pdf), уже предлагают **Grasping Neural Process (GNP)**. Один object-level latent sample задаёт action-feasibility classifier; posterior обновляется по набору совершённых действий и их success/failure, чтобы адаптироваться к скрытым mass/friction properties. Следовательно, «neural process для распределения grasp-feasibility functions» не нов. Отличие FiGO должно быть доказано не названием backbone’а, а observation regime и learning law: zero-interaction inference из coarsened RGB-D/PCD, hidden geometry вместо динамических свойств, nested visual signals и условная mixture/tower consistency.

Ещё один обязательный сосед — [Martingale Posterior Neural Processes, ICLR 2023](https://openreview.net/forum?id=-9PVqZ-IR_). Там *martingale posterior* — альтернативный способ породить functional uncertainty через conditionally identically distributed predictive continuation. В FiGO используется другой martingale: обычный Doob posterior process по вложенным $\sigma$-алгебрам наблюдаемой геометрии. Совпадение терминов опасно; в статье эти конструкции надо развести определениями и теоремой, а метод не называть «martingale neural process».

### 2.5 Прямой конкурент по external occlusion

[TARGO-Net](https://targo-benchmark.github.io/) сегментирует visible target, завершает target point cloud, объединяет completed target и full scene через transformer и предсказывает deterministic grasp quality/orientation/width. Его strong result означает:

- нельзя заявлять novelty для occlusion benchmark вообще;
- нельзя заявлять novelty для completion + scene attention;
- TARGO должен быть обязательным baseline и source dataset;
- новая работа должна показать преимущество именно при hidden-shape ambiguity, calibration и unseen occluders/classes, а не только общий top-1 success.

### 2.6 Реальный gap после жёсткой фильтрации

На дату этого обзора в просмотренных публично индексируемых источниках не найдено работы, которая одновременно:

1. формулирует hidden geometry только через task pushforward $S\mapsto F_S$, без shape reconstruction (сам function-posterior precedent существует в GNP, но не из visual coarsening);
2. учит **joint posterior over a grasp-outcome function** на continuous grasp manifold;
3. использует один shared latent sample для когерентных counterfactual outcomes всех query grasps;
4. требует measure-valued martingale/tower consistency между физически вложенными occlusion observations;
5. оценивает отдельно informational и physical effects external occlusion;
6. применяет это к single-view parallel-jaw grasp selection без RL/VLA.

Это рабочая, **не исчерпывающая** novelty claim, а не доказательство отсутствия unpublished/неиндексированной работы. Она должна звучать именно так; более широкие claims будут ложными.

---

## 3. Итерации идей и причины отбраковки

### Вариант A: occlusion augmentation + consistency encoder

**Идея.** Рендерить препятствие, удалять target points, заставлять clean и occluded encodings/grasp predictions совпадать.

**Почему привлекательно.** Дёшево; TARGO показывает пользу occlusion-induced negative grasps.

**Почему отвергнуто.** Это модификация существующего detector, а не новая постановка. Хуже того, pointwise clean/occluded consistency математически неверна: тяжёлое observation объективно должно иметь более широкое posterior. Принудительное совпадение стирает uncertainty и создаёт overconfidence. TARGO уже занимает практическую часть этого направления.

### Вариант B: probabilistic full completion + CVaR/LCB grasp selection

**Идея.** Сэмплировать meshes/PCDs, оценивать grasp на каждой, выбирать lower-tail best.

**Почему привлекательно.** Прямо соответствует Bayes decision theory; есть evidence, что uncertainty-aware planning лучше single completion.

**Почему отвергнуто.** Уже покрыто работами 2019–2026, особенно UNCLE-Grasp. Вычислительно тяжело, task-misaligned reconstruction loss, качество Chamfer не гарантирует корректных contacts/normals. Это именно нежелательный pipeline composition.

### Вариант C: worst-case certified grasp set по всем совместимым формам

**Идея.** (\underline m(X,g)=\inf_{S\in\mathcal C(X)}m(S,g)), выбирать grasp с наибольшей гарантией. При добавлении evidence set (\mathcal C(X)) сужается, поэтому certificate monotone.

**Почему привлекательно.** Сильная теорема и безопасная интерпретация.

**Почему отвергнуто как основной путь.** Без очень сильного shape prior adversary может добавить произвольную скрытую деталь и сделать почти любой grasp невалидным; safe set становится пустым. Сильный prior превращает метод обратно в completion/shape-family modeling. Полезен как limiting baseline, но не как practical SOTA framework.

### Вариант D: независимая conditional distribution $p(m\mid X,g)$

**Идея.** Вместо completion учить quantiles/flow grasp margin для каждого candidate.

**Почему привлекательно.** Task-direct, быстро, легко оптимизировать CVaR.

**Пересмотр после формального аудита.** Для one-shot utility $u(F(g))$, включая marginal CVaR, эти marginals статистически достаточны; joint correlations не могут улучшить идеальный Bayes selector для такой utility. Независимая модель всё ещё подвержена estimation/selection bias, но shared latent не устраняет его автоматически. Поэтому этот вариант не отбрасывается: **marginal distributional critic + Blackwell/KCM law становится минимальным core FiGO и сильнейшим обязательным baseline**. Без KCM он действительно выглядит как incremental uncertainty head.

### Вариант E (опциональное расширение): joint grasp-outcome process + occlusion filtration

Shared latent $z$ задаёт целую функцию $g\mapsto m_z(g)$. Это сохраняет correlations между candidates. Nested physical occlusions дают не обычную augmentation invariance, а закон условных posterior’ов. Он реализуется через conditional moment restriction и не требует full shape output.

Joint часть сохраняется условно, потому что она:

- отличается от TARGO/UNCLE/FFHFlow observation-to-output постановкой;
- отличается от GNP источником evidence (vision вместо interaction history) и от MPNP смыслом martingale;
- не требует dense scene representation;
- даёт общий ML contribution про learning posteriors under observation coarsening;
- имеет короткие и проверяемые теоретические свойства;
- допускает amortized, batched inference.

Её надо удалить, если она не помогает utility, действительно зависящей от joint law: tail regret, batch diversity или probability-at-least-one. Главный выбранный метод после аудита — **filtration-consistent outcome field**, который может быть marginal или joint; главный экспериментальный вопрос — полезность KCM, а не shared latent сам по себе.

---

## 4. Формальная постановка FiGO

### 4.1 Latent world и observation fiber

Пусть (S\sim P_S) — полная rigid surface target object вместе с фиксированными для эксперимента contact parameters. При известном class (c) используется (P_S(\cdot\mid c)); при неизвестном — unconditional/mixed prior. Classifier-free conditioning во время training позволяет одному model работать в обоих режимах.

Камера (v), single occluder (B), sensor noise (\epsilon) задают оператор видимости

\[
X=\mathcal V(S;v,B,\epsilon).
\]

Практический input не является SDF сцены:

\[
X=(P_{\mathrm{target}},R_{\mathrm{occ}},v[,c]),
\]

где $P_{\mathrm{target}}$ — segmented target PCD, а $R_{\mathrm{occ}}$ — опциональный небольшой subsample лучей первого возврата от obstacle около видимого контура target. Эти лучи оцениваются только из доступных depth/segmentation и **не требуют oracle amodal silhouette**; версия без них обязательна как baseline. Геометрия shelf/obstacle отдельно поступает в deterministic collision checker. Ray tokens могут различать missingness mechanisms при почти одинаковых target points, не превращая input в full-scene SDF.

Observation fiber

\[
\mathcal S_X=\{S:\mathcal V(S;v,B,\epsilon)\approx X\}
\]

содержит множество несовместимых скрытых форм. Point completion выбирает/сэмплирует элементы этого fiber. FiGO сразу отображает fiber в task space.

### 4.2 Grasp space

Parallel-jaw grasp:

\[
g=(R,t,w)\in SE(3)\times [w_{\min},w_{\max}].
\]

Перестановка губок создаёт физически тот же grasp, поэтому domain лучше считать quotient space

\[
\mathcal G=(SE(3)\times W)/C_2.
\]

Эта симметрия реализуется либо canonicalization, либо $C_2$-symmetrization decoder’а.

### 4.3 Локальный outcome, а не whole-cycle feasibility

Для complete (S) и grasp (g) вычисляется signed robust margin

\[
m(S,g)=\inf_{\delta\in\Delta_{\text{calib}}}
\Big[\mu_0-\mu_{\min}(S,g\oplus\delta)\Big],
\]

с отрицательным значением для jaw/object penetration или отсутствия допустимой contact pair. (\mu_{\min}) — минимальный coefficient of friction, при котором contact pair antipodal/force-closure; (\Delta_{\text{calib}}) — малое множество pose perturbations, соответствующее hand-eye и механической погрешности. Альтернатива — нормированный Ferrari–Canny (\epsilon)-margin. Оба label’а вычисляются по **полной target mesh**, но model её не предсказывает.

Такой label:

- локален к gripper interaction volume;
- непрерывен почти всюду и информативнее binary success;
- не включает motion planning и длинный lift cycle;
- позволяет binary success (Y=\mathbf 1[m>0]) для evaluation.

### 4.4 Task-induced random function

Каждая полная форма задаёт функцию

\[
F_S(g)=m(S,g),\qquad F_S\in\mathbb R^{\mathcal G}.
\]

Partial observation индуцирует posterior stochastic process

\[
\Pi_X=\mathcal L(F_S\mid X).
\]

Для конечного query basket (G=(g_1,\ldots,g_K)) модель возвращает joint distribution

\[
\Pi_X^G=\mathcal L\big(F_S(g_1),\ldots,F_S(g_K)\mid X\big).
\]

Именно конечномерные distributions нужны для обучения и выбора; по Kolmogorov consistency они определяют process при согласованной модели.

### 4.5 Proposition 1: task sufficiency — и её важное ограничение

Пусть любая downstream grasp decision использует (S) только через значения (F_S(g)). Тогда для любого bounded utility (u) и grasp (g)

\[
\mathbb E[u(F_S(g))\mid X]
=\int u(f(g))\,d\Pi_X(f).
\]

Следовательно, $\Pi_X$ достаточен для любой Bayes-optimal grasp decision в этом классе; posterior $P(S\mid X)$ содержит task-irrelevant information. Два shape posterior’а, имеющие одинаковый pushforward $\Pi_X$, неразличимы для всех таких grasp decisions.

Но для **одного** grasp и utility вида $u(F(g))$ достаточно даже набора одномерных marginals $\{\mathcal L(F(g)\mid X)\}_g$: correlations между candidates не меняют marginal CVaR. Полный joint process нужен лишь если loss сравнивает candidates в одном latent world (например, tail regret), выбирается batch/diverse set или планируется последующее observation/action. Это ограничение запрещает утверждать, что shared latent сам по себе улучшает обычный CVaR selector. В основных экспериментах должны быть оба режима: marginal-CVaR как простой и честный основной selector; joint-regret как отдельный тест ценности process structure.

Это не доказывает автоматически меньшую sample complexity, но строго объясняет, почему full reconstruction не является необходимой промежуточной задачей.

### 4.6 Occlusion filtration

Строим physically nested observations одного latent object:

\[
X_0\preceq X_1\preceq\cdots\preceq X_L,
\]

где $X_0$ наиболее закрыт, $X_L$ — unoccluded single-view observation (его backside всё ещё self-occluded). Формально coarse view должен быть **measurable garbling** fine view: shared rays используют один и тот же coupled sensor-noise draw, mask/occluder descriptor входит в observation, а $X_\ell=C_\ell(X_{\ell+1},U_\ell)$ для известного/сэмплированного garbling kernel. Тогда можно определить $\mathcal F_\ell=\sigma(X_0,\ldots,X_\ell)$ и получить $\mathcal F_\ell\subseteq\mathcal F_{\ell+1}$. Просто независимо перерендеренные noisy кадры этого свойства не имеют и не годятся для теоремы без расширения probability space.

Posterior measure

\[
\Pi_\ell(A)=P(F_S\in A\mid\mathcal F_\ell)
\]

является **measure-valued martingale**: для любого measurable (A)

\[
\mathbb E[\Pi_{\ell+1}(A)\mid\mathcal F_\ell]=\Pi_\ell(A).
\]

Эквивалентно, для любого bounded test functional (\phi)

\[
M_\ell^\phi=\int\phi(f)\,d\Pi_\ell(f)
\]

выполняет

\[
\mathbb E[M_{\ell+1}^\phi\mid\mathcal F_\ell]=M_\ell^\phi.
\]

Для mean margin это обычная tower property. Для variance:

\[
\mathrm{Var}(F(g)\mid\mathcal F_\ell)=
\mathbb E[\mathrm{Var}(F(g)\mid\mathcal F_{\ell+1})\mid\mathcal F_\ell]
+\mathrm{Var}(\mathbb E[F(g)\mid\mathcal F_{\ell+1}]\mid\mathcal F_\ell).
\]

Поэтому uncertainty с большей видимостью уменьшается **в среднем**, но не обязана pointwise монотонно уменьшаться для каждого конкретного кадра. Это важная защита от ложной regularization.

Связь с general ML не выдумана: martingale property уже использована как необходимое условие coherent Bayesian prediction в [Falck et al., ICML 2024](https://proceedings.mlr.press/v235/falck24a.html). Blackwell order также интерпретирует coarse observation как garbling более информативного experiment; posterior coarse signal является conditional mean fine posterior. Это классическая база, не собственная теорема FiGO. Потенциально новая часть — превратить этот закон в обучаемое conditional-moment ограничение для task-induced grasp field под контролируемой visual coarsening.

### 4.7 Почему clean/occluded KL неверен

Нельзя минимизировать

\[
D_{KL}(\Pi_{X_0}\|\Pi_{X_L})
\]

для пар одного latent object. (X_L) раскрывает именно истинную форму этого sample и его posterior уже; (X_0) должен быть смесью posterior’ов всех fine observations, совместимых с coarse observation:

\[
\Pi_{X_0}=\int \Pi_{X_L}\,dP(X_L\mid X_0).
\]

Pairwise matching заменяет условную смесь одной компонентой и искусственно схлопывает ambiguity.

---

## 5. Архитектура

### 5.1 SE(3)-equivariant sparse observation encoder

Encoder получает target points и небольшой набор occlusion-ray tokens. Он выдаёт:

- global invariant context (h_X);
- equivariant per-point features (h_i);
- локальные признаки в gripper-aligned crop для каждого query (g).

Для query grasp точки преобразуются в gripper frame:

\[
u_i=R_g^\top(x_i-t_g).
\]

Cross-attention выполняется только по points/rays около closing и contact volume. Это не dense SDF и не вся shelf scene.

Желаемая equivariance для любого конечного query basket $G$:

\[
\mathcal L\!\left((F_{HS}(Hg))_{g\in G}\mid HX\right)
=\mathcal L\!\left((F_S(g))_{g\in G}\mid X\right),\qquad H\in SE(3).
\]

OrbitGrasp и EquiGraspFlow дают косвенное evidence, что explicit equivariance улучшает data efficiency и spatial generalization; здесь она является не novelty claim сама по себе, а необходимой структурой query field.

### 5.2 Conditional latent grasp-outcome field (NP implementation)

Conditional prior:

\[
z\sim p_\theta(z\mid h_X[,c]),
\]

где (p_\theta) — компактный conditional normalizing flow или mixture prior. Один sampled (z) используется для **всех** (g_j\in G):

\[
\hat m_j=d_\psi(h_X,h_{G_j},z,g_j)+\eta_j.
\]

Это conditional latent field над $\mathcal G$; latent neural process является одной реализацией. Shared $z$ кодирует глобальную hidden-shape hypothesis только в task space. Модель не обязана уметь вывести mesh, backside points или SDF.

[Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a.html) показывают, что function distributions можно моделировать с permutation-invariant context и (O(n+m)) inference по числу context/target points. [Rényi Neural Processes](https://openreview.net/forum?id=qMt4KikFJg) дают современную robust variational alternative при misspecified conditional prior. Эти работы — general-ML inspiration, не robotics pipeline source.

GNP уже реализует object-level latent action-feasibility function в grasping. Поэтому эта секция — **implementation hypothesis**, не самостоятельный contribution. Если KCM/visual-filtration часть удалить, архитектурная novelty недостаточна.

### 5.3 Training posterior и ELBO

Во время training доступен query basket с ground-truth margins

\[
D_G=\{(g_j,m(S,g_j))\}_{j=1}^K.
\]

Permutation-invariant posterior encoder (q_\varphi(z\mid X,D_G)) используется только при обучении:

\[
\mathcal L_{NP}=
-\mathbb E_{q_\varphi}\sum_{j=1}^K\log p_\psi(m_j\mid X,g_j,z)
+\beta D\big(q_\varphi(z\mid X,D_G)\|p_\theta(z\mid X)\big).
\]

Можно начать с KL; если conditional-prior misspecification заметна, заменить на Rényi divergence. Не следует добавлять diffusion только ради модности: latent flow достаточен, быстрее и даёт tractable sampling.

### 5.4 Filtration consistency как conditional moment restriction

Для nested pair $X^-\preceq X^+$ и общего query basket $G$ берём characteristic feature map $\phi(F_G)$ (например, random Fourier features Gaussian kernel на joint margin vector). Важно: $G$ должен быть фиксирован, сэмплирован независимо или зависеть только от $X^-$; если candidates строятся из дополнительной информации $X^+$, tower equation ниже уже не следует. Модельные conditional mean embeddings:

\[
\mu_\theta(X,G)=
\mathbb E_{F_G\sim\hat\Pi_\theta(\cdot\mid X)}[\phi(F_G)].
\]

Tower property требует

\[
\mathbb E[\mu_\theta(X^+,G)-\mu_\theta(X^-,G)\mid X^-]=0.
\]

Пусть (r_i=\mu_\theta(X_i^+,G_i)-\mu_\theta(X_i^-,G_i)), (k_X) — universal kernel на coarse observation embeddings. Практический maximum-moment penalty:

\[
\mathcal L_{KCM}=
\frac1{B^2}\sum_{i,j=1}^B
k_X(e_i^-,e_j^-)\langle r_i,r_j\rangle.
\]

При characteristic outcome kernel, universal observation kernel, корректном sampling и достаточном query support нулевой population loss идентифицирует нужное conditional distribution relation для рассматриваемых finite-dimensional marginals. Конечный набор baskets сам по себе не доказывает равенство бесконечномерных процессов. [Kernel Conditional Moment Test](https://proceedings.mlr.press/v124/muandet20a.html) показывает, как RKHS maximum moment restriction характеризует conditional moment restriction и даёт вычислимый statistic. [Functional GEL](https://proceedings.mlr.press/v162/kremer22a.html) даёт практические kernel/neural реализации и consistency results для своего setting; перенос их guarantees на FiGO потребует отдельного доказательства assumptions.

Для scale используется random-feature approximation: (O(BD_\phi)), а не явная (O(B^2)) Gram matrix.

### 5.5 Полный objective

\[
\mathcal L=
\mathcal L_{NP}
+\lambda_{mart}\mathcal L_{KCM}
+\lambda_{eq}\mathcal L_{SE(3)/C_2}
+\lambda_{cal}\mathcal L_{margin-cal}.
\]

Последний term — proper scoring rule (CRPS или energy score) для calibration finite-dimensional distributions. Никакого reconstruction loss нет.

### 5.6 Risk-sensitive inference: marginal core и joint extension

Для каждого candidate (g) сэмплируем (M) shared latents (z^{(1:M)}) и получаем margins (m^{(1:M)}(g)). Основной score:

\[
R_\alpha(g\mid X)=\operatorname{CVaR}_{\alpha}
\big(F_S(g)\mid X\big),
\]

то есть среднее худших (\alpha)-долей posterior outcomes. Затем

\[
g^*=\arg\max_{g\in\mathcal C(X)}R_\alpha(g\mid X),
\]

где (\mathcal C(X)) уже отфильтрован observed obstacle/shelf collision checker’ом.

CVaR — design choice, не novelty. Ablation обязана сравнить mean, lower quantile, worst sample и CVaR. Этот selector использует только marginals и поэтому не является evidence пользы joint process.

Чтобы проверить именно joint structure, вводится отдельный posterior-regret score на **фиксированном общем** candidate pool $\mathcal C$:

\[
\rho_\alpha(g\mid X)=
\operatorname{CVaR}^{\mathrm{upper}}_{1-\alpha}
\left(\max_{h\in\mathcal C}F_S(h)-F_S(g)\mid X\right).
\]

Joint selector минимизирует $\rho_\alpha$ при ограничении на lower-tail absolute margin. В отличие от marginal CVaR, regret требует outcomes всех candidates под одним sample скрытого мира. Это не гарантированно лучший практический objective; его задача — честно установить, несут ли learned correlations decision value.

### 5.7 Candidate generation

FiGO лучше позиционировать как **proposal-agnostic posterior evaluator/selector**, потому что это чистая подзадача и позволяет честное сравнение при одинаковом candidate recall. Основные эксперименты используют candidates от двух сильных, существенно разных generators (например, GraspGen и TARGO/Contact-GraspNet-style), плюс oracle candidate pool с full mesh.

Опционально differentiable decoder позволяет 3–5 Riemannian gradient steps на (SE(3)) для refinement (R_\alpha). Это расширение, а не центральный contribution.

### 5.8 Calibration после selection

Обычная per-candidate calibration недостаточна: максимум из сотен candidates усиливает ошибки. На held-out calibration scenes запускается **полный фиксированный selector**, после чего conformal residual корректирует lower bound уже для selected grasp. При exchangeability это даёт marginal coverage для процедуры selection; не следует обещать conditional или OOD guarantee.

[Decision-Theoretic Foundations for Conformal Prediction](https://proceedings.mlr.press/v267/kiyani25a.html) даёт сильное косвенное основание: prediction sets естественно связаны с Value-at-Risk, а max-min policy оптимальна для описанного класса risk-averse agents. Calibration остаётся secondary component; без него core FiGO всё ещё полноценен.

---

## 6. Данные и обучение

### 6.1 Источники форм и grasp labels

[ACRONYM](https://research.nvidia.com/publication/2021-05_acronym-large-scale-grasp-dataset-based-simulation) содержит 17.7M parallel-jaw grasps для 8,872 объектов из 262 категорий. Это достаточный scale и прямое evidence, что large simulated grasp supervision улучшает planners.

[TARGO](https://targo-benchmark.github.io/) уже предоставляет depth, target masks, TSDF, target points, meshes/poses, grasp labels и controlled occlusion levels; его необходимо использовать для внешней сопоставимости.

Предлагаемый train corpus:

- ACRONYM meshes/labels для широкого shape distribution;
- TARGO-Synthetic для physically realistic target/occluder relations;
- held-out Google Scanned Objects/YCB-like real meshes только для test;
- собственные shelf renders с **одним** occluder, чтобы не смешивать постановку с clutter.

### 6.2 Nested occlusion generation

Для каждой (S,v):

1. рендерится clean single-view depth (X_L);
2. перед target перемещается один box/cylinder/held-out household occluder;
3. маски строятся вложенно по target pixel coverage;
4. levels: 0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90;
5. добавляются realistic depth quantization, edge flying pixels, holes и hand-eye perturbation;
6. сохраняется связь coarse/fine observations и точный visibility ratio.

Random point dropout — отдельный weak baseline, не основной augmentation: он не воспроизводит missing-not-at-random camera visibility.

### 6.3 Query baskets

Каждый basket должен содержать correlated counterfactual outcomes на одной complete shape:

- 35% known successful full-mesh grasps;
- 25% perturbations около decision boundary;
- 25% hard negatives, которые выглядят хорошими на partial PCD, но плохи на full mesh;
- 15% broad proposals для coverage.

Особенно важны pairs (g_a,g_b), где две hidden-shape hypotheses меняют их ranking местами. Без таких baskets joint process может выродиться в независимые marginal scores.

### 6.4 Class-known и class-free режимы

Во время training class token случайно удаляется с вероятностью 0.5. Evaluation содержит:

- same class / unseen instance;
- unseen class;
- correct class token;
- missing class token;
- deliberately corrupted class token как stress test.

Нельзя заявлять robustness, если выигрыш существует только при oracle class.

---

## 7. Экспериментальный протокол

### 7.1 Controlled simulation

Фиксируются object, pose и camera; меняется только occluder. Для каждой сцены выполняются два режима:

1. **Perception-only:** grasp quality и короткий lift считаются на target без физического occluder после capture.
2. **Perception+access:** occluder остаётся; простой collision checker удаляет candidates, но не меняет FiGO score.

Splits:

- unseen instances;
- unseen categories;
- unseen occluder geometry/material;
- train levels (0,0.3,0.6,0.9), test intermediate levels;
- sensor-noise shift;
- candidate-generator transfer.

### 7.2 Real shelf experiment

Минимально убедительный дизайн:

- 20–25 unseen household objects;
- один и тот же target pose удерживается скрытым fixture;
- occluder на линейной направляющей создаёт 4–5 visibility levels;
- 5 repeats на condition;
- wrist RGB-D camera;
- parallel-jaw gripper;
- success: target поднят и удерживается на 2–3 cm заданное короткое время;
- отдельно записываются perception failure, collision-filter rejection и contact/lift failure, но модель не обучается на causal taxonomy.

Порядок условий рандомизируется, friction-sensitive objects балансируются по методам.

### 7.3 Метрики

Главные:

- top-1 physical success;
- success-vs-occlusion curve;
- **AUOSC** — area under occlusion-success curve;
- worst-severity success (≥0.75 occlusion);
- oracle regret относительно лучшего grasp в общем candidate pool;
- posterior tail-regret и realized oracle regret для joint selector;
- risk–coverage curve, если разрешено abstention; при обязательном grasp — не использовать abstention для завышения success.

Uncertainty/process quality:

- NLL/CRPS marginal margin;
- energy score и variogram score joint basket distributions;
- selected-grasp calibration/coverage;
- KCM filtration violation;
- mean predicted variance vs visibility (только aggregate, не требовать pointwise monotonicity);
- вероятность «хотя бы один из top-k работает», где joint correlations действительно важны.

Efficiency:

- end-to-end latency;
- peak memory;
- число full-geometry queries (у FiGO ноль);
- scaling по (K) grasp queries и (M) latent samples.

### 7.4 Обязательные baselines

1. TARGO-Net.
2. GraspGen или актуальный сильный diffusion generator/evaluator.
3. Contact-GraspNet/AnyGrasp-style direct detector.
4. Equivariant direct model (OrbitGrasp или EquiGraspFlow adaptation).
5. Deterministic completion + planner.
6. Probabilistic completion + mean/LCB/CVaR planning (UNCLE-style generalized beyond strawberries).
7. Pointwise Bernoulli success critic.
8. Independent quantile/flow (p(m\mid X,g)).
9. FiGO without (\mathcal L_{KCM}).
10. FiGO with **wrong pairwise KL consistency**.
11. Full-PCD oracle и best-in-candidate oracle.

FFHFlow — conceptual comparison, но не прямой apples-to-apples baseline из-за dexterous hand. Нужен FFHFlow-style likelihood ranking на том же parallel-jaw generator.

### 7.5 Критические ablations

- shared global latent vs independent per-grasp noise;
- scalar mean vs marginal distribution vs joint process;
- marginal-CVaR vs joint tail-regret при одинаковых marginals и candidate pool;
- no martingale vs KCM vs pairwise KL;
- physical ray occlusion vs random point dropout;
- with/without occlusion-ray tokens;
- class-free vs class-conditioned;
- equivariant vs augmentation-only encoder;
- (M\in\{1,4,8,16,32\});
- different risk function (E,q_{0.1},CVaR_{0.1},\min);
- analytic margin labels vs short-lift simulation labels;
- fixed candidate pool vs two proposal generators.

---

## 8. Почему подход может сработать: только косвенные evidence, без завышения claims

1. **Uncertainty matters.** Robust planning по multiple completions улучшает real grasp success относительно point completion; UNCLE-Grasp и PUGS также находят benefit от uncertainty-aware selection.
2. **Ambiguity реально multimodal.** PSSNet показывает разные plausible hidden handles/shapes при одинаковом узком depth view; single averaged completion плох.
3. **External occlusion действительно создаёт performance gradient.** TARGO даёт controlled evidence по levels 0–0.9.
4. **Direct discriminative evaluation масштабируется.** Contact-GraspNet/GSNet/GraspGen и Get-a-Grip показывают силу discriminative evaluators для отбора generated candidates; FiGO расширяет evaluator до function posterior.
5. **Equivariance полезна.** OrbitGrasp и EquiGraspFlow показывают сильную data efficiency/generalization по rotations.
6. **Function distributions learnable amortized.** Neural Processes моделируют stochastic functions и дают линейное по context/query inference.
7. **Martingale coherence измерима.** ICML 2024 martingale perspective даёт тестируемый критерий Bayesian coherence; KCM/MMR даёт consistent estimation tool для conditional moment restrictions.
8. **Task-space inference может быть дешевле full posterior sampling.** Здесь это гипотеза, а не доказанный факт: один encoder pass + batched latent/query decoder должен сравниваться с multiple completions + repeated grasp evaluation. Нельзя писать «efficient» до latency measurements.

---

## 9. Adversarial novelty audit

### 9.1 Отличие от TARGO

TARGO: deterministic completed target + scene fusion + scalar grasp prediction.

FiGO: no completion; posterior random function of target-grasp margin; joint candidate correlations; filtration law; physical access отделён. Benchmark TARGO можно использовать, но method не является его модификацией.

### 9.2 Отличие от UNCLE-Grasp / 2019 robust completion planning

Они сначала сэмплируют complete geometry, затем propagates uncertainty к grasps. FiGO учит pushforward posterior напрямую. UNCLE использует LCB и abstention domain-specific strawberry pipeline; FiGO использует mandatory-selection-compatible joint process и calibration.

### 9.3 Отличие от FFHFlow

FFHFlow моделирует (p(g\mid X)) и использует grasp likelihood как uncertainty proxy. FiGO моделирует (\mathcal L(m(S,\cdot)\mid X)). «Частый хороший grasp в training data» и «grasp с хорошим worst-tail outcome по всем plausible hidden shapes» — разные quantities.

### 9.4 Отличие от Neural Processes

Latent NP — базовый model family, не новая сама по себе. Более того, GNP уже применяет его к grasp feasibility. Заявляемые элементы остаются только в связке: zero-interaction visual-coarsening posterior, task pushforward скрытой геометрии, Blackwell/Doob consistency под occlusion filtration и её KCM training objective. Grasp quotient manifold и equivariance полезны, но вторичны.

### 9.5 Отличие от Grasping Neural Process и Martingale Posterior Neural Processes

GNP получает labeled interaction set $D_t=\{(x,a,y)\}$ нового объекта и адаптирует posterior скрытых physical properties; FiGO получает один partial visual observation без пробных grasp’ов и оценивает ambiguity скрытой геометрии. Оба имеют shared object latent и feasibility decoder, поэтому это сходство надо признавать прямо.

MPNP строит martingale posterior через c.i.d. predictive pseudo-data как альтернативу latent-variable Bayes. FiGO не использует этот inference paradigm: его martingale — tower property семейства posteriors при переходе от fine visual experiment к его Blackwell garbling. Названия похожи, математические индексы и training signals различны.

### 9.6 Отличие от обычной consistency regularization

Обычная invariance требует одинаковых predictions после augmentation. Здесь это неверно; coarse distribution должна быть conditional mixture fine distributions. Conditional moment restriction сохраняет uncertainty и является содержательной частью метода.

### 9.7 Самая опасная reviewer criticism

> «Grasping Neural Process уже существует; Martingale Posterior Neural Process тоже существует. Здесь лишь другой encoder, risk score и consistency loss».

Чтобы criticism не стал фатальным, paper должен иметь все три результата:

1. формальную task-sufficiency и Blackwell/measure-valued-martingale proposition с корректным garbling construction;
2. empirical proof, что KCM улучшает calibration/generalization на unseen occlusion levels, а pairwise KL действительно collapse’ит uncertainty;
3. отдельный, не обязательный для marginal core, proof decision value joint correlations через tail-regret или batch selection — без ложного claim для marginal CVaR.

Если отсутствует пункт 2, contribution лучше позиционировать для CoRL/RSS, а не ICLR. Если отсутствует пункт 3, следует удалить joint-decision claim и оставить более простой marginal outcome field.

### 9.8 Что нельзя заявлять

- «first grasping under occlusion» — ложь из-за TARGO и других работ;
- «first uncertainty-aware grasping» — ложь;
- «first neural-process grasp model» или «first martingale neural process» — ложь;
- «first task-aware alternative to full completion» — слишком широко, TOSC близок;
- «provably safe» — conformal marginal coverage не равно physical safety;
- «joint posterior устраняет winner's curse при marginal-CVaR» — математически не следует;
- «SOTA» до одинакового candidate pool, TARGO split и real trials;
- «uncertainty always decreases with visibility» pointwise — математически неверно.

---

## 10. ICLR acceptance audit

Официальный [ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide) сводит решение к четырём вопросам: конкретна ли проблема, мотивирован ли подход и расположен ли он в литературе, поддержаны ли claims rigorous evidence, значимы ли новые знания. SOTA не обязателен сам по себе. [ICLR 2027 CFP](https://www.iclr.cc/Conferences/2027/CallForPapers) прямо включает uncertainty, probabilistic methods, geometric learning и robotics/autonomy и призывает к slow, complete work.

Текущая оценка **до результатов**:

| Dimension | Потенциал | Главный риск |
|---|---:|---|
| Problem significance | высокий | TARGO уже показывает problem и strong solution |
| Conceptual novelty | средний, условно средне-высокий | GNP и MPNP — сильные prior-art атаки; novelty держится на visual Blackwell/KCM law |
| Technical soundness | высокий при корректной постановке | finite-query process consistency и calibration требуют аккуратных proofs |
| Empirical rigor | пока неизвестно | большой baseline/real-robot burden |
| ICLR breadth | средне-высокий | надо показать general learning principle, а не только shelf system |
| SOTA potential | правдоподобен в high-occlusion top-1 selection | candidate recall может стать bottleneck |

Нужная paper story:

> Partial observation is an inverse problem, but grasping does not require solving that inverse problem in geometry space. The Bayes-sufficient object is the posterior of a task-induced function. When observations are progressively coarsened by occlusion, these function posteriors must form a measure-valued martingale. FiGO learns this posterior directly and enforces its law with conditional moment restrictions.

Это ICLR-level framing. Story «мы добавили uncertainty head к grasp detector» — нет.

Честная venue-оценка: **robotics-only результат с KCM regularizer скорее уровня CoRL/RSS**, даже при хорошем success rate. Путь к ICLR требует как минимум (i) theorem/identifiability statement с явно проверяемыми assumptions, (ii) controlled demonstration, что pairwise consistency даёт неверный posterior, а KCM восстанавливает unseen-garbling calibration, и (iii) второй masked inverse-problem domain или достаточно общий benchmark, показывающий перенос principle за grasping. Без этих трёх пунктов слово «ICLR-level» в proposal означает лишь направление, не прогноз принятия.

### Временной риск

На 2026-08-25 дедлайн full paper ICLR 2027 — **2026-09-25** ([official dates](https://iclr.cc/Conferences/2027/Dates)). Если инфраструктура, TARGO reproduction и significant experiments ещё не готовы, полноценная submission за месяц нереалистична и противоречит призыву CFP к slow science. Рациональная цель — ICLR 2028 либо ICLR 2027 только при уже существующем simulator/data/robot pipeline и сильных результатах первых двух недель.

---

## 11. Falsification / kill criteria

Идею следует остановить или понизить venue, если выполняется любое из следующего:

1. На fixed high-recall candidate pool independent distributional critic **с тем же KCM** равен joint FiGO по tail-regret/batch decisions: тогда joint process надо удалить из core, но marginal filtration idea ещё жива.
2. Joint energy/variogram scores улучшаются, но tail-regret/batch decision — нет: process modeling тогда не task-valuable.
3. $\mathcal L_{KCM}$ не улучшает unseen-severity calibration/AUOSC минимум на нескольких seeds: это убивает центральную filtration thesis.
4. Pairwise KL не хуже KCM: центральная filtration thesis не подтверждена practically.
5. TARGO-Net + deep ensemble или UNCLE-style completion-CVaR совпадает по success и укладывается в тот же latency/memory.
6. Выигрыш исчезает без correct class token.
7. Выигрыш существует только при synthetic random masks, но не при physical occluders/RealSense noise.
8. Proposal recall, а не selection uncertainty, объясняет почти все ошибки.
9. Real-world effect меньше variance между object sets/seeds.

Предварительные quantitative gates для продолжения full project:

- не менее +5 percentage points top-1 success над strongest fixed-pool uncertainty baseline при occlusion ≥0.6;
- заметное уменьшение selected-grasp calibration error;
- отсутствие существенного падения при occlusion 0–0.15;
- inference target <100 ms либо доказанный Pareto advantage latency/success над multi-completion baseline;
- effect сохраняется на class-free unseen-instance split.

Порог +5 pp — engineering gate, не статистическая guarantee; итоговые confidence intervals обязательны.

---

## 12. Минимальный feasibility study

### Неделя 1: controlled evidence и labels

- воспроизвести TARGO success-vs-occlusion хотя бы для одного direct baseline;
- построить information-only paired renderer;
- взять 300–500 ACRONYM objects;
- сформировать baskets по 64–128 grasps и full-mesh margins;
- проверить, что разные shapes с близкими coarse observations действительно меняют ranking candidates.

### Неделя 2: decisive ablation

Обучить одинаковый backbone в пяти версиях:

1. deterministic mean critic;
2. independent quantile critic;
3. independent distributional critic с KCM;
4. latent joint process без KCM;
5. joint FiGO с KCM.

Не тратить время на learned proposal, real robot и conformal layer до подтверждения:

- KCM benefit для marginal predictions;
- дополнительный joint correlation benefit только для joint-dependent decisions;
- filtration generalization benefit;
- acceptable latency.

### После gate

- TARGO-Net/GraspGen baselines;
- full ACRONYM scale;
- class/OOD splits;
- real shelf study;
- calibration;
- proofs и general masked-inverse toy experiment, показывающий, почему pairwise KL схлопывает coarse posterior.

---

## 13. Предлагаемые contributions будущей статьи

1. **Problem/formalism:** visual-coarsening family of task-pushforward posteriors for decisions from ambiguous partial geometry.
2. **Theory:** grasp-outcome posterior as Bayes-sufficient quotient of shape posterior; Blackwell/measure-valued martingale under occlusion filtration; correct conditional-mixture law and explicit failure of pairwise matching.
3. **Method:** $SE(3)/C_2$-structured conditional outcome field with KCM filtration consistency; shared latent is an optional joint extension, not the central novelty.
4. **Benchmarking insight:** decomposition of informational vs physical external occlusion, with nested paired observations.
5. **Empirics:** improved high-occlusion parallel-jaw selection, calibration and compute over deterministic direct models and probabilistic-completion planners.

Из них пункт 2 и его эмпирическая реализация в пункте 3 — core novelty. Пункт 1 частично предвосхищён GNP на другом observation regime; пункт 4 усиливает scientific value; пункт 5 необходим для принятия, но не является самостоятельной novelty claim.

---

## 14. Рабочие abstract и title

### Title

**Beyond Shape Completion: Blackwell-Consistent Grasp Outcome Fields under Visual Occlusion**

### Abstract draft

Single-view grasping is intrinsically ambiguous: many object geometries agree with the same visible point cloud, yet induce different grasp outcomes. Existing systems either ignore this ambiguity, model the likelihood of grasp poses, or reconstruct one or more complete shapes before planning. We argue that full shape inference is unnecessary. For any grasping objective, the posterior over the grasp-outcome function is a task-sufficient pushforward of the shape posterior. We further observe that posteriors conditioned on progressively revealed geometry obey a Blackwell/measure-valued-martingale law, whereas standard clean-to-occluded consistency incorrectly collapses uncertainty. We introduce FiGO, an $SE(3)$-structured conditional grasp-outcome field that predicts distributions of parallel-jaw grasp margins and enforces filtration consistency through kernel conditional moment restrictions. FiGO amortizes inference directly in grasp space without producing meshes, point completions, or scene SDFs. A controlled benchmark separates informational loss from physical obstruction under nested single-object occlusions. [Results placeholder: report only after experiments.] The framework suggests a general principle for task-directed inference under partial observations: learn the posterior of the decision-relevant function, and enforce coherence across observation filtrations.

---

## 15. Итоговое решение

**Продолжать FiGO, но не completion-based alternatives.**

Самая ценная научная часть — не CVaR, не flow, не neural process и не equivariant encoder отдельно. Финальный novelty search обнаружил два особенно близких прецедента: Grasping Neural Process уже учит latent action-feasibility function по истории взаимодействий, а Martingale Posterior Neural Processes уже соединяют neural processes с другой концепцией martingale posterior. Поэтому научное ядро сужается до следующего сочетания:

1. hidden geometry under **visual coarsening** quotient’ится в posterior task function без reconstruction;
2. posterior task functions по Blackwell-упорядоченным уровням visibility должны выполнять tower law, обучаемый без повторяющихся exact coarse observations через conditional moment restrictions;
3. физическая окклюзия даёт контролируемый и практически важный testbed, где обычная pairwise consistency заведомо задаёт неверную цель.

Если controlled experiments подтвердят, что KCM даёт независимый выигрыш над тем же probabilistic critic без KCM и pairwise matching, проект имеет правдоподобный ICLR-level path. Joint process остаётся только если отдельно помогает joint-dependent decisions. Если выигрыш даёт лишь occlusion augmentation или сильный encoder, честный вывод — practical CoRL/RSS-style improvement поверх TARGO, а не новая ML formulation.

---

## 16. Основные первичные источники

### Robotic grasping / occlusion

- Xia et al. [TARGO: Benchmarking Target-driven Object Grasping under Occlusions](https://arxiv.org/abs/2407.06168); [accepted IJCV 2026 project page](https://targo-benchmark.github.io/).
- Sundermeyer et al. [Contact-GraspNet](https://arxiv.org/abs/2103.14127).
- Wang et al. [Graspness Discovery / GSNet](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.html).
- Fang et al. [AnyGrasp](https://arxiv.org/abs/2212.08333).
- Lim et al. [EquiGraspFlow](https://openreview.net/forum?id=5lSkn5v4LK).
- Song et al. [Implicit Grasp Diffusion](https://proceedings.mlr.press/v270/song25b.html).
- Murali et al. [GraspGen](https://arxiv.org/abs/2507.13097).
- Eppner et al. [ACRONYM](https://research.nvidia.com/publication/2021-05_acronym-large-scale-grasp-dataset-based-simulation).

### Shape uncertainty / completion

- Lundell et al. [Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645).
- Saund & Berenson [Diverse Plausible Shape Completions from Ambiguous Depth Images](https://arxiv.org/abs/2011.09390).
- Humt et al. [Shape Completion with Prediction of Uncertain Regions](https://arxiv.org/abs/2308.00377).
- Duarte et al. [Measuring Uncertainty in Shape Completion to Improve Grasp Quality](https://arxiv.org/abs/2504.16183).
- Wu et al. [TOSC](https://ojs.aaai.org/index.php/AAAI/article/view/38053).
- Mansour et al. [UNCLE-Grasp](https://arxiv.org/abs/2601.14492).
- Feng et al. [FFHFlow](https://proceedings.mlr.press/v305/feng25a.html).
- Bagoren et al. [PUGS](https://arxiv.org/abs/2502.09824).
- Noseworthy et al. [Amortized Inference for Efficient Grasp Model Adaptation / Grasping Neural Process](http://groups.csail.mit.edu/rrg/papers/noseworthy_shaw_icra24.pdf).

### General ML / mathematics inspiration

- Garnelo et al. [Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a.html).
- Lee et al. [Martingale Posterior Neural Processes](https://openreview.net/forum?id=-9PVqZ-IR_).
- Falck et al. [Is In-Context Learning in Large Language Models Bayesian? A Martingale Perspective](https://proceedings.mlr.press/v235/falck24a.html).
- Muandet et al. [Kernel Conditional Moment Test via Maximum Moment Restriction](https://proceedings.mlr.press/v124/muandet20a.html).
- Kremer et al. [Functional Generalized Empirical Likelihood Estimation for Conditional Moment Restrictions](https://proceedings.mlr.press/v162/kremer22a.html).
- Kiyani et al. [Decision-Theoretic Foundations for Conformal Prediction](https://proceedings.mlr.press/v267/kiyani25a.html).
- Kolchinsky [Coarse-Graining and the Blackwell Order](https://arxiv.org/abs/1701.07602).
- [ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide).
- [ICLR 2027 Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers).
