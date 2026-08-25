# Learn the Feasibility Profile, Not the Hidden Shape

## Исследовательский меморандум: новый learning objective и архитектура для parallel-jaw grasp selection при foreground-окклюзии

**Дата novelty-аудита:** 25 августа 2026 г.  
**Рабочее название метода:** **GraspLFP** — *Grasp Latent Feasibility Profiles*  
**Главный тезис:** вместо completion скрытой формы или бинарного ответа «этот grasp сработает?» учить условное распределение **signed distance кандидата до границы множества допустимых grasps**, индуцированного неизвестной полной геометрией. Это компактный task-induced posterior: один скаляр на grasp, но он сохраняет всю информацию, необходимую для выбора grasp при любом требуемом допуске к ошибке.

> Статус документа: это research design с проверенной по доступной литературе novelty-гипотезой, а не утверждение уже достигнутого SOTA. Отсутствие статьи невозможно доказать поиском; формулировка novelty ниже означает «не найдено на дату аудита» и должна быть повторно проверена перед submission.

---

## 1. Решение в одном абзаце

Пусть полная, но при test-time неизвестная геометрия target и foreground-объекта задаёт множество `C(W)` всех parallel-jaw grasps, допустимых для **локального terminal close-and-hold**: без столкновения корпуса/тыльных поверхностей пальцев во время закрытия, с двумя контактами с target, допустимой шириной и antipodal force closure. Для кандидата `g` определим `rho(W,g)` как signed distance в метризованном grasp space до границы `C(W)`: положительная величина — радиус допустимых perturbations до первого failure, отрицательная — расстояние до ближайшего допустимого grasp. По одному noisy RGB-D наблюдению `X` скрытый мир `W` неоднозначен, поэтому правильный prediction target — не число и не completion, а

`F*(t | X,g) = P[rho(W,g) <= t | X,g]`.

Модель учится новым objective — **Boundary-Amplified Integrated Brier Score**:

`L_BAIBS = E integral a(t) [F_theta(t|X,g) - 1{rho<=t}]^2 dt`,

где `a(t)>0`, с усилением около `t=0`. Это строго proper distributional loss; обычный binary grasp BCE является только одним его срезом при `t=0`. На inference можно без retraining выбирать grasp по `P(rho>t_req)`, где `t_req` соответствует фактической точности робота, либо максимизировать нижний posterior quantile. Геометрия не реконструируется ни явно, ни как local occupancy/SDF.

---

## 2. Точная область и намеренные исключения

### Рассматривается

- один target на полке;
- один foreground-препятствующий/окклюдирующий объект, не clutter;
- один RGB-D кадр с wrist camera;
- шум глубины, holes, умеренная ошибка target mask;
- неизвестная скрытая grasp-релевантная геометрия target, известная только через обучающее распределение форм;
- rigid parallel-jaw gripper;
- выбор/локальная доводка grasp среди candidates;
- проверка terminal placement, закрытия пальцев и удерживаемого antipodal contact.

### Не рассматривается

- RL и VLA;
- active view / next-best-view;
- removal/rearrangement occluder;
- reachability руки, IK, полный approach trajectory, полный lift trajectory;
- causal taxonomy failure modes;
- full-object reconstruction, scene SDF, dense occupancy, local occupancy reconstruction;
- долговременная closed-loop политика.

В физическом эксперименте допустим короткий подъём на 1–2 см **только как бинарный тест факта удержания**; он не входит в моделируемый state/action cycle.

---

## 3. Что уже существует и где настоящий gap

### 3.1 Точный benchmark уже существует, но его objective остаётся стандартным

[TARGO / TARGO-Net](https://targo-benchmark.github.io/) (IJCV 2026; [paper](https://arxiv.org/abs/2407.06168)) изучает direct target grasping из одного RGB-D при occlusion. TARGO-Net сегментирует target, **полностью completes его point cloud**, fused cross-attention объединяет completed target и scene, затем обычные heads предсказывают grasp quality/orientation/width. Его loss использует BCE для `q` и regression для pose/width. На balanced synthetic test сильная окклюзия уменьшает GSR TARGO-Net примерно на 7%, тогда как VGN/GIGA падают примерно на 20%, EdgeGraspNet-варианты — до 30%. Это одновременно:

1. подтверждает важность hidden geometry;
2. показывает, что точная задача открыта;
3. оставляет незакрытым вопрос: зачем восстанавливать всю форму, если downstream нужен только выбор grasp?

### 3.2 «Восстановим только нужную часть» уже занято

- [Local Occupancy-Enhanced Object Grasping](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09354.pdf) (ECCV 2024) infers occupancy только в grasp-related regions и сообщает существенные улучшения на GraspNet-1Billion.
- [NeuGraspNet](https://www.roboticsproceedings.org/rss20/p046.html) (RSS 2024) совместно учит implicit scene geometry и grasp quality, затем renders local surface inside the gripper; его ablations подтверждают пользу action-local surface information.
- [PartialBiGrasp](https://arxiv.org/abs/2608.19188) (arXiv, 19 августа 2026) уже прямо заявляет отсутствие full reconstruction, но всё равно учит convolutional occupancy features скрытой local geometry для bimanual grasp refinement.
- [TOSC](https://ojs.aaai.org/index.php/AAAI/article/view/38053) (AAAI 2026) completes task-relevant contact regions, а не всю форму, для dexterous task-oriented grasping.
- Ранние [Shape Completion Enabled Robotic Grasping](https://arxiv.org/abs/1609.08546), [ShellGrasp-Net](https://arxiv.org/abs/2109.06837) и новые completion pipelines показывают долгую линию object-shell/full-shape reconstruction.

Следовательно, **local completion, contact-region completion или implicit occupancy — недостаточная novelty**.

### 3.3 Uncertainty-aware grasp scoring тоже не новый сам по себе

- [Dex-Net 2.0](https://arxiv.org/abs/1703.09312) учит probability of robust grasp success из synthetic depth и analytic metrics.
- [Robotic Pick-and-Place With Uncertain Segmentation and Shape Completion](https://arxiv.org/abs/2010.07892) сравнивает uncertainty-aware costs; learned grasp/place probability оказывается лучше uncertainty-unaware costs и быстрее Monte Carlo, а real packing улучшается на 7.8 процентного пункта.
- [Measuring Uncertainty in Shape Completion to Improve Grasp Quality](https://arxiv.org/abs/2504.16183) (2025) добавляет completion uncertainty в grasp ranking.
- [FFHFlow](https://proceedings.mlr.press/v305/feng25a.html) (CoRL 2025) моделирует uncertainty partial point clouds и risk-aware ranking для dexterous grasps.
- [SpringGrasp](https://tml.stanford.edu/SpringGrasp/) оптимизирует compliant dexterous grasp under shape uncertainty и сообщает 84% real success from a single view, минимум на 18 п.п. выше force-closure planner.

Следовательно, **добавить variance/CVaR/ensemble к бинарному score — недостаточная novelty**.

### 3.4 Robust margin существует, но не как posterior hidden-geometry feasibility profile

- [Toward an Analytic Theory of Intrinsic Robustness](https://arxiv.org/abs/2403.07249) связывает perturbations friction cones с сохранением force closure и показывает hardware benefit robust metrics.
- [FIRMGrasp](https://arxiv.org/abs/2607.25049) (июль 2026) уже вводит CVaR-based friction-aware force-closure margin; положительная risk margin даёт probabilistic closure certificate при заданном friction prior. Поэтому «наш новый objective — CVaR grasp score» был бы слишком близок к prior art.
- [Grasp Distance Fields](https://arxiv.org/abs/2608.00600) (август 2026) строит smooth distance до **набора заранее синтезированных grasp configurations** в arm-hand configuration space для trajectory-free feedback execution. Это не conditional signed distance кандидата до world-dependent feasible set и не inference скрытой формы, но термин `grasp distance field` использовать нельзя: он уже занят и создаст ложное пересечение.

### 3.5 Сильное косвенное свидетельство в пользу task-sufficient direct target

Свежий preprint [Object Pose and Shape Estimation for Grasping: Does it Work?](https://arxiv.org/abs/2605.26944) сообщает, что modular shape-first methods в его экспериментах дают примерно 1.6–2× success AnyGrasp, но требуют 45–105 секунд вместо 0.33 секунды; scale errors составляют 38–50% failure, а completion/sampling — bottleneck. Это не доказательство GraspLFP, но очень сильная мотивация: physics-rich hidden geometry действительно полезна, однако full reconstruction дорого и само создаёт ошибки. Нужен direct low-dimensional surrogate, сохраняющий decision information.

---

## 4. Итерации идей и почему четыре альтернативы отброшены

### Вариант A: stochastic local occupancy + CVaR

**Идея:** предсказывать samples occupancy внутри gripper и выбирать grasp по lower-tail quality.  
**Отброшен:** LOE, NeuGraspNet и PartialBiGrasp занимают local hidden geometry; FFHFlow и FIRMGrasp занимают generic uncertainty/CVaR motifs. Получилась бы композиция известных robotic components.

### Вариант B: distribution of first contacts along finger rays

**Идея:** competing-risks/survival model для первых left/right contacts и collision along closing rays.  
**Плюс:** существенно дешевле SDF.  
**Отброшен как top-1:** ShellGrasp already predicts entry/exit surfaces along camera rays; local surface rendering and occupancy-query methods создают близкое conceptual neighborhood. Кроме того, joint normals/contact times снова образуют длинный structured output, противоречащий требованию компактной постановки.

### Вариант C: Blackwell/martingale consistency across nested occlusions

**Идея:** posterior from a more occluded view должен быть coherent с mixture posteriors from refinements.  
**Причина отказа:** pointwise convex-order constraint между конкретной coarse/fine парой в общем случае неверен; tower property выполняется только после averaging по refinements conditional on coarse observation. Надёжная оценка такого conditional mixture потребует отдельной generative model или matched equivalence classes. Это интересный future regularizer, но не чистое ядро.

### Вариант D: one binary robust-success score

**Идея:** label success после random perturbations, BCE, uncertainty head.  
**Отброшен:** это близко к Dex-Net, pose-uncertainty grasp functions и Monte-Carlo robustness. Один Bernoulli target не различает grasp с радиусом 0.2 мм и 20 мм и фиксирует единственный noise budget.

### Выбранный вариант E: posterior signed feasibility profile

Он сохраняет компактность одного scalar outcome, но не теряет continuum tolerance decisions; имеет собственный strictly proper objective; не декодирует hidden geometry; допускает theorem-level characterization как loss-dependent Bayes-sufficient posterior.

---

## 5. Формальная постановка

### 5.1 Скрытый мир и наблюдение

Пусть

- `W = (S_T, S_O, eta)` — полная target geometry, foreground-object geometry и фиксированные/рандомизированные локальные физические параметры;
- `X = R(W, v, epsilon)` — single-view RGB-D, target mask и camera calibration после rendering/sensor noise;
- `g = (R_g, t_g, w_g)` — parallel-jaw terminal grasp candidate.

Не требуется восстанавливать `W`.

### 5.2 Локальное множество допустимых grasps

`C(W)` содержит grasp configurations, для которых при фиксированной palm pose и одномерном закрытии jaws:

1. forbidden gripper geometry не пересекает target или foreground object;
2. обе contact pads впервые контактируют с target, а не с occluder;
3. opening/closure находится в stroke limits;
4. contact pair удовлетворяет выбранному antipodal/force-closure certificate при фиксированном friction protocol.

Здесь нет arm approach, IK и full lift.

### 5.3 Метрика grasp space

Используем физически интерпретируемую локальную метрику

`d_G(g,g')^2 = ||Delta t||^2/sigma_t^2 + ||Log(R_g^T R_g')||^2/sigma_R^2 + |Delta w|^2/sigma_w^2`.

`sigma_t`, `sigma_R`, `sigma_w` — единицы robot uncertainty budget, а не tunable weights без смысла. Например, одно расстояние означает 2 мм translation, 2 градуса rotation или 1 мм width error.

### 5.4 Signed local feasibility radius

Для bounded neighborhood radius `R_max`:

`rho(W,g) = +min(R_max, dist(g, C(W)^c))`, если `g in C(W)`;

`rho(W,g) = -min(R_max, dist(g, C(W)))`, если `g notin C(W)`.

Если локально допустимого grasp нет, `rho=-R_max`. Тогда:

- `rho>0` эквивалентно nominal local feasibility;
- `rho>t>0` означает, что все perturbations ближе `t` остаются допустимыми;
- величина непрерывно различает fragile и robust positives, а также near-miss и hopeless negatives.

Практический label oracle может вычислять `rho` batched adversarial line search по Sobol directions в локальном `se(3) × width`, затем локально уточнять ближайшую boundary. Более дешёвая версия использует минимум Lipschitz-normalized constraint slacks; её положительное значение является lower bound на exact radius. В основной статье exact/approximation gap следует измерить на mesh subset, а не замалчивать.

### 5.5 Правильный prediction target

`F*(t | X,g) = P(rho(W,g) <= t | X,g)`.

Это posterior pushforward скрытого world distribution через mechanics map `W -> rho(W,g)`. Разные скрытые shapes, неразличимые в RGB-D, могут давать разные `rho`; multimodality является правильным ответом, а не reconstruction error.

---

## 6. Новый learning objective

### 6.1 Boundary-Amplified Integrated Brier Score (BAIBS)

Пусть `F_theta(t|X,g)` — predicted CDF. Определим

`a(t) = 1 + lambda exp(-t^2/(2 sigma_0^2))`, поэтому `a(t)>0` всюду, но thresholds около feasibility boundary `t=0` получают больше веса.

`L_BAIBS(theta) = E_[W,X,g] integral_[-Rmax,Rmax] a(t) (F_theta(t|X,g) - 1{rho(W,g)<=t})^2 dt`.

На практике integral оценивается 8–16 stratified threshold samples; половина uniform по диапазону, половина из narrow distribution около нуля, с importance correction.

### 6.2 Почему это не «CRPS ради CRPS»

- Каждый threshold `t` — отдельная физическая задача: классифицировать, выдерживает ли grasp требуемый error tolerance `t`.
- Обычный grasp BCE учит только `t=0`.
- BAIBS учит все robot precision budgets одновременно и позволяет менять safety requirement без retraining.
- Weight остаётся строго положительным, поэтому boundary emphasis не разрушает propriety.
- Output остаётся одномерным; model не обязана быть generative model формы.

[Gneiting & Raftery](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437) дают общую теорию strictly proper scoring rules; CRPS эквивалентен integral Brier scores по thresholds. Современная работа по [nonparametric distributional regression](https://proceedings.mlr.press/v258/peng25a.html) дополнительно подтверждает связь CRPS с одновременным обучением всех quantiles без quantile crossing.

### 6.3 Простая теорема propriety

Для фиксированных `(X=x,g)` и каждого `t` conditional expected Brier loss единственным образом минимизируется в

`F(t)=P(rho<=t|x,g)`.

Так как `a(t)>0` почти всюду, интеграл минимизируется истинной CDF почти всюду. Следовательно, при достаточной model capacity population minimizer BAIBS равен `F*`. Это theorem, а не calibration heuristic.

---

## 7. Почему это task-sufficient и broad general ML

### 7.1 Feasibility quotient

Два hidden worlds `W1` и `W2` эквивалентны относительно candidate family, если они индуцируют одинаковые `rho(W,g)` для всех рассматриваемых `g`. Их полные shapes могут сильно отличаться; никакой downstream selector из рассматриваемого класса не должен их различать.

Идея соответствует loss-dependent Bayes sufficiency: [Bayes-Sufficient Representations in Supervised Learning](https://arxiv.org/abs/2606.04045) формализует, что требуемая representation определяется distribution **и loss**, а full predictive distribution требуется при strictly proper loss. Здесь representation target не hidden shape posterior, а его action-conditioned feasibility pushforward.

### 7.2 Proposition: sufficiency для семейства utilities

Для любой bounded utility вида

`U(W,g)=psi(rho(W,g))`

conditional expected utility равна

`E[U|X,g] = integral psi(r) dF*(r|X,g)`.

Значит, `F*` Bayes-sufficient для всего семейства utilities: threshold success, expected robustness, lower-tail selection, asymmetric penalties, разные robot tolerances. Posterior `P(W|X)` достаточен, но избыточен.

### 7.3 Selection regret bound

Для фиксированного tolerance `t`, пусть `p_t(g)=1-F*(t|X,g)` и `sup_g |F_theta(t|X,g)-F*(t|X,g)| <= epsilon`. Если `g_hat` максимизирует predicted `p_t`, а `g_star` — true `p_t`, то

`p_t(g_star)-p_t(g_hat) <= 2 epsilon`.

Это прямо связывает CDF calibration на нужном threshold с quality выбранного grasp.

### 7.4 Общая идея за пределами grasping

**Latent Feasibility Profile Learning:** при partial observation скрытое состояние задаёт random feasible action set; вместо reconstruct-then-optimize учится conditional law signed distance query-action до этого set. GraspLFP — первая предметная инстанциация. Этот framing применим к occluded collision checking, safe placement, insertion under tolerance и design feasibility, но статья должна доказать его сначала на grasping, а не размывать scope.

---

## 8. Новая архитектура: Dual-Frame Feasibility Transformer

### 8.1 Почему нужны два frame

Occlusion структурирована camera rays, а grasp mechanics — gripper coordinates. Перевести всё только в camera frame теряет inductive bias gripper interaction; перевести всё только в gripper frame скрывает line-of-sight ordering. Поэтому архитектура обрабатывает обе структуры явно.

### 8.2 Input tokens без completion

1. **Visible point tokens:** target и foreground-obstacle points с RGB, depth residual/noise estimate, segmentation confidence и class bit `target/occluder`.
2. **Gripper interaction tokens:** 32–64 canonical sites на contact pads, finger backs, palm lip и closing corridor, transformed candidate pose `g`.
3. **Visibility-order code для каждого interaction site:** по calibrated projection сравнивается его camera depth с measured depth в соответствующем pixel neighborhood. Код различает measured-free-space, on-visible-surface, behind-visible-target, behind-foreground-occluder и outside-valid-depth. Это deterministic evidence, не predicted occupancy.

### 8.3 Encoder

- Sparse point encoder с SE(3)-equivariant/vector features извлекает global visible-shape prior и local point features. [SE(3)-Transformer](https://proceedings.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html) и [Vector Neurons](https://openaccess.thecvf.com/content/ICCV2021/html/Deng_Vector_Neurons_A_General_Framework_for_SO3-Equivariant_Networks_ICCV_2021_paper.html) дают прямые основания ожидать лучшую rotation generalization и weight sharing на point clouds.
- Candidate-conditioned cross-attention идёт от interaction tokens к cached point tokens.
- Relative geometry входит дважды: offsets in gripper frame для contact/collision compatibility и camera-ray depth/angle residuals для visibility ordering.
- Invariant pooling выдаёт один vector `h(X,g)`.

Это не voxel grid, не tri-plane, не SDF и не hidden surface decoder.

### 8.4 Distribution head

Основной вариант — conditional monotone rational-quadratic spline CDF на `[-Rmax,Rmax]`; inversion analytic, quantiles не пересекаются. Более простой baseline — mixture of 5 logistic CDFs. Head выдаёт только scalar law, а не geometry field.

### 8.5 Candidate selection и refinement

- Candidate pool берётся из одного фиксированного generator для всех сравниваемых rerankers; это изолирует contribution selection objective.
- Для known robot tolerance `t_req`: `score(g)=1-F_theta(t_req|X,g)`.
- Для posterior-conservative режима: `score(g)=Q_alpha[rho|X,g]`, например `alpha=0.1`.
- Optional local refinement: gradient ascent по `SE(3)` Lie-algebra coordinates через differentiable quantile/CDF head, максимум 3–5 итераций. Основная claim должна сохраняться и без refinement.
- Если лучший lower quantile не положителен, model может abstain. Abstention оценивается coverage–risk curve, но не должно скрывать failures при фиксированном coverage.

### 8.6 Complexity target

Point features кэшируются один раз; каждый grasp использует малый fixed stencil и local attention. Цель — score 1,024 candidates менее чем за 100 мс на desktop GPU, без 3D grid. Это реалистичная гипотеза, не обещанный результат.

---

## 9. Label generation без full reconstruction в model

Full meshes разрешены **только training oracle**.

1. Собрать watertight target meshes из Acronym/ShapeNet/Objaverse subset; разделить по object identity и category.
2. Разместить target на shelf и ровно один foreground occluder; контролировать 2D visibility bins `0–0.2, ..., 0.8–0.9`.
3. Render RGB-D с axial depth noise, quantization, missing pixels, edge flying pixels, calibration perturbations и mask erosion/dilation.
4. Сгенерировать candidate pool одинаковым base detector или mesh sampler.
5. На полном training world вычислить membership `g in C(W)` и signed boundary radius `rho` batched GPU oracle.
6. Boundary-stratified sampling увеличивает долю fragile grasps около `rho=0`; importance weights восстанавливают исходную distribution.

Критически важный split — **observation-equivalence stress set**: несколько разных hidden-side morphs имеют практически одинаковую visible point cloud до установленного Chamfer threshold, но разные grasp feasibility profiles. Без него модель может демонстрировать хорошую среднюю accuracy, не решая ambiguity.

---

## 10. Эксперимент, способный подтвердить или уничтожить идею

### 10.1 Datasets

- **Primary synthetic:** single-target/single-occluder shelf benchmark на широком mesh distribution.
- **External benchmark:** TARGO balanced occlusion split, адаптированный так, чтобы candidate pool и labels были доступны; отдельно сообщить, что оригинальный TARGO — cluttered benchmark.
- **Real RGB-D offline:** wrist-camera captures полного lab setup, meshes/poses получены отдельно только для evaluation labels.
- **Real robot:** humanoid/arm с parallel-jaw gripper, короткий 1–2 см hold test.

### 10.2 Baselines

1. Base grasp generator raw score: AnyGrasp/GSNet/Contact-GraspNet family.
2. TARGO-Net-like full completion + binary quality.
3. LOE/NeuGraspNet-like local occupancy scorer.
4. Same Dual-Frame architecture + BCE on `1{rho>0}`.
5. Same architecture + deterministic Huber regression of `rho`.
6. Same architecture + Gaussian NLL.
7. Multi-completion posterior + analytic feasibility + Monte Carlo/CVaR, matched candidate pool.
8. GraspLFP with uniform CRPS weight.
9. GraspLFP with BAIBS boundary amplification.

### 10.3 Главные metrics

- top-1 grasp success vs occlusion bin;
- success at fixed coverage и coverage–risk curve при abstention;
- `P(rho>t)` Brier/ECE for several preregistered `t`;
- CRPS/BAIBS;
- selected true margin и 10th-percentile margin;
- target-contact failure и terminal gripper collision, только как diagnostic metrics, не отдельные learned failure heads;
- runtime, memory, candidates/sec;
- regret to full-geometry oracle;
- performance on observation-equivalence stress set;
- cross-noise-budget transfer: один trained model, несколько `t_req`, без retraining.

### 10.4 Real protocol

- минимум 20 held-out objects;
- три occlusion bands: 20–40%, 40–60%, 60–80%;
- randomized object/occluder placement;
- одинаковый candidate generator для всех rerankers;
- 95% bootstrap confidence intervals и paired comparison по одной scene sequence;
- заранее определить success: target удерживается после вертикального displacement 1–2 см в течение 2 секунд;
- не отбрасывать трудные scenes post hoc.

### 10.5 Falsification gates

Идею следует считать неудачной, если выполняется хотя бы одно:

1. GraspLFP не превосходит same-architecture BCE на high-occlusion top-1 success хотя бы статистически, а даёт только prettier calibration plots.
2. Cross-tolerance selection не лучше retrained/fixed binary heads.
3. Observation-equivalence set не показывает benefit distributional prediction над mean/Gaussian.
4. Matched-compute local occupancy превосходит GraspLFP на success и latency.
5. Exact boundary radius слишком дорог для scalable labeling, а cheap normalized margin плохо коррелирует с real success.
6. Full-completion posterior при сопоставимом compute стабильно лучше.
7. Non-occluded performance падает более чем на 2 п.п. без компенсирующего high-occlusion gain.

Preregistered success target для серьёзной SOTA claim: не менее +5 п.п. absolute top-1 real success над лучшим matched candidate-pool baseline в 60–80% occlusion, с non-inferiority в 0–20%, плюс существенное улучшение BAIBS/CRPS и latency ниже completion baseline.

---

## 11. Ablations, без которых reviewer не поверит

1. CDF objective vs BCE vs scalar regression при идентичной архитектуре/данных.
2. Camera-frame visibility code removed.
3. Gripper-frame relative geometry removed.
4. Global visible-target token removed.
5. Foreground-object tokens removed.
6. Equivariant encoder vs non-equivariant PointNet/Transformer с matched parameters.
7. Boundary amplification `lambda=0` vs preregistered `lambda`.
8. Exact radius labels vs Lipschitz-normalized approximate margin.
9. Mixture/spline distribution head vs Gaussian.
10. Candidate reranking only vs optional gradient refinement.
11. Known clean target mask vs noisy predicted mask.
12. Seen instances, unseen instances, unseen categories, real sensor transfer.

Особенно важна ablation №1: без неё contribution будет выглядеть как новая feature architecture, а не новый learning objective.

---

## 12. Косвенные основания ожидать высокую эффективность

Это не прямое доказательство будущего SOTA, но цепочка evidence согласована:

1. **Hidden geometry действительно нужна.** TARGO-Net теряет около 7% при extreme occlusion против примерно 20–30% у completion-free/local baselines в его benchmark.
2. **Task-local geometry помогает.** LOE и NeuGraspNet улучшают grasping, когда hidden/local surface information сосредоточена около gripper.
3. **Physics-aware reconstructed shape может сильно превосходить end-to-end score**, но слишком медленна и страдает scale error: shape-first study 2026 сообщает 1.6–2× success против AnyGrasp, 45–105 с latency и 38–50% failures из-за scale.
4. **Learned uncertainty-aware costs могут быть быстрее MC и лучше uncertainty-unaware ranking.** Это показано Gualtieri & Platt.
5. **Robust margins correlate with real success лучше nominal force closure.** Intrinsic-robustness hardware trials и FIRMGrasp stress tests подтверждают ценность margins/tails.
6. **Strong fast proposal generator уже существует.** [AnyGrasp](https://doi.org/10.1109/TRO.2023.3281153) сообщает 93.3% bin-clearing success, >900 picks/hour и robustness to depth noise; значит, эксперимент может изолировать hard selection under occlusion вместо заново решать proposal generation.
7. **Proper distributional objectives имеют корректный population target.** В отличие от ad-hoc variance penalty, BAIBS не поощряет ложную uncertainty при идеальной capacity.

Главная гипотеза эффективности: GraspLFP должен приблизиться к physics benefit shape-first planners, но сохранить runtime direct scorer и не наследовать reconstruction Chamfer/scale hallucination errors.

---

## 13. Novelty matrix

| Line of work | Что predicts | Как использует uncertainty | Full/local geometry output | Отличие GraspLFP |
|---|---|---|---|---|
| TARGO-Net | binary quality + pose/width | implicit via completion | full target completion | conditional CDF signed feasible-set radius |
| LOE / NeuGraspNet / PartialBiGrasp | occupancy/surface features | mostly implicit | local/global occupancy | no occupancy or surface decoder |
| Dex-Net / GQ-CNN | probability of success | fixed perturbation model | no completion required | one threshold vs continuum tolerances |
| Completion uncertainty methods | shape/completion uncertainty + score | ensemble/variance/MC | yes | direct mechanics pushforward |
| FFHFlow | distribution over dexterous grasps + likelihood uncertainty | latent flow | no explicit full shape, but grasp generative law | distribution of candidate robustness, parallel-jaw selection |
| SpringGrasp / intrinsic robustness | analytic robust grasp metric | known uncertain surface/contact model | requires surface model | inferred posterior from single occluded RGB-D |
| FIRMGrasp | CVaR friction margin | explicit friction prior | full contacts/wrenches | hidden-geometry conditional full CDF, not CVaR-only |
| Grasp Distance Fields | distance to stored candidate set for feedback | execution constraints | candidate/configuration field | signed distance to world-dependent feasible set for perception/selection |

**Не найдено:** работа, которая из одного occluded noisy RGB-D, без shape/local occupancy reconstruction, учит strictly proper conditional distribution signed distance candidate grasp до boundary hidden-world feasible grasp set и использует её для zero-retraining tolerance adaptation.

---

## 14. ICLR acceptance audit

[ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide) формулирует четыре ключевых вопроса: конкретная проблема; мотивация и placement in literature; поддержаны ли claims; significance/new knowledge. SOTA сам по себе не обязателен.

### Потенциально сильные стороны

- Новый general-ML object: action-conditioned latent feasibility profile.
- Новый strictly proper objective, стандартный BCE показан как вырожденный single-threshold case.
- Нетривиальная low-dimensional sufficiency story вместо pipeline composition.
- Архитектура следует структуре camera occlusion и gripper mechanics, не декодирует SDF.
- Theory, controlled ambiguity benchmark, simulation, real RGB-D и real robot могут поддержать claims.
- Zero-retraining adaptation к разным robot uncertainty budgets — falsifiable new knowledge, а не просто AP gain.

### Главные reviewer attacks

1. **«Это всего лишь CRPS на новой label».** Ответ возможен только через feasibility-quotient theorem, continuum tolerance decision, observation-equivalence experiment и large gain над same-architecture BCE.
2. **«Signed radius label — ручная metric engineering».** Нужно показать correlation с physical success, compare exact/approximate oracle и robustness к metric scales.
3. **«Posterior не calibrated under sim-to-real».** Нельзя обещать distribution-free guarantee; нужны real calibration plots, held-out sensor conditions и честная limitation.
4. **«Candidate generator ограничивает ceiling».** Нужен oracle-pool recall и минимум два generator pools.
5. **«Foreground obstacle делает это просто collision prediction».** Feasible set должен включать target contacts, width и force closure; collision-only baseline обязателен.
6. **«Incremental robotics reranker».** Paper framing и experiments должны доказывать general objective, а не продавать ещё один module после AnyGrasp.

### Текущая оценка потенциала

- **Novelty hypothesis:** высокая, но требует повторного search audit перед submission.
- **Technical depth:** высокая при exact signed-radius oracle + propriety/sufficiency/regret results.
- **Empirical risk:** высокий; conditional multimodality может оказаться слабой на обычных datasets, поэтому equivalence stress set и broad shape prior обязательны.
- **ICLR fit:** потенциально сильный, если paper отвечает «какой posterior действительно нужен downstream decision?» и показывает перенос objective за пределы одного фиксированного tolerance. Без этого вероятнее CoRL/RSS, чем ICLR.

---

## 15. Минимальный publishable implementation path

### Phase 0: oracle sanity, 2 недели

- Реализовать `C(W)` и approximate `rho` для watertight meshes.
- На 100 объектах проверить monotonic relation `rho` с empirical success under pose perturbations.
- Измерить label cost и exact-vs-approx gap.
- Stop, если Spearman correlation с empirical robustness низкая.

### Phase 1: objective sanity, 2–3 недели

- Фиксированный candidate pool, простой PointNet encoder.
- BCE, Huber, Gaussian NLL, CRPS/BAIBS.
- Artificial ambiguity pairs с одинаковой visible half и разной hidden half.
- Stop, если distributional objective не выигрывает selection regret.

### Phase 2: dual-frame architecture, 3–4 недели

- Visibility-order tokens, gripper interaction stencil, equivariant point features.
- Profile runtime и memory с 1,024 grasps.
- Ablate оба coordinate frames.

### Phase 3: benchmark and real transfer, 4–6 недель

- Large shape distribution, held-out categories, TARGO adaptation.
- Real RGB-D calibration, затем paired robot trials.
- Только после этого формулировать SOTA claim.

---

## 16. Рекомендуемый paper pitch

**Title:** *Learn the Feasibility Profile, Not the Hidden Shape: Decision-Sufficient Grasping under Occlusion*

**One-sentence claim:**

> Under partial observability, reconstructing the latent world is unnecessary for a family of robust decisions: it is sufficient to predict the conditional distribution of an action's signed distance to the latent feasible set; for occluded parallel-jaw grasping, this yields a compact, calibrated, tolerance-adaptive selector without shape completion.

**Три contributions, не больше:**

1. Latent Feasibility Profile formulation + Bayes-sufficiency/regret characterization.
2. BAIBS, a boundary-focused strictly proper continuum-tolerance objective, and Dual-Frame Feasibility Transformer without geometry decoding.
3. Occlusion/ambiguity benchmark plus controlled and real evidence that feasibility profiles beat binary scoring and completion/local-occupancy baselines at matched candidate pools.

---

## 17. Финальный вывод

Сильнейшая найденная постановка — не «угадать скрытую поверхность лучше», а **сменить объект обучения**. Полная hidden shape является nuisance latent variable; binary grasp success слишком груб. Между ними находится компактный объект: posterior signed distance grasp до boundary скрытого feasible set. Он:

- не reconstructs объект;
- представляет hidden ambiguity, а не скрывает её в одном score;
- даёт один scalar distribution вместо большого geometry field;
- обучается strictly proper loss;
- позволяет выбирать под любой hardware tolerance без retraining;
- допускает clear theorem, hard falsification и direct comparison с текущими SOTA paradigms.

Именно сочетание **нового prediction target**, **continuum-threshold objective**, **task-sufficiency theory** и **dual camera/gripper inductive bias** делает идею заметно сильнее, чем очередной modified grasp pipeline.

---

## Основные источники

1. [ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide)
2. [TARGO / TARGO-Net, IJCV 2026](https://targo-benchmark.github.io/)
3. [Local Occupancy-Enhanced Object Grasping, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09354.pdf)
4. [NeuGraspNet, RSS 2024](https://www.roboticsproceedings.org/rss20/p046.html)
5. [PartialBiGrasp, arXiv 2026](https://arxiv.org/abs/2608.19188)
6. [TOSC, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38053)
7. [Dex-Net 2.0](https://arxiv.org/abs/1703.09312)
8. [Uncertain Segmentation and Shape Completion for Pick-and-Place](https://arxiv.org/abs/2010.07892)
9. [Measuring Uncertainty in Shape Completion](https://arxiv.org/abs/2504.16183)
10. [FFHFlow, CoRL 2025](https://proceedings.mlr.press/v305/feng25a.html)
11. [SpringGrasp](https://tml.stanford.edu/SpringGrasp/)
12. [Intrinsic Robustness for Dexterous Grasping](https://arxiv.org/abs/2403.07249)
13. [FIRMGrasp](https://arxiv.org/abs/2607.25049)
14. [Configuration-Space Grasp Distance Fields](https://arxiv.org/abs/2608.00600)
15. [Object Pose and Shape Estimation for Grasping: Does it Work?](https://arxiv.org/abs/2605.26944)
16. [AnyGrasp, IEEE T-RO 2023](https://doi.org/10.1109/TRO.2023.3281153)
17. [Strictly Proper Scoring Rules, Prediction, and Estimation](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437)
18. [Bayes-Sufficient Representations in Supervised Learning](https://arxiv.org/abs/2606.04045)
19. [SE(3)-Transformer, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html)
20. [Vector Neurons, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Deng_Vector_Neurons_A_General_Framework_for_SO3-Equivariant_Networks_ICCV_2021_paper.html)
