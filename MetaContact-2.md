# Grasp What You Cannot See: Blackwell-consistent outcome processes for parallel-jaw grasping under external occlusion

**Статус:** исследовательский checkpoint v0.2 после второго novelty pass, 25 августа 2026.  
**Целевая площадка:** ориентир на ICLR (на практике — следующий доступный цикл); принятие не гарантируется.  
**Главный вывод:** наиболее перспективная постановка — не восстанавливать единственную полную форму и не генерировать лишь распределение grasp poses, а учить **условное распределение целой функции исхода grasp-а** на пространстве `SE(3) × width`. Это распределение должно быть согласовано между вложенными уровнями внешней окклюзии по tower property условного ожидания / Blackwell order. Рабочее название метода: **OC-GOP — Occlusion-Consistent Grasp Outcome Process**. После жёсткого novelty audit это **conditional go**, а не готовый claim: общая conditioning consistency уже имеет близкий prior art, поэтому ICLR-ценность должна прийти из decision-sufficient representation, physical garblings, label-free projection и доказанной sample efficiency.

---

## 0. Короткий ответ на исходный вопрос

Да, evidence того, что ухудшение обычного partial single-view PCD усиливается при потере видимой геометрии, уже достаточно сильный. Но литература смешивает три разных режима:

1. self-occlusion чистого одиночного объекта;
2. inter-object occlusion в clutter;
3. контролируемая внешняя окклюзия целевого одиночного объекта передним препятствием.

Для первого режима падение измерено прямо. Например, на unseen EGAD objects в NeuGraspNet переход от fixed top view к hard 15° view снижает GSR AnyGrasp с `68.46` до `56.23`, Contact-GraspNet с `48.98` до `30.46`, GIGA с `27.49` до `17.10`, а самого NeuGraspNet с `79.76` до `67.21` ([NeuGraspNet, RSS 2024, Table III](https://www.roboticsproceedings.org/rss20/p046.pdf)). GraspLDM отдельно сообщает, что на single-view partial clouds значительная доля ошибок — столкновения с невидимой частью объекта, а неполные поверхности принимаются моделью за настоящие края ([GraspLDM](https://arxiv.org/abs/2312.11243)).

Для третьего режима самое прямое новое evidence даёт domain-specific UNCLE-Grasp: при росте leaf occlusion partial-cloud baseline становится плохим и нестабильным; completion помогает, но при тяжёлой окклюзии остаётся недостаточным; uncertainty-aware selection улучшает success среди предпринятых попыток на максимальной simulated occlusion с `0.780` до `0.870`, а на physical robot при примерно 87% окклюзии — с `0.483` до `0.800`, ценой abstention ([UNCLE-Grasp, preprint 2026](https://arxiv.org/abs/2601.14492)). Это специализированный объект и pipeline, не общее решение.

Следовательно, исходная гипотеза хорошо мотивирована, но публикационная задача не может звучать как «показать, что ещё одна окклюзия ухудшает grasping»: это ожидаемо и частично уже показано. Сильная новая задача — **научиться представлять и согласованно переносить именно task uncertainty, возникающую из-за структурной потери поверхности**, не восстанавливая всю форму.

---

## 1. Точный scope

### В задаче

- parallel-jaw gripper;
- один заранее выделенный target object на полке;
- один RGB-D кадр wrist camera;
- noisy partial PCD;
- target может быть частично закрыт спереди отдельным физическим препятствием;
- class label может быть доступен, но метод обязан иметь class-agnostic режим;
- primitive outcome: устойчивое замыкание захвата и короткий подъём, а не оценка длинной траектории и не long-horizon policy;
- допускается синтетическое обучение на полном mesh и известных grasp outcomes.

### Явно вне задачи

- RL;
- VLA/VLM как основной метод;
- rearrangement, pushing и многошаговое освобождение target;
- предсказание исполнимости всего цикла approach-to-lift как главный contribution;
- полный scene SDF/TSDF как обязательная внутренняя переменная;
- causal failure-mode taxonomy;
- deterministic full-shape completion как главный ответ.

### Что считать внешней окклюзией

Нельзя имитировать её iid point dropout. Это **ray-consistent, input-dependent missingness**: передний объект заменяет глубину target своими depth returns в связной области изображения. Маска пропусков связана с геометрией камеры, target и blocker. В терминах missing-data literature это ближе к structured non-random missingness; игнорирование механизма missingness вообще может оставлять систематический bias даже при больших данных ([Identifiable Generative Models for MNAR, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/e8a642ed6a9ad20fb159472950db3d65-Abstract.html); [NeuMiss, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/42ae1544956fbe6e09242e6cd752444c-Paper.pdf)). Здесь это статистическая аналогия, а не causal failure-mode постановка.

---

## 2. Карта литературы и фактический gap

### 2.1 Direct grasp detection на partial PCD

Классическая линия напрямую связывает наблюдаемую геометрию с grasp poses/quality:

- GPD работает с noisy, partially occluded RGB-D/PCD без CAD ([ten Pas et al.](https://arxiv.org/abs/1706.09911));
- GraspNet-1Billion дал большой single-view RGB-D benchmark, >1B annotations и analytic evaluation ([CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html));
- Contact-GraspNet обучен на 17M simulated grasps и привязывает proposal к наблюдаемым contact points ([Contact-GraspNet](https://arxiv.org/abs/2103.14127));
- GSNet/Graspness учит видимый graspable landscape и существенно улучшает sampling ([ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.html));
- AnyGrasp добавляет dense supervision, stability cues и temporal correspondence; в bin clearing сообщает 93.3% success, но это не контролируемый external-occlusion test ([AnyGrasp](https://arxiv.org/abs/2212.08333));
- L2G прямо генерирует 6-DoF parallel-jaw grasps из partial view, причём contact sampling по определению начинается с видимых points ([L2G](https://arxiv.org/abs/2203.05585));
- новые генеративные варианты включают GraspLDM ([arXiv](https://arxiv.org/abs/2312.11243)), Grasp Diffusion Network ([preprint 2024](https://arxiv.org/abs/2412.08398)) и Implicit Grasp Diffusion ([CoRL 2024 proceedings](https://proceedings.mlr.press/v270/song25b.html)).

**Gap этой линии:** output diversity не равна epistemic uncertainty. Diffusion может моделировать много хороших способов схватить один известный объект, но это не означает, что её samples являются posterior hypotheses о том, какие grasp outcomes возможны при разных скрытых формах. GraspLDM прямо отмечает collision с невидимыми частями как главный failure mode partial input.

### 2.2 Geometry/completion как auxiliary или основной путь

- Varley et al. восстанавливают voxel shape из 2.5D view и планируют на completed object ([Shape Completion Enabled Robotic Grasping](https://arxiv.org/abs/1609.08546)).
- GIGA совместно учит implicit geometry и affordance; на packed/pile scenes превосходит VGN и показывает пользу geometry auxiliary task ([RSS 2021](https://www.roboticsproceedings.org/rss17/p024.html)).
- NeuGraspNet делает global и gripper-local neural surface rendering; completion особенно помогает в hard views ([RSS 2024](https://www.roboticsproceedings.org/rss20/p046.html)).
- Local Occupancy-Enhanced Grasping восстанавливает только grasp-local occupancy. На GraspNet AP: `49.72` без occupancy, `53.84` с local occupancy, `58.78` при completed 256-view observation; real success: GSNet `87.72%`, method `94.34%` ([ECCV 2024](https://arxiv.org/abs/2407.15771)). Это сильное evidence, что missing local geometry реально ограничивает grasping, но severe lack of observation остаётся failure case.
- Domain-prior method CVPR 2024 прямо пишет, что novel object shape нельзя вывести из partial PCD без prior, и поэтому использует multi-view TSDF ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Generalizing_6-DoF_Grasp_Detection_via_Domain_Prior_Knowledge_CVPR_2024_paper.html)).
- ZeroGrasp совместно предсказывает octree shape и grasps, использует explicit 3D occlusion fields, 1M synthetic RGB-D, 12K objects и 11.3B validated grasps. Он сообщает SOTA на GraspNet-1B, `5 FPS`, improvement collision score с partial depth `59.93` до reconstruction-based `70.53`, real-object pick rate с `56.25%` до `75%` ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.html)). ReOcS делит real RGB-D на easy/normal/hard occlusion, но оценивает прежде всего reconstruction; это ещё не controlled grasp-under-occlusion benchmark.

**Gap этой линии:** completion решает более трудную задачу, чем требуется decision maker. Chamfer/F1 оптимизируют и геометрию, не влияющую ни на один допустимый grasp. Одна plausible reconstruction может быть task-wrong; несколько reconstructions дороги, их geometric diversity не гарантирует calibrated diversity grasp outcomes. Даже локальная occupancy остаётся point estimate.

### 2.3 Shape uncertainty и risk-aware grasping

- Lundell et al. генерируют MC-dropout voxel completions, строят grasps на mean shape и усредняют analytic quality по shape samples; 90k simulated и 200 real grasps показывают статистически значимое улучшение над point completion ([Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645)).
- Duarte et al. добавляют uncertainty completed points в grasp score и улучшают ranking real parallel-jaw grasps ([preprint 2025](https://arxiv.org/abs/2504.16183)).
- SpringGrasp планирует dexterous/compliant grasp под shape uncertainty и может принимать raw PCD либо completion output ([RSS 2024](https://www.roboticsproceedings.org/rss20/p042.pdf)).
- FFHFlow учит uncertainty-aware flow для dexterous hand из partial observations и risk-aware ranking ([CoRL 2025](https://proceedings.mlr.press/v305/feng25a.html)); embodiment не parallel-jaw.
- UNCLE-Grasp семплирует strawberry completions, фильтрует grasps, агрегирует force closure и использует LCB/abstention ([preprint 2026](https://arxiv.org/abs/2601.14492)).

**Gap этой линии:** uncertainty почти всегда живёт в geometry space, затем дорого проталкивается через grasp planner. Обычно оцениваются marginal mean/variance отдельного grasp-а или object-level LCB. Не моделируется joint correlation outcomes разных grasps: две формы могут одновременно поднять/опустить качества целого семейства grasпов. Нет требования, чтобы posterior при 60% occlusion был корректной marginalization posterior-а при 20% occlusion.

### 2.4 Очень свежие collisions, которые нельзя игнорировать

- **A Hybrid Optimization Framework for Grasp Synthesis under Partial Observations** сочетает EBM, ICP и SVGD; на 5,360 attempts сообщает `60.9%` против AnyGrasp `31.1%` ([preprint, июнь 2026](https://arxiv.org/abs/2606.18053)). Это сильный optimization baseline, но не posterior/coherence method.
- **Cross-view Fusion for Robust 6-DoF Grasp Pose Estimation** использует auxiliary view и избегает полной reconstruction ([CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_A_Cross-view_Fusion_Framework_for_Robust_6-DoF_Grasp_Pose_Estimation_CVPR_2026_paper.html)). Наш single-view scope отличен, но этот baseline задаёт ceiling для дополнительной информации.
- **Fixed Reality, Diffused Possibility: Disentangling Stochastic and Deterministic Latent for Cluttered Grasping** принят на ECCV 2026. Уже доступны [project page](https://hritam-98.github.io/SplitDiffGrasp/) и [официальный code/README](https://github.com/hritam-98/SplitDiffGrasp): fixed multi-view geometric latent + stochastic global latent, global/local conditional diffusion, point-wise affordance/grasp fields, Harmonic Grasp Field и GraspVAE. Публичная objective — diffusion losses + harmonic regularizer + ELBO; ни nested sensor pairs, ни outcome posterior, ни tower/CMR там не заявлены. Значит эта работа занимает «stochastic grasp field» и является обязательным сильным baseline, но по доступным материалам не занимает узкий physical-garbling claim. Ссылка `Paper` на project page пока ведёт на homepage, поэтому финальный camera-ready PDF всё ещё надо проверить.

### 2.5 Benchmark gap

Текущие benchmarks не изолируют нужную переменную:

- GraspNet-1B: реальные cluttered scenes, но окклюзия не является controlled axis, один и тот же target не сравнивается под вложенными blocker masks;
- NeuGraspNet: hard camera viewpoint и clutter, не отдельное внешнее препятствие при фиксированной camera/target geometry;
- ReOcS: controlled severity для reconstruction, не dense grasp outcomes / executed success curves;
- UNCLE: controlled external occlusion, но один узкий strawberry distribution и strong task-specific filters.

Нужен paired benchmark, где один и тот же world state наблюдается через физически вложенные occlusion channels.

---

## 3. Последовательно исследованные и отброшенные направления

### Кандидат A — probabilistic completion + LCB

**Интуиция:** sample multiple shapes, проверить grasp на каждой, выбрать lower confidence bound.

**Почему должен работать:** Lundell и UNCLE дают прямое положительное evidence.

**Почему отвергнут как main contribution:** формализация уже существует почти буквально; ZeroGrasp усиливает completion большой synthetic prior; novelty сведётся к новой completion architecture или новому aggregation heuristic. Это соответствует нежелательному сценарию «скомпоновать несколько robotic pipelines».

### Кандидат B — сильная occlusion augmentation и feature invariance

**Интуиция:** заставить clean и occluded observations иметь близкие features/grasps.

**Критическая математическая ошибка:** правильный Bayesian predictor не обязан быть invariant. Если reveal показывает выступ, hole или вторую опорную поверхность, posterior и лучший grasp должны измениться. Pairwise loss

`||f(O_coarse) - f(O_fine)||²`

стирает полезный information gain и ведёт к conservative/common-denominator predictor. Нужна equality **в условном ожидании**, а не для каждой пары.

**Решение этого провала** стало частью финальной идеи: tower consistency через conditional moment restriction.

### Кандидат C — только grasps, у которых обе contact patches видимы

**Плюс:** геометрически интерпретируемая safety certificate; не требуется prior скрытой формы.

**Минусы:** резко падает candidate coverage; для фронтальной камеры одна из antipodal surfaces часто невидима даже без внешнего blocker; метод сознательно отказывается от главной задачи обобщения. Подходит как high-precision baseline, но не как потенциальный общий SOTA.

### Кандидат D — conformal risk control / selective prediction

**Плюс:** finite-sample control ожидаемого monotone loss возможен ([Conformal Risk Control, ICLR 2024](https://openreview.net/pdf?id=33XGfHLtZg)); decision-theoretic conformal sets для risk-averse agents развиты дальше ([ICML 2025](https://proceedings.mlr.press/v267/kiyani25a.html)).

**Почему не main:** это post-hoc calibration, не representation/learning solution; exact conditional coverage по каждой степени окклюзии без assumptions недостижима, а marginal guarantee может скрыть failures именно в severe-occlusion stratum. UNCLE уже занимает простую abstention/LCB историю. Conformal calibration полезна как secondary module после появления real calibration set.

### Кандидат E — stochastic grasp/affordance field без дополнительной структуры

**Плюс:** перенос uncertainty из geometry space в task space, меньше вычислений.

**Почему недостаточно:** deterministic grasp functions существуют давно ([Johns et al. 2016](https://arxiv.org/abs/1608.02239)); GIGA — implicit affordance field; Neural Diffusion Processes дают function distributions ([ICML 2023](https://proceedings.mlr.press/v202/dutordoir23a.html)); ECCV 2026 work уже заявляет uncertainty-aware stochastic grasp fields. Нужна новая структурная аксиома и новый learnable objective.

### Выживший кандидат

**Task-outcome stochastic process + observation-filtration consistency + conditional-moment learning.** Его novelty не в словах «uncertainty» или «field», а в том, что posteriors под разными информационными каналами образуют одну coherent measure-valued process.

---

## 4. Финальная broad idea: learning decision-sufficient posteriors over outcome functions

### 4.1 Мир, наблюдение и grasp outcome field

Пусть `X` — полный, но неизвестный world state target для одного grasp primitive. Он включает только то, что определяет локальный grasp outcome; не требуется явный scene SDF.

Grasp:

$$
g=(R,t,w)\in\mathcal G=SE(3)\times[0,w_{max}].
$$

Полный state индуцирует функцию

$$
U_X:\mathcal G\to[0,1],\qquad
U_X(g)=P(Y_g=1\mid X,g),
$$

где `Y_g` — успех closure + короткого lift primitive. В полностью deterministic simulator `U_X(g)` может быть robust quality или Monte-Carlo success frequency под малыми execution perturbations.

Из RGB-D получено наблюдение

$$
O=C(X,B,\varepsilon),
$$

где `B` — ray-consistent foreground blocker, `ε` — sensor noise. Искомый объект — не `X` и не shape posterior, а pushforward posterior

$$
\Pi_O=\mathcal L(U_X\mid O),
$$

то есть распределение случайной функции на grasp space.

### 4.2 Почему это decision-sufficient, а shape — избыточна

**Proposition 1 (task sufficiency).** Если loss любого допустимого решения зависит от полного state только через grasp outcome field,

$$
L(g,X)=\widetilde L(g,U_X),
$$

то любой Bayes-optimal grasp может быть найден только из `Π_O`; posterior `P(X|O)` содержит лишнюю для этого решения информацию.

**Proof sketch.** Conditional Bayes risk равен

$$
E[L(g,X)\mid O]
=\int \widetilde L(g,u)\,d\Pi_O(u).
$$

Минимизация не требует другого functional от `P(X|O)`. □

Это не утверждает, что geometry бесполезна вообще; оно утверждает, что full-shape reconstruction — не минимальная representation для данного primitive.

### 4.3 Почему нужен posterior над целой функцией, а не независимые confidence scores

Для query set `G={g_1,…,g_M}` нужна joint distribution

$$
\Pi_O^G=\mathcal L((U_X(g_1),\ldots,U_X(g_M))\mid O).
$$

Скрытая ручка чашки, неизвестная толщина или симметрия одновременно меняют качества многих соседних grasps. Независимые Bernoulli/Beta scores не сохраняют эту correlation. Без неё нельзя корректно оценить:

- disagreement о том, **какой** grasp лучший;
- posterior regret относительно лучшего grasp-а в каждой plausible world;
- probability, что хотя бы один candidate надёжен;
- winner's curse при выборе maximum среди сотен noisy scores.

### 4.4 Внешняя окклюзия как Blackwell garbling

Строим nested observation pair:

$$
X\rightarrow O^+\rightarrow O^-,
$$

где `O⁻` получен из `O⁺` дополнительным ray-consistent blocker и потому менее информативен в Blackwell sense. Эквивалентно, `\mathcal F^-\subseteq\mathcal F^+`.

Для любого bounded functional `φ(U_X)` истинный posterior moment

$$
M_\phi(O)=E[\phi(U_X)\mid O]
$$

подчиняется tower property:

$$
E[M_\phi(O^+)\mid O^-]=M_\phi(O^-). \tag{1}
$$

Это корректная «occlusion consistency». Равенство `M(O⁺)=M(O⁻)` для каждой пары неверно.

### 4.5 Consequence: value of information

Для posterior-mean utility `m_O(g)=E[U_X(g)|O]` определим

$$
V(O)=\max_{g\in\mathcal G}m_O(g).
$$

Из (1) и convexity `max`:

$$
E[V(O^+)\mid O^-]\ge V(O^-). \tag{2}
$$

То есть более информативный канал не может в среднем уменьшить достижимую Bayes utility. Это не pointwise monotonicity: конкретный reveal может показать плохую новость. (2) даёт новый diagnostic для grasp models — **Blackwell violation rate**.

### 4.6 Irreducible ambiguity

Если две формы имеют одинаковое распределение `O⁻`, но непересекающиеся множества хороших grasps, никакой deterministic predictor не может быть корректен для обеих. Category label меняет prior mixture weights, но не создаёт отсутствующую информацию. Правильный output в таком случае — multimodal `Π_O`, а forced action выбирается по decision rule; при разрешённом abstention модель должна честно показать отсутствие надёжного решения.

---

## 5. Learnable formalization: OC-GOP

### 5.1 Finite-query neural process

Для любого unordered query set `G` модель задаёт

$$
p_\theta(u_G\mid O,G)
=\int \prod_{i=1}^{M}
p_\psi(u_i\mid g_i,E_\theta(O),z)\;
p_\theta(z\mid O)\,dz. \tag{3}
$$

- `Eθ(O)` — sparse ray-aware point encoder;
- `z` — low-dimensional stochastic latent, общий для всех grasps и потому несущий correlated task ambiguity;
- decoder query-conditioned и применим к любому числу grasp poses.

Shared latent + pointwise decoder дают permutation exchangeability и marginal consistency по query set по построению: удаление части queries не меняет marginal остальных. Это дешевле diffusion по dense 3D shape.

Для multimodal `p(z|O)` нужен conditional normalizing flow или короткий rectified flow в 16–64D latent. Простой diagonal Gaussian — обязательный ablation, но не предпочтительная финальная модель.

### 5.2 Task-field autoencoding вместо shape autoencoding

На полном training object:

1. sample diverse grasp query set `G_X`;
2. получить simulator/analytic outcomes `u_X(G_X)`;
3. Set Transformer teacher `q_φ(z|{(g_i,u_i)})` кодирует **outcome field**, а decoder восстанавливает utility на held-out grasp queries;
4. partial-observation prior `p_θ(z|O)` учится совпадать с teacher latent distribution.

Две разные формы с одинаковым grasp outcome field намеренно могут иметь один latent. Это task quotient shape space, а не скрытая попытка восстановить mesh.

### 5.3 Вход без полного scene SDF

В target-centered crop используются три типа sparse tokens:

- visible target points с xyz/rgb и confidence;
- foreground blocker points только в локальном target crop;
- camera-ray state/depth tokens: target hit, blocker hit, free/no-return.

Опциональный category embedding `c` включается с class dropout: одна модель поддерживает `c=known` и `c=unknown`. Это позволяет измерить, сколько даёт class prior, не превращая его в обязательное условие.

Предпочтителен SE(3)-equivariant point transformer либо canonical gripper-frame queries. Exact equivariance снижает объём augmentation и не является главным novelty claim.

### 5.4 Candidate proposal

OC-GOP — прежде всего evaluator/belief model, но candidate recall нельзя оставить скрытой переменной. Практический union:

$$
G=G_{vis}\cup G_{prior}.
$$

- `G_vis`: быстрые GSNet/AnyGrasp-style proposals, anchored на видимой поверхности;
- `G_prior`: lightweight conditional `SE(3)` flow, способный предлагать center/orientation, не anchored строго в видимом point.

Оба rank-ятся одним OC-GOP. В экспериментах обязательно отдельно сообщать oracle recall `max_{g∈G} U_X(g)`, иначе улучшение evaluator-а невозможно отделить от proposal luck.

### 5.5 Supervised proper objective

Базовый loss на случайных query sets:

$$
\mathcal L_{pred}
=-E\log p_\theta(u_X(G)\mid O,G), \tag{4}
$$

или proper energy/kernel score для implicit samples. На всех occlusion levels есть один и тот же full-state outcome target, но разные условные distributions.

### 5.6 Новая часть: distribution-level tower constraint

Для feature map `Φ` joint utility vector определим predicted posterior embedding

$$
\mu_\theta(O,G)=E_{u\sim p_\theta(\cdot|O,G)}[\Phi(u)].
$$

`Φ` включает первые два момента и random Fourier features characteristic Gaussian kernel. Для nested pair:

$$
\Delta_\theta=
\mu_\theta(O^+,G)-\mu_\theta(O^-,G).
$$

Истинная tower property эквивалентна conditional moment restriction

$$
E[\Delta_\theta\mid O^-,G]=0. \tag{5}
$$

Нельзя минимизировать `||Δ||²` по каждой паре: это ошибочный invariance loss. Вместо этого решается adversarial GMM:

$$
\mathcal L_{tower}(\theta)=
\max_\omega
\left{
2E[h_\omega(O^-,G)^\top\Delta_\theta]
-E\|h_\omega(O^-,G)\|^2
\right}. \tag{6}
$$

Rich critic ищет любую coarse-observation subgroup, в которой residual имеет ненулевое условное среднее. Такой приём опирается на conditional moment literature, но сама restriction выведена из observation filtration, а не из instrumental variables: [Adversarial GMM](https://arxiv.org/abs/1803.07164), [Minimax Estimation of Conditional Moment Models, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/8fcd9e5482a62a5fa130468f4cf641ef-Abstract.html), [Kernel Conditional Moment Test, UAI 2020](https://proceedings.mlr.press/v124/muandet20a.html).

Стабильная альтернатива (6) — closed-form RKHS maximum moment restriction по minibatch Gram matrix. В первой реализации стоит начать с kernel version, затем сравнить с neural critic.

Итог:

$$
\mathcal L
=\mathcal L_{pred}
+\beta\mathcal L_{latent}
+\lambda\mathcal L_{tower}. \tag{7}
$$

`L_pred` не даёт trivial constant solution; `L_tower` связывает статистику разных occlusion severities.

#### Важная асимптотическая оговорка

При бесконечном количестве размеченных данных на **каждом** уровне окклюзии, realizable model class и глобальной оптимизации строго proper `L_pred` уже восстанавливает истинные conditional laws; tower property тогда выполняется автоматически. Поэтому (6) нельзя продавать как новый population target или как способ идентифицировать то, что proper supervision принципиально не идентифицирует. Его честная роль:

1. finite-sample structural regularization, связывающая редкие severe-occlusion bins с более информативными наблюдениями;
2. обучение на paired nested observations без real grasp labels;
3. diagnostic misspecification: отдельно измерить, нарушает ли learned posterior законы обновления информации;
4. inductive bias при shared-capacity model, где разные occlusion levels конкурируют за параметры.

Это делает empirical sample-efficiency claim центральным и фальсифицируемым: при matched labeled sample size и capacity tower-CMR обязан улучшить severe-bin proper score/ranking, иначе его вклад остаётся в основном диагностическим.

#### Projection bound, который можно доказать без сильных допущений

Пусть `M^+=E[Phi(U)|O^+]`, `M^-=E[Phi(U)|O^-]`, fine teacher даёт `\widehat M^+`, а coarse student — `\widehat M^-`. Тогда

$$
\begin{aligned}
\|\widehat M^- - M^-\|_2
&\le
\|\widehat M^- - E[\widehat M^+\mid O^-]\|_2
+\|\widehat M^+-M^+\|_2. \tag{8}
\end{aligned}
$$

Первый член — population conditional-moment residual, который аппроксимирует rich MMR/critic; второй — ошибка fine teacher. Это следует из triangle inequality, tower property и `L2`-contractivity conditional expectation. Для finite candidate set ошибка mean-utility `\sup_g|\widehat m^-(g)-m^-(g)|\le\epsilon` дополнительно даёт не более `2\epsilon` regret у greedy grasp. Это ещё не finite-sample theorem для neural estimator: к (8) надо добавить critic approximation, empirical-process и optimization errors. Именно такой theorem, а не повторное доказательство tower property, имеет шанс быть ICLR-level вкладом.

### 5.7 Semi-supervised real adaptation

Преимущество (5): для real nested blocker pairs grasp labels не нужны. Fine model/EMA teacher выдаёт moments, coarse model учится их conditional projection. Теоретическая опора:

$$
\|E[\widehat M^+\mid O^-]-M^-\|_2
\le \|\widehat M^+-M^+\|_2,
$$

поскольку conditional expectation — `L2` contraction. То есть projection хорошего fine predictor-а не усиливает его mean-square error. Это потенциально сильнее обычного sim-to-real augmentation story.

Но CMR-loss сам по себе не гарантирует, что конечная сеть реализовала точную projection: нужны достаточно богатый critic, coverage paired observations и контроль optimization error. В экспериментах поэтому надо показывать не только downstream success, но и estimated conditional residual на held-out blocker mechanisms.

---

## 6. Inference и decision rule

Для `M` candidates и `K` latent samples получаем matrix

$$
U^{(k)}_i=U^{(k)}(g_i),
\qquad K\times M.
$$

### Forced-attempt режим

Чистый Bayes rule для success rate:

$$
g^*_{mean}=\arg\max_i \frac1K\sum_k U^{(k)}_i. \tag{8}
$$

Чтобы joint posterior реально участвовал в выборе, вводится sample-wise regret

$$
r^{(k)}_i=\max_j U^{(k)}_j-U^{(k)}_i. \tag{9}
$$

Risk-sensitive вариант:

$$
g^*=\arg\max_i
\left(E[U_i]-\lambda_r\,CVaR_\alpha(r_i)\right). \tag{10}
$$

В отличие от per-grasp LCB, regret учитывает correlations: если все grasps одновременно ухудшаются на трудной форме, relative ambiguity мала; если разные hypotheses требуют несовместимых grasps, tail regret велик.

### Selective режим

Если abstention допустим, попытка совершается при

$$
P(U(g^*)>\tau\mid O)>1-\delta.
$$

После real calibration можно добавить conformal risk control, но это secondary contribution.

### Blocker collision

Observed blocker не надо «воображать». После ranking выполняется быстрый deterministic collision check gripper closing/small local approach volume против raw blocker PCD. Это post-filter, не learned long-horizon feasibility model.

---

## 7. Почему framework потенциально эффективнее completion pipelines

Это гипотеза, которую надо проверять, а не готовый факт. Косвенные основания:

1. **Task auxiliary geometry помогает.** GIGA, NeuGraspNet и Local Occupancy показывают, что grasp-local geometric information улучшает результат.
2. **Full observation остаётся заметным upper bound.** Local Occupancy: `53.84 AP` против `58.78` completed observation.
3. **Deterministic completion недостаточна.** Lundell и UNCLE показывают пользу uncertainty across completions.
4. **Joint outcome predictor может быть дешевле.** ZeroGrasp делает high-resolution octree reconstruction в `5 FPS`; GraspLDM требует `0.75s` для 100 DDIM grasp samples в сообщённой конфигурации. OC-GOP семплирует только low-D task latent и batched decoder values. Целевой latency надо измерить, а не заявлять заранее.
5. **Large data доступны.** ACRONYM содержит 17.7M parallel-jaw grasps на 8,872 objects / 262 categories ([ACRONYM](https://arxiv.org/abs/2011.09584)); ZeroGrasp-11B демонстрирует масштаб физически validated annotations. Нужные outcome fields можно получить без real trials на каждом query.
6. **Function-distribution tools зрелы.** CNP задаёт conditional stochastic processes ([ICML 2018](https://proceedings.mlr.press/v80/garnelo18a.html)); Neural Diffusion Processes моделируют non-Gaussian correlated functions ([ICML 2023](https://proceedings.mlr.press/v202/dutordoir23a.html)); Markov Neural Processes сохраняют exchangeability/consistency ([NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7749f9c0d5ff109231be21e910a3ced2-Abstract-Conference.html)). Наша задача добавляет другую consistency axis — между information channels.

---

## 8. Dataset / benchmark design

Рабочее имя: **NestedOcclusion-Grasp (NOG)**.

### 8.1 Synthetic paired construction

Для каждого `(mesh, pose, camera)`:

1. render clean noisy RGB-D;
2. выбрать foreground blocker family;
3. из одного fine render последовательно overlay blocker так, чтобы masks были вложены;
4. severity измерять не параметром генератора, а долей исчезнувшей **видимой target surface** относительно unblocked single view;
5. сохранить levels, например `0, 10, 25, 40, 55, 70, 85%`;
6. full mesh используется только для labels/evaluation;
7. физический blocker сохраняется отдельным local PCD для collision check.

Нужны разные blocker families: planar shelf lip, box edge, цилиндр, irregular household obstacle. Train/test split должен разделять и target meshes, и blocker meshes.

### 8.2 Real paired capture

Target и wrist camera фиксированы; calibrated opaque blocker перемещается по направляющей между уровнями. Это даёт зарегистрированные nested observations без grasp execution на каждом кадре. Для небольшого subset выполняются short-lift trials.

### 8.3 Два prior режима

- **category unknown:** общий shape distribution;
- **category known:** class embedding, включая held-out instances;
- отдельный **wrong/shifted category** stress test, чтобы измерить вред ошибочного YOLO prior.

### 8.4 Noise protocol

- depth quantization/dropout;
- edge flying pixels;
- pose calibration perturbation;
- RGB/segmentation errors;
- sim-to-real sensor model.

Важно: sensor noise должен сначала появляться в fine observation, после чего coarse observation получается Markov garbling. Независимый повторный шум разрушает точную nested relation и должен быть отдельным robustness test.

---

## 9. Evaluation protocol

### 9.1 Основные outcome metrics

- executed grasp success vs occlusion severity;
- AUC этой кривой;
- forced-attempt success (главная метрика лабораторного режима);
- success–coverage curve и AURC для selective режима;
- oracle candidate recall;
- real latency, memory, number of posterior samples.

### 9.2 Probabilistic metrics

- NLL / proper energy score joint outcomes;
- Brier/ECE для per-grasp success;
- calibration по severity bins;
- correlation error между nearby/competing grasps;
- posterior mode coverage на toy ambiguous shapes.

### 9.3 Новые coherence metrics

Для random `φ,h`:

$$
TVio=\left\|E[h(O^-)(M_\phi(O^+)-M_\phi(O^-))]\right\|;
$$

- kernel maximum moment tower violation;
- law-of-total-variance residual;
- Blackwell value violation: положительная часть
  `V(O⁻) - E[V(O⁺)|O⁻]`;
- expected posterior spread vs severity — только diagnostic, не pointwise constraint.

### 9.4 Baselines

Минимальный сильный набор:

1. Contact-GraspNet;
2. GSNet / AnyGrasp;
3. GraspLDM или Grasp Diffusion Network;
4. GIGA;
5. NeuGraspNet;
6. Local Occupancy-Enhanced Grasping;
7. ZeroGrasp;
8. Hybrid EBM+ICP+SVGD 2026, если code доступен;
9. deterministic occlusion-augmented evaluator;
10. deep ensemble / MC-dropout grasp scores;
11. probabilistic shape completion + mean/LCB по Lundell/UNCLE principle;
12. visible-two-contact certified baseline;
13. OC-GOP без tower loss;
14. oracle full-shape grasp planner.

SplitDiffGrasp / ECCV 2026 Fixed Reality уже имеет public code и обязателен как baseline; после появления camera-ready надо сверить точные protocol и objective.

### 9.5 Ключевые ablations

- stochastic outcome process → deterministic mean;
- joint latent → independent per-grasp uncertainty;
- tower CMR off;
- incorrect pairwise invariance вместо CMR;
- mean-only CMR vs characteristic distribution features;
- kernel critic vs neural critic;
- ray/blocker tokens off;
- class known / dropped / wrong;
- `G_vis` vs `G_vis∪G_prior`;
- mean rule (8) vs regret-CVaR (10);
- number of posterior samples;
- synthetic labeled vs added unlabeled real nested pairs.

---

## 10. Falsification tests

Идею надо отвергнуть или серьёзно упростить, если выполнится любое из следующего:

1. После фиксации одного candidate set OC-GOP не превосходит deterministic evaluator: значит stochastic field не даёт decision value.
2. `L_tower` улучшает coherence metrics, но не calibration/SR severe bins: consistency эстетична, но practically irrelevant.
3. Candidate oracle recall падает быстрее evaluator accuracy: bottleneck — proposal, тогда paper должен честно сменить фокус.
4. Shape-completion samples + общий evaluator при том же compute дают лучшую joint calibration и SR: task quotient не окупается.
5. Pairwise outcome correlations не воспроизводятся latent process; regret-CVaR хуже mean.
6. Gains исчезают при held-out blocker geometry или real ray noise: модель выучила occluder textures/positions.
7. Category-conditioned gains оборачиваются catastrophic errors при wrong class.
8. При severe ambiguity модель остаётся overconfident на двух-shape counterexample: posterior learning не решило исходную проблему.
9. Финальный ECCV 2026 PDF добавит эквивалентную observation-garbling consistency formulation, которой пока нет в project page/code objective.

---

## 11. Novelty audit

### Claim, который пока защищаем

> Насколько показывает поиск на 25 августа 2026 года, это первый grasp-learning framework, который (i) выбирает decision-sufficient posterior над joint outcome function вместо posterior над shape или distribution хороших poses и (ii) учит его Bayesian coherence по **физически вложенным RGB-D garblings** через conditional moment restrictions, включая обучение на real blocker pairs без grasp labels.

Это узкий, составной и пока provisional claim. Нельзя расширять его до «первого conditionally consistent neural process»: свежие general-ML работы уже анализируют или конструируют conditioning-consistent neural processes. Перед submission claim должен пройти библиографическую проверку человеком и обновление после выхода camera-ready PDF ближайшей ECCV 2026 работы.

### Что не заявлять

- первый grasp field;
- первый stochastic grasp model;
- первый uncertainty-aware grasp planner;
- первый grasping under occlusion;
- первый use of completion uncertainty;
- первый use of diffusion/flow for grasps.

### Различия с ближайшими работами

| Работа/линия | Что представляет | Как обрабатывает missing geometry | Чего нет относительно OC-GOP |
|---|---|---|---|
| GraspLDM / GDN / IGD | distribution хороших grasp poses | conditioning на partial PCD | samples — action multimodality, не posterior joint outcome field; нет nested consistency |
| GIGA / NeuGraspNet | deterministic implicit geometry + affordance | learned scene/local completion | point prediction; нет calibrated task posterior |
| Local Occupancy | local occupancy point estimate | grasp-local completion | нет uncertainty и cross-occlusion coherence |
| ZeroGrasp | probabilistic octree latent + shape + grasps | explicit occlusion fields, reconstruction | full geometry target; нет posterior over utility functions/tower |
| Lundell / UNCLE | shape samples + grasp aggregation | MC-dropout completions | дорогой geometry intermediary; mostly marginal mean/LCB; нет function-level coherence |
| FFHFlow | uncertainty-aware dexterous grasp flow | latent uncertainty from partial PCD | другая hand/task; нет nested observation law |
| Fixed Reality, Diffused Possibility / SplitDiffGrasp (ECCV 2026) | fixed multi-view latent + global/local diffusion + point-wise affordance/grasp fields + GraspVAE | partially observed clutter, obstacle-aware harmonic field | project/code objective не содержит decision-sufficient outcome posterior, physical nested pairs, Blackwell/tower law или CMR; camera-ready PDF ещё обязателен для проверки |
| Neural Processes / NDP / MNP | general distributions over functions | context conditioning | нет structured observation lattice и decision-utility application |
| Score-Based Neural Processes (under review) | expressive correlated stochastic process; заявляет conditional consistency через conditional diffusion guidance | conditioning на observed function context | серьёзный general-method prior art; его notion добавляет observed `(x,y)` context, тогда как здесь меняется exogenous sensor information `O^+ -> O^-`, а grasp outcomes остаются latent |
| Conditioning Consistency Gap in CNPs (TMLR submission, 2026) | KL-gap между re-encoding enlarged context и conditioning joint prediction; bound `O(1/n^2)` | few-shot function observations | показывает, что термин и общая consistency-проблема уже заняты; не изучает Blackwell garblings, task utility или unlabeled paired sensor adaptation |
| Information-martingale / probability-path work | coherent evolution вероятностных прогнозов по мере информации | временно поступающие signals | tower/martingale interpretation не нова сама по себе; новыми должны быть grasp outcome representation, physical garbling design, estimator и evidence |
| AGMM/MMR | conditional moment estimation | не про perception | даёт optimization machinery, но не постановку/теоремы OC-GOP |

#### Две разные consistency, которые reviewer почти наверняка смешает

В NP literature conditioning consistency обычно спрашивает: совпадёт ли prediction после добавления уже наблюдённой пары функции `(g,u_g)` в context с условным распределением, полученным из прежнего joint process? В OC-GOP не раскрывается ни одного grasp outcome. Меняется sensor experiment: `O^+` проходит через известное/изучаемое garbling kernel и даёт `O^-`; latent outcome function та же. Нужен commuting diagram

`full state X -> O^+ -> O^-` и `X -> U`,

а ограничение сравнивает posterior laws `Law(U|O^+)` и `Law(U|O^-)` после усреднения по reveal, совместимым с `O^-`. Это observation-filtration coherence, не query marginalization и не обычное context augmentation. Тем не менее обе линии опираются на одну вероятностную основу, поэтому architecture-level novelty нельзя заявлять без прямого сравнения или использования ScoreNP как backbone.

### Риск «obvious combination»

Он реален: reviewer может сказать «Neural Process + consistency regularizer + grasping». Для защиты нужны одновременно:

1. theorem о decision sufficiency;
2. ясное различие query consistency и observation-filtration consistency;
3. доказательство, что naive pairwise invariance неправильна;
4. CMR/RKHS objective как principled estimator tower law;
5. semi-supervised projection result для unlabeled real nested pairs;
6. новый controlled benchmark;
7. gains не только над direct models, но и над ZeroGrasp / uncertain completion при compute matching;
8. хотя бы один controlled non-robotic/synthetic experiment с известным истинным posterior, показывающий generality метода.

Дополнительный риск второго прохода: в population limit tower constraint следует из правильных conditional laws и потому асимптотически избыточен при полном proper supervision. Следовательно, paper должен доказать и показать **sample-efficiency / unlabeled-transfer advantage**, а не ограничиться красивой coherence metric. Наиболее защищаемая теория — decomposition (8) плюс finite-sample control critic/teacher/optimization errors; наиболее защищаемая эмпирика — labeled-data scaling curve и transfer на новый физический blocker kernel.

Без пунктов 4–8 работа, вероятно, будет хорошей robotics paper, но ICLR novelty останется borderline.

---

## 12. ICLR acceptance audit

Последняя доступная официальная формулировка ICLR говорит, что review определяет, приносит ли работа sufficient value и new knowledge; проверяются motivation/literature placement, correctness/rigor, support of claims и significance. SOTA сам по себе не обязателен ([ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide)). Robotics, probabilistic methods, UQ, structured prediction и learning theory входят в scope ([ICLR 2026 CFP](https://iclr.cc/Conferences/2026/CallForPapers)).

### Предварительная оценка

- **Specific question:** strong — как учить coherent task posterior при structured information loss.
- **Motivation:** strong — несколько независимых линий показывают partial-view degradation и пользу missing-geometry uncertainty.
- **Technical originality:** medium, потенциально medium-high только при theorem/sample-efficiency result и отсутствии эквивалентной observation-garbling objective в полном тексте ECCV 2026; общая conditioning consistency уже занята соседними NP papers.
- **Correctness:** потенциально strong; propositions просты и проверяемы, но estimator analysis ещё нужно написать.
- **Empirical rigor:** пока отсутствует; главный риск.
- **Significance:** выше robotics niche, если framework сформулирован для arbitrary utility fields under garblings и показан хотя бы на одном дополнительном controlled domain.
- **Clarity/reproducibility:** feasible; NOG generator и evaluation должны быть released.

### Вердикт

**Conditional go.** Идея достойна prototype и полноценного novelty search. На текущем этапе это не «готовая ICLR paper» и не обещание SOTA. Наиболее вероятные причины reject: apparent combination, отсутствие прямого сравнения с public SplitDiffGrasp, слабая реальная выборка, coherence improvements без success gains.

### Policy note

ICLR 2026 Author Guide требует раскрывать significant LLM usage в research ideation/writing ([Author Guide](https://iclr.cc/Conferences/2026/AuthorGuide)). Поскольку этот документ существенно использует LLM для ideation, при подаче надо честно описать точную роль модели; ответственность за novelty, proofs, citations и experiments остаётся у авторов.

---

## 13. Минимальный путь к paper, в порядке снятия риска

### Milestone 1 — дешёвое опровержение/подтверждение gap

- ACRONYM subset, 200–500 held-out objects;
- nested synthetic blocker renderer;
- AnyGrasp/Contact/GraspLDM baseline curves vs removed-visible-surface ratio;
- separate candidate recall and ranking loss;
- проверить, даёт ли внешний blocker дополнительное падение при фиксированном clean view.

### Milestone 2 — deterministic mean + tower CMR

Пока без stochastic process. Учить `m(O,g)` proper supervised loss и mean tower CMR. Если severe-bin calibration/ranking не улучшаются, full method под вопросом.

### Milestone 3 — joint latent outcome process

- field autoencoder;
- conditional flow latent;
- energy score / held-out queries;
- проверить correlation и ambiguous-pair toy dataset.

### Milestone 4 — strong comparisons

- ZeroGrasp / Local Occupancy / uncertain completion;
- matched candidate sets и matched compute;
- full ECCV 2026 comparison;
- unlabeled real nested-pair adaptation.

### Milestone 5 — real robot

- минимум 20–30 objects × несколько blocker levels × повторения;
- primary paired statistical test на success difference;
- заранее определить failure taxonomy только для анализа, не делать causal failure modes центральной постановкой.

---

## 14. Paper skeleton

Возможный title:

> **Grasp What You Cannot See: Blackwell-Consistent Outcome Processes under Structured Occlusion**

Contributions:

1. Decision-sufficient grasp outcome posterior вместо geometry completion.
2. Observation-filtration consistency law и value-of-information diagnostics.
3. Conditional-moment/RKHS training objective, не collapsing pairwise invariance.
4. NOG benchmark с вложенной external occlusion.
5. SOTA или конкурентный forced-attempt success при существенно меньшем inference compute, плюс calibrated selective mode.

Main figure:

`full state → same hidden outcome field → nested RGB-D observations → coherent posterior function samples → risk-aware grasp`.

Главная визуализация должна показать две одинаковые видимые partial surfaces, два разных скрытых объекта, несовместимые optimal grasps и то, как posterior modes сужаются при reveal.

---

## 15. Источники general-ML inspiration

- Blackwell order сравнивает information channels через максимальную ожидаемую utility и garbling ([Rauh et al.](https://arxiv.org/abs/1701.07602)).
- Conditional Neural Processes прямо параметризуют conditional stochastic process ([ICML 2018](https://proceedings.mlr.press/v80/garnelo18a.html)).
- Neural Diffusion Processes моделируют correlated non-Gaussian function distributions и обсуждают exchangeability/marginal consistency ([ICML 2023](https://proceedings.mlr.press/v202/dutordoir23a.html)).
- Markov Neural Processes строят expressive process через consistency-preserving transition operators ([NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7749f9c0d5ff109231be21e910a3ced2-Abstract-Conference.html)).
- [Score-Based Neural Processes](https://openreview.net/pdf?id=rZzcaduYU1) заявляет correlated non-Gaussian samples, marginal и conditioning consistency через conditional diffusion guidance; на момент поиска это анонимная under-review версия, поэтому использовать её надо как серьёзный prior art, а не как установленный SOTA.
- [On the Conditioning Consistency Gap in Conditional Neural Processes](https://openreview.net/pdf/e4f9b81d0c5775228e4c6adad91729611ef789fb.pdf) (TMLR submission, 2026) формализует KL-gap между context update и probabilistic conditioning и получает `O(1/n^2)` rate для CNP при своих условиях. Это ближайшая терминологическая collision, хотя observation filtration в OC-GOP другая.
- [Probability Paths and the Structure of Predictions over Time](https://openreview.net/forum?id=5CKM8jCfEmM) (NeurIPS 2021) моделирует прогнозы как information martingale; [martingale audit Bayesian ICL](https://openreview.net/forum?id=z4YTZ01NT4) и более свежий [Martingale Score](https://openreview.net/forum?id=BfO6od6JD6) показывают, что martingale/tower law уже используется как diagnostic learned beliefs. Поэтому сама идея «проверять tower property» не является новой.
- Conditional moment restrictions обучаемы minimax/RKHS methods: [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/8fcd9e5482a62a5fa130468f4cf641ef-Abstract.html), [UAI 2020](https://proceedings.mlr.press/v124/muandet20a.html), [ICML 2022](https://proceedings.mlr.press/v162/kremer22a.html).
- Conformal Risk Control контролирует expected monotone loss и полезен как последующая calibration layer ([ICLR 2024](https://openreview.net/pdf?id=33XGfHLtZg)).

---

## 16. Открытые вопросы следующего research pass

1. Camera-ready PDF ECCV 2026 Fixed Reality work: project page и code уже проверены, но ссылка на paper пока не даёт PDF.
2. Стоит ли взять ScoreNP как backbone для query/context consistency и оставить contribution только в observation-garbling CMR, либо exact joint construction окажется слишком дорогой?
3. Какой proper score лучше для discontinuous grasp outcome fields: energy score, kernel score или discretized Bernoulli likelihood?
4. Достаточен ли один global latent для topology-changing multimodality или нужен hierarchical global/local latent?
5. Как генерировать `G_prior` без того, чтобы proposal distribution путалась с posterior uncertainty?
6. Какой физический outcome label лучше всего переносится sim-to-real: short-lift frequency, robust epsilon margin или их learned calibration?
7. Насколько class prior помогает после контроля за geometric similarity, и как безопасно отключать его при OOD?
8. Как оценить joint field calibration при невозможности исполнить сотни grasps на одном real object state?
9. Можно ли получить finite-sample bound между tower violation и excess grasp decision regret?
10. Нужна ли вторая general-ML task для ICLR, или достаточно synthetic known-posterior theorem experiment + полноценной robotics validation?

### Итог второго novelty pass

Второй проход **не опроверг** task-posterior + physical-garbling formulation, но существенно сузил её. Outcome-process representation остаётся содержательно отличной от shape completion и pose generation; controlled external-occlusion benchmark остаётся явным практическим gap. Однако tower law — классическое свойство, conditioning-consistent neural processes уже появляются, а при полной population supervision дополнительный tower loss избыточен. Поэтому окончательный ICLR bet теперь не «новая consistency сама по себе», а связка:

`decision-sufficient joint outcome posterior + physically nested sensor experiments + label-free conditional projection + finite-sample/transfer evidence`.

Если theorem и scaling/transfer experiments не материализуются, честный downgrade — robotics submission про benchmark и uncertainty-aware outcome modeling, без broad ICLR claim.

---

## 17. Текущая рекомендация лаборатории

Не начинать с большой stochastic architecture. Сначала построить NOG-lite и сделать две дешёвые проверки:

1. decomposition `candidate recall` vs `ranking error` по occlusion severity;
2. deterministic evaluator с правильным mean-tower CMR против обычной occlusion augmentation и ошибочного pairwise invariance.

Если tower objective не улучшает severe-occlusion ranking/calibration при фиксированных candidates, главная идея должна быть пересмотрена до затрат на full neural process. Если улучшает, следующий шаг — joint outcome latent и comparison с uncertainty-through-completion.

---

# Новая независимая итерация — 2026-08-25

## Итоговый кандидат

**FiberGrasp: Learning Necessary and Possible Grasp Sets from Occluded RGB-D**

Короткая формулировка идеи:

> Вместо восстановления единственной скрытой формы или оценки средней вероятности успеха FiberGrasp предсказывает непосредственно на многообразии параллельных захватов два вложенных множества: захваты, допустимые для **всех** полных сцен, неразличимых по текущему RGB-D, и захваты, допустимые хотя бы для **одной** такой сцены.

Это непрерывный action-space аналог нижнего и верхнего приближений rough-set theory. Невидимая геометрия не превращается в одну «наиболее вероятную» форму, а задаёт **волокно наблюдения** — множество физических миров, совместимых с тем, что камера действительно увидела. Метод обучает амортизированный оператор

$$
o\longmapsto\bigl(\mathcal G_-(o),\mathcal G_+(o)\bigr)
$$

без генерации полных форм во время инференса.

Предварительный вывод после поиска литературы: именно такая комбинация

1. observation-fiber semantics;
2. необходимых и возможных множеств в непрерывном пространстве захватов;
3. прямого equivariant implicit-предсказания этих множеств;
4. максимальности гарантируемого множества и границы наблюдательной идентифицируемости;
5. теста с парами геометрий, неразличимых для камеры,

не обнаружена в работах по robotic grasping. Это не доказательство абсолютного отсутствия статьи, поэтому novelty claim остаётся проверяемой научной гипотезой. Если убрать пункты 2, 4 или 5, работа становится слишком похожей на uncertainty-through-completion и теряет уровень ICLR.

## 1. Что подтверждает исходную гипотезу

Да, гипотеза о существенном падении качества при внешней окклюзии подтверждается прямыми результатами.

- [TARGO / Target-driven grasping under occlusions](https://arxiv.org/abs/2407.06168) вводит систематический benchmark целевого захвата при окклюзии. На [странице проекта](https://targo-benchmark.github.io/) показано, что обычные методы заметно деградируют с ростом occlusion rate; авторская модель снижает падение, но сама использует completed target shape.
- [TARGO-Net, финальная версия IJCV 2026](https://doi.org/10.1007/s11263-025-02716-9) объединяет сегментацию, AdaPoinTr completion и target-scene fusion. Поэтому «лучше достроить форму и затем предсказать grasp» уже является ближайшим сильным baseline, а не новым направлением.
- [Local Occupancy-Enhanced Grasping, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09354.pdf) показывает большой разрыв между partial и complete geometry и улучшает grasp detection локальной occupancy-реконструкцией. Значит, просто заменить глобальный completion локальным тоже недостаточно ново.
- [Generalizing 6-DoF Grasp Detection, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Generalizing_6-DoF_Grasp_Detection_via_Domain_Prior_Knowledge_CVPR_2024_paper.pdf) прямо отмечает невозможность надёжно вывести невидимые части новых объектов из одного partial point cloud и использует multi-view TSDF как способ получить дополнительную геометрию.

Практический вывод: постановка реальна, эффект окклюзии велик, но completion-направление уже плотно занято. Нужен иной объект предсказания.

## 2. Карта ближайшей литературы и занятые направления

### 2.1. Неопределённая реконструкция с последующей робастной оценкой

- [Robust Grasp Planning over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645): MC-dropout создаёт несколько полных форм, после чего один grasp оценивается на выборке форм.
- [PSSNet: Planar Shape Sampling Network](https://proceedings.mlr.press/v155/saund21a.html): несколько правдоподобных completions помогают именно в неоднозначных окклюдированных сценах.
- [Measuring Uncertainty in Shape Completion to Improve Grasp Quality, IROS 2025](https://arxiv.org/abs/2504.16183): uncertainty из множества completions штрафует grasp score и улучшает результат, но требует десятков проходов completion-модели.
- [UNCLE-Grasp, 2026](https://arxiv.org/abs/2601.14492): несколько MC-dropout completions, force-closure, conservative lower-confidence selection и возможность отказа.
- [Robotic Pick-and-Place With Uncertain Instance Segmentation and Shape Completion](https://pmc.ncbi.nlm.nih.gov/articles/PMC8022832/): сравнивает sampling-based uncertainty с прямым предсказанием стоимости ошибки.

Следствие: posterior sampling, variance penalty, CVaR, LCB и «наихудший grasp по K completions» сами по себе не являются новым вкладом.

### 2.2. Прямые генераторы и implicit-поля захватов

- [Contact-GraspNet](https://arxiv.org/abs/2103.14127) напрямую генерирует 6-DoF parallel-jaw grasps из depth/point cloud.
- [6-DoF GraspNet](https://openaccess.thecvf.com/content_ICCV_2019/papers/Mousavian_6-DOF_GraspNet_Variational_Grasp_Generation_for_Object_Manipulation_ICCV_2019_paper.pdf) использует вариационный генератор grasps из partial point clouds.
- [GraspLDM](https://arxiv.org/abs/2312.11243) моделирует распределение успешных SE(3)-захватов latent diffusion-моделью; среди причин ошибок остаются столкновения с невидимой частью объекта.
- [Implicit Grasp Diffusion](https://proceedings.mlr.press/v270/song25b.html) совмещает локальные implicit features и conditional diffusion для мультимодальной генерации grasps.
- [ShellGrasp](https://arxiv.org/abs/2109.06837) предсказывает camera-centric вход/выход через оболочку объекта и deterministic grasp map.
- [GraspGen-X, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Han_GraspGen-X_Cross-Embodiment_6-DOF_Diffusion-based_Grasping_CVPR_2026_paper.html) развивает diffusion grasp generation для разных grippers.

Следствие: «новый implicit grasp field», «diffusion grasps из partial cloud» или «contact-ray representation» недостаточны. Новизна FiberGrasp должна быть в **семантике поля и объекте обучения**, а не в том, что используется neural field.

### 2.3. Другие близкие направления

- [SpringGrasp, RSS 2024](https://arxiv.org/abs/2404.13532) строит дифференцируемую физическую метрику для compliant dexterous grasping из single-view данных, но решает другую задачу: многопальцевый захват и динамику пружинной податливости.
- [Task-Informed Grasping of Partially Observed Objects, RA-L 2024](https://pure-oai.bham.ac.uk/ws/portalfiles/portal/239318079/deFariasC2024Task-informed.pdf) комбинирует GPIS-реконструкцию и перенос функциональных областей, то есть остаётся reconstruction-first.
- [Grasp configuration planning for an underactuated three-fingered hand, 2018](https://doi.org/10.1016/j.mechmachtheory.2018.06.019) уже использует выражение rough-set mixed neural networks. Там rough sets предобрабатывают human-experience/taxonomy attributes формы и размера для выбора одной из шести конфигураций underactuated hand. Работа не строит lower/upper feasible sets, observation equivalence по RGB-D или continuous 6-DoF action field. Следовательно, нельзя заявлять «первое применение rough sets к grasping»; допустим только более узкий claim об observation-induced rough action sets.
- [Sample-Efficient Safety Assurances using Conformal Prediction](https://arxiv.org/abs/2109.14082) уже применяет conformal prediction к предупреждению ошибок grasping. Поэтому conformal calibration может быть только вспомогательным слоем, но не главным claim.

## 3. Отброшенные кандидаты

| Кандидат | Почему привлекателен | Почему отклонён |
|---|---|---|
| Probabilistic full shape + CVaR/LCB | Явно учитывает скрытую форму | Занято Lundell, PSSNet, IROS 2025 и UNCLE-Grasp; дорого во время инференса |
| Local occupancy около каждого gripper | Не нужен global SDF | Слишком близко к Local Occupancy ECCV 2024 и ShellGrasp |
| Direct success probability $p(y=1\mid o,g)$ | Просто и быстро | Усредняет observation-equivalent миры согласно частоте training prior; не отделяет unknowable от unlikely |
| Quantile/conformal grasp score | Даёт статистическую осторожность | LCB и conformal safety для grasping уже существуют; coverage не равна физической необходимости |
| Diffusion over robust grasps | Мультимодальность | Без новой target semantics это очередной conditional grasp generator |
| Moment/SOS-робастность скрытой массы и поверхности | Красивые гарантии | Слишком много жёстких предположений и переменных; плохо согласуется с noisy PCD и требованием эффективного обучения |
| Full-scene neural SDF + planning | Универсальная геометрия | Именно тот тяжёлый reconstruction/planning pipeline, который исключён постановкой |
| Causal taxonomy failures | Может объяснять ошибки | Явно исключена пользователем и не решает выбор grasp |

## 4. Формальная постановка FiberGrasp

### 4.1. Состояние, наблюдение и действие

Пусть $x\in\mathcal X_c$ — полное локальное физическое состояние: поверхность целевого объекта, препятствие, полка и параметры, необходимые для локальной проверки закрытия gripper. Индекс $c$ — известный класс; если класс неизвестен, берётся объединение допустимых supports.

Камера реализует оператор наблюдения

$$
\mathcal R:\mathcal X\rightarrow\mathcal O.
$$

Наблюдение $o$ содержит только доступные из wrist RGB-D данные: sparse target/obstacle points, known-free camera rays и uncertainty/noise metadata. Полная форма не создаётся на инференсе.

Для noiseless-случая все $x$, имеющие одно наблюдение, образуют equivalence class. Для реального сенсора используется tolerance fiber

$$
\mathcal F_\varepsilon(o,c)
=
\left\{
x\in\mathcal X_c:
d_{\rm obs}\bigl(\mathcal R(x),o\bigr)\leq\varepsilon
\right\}.
$$

Это множество всех сцен из выбранного support, которые камера не умеет различить с учётом шума и окклюзии.

Grasp:

$$
g\in\mathcal G
=
\bigl(SE(3)/C_2\bigr)\times[w_{\min},w_{\max}],
$$

где фактор $C_2$ учитывает 180-градусную симметрию parallel-jaw gripper.

### 4.2. Только локальная физическая допустимость

Вводится signed margin

$$
m(x,g)\in\mathbb R,
\qquad
m(x,g)\geq0
\iff
g\text{ локально допустим в }x.
$$

Margin агрегирует только:

- clearance пальцев и ладони при локальном открытии/закрытии;
- существование двух совместимых contact regions;
- antipodal/friction-cone margin;
- соответствие ширине gripper.

Arm reachability, полный approach trajectory, whole-cycle motion planning и большой lift trajectory намеренно не входят в определение. Малый подъём используется только как конечный экспериментальный критерий успешности.

Для полного состояния:

$$
\mathcal A(x)=\{g:m(x,g)\geq0\}.
$$

### 4.3. Необходимое и возможное множества захватов

Два экстремальных поля:

$$
m_-(o,g)
=
\inf_{x\in\mathcal F_\varepsilon(o,c)}m(x,g),
$$

$$
m_+(o,g)
=
\sup_{x\in\mathcal F_\varepsilon(o,c)}m(x,g).
$$

Они порождают

$$
\mathcal G_-(o)
=
\{g:m_-(o,g)\geq0\}
=
\bigcap_{x\in\mathcal F_\varepsilon(o,c)}\mathcal A(x),
$$

$$
\mathcal G_+(o)
=
\{g:m_+(o,g)\geq0\}
=
\bigcup_{x\in\mathcal F_\varepsilon(o,c)}\mathcal A(x).
$$

Интерпретация:

- $\mathcal G_-$: grasp допустим независимо от того, какая из неразличимых скрытых геометрий истинна;
- $\mathcal G_+$: grasp допустим хотя бы в одном совместимом мире;
- $\mathcal G_+\setminus\mathcal G_-$: область решений, которую single-view RGB-D принципиально не может разрешить без более сильного prior или нового observation.

Рабочая политика:

$$
g^\star
=
\arg\max_{g\in\mathcal G}m_-(o,g).
$$

Если максимум положителен, получен necessary grasp. Если он отрицателен, тот же $g^\star$ является maximin fallback — наименее хрупким решением в принятой модели support. Отказ можно измерять как диагностический режим, но он не является главным результатом.

## 5. Чем это отличается от вероятности успеха

Обычный score

$$
p(g\text{ succeeds}\mid o)
=
\int\mathbf 1[m(x,g)\geq0]p(x\mid o)\,dx
$$

зависит от того, как часто скрытые формы встречались в training distribution. Редкая, но observation-compatible геометрия может почти не повлиять на probability, хотя именно на ней grasp сталкивается с невидимой поверхностью.

FiberGrasp использует support:

$$
\mathcal F_\varepsilon(o,c)
=
\operatorname{supp}p(x\mid o,c)
\quad\text{с сенсорным tolerance}.
$$

Поэтому при изменении частот форм, но неизменном support, $\mathcal G_-$ и $\mathcal G_+$ не меняются. Это не «лучше откалиброванная вероятность», а другой объект предсказания.

Ограничение честно: гарантия верна только относительно заданного support и $\varepsilon$. Неизвестную форму вне support никакой single-view метод не может магически исключить.

## 6. Теоретический пакет

### Proposition 1. Наблюдательная невозможность

Пусть $\mathcal R(x_1)=\mathcal R(x_2)=o$ и

$$
\mathcal A(x_1)\cap\mathcal A(x_2)=\varnothing.
$$

Тогда любая детерминированная observation-only policy $g=\pi(o)$ терпит неудачу хотя бы в одном из $x_1,x_2$.

**Proof sketch.** Policy получает одинаковый input и обязана вернуть один и тот же grasp. Этот grasp не может принадлежать двум непересекающимся feasible sets.

Это формализует не просто sensor uncertainty, а предел идентифицируемости задачи.

### Theorem 1. Максимальность certifiable action set

$\mathcal G_-(o)$ является максимальным множеством, каждый элемент которого можно гарантировать локально допустимым, зная только $o$, support $\mathcal X_c$ и tolerance $\varepsilon$.

Если $g\notin\mathcal G_-(o)$, существует $x'\in\mathcal F_\varepsilon(o,c)$, для которого $m(x',g)<0$. Следовательно, ни один метод с тем же observation и теми же assumptions не может soundly сертифицировать такой $g$.

Аналогично, если $g\notin\mathcal G_+(o)$, он невозможен во всех observation-compatible мирах.

### Theorem 2. Оценка ошибки конечного fiber oracle

Пусть $\{x_j\}_{j=1}^{K}$ — $\delta$-net множества $\mathcal F_\varepsilon(o,c)$ в метрике $d_\mathcal X$, а $m(\cdot,g)$ является $L_x$-Lipschitz. Тогда

$$
0\leq
\min_j m(x_j,g)-m_-(o,g)
\leq L_x\delta,
$$

$$
0\leq
m_+(o,g)-\max_j m(x_j,g)
\leq L_x\delta.
$$

Если ошибка neural approximation empirical extrema не выше $\eta$, sound margins:

$$
\widehat m_-^{\,\rm cert}
=
\widehat m_- - (L_x\delta+\eta),
\qquad
\widehat m_+^{\,\rm cert}
=
\widehat m_+ + (L_x\delta+\eta).
$$

Условие $\widehat m_-^{\,\rm cert}\geq0$ достаточно для робастной локальной допустимости внутри model class.

Практический caveat: глобальный $L_x$ трудно оценить. В первой работе нужны либо контролируемая simulator metric и spectral/Lipschitz bounds, либо эмпирическая held-out correction. Conformal widening допустим как инструмент проверки, но не как novelty claim.

### Proposition 2. Инвариантность к frequency shift

Пусть две условные меры $P_1(x\mid o)$ и $P_2(x\mid o)$ имеют одинаковый support $\mathcal F_\varepsilon(o,c)$. Тогда их необходимые и возможные множества совпадают, хотя posterior success probabilities могут различаться.

Это даёт отдельный falsifiable experiment: reweight частоты форм, не меняя support.

### Proposition 3. Дискретизация grasp manifold

Если $m_-(o,\cdot)$ является $L_g$-Lipschitz, а query set — $h$-cover пространства $\mathcal G$, лучший дискретный maximin grasp отстаёт от непрерывного optimum не более чем на $L_gh$. Это обосновывает Sobol queries с последующим manifold gradient refinement.

## 7. Модель: Equivariant Fiber Operator

### 7.1. Вход

Один sparse input:

- partial target point tokens;
- obstacle/shelf point tokens;
- free-space/occlusion-shadow tags, выводимые из camera rays;
- optional class token;
- gripper geometry constants.

Нет dense full-scene SDF, mesh completion, object-centric canonical reconstruction и отдельного планировщика.

### 7.2. Запрос

Для каждого candidate $g$ локальные tokens переводятся в gripper frame. SE(3)-equivariant point encoder создаёт scene features, а query decoder предсказывает два ordered scalar fields:

$$
\widehat m_-(o,g)=a_\theta(o,g),
$$

$$
\widehat m_+(o,g)
=
a_\theta(o,g)+\operatorname{softplus}b_\theta(o,g).
$$

Так архитектурно гарантируется

$$
\widehat m_-(o,g)\leq\widehat m_+(o,g).
$$

Нужна одна shared backbone и один двухголовый decoder. Дополнительный diffusion-generator не требуется.

### 7.3. Выбор grasp

1. Сэмплировать 1–4 тысячи Sobol poses/widths в допустимом workspace.
2. Одним batched forward pass получить $\widehat m_-$.
3. Взять top-$k$.
4. Выполнить несколько шагов Riemannian gradient ascent по $SE(3)/C_2$ и width.
5. Исполнить максимум certified/adjusted lower field.

Это direct inference; полные формы во время работы робота не генерируются.

## 8. Как строится supervision без unstable reconstruction pipeline

Полные meshes используются **только офлайн для построения labels**, как ground truth в simulator, а не как промежуточный prediction target.

### 8.1. Observation groups

Для одной RGB-D проекции собирается fiber bank:

- разные полные объекты, чьи rendered visible surfaces совпадают в tolerance;
- hidden-part morphs, меняющие геометрию только внутри occlusion shadow;
- class-preserving shapes при known class;
- mixture support при unknown class;
- sensor perturbations и segmentation noise.

### 8.2. Hard witnesses

Для каждого $(o,g)$ ищутся witness shapes:

$$
z^-
=
\arg\min_z m(S_\phi(z),g)
$$

и

$$
z^+
=
\arg\max_z m(S_\phi(z),g)
$$

при ограничениях

$$
d_{\rm obs}\bigl(\mathcal R(S_\phi(z)),o\bigr)\leq\varepsilon,
\qquad
S_\phi(z)\in\mathcal X_c.
$$

Используются multiple starts, augmented Lagrangian, adversarial refinement и replay bank найденных witnesses. Shape model здесь служит офлайн генератором контрпримеров; online model никогда не выдаёт completed shape.

### 8.3. Loss

Для margins $M_j=m(x_j,g)$:

$$
\mathcal L_{\rm contain}
=
\frac1K\sum_j
\left[
\operatorname{ReLU}(\widehat m_- - M_j)^2
+
\operatorname{ReLU}(M_j-\widehat m_+)^2
\right],
$$

$$
\mathcal L_{\rm tight}
=
\left|\widehat m_--\min_jM_j\right|
+
\left|\widehat m_+-\max_jM_j\right|,
$$

$$
\mathcal L
=
\mathcal L_{\rm contain}
+\lambda_{\rm tight}\mathcal L_{\rm tight}
+\lambda_{\rm gap}(\widehat m_+-\widehat m_-)
+\lambda_{\rm eq}\mathcal L_{\rm equivariance}
+\lambda_{\rm Lip}\mathcal L_{\rm Lipschitz}.
$$

Containment отвечает за sound interval, extrema regression — за tightness, gap penalty не позволяет выдавать тривиальный бесконечно широкий interval.

### 8.4. Iterative oracle-model loop

1. Обучить на random fiber bank.
2. Найти grasps, где model optimistic.
3. Для них решить constrained hidden-shape minimization.
4. Добавить найденные failure witnesses.
5. Повторять до насыщения held-out violation rate.

Это аналог separation oracle: model учится на тех неразличимых геометриях, которые опровергают её текущую уверенность.

## 9. Почему это должно быть эффективно обучаемо

### Инференс

Uncertain-completion pipeline имеет приблизительную стоимость

$$
O(KC_{\rm completion}+KQ C_{\rm physics}),
$$

где $K$ — число completions, $Q$ — candidates.

FiberGrasp после scene encoding имеет

$$
O(C_{\rm encoder}+Q C_{\rm decoder}),
$$

без множителя $K$ и без online mesh processing.

Заявлять конкретные миллисекунды заранее нельзя. Paper-level target: batched inference вместе с refinement менее 100 ms на лабораторном GPU и выигрыш не менее $5\times$ против 30–60 completion samples.

### Обучение

Задача сводится к supervised interval regression по offline physical margins. Нет reinforcement learning, rollout credit assignment или VLA pretraining. Hard oracle дорог, но embarrassingly parallel и используется только при создании/reﬁnement dataset.

## 10. Что в идее пришло из общей математики/ML, а не из сборки robotics pipeline

- Rough-set theory задаёт lower/upper approximations по классам неразличимости: [вводный источник по rough sets](https://people.eecs.ku.edu/~jerzygb/Rough-sets.pdf).
- Для noisy observation вместо строгой equivalence используется tolerance relation; см. [rough approximations through general binary relations](https://arxiv.org/abs/1811.09609).
- Random-set view рассматривает неизвестное feasible set как set-valued random object; adversarial sampling уже полезен для ускорения нелинейной reachability: [Neural Bridge Sampling for evaluating safety-critical autonomous systems](https://proceedings.mlr.press/v155/lew21a.html).
- Equivariance для stochastic fields даёт правильный inductive bias: [Equivariant Conditional Neural Processes](https://proceedings.mlr.press/v139/holderrieth21a.html).
- Идея предсказывать достаточный для решения proxy вместо всей неопределённой величины согласуется с [Sufficient Decision Proxies for Decision-Focused Learning](https://arxiv.org/abs/2505.03953).

Robotics определяет только физический margin и эксперимент. Главный объект — learned set-valued decision operator under partial observability — является общей ML-постановкой.

## 11. Сравнение с ближайшими работами

| Работа | Что моделирует | Online completion | Семантика uncertainty | Что отсутствует относительно FiberGrasp |
|---|---|---:|---|---|
| TARGO-Net | completed target + grasp field | Да | одна reconstructed shape | observation fiber, necessary/possible sets, identifiability theorem |
| Lundell 2019 | grasp score на MC completions | Да | sample average | support extrema как direct field; maximal set |
| PSSNet | diverse plausible shapes | Да | несколько modes | action-space lower/upper operator |
| IROS 2025 uncertainty | variance penalty | Да, около 60 samples | heuristic point uncertainty | set semantics и soundness |
| UNCLE-Grasp | force closure на MC shapes + LCB | Да | conservative confidence | generic direct field, support invariance, upper set, observation-pair theorem/benchmark |
| Local Occupancy ECCV | local hidden occupancy | Да, локально | point occupancy | no-reconstruction action-set target |
| Contact-GraspNet / GraspLDM / IGD | distribution/score of grasps | Нет | learned training frequency | necessary vs possible under indistinguishable worlds |
| FiberGrasp | lower/upper margins on grasp manifold | Нет | support of observation fiber | предлагаемый вклад |

Самый опасный reviewer objection: «FiberGrasp — amortized UNCLE-Grasp». Ответ будет убедителен только при наличии одновременно:

- upper set, а не только conservative score;
- theorem of maximal certifiable set;
- frequency-shift experiment;
- indistinguishable-pair benchmark;
- отсутствие sampled completions на инференсе;
- generic parallel-jaw shelf setup, а не один узкий вид объекта.

Без этого objection справедлив.

## 12. Экспериментальная программа

### 12.1. Данные

1. TARGO balanced occlusion bins как внешний benchmark.
2. Отдельный TARGO subset с ровно одним фронтальным obstacle.
3. Новый controlled Shelf-Fiber benchmark:
   - wrist RGB-D;
   - target на полке;
   - один передний obstacle;
   - occlusion ratio 0–90%;
   - realistic depth dropout, quantization и pose noise;
   - class-known и class-unknown protocols;
   - полная simulator geometry доступна только для labels/evaluation.
4. Real robot: humanoid arm + parallel-jaw gripper; успех — стабильный grasp и малый подъём.

### 12.2. Baselines

- Contact-GraspNet или сильный direct grasp detector;
- GraspLDM / Implicit Grasp Diffusion;
- TARGO-Net;
- Local Occupancy-Enhanced Grasping;
- deterministic completion + grasp;
- Lundell-style MC completion scoring;
- PSSNet-style diverse completion;
- IROS 2025 uncertainty penalty;
- UNCLE-style lower confidence selection, адаптированный к parallel jaw;
- full-geometry oracle как верхняя граница.

### 12.3. Основные метрики

- grasp success rate по bins окклюзии;
- worst-bin success;
- падение easy $\rightarrow$ severe occlusion;
- regret к full-geometry oracle;
- precision/soundness necessary set;
- fraction сцен с непустым $\mathcal G_-$;
- violation rate: доля predicted-necessary grasps, опровергнутых held-out compatible shape;
- tightness $\widehat m_+-\widehat m_-$;
- success-coverage curve maximin/certified policy;
- latency, memory и число geometry-model calls.

### 12.4. Killer experiment

Создать пары $(x_1,x_2)$, которые:

$$
d_{\rm obs}\bigl(\mathcal R(x_1),\mathcal R(x_2)\bigr)\leq\varepsilon,
$$

но имеют разные hidden collisions или contact feasibility.

Для каждой пары проверить:

1. posterior-mean detector выбирает высокочастотный, но не necessary grasp;
2. FiberGrasp исключает его из $\mathcal G_-$;
3. shared feasible grasp, если он существует, остаётся в lower set;
4. при пустом intersection model показывает отрицательный maximin margin, а не искусственную уверенность.

Это наиболее чистое доказательство того, что работа изучает partial identifiability, а не просто data augmentation.

### 12.5. Distribution stress

- менять частоты shapes при фиксированном support;
- rare hidden protrusions;
- unseen instances и новые классы;
- неверный и отсутствующий class token;
- mismatch depth noise;
- segmentation boundary corruption;
- obstacle pose errors;
- расширение/сужение $\varepsilon$.

### 12.6. Критические ablations

- posterior mean vs lower field;
- lower only vs lower+upper joint learning;
- random fibers vs hard witnesses;
- без ray/free-space tokens;
- без equivariance;
- class known vs unknown;
- разное $K$ в offline fiber bank;
- без $\delta,\eta$ correction;
- Sobol only vs gradient refinement;
- true analytic margins vs learned outcome labels.

## 13. Проверяемая гипотеза о превосходстве над SOTA

Гипотеза:

> При severe single-view occlusion и fixed inference budget direct prediction of the necessary grasp set уменьшит catastrophic hidden-geometry failures сильнее, чем point-estimate completion, posterior-mean grasp scoring и finite-sample uncertainty penalties.

Основания:

- TARGO показывает систематическое падение с окклюзией;
- diverse/uncertain completions уже улучшают grasping, то есть ambiguity действительно полезна;
- IROS 2025 сообщает прирост от uncertainty penalty;
- Local Occupancy показывает, что невидимая локальная геометрия объясняет заметную часть разрыва;
- прямой action-space proxy убирает online completion cost и training-frequency averaging.

Но это **не гарантия SOTA**. Strict support worst-case может оказаться слишком консервативным. Поэтому работа обязана показать Pareto frontier «success vs coverage», а не скрывать пустые lower sets.

Минимальный go/no-go результат:

- не менее +5 percentage points success в severe occlusion против лучшего matched-latency baseline;
- не менее 2× снижение held-out fiber violation;
- не более 20% относительного падения coverage;
- минимум 5× ускорение против MC completion uncertainty;
- преимущество сохраняется при frequency shift.

Если эти условия не выполнены, claims уровня ICLR не подтверждены.

## 14. ICLR-аудит

[ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide) оценивает ясность проблемы, связь с литературой, поддержку claims, значимость нового знания, техническую корректность и воспроизводимость. [Call for Papers](https://iclr.cc/Conferences/2026/CallForPapers) явно включает robotics, uncertainty, structured prediction и representation learning.

### Specific problem

Да: выбор 6-DoF parallel-jaw grasp из одного noisy occluded RGB-D observation, когда разные физические сцены дают практически одинаковый input.

### Новый ML-вопрос

Как амортизированно предсказывать нижнее и верхнее множества непрерывных действий по observation-induced equivalence/tolerance classes, не реконструируя скрытое состояние?

### Claims, которые реально можно защищать

1. Observation-only grasp certification ограничена intersection feasible sets.
2. Это intersection — максимальный certifiable set при данных assumptions.
3. Fiber extrema допускают bounded approximation через $\delta$-net.
4. Support-based sets инвариантны к reweighting frequency.
5. Equivariant implicit operator оценивает их быстрее online sampling.
6. На severe occlusion такой target уменьшает hidden-geometry failures.

Пункты 1–4 теоретические; 5–6 требуют эксперимента.

### Почему это потенциально ICLR, а не только robotics paper

- новая set-valued learning target;
- continuous rough approximations на transformation group/action manifold;
- identifiability perspective;
- adversarial fiber supervision;
- generalization к другим partially observed decision problems: medical action selection, collision avoidance, manipulation and control.

### Оценка новизны

- **8/10**, если реализованы обе границы, теоремы, hard-witness oracle, paired benchmark и frequency-shift result;
- **5/10**, если реализован только neural lower-confidence score;
- **3/10**, если метод фактически делает K completions и min/variance aggregation.

### Главные риски

1. **Support misspecification.** Гарантия условна. Нужно публиковать violation under support expansion.
2. **Пустой lower set.** Нужны coverage curve и maximin fallback.
3. **Triviality theorem.** Intersection identity проста; научная ценность должна идти от learnable operator, approximation bounds и benchmark.
4. **Oracle bias.** Hard witnesses могут не покрыть fiber; нужны held-out generators и $\delta$-net/empirical corrections.
5. **Amortized robust optimization objection.** Отбивается только полным набором отличий из раздела 11.
6. **Metric ambiguity.** $d_{\rm obs}$, shape support и physical margin должны быть определены до экспериментов, а не подогнаны после.
7. **Overconservatism.** Нужно сравнить support extrema с $\alpha$-trimmed fibers, но robust version оставить основной.

## 15. Фальсификация новизны до дорогого обучения

Работу следует остановить или радикально изменить, если поиск обнаружит статью, которая одновременно:

- определяет observation-equivalence/tolerance fiber;
- выводит intersection/union feasible grasps;
- напрямую учит обе границы в continuous 6-DoF grasp space;
- доказывает maximal certifiability;
- не требует completion samples на инференсе.

Поиск должен продолжаться по терминам:

- necessary and possible grasp sets;
- lower/upper approximation robotic grasp;
- rough set grasp planning;
- observation fiber action set;
- set-valued grasp prediction partial observation;
- robust feasible action set point cloud;
- indistinguishable shapes grasping.

На момент этой записи прямого совпадения не найдено. Найдены Random Set Neural Networks для других задач и историческая rough-set mixed neural network для taxonomy-based выбора конфигурации трёхпальцевой кисти. Поэтому слово rough set само по себе заведомо не является новым. Не найден именно continuous lower/upper action-set operator, индуцированный классами неразличимых RGB-D наблюдений.

## 16. Минимальная реализация, проверяющая сущность идеи

До большой модели достаточно 2D/2.5D прототипа:

1. Сгенерировать силуэты объектов с одинаковой видимой фронтальной частью и разными скрытыми protrusions.
2. Перечислить planar parallel-jaw grasps.
3. Аналитически вычислить $m_-$ и $m_+$.
4. Обучить небольшой equivariant query network.
5. Сравнить с mean-shape, success probability и min по малому $K$.
6. Выполнить paired indistinguishability и frequency-reweighting tests.

Go-критерий: network сохраняет necessary-set precision при смене частот и находит shared grasp, когда mean-probability baseline выбирает несовместимый с одной из скрытых форм.

Только после этого оправдан переход к 3D mesh oracle и real robot.

## 17. Рекомендуемый paper claim

Наиболее сильная и честная формулировка:

> We introduce observation-fiber grasping: a set-valued learning problem that maps a partial RGB-D observation to the maximal necessary and possible subsets of a continuous grasp manifold. We derive identifiability and finite-fiber approximation results, propose an equivariant implicit fiber operator trained with adversarially discovered hidden-shape witnesses, and show that direct support-aware action-set inference reduces severe-occlusion failures without online shape completion.

Не следует заявлять:

- «первая uncertainty-aware grasping model»;
- «первая robust grasping under occlusion»;
- «гарантия для любой реальной формы»;
- «решает весь manipulation cycle»;
- «доказывает SOTA» до эксперимента.

## 18. Финальный вердикт

**Рекомендация: развивать FiberGrasp, но только в сильной set-theoretic версии.**

Это критически более новое направление, чем очередной completion, fusion module или diffusion grasp generator, потому что меняет сам prediction target: от скрытой формы или средней вероятности к границам того, что наблюдение позволяет и не позволяет утверждать о действиях.

Суть работы можно потерять очень легко. Если реализация станет «K hidden shapes $\rightarrow$ min score», это будет инженерное ускорение существующей линии. Если же будет direct lower/upper operator, maximality result, hard counterexample supervision и paired indistinguishability benchmark, возникает самостоятельный ML-вклад с разумной ICLR-мотивацией.

## Источники, использованные в этой итерации

- TARGO: https://arxiv.org/abs/2407.06168
- TARGO project: https://targo-benchmark.github.io/
- TARGO IJCV 2026: https://doi.org/10.1007/s11263-025-02716-9
- Local Occupancy-Enhanced Grasping: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09354.pdf
- Lundell et al., uncertain completions: https://arxiv.org/abs/1903.00645
- PSSNet: https://proceedings.mlr.press/v155/saund21a.html
- IROS 2025 shape-completion uncertainty: https://arxiv.org/abs/2504.16183
- UNCLE-Grasp: https://arxiv.org/abs/2601.14492
- Pick-and-place with segmentation and completion uncertainty: https://pmc.ncbi.nlm.nih.gov/articles/PMC8022832/
- Contact-GraspNet: https://arxiv.org/abs/2103.14127
- 6-DoF GraspNet: https://openaccess.thecvf.com/content_ICCV_2019/papers/Mousavian_6-DoF_GraspNet_Variational_Grasp_Generation_for_Object_Manipulation_ICCV_2019_paper.pdf
- GraspLDM: https://arxiv.org/abs/2312.11243
- Implicit Grasp Diffusion: https://proceedings.mlr.press/v270/song25b.html
- ShellGrasp: https://arxiv.org/abs/2109.06837
- GraspGen-X: https://openaccess.thecvf.com/content/CVPR2026/html/Han_GraspGen-X_Cross-Embodiment_6-DOF_Diffusion-based_Grasping_CVPR_2026_paper.html
- SpringGrasp: https://arxiv.org/abs/2404.13532
- Task-Informed Grasping: https://pure-oai.bham.ac.uk/ws/portalfiles/portal/239318079/deFariasC2024Task-informed.pdf
- Rough-set mixed grasp configuration planning (2018): https://doi.org/10.1016/j.mechmachtheory.2018.06.019
- Conformal safety for grasping: https://arxiv.org/abs/2109.14082
- Generalizing 6-DoF Grasp Detection: https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Generalizing_6-DoF_Grasp_Detection_via_Domain_Prior_Knowledge_CVPR_2024_paper.pdf
- Rough sets: https://people.eecs.ku.edu/~jerzygb/Rough-sets.pdf
- Rough approximations with general relations: https://arxiv.org/abs/1811.09609
- Neural Bridge Sampling / random-set reachability: https://proceedings.mlr.press/v155/lew21a.html
- Equivariant Conditional Neural Processes: https://proceedings.mlr.press/v139/holderrieth21a.html
- Sufficient Decision Proxies: https://arxiv.org/abs/2505.03953
- ICLR 2026 Reviewer Guide: https://iclr.cc/Conferences/2026/ReviewerGuide
- ICLR 2026 Call for Papers: https://iclr.cc/Conferences/2026/CallForPapers


---

# Research cycle: 2026-08-25 — initial landscape audit and first surviving formulation

## Status

This is a progress checkpoint, not a declaration that the goal is complete. The literature audit has
eliminated several superficially attractive ideas. One formulation survives the first novelty attack,
but ICLR-level acceptance is not yet objectively established because the theorem, implementation,
and decisive experiments are still missing.

Working paper title:

> **Infer Contact Consequences, Not Hidden Shape: Task-Quotient Posterior Processes for Reliable
> Parallel-Jaw Grasping under Occlusion**

Working framework name: **TQ-Grasp**, an instance of a general **Task-Quotient Posterior Process
(TQPP)**.

The core claim to test is not “shape completion helps grasping.” That claim is old. The proposed
new object of inference is the conditional law of an action-indexed mechanics function. The model
never decodes a complete point cloud, mesh, voxel occupancy, SDF, or scene field.

## Exact task boundary

The task is deliberately narrower than generic cluttered grasping and broader than a single
application-specific heuristic.

Input:

- one noisy wrist-camera RGB-D observation;
- a target mask or target prompt from which a target mask is obtained;
- one foreground obstacle or shelf lip that hides part of the target;
- gravity direction and camera calibration;
- a parallel-jaw gripper model.

Output:

- a reliable 6-DoF terminal parallel-jaw grasp, optionally with an explicit risk score;
- “terminal” means contact/closure and a tiny quasi-static lift, not complete arm approach,
  trajectory feasibility, or long-horizon manipulation.

Excluded:

- RL;
- VLA;
- active view selection or multi-view exploration;
- full approach-to-lift feasibility prediction;
- causal failure-mode taxonomies;
- full object or scene reconstruction;
- dense scene SDF/TSDF as a learned target;
- generic clutter removal.

The foreground obstacle is treated as observed geometry for conservative final-pose/short-closure
collision filtering. The learned uncertainty concerns the target's hidden grasp-relevant geometry.
The approach path remains a separate motion-planning concern.

## Literature audit: occupied directions

### 1. Direct grasp detectors from partial point clouds

GraspNet-1Billion established dense analytic grasp supervision and a unified 6-DoF benchmark.
Contact-GraspNet directly predicts a 6-DoF grasp distribution from a raw partial scene cloud.
GSNet/“graspness” filters the action space early using geometric graspability. AnyGrasp adds dense
spatial-temporal supervision, noise robustness, and center-of-mass awareness and reports strong
real-robot performance.

Primary sources:

- [GraspNet-1Billion, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html)
- [Contact-GraspNet, ICRA 2021](https://elib.dlr.de/145798/1/Contact-GraspNet.pdf)
- [Graspness/GSNet, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.html)
- [AnyGrasp](https://arxiv.org/abs/2212.08333)

Consequence: “directly predict grasp quality from a noisy partial cloud” is not a new task. A scalar
confidence head, ensemble, evidential head, or quantile head on such a detector is unlikely to clear
the ICLR novelty bar.

### 2. Implicit geometry and local completion for grasping

NeuGraspNet jointly learns a neural surface representation and grasping functions and explicitly
targets partially observed single-view scenes. Local Occupancy-Enhanced Grasping predicts occupancy
only around proposed grasp points. ShellGrasp-Net predicts a camera-centric entry/exit shell together
with grasp feasibility and quality. GIGA and VGN learn voxel/implicit fields for grasp prediction.
ZeroGrasp performs reconstruction and grasp prediction jointly at large scale.

Primary sources:

- [NeuGraspNet, RSS 2024](https://arxiv.org/abs/2306.07392)
- [Local Occupancy-Enhanced Object Grasping](https://arxiv.org/abs/2407.15771)
- [ShellGrasp-Net / camera-centric object shell](https://arxiv.org/abs/2109.06837)
- [ZeroGrasp, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.html)
- [VGN](https://arxiv.org/abs/2101.01132)

Consequence: “complete only local grasp regions,” “predict a compact shell,” “use an implicit local
surface,” and “jointly reconstruct and grasp” are occupied. Merely replacing a global completion
network by a local completion network is an incremental architecture change.

### 3. Occlusion-specific target grasping

TARGO is extremely close to the laboratory scene at the task level: direct target-driven grasping
from one RGB-D image under measured visual occlusion. TARGO-Net segments the target, completes its
shape with AdaPoinTr, fuses completed target and scene features, and predicts grasp fields. Its
benchmark shows a strong degradation of earlier models as occlusion increases and much smaller
degradation for its completion-aware model.

Primary source:

- [TARGO project and IJCV 2026 update](https://targo-benchmark.github.io/)
- [TARGO paper](https://arxiv.org/html/2407.06168v1)

Important facts for our novelty boundary:

- the task “target-driven grasping under occlusion” is already named and benchmarked;
- TARGO-Net owns the deterministic completion-plus-grasp architecture;
- its benchmark is still valuable as evidence that the gap is real;
- our paper must not claim novelty merely from foreground occlusion.

### 4. Shape-distribution-aware robust grasping

Robust Grasp Planning Over Uncertain Shape Completions samples voxel completions using Monte Carlo
dropout and evaluates grasps over those samples. It reports statistically significant simulation and
real-robot improvements over a point completion. A 2025 paper adds completion uncertainty to grasp
ranking and reports improved ranking/success.

Primary sources:

- [Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645)
- [Measuring Uncertainty in Shape Completion to Improve Grasp Quality](https://arxiv.org/abs/2504.16183)

Consequence: “sample plausible completed shapes and choose a robust grasp,” “Monte Carlo over
completions,” and “penalize uncertain completed points near a grasp” are occupied. A modern diffusion
completion model does not make this concept new.

### 5. General probabilistic ingredients

Neural Processes and Neural Diffusion Processes model distributions over functions. Work on
posterior predictive correlations demonstrates that marginal uncertainty can be insufficient for
downstream decisions. Loss-calibrated inference and decision-focused uncertainty quantification
already connect posterior approximation or prediction sets to utility. A 2026 computational
mechanics preprint directly learns posterior-predictive quantities of interest and reports better
accuracy and lower online cost than a conventional infer-parameters-then-propagate pipeline.

Primary sources:

- [Functional Neural Process, NeurIPS 2019](https://openreview.net/forum?id=H1gk0NrgIH)
- [Neural Diffusion Processes, ICML 2023](https://proceedings.mlr.press/v202/dutordoir23a.html)
- [Beyond Marginal Uncertainty, AISTATS 2021](https://proceedings.mlr.press/v130/wang21g.html)
- [Post-hoc loss-calibration, UAI 2021](https://proceedings.mlr.press/v161/vadera21a.html)
- [Decision-Focused Uncertainty Quantification](https://arxiv.org/abs/2410.01767)
- [Direct posterior-predictive variational inference](https://arxiv.org/abs/2605.03710)

Consequence: neural stochastic processes, loss calibration, direct quantities-of-interest inference,
or conformalization cannot individually be claimed as the paper's generic ML novelty. They can be
used as mathematical machinery and as indirect evidence.

## Sequentially rejected formulations

### Rejected A: deterministic grasp-centric hidden-contact completion

Idea: for each candidate grasp, predict the hidden opposing contact patch, normal, and local
occupancy; score antipodality.

Why it initially looked good: it avoids a full object reconstruction and seems mechanically local.

Why it is rejected:

- Local Occupancy-Enhanced Grasping already predicts hidden occupancy near proposed grasp points.
- ShellGrasp-Net already couples a compact hidden shell with grasp maps.
- NeuGraspNet already queries local implicit surface features for grasp evaluation.
- The remaining difference would be a change of output channels, not a new scientific object.

### Rejected B: stochastic completion plus CVaR/worst-case grasp selection

Idea: sample full or local completions, compute a mechanics score on each, then maximize its lower
quantile or CVaR.

Why it initially looked good: tail risk is more appropriate than mean completion quality.

Why it is rejected as the main contribution:

- the infer-shape-then-propagate structure is already explicit in 2019 robust uncertain completion;
- a different generative model or risk functional is insufficient novelty;
- it pays to model many hidden surfaces that no candidate grasp queries;
- approximation error compounds through completion and contact simulation.

Tail risk can remain a decision rule in the surviving formulation, but cannot be the core novelty.

### Rejected C: conformalized direct grasp confidence

Idea: attach a conformal lower success bound to a direct detector and abstain when no candidate is
safe.

Why it initially looked good: it creates a finite-sample reliability statement.

Why it is rejected as the main idea:

- decision-focused conformal UQ is already a general framework;
- marginal coverage is not a conditional guarantee for a specific heavily occluded scene;
- it does not represent multimodal hidden geometry;
- it is a wrapper around an existing detector rather than a new learnable formulation.

Conformal calibration may be a secondary evaluation/calibration layer, with claims worded carefully.

### Rejected D: prior-free worst-case certificate over all unobserved geometry

Idea: define the set of all shapes consistent with visible depth rays and accept a grasp only if it
succeeds for every shape.

Why it is rejected:

- for a grasp whose second contact is hidden, an adversarial unseen surface can almost always remove
  contact or violate aperture, making the certificate vacuous;
- useful nontrivial guarantees require a learned shape prior or a strongly restricted shape family;
- after adding a learned prior, the problem becomes distributional inference, not pure geometry.

This rejection clarifies that training-distribution uncertainty is essential, not an inconvenience.

## Surviving formulation: posterior over a mechanics quotient

### Latent setup

Let $S\in\mathcal S$ be the complete target shape drawn from the training shape distribution
$p_{\rm train}(S)$. Let $O$ be one noisy, partially occluded RGB-D observation generated by the
camera/occluder process:

$$
O \sim p(O\mid S).
$$

Let the terminal parallel-jaw grasp domain be

$$
\mathcal G \subset (\mathrm{SE}(3)\times [w_{\min},w_{\max}]) / C_2,
$$

where $C_2$ identifies the 180-degree wrist symmetry of a parallel-jaw gripper. In practice the
network is queried on a candidate subset $\mathcal G_O$, then can continuously refine candidates.

### Parallel-jaw contact consequence transform

For a complete shape $S$ and terminal grasp $g$, close the finite-area jaws in simulation and
compute only a small mechanics tuple:

$$
C_S(g) =
\big(
d_-(g),d_+(g),n_-(g),n_+(g),c_{\rm body}(g),\eta_{\rm lift}(g)
\big).
$$

Here:

- $d_-,d_+$ are first-contact closing distances;
- $n_-,n_+$ are the corresponding local contact normals or pad-averaged normals;
- $c_{\rm body}$ is signed clearance of the non-contact gripper body in the terminal pose and
  short closure sweep;
- $\eta_{\rm lift}$ is a signed quasi-static wrench margin for a tiny lift under a finite-pad
  soft-finger friction model.

For the wrench term, let $\mathcal W_g(S)$ be the convex set of wrenches producible by the two
finite pads under bounded normal force and fixed friction coefficient, and let $w_{\rm grav}(S)$
be the gravitational wrench under a declared density assumption. A clean definition is

$$
\eta_{\rm lift}(g,S)
=
\operatorname{sd}
\left(
-w_{\rm grav}(S),\partial\mathcal W_g(S)
\right),
$$

positive when the gravity-balancing wrench lies inside the feasible wrench set and negative
otherwise. This avoids the physically false claim that two ideal 3-D point contacts necessarily
provide full six-dimensional force closure.

Define a normalized scalar robustness margin

$$
m_S(g)=\min
\left\{
\widetilde c_{\rm body},
\widetilde a_{\rm aperture},
\widetilde a_{\rm antipodal},
\widetilde\eta_{\rm lift}
\right\}.
$$

Positive means the terminal closure has clearance, fits the gripper, is frictionally opposed, and
can resist the tiny-lift wrench. The exact normalization constants are gripper-specific, declared,
and fixed. A binary simulation success label is retained for validation, but the continuous margin
is the primary dense target.

This tuple is not a reconstruction. It is a query result of contact mechanics.

### Task equivalence and quotient

Define an equivalence relation on complete shapes for the current grasp domain:

$$
S_1 \sim_{\mathcal G_O} S_2
\quad\Longleftrightarrow\quad
m_{S_1}(g)=m_{S_2}(g)
\;\text{for every }g\in\mathcal G_O.
$$

The equivalence class discards every geometric distinction that cannot change the mechanics margin
of any relevant candidate. The random function

$$
M_S:\mathcal G_O\rightarrow\mathbb R,\qquad M_S(g)=m_S(g)
$$

is a coordinate-free representative of this task quotient.

The actual inference target is the posterior pushforward

$$
\Pi_O
=
(M_\cdot)_\# p_{\rm train}(S\mid O),
$$

a conditional distribution over mechanics functions. TQ-Grasp estimates $\Pi_O$ directly. It does
not estimate $p(S\mid O)$ and then push samples through a grasp simulator.

### Theorem target: decision sufficiency and coarseness

A first formal result should be proved cleanly.

**Proposition (task sufficiency).** Suppose the grasp decision loss depends on latent shape only
through the mechanics function, i.e.

$$
L(g,S)=\ell(g,M_S)
$$

for a measurable $\ell$. Then the posterior quotient law $\Pi_O$ is sufficient to compute every
Bayes action and Bayes risk for this loss family. No posterior over full shape is required.

**Proposition (coarsest deterministic latent representation).** If a deterministic representation
$T(S)$ preserves $m_S(g)$ for every $g\in\mathcal G_O$, then $T$ refines the equivalence
classes induced by $\sim_{\mathcal G_O}$. Therefore the quotient is the coarsest deterministic
shape representation that preserves all candidate mechanics evaluations.

These results are elementary enough to be believable but useful enough to pin down the exact new
scientific object. They must not be oversold as a new statistical minimal-sufficiency theorem.

A more valuable theoretical extension to attempt:

- derive an upper bound on downstream risk regret in terms of a probability metric between the true
  and learned quotient posteriors;
- for a Lipschitz risk functional and compact grasp domain, show
  $$
  |\mathcal R_{\Pi}(g)-\mathcal R_{\widehat\Pi}(g)|
  \le L\,W_1(\Pi,\widehat\Pi);
  $$
- extend this to the selected grasp with a factor of two via the standard argmax comparison;
- connect finite random-query training to uniform control using a covering number of the
  symmetry-quotiented grasp domain and a Lipschitz assumption on mechanics away from contact-mode
  discontinuities.

The contact-mode discontinuities are important. The theorem may need a piecewise-Lipschitz domain
or an explicit margin away from contact transitions.

## Learnable model: TQ-Grasp

### 1. Sparse ray-aware observation encoder

Encode only sparse evidence from the RGB-D observation:

- visible target points with RGB, depth confidence, normal estimates, and camera ray;
- foreground-obstacle/shelf points with a separate type tag;
- free-space ray segments up to measured surfaces;
- a compact token for the target's image-plane occlusion boundary.

Use a point/set encoder with vector features or an SE(3)-aware backbone. Because gravity and shelf
orientation are meaningful, equivariance claims should be stated relative to simultaneous rigid
coordinate changes, not as invariance to physically changing gravity.

No dense scene field is constructed.

### 2. Symmetry-aware action queries

A query token represents $g$ in the camera/shelf frame, gripper dimensions, and sparse features
pooled in the two finger pads and short closure sweep. Encode wrist angle with the $C_2$ symmetry
so equivalent parallel-jaw poses cannot receive contradictory predictions.

### 3. Projective latent-function decoder

Draw a base random seed $\omega$ once per posterior function sample and define

$$
\widehat M_{\theta,\omega}(g)
=
F_\theta\big(E_\phi(O),q(g),z_\omega,\xi_\omega(g)\big).
$$

Requirements:

- for a fixed $\omega$, queries at different grasps are values of one coherent differentiable
  function;
- querying a new grasp does not resample the hidden-shape hypothesis;
- arbitrary query sets are allowed;
- global latent randomness captures discrete hidden shape modes;
- a continuous random-feature field $\xi_\omega(g)$ captures local variation while preserving
  correlation between nearby grasps.

This construction gives projective consistency by design: the generated function exists before a
finite query subset is chosen. A Neural Process or Neural Diffusion Process is a baseline
implementation, not the novelty claim.

The decoder may output the mechanics tuple $C_S(g)$, then deterministically compute $m_S(g)$.
A scalar-only ablation is mandatory. Tuple supervision is expected to improve sample efficiency and
diagnosability without reconstructing geometry.

### 4. Joint conditional distribution training

For each complete training mesh:

1. render many noisy RGB-D observations with controlled foreground occlusion;
2. sample a set of $K$ grasp queries for the same latent object;
3. compute all contact/mechanics tuples offline from the mesh;
4. train the conditional generator on the joint $K$-query target.

A likelihood-free proper kernel/energy score can train implicit samples:

$$
\mathcal L_{\rm joint}
=
\mathbb E\,k(\widehat Y,\widehat Y')
-
2\mathbb E\,k(\widehat Y,Y),
$$

where $Y=[C_S(g_1),\ldots,C_S(g_K)]$, $\widehat Y,\widehat Y'$ are independent samples from
the predicted joint law, and $k$ is characteristic. The energy score is a practical special case;
a variogram or characteristic-kernel term should be evaluated because energy scores can be weakly
sensitive to correlation errors.

Reference for proper scores:

- [Gneiting and Raftery, strictly proper scoring rules](https://stat.uw.edu/research/tech-reports/strictly-proper-scoring-rules-prediction-and-estimation-revised)
- [Scoring rule nets and the correlation-sensitivity caveat](https://arxiv.org/abs/2409.14456)

Training batches must contain multiple grasps from the same object/observation. Independent
single-grasp batches cannot identify the posterior dependence structure.

### 5. Risk-aware continuous selection

For posterior samples $\widehat M^{(1)},\ldots,\widehat M^{(J)}$, rank a candidate by a lower-tail
functional such as

$$
R_\alpha(g\mid O)=\operatorname{CVaR}^{\rm lower}_\alpha
\left[\widehat M(g)\mid O\right].
$$

Choose

$$
g^\star=\arg\max_{g\in\mathcal G_O}R_\alpha(g\mid O).
$$

Use common posterior function samples while locally refining $g$. Coherent sample paths provide a
stable risk gradient; an independent marginal density estimator would resample incompatible hidden
shapes across neighboring queries.

Report the full risk-coverage curve over $\alpha$, not only the best tuned point. If abstention is
included, report both grasp success and coverage, and do not hide failures through aggressive
abstention.

## Why the joint posterior process is not just decoration

For a fixed finite candidate set and expected success of one action, calibrated marginals are
mathematically sufficient. Therefore the paper must not falsely claim that cross-grasp correlation
is always necessary.

The joint process becomes substantively useful for:

- continuous risk optimization and gradient-based refinement over $g$;
- coherent behavior under adaptively chosen new grasp queries;
- uncertainty over contact-mode boundaries;
- selecting a compact diverse top-$K$ fallback set, if that extension is evaluated;
- testing whether two nearby grasps fail under the same hidden-shape mode.

A decisive ablation must compare the joint process to a marginal conditional density model with the
same encoder, parameter count, number of posterior samples, and grasp candidates. If joint modeling
does not improve risk optimization or calibration, the architecture should be simplified and the
paper's claim narrowed.

## Indirect evidence that the direction can work

The evidence is triangulated rather than treated as proof:

1. TARGO shows that current direct methods degrade strongly with increasing occlusion and that
   hidden target geometry is useful.
2. Robust uncertain shape completion shows that integrating over geometric uncertainty improves
   analytic grasp ranking and real-robot success over a point estimate.
3. The 2025 uncertainty-ranking paper independently reports that completion uncertainty changes
   grasp ranking beneficially.
4. NeuGraspNet and local occupancy work show that grasp-level/local geometric queries can be more
   efficient and effective than blindly processing all geometry.
5. Direct posterior-predictive inference in computational mechanics reports better predictive
   distributions and lower online cost than a two-stage latent-posterior propagation workflow.
6. Work on posterior predictive correlations shows that joint functional uncertainty can matter to
   downstream decisions.

These findings support the premises. None establishes the proposed method's result in advance.

## Benchmark and experimental design

### Controlled benchmark: OCCL-Shelf

Generate a task-specific benchmark rather than relying only on generic clutter.

Scene:

- one target object on a shelf;
- one foreground occluder or shelf lip;
- wrist-camera viewpoints matched to the humanoid setup;
- no pile and no generic clutter.

Factors:

- target visibility bins, e.g. 10–30%, 30–50%, 50–70%, 70–90%;
- RealSense-like depth dropout, axial noise, flying pixels, and mask error;
- seen instances, novel instances from seen categories, and novel categories;
- occluder-target depth gap and lateral offset;
- gripper calibration/pose perturbations.

Labels:

- complete mesh used only offline;
- candidate contact tuple and robustness margin;
- binary dynamic or quasi-static tiny-lift validation in simulation;
- a real subset with repeated trials.

A filtered one-occluder subset of TARGO can be an external benchmark, but it should not replace the
controlled non-cluttered setup.

### Baselines

Required baseline families:

- direct partial-cloud detector and its original score: Contact-GraspNet, GSNet/AnyGrasp where
  licenses/access permit;
- deterministic direct mechanics regressor;
- marginal heteroscedastic/quantile mechanics predictor;
- deterministic shape completion plus grasp detector;
- stochastic shape completion plus Monte Carlo mechanics and the same CVaR decision;
- local occupancy-enhanced grasping;
- TARGO-Net or the closest reproducible completion-aware target grasp model;
- TQ-Grasp without tuple supervision;
- TQ-Grasp with independent per-grasp latents;
- full TQ-Grasp with coherent function samples.

Compute-matched comparisons are essential. The stochastic completion baseline must receive enough
samples to show the accuracy/latency frontier, not an artificially tiny budget.

### Metrics

Primary:

- physical grasp success after a tiny lift;
- success versus visibility/occlusion;
- failure rate among the top predicted risk decile;
- area under the risk-coverage curve;
- expected calibration error and reliability diagrams for success;
- lower-tail margin calibration;
- inference latency, memory, and number of physics/geometry queries.

Distributional:

- marginal CRPS;
- multivariate energy/kernel score on held-out grasp query sets;
- covariance or variogram error across nearby grasps;
- posterior mode coverage on deliberately ambiguous paired shapes.

Mechanics:

- contact distance/normal error;
- signed lift-margin error;
- ranking regret against an oracle with the complete mesh.

### Decisive synthetic ambiguity test

Construct paired shapes that produce nearly identical visible RGB-D observations but have different
hidden backsides and therefore different optimal grasps. This prevents a network from winning by
ordinary visible-surface regression.

For each observation family:

- completion methods must choose or sample hidden geometry;
- a marginal classifier can estimate individual success probabilities;
- TQ-Grasp must recover coherent alternative mechanics functions;
- posterior samples should correspond to the alternative grasp rankings without being required to
  form valid full shapes.

This is the cleanest empirical demonstration of the quotient posterior.

### Real-robot protocol

- fixed wrist view;
- object on shelf;
- one controlled foreground obstacle;
- parallel-jaw grasp;
- tiny lift only;
- at least 20–30 repeated trials per key visibility/risk condition or enough trials for meaningful
  binomial intervals;
- report all attempted grasps and perception failures;
- freeze risk threshold before the final test.

## Efficiency hypothesis

At inference, completion-based robust planning costs approximately

$$
\text{shape samples}
\times
\text{geometry decode cost}
+
\text{shape samples}
\times
\text{grasp evaluation cost}.
$$

TQ-Grasp costs approximately

$$
\text{posterior function samples}
\times
\text{queried grasps}
\times
\text{small tuple-decoder cost},
$$

with no high-resolution 3-D output. This should dominate when the number of queried grasps is much
smaller than the number of spatial samples needed to reconstruct a surface.

This is a hypothesis, not yet a theorem. It must be plotted as accuracy/calibration versus wall-clock
latency and memory. A completion method may still win when a dense shape is amortized across very
many downstream tasks; the present claim is specifically one-shot grasp selection.

## Novelty matrix

| Axis | Existing work | TQ-Grasp hypothesis |
|---|---|---|
| Occlusion task | TARGO benchmarks and completes hidden target | Uses same hard setting; no task novelty claim |
| Hidden geometry | Deterministic or sampled complete/local geometry | No geometry decoder |
| Prediction target | grasp pose/score, occupancy, SDF, shell | posterior over action-indexed contact mechanics |
| Uncertainty | marginal score or samples of completed shapes | coherent conditional process over the continuous grasp domain |
| Decision | top score or score penalized by completion variance | lower-tail optimization on quotient posterior samples |
| Supervision | points/voxels/SDF/grasps | sparse contact-mechanics queries from complete meshes |
| Formal object | scene/object representation | task equivalence class and posterior pushforward |
| General ML value | application-specific grasp predictor | general recipe for latent-state decisions when a simulator exposes task queries |

The likely defensible “first” claim, pending a more exhaustive search, is:

> first direct posterior process over parallel-jaw contact-mechanics consequences conditioned on a
> partial observation, learned from sparse complete-shape physics queries without reconstructing
> hidden shape or occupancy.

Do not claim “first occlusion-aware grasping,” “first uncertainty-aware grasping,” “first
task-oriented completion,” or “first neural process for robotics.”

## ICLR acceptance audit against official criteria

The ICLR 2026 reviewer guide asks whether the problem is specific, the approach is motivated and
well placed in literature, claims are supported rigorously, and the work contributes significant new
knowledge; SOTA is not itself required.

Official source:

- [ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide)
- [ICLR 2026 Call for Papers](https://iclr.cc/Conferences/2026/CallForPapers)

Current honest assessment:

- **Problem specificity: strong.** The latent ambiguity caused by a single occluded observation is
  clean and directly measurable.
- **Motivation/significance: strong.** TARGO supplies direct empirical evidence that occlusion causes
  a systematic performance collapse.
- **Originality: promising but not yet locked.** The quotient posterior and action-indexed mechanics
  process differ materially from deterministic completion, stochastic completion, and scalar grasp
  confidence. A deeper search for direct “posterior of simulator quantities at action queries” in
  robotics is still required.
- **Soundness: incomplete.** Mechanics definitions, contact discontinuities, and posterior training
  need proofs/ablations.
- **Empirical support: absent.** No acceptance claim is objective until the ambiguity test,
  compute-matched baselines, and real trials exist.
- **Broad ICLR value: plausible.** The general latent-task quotient formulation can extend to
  placement stability, insertion contact, and other simulator-query decisions, but the paper should
  demonstrate at least one non-grasp toy inverse problem or a general theorem rather than merely
  asserting breadth.

### Objective go/no-go gates

Continue only if all of these pass:

1. **Novelty gate:** no prior paper directly learns a coherent posterior process of action-indexed
   grasp mechanics from partial observation without geometry reconstruction.
2. **Identifiability gate:** paired-ambiguity data show calibrated multimodal posterior samples,
   not mean collapse.
3. **Decision gate:** lower-tail selection materially beats both a marginal distributional critic
   and stochastic-completion CVaR at equal compute.
4. **Efficiency gate:** a clear accuracy/calibration–latency Pareto improvement over stochastic
   completion.
5. **Robotics gate:** significant improvement at high occlusion on controlled simulation and real
   hardware, with confidence intervals.
6. **Theory gate:** quotient sufficiency/coarseness theorem plus a non-vacuous posterior-error to
   decision-regret result.
7. **Scope gate:** no hidden reintroduction of a mesh, point completion, local occupancy grid, or
   scene SDF in the architecture.
8. **ICLR gate:** the main contribution reads as a probabilistic representation/inference result
   instantiated in grasping, not as a robotics pipeline assembled from existing modules.

If gates 2–4 fail, reject TQ-Grasp and begin a new direction rather than cosmetically modifying the
architecture.

## Immediate next research steps

1. Search robotics, computational mechanics, and simulation-based inference for action-indexed
   posterior quantities of interest and query-consistent task surrogates.
2. Formalize the soft-finger finite-pad wrench margin and contact-mode regularity assumptions.
3. Prove the quotient propositions and derive the decision-regret bound.
4. Implement the paired-shape ambiguity toy experiment before any large benchmark.
5. Compare three equal-capacity models: deterministic margin, marginal conditional density, and
   coherent posterior process.
6. Only after the toy test passes, build OCCL-Shelf and completion baselines.
7. Keep the goal active until the novelty search and empirical/theoretical gates make the ICLR case
   objectively defensible.


---

## Critical novelty correction: goal-oriented Bayesian inference prior art

The first checkpoint's generic framing was still too broad. A non-robotics literature search found
direct prior art on exactly the high-level principle “infer the posterior prediction/QoI without
resolving the latent parameter.”

Key sources:

- [Goal-Oriented Inference: Approach, Linear Theory, and Application to
  Advection-Diffusion, SIAM Review 2013](https://epubs.siam.org/doi/10.1137/130913110)
- [Nonlinear Goal-Oriented Bayesian Inference, SIAM J. Scientific Computing
  2014](https://epubs.siam.org/doi/10.1137/130928315)
- [Goal-oriented optimal approximations of Bayesian linear inverse
  problems](https://arxiv.org/abs/1607.01881)
- [Goal-Oriented Bayesian Optimal Experimental Design](https://arxiv.org/abs/1802.06517)
- [Neural Conditional Probability, NeurIPS 2024](https://openreview.net/pdf?id=zXfhHJnMB2)

The 2014 nonlinear method explicitly learns the joint density of observations and low-dimensional
prediction quantities offline, then conditions it online to obtain a probabilistic prediction without
recovering the high-dimensional latent parameter. Therefore:

- “direct posterior-predictive inference” is not novel;
- “do not reconstruct nuisance latent state” is not novel;
- the quotient sufficiency proposition alone is not an ICLR contribution;
- computational efficiency from low-dimensional QoIs is established prior art.

This literature is valid mathematical inspiration under the research rules, but it must be cited as
a foundation, not presented as novelty.

### Refined general-ML formulation

The potentially new general problem is narrower:

> **Amortized function-valued goal-oriented inference for decision-indexed, non-smooth quantities of
> interest.**

Classical goal-oriented inference mostly targets a fixed finite-dimensional QoI vector. Here the QoI
is a stochastic process indexed by a continuous decision:

$$
Q_S:\mathcal A\to\mathbb R^d,\qquad a\mapsto Q(S,a),
$$

and the output is the conditional process law

$$
\mathcal L(Q_S(\cdot)\mid O),
$$

which must support arbitrary/adaptive decision queries, preserve joint dependence, respect action
symmetries, and permit risk optimization. For grasping, $\mathcal A$ is the parallel-jaw pose
space modulo wrist symmetry and $Q$ is the contact-mechanics consequence transform.

The refined novelty hypothesis is the combination of:

1. a continuous decision-indexed QoI rather than one fixed QoI vector;
2. projectively consistent posterior function samples under arbitrary action queries;
3. sparse physics-query supervision with no latent-state decoder;
4. risk optimization over the learned posterior process;
5. an instantiation where the QoI has contact-mode discontinuities and an
   $\mathrm{SE}(3)/C_2$ action geometry.

The paper must compare against:

- fixed-vector goal-oriented inference applied to a discretized action set;
- a marginal conditional distribution estimator;
- Neural Process / Neural Diffusion Process backbones;
- latent-shape posterior propagation.

### Stronger theory requirement

The elementary quotient theorem is now background. A publishable theorem should relate
function-posterior approximation to continuous decision quality.

Candidate result:

Let $\Pi_O$ and $\widehat\Pi_O$ be probability measures on a bounded function space over compact
$\mathcal A$. Let $\rho$ be a law-invariant risk functional that is
$L_\rho$-Lipschitz under the scalar $W_1$ metric. If

$$
W_1^{\|\cdot\|_\infty}(\Pi_O,\widehat\Pi_O)\le \varepsilon,
$$

then uniformly in action

$$
|\rho_{\Pi_O}(Q(a))-\rho_{\widehat\Pi_O}(Q(a))|
\le L_\rho\varepsilon.
$$

For the true and learned maximizers $a^\star,\widehat a$, this yields

$$
\rho_{\Pi_O}(Q(a^\star))-\rho_{\Pi_O}(Q(\widehat a))
\le 2L_\rho\varepsilon.
$$

The nontrivial part is to connect finite random action-query training to the function-space error.
Under a piecewise-Lipschitz mechanics class and a $\delta$-net of the regular action regions, seek
a bound of the form

$$
W_1^{\|\cdot\|_\infty}
\lesssim
W_1^{\text{queried values}}
+
2L_Q\delta
+
\text{contact-boundary mass}.
$$

The last term measures posterior probability near grazing/contact-mode boundaries. This makes the
robotics pathology part of the theory rather than hiding it under a global smoothness assumption.

### Important compression caveat

Across every possible grasp pose, first-contact data may contain enough information to recover much
of the object's surface. Therefore it is unsafe to claim that the complete continuous contact
transform is always information-theoretically lower-dimensional than shape.

The defensible efficiency claim is computational and decision-local:

- only queried candidate consequences are evaluated;
- no dense spatial discretization is decoded;
- query complexity adapts to the grasp optimizer;
- nuisance geometry outside all queried gripper interaction volumes receives no explicit loss.

Any statement that the quotient is inherently low-dimensional must be supported empirically or
restricted to a finite candidate set.

### Revised novelty claim

Pending further falsification, use:

> We extend goal-oriented Bayesian inference from fixed prediction vectors to projectively
> consistent, decision-indexed posterior processes, and instantiate it as a contact-mechanics
> process for occluded parallel-jaw grasping learned from sparse physics queries without hidden-shape
> reconstruction.

Do not use:

> We are the first to infer task-relevant posterior quantities without reconstructing the latent
> state.

### Revised go/no-go decision

TQ-Grasp remains alive, but its ICLR case is now harder and more honest. It passes the first robotics
novelty screen and fails any claim of generic “direct QoI posterior” novelty. It advances only if:

- no existing function-valued goal-oriented inference method already covers arbitrary
  decision-indexed posterior queries with the same consistency/optimization guarantees;
- the process model beats a discretized fixed-vector goal-oriented baseline and a marginal critic;
- the contact-boundary theory is non-vacuous;
- the empirical efficiency gain comes from adaptive action queries, not a weak completion baseline.

# Итерация: аудит contact-process и переход к observation-fiber supervision

Дата проверки: 2026-08-25.

## 1. Новые предшественники, меняющие вывод

### 1.1. Transformer Neural Decision Process уже покрывает общий decision-indexed process

Найден близкий общий ML-предшественник:

- D. Huang, Y. Guo, L. Acerbi, S. Kaski, **Amortized Bayesian Experimental Design for Decision-Making**, NeurIPS 2024, arXiv:2411.02064.
- Работа задаёт joint predictive distribution исходов как stochastic process, индексированный design set, использует Transformer Neural Decision Process, query set в непрерывном design space и downstream utility.
- Источники:
  - https://arxiv.org/abs/2411.02064
  - https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf

Следствие: формулировка «амортизированный posterior-process над непрерывным множеством решений вместо posterior над латентным состоянием» **не нова**. Вместе с классическим goal-oriented inference это снимает общий TQPP-claim.

Что остаётся полезным, но не может быть главным claim:

- posterior механических исходов достаточен для решения, если utility факторизуется через эти исходы;
- общие latent samples дают согласованные исходы для нескольких grasp queries;
- query-local inference может быть дешевле shape completion.

Это теперь supporting design rationale, а не фундаментальная новизна.

### 1.2. AnyDexGrasp уже покрывает детерминированное contact-centric representation из partial observation

Найден особенно близкий robotics-предшественник:

- H.-S. Fang et al., **AnyDexGrasp: General Dexterous Grasping for Different Hands with Human-level Learning Efficiency**, arXiv:2502.16420; ICLR 2025 submission.
- Источники:
  - https://arxiv.org/abs/2502.16420
  - https://graspnet.net/anydexgrasp/assets/files/AnyDexGrasp.pdf
- Модель отображает single-view partial scene point cloud в contact-centric grasp representation (CGR).
- CGR содержит расстояния до поверхности и углы нормалей на локальных сечениях; в статье явно сказано, что сеть выводит contact positions/normals на невидимых поверхностях.
- Представление строится плотно (более миллиарда CGR labels) и затем используется hand-specific decision model.

Следствие: идеи «не реконструировать весь object; предсказывать локальные contact distances/normals» и «плотная механическая supervision делает задачу эффективно обучаемой» **уже заняты**.

Поэтому предыдущий кандидат `Occlusion-Conditioned Contact Stopping Process` в детерминированной форме отклонён. Survival/hazard parameterization сам по себе также стандартен в survival analysis; first contact по swept volume давно вычисляется при известной геометрии.

### 1.3. Другие близкие границы

- ShellGrasp-Net: camera-centric entry/exit shell плюс grasp quality; https://arxiv.org/abs/2109.06837
- Local Occupancy-Enhanced Grasping: query-local hidden occupancy around grasp points; https://arxiv.org/abs/2407.15771
- NeuGraspNet: grasping as neural surface rendering; https://arxiv.org/abs/2306.07392
- Diverse Plausible Shape Completions from Ambiguous Depth Images (PSSNet): multimodal full-shape completion и физический grasp demo; https://proceedings.mlr.press/v155/saund21a.html
- TARGO-Net: точная task setting single RGB-D target grasping under foreground occlusion, но через deterministic target completion; https://targo-benchmark.github.io/

## 2. Решение по предыдущей ветке

### Отклонено как главный вклад

**Task-Quotient Posterior Process / generic contact field**.

Причины:

1. Direct QoI inference занят goal-oriented Bayesian inference.
2. Decision-indexed stochastic outcome process занят TNDP/Neural Processes.
3. Contact-centric hidden geometry representation занят AnyDexGrasp.
4. Stochastic full-shape ambiguity занят PSSNet и uncertain shape completion.
5. Без ещё одного принципиального элемента предложение сводится к conditional flow head на новом robotics target, чего недостаточно для объективно сильной ICLR-новизны.

Полезные части сохраняются как компоненты следующего кандидата:

- низкоразмерный joint contact outcome;
- физически корректная pair dependence;
- proper distributional loss;
- risk-aware grasp selection;
- отсутствие shape output.

## 3. Новый кандидат: Same View, Different Grasp

Рабочее название:

> **Same View, Different Grasp: Observation-Fiber Supervision for Contact-Posterior Grasping under Occlusion**

Короткая формулировка:

> Главная проблема single-view grasping под окклюзией не в том, что сеть плохо «достраивает» форму, а в том, что стандартный датасет почти всегда предъявляет каждому RGB-D наблюдению ровно одну скрытую геометрию. Он не показывает модели саму неидентифицируемость. Нужно обучать на группах разных полных форм, которые дают один и тот же noisy RGB-D в пределах sensor tolerance, но имеют разные grasp-relevant hidden contacts. Модель должна предсказывать не форму, а совместный posterior пары контактов и механического margin для query grasp.

Обозначение:

- `S` — полная target geometry;
- `O = H(S, E, eta)` — RGB-D observation при shelf environment `E`, foreground occluder и sensor noise `eta`;
- `g` — parallel-jaw grasp query;
- `Z_g(S)` — контактно-механический исход при квазистатическом закрытии губок;
- `p(Z_g | O)` — требуемый posterior outcome, не `p(S | O)`.

### 3.1. Observation fiber

Для tolerance `epsilon`, согласованного с шумом depth sensor:

`F_epsilon(O) = {S : d_obs(H(S,E), O) <= epsilon и S не нарушает observed free-space rays}`.

Это не произвольный set of completions. Члены fiber должны быть неразличимы **в фактическом sensor observation**, включая:

- depth на видимых target pixels;
- silhouette вне foreground-occluder mask;
- RGB, если appearance используется моделью;
- camera free-space constraints;
- shelf/occluder geometry;
- реалистичный noise tolerance.

При этом они специально различаются в механике query grasps:

`Z_g(S_i) != Z_g(S_j)` для части `g`.

### 3.2. Почему это не просто data augmentation

Обычная occlusion augmentation держит форму фиксированной и меняет видимость. TARGO single-scene augmentation добавляет occluder-induced failed grasps, но для одного наблюдения остаётся одна target geometry.

Здесь операция обратная:

- observation фиксирован в пределах sensor equivalence;
- hidden geometry меняется;
- механический outcome меняется;
- conditional model вынужден представить aleatoric ambiguity, а не выдать одно усреднённое скрытое касание.

Это **fiber conditional Monte Carlo supervision**: несколько условных исходов при одном дорогом контексте `O`.

### 3.3. Как строить fiber groups без test-time reconstruction

Два режима данных.

#### Режим A: контролируемые synthetic twins

- Объект строится как visible canonical front shell плюс sampled hidden module.
- Hidden modules принадлежат обучающему shape distribution: разная backside thickness, concavity, handle, rib, taper, local contact patch.
- Foreground obstacle и shelf lip скрывают все места соединения и различия.
- Рендеринг проверяет pixelwise RGB-D equivalence.
- Это даёт точный диагностический benchmark с известной conditional distribution.

Этот режим нужен для identifiability theorem и calibration, но сам по себе недостаточен для real-world claim.

#### Режим B: empirical CAD fibers

- Предварительно рендерятся CAD meshes/poses с большим набором shelf/occluder masks.
- Для каждого observation вычисляется visible signature: target depth residuals, silhouette, visible normals, ray-free-space violations.
- ANN retrieval находит meshes/poses с близкой visible signature.
- Exact renderer отбрасывает пары, отличимые сильнее RealSense noise model.
- Группа сохраняется только если члены имеют разные contact outcomes хотя бы для заданной доли query grasps.

Полные meshes нужны только offline для label generation и проверки equivalence. Ни mesh, ни voxel grid, ни SDF не являются prediction target или test-time variable.

### 3.4. Contact outcome

Для candidate `g` и фиксированного rigid object counterfactual closing:

`Z_g = (tau_L, u_L, n_L, tau_R, u_R, n_R, c_body, m_wrench)`,

где:

- `tau_L, tau_R` — first-contact jaw displacements или `MISS` atom;
- `u_L, u_R` — coordinates контакта на finite finger pad;
- `n_L, n_R` — contact normals;
- `c_body` — minimum non-fingertip gripper/body clearance для терминальной grasp pose;
- `m_wrench` — signed tiny-lift wrench margin при finite-pad soft-finger approximation.

Нельзя использовать ложный claim «две идеальные point contacts дают general 6-D force closure». Механика должна либо:

- использовать soft-finger torsional moment и finite patches;
- либо честно ограничить success gravity-resistance criterion, а не universal force closure.

Approach trajectory не моделируется; проверяется только заданная terminal insertion/closing primitive и tiny vertical lift. Это удерживает scope задачи.

## 4. Архитектура FiberContact

### 4.1. Observation encoder

- RGB-D wrist view;
- visible target mask/ID как стандартный task input, чтобы не смешивать вклад с open-vocabulary segmentation;
- explicit shelf, foreground obstacle и target tags;
- point/ray tokens: 3-D visible point, RGB feature, normal, camera ray, free-space interval, occlusion-boundary flag;
- sparse point-transformer или equivariant point encoder; никакого dense scene SDF.

### 4.2. Candidate generator

Не заявлять novelty здесь. Использовать воспроизводимый visible-surface proposal mechanism:

- seeds на видимой target surface;
- SO(3)/parallel-jaw symmetry-aware orientations;
- explicit quotient `g ~ g R_pi` при swapping identical jaws;
- один и тот же candidate set для всех evaluators.

### 4.3. Bi-Contact Survival Flow

Evaluator получает `(E(O), q(g))` и noise `xi` и генерирует joint sample `Z_hat_g`.

Структура decoder:

1. categorical atoms: left miss, right miss, body collision;
2. monotone spline marginals для `tau_L`, `tau_R`;
3. conditional copula/shared latent для pair dependence;
4. pad-coordinate density;
5. vMF/Bingham-compatible normal marks;
6. deterministic differentiable mechanics layer, вычисляющий aperture, antipodality и wrench margin из sample.

Критическая причина joint model: успешность одного grasp зависит от **совместимости** левого и правого контакта. Две отдельно хорошо откалиброванные marginal distributions могут соединить левый contact одного hidden mode с правым contact другого и создать несуществующий стабильный grasp.

Для выбора одного действия не нужен joint process между разными `g`; достаточно корректного joint posterior внутри каждого `g`. Cross-grasp coherence — опциональная функция для continuous optimization, но не главный claim.

### 4.4. Fiber-set loss

Для одного `(O,g)` доступен set true outcomes `{Z_g(S_k)}_{k=1}^K`.

Базовый objective — conditional log likelihood всех `K` outcomes. Более устойчивый sample-based вариант — energy distance между generated set и fiber target set в mechanics embedding `phi(Z)`.

`phi` должен содержать:

- contact distances/width;
- paired normals и antipodality;
- wrench witnesses;
- body clearance;
- explicit miss/collision indicators.

Стандартный marginal CRPS недостаточен для pair correlation. Energy score строго proper, но может быть слабо чувствителен к зависимости; обязательны сравнения с conditional CRPS/copula likelihood и отдельная correlation diagnostic.

First-contact time даёт dense nested supervision:

`CRPS(F, tau) = integral (F(t) - 1[tau <= t])^2 dt`.

То есть один точный `tau` одновременно обучает все nested jaw sweeps. Это сильная learnability motivation, но стандартное свойство CRPS, не claim новизны.

### 4.5. Decision rule

Для каждого candidate:

- sample `M` contact outcomes;
- compute mechanics margin `m_g^(j)`;
- оценить `P(m_g > 0 | O)` и lower-tail `CVaR_alpha(m_g | O)`;
- фильтровать visible obstacle/shelf collision;
- выбрать grasp по calibrated lower-tail score.

Главный baseline обязан использовать stochastic shape completion samples с тем же `M`, тем же mechanics layer и тем же CVaR. Иначе нельзя отделить direct contact posterior от более дорогого posterior propagation.

## 5. Формальные результаты, которые действительно поддерживают paper

### 5.1. Ambiguity gap

Пусть `s(g,S) in [0,1]` — physical success indicator/score.

Posterior-optimal single-view success:

`V_obs(O) = sup_g E[s(g,S) | O]`.

Shape-oracle success:

`V_oracle(O) = E[sup_g s(g,S) | O]`.

`Delta(O) = V_oracle(O) - V_obs(O) >= 0`.

`Delta` измеряет irreducible observation ambiguity. Если два равновероятных fiber members имеют disjoint successful-grasp sets, любой single-view decision rule ошибается как минимум с вероятностью `1/2`. Это не глубокая новая decision theory, но даёт точный benchmark quantity и запрещает нечестное сравнение с full-shape oracle.

### 5.2. Sufficiency contact pushforward

Если `s(g,S) = s_tilde(g,Z_g(S))`, то Bayes risk grasp `g` зависит только от `p(Z_g | O)`. Следовательно, полная `p(S|O)` вычислительно не обязательна.

Это классический goal-oriented pushforward result и должно быть представлено как proposition/rationale, не как новый общий theorem.

### 5.3. Compute-optimal number fiber replicates

Пусть stochastic gradient при fixed context имеет разложение variance:

- `A = Var_O(E[G | O])` — between-observation component;
- `C = E_O Var(G | O)` — within-fiber component.

При `N` observations и `K` conditionally independent fiber outcomes на observation:

`Var(G_bar) = A/N + C/(N K)`.

Если стоимость нового rendered/encoded observation равна `c_O`, а дополнительного contact label на существующем observation — `c_Z`, то при бюджете

`B = N(c_O + K c_Z)`

variance proportional to

`V(K) = ((c_O + K c_Z)/B)(A + C/K)`.

Непрерывный optimum:

`K* = sqrt(C c_O / (A c_Z))`.

Это даёт falsifiable prediction: fiber batching полезен именно когда ambiguity component `C` велик, а новый observation/render/encoder context дороже дополнительной mechanics annotation. Нужно проверить empirical variance curve и wall-clock, а не только downstream success.

### 5.4. Selection regret from contact-law error

Для bounded/Lipschitz mechanics utility `u` и uniform Wasserstein error

`sup_g W_1(p_hat(Z_g|O), p(Z_g|O)) <= epsilon`,

ошибка риска каждого grasp не больше `L epsilon`, а regret выбранного grasp относительно posterior Bayes grasp не больше `2 L epsilon`.

Из-за contact discontinuities bound применять к smoothed finite-pad margin или отдельно учитывать mass в `delta`-neighborhood contact boundary. Не скрывать этот caveat.

## 6. Novelty matrix после коррекции

### TARGO-Net

- Есть: точная target-under-occlusion task, single RGB-D, real/sim benchmark, target completion.
- Нет: groups одного observation с разными hidden geometries; calibrated joint contact posterior; ambiguity gap.

### PSSNet / stochastic shape completion

- Есть: diverse plausible full shapes из ambiguous depth; grasp application.
- Нет: no-reconstruction mechanics posterior; fiber-supervised action outcomes; paired-contact distribution; equal-compute direct-vs-propagation test.

### Robust grasp planning over uncertain completions / uncertainty re-ranking

- Есть: samples формы и robust grasp evaluation.
- Нет: direct conditional mechanics law; exact observation-equivalent training groups.

### AnyDexGrasp

- Есть: partial observation -> hidden contact distances/normals, compact contact-centric representation, dense supervision.
- Нет: conditional distribution/aleatoric fiber ambiguity; pair-correlated hidden contact modes; occlusion-controlled shelf setting.

### Goal-oriented inference / TNDP

- Есть: direct QoI posterior; decision-indexed stochastic outcomes; downstream utility.
- Нет: observation-equivalence data construction for non-identifiable visual mechanics; structured bi-contact survival law; physical benchmark.

### Честная граница claim

Нельзя заявлять:

- первый direct QoI posterior;
- первый stochastic process over decisions;
- первый contact-centric representation;
- первый uncertainty-aware occluded grasping;
- первый diverse posterior under ambiguous depth.

Можно проверять claim:

> первый framework для обучения parallel-jaw grasp risk на **observation-equivalent hidden-shape groups**, напрямую как совместный posterior контактной механики без shape reconstruction, с controlled ambiguity gap и equal-compute comparison против stochastic completion.

По текущему поиску exact prior не найден, но этот claim ещё требует author-level search по citations PSSNet, AnyDexGrasp, TARGO и conditional simulation literature.

## 7. OCCL-Fiber benchmark

### Сцена

- shelf plane + back wall;
- один target;
- один foreground occluder или shelf lip;
- wrist RGB-D at fixed/randomized known camera pose;
- target mask/ID supplied;
- никаких generic clutter piles.

### Splits

- seen categories / novel instances;
- novel categories;
- visibility bins;
- fiber ambiguity bins по `Delta(O)` и entropy contact mode;
- sensor noise bins;
- real 3-D printed twins с одинаковой видимой частью и разными скрытыми контактными модулями.

### Особенно важный real test

Напечатать family объектов с одним и тем же visible front и interchangeable hidden backsides. Foreground obstacle гарантирует одинаковый camera observation. Для каждого capture рандомизировать hidden member, не сообщая модели. Тогда calibration и Bayes-risk проверяются физически, а не только на CAD labels.

### Метрики

- tiny-lift physical success;
- success vs visibility и vs `Delta`;
- negative log likelihood / energy / conditional CRPS contact outcomes;
- calibration `P(success)`;
- left-right contact covariance/coupling error;
- posterior risk regret против fiber Bayes oracle;
- latency, peak memory, labels/render, labels/second;
- Pareto success vs inference compute;
- diversity без physical mode hallucination.

## 8. Обязательные baselines и ablations

Baselines:

1. AnyGrasp/GSNet-style direct detector.
2. TARGO-Net completion pipeline.
3. PSSNet/diffusion diverse completion + same mechanics + CVaR.
4. MC-dropout shape completion + robust re-ranking.
5. AnyDex-style deterministic contact representation adapted to parallel jaw.
6. Deterministic contact mean.
7. Conditional flow trained one-shape-per-observation.
8. Fiber flow trained `K>1`.
9. Oracle empirical fiber distribution.
10. Full-shape oracle, clearly labelled unattainable upper bound.

Ablations:

- `K = 1,2,4,8,16` at equal wall-clock and equal label budgets separately;
- true observation-equivalent fibers vs ordinary random hidden shapes;
- marginal independent contacts vs copula/joint flow;
- binary success posterior vs structured contact outcomes;
- NLL vs energy vs conditional CRPS;
- RGB-D vs depth only;
- without ray/free-space tokens;
- risk-neutral mean vs CVaR;
- synthetic twins only vs empirical CAD fibers;
- exact sensor-equivalence vs loose visible-patch matching.

## 9. Falsification gates

Кандидат нужно немедленно отвергнуть, если выполняется хотя бы одно:

1. После адаптации обычный conditional flow с `K=1` достигает той же calibration/success при equal total label budget.
2. Stochastic shape completion + same CVaR совпадает по success и Pareto compute, то есть direct posterior не даёт преимущества.
3. Fiber retrieval создаёт нереалистичные hidden geometries или leaking visible differences, которые сеть использует как shortcut.
4. Marginal independent contacts достаточно; joint contact law не улучшает ни likelihood зависимости, ни physical success.
5. Real printed-twin experiment не воспроизводит synthetic ambiguity gap.
6. AnyDex-style deterministic CGR с хорошо calibrated scalar success head решает задачу так же.
7. Paper остаётся комбинацией стандартного conditional flow, data augmentation и CVaR без отдельного сильного empirical phenomenon.

## 10. Текущий статус

Кандидат **сильнее предыдущего**, потому что:

- у него есть конкретная незакрытая failure mode стандартных datasets: один hidden world на observation;
- есть exact controlled experiment, где deterministic completion объективно не может быть правильной;
- representation и architecture теперь привязаны к парной contact mechanics;
- есть compute-allocation theorem с проверяемым `K*`;
- novelty можно сформулировать узко и сравнить с самыми близкими предшественниками.

Но цель ещё не достигнута объективно. До статуса «очевидно ICLR» не хватает:

- author-level citation search по PSSNet/AnyDex/TARGO;
- toy/synthetic empirical demonstration fiber batching vs `K=1` и stochastic completion;
- доказательства, что empirical CAD fibers можно строить без visible leakage;
- real-world printed-twin protocol хотя бы на уровне реализуемого bill of materials;
- evidence, что joint contact posterior улучшает physical selection, а не только probabilistic score.

Поэтому исследование продолжается; текущий direction не объявляется финальным.

# Минимальный empirical sanity check для fiber batching

Дата: 2026-08-25.

Цель теста: проверить только статистико-вычислительный claim из формулы

`K* = sqrt(C c_O / (A c_Z))`,

а не robotics performance.

## Toy setup

- Context `x in [-1,1]^2` играет роль дорогого observation.
- Hidden binary mode имеет нелинейную conditional probability `p(h=1|x)`.
- Есть три действия:
  - mode-1-specific grasp с success `p`;
  - mode-0-specific grasp с success `1-p`;
  - conservative common grasp с success `0.72`.
- Bayes rule выбирает максимум из `{p, 1-p, 0.72}`.
- Deterministic MAP-completion всегда выбирает один mode-specific grasp.
- Conditional probability оценивается корректно специфицированной logistic regression.
- 100 independent training seeds, 50,000 fixed evaluation contexts.

Cost model:

- новый observation `c_O = 100`;
- дополнительный conditional hidden outcome `c_Z = 1`;
- общий budget около `20,200`.

## Результат

| setting | unique contexts | labels | cost | Brier к true conditional p | decision value | regret к Bayes |
|---|---:|---:|---:|---:|---:|---:|
| equal-cost `K=1` | 200 | 200 | 20,200 | 0.007333 ± 0.000447 | 0.770464 ± 0.000697 | 0.009816 ± 0.000697 |
| equal-cost `K=8` | 187 | 1,496 | 20,196 | 0.000923 ± 0.000052 | 0.779165 ± 0.000089 | 0.001115 ± 0.000089 |
| equal-label `K=1` | 1,496 | 1,496 | 151,096 | 0.000862 ± 0.000053 | 0.779273 ± 0.000072 | 0.001007 ± 0.000072 |

Дополнительно:

- deterministic MAP-completion expected value: `0.733935`;
- Bayes oracle value: `0.780280`.

## Интерпретация

1. При заданном cost ratio grouped fiber outcomes почти достигают equal-label estimator, но примерно в `7.5x` меньшем бюджете.
2. У deterministic single completion есть заметный ambiguity gap даже при идеальном знании posterior mode probability, потому что оно не рассматривает conservative action.
3. Результат подтверждает variance/cost derivation и оправдывает реальный profiling experiment.

## Что этот toy **не** доказывает

- Он не показывает, что contact labels действительно в 100 раз дешевле нового shelf observation.
- Он не показывает, что empirical CAD fibers репрезентативны.
- Он не показывает преимущества structured contact posterior над direct scalar success probability.
- Он не показывает, что neural conditional model с `K=1` проиграет при большом обычном датасете.
- Он не является result для paper; это только go/no-go sanity check.

Критический следующий тест должен измерить реальные:

- `c_O`: render + RGB-D noise + encoder/context storage;
- `c_Z`: найти/проверить fiber member + collision/contact labels для batch queries;
- empirical `A` и `C` из gradient variance;
- observed optimum `K` против формулы.

Временный toy script удалён после выполнения; лишних файлов в workspace не создано.

# Коррекция после чтения полного PSSNet и uncertainty-region prior art

Дата: 2026-08-25.

## 1. PSSNet ближе, чем следовало из abstract

Полный текст:

- https://proceedings.mlr.press/v155/saund21a/saund21a.pdf

У PSSNet уже есть:

- явное множество plausible completions `P(x)` для конкретного partial depth input;
- shape-matching процедура, находящая разные meshes/poses, согласованные с одним наблюдением;
- set metrics coverage/plausibility/diversity;
- 20 sampled completions в physical grasp demo;
- side grasp, выбранный потому, что он охватывает разные plausible формы Cheez-it box;
- пример скрытой mug handle ambiguity.

Следовательно, нельзя заявлять как новое:

- построение множества разных скрытых форм, согласованных с одним depth input;
- использование diverse plausible shapes для robust grasp choice;
- сам факт, что single completion усредняет/теряет ambiguity.

Отличие рабочего кандидата от PSSNet должно быть строго следующим:

1. plausible groups используются как **training and calibration replicates**, а не только evaluation set;
2. прогнозируется likelihood-weighted law парных contact outcomes, а не voxel shapes;
3. группа специально отбирается по **mechanical disagreement**, а не только shape diversity;
4. test оценивает probabilities и Bayes regret на repeated identical observations, а не set Chamfer coverage;
5. direct posterior сравнивается equal-compute с PSSNet/diffusion completion propagation.

### PSSNet caveat, полезный для нашего benchmark

Авторы сами отмечают, что их observation model может игнорировать коррелированные depth differences, по которым сеть способна различить формы. Это подтверждает, что simple visible-patch threshold опасен. Нужен calibrated sensor likelihood и explicit leakage test.

## 2. Humt et al. уже покрывают identical views и irreducible uncertain regions

- M. Humt, D. Winkelbauer, U. Hillenbrand, **Shape Completion with Prediction of Uncertain Regions**, IROS 2023.
- https://arxiv.org/abs/2308.00377
- https://hummat.github.io/2023-iros-uncertain/

У них уже есть:

- identical views при разных handle positions;
- ground-truth uncertain-region annotation;
- direct trinary prediction `free / occupied / uncertain`;
- explicit claim objective/viewpoint-induced, irreducible uncertainty;
- grasp filtering, избегающий predicted uncertain region;
- synthetic and real evaluation.

Следовательно, нельзя заявлять:

- первый direct predictor irreducible geometric uncertainty;
- первый dataset identical views с разной hidden geometry;
- первый uncertainty-aware grasp filtering без MC posterior.

Отличие остаётся в том, что binary spatial uncertainty region не сообщает:

- какие левый и правый contacts совместно возможны;
- вероятности contact modes;
- какие uncertain regions совместно принадлежат одному hidden shape;
- может ли grasp быть безопасен через конкретную correlated pair geometry;
- lower-tail mechanics margin.

## 3. Ещё один новый close prior: SpaHybGen

- X. Wang, L. M. Tam, Q. Xu, **Learning contact representations in real-world clutter for universal robotic grasping**, Nature Machine Intelligence, 2026.
- https://doi.org/10.1038/s42256-026-01292-y

SpaHybGen выводит hardware-agnostic spatial contact features из noisy depth и оптимизирует grasps для разных hands. Это вместе с AnyDexGrasp окончательно исключает claim «первая learned contact representation».

По доступному описанию SpaHybGen не решает grouped observation ambiguity и не выводит likelihood-weighted pair-contact posterior, но его нужно включить как contact-representation baseline/positioning reference.

# Финально уточнённый research object: Grasp Metamers

Рабочий paper title:

> **Grasp Metamers: Learning Paired-Contact Beliefs from Indistinguishable RGB-D Views**

Название метода:

> **MetaContact** — metamer-supervised bi-contact mixture evaluator.

Название benchmark:

> **OCCL-Meta** — mechanically conflicting RGB-D metamer groups on shelves.

## 4. Определение grasp metamer

Две target geometries `S` и `S'` при фиксированных camera/environment variables являются `epsilon`-sensor metamers, если

`D_sensor(P(.|S,E), P(.|S',E)) <= epsilon`,

то есть distributions noisy RGB-D observations неразличимы с учётом реального noise model.

Они являются **grasp-conflicting metamers** для query family `G`, если

`D_mech({Z_g(S)}_{g in G}, {Z_g(S')}_{g in G}) >= delta`.

Практический `D_mech`:

- disagreement probability успешности на общем candidate set;
- Jaccard distance successful-grasp sets;
- difference posterior-optimal grasp;
- oracle-to-observation ambiguity gap;
- distance между paired contact outcomes после mechanics embedding.

Это не model metamer и не adversarial pixel noise. Равенство задаётся calibrated physical RGB-D forward process; различие — физической contact mechanics.

## 5. Генератор механически конфликтующих metamers

### 5.1. Retrieval path

1. Sample shelf scene, target pose, foreground obstacle, camera pose.
2. Render base target and store full noisy-observation likelihood signature.
3. Retrieve CAD meshes/poses from training shape distribution с близким visible signature.
4. Re-render с теми же environment parameters.
5. Reject при free-space violation, silhouette leak или sensor two-sample distinguishability.
6. Compute dense `Z_g` на общем grasp-query bank.
7. Сохранять только groups с `D_mech >= delta`.

### 5.2. Hidden-side deformation path

Если retrieval даёт мало метамеров:

- зафиксировать visible vertices/surfaces;
- деформировать только vertices, всегда скрытые camera + foreground occluder;
- использовать category shape basis/generative prior, watertightness и printability constraints;
- оптимизировать hidden deformation на увеличение `D_mech` при ограничении `D_sensor <= epsilon`;
- окончательно проверять independent renderer и sensor-noise classifier.

Это можно называть **mechanics-seeking nullspace augmentation**, но не заявлять общую новизну nullspace learning: null-space networks и measurement-consistent inverse methods давно существуют.

## 6. Вероятностные веса — обязательное отличие от unweighted plausible set

Просто равномерно усреднить найденные plausible shapes неправильно. Для proposal distribution `q(S)` нужны importance weights

`w_k proportional p_train(S_k) p(O | S_k,E) / q(S_k)`.

В synthetic twin families prior weights задаются частотами generation. В empirical CAD retrieval:

- `p_train(S)` оценивается category/shape density или задаётся empirical prior;
- `p(O|S,E)` приходит из calibrated RGB-D sensor likelihood;
- `q(S)` — известная retrieval/proposal probability.

Target contact law:

`p(Z_g | O) approx sum_k normalized(w_k) delta[Z_g(S_k)]`.

Без этой поправки модель учит дизайн retrieval algorithm, а не training shape distribution.

## 7. Почему grouped metamer test принципиально полезен

Стандартный test set имеет один realized hidden shape на почти уникальный continuous observation. Он может оценить marginal NLL/Brier, но не даёт прямой частотной проверки `p(success | exactly this O,g)`.

OCCL-Meta повторяет один sensor-equivalence class с известными hidden-member frequencies. Поэтому можно непосредственно проверить:

- conditional calibration в одной ambiguity group;
- recovery rare hidden modes;
- Bayes action против MAP completion;
- selective risk/abstention;
- whether predicted uncertainty comes from the correct hidden alternatives.

Это benchmark contribution сильнее обычного visibility binning: два метода с одинаковой средней success могут радикально отличаться по conditional belief внутри одной metamer group.

## 8. Минимальная реалистичная MetaContact architecture

### Encoder

- one sparse RGB-D target/scene encoder;
- ray/free-space/occlusion-boundary features;
- target, obstacle, shelf tags;
- features вычисляются один раз на observation.

### Query

- candidate grasp transformed в local gripper coordinates;
- jaw-swap `C2` symmetry enforced;
- cross-attention только к tokens в projected terminal gripper neighborhood и global target token.

### Hybrid `M`-component bi-contact mixture

Для каждого `g` output:

- mixture weights `pi_1...pi_M`;
- categorical atoms `MISS_L`, `MISS_R`, body collision;
- joint Gaussian/spline parameters для `tau_L,tau_R`;
- pad coordinates обоих contacts;
- unit-normal distributions в gripper frame;
- within-mode covariance, не independent marginal product.

Reasonable first implementation: `M=8`, low-rank covariance, normals as normalized 3-D vectors plus concentration. Не нужен diffusion over a giant scene field.

### Mechanics layer

- aperture feasibility;
- finite-pad contact inclusion;
- paired friction-cone/gravity wrench test;
- terminal gripper-target and gripper-scene collision;
- tiny-lift signed margin.

### Loss

- likelihood-weighted hybrid mixture NLL на всех member outcomes;
- conditional-CRPS/copula or energy-score auxiliary;
- jaw-swap consistency;
- optional mechanics-projected score;
- no mesh/occupancy/SDF reconstruction loss.

### Selection

`score(g) = CVaR_alpha(m_g | O)` или calibrated `P(m_g>0|O)` с minimum threshold.

Forced-choice success и selective success-coverage curve сообщаются отдельно. Abstention не должно скрывать failures.

## 9. Joint-pair necessity: точный toy counterexample

Пусть hidden contact marks `(L,R) in {-1,+1}^2`.

- Group A: `(+,+)` или `(-,-)` с вероятностью `1/2`.
- Group B: `(+,-)` или `(-,+)` с вероятностью `1/2`.

Обе группы имеют одинаковые marginal laws для `L` и `R`. Любой evaluator, который выводит только независимые marginals, не различает их.

Пусть mechanics success `s=1[L R = +1]`. Тогда:

- true success A = 1;
- true success B = 0;
- independent-marginal reconstruction даёт 1/2 для обеих.

Значит pair correlation необходима для некоторого класса mechanics utilities. Это не доказывает superiority над direct scalar success head; scalar head остаётся обязательным baseline. Structured pair claim должен выигрывать sample efficiency, transfer across friction/pad settings или calibration diagnostics.

## 10. Leakage protocol

Metamer group допустима только если выполнены все проверки:

1. `D_sensor` не превышает distribution реальных repeated captures одного объекта.
2. Twin-ID classifier на raw RGB-D не лучше chance с confidence interval.
3. Twin-ID classifier на encoder features best baseline также не лучше chance.
4. Pixel residual correlations проверяются, не только independent thresholds.
5. Free-space ray violations равны нулю в noiseless renderer.
6. Appearance/texture либо одинакова, либо sampled так, чтобы не кодировать hidden member.
7. Train/test split группируется по visible shell и hidden-module families, предотвращая memorization.

Если leakage classifier различает twins, observation-equivalence claim недействителен.

## 11. Численные go/no-go критерии для ICLR paper

Идея считается поддержанной только при всех следующих результатах.

### Phenomenon

- High-ambiguity OCCL-Meta имеет median `Delta(O) >= 0.15` absolute success.
- MAP deterministic completion проигрывает fiber Bayes action минимум на 10 percentage points в high-ambiguity bin.

### Method

- MetaContact лучше сильнейшего direct scalar/contact baseline минимум на 5 pp physical success в high-ambiguity bin, 95% bootstrap CI не пересекает 0.
- MetaContact не хуже stochastic completion + identical mechanics/CVaR по success более чем на 2 pp и использует не более 1/3 inference latency или memory.
- Joint mixture лучше independent contacts минимум на 3 pp physical success **или** даёт заранее заданный существенный gain в joint NLL/correlation плюс transfer test; иначе joint component удаляется.

### Fiber supervision

- `K>1` улучшает conditional proper score при equal wall-clock.
- Equal-total-label experiment честно показывает, какая часть gain происходит только от большего числа labels.
- Empirical optimum `K` согласуется по порядку величины с measured `A,C,c_O,c_Z`; иначе compute theorem не является практическим вкладом.

### Real

- Не менее 8 visible-shell families, минимум 3 hidden members каждая, минимум 2 occlusion configurations.
- Hidden member рандомизируется с заранее заданными frequencies.
- Достаточно trials, чтобы 95% interval различия high-ambiguity success исключал 0; точное число определяется power analysis, а не удобством.
- Одновременно сообщаются forced-choice и selective curves.

### Generalization

- unseen hidden modules и unseen visible shells;
- at least one held-out category family;
- RealSense noise/camera pose perturbation;
- friction/pad parameter transfer для structured contact law.

Если два или более ключевых thresholds не достигнуты, paper нельзя продавать как ICLR-сильный; направление закрывается или позиционируется как benchmark/robotics workshop.

## 12. Финальная novelty sentence после всех угроз

Наиболее защищаемая формулировка сейчас:

> We introduce mechanics-conflicting RGB-D grasp metamers—shape-prior-weighted groups that are indistinguishable to a calibrated sensor yet induce different parallel-jaw contact modes—and use them to train and conditionally calibrate a reconstruction-free bi-contact belief model for Bayes-risk grasp selection.

Она **не** утверждает первую ambiguity modeling, shape uncertainty, plausible set, contact representation, direct QoI posterior или robust grasping.

Проверяемая novelty состоит из связки:

1. mechanics-seeking sensor-metamer construction;
2. repeated/grouped conditional calibration benchmark;
3. likelihood-weighted joint pair-contact posterior;
4. no explicit shape prediction;
5. equal-compute comparison с posterior shape propagation.

По текущему поиску exact combination не найдена. Самая опасная reviewer-formulation остаётся:

> «PSSNet plausible sets + Humt uncertain regions + AnyDex/SpaHybGen contact representation + standard mixture density network».

Эту критику можно победить только сильным empirical phenomenon, conditional calibration protocol и убедительным efficiency/physical-success gain; архитектурного переименования недостаточно.

# Final research decision memo

Дата: 2026-08-25.

## Выбранная идея

**Grasp Metamers: Learning Paired-Contact Beliefs from Indistinguishable RGB-D Views.**

Один sentence:

> Создать likelihood-weighted группы разных hidden shapes, которые неразличимы для calibrated noisy RGB-D sensor в shelf scene, но дают конфликтующие parallel-jaw contacts; обучить на repeated outcomes query-conditioned joint bi-contact mixture без shape reconstruction и выбирать grasp по posterior lower-tail mechanics risk.

## Постановка окончательно фиксирована

- один wrist RGB-D frame;
- известная camera pose;
- один target с supplied visible target mask/ID;
- shelf, back wall и ровно один foreground occluder/lip;
- hidden target geometry известна только через training shape distribution;
- parallel-jaw terminal grasp pose + jaw closure + tiny vertical lift;
- no RL, no VLA, no active view/touch, no generic clutter, no full approach-to-lift planning;
- no explicit mesh/point completion, voxel/SDF output или giant scene variable set.

## Архитектура окончательно фиксирована

1. Sparse ray-aware RGB-D target/scene encoder.
2. Standard symmetry-aware visible-surface grasp proposals.
3. Gripper-local query cross-attention.
4. Hybrid `M=8` bi-contact mixture с joint left/right contact distances, pad points, normals, MISS/collision atoms и covariance.
5. Differentiable finite-pad gravity-wrench/clearance mechanics layer.
6. Likelihood/importance-weighted grouped NLL + joint proper-score auxiliary.
7. `CVaR`/calibrated probability selection; forced-choice and selective evaluation separately.

## Главная novelty окончательно фиксирована

Не «uncertainty», не «contact representation» и не «direct QoI» сами по себе, а одновременно:

1. **mechanics-conflicting physical sensor metamers** вместо произвольных diverse completions;
2. **repeated exact-equivalence groups** для conditional calibration и Bayes-regret evaluation;
3. **prior/likelihood-weighted joint contact modes**, сохраняющие корреляцию двух губок;
4. **reconstruction-free mechanics posterior**;
5. **equal-compute test** против stochastic full-shape posterior propagation.

## Самые близкие работы и точная дистанция

- TARGO: та же shelf/foreground-occlusion task, но deterministic target completion и нет metamer calibration groups.
- PSSNet: plausible shape sets и grasp over diverse completions, но full voxel reconstruction; groups главным образом для evaluation; нет weighted paired-contact posterior.
- Humt et al.: identical ambiguous views и direct uncertain spatial regions, но не вероятностные correlated contact modes.
- AnyDexGrasp/SpaHybGen: learned contact representation, но deterministic и не calibrated on hidden-geometry equivalence groups.
- uncertain completion/CVaR: posterior propagation через полную форму, обязательный equal-compute baseline.
- goal-oriented inference/TNDP: общие direct QoI/decision-process идеи, поэтому они не входят в novelty claim.

## Почему идея broad/open

На уровне ML это проверка **task sufficiency измерительного оператора**: разные latent worlds могут быть sensor-equivalent, но decision-conflicting. Grasping даёт редкий случай, где downstream discrepancy можно вычислить точной contact mechanics и проверить физически на interchangeable printed twins. Тот же protocol переносим на collision checking, tool contact и другие inverse decisions, но paper не обязан демонстрировать все области.

## Почему efficiently learnable

- output low-dimensional и query-local;
- один observation encoding используется для многих hidden outcomes и grasp queries;
- exact first-contact label даёт dense nested-width supervision;
- toy подтвердил compute/variance mechanism: при `c_O:c_Z=100:1` equal-cost `K=8` снизил decision regret с `0.009816` до `0.001115`, почти сравнявшись с equal-label `K=1` за ~7.5x меньшую стоимость;
- реальный claim допускается только после profiling истинных `c_O,c_Z,A,C`.

## Решающий theorem/metric package

- `Delta(O)=E[max_g s(g,S)|O]-max_g E[s(g,S)|O]` — metamer ambiguity/Jensen gap;
- `Delta=0` iff существует один posterior-common optimal grasp (для finite candidate set, с обычными условиями ties);
- grouped fibers позволяют непосредственно оценивать `Delta`, conditional calibration и Bayes regret;
- contact-pushforward sufficiency поддерживает отсутствие reconstruction;
- compute-optimal replicate count `K*=sqrt(C c_O/(A c_Z))`;
- joint-contact counterexample доказывает невозможность recovery mechanics risk из одних marginals.

Эти результаты в основном собирают известную probability/decision theory для новой измеримой постановки; paper не должен выдавать их за глубокую универсальную теорию.

## Объективный ICLR verdict

**Идея достаточно ясна и потенциально ICLR-сильна, но только как phenomenon + benchmark + method paper.** Одна архитектура mixture head недостаточна.

ICLR case становится сильным, если выполнены заранее зафиксированные gates:

- metamer ambiguity gap >=15 pp;
- >=5 pp physical high-ambiguity gain над сильнейшим direct baseline с CI;
- parity по success с stochastic completion при <=1/3 latency/memory;
- leakage classifiers at chance;
- real interchangeable-twin calibration;
- joint posterior даёт физический/transfer gain;
- equal-wall-clock fiber gain не исчезает после честного equal-label анализа.

Если gates не выполняются, правильный научный вывод заранее известен: direct scalar success или stochastic completion достаточны, а MetaContact не является ICLR-вкладом.

## Статус цели

Постановка, архитектура, защищаемая novelty, closest-prior boundary, theory targets, benchmark, real protocol и falsification thresholds теперь однозначны. Исследовательская цель формирования идеи завершена; следующая отдельная цель — реализация OCCL-Meta и empirical validation.
