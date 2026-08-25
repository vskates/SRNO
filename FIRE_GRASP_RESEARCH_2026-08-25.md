# FIRE-Grasp: fiberwise entropic regret для grasp selection без реконструкции скрытой формы

Дата исследования: 25 августа 2026 года  
Статус: исследовательская гипотеза и план проверки, а не утверждение о достигнутом SOTA

## 0. Короткий вердикт

Предлагаемая ставка — не предсказывать скрытую геометрию, occupancy, SDF или набор shape completions. Вместо этого надо обучать **распределительно-устойчивый regret конкретного grasp по множеству полных форм, неразличимых в данном RGB-D наблюдении**.

Рабочее название:

> **FIRE-Grasp — Fiberwise Inverse-decision Regret Elicitation for Occluded Grasping.**

Главная general-ML идея:

> В неидентифицируемой inverse problem следует аугментировать не обычными label-preserving преобразованиями, а преобразованиями в null space измерительного оператора, которые **сохраняют наблюдение, но меняют полезность решений**. Такие варианты нельзя заставлять иметь одинаковый label и не обязательно нужно реконструировать. Их надо объединять в observation fiber и непосредственно обучать tail-risk hindsight regret решения внутри fiber.

Для grasping это означает следующее. В training имеется полная mesh-геометрия. За foreground-occluder создаются несколько правдоподобных hidden-only вариантов объекта, которые дают то же RGB-D изображение в пределах sensor tolerance. Для каждого варианта и каждого одного и того же candidate grasp вычисляется только terminal closure quality. Затем grasp получает label не «успех на одной форме», а tail-sensitive gap до grasp-оракула, знающего полную форму. На inference модель видит только один шумный RGB-D кадр и сразу оценивает этот regret; ни форма, ни локальная occupancy, ни completion samples не декодируются.

Это направление выглядит сильнее рассмотренных альтернатив по трем причинам:

1. Оно вводит новый объект обучения — **fiberwise conditional decision regret**, а не модифицирует reconstruction pipeline.
2. Оно эксплуатирует полную форму только как privileged training information и устраняет Monte Carlo over completions на inference.
3. Оно переносится на широкий класс ill-posed inverse-decision задач: null-space perturbation сохраняет measurement, oracle оценивает downstream actions, а learner учит pushforward риска, не latent state.

Но novelty нельзя считать доказанной поиском литературы. Самые опасные соседи — direct grasp scoring, learning with privileged information, decision-focused regret, robust DFL и entropic/DRO risk — существуют по отдельности. ICLR-level вклад появится только если одновременно сработают: (i) fiber construction; (ii) новый regret-elicitation objective; (iii) теоретическая decision-sufficiency часть; (iv) убедительное преимущество над completion, mean-score и direct-CVaR baselines на occlusion twins и реальном роботе.

## 1. Точная область и сознательные ограничения

Рассматривается один целевой rigid object на полке, один foreground-occluder, wrist RGB-D camera, parallel-jaw gripper и один кадр. RGB-D может быть шумным. Цель — выбрать terminal grasp pose, после закрытия которого объект можно надежно оторвать на минимальную высоту.

Не рассматриваются:

- RL и VLA;
- rearrangement, active view selection и удаление occluder;
- clutter из многих объектов;
- оценка всего цикла approach–close–lift–transport;
- scene-wide SDF/occupancy и полная object reconstruction;
- causal failure-mode taxonomies;
- обучение генератора shape completions как части test-time метода.

Минимальные предположения:

- доступна маска видимой части target и маска foreground-occluder либо они даны upstream-модулем;
- candidate set строится одинаковым способом для всех сравниваемых scorers;
- грубая reachability и collision с наблюдаемой полкой/препятствием проверяются отдельным консервативным geometric filter;
- core model оценивает только grasp closure и устойчивость начала lift, а не trajectory feasibility;
- training meshes и physics labels доступны в simulation.

## 2. Что уже сделано и почему очевидные направления недостаточны

### 2.1 Full/local completion уже является сильной и занятой линией

[Shape Completion Enabled Robotic Grasping](https://arxiv.org/abs/1609.08546) уже в 2016 году обучал full voxel completion из single-view point cloud, после чего планировал grasp на completed mesh. [CenterGrasp](https://arxiv.org/abs/2312.08240) совместно декодирует полную форму и grasp manifold; авторы сообщают среднее преимущество в 33 процентных пункта grasp success над GIGA и показывают grasps на невидимой стороне объекта. [Local Occupancy-Enhanced Object Grasping](https://arxiv.org/abs/2407.15771) специально завершает occupancy в локальных grasp regions. [NeuGraspNet](https://roboticsproceedings.org/rss20/p046.pdf) рендерит локальную поверхность из implicit scene representation; его ablations показывают, что local rendering и local occupancy supervision существенно помогают при hard viewpoints.

Следствие: «предсказывать только локальную скрытую геометрию около пальцев» уже не является достаточно новой идеей. FIRE-Grasp принципиально не имеет geometry decoder и не использует occupancy/SDF loss.

### 2.2 Uncertainty over completions тоже уже исследуется

[Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645) в IROS 2019 использовал MC dropout, несколько completed shapes и joint analytic evaluation; авторы получили статистически значимое улучшение над single completion. [UNCLE-Grasp](https://arxiv.org/abs/2601.14492) в 2026 году снова строит несколько completions, агрегирует force-closure feasibility и применяет lower-confidence bound с abstention. [Shape Completion with Prediction of Uncertain Regions](https://arxiv.org/abs/2308.00377) показывает, что избегание predicted uncertain regions повышает grasp quality.

Следствие: «sample completions + variance/LCB/CVaR» — не новая постановка. FIRE переносит marginalization из inference в training objective и вообще не материализует completion.

### 2.3 Прямой grasp evaluator — сильный baseline, но учит другой функционал

[Dex-Net 2.0](https://www.roboticsproceedings.org/rss13/p58.html) уже обучал GQ-CNN прямо предсказывать robust grasp success из synthetic depth, используя 6.7 млн point-cloud/grasp/metric примеров; сообщалось 93% success на adversarial known objects и высокая precision на novel household objects. [Get a Grip](https://proceedings.mlr.press/v270/lum25b.html) показывает на multi-finger setting, что масштабный discriminative evaluator может превзойти analytic и generative baselines, причем качество evaluator деградирует при уменьшении данных.

Следствие: сам тезис «не реконструировать, а score grasp напрямую» не нов. Отличие FIRE обязано быть в target: не expected success одной sample shape, а tail-risk hindsight regret по observation-equivalent latent shapes.

### 2.4 Точная occlusion problem уже имеет benchmark и сильный completion baseline

[TARGO/TARGO-Net](https://targo-benchmark.github.io/) — наиболее важный сосед. Опубликованный в IJCV 2026 benchmark непосредственно изучает target-driven 6-DoF grasping из одного RGB-D кадра при occlusion. TARGO-Net завершает target shape, fusion-ит ее с scene features и, по данным проекта, теряет около 7% на extreme synthetic occlusion, тогда как несколько предыдущих методов теряют около 20% или больше; в real experiments разрыв easy-to-hard также существенно меньше. TARGO дополнительно показывает, что occlusion-induced negative grasps улучшают baselines минимум примерно на 5%, а TARGO-Net примерно на 10%.

Это одновременно угроза novelty и сильное косвенное подтверждение FIRE: supervision именно на изменившейся из-за occlusion decision quality полезна. Однако TARGO остается completion-based и использует binary/quality grasp labels, а не repeated latent variants одного observation и не fiberwise risk.

### 2.5 General ML соседи

- [Decision-Focused Learning through Learning to Rank](https://proceedings.mlr.press/v162/mandi22a.html) трактует DFL как корректное ранжирование feasible solutions.
- [DF²](https://proceedings.mlr.press/v286/kong25a.html) напрямую учит expected optimization function вместо параметрической predictive distribution.
- [Robust Decision-Focused Learning via Worst-Case Regret](https://proceedings.mlr.press/v337/yamao26a.html) рассматривает measurement error и deployment shift через worst-case regret, uncertainty sets и Wasserstein ambiguity. Эта работа опубликована 17 августа 2026 года и является contemporaneous для ICLR 2027, но ее все равно надо обсуждать.
- [Learning Using Privileged Information](https://proceedings.neurips.cc/paper/2010/hash/c73dfe6c630edb4c1692db67c510f65c-Abstract.html) формализует training-only teacher information.
- [NPN](https://proceedings.neurips.cc/paper_files/paper/2025/hash/acb1891d79e617134ed604084ebbc919-Abstract-Conference.html) напоминает, что inverse problem имеет бесконечное множество решений в null space measurement operator, но занимается reconstruction.

Следствие: regret, direct objective learning, privileged information, null spaces и DRO нельзя заявлять новыми по отдельности. Потенциальная novelty FIRE — их новая problem-level связь: **measurement-equivalent latent variants с разными decision labels образуют fiber, а conditional entropic oracle gap elicited напрямую без prediction/optimization и без latent reconstruction**.

## 3. Отброшенные направления

### A. Probabilistic local contact signature

Идея: предсказывать joint distribution first-contact depths/normals двух пальцев и clearance. Отброшена как слишком близкая к local occupancy/surface rendering и как скрытая реконструкция grasp-local geometry.

### B. Full conditional quality distribution / stop-loss operator

Идея: предсказывать conditional CDF, quantiles или stop-loss transform физического grasp loss и получать CVaR. Отброшена как потенциально полезная baseline, но недостаточно новая: это в значительной степени distributional regression с другим head.

### C. Completion samples + conformal/LCB

Идея: sample completions, затем simultaneous lower bounds после adaptive grasp selection. Отброшена из-за близости к IROS 2019 и UNCLE-Grasp 2026, а также из-за pipeline-композиции.

### D. Random-closed-set capacity functional

Идея: учить вероятность пересечения hidden shape с gripper query volumes. Математически интересна, но force closure требует совместных marked contact events; простой capacity functional недостаточен, а расширение фактически возвращает локальную geometry model.

### E. Wrench-space support-function prediction

Идея: предсказывать support function latent grasp wrench set в dual space. Элегантна, но порядок min over wrench directions и expectation over hidden shapes дает опасные bounds; architecture получается сложнее, а выигрыш над direct quality неочевиден.

### F. Causal failure modes и full-cycle feasibility

Исключены постановкой задачи и дополнительно размывают central ML claim.

## 4. Новая broad постановка: decision learning on observation fibers

Пусть

- \(x\in\mathcal X\) — полный latent physical state целевого объекта: mesh и nuisance variables, используемые только при генерации training labels;
- \(c\) — известная camera/occluder configuration;
- \(y=\mathcal A_c(x)+\epsilon\) — одно RGB-D наблюдение;
- \(\mathcal G(y)\) — конечный набор candidate parallel-jaw grasps после одинакового reachability/visible-collision filter;
- \(u(x,g)\in[-1,1]\) — closure-only mechanics utility;
- \(g=\bot\) — разрешенный reject/abstain action с \(u(x,\bot)=0\).

Для tolerance \(\tau\) определим observation fiber:

\[
\mathcal F_\tau(y,c)=\{x':d(\mathcal A_c(x'),y)\le \tau\}.
\]

Это множество полных форм, которые RGB-D sensor не может различить в данной конфигурации. В отличие от стандартной data augmentation, изменения внутри fiber могут менять правильный grasp.

Full-information oracle знает \(x\), но ограничен тем же candidate set:

\[
u^*(x,y)=\max\left(0,\max_{h\in\mathcal G(y)}u(x,h)\right).
\]

Нормированный hindsight gap grasp \(g\):

\[
\Delta(x,y,g)=\frac{u^*(x,y)-u(x,g)}{2}\in[0,1].
\]

Reject action нужен не для превращения задачи в selective prediction, а для корректного масштаба regret: если все grasps небезопасны, omniscient oracle тоже может отказаться. Если deployment обязан сделать попытку, \(\bot\) удаляется только после scoring, но coverage–risk curve все равно следует публиковать.

### 4.1 Closure-only utility

Не нужен длинный вектор scene variables. Практичная scalar utility:

\[
u(x,g)=2\,p_{\text{term}}(x,g)-1,
\]

где \(p_{\text{term}}\) — вероятность bilateral antipodal/force-closure terminal contact без gripper penetration при малых SE(3), depth и friction perturbations. Она оценивается офлайн на полной mesh с небольшим батчем perturbations. Никакой approach trajectory или полный lift не симулируется. Реальный критерий — удержание после минимального вертикального отрыва — используется только для финальной hardware evaluation.

Альтернатива для более дешевого первого прототипа: normalized robust wrench-resistance / epsilon metric, аналогичный analytic supervision в Dex-Net/GraspNet. Binary success нельзя использовать как единственный \(u\): для Bernoulli outcome многие risk transforms сводятся к тому же ranking по success probability; нужен непрерывный mechanics margin или perturbation-averaged utility.

## 5. FIRE objective: conditional entropic hindsight regret

Для \(\beta>0\):

\[
R_\beta(y,g)
=\frac{1}{\beta}\log
\mathbb E_{X\mid Y=y}
\left[\exp\big(\beta\Delta(X,y,g)\big)\right].
\]

Deployment rule:

\[
g_\theta(y)=\arg\min_{g\in\mathcal G(y)\cup\{\bot\}}
\widehat R_{\beta,\theta}(y,g).
\]

Интерпретация:

- \(\beta\to0\): expected hindsight regret;
- большой \(\beta\): приближение к worst plausible completion regret;
- конечный \(\beta\): плавный компромисс, который не определяется одним hallucinated completion.

Через variational duality:

\[
R_\beta(y,g)=
\sup_{Q\ll P(\cdot\mid y)}
\left\{
\mathbb E_Q[\Delta(X,y,g)]-
\frac{1}{\beta}\mathrm{KL}(Q\Vert P(\cdot\mid y))
\right\}.
\]

Следовательно, score отвечает adversary, который reweight-ит plausible hidden shapes, но платит KL penalty. Аналогичная KL-DRO duality используется в современной robust optimization literature; см., например, lemma и формулу в [Distributionally Robust Q-Learning](https://proceedings.mlr.press/v162/liu22a.html). В FIRE это не RL: одно статическое supervised decision из одного наблюдения.

### 5.1 Sampling-unbiased shifted exponential elicitation

Наивная реализация сначала считает log-mean-exp по всем variants в fiber. Это требует держать их в одном batch и дает biased stochastic gradients при minibatch approximation. Можно elicitate conditional log-moment одним sample variant.

Пусть

\[
z=\exp\{\beta(\Delta-1)\}\in[e^{-\beta},1]
\]

и network output \(f_\theta(y,g,\beta)\in[-\beta,0]\). Определим FIRE loss:

\[
\ell_{\mathrm{FIRE}}(f;\Delta,\beta)
=e^f-zf.
\]

Условный population risk строго выпуклый по scalar \(f\), и

\[
f^*(y,g,\beta)
=\log\mathbb E[z\mid y,g]
=\beta\big(R_\beta(y,g)-1\big).
\]

Поэтому

\[
\widehat R_{\beta,\theta}(y,g)=1+f_\theta(y,g,\beta)/\beta.
\]

Это важно practically: один случайно выбранный hidden variant дает unbiased gradient правильного fiber log-moment. На inference нет ни sampling, ни log-sum-exp по shapes.

Для fixed \(\beta\) можно параметризовать \(f=-\beta\,\sigma(a_\theta)\). Для risk spectrum обучать один head на \(\beta\sim p(\beta)\), задавая \(f(0)=0\) и штрафуя нарушения convexity по \(\beta\). Первую статью лучше строить вокруг одного заранее выбранного \(\beta\) плюс sensitivity curve, чтобы не размывать вклад.

### 5.2 Почему regret, а не absolute quality

Absolute quality учит «насколько хорош grasp». Regret учит «сколько мы теряем именно из-за выбора этого grasp, если скрытая форма окажется такой». Две формы могут быть одинаково сложными в absolute terms, но иметь совершенно разные доступные альтернативы. Oracle gap:

- нормирует inherent difficulty объекта/scene;
- непосредственно обучает ranking относительно альтернатив;
- превращает полную training geometry в privileged teacher, не требуя ее декодировать;
- сохраняет failure cost через reject oracle и знак utility.

Критический baseline — entropic risk absolute loss. Если hindsight gap не дает преимущества при одинаковом encoder и risk transform, central claim не выдержан.

## 6. Как построить finite observation fibers

### 6.1 Exact hidden-only deformation

Для mesh \(x\), camera и occluder вычисляется visibility mask. Строится smooth spatial mask \(m(p)\), равная нулю на visible surface и в safety band около occlusion boundary, и единице глубоко в occlusion shadow. Hidden geometry меняется только полем

\[
p' = p + m(p)\,d_\phi(p),
\]

где \(d_\phi\) — ограниченное diffeomorphic deformation либо watertight primitive graft/cut. Variant принимается, только если:

- повторный RGB-D render отличается не больше sensor tolerance;
- mesh watertight и не self-intersecting;
- stable shelf pose сохраняется;
- изменение реально меняет хотя бы один hidden-contact grasp label, иначе variant малоинформативен.

Типы hidden variations должны быть физически осмысленными: backside thickness, скрытая concavity, rim, handle continuation, taper, asymmetric bulge. Visible front не меняется.

### 6.2 Natural near-fibers

Чтобы exact deformations не задали искусственный prior, нужен второй источник. Среди большого CAD corpus ищутся shapes, чьи renders под данной маской совпадают по visible depth/RGB/normal features, но hidden geometry различается. Variants получают веса

\[
w_j\propto p_{\text{shape}}(x_j)
\exp\left[-d(\mathcal A_c(x_j),y)^2/(2\sigma^2)\right].
\]

Loss легко расширяется: conditional moment использует weighted expectation. Exact fibers дают controlled identifiability test; natural near-fibers — realism.

### 6.3 Почему это не shape completion

Finite shapes существуют только в offline data generation. Test-time model:

- не кодирует mesh latent с geometry decoder;
- не запрашивает occupancy в 3D points;
- не рендерит hidden surfaces;
- не выбирает или усредняет completions;
- не оптимизирует grasp на reconstructed geometry.

Он учит только pushforward latent fiber через scalar decision regret.

## 7. Architecture: FiberFormer

Архитектура намеренно компактна и query-conditioned.

### 7.1 Observation tokens

Из RGB-D строятся три sparse token set:

1. visible target points с RGB, normal estimate и depth confidence;
2. visible foreground-occluder points;
3. camera-ray/occlusion-boundary tokens, описывающие known free space и направление occlusion shadow.

Это не voxel grid и не SDF. Ray tokens важны: одинаковый visible target при разных foreground masks задает разные fibers.

### 7.2 Shared encoder

SE(3)-equivariant либо carefully canonicalized point transformer кодирует target глобально и локально. Global token нужен для learned shape prior; point tokens сохраняют metric geometry. RGB branch допускается, но depth-only ablation обязательна.

### 7.3 Gripper-query decoder

Candidate \(g=(R,t,w)\) представляется небольшим фиксированным набором anchor points на двух пальцах, inner closing slab и palm, перенесенных в camera frame. Gripper anchors cross-attend к observation tokens. Они служат queries, но **не запрашивают occupancy** и не рендерят local surface.

Decoder выдает \(a_\theta(y,g,\beta)\), затем bounded head \(f=-\beta\sigma(a)\) и \(R=1+f/\beta\).

### 7.4 Вычислительная сложность

Observation кодируется один раз. После этого каждый grasp требует только несколько query tokens. Нет \(64^3\) grid, marching cubes, diffusion completion или \(M\) forward passes. Ожидаемая inference cost — один encoder pass плюс batched candidate decoder. Заявлять speed advantage можно только после wall-clock/memory измерений против TARGO-Net, NeuGraspNet и MC-completion baseline.

## 8. Проверяемые теоретические результаты

Минимальный theory package для ICLR:

### Proposition 1: Fisher consistency FIRE loss

Для bounded \(\Delta\) условный minimizer \(\mathbb E[\ell_{\mathrm{FIRE}}\mid y,g]\) единственен и равен \(f^*=\log\mathbb E[e^{\beta(\Delta-1)}\mid y,g]\). Доказательство — производная \(e^f-\mathbb E[z]\) и положительная вторая производная \(e^f\).

### Proposition 2: finite-fiber concentration

Для \(m\) iid variants и \(\Delta\in[0,1]\), поскольку \(e^{\beta\Delta}\in[1,e^\beta]\), Hoeffding и Lipschitz log на \([1,\infty)\) дают с probability \(1-\eta\) bound порядка

\[
|\widehat R_\beta-R_\beta|
\le
\frac{e^\beta-1}{\beta}
\sqrt{\frac{\log(2/\eta)}{2m}}.
\]

Это также показывает цену слишком большого \(\beta\): sample complexity экспоненциально ухудшается.

### Proposition 3: decision regret from uniform estimation

Если \(\sup_g|\widehat R_\beta(y,g)-R_\beta(y,g)|\le\varepsilon\), то выбранный grasp имеет entropic regret не более optimum + \(2\varepsilon\). Это стандартный argmin stability proof, но он связывает learning error с конечным decision.

### Proposition 4: decision sufficiency, не reconstruction sufficiency

Для bounded \(\Delta\) функция \(\beta\mapsto\log\mathbb E[e^{\beta\Delta}\mid y,g]\) на любой окрестности нуля однозначно определяет conditional distribution oracle gap. Значит risk operator сохраняет всю информацию, нужную для любого analytic entropic risk в данном action space, но может отбрасывать произвольные latent shape details, не меняющие regret.

### Proposition 5: impossibility/strict separation toy model

Нужен конструктивный пример двух observation-equivalent shapes и трех actions, где:

- deterministic completion + plan выбирает катастрофический grasp при одной из форм;
- expected score выбирает grasp с меньшим mean error, но большим tail oracle regret;
- FIRE выбирает compromise либо reject;
- никакая deterministic reconstruction, оптимальная по Chamfer, не обязана быть decision-optimal.

Последний пункт должен быть сформулирован как theorem/counterexample: geometry distortion и decision regret не имеют общего monotone ordering.

## 9. Экспериментальный дизайн, способный подтвердить или убить идею

### 9.1 Datasets

1. **Occlusion-Twins controlled set.** Exact fibers из 8–32 hidden variants с идентичным render. Train/val/test split по base shapes и hidden deformation families.
2. **TARGO single-occluder subset.** Фильтр сцен с одним foreground object; same candidate pools; evaluation по visibility bins. TARGO является главным published benchmark, хотя исходно включает clutter.
3. **Natural CAD near-fibers.** ShapeNet/Objaverse-like meshes с matching visible renders и различным hidden geometry.
4. **Real twin objects.** 3D-printed пары с одинаковой видимой передней частью и разной backside/contact geometry плюс household objects; один фиксированный occluder, wrist camera, 10+ повторов каждого grasp condition.

### 9.2 Baselines

Все learned baselines получают одинаковый encoder budget и candidate set:

- visible-only analytic/PointNet scorer;
- BCE success / MSE mean utility (Dex-Net-like direct evaluation);
- heteroscedastic Gaussian и quantile/CVaR direct regression;
- entropic absolute-loss head;
- pairwise/listwise ranking loss;
- single deterministic completion + planner;
- MC completion + mean, LCB и CVaR;
- TARGO-Net;
- NeuGraspNet/local occupancy, если удается честно адаптировать;
- robust DFL-inspired worst-case regret baseline;
- full-geometry oracle и candidate-set oracle ceiling.

### 9.3 Главные metrics

- physical success after minimal lift;
- success@attempt и coverage при наличии reject;
- worst-decile и worst-visibility-bin success;
- expected и 90/95%-tail hindsight regret against full-shape oracle;
- calibration of predicted risk within visibility bins;
- regret under depth noise, mask error и unseen hidden-deformation family;
- inference latency, GPU memory и number of geometry queries.

Обычный mean AP недостаточен: central claim именно tail risk under ambiguity.

### 9.4 Обязательные ablations

- один shape per observation против finite fiber;
- random shape augmentation против measurement-null augmentation;
- label-preserving invariance loss против FIRE aggregation;
- absolute loss против oracle gap;
- \(\beta=0\), несколько finite \(\beta\), near-worst-case;
- без occlusion-ray tokens;
- без global target token;
- RGB-D против depth-only;
- exact fibers против natural near-fibers;
- varying number of variants \(m\) для проверки concentration trend;
- fixed candidate pool quality и oracle ceiling.

### 9.5 Go/no-go gates

До большой real-robot кампании направление должно пройти три дешевых gate:

1. На synthetic twins FIRE статистически значимо превосходит mean utility, direct CVaR и MC completion при одинаковом candidate pool.
2. Advantage сохраняется на unseen deformation families, а не только на тех, которыми создан training fiber.
3. Корреляция chosen risk с real minimal-lift failure лучше, чем у analytic metric и predicted success.

Если любой gate провален после честного tuning, paper claim следует сузить или направление закрыть. Особенно разрушительный результат: entropic absolute loss равен FIRE; тогда oracle gap не приносит нового знания.

## 10. Почему потенциальный SOTA правдоподобен, но не гарантирован

Косвенные свидетельства складываются в непротиворечивую цепочку:

1. [TARGO](https://targo-benchmark.github.io/) показывает большой performance drop существующих grasp models с ростом occlusion и полезность occlusion-induced negative supervision.
2. [Robust planning over uncertain completions](https://research.aalto.fi/en/publications/robust-grasp-planning-over-uncertain-shape-completions/) показывает, что учет нескольких plausible shapes статистически улучшает grasp quality/success относительно одной completion.
3. [CenterGrasp](https://arxiv.org/abs/2312.08240) показывает, что training shape prior способен дать grasps на невидимой стороне и большой gain в grasp success, то есть hidden geometry не полностью непредсказуема из распределения.
4. [NeuGraspNet](https://roboticsproceedings.org/rss20/p046.pdf) показывает важность grasp-local geometry interaction и сильную real-world performance, но платит за implicit reconstruction/rendering.
5. [Dex-Net 2.0](https://www.roboticsproceedings.org/rss13/p58.html) и [Get a Grip](https://proceedings.mlr.press/v270/lum25b.html) подтверждают, что discriminative evaluators, обученные на масштабных physics labels, могут быть практичнее generative planning.
6. [DF²](https://proceedings.mlr.press/v286/kong25a.html) поддерживает general principle прямого обучения downstream objective вместо полной predictive distribution.

FIRE пытается занять незаполненное пересечение: benefits of shape-distribution marginalization + direct discriminative inference + decision-focused tail regret, но без test-time geometry. Это только plausibility argument. Настоящее превосходство должно быть показано на TARGO subset, occlusion twins и hardware.

## 11. Novelty matrix

| Method family | Test-time hidden geometry | Несколько hypotheses | Что учится | Risk under ambiguity | Oracle-relative | Exact same-observation fibers |
|---|---:|---:|---|---:|---:|---:|
| Varley/full completion | full voxel/mesh | нет | geometry | нет | нет | нет |
| Lundell 2019 | sampled voxel completions | да | geometry + analytic score | joint samples | нет | нет |
| GIGA/NeuGraspNet/local occupancy | implicit/local surface | обычно нет | geometry features + quality | обычно нет | нет | нет |
| TARGO-Net | completed target + scene fusion | нет | completion + grasp | implicit | нет | нет |
| Dex-Net/GQ-CNN | нет | нет | expected success/robustness | expected nuisance | нет | нет |
| UNCLE-Grasp | sampled completions | да | completion + LCB | да | нет | нет |
| Generic robust DFL | predicted coefficients/uncertainty set | иногда | downstream regret | worst-case/DRO | да | нет measurement fiber |
| **FIRE-Grasp** | **нет** | **только offline fibers** | **conditional entropic hindsight gap** | **KL-DRO/tail** | **да** | **да** |

Честная novelty claim должна звучать узко:

> Насколько показал поиск, ранее не была сформулирована supervised inverse-decision задача, где hidden, measurement-equivalent physical states группируются в observation fibers, а query-conditioned model через sampling-unbiased exponential loss непосредственно elicitate-ит entropic hindsight regret grasp без reconstruction или sampling latent states на inference.

Нельзя заявлять новыми: entropic risk, regret, LUPI, direct grasp scoring, point transformers, cross-attention или null-space augmentation сами по себе.

## 12. Adversarial ICLR 2027 audit

Официальный [ICLR 2027 Reviewer Guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines) предлагает проверять: конкретность вопроса, мотивацию и placement в литературе, поддержку claims и significance/new knowledge; SOTA сам по себе не обязателен. [Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers) явно включает uncertainty quantification, probabilistic methods, optimization, learning theory и robotics applications.

### Конкретный вопрос

Можно ли learn-ить decision-risk hidden geometry непосредственно на equivalence classes измерительного оператора, не восстанавливая latent state, и тем самым надежнее выбирать parallel-jaw grasp при occlusion?

Оценка: сильный и ясный вопрос.

### Technical novelty

Сильная часть: observation-fiber construction с decision-changing null transformations + oracle-gap risk elicitation + no-inference-sampling.

Слабая часть: каждый компонент имеет соседей; reviewer может назвать работу «entropic risk on regret labels with a point transformer».

Что необходимо: formal separation от expected score/reconstruction, доказательство elicitation, general inverse-decision formulation и хотя бы один non-grasp toy benchmark.

Текущая оценка до экспериментов: 3.5/5.

### Significance

Сильная: TARGO подтверждает реальный unsolved degradation under occlusion; framework может обобщиться на masked imaging, tomography-guided intervention и inspection decisions.

Слабая: узкая single-object shelf setup может выглядеть как robotics niche.

Что необходимо: abstract/theory писать про inverse decisions on observation fibers, grasping — demanding primary instantiation.

Текущая оценка: 4/5 при наличии cross-domain controlled experiment, иначе 3/5.

### Correctness/soundness

Сильная: population optimum loss и KL dual просты и проверяемы.

Риски: finite fiber может не соответствовать real conditional shape distribution; analytic utility может плохо коррелировать с hardware success; exact-render invariance легко нарушить RGB cues.

Что необходимо: weighted natural fibers, sensor-tolerance tests, utility–hardware correlation, confidence intervals и preregistered primary metric.

Текущая оценка: 3/5 до validation.

### Empirical rigor

Нужны одинаковые candidate pools, strong recent baselines, unseen-object/hidden-family splits, hardware twins, latency и failure taxonomy только как analysis, не как method.

Без TARGO-Net, MC-completion, mean direct scorer, direct-CVaR и full-shape oracle paper следует ожидать reject.

### Broad interest

Нужен general recipe:

1. задать measurement operator;
2. sample/generate its approximate fibers;
3. вызвать privileged oracle на actions;
4. elicitate risk of oracle gap;
5. принимать decision без latent reconstruction.

Показать второй controlled domain: например, masked 2D structural design, где одинаковая проекция скрывает разные внутренние дефекты, а action — выбор load/test location. Это general ML/physics, не RL.

### Overall acceptance potential

- Только идея + TARGO experiment: borderline/reject.
- Идея + solid theory + twins + strong baselines: plausible ICLR poster.
- Плюс real robot, cross-domain task и явная SOTA under severe occlusion: сильный ICLR candidate.

Никаких оснований обещать acceptance или SOTA до результатов нет.

## 13. Самые вероятные reviewer objections и ответы, которые должны быть заработаны экспериментом

1. **«Это просто transformed regression target».**  
   Ответ должен опираться не на риторику, а на fiber benchmark, unbiased elicitation и separation theorem; сравнить с MSE regret, quantile и distributional heads.

2. **«Fibers синтетические и не являются posterior».**  
   Нужны natural near-fibers, prior weights, real twins и sensitivity к misspecified variant distribution.

3. **«Почему regret лучше absolute risk?»**  
   Обязательная matched ablation и examples, где oracle-relative normalization меняет решение полезно.

4. **«Completion дает больше информации и может выполнить любую downstream задачу».**  
   Да, но при fixed data/compute она решает более трудную high-dimensional problem. Нужно показать better grasp regret, calibration, latency и memory при одинаковом training corpus.

5. **«Candidate generator определяет результат».**  
   Одинаковые frozen candidate sets, oracle ceiling и отдельная recall metric.

6. **«Entropic risk очень чувствителен к \(\beta\) и samples».**  
   Theory bound, fixed preregistered \(\beta\), sensitivity curve и comparison с CVaR.

7. **«Force-closure metric не равен реальному lift success».**  
   Perturbation-averaged terminal utility, calibration subset и real minimal-lift trials.

8. **«Robust DFL 2026 уже делает worst-case regret».**  
   Четко отделить coefficient-prediction DRO от measurement-fiber latent ambiguity, sampling-unbiased conditional log-moment learning и reconstruction-free physical inverse decision.

## 14. Минимальный план реализации

### Phase 1: falsification prototype

- procedural exact twins;
- 64–256 candidates на observation;
- frozen point transformer encoder;
- mean utility, MSE regret, entropic absolute и FIRE heads;
- analytic terminal closure labels;
- 5 seeds, paired bootstrap CI.

### Phase 2: benchmark-grade simulation

- natural near-fibers;
- TARGO one-occluder subset;
- comparison с TARGO-Net/NeuGraspNet/MC completion;
- unseen-object и unseen-hidden-variation splits;
- latency/memory.

### Phase 3: theory and generality

- propositions 1–5;
- controlled non-robotics inverse-decision task;
- misspecified fiber analysis.

### Phase 4: hardware

- printed twins и household objects;
- wrist RGB-D, one foreground obstacle;
- minimal lift, no transport;
- randomized object pose/noise, repeated trials;
- coverage–success and tail-regret reporting.

## 15. Что именно является ожидаемым substantial knowledge

Успешная статья должна установить не только новый score, но четыре эмпирико-теоретических факта:

1. Geometry metrics на completion могут улучшаться без улучшения decision regret; hidden-shape reconstruction является избыточной и иногда неверно направленной supervision для grasp selection.
2. Measurement-null augmentations нельзя автоматически считать invariances: если downstream label меняется, правильный объект обучения — distribution/risk по fiber.
3. Hindsight oracle gap является более transferable target, чем absolute grasp success, когда latent state меняет множество хороших альтернатив.
4. Conditional entropic regret можно elicitate одним latent sample на update и вычислять одним forward pass на inference, сохраняя KL-DRO interpretation.

Если установлены только performance gains архитектуры, вклад недостаточно general для заявленной цели.

## 16. Disclosure и процессуальная заметка

Существующие файлы проекта и прежние occlusion ideas при подготовке этого отчета не читались; поэтому независимость соблюдена, но прямую проверку текстового пересечения с ними выполнить логически невозможно. Метод специально выведен вдали от completion/occupancy/LCB/causal/full-cycle линий, которые были обозначены как нежелательные.

Для ICLR 2027 потребуется AI disclosure: [официальная policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors) требует раскрывать использование generative AI для conceptual frameworks, mathematical claims, hypotheses, methodology и experiments. Авторы остаются ответственными за ручную проверку novelty, доказательств, кода и всех factual claims.

## 17. Итоговая формулировка paper-level идеи

> **FIRE-Grasp studies grasp selection as fiberwise inverse decision learning.** Training-time full geometry is used only to construct multiple physically plausible hidden states that are indistinguishable under the same occluded RGB-D measurement and to compute each candidate’s hindsight gap to a full-information terminal-grasp oracle. A grasp-conditioned FiberFormer then minimizes a sampling-unbiased exponential elicitation loss whose population solution is the conditional entropic oracle regret. The resulting model selects or rejects parallel-jaw grasps in a single forward pass, without shape completion, occupancy supervision, latent sampling, RL, VLA, or full-cycle prediction.

Самая короткая testable claim:

> При одинаковом candidate recall и training shape prior прямое обучение tail hindsight regret на observation fibers даст меньший worst-bin physical grasp failure и меньшую latency, чем deterministic completion, MC completion и direct expected-success scoring, особенно когда скрытые стороны объектов неоднозначны, но их видимые RGB-D проекции совпадают.

