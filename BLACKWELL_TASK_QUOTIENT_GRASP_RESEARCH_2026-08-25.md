# Predict the outcome field, not the hidden shape

## Blackwell-consistent task-quotient posterior processes for reliable parallel-jaw grasp selection under occlusion

**Статус документа:** исследовательская гипотеза и план её фальсификации, а не обещание SOTA или принятия в ICLR.  
**Дата среза литературы:** 25 августа 2026.  
**Рабочее имя метода:** **TQ-Grasp** (Task-Quotient Grasp Process).  
**Рабочее имя learning objective:** **BTPS** (Blackwell-Tower Proper Score).

---

## 1. Итог в одном абзаце

Наиболее сильная найденная постановка — не реконструировать скрытую часть объекта и не выдавать один детерминированный grasp score, а непосредственно учить **условный posterior случайного grasp-outcome field**

\[
\Pi_o = \mathcal L\big(M_S(\cdot)\mid O=o\big),
\]

где \(S\) — полная, но скрытая геометрия объекта, \(O\) — единственное шумное RGB-D наблюдение с foreground-окклюзией, а \(M_S(g)\) — один signed margin локальной устойчивости parallel-jaw grasp \(g\). Полные формы \(S,S'\) отождествляются, если они дают одинаковую функцию \(M(\cdot)\) на допустимых grasps; поэтому модель учит posterior не в пространстве форм, а в гораздо более узком **task quotient**. Архитектура с одним общим latent sample на целое поле выдаёт согласованные samples margins только в запрошенных grasp poses и никогда не декодирует voxel/SDF/mesh. Новый objective состоит из strictly proper energy/variogram score по случайным конечным наборам grasp-запросов и условного moment restriction, заставляющего предсказанные posteriors удовлетворять tower property при физически корректном garbling наблюдения дополнительной окклюзией. Это даёт general-ML тезис: *при частичной наблюдаемости следует амортизировать posterior минимального decision-complete функционала скрытого мира и обучать его быть согласованным с порядком информации, а не амортизировать весь скрытый мир*.

Это top-1 не потому, что shape completion заведомо хуже: напротив, свежие результаты делают shape-first подход самым опасным baseline. Идея сильна только если совместно подтвердятся три утверждения: (i) на одинаковом candidate pool task-quotient posterior даёт не худший top-1 outcome и лучший selected-grasp tail risk; (ii) BTPS действительно улучшает high-occlusion calibration сверх обычного proper scoring; (iii) это достигается заметно дешевле multi-sample completion. Candidate recall — отдельное свойство proposer, которое нельзя приписывать selector.

---

## 2. Жёсткая граница задачи

### Входит

- один target object на полке;
- один или несколько foreground-элементов только как источник окклюзии/локального препятствия, но не clutter-removal;
- одно RGB-D наблюдение с wrist camera;
- умеренно шумный depth/point cloud;
- неизвестная hidden grasp-relevant geometry, prior которой доступен только через обучающее распределение полных форм;
- parallel-jaw gripper;
- выбор одного grasp из observation-measurable candidate set;
- локальный захват и очень малый контрольный подъём как hardware-метрика.

### Не входит

- RL, VLA, active view selection, removal/pushing occluder;
- оценка всего цикла approach–IK–trajectory–full lift;
- causal taxonomy failure modes;
- полный SDF/occupancy/mesh scene reconstruction;
- dense clutter как предмет исследования;
- language/task-oriented grasping.

Чтобы не скрыть motion-planning задачу внутри learning target, путь руки и глобальная кинематическая достижимость должны фильтроваться одинаковым внешним модулем для всех методов. TQ-Grasp оценивает только grasp-local outcome при фиксированном terminal pose и малых execution perturbations.

---

## 3. Что уже занято и где остаётся gap

### 3.1 Прямые amodal grasp predictors уже существуют

[S4G](https://proceedings.mlr.press/v100/qin20a.html) уже формулирует amodal single-view, single-shot SE(3) grasp detection для parallel gripper и напрямую предсказывает grasp proposals из single-view point cloud. [Contact-GraspNet](https://arxiv.org/abs/2103.14127) также напрямую генерирует распределение 6-DoF grasps из partial scene cloud и достигает высокой эффективности, привязывая grasp representation к наблюдаемым contact points. Поэтому «direct grasp prediction без completion» само по себе не ново.

Gap не в наличии direct predictor, а в том, что эти методы обычно выдают proposal/quality как детерминированный conditional output или распределение **поз**, не posterior того, как один и тот же grasp сработает при альтернативных скрытых геометриях, совместимых с наблюдением.

### 3.2 Geometry supervision и completion реально помогают при occlusion

[GIGA](https://roboticsproceedings.org/rss17/p024.pdf) совместно учит occupancy и grasp affordance. В packed scenes geometry supervision даёт около 5 percentage points над affordance-only ablation; авторы связывают преимущество именно с частично видимыми объектами. При этом GIGA-Aff без reconstruction всё равно превосходит VGN, а continuous implicit affordance query оказывается сильнее voxel-snapped predictor. Это одновременно подтверждает две вещи:

1. скрытая геометрическая prior-информация нужна;
2. её не обязательно выдавать как финальную reconstruction, если task field достаточно выразителен.

[TARGO/TARGO-Net](https://arxiv.org/abs/2407.06168) — ближайший benchmark по occlusion. TARGO-Net сначала completes target, затем совместно рассуждает о target и scene. На высокой occlusion ablation без shape completion теряет до 18%, а полный метод остаётся около 85% synthetic success. Однако paper также сообщает больший real-world drop (14% против 7% в simulation) и прямо связывает разрыв с real noise, затрудняющим completion. Это главный empirical gap для нового метода: сохранить prior benefit без обязательного точного геометрического decoding.

[ZeroGrasp](https://cvpr.thecvf.com/virtual/2025/poster/32440) одновременно восстанавливает 3D geometry и grasps, используя 1M photorealistic images и 8.9B grasp annotations; это сильный современный reconstruction-based comparator. [Single-View Shape Completion for Robotic Grasping in Clutter](https://arxiv.org/abs/2512.16449) сообщает +23 p.p. к partial-cloud baseline и +19 p.p. к ZeroGrasp в preliminary real experiments, но полный pipeline занимает примерно 4–5 s, из которых около 3 s уходит на completion.

Самый неприятный контраргумент — свежая работа [Object Pose and Shape Estimation for Grasping: Does it Work?](https://arxiv.org/abs/2605.26944): три modular shape-first метода превосходят AnyGrasp во всех проведённых сравнениях по collision-free generation, force closure и stability. Следовательно, статья TQ-Grasp не может строиться на лозунге «reconstruction — лишняя». Проверяемый тезис должен быть уже: **для posterior tail-risk одного grasp task-quotient является достаточным и потенциально статистически/вычислительно выгоднее полной формы, но shape-first может оставаться лучше по proposal recall и transfer**.

### 3.3 Uncertain completion + robust aggregation тоже уже заняты

[Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645) семплирует voxel completions с MC dropout и оценивает grasp на всех samples; авторы показывают статистически значимое преимущество над single completion. [Shape Completion with Prediction of Uncertain Regions](https://arxiv.org/abs/2308.00377) учит uncertain regions и улучшает grasp quality, избегая их. [Measuring Uncertainty in Shape Completion to Improve Grasp Quality](https://arxiv.org/abs/2504.16183) делает 60 completion passes, штрафует grasp scores локальной variance и сообщает +7 p.p. над deterministic completed-cloud pipeline, но около 6 s на весь pipeline. [UNCLE-Grasp](https://arxiv.org/abs/2601.14492) уже использует multiple incompatible completions, force-closure metrics, lower confidence bound и abstention.

Поэтому следующие формулировки **не проходят novelty audit**:

- MC dropout completions + mean/std/LCB;
- CVaR по набору reconstructed shapes;
- uncertainty penalty в grasp score;
- «избегать uncertain completed regions»;
- completion diffusion + существующий grasp detector.

### 3.4 Точный gap

В проверенной литературе нашлись:

- deterministic direct amodal grasp fields;
- full/shell/point/voxel shape completion с grasp head;
- sampling full completions и robust aggregation;
- generic direct expected-cost amortization;
- generic stochastic-process models;
- generic measure-valued martingale calibration.

Не была найдена работа, которая одновременно:

1. определяет скрытый объект только через equivalence class его **полной action-outcome function**;
2. учит **posterior process** этой функции непосредственно из partial observation и simulator queries;
3. использует strictly proper finite-marginal objective, а не pointwise BCE/mean regression;
4. навязывает корректную tower/Blackwell consistency между physically nested occlusions;
5. делает это для continuous parallel-jaw grasp queries без shape decoder.

Это и есть defendable novelty claim. Он должен формулироваться как «first combination/instantiation under checked literature», а не как абсолютное «first uncertainty-aware occluded grasping».

---

## 4. Отброшенные направления

| Вариант | Почему сначала выглядел разумно | Почему отброшен |
|---|---|---|
| Completion ensemble + LCB/CVaR | Явно представляет hidden-shape ambiguity | Уже сделано Lundell et al., UNCLE-Grasp и uncertainty-scoring работами; дорого; reconstructs more than needed |
| Один uncertainty head над grasp score | Очень дёшево | Почти обычная heteroscedastic regression/calibration; не представляет multimodal hidden hypotheses и joint action structure |
| Direct probability \(P(success\mid O,g)\) | Для forced single binary decision теоретически достаточно | S4G/direct grasp scoring уже близки; нет нового general-ML объекта. Остаётся обязательным strong baseline и возможным falsifier TQ-Grasp |
| Contact-signature posterior | Декодирует только две local contact sections, физически интерпретируем | Для nonconvex geometry и finite finger pads signature быстро разрастается; это всё ещё локальная reconstruction с error propagation. Хороший runner-up, но слабее broad-ML story |
| Random feasible-grasp set / capacity functional | Элегантная random-set математика | Hitting probabilities не дают автоматически надёжность конкретного выбранного grasp; continuous set learning и adaptive refinement усложняют первую paper |
| Только martingale/tower loss над scalar scores | Новый information-order regularizer | В population обычный proper BCE на всех mask levels уже восстанавливает conditional means; без posterior-process target вклад легко назвать redundant regularization |
| Морфологическая erosion feasible grasp set | Естественная robustness к pose error | Родственно классическому shrinking operation-space и robust grasp metrics; novelty недостаточна |

Итоговый TQ-Grasp сохраняет сильные части двух последних general-ML направлений — task quotient и information consistency — но не пытается реконструировать contacts или форму.

---

## 5. Новая broad идея: Task-Quotient Posterior Process

### 5.1 Скрытый мир и наблюдение

Пусть

- \(S\in\mathcal S\) — полное состояние target geometry в camera/shelf frame;
- \(O\in\mathcal O\) — noisy RGB-D target observation вместе с target mask, foreground cloud и depth-validity mask;
- \(\overline{\mathcal G}\subset SE(3)\times [w_{\min},w_{\max}]\) — compact global domain допустимых terminal grasps;
- \(G(O)\subset\overline{\mathcal G}\) — finite candidate set, построенный **только** из \(O\);
- \(g=(R,t,w)\) — parallel-jaw terminal grasp pose и width.

В training simulator полная форма доступна только для построения supervision. Модель её не получает и не восстанавливает.

### 5.2 Один, а не составной learning target

Определим один bounded signed margin

\[
M_S(g)\in[-1,1].
\]

Он должен вычисляться единым grasp-local evaluator: например, normalized signed force-closure/stability margin после jaw closure, усреднённый или минимизированный по малому фиксированному набору calibration/pose perturbations. Отрицательное значение означает локально неустойчивый grasp; положительное — запас. Terminal collision с уже наблюдаемым shelf/foreground лучше фильтровать одинаковым детерминированным модулем до модели, чтобы не превращать \(M\) в длинный список failure variables.

На hardware primary label — удержал ли gripper target после короткого 1–2 cm lift. Модель не оценивает arm approach, IK или полный lift.

### 5.3 Quotient скрытых форм

Каждая полная форма индуцирует функцию

\[
T(S)=M_S(\cdot):\overline{\mathcal G}\rightarrow[-1,1].
\]

Вводим отношение

\[
S\sim_{\overline{\mathcal G}}S'
\quad\Longleftrightarrow\quad
M_S(g)=M_{S'}(g)\;\;\forall g\in\overline{\mathcal G}.
\]

То есть две формы считаются одинаковыми, если никакой допустимый parallel-jaw grasp не различает их по локальному outcome. Все back-side детали, которые не меняют ни один доступный grasp, исчезают автоматически. Это не latent shape compression по heuristic bottleneck, а exact task-defined quotient.

### 5.4 Что именно предсказывает модель

TQ-Grasp учит

\[
\Pi_o := \mathcal L(T(S)\mid O=o),
\]

условное распределение на пространстве функций. На практике запрашиваются finite marginals:

\[
\Pi_o^{G}=\mathcal L\left(
[M_S(g_1),\ldots,M_S(g_m)]\mid O=o
\right).
\]

Важно различать три распределения:

- distribution of grasp poses — что генерирует большинство grasp generators;
- posterior of hidden shape — что генерирует completion/SBI;
- posterior of grasp outcomes over a common action domain — предлагаемый объект.

TQ-Grasp относится к третьему.

### 5.5 Почему не достаточно expected score

Для одного фиксированного binary loss и risk-neutral forced pick действительно достаточно \(p_g=P(Y_g=1\mid O)\). Это не следует замалчивать. Полный task posterior нужен, когда хотя бы одно из следующего является частью paper:

- grasp quality — continuous margin, а не Bernoulli;
- необходим lower-tail criterion, chance constraint или смена risk level без retraining;
- требуется calibrated abstention;
- training labels доступны лишь для sparse sets of grasps, а shared functional latent даёт статистическое sharing;
- selection идёт по сотням candidates и нужна coherent, а не независимая, uncertainty field.

Если independent per-grasp quantile model достигает той же selected-grasp success/calibration, TQ-process complexity не оправдана. Это заранее объявленный falsifier.

---

## 6. Архитектура TQ-Grasp

### 6.1 Observation encoder без voxel/SDF

Вход разделяется на два point sets:

\[
P_{tar}=\{(x_i,c_i,n_i,v_i,r_i)\},\qquad
P_{ctx}=\{(x_j,c_j,v_j,r_j)\},
\]

где \(x\) — 3D point, \(c\) — RGB feature, \(n\) — normal для валидных target points, \(v\) — depth-validity/noise feature, \(r\) — camera-ray/occlusion-boundary feature. Target и foreground/shelf имеют разные type embeddings.

Sparse point transformer кодирует:

- global evidence token \(h_o\), несущий prior о hidden shape;
- local visible tokens \(H_o=\{h_i\}\), используемые grasp queries.

Никакого dense scene grid не создаётся.

### 6.2 Общий latent hidden-outcome hypothesis

Для каждого posterior sample:

\[
\epsilon_k\sim\mathcal N(0,I_d),\qquad
z_k=A_\theta(\epsilon_k;h_o),
\]

где \(A_\theta\) — малый conditional implicit generator или 4–6 layer conditional flow, \(d\approx16\text{–}32\). Один \(z_k\) используется для **всех** grasp queries данной сцены. Поэтому sample \(k\) означает одну согласованную hypothesis в task quotient, а не независимый шум каждого grasp.

### 6.3 Projective query decoder

Для каждого \(g_j\):

1. nearby visible tokens переводятся в gripper frame;
2. query cross-attends к target/context tokens в terminal closing volume и camera occlusion cone;
3. shared decoder выдаёт

\[
\widehat M^{(k)}(g_j)=D_\theta(g_j,H_o,h_o,z_k).
\]

Для набора \(G=\{g_1,\ldots,g_m\}\):

\[
F_\theta(O,G,z_k)=
[D_\theta(O,g_1,z_k),\ldots,D_\theta(O,g_m,z_k)].
\]

Между candidate queries **нет self-attention**. Это намеренное ограничение:

- permutation equivariance выполняется автоматически;
- marginal prediction для \(g\) не меняется, если рядом добавили другой query;
- finite marginals projectively consistent по конструкции;
- вычисление батчится как \(O(Km)\).

Корреляция между grasps проходит через общий \(z_k\) и общий observation representation. Если этой ёмкости недостаточно, \(z_k\) может параметризовать low-rank adapters query decoder, не нарушая projectivity.

### 6.4 Candidate interface

Первая статья должна изолировать **selection**, а не смешивать её с proposal recall:

- frozen observation-only proposer строит \(m=256\text{–}1024\) candidates;
- один и тот же pool получают все ranking baselines;
- отдельно публикуется oracle success@pool, чтобы отсутствие хорошего grasp не приписывать selector;
- для tower training один и тот же \(G\) строится из более coarse observation и затем подаётся и coarse, и fine branch.

Можно использовать union из visible-contact anchored proposals и широкого object-bounding-frustum sampler. Сильные published proposers допустимы как baseline interface, но не должны объявляться частью novelty.

---

## 7. Новый learning objective: Blackwell-Tower Proper Score

### 7.1 Proper finite-marginal learning

Для training shape \(S_i\), observation \(o_i\) и random query bundle \(G_i=\{g_j\}_{j=1}^m\) simulator даёт

\[
y_i^G=[M_{S_i}(g_1),\ldots,M_{S_i}(g_m)].
\]

Модель генерирует \(K\) samples \(\hat y^{1:K}\). Базовая loss — unbiased ensemble energy score при \(0<\beta<2\):

\[
\mathcal L_{ES}=
\frac1K\sum_{k=1}^K
\|W_G(\hat y^k-y)\|_2^\beta
-\frac{1}{2K(K-1)}
\sum_{k\ne l}
\|W_G(\hat y^k-\hat y^l)\|_2^\beta.
\]

Первый член обеспечивает accuracy, второй не даёт samples схлопнуться. \(W_G\) зависит только от observation/query sampling density, но не от realized outcome; importance weights корректируют oversampling около high-quality и decision-boundary grasps. Для \(\beta=1\) score strictly proper для distributions с конечным первым моментом. Теория proper scores изложена в [Gneiting & Raftery](https://www.eecs.harvard.edu/cs286r/courses/fall10/papers/Gneiting07.pdf); energy-score training implicit generative networks не требует tractable likelihood.

Поскольку energy score может быть слабо чувствителен к high-dimensional dependence, добавляется небольшой proper variogram score на случайных query pairs. Сумма strictly proper ES и proper variogram score остаётся strictly proper. На каждой итерации достаточно \(m=16\text{–}64\) queries; random bundles покрывают action space за training.

### 7.2 Физически корректные nested occlusions

Из одного fine same-camera observation строится coarse observation:

\[
O^- = \mathcal K(O^+,U),
\]

где \(\mathcal K\) физически накладывает более крупный foreground depth layer, удаляет закрытые target pixels и добавляет calibrated sensor noise. Важно, что coarse signal получается post-processing/garbling fine signal; изменение camera viewpoint не подходит, потому что два view обычно не упорядочены по Blackwell.

Имеем filtrations \(\mathcal F^-\subset\mathcal F^+\). Истинные conditional outcome laws обязаны удовлетворять tower property:

\[
\mathbb E[\Pi_{O^+}^{G}(A)\mid O^-,G]
=\Pi_{O^-}^{G}(A)
\quad\forall A.
\]

Иначе говоря, coarse posterior — mixture fine posteriors, а не копия конкретного fine posterior. Поэтому обычная \(\|\Pi^- - \Pi^+\|\) consistency loss **неверна** и уничтожила бы нужный рост неопределённости.

### 7.3 Trainable tower moment

Пусть \(\varphi(y)\in\mathbb R^r\) — random Fourier features characteristic kernel на margin vectors, а

\[
\mu_\theta(o,G)=\mathbb E_{z}[\varphi(F_\theta(o,G,z))]
\]

— kernel mean embedding predicted finite marginal. Тогда truth удовлетворяет

\[
\mathbb E[
\mu_\theta(O^+,G)-\mu_\theta(O^-,G)
\mid O^-,G]=0.
\]

Условие оценивается как maximum conditional moment restriction:

\[
\mathcal L_{tower}(\theta)
=\sup_{h\in\mathcal H}
\frac{
\left\|
\mathbb E\left[
h(O^-,G)^\top
(\mu_\theta(O^+,G)-\mu_\theta(O^-,G))
\right]
\right\|_2^2
}{\mathbb E\|h(O^-,G)\|_2^2+\varepsilon}.
\]

Практический вариант — bank из fixed random probes над stop-gradient coarse encoder; более сильный — маленький learned critic с alternating updates. [Adversarial GMM](https://arxiv.org/abs/1803.07164) подтверждает, что continuum conditional moment restrictions можно обучать нейросетевым adversary; здесь conditional moment вытекает не из causal IV assumptions, а из exact observation garbling.

### 7.4 Полный objective

\[
\boxed{
\mathcal L_{BTPS}
=\mathcal L_{ES}(O^-)
+\mathcal L_{ES}(O^+)
+\lambda_V\mathcal L_{vario}
+\lambda_T\mathcal L_{tower}
}
\]

Новый вклад — не сам energy score, не martingale fact и не GMM по отдельности, а **proper finite-marginal learning task-quotient process, regularized exact conditional moments induced by a known information-garbling operator**.

### 7.5 Почему это не просто consistency regularization

Если модель истинна, adding \(\mathcal L_{tower}\ge0\) не сдвигает population optimum strictly proper term: truth уже даёт zero tower residual. В finite sample regularizer связывает разные occlusion regimes и запрещает две типичные ошибки:

- coarse observation искусственно уверенно повторяет одну fine hypothesis;
- fine prediction добавляет систематический drift, который не усредняется обратно в coarse belief.

Недавняя работа [Calibrated Probability Forecast Sequences and Measure-Valued Martingales](https://arxiv.org/abs/2606.31621) показывает эквивалентность auto-calibration последовательности probabilistic forecasts и measure-valued martingale property и предлагает тестирование. TQ-Grasp отличается тем, что строит **training objective** для action-indexed posterior fields по управляемой lattice окклюзий и сочетает его с identifying proper score.

---

## 8. Inference и grasp selection

Для test observation:

1. frozen proposer выдаёт \(G(O)\);
2. observed-cloud collision filter удаляет terminal poses, явно пересекающие shelf/foreground;
3. один observation encoding используется для \(K\approx8\text{–}16\) latent samples;
4. для каждого candidate вычисляется lower posterior quantile

\[
s_\alpha(g)=Q_\alpha(M_S(g)\mid O),\qquad \alpha\in[0.05,0.2];
\]

5. выбирается

\[
g^*=\arg\max_{g\in G(O)}s_\alpha(g).
\]

Если \(\max_gs_\alpha(g)\le0\), система может abstain. В forced-pick режиме всё равно берётся argmax; обе метрики следует публиковать.

Optional real-calibration layer может conformalize threshold на held-out hardware data, но не является contribution: decision-aware conformal methods уже быстро развиваются, включая [Utility-Directed Conformal Prediction](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0c6b452f1bbfb6905f6bac957d73b321-Abstract-Conference.html), [Conformal Robustness Control](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a80188f9b1246d4b95d396165cf99207-Abstract-Conference.html) и [action-conditional conformal decision making](https://arxiv.org/abs/2606.05551). Conformalization надо позиционировать только как deployment wrapper.

---

## 9. Теоретический пакет, достаточный для ICLR

### Proposition 1: universal grasp-decision sufficiency

Пусть downstream loss зависит от hidden world только через \(M_S(g)\), а risk functional \(\rho\) law-invariant. Тогда для любого \(g\)

\[
R_\rho(g\mid o)
=\rho\big((\mathrm{ev}_g)_\#\Pi_o\big),
\]

где \(\mathrm{ev}_g(f)=f(g)\). Следовательно, \(\Pi_o\) достаточно для всех mean/quantile/CVaR/chance-constrained selectors на данном action domain.

### Proposition 2: quotient minimality

Если \(\overline{\mathcal G}\) compact/separable, \(M_S(\cdot)\) continuous almost surely и representation \(R(o)\) позволяет восстановить joint law \([M(g_1),\ldots,M(g_m)]\mid o\) для любого конечного query set, то \(R\) определяет \(\Pi_o\) с точностью до almost-sure isomorphism. Ни одна универсально decision-complete representation не может отождествить два разных task-quotient posteriors.

Это отличается от свежего понятия [Bayes quotient](https://arxiv.org/abs/2606.04045), которое для **фиксированных loss и decision problem** может сохранять только optimal action. Здесь сохраняется posterior hidden-world outcome function, минимальный для **семейства** grasp risks и thresholds. В paper это различие надо доказать, а не заявить терминологически.

### Proposition 3: finite-marginal Fisher consistency

Для \(0<\beta<2\) expected energy score уникально минимизируется истинным \(\Pi_o^G\) при каждом \((o,G)\). Если random query distribution имеет dense support, а sample paths continuous, совпадение всех sampled finite marginals определяет stochastic process.

### Proposition 4: BTPS не меняет population truth

Истинный process одновременно минимизирует proper-score terms и обнуляет tower moment. Поэтому при достаточно богатом critic class добавление \(\lambda_T\mathcal L_{tower}\) Fisher-consistent. Нужен отдельный bound: learned violation любой normalized witness moment не превышает \(\sqrt{\mathcal L_{tower}}\).

### Proposition 5: Blackwell implication

Из tower property и Jensen следует, что distribution posterior beliefs при fine observation является mean-preserving refinement coarse beliefs. Для любого bounded decision problem expected Bayes value fine information не ниже coarse. Это даёт не только theorem, но и diagnostic: learned model не должен систематически утверждать, что дополнительное раскрытие target information ухудшает optimal expected decision value.

### Что не стоит обещать

- conditional coverage для каждого конкретного RGB-D кадра без дополнительных assumptions;
- OOD guarantee на shape distribution;
- физическую безопасность всего arm trajectory;
- точное восстановление friction/mass, которых нет во входе;
- «posterior» в субъективно Bayesian смысле вне simulator/data-generating distribution.

---

## 10. Почему метод может работать: косвенные evidence

### 10.1 Task posterior может быть дешевле full latent posterior

[Amortized Bayesian Decision Making](https://arxiv.org/abs/2312.02674) показывает, что expected cost как функция observation/action может близко совпадать с true-posterior decisions и иногда быть точнее решения через misspecified posterior approximation. [Optimal simulation-based Bayesian decisions](https://arxiv.org/abs/2311.05742) сообщает 100–1000× меньше simulator calls, чем posterior-inference/Monte-Carlo alternatives, когда напрямую учится utility surrogate. Это не доказывает преимущество на grasping, но поддерживает главный statistical intuition: inference большого latent state может тратить capacity на decision-irrelevant directions.

TQ-Grasp идёт дальше scalar expected cost: учит law целого outcome field, сохраняя risk sensitivity и совместимость разных risk thresholds.

### 10.2 Function distributions learnable через finite marginals

[Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a/garnelo18a.pdf) формализуют обучение stochastic processes через conditional finite-dimensional predictions. [Neural Diffusion Processes](https://icml.cc/virtual/2023/poster/25009) показывают, что exchangeable finite-marginal architecture может приближать rich function distributions и Bayesian posteriors. Это косвенно подтверждает learnability process target. TQ-Grasp выбирает более дешёвый implicit latent generator + proper score вместо многократного diffusion sampling.

### 10.3 Proper scores дают корректный distributional target

Strictly proper scoring rule мотивирует честное conditional distribution, а repulsive sample term препятствует collapse. Energy score уже применяется к implicit ensembles и имеет Hilbert-space extensions. Это гораздо более defendable objective, чем произвольный «uncertainty penalty» в grasp ranking.

### 10.4 Uncertainty-aware grasping эмпирически лучше point completion

Lundell et al., uncertainty-region prediction, 2025 uncertainty scoring и UNCLE-Grasp независимо показывают пользу агрегирования shape ambiguity, особенно на complex/unknown/high-occlusion objects. TQ-Grasp не доказывает superior uncertainty, но удаляет дорогой intermediate full geometry и оптимизирует именно downstream random variable.

### 10.5 Риск реального опровержения остаётся

GIGA, TARGO, ZeroGrasp, diffusion completion и 2026 shape-first comparison показывают, что geometry supervision улучшает generalization и candidate recall. Возможны два исхода, при которых TQ-Grasp проиграет:

1. full geometry — более сильный inductive bias и требует меньше shapes для transfer;
2. task-outcome simulator labels недостаточно реалистичны, а geometric pretraining лучше переносится sim-to-real.

Поэтому потенциальный SOTA следует обосновывать только после matched-compute и matched-data экспериментов.

---

## 11. Dataset и training protocol

### 11.1 Synthetic distribution

- полные object meshes из ACRONYM/ShapeNet/Objaverse-LVIS с category-disjoint split;
- полка, один target, foreground occluder без pile/clutter dynamics;
- side/oblique wrist-camera RGB-D;
- nested occlusion levels при одной camera pose;
- RealSense-подобный depth noise: quantization, edge dropout, axial noise, missing pixels;
- explicit seen/similar/novel shape split;
- отдельный OOD split с hidden backside modifications.

### 11.2 Counterfactual occlusion twins — критически важный benchmark

Нужно специально строить pairs \((S_a,S_b)\), у которых:

- visible RGB-D under chosen occluder почти одинаков;
- hidden geometry различна;
- хотя бы для одного common candidate \(M_{S_a}(g)>0\), \(M_{S_b}(g)<0\).

Именно здесь deterministic completion, scalar mean, independent quantiles и posterior process имеют различимые predictions. Без такого controlled ambiguity benchmark paper может показать лишь очередное улучшение на natural scenes, не подтверждая центральный scientific claim.

### 11.3 Labels

Для каждого full shape:

- observation-only proposer создаёт candidate pool;
- GPU local grasp evaluator вычисляет continuous margin для random query bundles;
- небольшой subset получает repeated perturbation trials;
- никаких occupancy/SDF/mesh losses TQ-Grasp не использует.

Training curriculum: low→high occlusion использовать допустимо только как sampling schedule, но не как отдельную pipeline. Каждый batch должен содержать paired \((O^-,O^+)\), общий \(G\), один truth margin vector и \(K\) generated samples каждой branch.

### 11.4 Real data

- target objects и occluders, не встречавшиеся в synthetic meshes;
- wrist RGB-D с зафиксированными intrinsics/extrinsics;
- visibility bins;
- randomized object pose и foreground obstacle position;
- 20–30 objects × 3 occlusion levels × не менее 10 trials для primary hardware comparison;
- separate calibration objects, не используемые для reporting.

---

## 12. Baselines и ablations

### Robotics baselines

1. S4G/AnyGrasp/Contact-GraspNet-style direct partial-cloud predictor.
2. GIGA-Aff: continuous direct affordance без geometry head.
3. GIGA: joint occupancy+affordance.
4. TARGO-Net: target completion + grasp detection.
5. ZeroGrasp или strongest available shape-first pipeline.
6. MC-dropout/multi-completion robust aggregation, matched по числу samples.

### На одном frozen candidate pool

1. deterministic mean margin regression;
2. calibrated Bernoulli success probability;
3. independent heteroscedastic Gaussian;
4. independent quantile regression;
5. TQ process + ES, без tower;
6. TQ process + ES + naive pairwise consistency (должен быть хуже на ambiguity);
7. TQ process + full BTPS;
8. BTPS без shared latent;
9. BTPS без local gripper-frame tokens;
10. BTPS при случайном, не physical garbling.

### Обязательный matched-compute protocol

- одинаковое training shape distribution;
- одинаковый number of simulator-evaluated grasps;
- одинаковый candidate pool для selection study;
- отдельный end-to-end comparison, где каждый метод использует свой proposer;
- latency при одинаковом GPU, batch size и number of uncertainty samples;
- параметры и peak memory.

---

## 13. Метрики и decisive experiments

### Primary

- hardware Success@1 короткого lift;
- Success@1 по occlusion bins;
- worst-bin success;
- forced-pick success и selective risk–coverage curve;
- oracle success@candidate-pool.

### Distributional

- held-out energy/variogram score margin vectors;
- calibration of \(P(M>0\mid O,g)\);
- lower-quantile coverage;
- selected-action calibration отдельно от all-candidate calibration;
- tower residual на unseen physical occlusion chains;
- Blackwell value monotonicity violations.

### Efficiency

- encoder latency;
- cost на один дополнительный posterior sample;
- total selection latency при 256/512/1024 candidates;
- memory;
- comparison с 8/16/60 shape completions.

### Четыре decisive hypotheses

**H1 — quotient sufficiency.** На одинаковом candidate pool TQ-Grasp не хуже strongest shape-posterior method по top-1 success и лучше по tail calibration/latency.

**H2 — posterior, а не mean.** На occlusion twins full process существенно снижает low-quantile regret против deterministic/independent predictors.

**H3 — tower objective.** BTPS уменьшает calibration/tower violations и selected-grasp failures на high occlusion без ухудшения low-occlusion accuracy.

**H4 — no hidden reconstruction.** Выигрыш сохраняется при полном запрете geometry supervision; иначе paper фактически подтверждает GIGA, а не новую постановку.

### Kill criteria до большого robot study

Проект следует остановить или сузить, если на synthetic twins после matched tuning выполняется любое:

- independent quantile baseline имеет тот же risk regret в пределах confidence interval;
- tower term не улучшает ни distribution score, ни high-occlusion calibration;
- TQ process требует столько же latency/memory, сколько sparse shape completion;
- candidate recall из observation-only pool ниже уровня, при котором selector способен соревноваться;
- energy-score samples collapse или не восстанавливают bimodal truth на toy benchmark.

---

## 14. Novelty audit против ближайших general-ML работ

| Работа/направление | Что уже даёт | Что остаётся новым в TQ-Grasp |
|---|---|---|
| Amortized Bayesian Decision Making | Direct expected cost \((x,a)\) без явного posterior | Posterior **function** outcomes, risk-flexible finite marginals, physical information-order consistency |
| Optimal simulation-based decisions | Utility surrogate и active simulator allocation | Amortized conditional stochastic process из partial sensor input, не per-instance BO |
| Bayes-sufficient representations | Minimal representation для фиксированных loss/Bayes action | Hidden-world equivalence по всей action-outcome function и posterior, decision-complete для семейства risks |
| Conditional/Neural Diffusion Processes | Generic distributions over functions | Task quotient, projective grasp-query decoder, known garbling/tower objective |
| Proper scoring rules | Distribution-identifying losses | Random grasp-query finite marginals + action-field architecture + tower moment |
| Measure-valued forecast martingales | Characterization/testing sequential calibration | Обучение posterior outcome fields по controlled nested occlusions |
| Adversarial GMM | Neural conditional-moment estimation | Новый exact moment из Blackwell garbling, не IV/causal identification |

### Наиболее вероятные reviewer objections

1. **«Это Neural Process applied to grasping».** Ответ возможен только при наличии quotient theorem, BTPS, controlled twins и tower ablation. Без них objection справедлив.
2. **«Proper score уже автоматически даёт tower property».** В population — да, если каждый conditional law выучен идеально. Contribution должен быть finite-sample coupling across known garblings; нужны rate/bound или убедительный low-data experiment.
3. **«Для одного grasp достаточно success probability».** Верно для фиксированного binary risk. Нужны continuous margin, risk-sweep и empirical win над calibrated scalar baseline.
4. **«Shape reconstruction даёт reusable geometry».** Верно, но вне узкой задачи selection. Нельзя объявлять TQ-Grasp универсальной заменой reconstruction.
5. **«Synthetic prior не соответствует real objects».** Нужны category-disjoint/OOD splits, real calibration и abstention.
6. **«Foreground occlusion не всегда является pure garbling».** Tower pairs создаются только same-camera overlay operator; natural pairs используются для evaluation, а не для theorem.
7. **«Joint process не projectively consistent».** Pointwise shared-latent decoder обеспечивает exact consistency; query self-attention запрещён.

---

## 15. ICLR acceptance audit

Официальный [ICLR 2027 Reviewer Guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines) сводит оценку к конкретному вопросу, мотивации и месту в литературе, корректности/rigor, поддержке claims и значимости нового знания; SOTA сам по себе не обязателен. По этим критериям:

### Потенциальные сильные стороны

- новый general-ML объект: posterior task quotient как альтернатива full latent posterior;
- новый, математически проверяемый objective через proper scores + Blackwell tower moments;
- clean application, где full reconstruction очевидно избыточна, но hidden uncertainty принципиальна;
- теоремы о sufficiency, propriety и information ordering;
- controlled ambiguity benchmark даёт новое знание, а не только leaderboard gain;
- измеримый compute–risk trade-off;
- no RL/VLA, clean supervised/simulation-based learning.

### Что опустит paper до robotics workshop/RA-L

- только новая grasp architecture без general theorem/toy benchmark;
- сравнение только с устаревшими direct baselines;
- отсутствие TARGO/ZeroGrasp/modern shape-first comparator;
- обычный mean/std uncertainty вместо posterior field;
- claims SOTA на малом числе hardware trials;
- BTPS без доказательства, что naive posterior equality неверна;
- использование reconstruction loss «для стабильности», уничтожающее основной claim.

### Реалистичная оценка до экспериментов

- **Novelty:** высокая, но зависит от строгого отличия от Neural Processes + BAM + Bayes quotient.
- **Technical depth:** потенциально высокая при 4 propositions и корректном conditional-moment estimator.
- **Broad interest:** средне-высокий, если показать generic inverse-decision toy/PDE task помимо grasping.
- **Empirical risk:** высокий из-за силы современных shape-first методов.
- **ICLR potential:** правдоподобный, но не высокий-confidence до controlled twins и real high-occlusion results.

---

## 16. Минимальный paper package

### Main claim

> Full latent reconstruction is not the uniquely principled response to partial observability. For decision families that access a hidden world only through action outcomes, the posterior of the task-quotient outcome process is decision-complete; it can be learned from sparse simulator queries with proper finite-marginal scores and made information-consistent across known observation garblings.

### Предлагаемый title

**Predict the Outcome Field, Not the Hidden Shape: Blackwell-Consistent Task-Quotient Processes for Grasping under Occlusion**

### Abstract skeleton

1. Single-view occlusion makes many hidden shapes compatible with the same RGB-D input.
2. Shape completion estimates far more latent detail than grasp selection consumes; direct grasp scores discard ambiguity.
3. Define task quotient \(S/\!\sim\) through equality of action-outcome functions and learn its conditional posterior process.
4. Introduce projectively consistent shared-latent query architecture and BTPS.
5. Prove universal decision sufficiency, finite-marginal propriety and tower consistency.
6. Demonstrate exact posterior recovery on controlled ambiguity, better risk/latency trade-off on TARGO-like scenes and higher real Success@1 under severe foreground occlusion.

### General-ML second domain

Для broad ICLR evidence полезен один небольшой non-robotics simulator, где expensive latent field скрыт, а decisions query scalar outcomes — например, selecting a design point under partially observed PDE coefficient field. Нельзя превращать это в второй большой проект; задача нужна только для подтверждения, что TQPP и BTPS не grasp-specific tricks.

---

## 17. Практический roadmap

### Phase A — 2D analytic sanity check

- random hidden contours;
- one-dimensional parallel-jaw action domain;
- exact enumeration posterior;
- проверить ES recovery, projectivity, tower moments и risk regret.

### Phase B — 3D occlusion twins

- 100–500 base meshes с controlled hidden backside variants;
- fixed candidate pool;
- deterministic, independent quantile, TQ-ES и TQ-BTPS;
- kill criteria до масштабного training.

### Phase C — large synthetic

- category-disjoint objects;
- realistic same-camera occluders/noise;
- TARGO/GIGA/ZeroGrasp comparisons;
- matched compute and simulator budget.

### Phase D — hardware

- shelf, wrist RGB-D, parallel jaw;
- forced pick + abstention;
- 1–2 cm lift;
- confidence intervals и pre-registered primary metric.

### Phase E — paper theory

- formal quotient/minimality;
- random-query finite-marginal identification;
- BTPS Fisher consistency;
- finite-critic tower violation bound;
- limitations and explicit binary-probability special case.

---

## 18. Финальная рекомендация

Развивать **TQ-Grasp / BTPS** как top-1. Это не модификация конкретного robotic pipeline, а новая decision-theoretic постановка, выведенная из достаточности, stochastic processes, proper scoring и Blackwell information order. Её наиболее ценная часть — не latent architecture сама по себе, а связка:

\[
\text{hidden world}
\longrightarrow
\text{task quotient outcome function}
\longrightarrow
\text{posterior process}
\longrightarrow
\text{proper finite-marginal learning}
\longrightarrow
\text{tower-consistent risk selection}.
\]

При этом результат надо считать **high-risk/high-upside**. Strongest shape-first evidence уже очень серьёзно. Наиболее честный и быстрый следующий шаг — controlled occlusion-twins benchmark. Если full TQ process и BTPS там не превосходят calibrated independent quantiles, идею следует остановить до больших роботических затрат. Если превосходят и сохраняют преимущество на TARGO-like noise, получится не просто occlusion grasping method, а broad general-ML statement о том, какой posterior объект следует учить для решений под частичной наблюдаемостью.

---

## 19. Ключевые источники

### Robotic grasping / gap and evidence

- [TARGO: Benchmarking Target-driven Object Grasping under Occlusions](https://arxiv.org/abs/2407.06168)
- [S4G: Amodal Single-view Single-Shot SE(3) Grasp Detection](https://proceedings.mlr.press/v100/qin20a.html)
- [Contact-GraspNet](https://arxiv.org/abs/2103.14127)
- [GIGA: Synergies Between Affordance and Geometry](https://roboticsproceedings.org/rss17/p024.pdf)
- [ZeroGrasp](https://openaccess.thecvf.com/content/CVPR2025/papers/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.pdf)
- [Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645)
- [Shape Completion with Prediction of Uncertain Regions](https://arxiv.org/abs/2308.00377)
- [Measuring Uncertainty in Shape Completion to Improve Grasp Quality](https://arxiv.org/abs/2504.16183)
- [UNCLE-Grasp](https://arxiv.org/abs/2601.14492)
- [Single-View Shape Completion for Robotic Grasping in Clutter](https://arxiv.org/abs/2512.16449)
- [Object Pose and Shape Estimation for Grasping: Does it Work?](https://arxiv.org/abs/2605.26944)

### General ML / mathematical inspiration

- [Amortized Bayesian Decision Making for simulation-based models](https://arxiv.org/abs/2312.02674)
- [Optimal simulation-based Bayesian decisions](https://arxiv.org/abs/2311.05742)
- [Bayes-Sufficient Representations in Supervised Learning](https://arxiv.org/abs/2606.04045)
- [Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a/garnelo18a.pdf)
- [Neural Diffusion Processes](https://arxiv.org/abs/2206.03992)
- [Strictly Proper Scoring Rules, Prediction, and Estimation](https://www.eecs.harvard.edu/cs286r/courses/fall10/papers/Gneiting07.pdf)
- [Adversarial Generalized Method of Moments](https://arxiv.org/abs/1803.07164)
- [Calibrated Probability Forecast Sequences and Measure-Valued Martingales](https://arxiv.org/abs/2606.31621)
- [ICLR 2027 Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines)
