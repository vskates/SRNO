# Не достраивать объект: учить posterior над тем, что с ним можно сделать

## Исследовательский журнал и спецификация идеи для ICLR 2027

**Дата среза литературы:** 25 августа 2026 года.  
**Рабочее название общей идеи:** **Decision-Quotient Posterior Learning (DQPL)**.  
**Рабочее название инстанциации:** **Conditional Random Feasible-Set Process (CRFSP)**, или короче **Grasp-Quotient Field**.  
**Вердикт:** направление стоит проверять. Оно заметно сильнее первоначальных вариантов с shape completion, обычной uncertainty head или conformal reranking. Однако новизна имеет смысл только при строгой постановке через *posterior над grasp-relevant функционалом*, а не как «ещё один probabilistic grasp scorer». Вероятность сильной ICLR-подачи оцениваю как умеренно высокую **при наличии** общей теории, убедительной синтетической задачи, сильных completion/uncertainty baselines и реального робота. Без этих четырёх элементов — вероятный reject как application of functional flow matching.

> **Обновление 25 августа 2026 — этот ранний вердикт больше не является основным.** После дополнительного поиска CRFSP через functional flow matching оставлен как сильный baseline, но не как рекомендуемый headline: близость к FlowSDF/стохастической сегментации слишком велика, а для обычного top-1 expected success полный posterior поля вообще не нужен. Текущий лучший кандидат — **Fell-Event Elicitation for Random Feasible-Action Sets (FELLAS)** с новым proper objective **Choquet Query Score** и новой архитектурой **Choquet Excursion Network**. Полная спецификация, отрицательные результаты и условия, при которых идею следует закрыть, находятся в разделах 22–36.

---

## 1. Коротко: что именно предлагается

Пусть скрытая полная геометрия объекта равна $S$, единственное наблюдение — шумное RGB-D изображение $x$, а $g\in\mathcal G\subset SE(3)\times\mathbb R_+$ — поза и раскрытие parallel-jaw gripper. Полная геометрия индуцирует не только отдельную метку «этот захват успешен», но целое множество допустимых локальных захватов

$$
F_S=\{g\in\mathcal G:Y(S,g)=1\}.
$$

Из-за окклюзии одно и то же $x$ совместимо с несколькими $S$, а значит — с несколькими разными $F_S$. Главный объект предсказания должен быть не реконструкция $S$, не одна grasp pose и не независимый Bernoulli-score для каждого кандидата, а условный закон случайного множества

$$
\Pi_x = \mathcal L(F_S\mid x).
$$

Для вычислимости множество кодируется его **знаковым action-space margin field** $r_S:\mathcal G\to\mathbb R$: положительное значение означает допустимый grasp, модуль — минимальное возмущение позы/ширины, нужное для смены допустимости. Модель учит непосредственно

$$
\mathcal L(r_S(\cdot)\mid x),
$$

запрашивая значения только в $K$ кандидатах. Один сэмпл модели даёт одну согласованную гипотезу о всей границе множества допустимых действий. На инференсе выбирается grasp с лучшим нижним CVaR/квантилем margin; робот воздерживается, если даже лучший нижний risk-score не положителен.

**Ключевой тезис:** скрытую форму не нужно восстанавливать, если downstream-решения зависят лишь от её образа в пространстве действий. Мы учим posterior сразу на этом decision quotient.

---

## 2. Зафиксированные границы задачи

### 2.1 Что входит

- Humanoid robot с parallel-jaw gripper.
- Одна wrist-camera и одно RGB-D наблюдение.
- Целевой объект стоит на полке.
- Один foreground object может закрывать значительную часть цели.
- Depth содержит реалистичный шум, выбросы, дырки и ошибки по границе маски.
- Выбирается один локальный grasp; затем робот замыкает пальцы и приподнимает объект на небольшую высоту только для физической проверки.
- Полная геометрия доступна лишь offline при создании supervision.
- Семантическая идентификация цели и получение target/occluder masks считаются заданными; иначе задача незаметно превращается в VLM/VLA.

### 2.2 Что намеренно не входит

- RL и VLA.
- Планирование всей траектории approach–close–lift.
- Активный сбор дополнительных видов или tactile regrasp.
- Полная object/scene reconstruction, dense occupancy, TSDF или SDF в трёхмерном рабочем объёме как выход модели.
- Полная физика падения/динамики после захвата.
- Clutter benchmark с множеством равноправных объектов: здесь важна контролируемая пара «target + foreground occluder».

### 2.3 Узкое определение успеха

Метка $Y(S,g)$ означает **локальную grasp formation feasibility**:

1. terminal gripper geometry в позе $g$ не пересекает наблюдаемый foreground obstacle и полку;
2. замыкание пальцев по короткому фиксированному локальному движению образует допустимые контакты с target;
3. квазистатический antipodal/force-closure proxy и ограниченный perturbation test проходят порог.

Reachability руки, дальний approach path и полноценный lift trajectory не входят в $Y$. В реальном опыте небольшой lift служит только бинарным readout качества сформированного захвата.

---

## 3. Как менялось направление: журнал отбраковки

### Ветка A — posterior shape completion + pessimistic selection

**Идея.** Сэмплировать несколько полных форм, оценивать grasp на каждой и выбирать по lower confidence bound.

**Почему отвергнута.** Это уже не новая постановка. Varley et al. восстанавливали полную форму из 2.5D и планировали grasp [R1]. Lundell et al. использовали MC-dropout shape completions и проверяли кандидаты по нескольким формам [R2]. UNCLE-Grasp уже делает target-level selective prediction по lower confidence bound над множеством completion-гипотез и сообщает особенно сильный выигрыш при тяжёлой окклюзии [R7]. Это отличный baseline и свидетельство полезности uncertainty, но не наша идея.

**Вывод.** Нельзя заявлять новизну через «несколько plausible shapes + robust grasp».

### Ветка B — достраивать только скрытую contact region

**Идея.** Не строить весь объект, а реконструировать только локальные области потенциального контакта пальцев.

**Почему отвергнута.** TOSC прямо формулирует task-oriented shape completion с фокусом на potential contact regions [R8]. AnyDexGrasp выводит скрытые локальные contact positions/normals из partial observation [R23]. Свежие работы продолжают занимать нишу partial/local geometry. Даже если сделать representation меньше, reviewer справедливо назовёт это contact-aware completion.

**Вывод.** Выход не должен жить в object space вообще.

### Ветка C — scalar grasp critic с uncertainty head

**Идея.** Для каждого $(x,g)$ предсказывать среднее и дисперсию или evidential distribution и сортировать по uncertainty-adjusted score.

**Почему отвергнута.** vMF-Contact уже моделирует направленную неопределённость контактов и разделяет типы uncertainty, одновременно используя auxiliary point reconstruction [R26]. Probabilistic directional models и FFHFlow уже демонстрируют пользу distributional prediction для noisy/partial point clouds [R25, R27]. Lundell, completion-uncertainty reranking и UNCLE также покрывают «uncertainty helps ranking» [R2, R5, R7].

**Вывод.** Независимые uncertainty scores — baseline, не contribution.

### Ветка D — conformal lower bound для выбранного grasp

**Идея.** Добавить conformal calibration и получить гарантированное воздержание.

**Почему отвергнута как ядро.** Современная conformal decision theory уже выводит prediction sets для risk-averse agents [M7], conformal robust optimization связывает coverage с downstream decision [M8], а в 2026 появились action-conditional и policy-coupled варианты [M9, M10]. Conformal слой полезен, но сам по себе будет выглядеть как wrapper.

**Вывод.** Conformalization допустима только как калибровка уже нового объекта — случайного feasible-set process. Гарантия должна называться marginal under exchangeability; нельзя выдавать её за per-scene conditional safety.

### Ветка E — posterior над wrench-space convex body

**Идея.** Учить support function множества достижимых wrench вместо формы.

**Почему отвергнута.** Математически красиво, но для двух пальцев приходится отдельно восстанавливать наличие/нормали контактов, friction cone, collision и ширину gripper. Representation перестаёт быть компактной и снова скрыто кодирует механику объекта. Кроме того, задача быстро расползается в causal failure modes и lift dynamics, которые исключены.

### Ветка F — выбранная: random feasible set в action space

**Почему выжила.** Она переносит неопределённость ровно в пространство downstream-решения, не выдаёт ни mesh, ни point completion, поддерживает multimodal hidden geometry, допускает query-only вычисление, имеет общую ML-постановку за пределами grasping и порождает проверяемую decision-theoretic теорию.

### Дополнительные угрозы, найденные после выбора

- **Grasping Neural Process (GNP)** учит latent posterior над неизвестными mass/friction/center-of-mass из *истории исполненных grasp trials* и декодирует feasibility [R31]. Это близко концептуально, но вход — интеракции, скрытые переменные — динамические свойства, а не single-view occluded geometry; модель не задаёт posterior над boundary/margin field из RGB-D.
- **Grasp Distance Fields (GDF, август 2026)** задаёт гладкое расстояние до дискретного набора grasp configurations в arm-hand configuration space для controller execution [R32]. Это детерминированное поле вокруг уже синтезированных grasпов, не условный posterior feasible set, не single-view uncertainty и не selection. Термин “distance field” нельзя подавать как саму новизну.
- **Deep Learning a Grasp Function** уже предсказывает score для всех image-plane grasпов и свёртывает его с pose uncertainty [R29]. Поэтому contribution не «grasp field», а **law of a structured random field induced by missing geometry**, плюс quotient theory.
- **Learning to Generate All Feasible Actions** учит распределение по всем feasible actions [R28]. Там полностью наблюдаемое состояние, преимущественно 2D proof-of-concept и цель — равномерная генерация; нет posterior над меняющимся feasible set при частичном наблюдении.

---

## 4. Общая постановка: Decision-Quotient Posterior Learning

Пусть $Z$ — латентное состояние мира, $X\sim p(x\mid Z)$ — неполное наблюдение, $a\in\mathcal A$ — действие, а $q_Z(a)$ — любой downstream-релевантный функционал: feasibility margin, constraint slack, robust cost или reachable-set membership. Определим отображение

$$
\pi: Z\mapsto q_Z(\cdot).
$$

Два скрытых состояния эквивалентны, если они индуцируют один и тот же функционал:

$$
Z_1\sim_{\mathcal D} Z_2 \iff q_{Z_1}(a)=q_{Z_2}(a)\quad\forall a\in\mathcal A.
$$

Фактор $Z/\!\sim_{\mathcal D}$ — **decision quotient**. DQPL предлагает аппроксимировать pushforward posterior

$$
\pi_{\mathrm{push}}p(Z\mid X=x)=\mathcal L(q_Z(\cdot)\mid x)
$$

напрямую, минуя posterior над $Z$.

Это широкая идея, применимая к grasping, safe control при частично наблюдаемых препятствиях, экспериментальному дизайну, inverse design и constraint satisfaction. Grasping — хороший тест, потому что object-space reconstruction дорога, а локальная допустимость действий зависит от маленькой, но часто скрытой части формы.

---

## 5. Специализация для parallel-jaw grasping

### 5.1 Action space и метрика

$$
g=(R,t,w)\in\mathcal G\subset SE(3)\times[w_{\min},w_{\max}].
$$

Используем геодезическую product metric

$$
d_{\mathcal G}(g,g')^2 =
\frac{\|t-t'\|_2^2}{\sigma_t^2}+
\frac{\mathrm{ang}(R^\top R')^2}{\sigma_R^2}+
\frac{(w-w')^2}{\sigma_w^2}.
$$

Масштабы имеют физический смысл: допустимые ошибки position, rotation и width. Для parallel jaws следует факторизовать симметрию поворота на $\pi$ вокруг closing axis, иначе одно физическое действие будет представлено дважды.

### 5.2 Feasible set

Для полной сцены $S$, включая target и геометрию foreground obstacle в terminal neighborhood,

$$
F_S=\{g:Y(S,g)=1\}.
$$

Важно: это не множество arm trajectories и не policy support. Это локальное множество terminal grasps.

### 5.3 Signed robustness field

$$
r_S(g)=
\begin{cases}
d_{\mathcal G}(g,\mathcal G\setminus F_S), & g\in F_S,\\
-d_{\mathcal G}(g,F_S), & g\notin F_S.
\end{cases}
$$

Тогда $F_S=\{g:r_S(g)\ge 0\}$. Знак даёт feasibility, величина — pose/width robustness. Для численно шумного oracle boundary points можно считать infeasible; это делает $F_S$ closed после явного margin threshold.

Практически exact nearest boundary слишком дорога. Label approximation:

1. вычислить dense local cloud кандидатов вокруг $g$;
2. прогнать analytic/physics oracle $Y$;
3. взять расстояние до ближайшего противоположного класса;
4. censor значения сверху $r_{\max}$, поскольку далеко от boundary точность не нужна.

Альтернатива для первого прототипа — two-channel field $(d(g,F_S),d(g,F_S^c))$, где обе компоненты гарантированно 1-Lipschitz. Это безопаснее, если regularity signed distance на выбранном quotient space неудобно доказывать.

### 5.4 Наблюдение без реконструкции

$$
x=(I,D,M_T,M_O,K_{cam}).
$$

Из depth строятся только **наблюдаемые** target points, occluder points и ray tags:

- visible target return;
- foreground-occluder return;
- unknown target shadow за occluder;
- invalid/noisy return.

Никакие точки внутри shadow volume не генерируются. Model encoder может извлечь latent features, но supervision и output существуют только в action space. Запрещены occupancy/Chamfer losses, скрытые decoder heads, выдающие object SDF, и test-time mesh completion.

### 5.5 Что именно случайно

Для одного изображения $x$ training distribution задаёт

$$
R_x(\cdot)\sim\mathcal L(r_S(\cdot)\mid X=x).
$$

Например, видимая часть кружки за коробкой может быть совместима с ручкой слева и справа. В первом sampled field безопасны одни grasps, во втором — другие. Независимые per-grasp logits могут совпасть по marginals, но не показывают, какие feasibility events должны меняться совместно.

---

## 6. Математический каркас

### 6.1 Связь со случайными замкнутыми множествами

Классическая теория random closed sets говорит, что закон случайного замкнутого множества характеризуется capacity/hitting functional

$$
T(K)=\Pr(F\cap K\neq\varnothing)
$$

для компактных $K$ [M11, M12]. Это даёт правильный вероятностный объект: не fuzzy membership map, а распределение над множествами. Signed distance embedding удобнее capacity functional для оптимизации отдельного действия, но должен сохранять тот же sampled-set semantics.

### 6.2 Proposition 1 — decision sufficiency

Пусть downstream loss имеет вид

$$
\ell(Z,a)=\bar\ell(q_Z(a),a).
$$

Тогда для любой decision rule, выбирающей одно действие, условный Bayes risk полностью определяется $\mathcal L(q_Z(\cdot)\mid x)$. Если два posteriors над $Z$ имеют один pushforward через $\pi$, никакое решение этого класса не может их различить.

**Смысл.** Это не утверждение, что quotient сохраняет всю форму. Он сохраняет ровно всю информацию для заранее определённого класса локальных grasp decisions.

### 6.3 Proposition 2 — ошибка posterior и decision regret

Пусть $U(a,q(a))$ является $L$-Lipschitz по второму аргументу. Если истинный и выученный process posteriors удовлетворяют

$$
W_1(\widehat\Pi_x,\Pi_x;\|\cdot\|_\infty)\le \varepsilon,
$$

то ошибка ожидаемой полезности каждого действия не превосходит $L\varepsilon$, а regret оптимизатора по выученному posterior относительно Bayes action — не более $2L\varepsilon$.

Для нижнего CVaR уровня $\alpha$ аналогичная граница масштабируется как $O(\varepsilon/\alpha)$. Для quantile без предположения о плотности около квантиля стабильной границы нет; поэтому CVaR теоретически чище и должен быть главным risk functional.

### 6.4 Proposition 3 — покрытие непрерывного action space

В геодезическом length space signed distance до регулярного feasible set 1-Lipschitz. Если конечный candidate set $C$ является $\delta$-net интересующей области, то разрыв между лучшим margin в непрерывном пространстве и лучшим margin среди кандидатов не больше $\delta$. Если геометрические условия для signed field не выполнены, two-channel distances дают константу не хуже 2 после восстановления знака.

Это связывает representation с candidate density и даёт осмысленную метрику sample efficiency: сколько action queries нужно, чтобы не пропустить robust grasp.

### 6.5 Почему Chamfer reconstruction не даёт такой гарантии

Средний Chamfer может стремиться к нулю, если ошибка сосредоточена на малой скрытой области. Но тонкий выступ, край или участок второго контакта может иметь исчезающе малую площадь и одновременно переключать $Y(S,g)$ для лучшего grasp. Поэтому малый average reconstruction error не гарантирует малый decision regret. Это нужно показать явным counterexample и synthetic experiment, а не оставить риторикой.

### 6.6 Важное ограничение sufficiency claim

Для **одного фиксированного risk-neutral решения** scalar $p(Y=1\mid x,g)$ уже Bayes-sufficient. Нельзя утверждать, что joint random field информационно минимален для top-1 expected success.

Joint field оправдан для более богатого, но всё ещё one-step класса запросов:

- разные post-hoc risk preferences без переобучения;
- robust margin, а не только Bernoulli success;
- coherent sampled alternatives при multimodal hidden geometry;
- simultaneous post-selection bounds по адаптивно выбранному кандидату;
- выбор набора разнообразных backup grasps;
- continuous local refinement и candidate-coverage guarantees.

Эксперимент обязан отдельно показать, что structure приносит sample-efficiency или calibration benefit против мощного per-action distributional critic. Иначе честный вывод — scalar critic достаточно.

---

## 7. Модель CRFSP

### 7.1 Interface

Вход:

- один RGB-D frame и intrinsics;
- target/occluder masks;
- query set $C=\{g_1,\ldots,g_K\}$.

Выход:

- $M$ совместных сэмплов $(r^{(m)}(g_1),\ldots,r^{(m)}(g_K))$;
- marginal mean, lower-CVaR и feasibility probability для каждого кандидата;
- optional scene-level abstention/OOD score.

### 7.2 Observation encoder

Sparse point/ray transformer кодирует видимые target points и foreground points. Каждая точка несёт RGB, depth confidence, mask role, viewing ray и normal confidence. Shadow tokens описывают только направление и границу неизвестной области, не заполненную геометрию.

Encoder вычисляется один раз. Он должен быть equivariant к общей SE(3) смене координат или, как минимум, использовать camera-to-shelf canonical frame с сильной SE(3) augmentation.

### 7.3 Query encoder

Каждый $g_i$ кодируется через:

- relative pose к видимым target features;
- gripper closing axis и approach axis;
- finger swept volumes только для короткого closing motion;
- width;
- расстояния до *наблюдаемой* полки/foreground points.

Cross-attention извлекает локальный видимый контекст. Observed collision можно задать hard mask, не тратя stochastic capacity на заведомо невозможные действия.

### 7.4 Conditional stochastic-process decoder

Предпочтительный вариант — conditional **Flow-Matching Neural Process** или function-space flow model, вдохновлённый Functional Flow Matching [M1] и stochastic-process extensions [M2–M5]. Он моделирует конечномерные распределения

$$
p_\theta(r_C\mid x,C),\qquad r_C=(r(g_1),\ldots,r(g_K)),
$$

с shared scene latent и attention между grasp queries.

Базовый conditional flow-matching loss для interpolant $u_t=(1-t)u_0+t r_C$:

$$
\mathcal L_{FM}=\mathbb E_{t,u_0,r_C}
\left[\left\|v_\theta(t,u_t,C,h_x)-(r_C-u_0)\right\|_W^2\right].
$$

Здесь $W$ усиливает область около $r=0$, потому что именно boundary определяет решение.

**Критическая инженерная оговорка:** permutation equivariance по queries сама по себе не обеспечивает projective consistency. Архитектура должна сэмплировать один глобальный latent function/object и затем детерминированно query его в любых $g$, либо явно обучаться marginal-consistency loss на вложенных query sets. Consistency надо измерять в эксперименте.

### 7.5 Structure losses

$$
\mathcal L=\mathcal L_{FM}+\lambda_b\mathcal L_{boundary}
+\lambda_L\mathcal L_{Lip}+\lambda_c\mathcal L_{cons}.
$$

- $\mathcal L_{boundary}$: повышенный вес ошибкам знака и margin около нуля.
- $\mathcal L_{Lip}$: штраф за нарушение известной Lipschitz bound между соседними action queries.
- $\mathcal L_{cons}$: согласованность marginals на $C_1\subset C_2$.
- Hard observed-collision mask: sampled margin не может быть положительным для gripper volume, пересекающего достоверно наблюдаемую полку/occluder.

Не следует навязывать Eikonal equality $\|\nabla r\|=1$ везде: приблизительный physics oracle, quotient symmetries и негладкие пересечения feasible components могут её нарушать. Lipschitz inequality безопаснее; Eikonal можно тестировать как ablation только вдали от medial axis.

### 7.6 Training data

Полные CAD meshes и simulator используются **только как label oracle**:

1. sample target shape, pose, shelf and foreground occluder;
2. render one noisy RGB-D;
3. sample many grasps $g$, особенно около feasibility boundary;
4. вычислить $Y(S,g)$ и approximate $r_S(g)$;
5. обучать на случайных subsets queries и на нескольких независимых occlusions одной shape.

Ни один loss не сравнивает предсказанную геометрию с mesh. Это принципиальная ablation boundary: если auxiliary reconstruction улучшает результат, её можно показать отдельно, но основная модель должна оставаться reconstruction-free.

### 7.7 Candidate proposal

Работа должна честно называться **selection**, а не полной grasp detection. Для всех selectors используется один high-recall candidate generator. Возможные варианты:

- proposals от AnyGrasp/GSNet, замороженные;
- геометрический sampler, anchored на видимых target points;
- union нескольких proposal families для высокой recall.

Чтобы скрытая contact geometry не создавала невозможную зависимость от proposal network, нужен oracle-candidate experiment: если в candidate set добавлены качественные full-shape grasps, насколько хорошо selector их находит из partial observation?

### 7.8 Inference и abstention

Для $M$ сэмплов поля:

$$
s_\alpha(g\mid x)=\mathrm{LCVaR}_\alpha
\{r^{(m)}(g)\}_{m=1}^M,
\qquad
g^*=\arg\max_{g\in C}s_\alpha(g\mid x).
$$

Исполнять grasp, если $s_\alpha(g^*\mid x)>\tau$; иначе abstain. $\alpha$ и $\tau$ подбираются только на validation/calibration set.

LCVaR предпочтительнее LCB вида $\mu-\beta\sigma$: он не предполагает Gaussian posterior и корректно реагирует на multimodality.

### 7.9 Optional conformal calibration

На exchangeable calibration scenes можно строить simultaneous lower band для конечного candidate set, используя nonconformity, зависящий от максимального нарушения по всем queries. Это защищает от selection-after-evaluation лучше, чем pointwise intervals.

Но формулировка гарантии должна быть узкой:

- coverage marginal по новым IID scenes;
- не conditional для данного occlusion;
- не гарантирована при sim-to-real shift;
- candidate generator должен быть включён в calibration protocol;
- calibration — вторичный модуль, не headline novelty.

### 7.10 Epistemic uncertainty и OOD

Разброс sampled fields в generative model главным образом описывает **conditional ambiguity within the training distribution**. Он не гарантирует корректную epistemic uncertainty на новом типе объектов или сенсорном сдвиге. Для OOD-abstention нужны ensemble/last-layer Bayesian approximation или отдельный density/distance score. Надо отчётливо разделять:

- hidden-geometry ambiguity;
- sensor aleatoric noise;
- parameter uncertainty;
- distribution shift.

---

## 8. Почему representation может быть эффективнее reconstruction

### 8.1 Вычислительно

Вместо dense $128^3$–$256^3$ grid, mesh extraction и повторного grasp analysis модель делает:

$$
O(\text{encode one RGB-D})+O(MTKL)
$$

для $K$ queries, $M$ samples и $L$ flow steps. Цель прототипа: $K=256$–1024, $M=8$–32, $L=4$–8 с batching. Нельзя заранее обещать real-time; wall-clock и peak memory должны измеряться против completion pipelines.

### 8.2 Статистически

Object space содержит множество вариаций, не меняющих ни одного допустимого grasp: текстуру, внутреннюю геометрию, далёкую от контактов поверхность. Quotient supervision отбрасывает их. Гипотеза: при фиксированном количестве shapes/query labels это уменьшает sample complexity.

Это пока **гипотеза**, не теорема. Её надо проверять learning curves по числу training objects и по числу labeled actions на object.

### 8.3 Для decision quality

Reconstruction loss усредняет ошибки по поверхности. Action margin loss концентрируется на boundary, где меняется решение. Это аналог loss-calibrated Bayesian inference [M6], но calibration происходит на уровне выбранного статистического объекта, а не только weighting готового posterior.

---

## 9. Карта ближайших работ и точная дельта

| Семейство | Что уже сделано | Чего там нет относительно CRFSP |
|---|---|---|
| Full completion → grasp [R1–R4, R9, R10] | Восстановление формы/scene representation, затем grasping | Прямого posterior над action-feasible set без reconstruction |
| Uncertain completions [R2, R5–R7] | Несколько форм, uncertainty maps, pessimistic reranking/abstention | Object-space sampling всё ещё обязательный промежуточный объект |
| Joint reconstruction + grasp [R11, R12] | GIGA/NeuGraspNet связывают geometry и affordance | Dense/implicit scene field; обычно point estimate grasp quality |
| Direct grasp detectors [R13–R18] | Быстрый score/proposal prediction из partial points | Coherent stochastic law of the whole margin field under hidden geometry |
| Local/contact inference [R8, R23] | Task/contact-oriented completion, hidden contacts/normals | Всё ещё выводится geometry/contact structure, не decision quotient |
| Probabilistic grasp models [R25–R27, R33] | Distribution over proposed actions или orientation/contact uncertainty | Posterior random variable — обычно выбранный action/pose, не latent feasible set for fixed queries |
| Grasp function [R29] | Dense deterministic score, smoothing by pose noise | Нет posterior из missing geometry и sampled joint alternatives |
| All feasible actions [R28] | Generator, покрывающий feasible actions | Нет частичного наблюдения и posterior над множествами |
| GNP [R31] | Neural-process latent posterior над скрытыми physical properties из grasp trials | Не single-view occluded geometry; нет margin-boundary process |
| GDF [R32] | Deterministic configuration-space distance to known candidate grasps for execution | Не learning from RGB-D, не random set, не selection |
| Random-set ML [M13] | Belief functions/random sets по конечным class labels | Не continuous feasible-action sets и не stochastic process over margins |
| Functional generative models [M1–M5] | Законы случайных функций на arbitrary query sets | Нет decision quotient и grasp specialization |

### Самая опасная reviewer-формулировка

> “This is Functional Flow Matching applied to a grasp-quality function.”

Ответ будет убедителен только если paper содержит:

1. новую DQPL formalization и sufficiency/regret results;
2. random-set/margin construction, а не произвольный regression target;
3. benchmark, где одинаковое наблюдение действительно задаёт multimodal feasible sets;
4. сравнение с equally expressive per-query distributional critic;
5. evidence, что coherent field и quotient supervision дают data/calibration/decision gain.

---

## 10. Косвенные эмпирические основания

Прямых результатов CRFSP ещё нет, поэтому ниже — только triangulation, не доказательство.

### 10.1 Hidden geometry действительно влияет на grasp

- GIGA сообщает более чем 10-point improvement над VGN при joint geometry/affordance reasoning [R11].
- PCF-Grasp сообщает +17.8% real-world grasping performance против использованных SOTA baselines благодаря completion-informed features [R4].
- Single-View Shape Completion for Robotic Grasping сообщает +23% к no-completion и +19% к недавнему completion baseline в своей постановке [R9].

Следовательно, игнорировать скрытую форму недостаточно. Но эти результаты не доказывают, что её нужно реконструировать.

### 10.2 Uncertainty-aware selection помогает

- Lundell et al. получили существенный прирост, оценивая grasps на uncertain completions [R2].
- vMF-Contact в приведённых авторами real tests поднимает success с 39.2/45.5 до 72.7/65.0% (ID/OOD) при reconstruction+uncertainty [R26].
- UNCLE-Grasp v3 сообщает при самом тяжёлом simulated occlusion attempted success 0.870 против 0.780 у сильнейшего completed baseline, а в physical high-occlusion subset — 0.800 против 0.483 ценой меньшего attempt rate [R7].

Следовательно, selective/risk-aware grasping — реальный эффект. Но CRFSP должен выиграть при одинаковом candidate generator и attempt coverage.

### 10.3 Локальная/task-relevant геометрия эффективна

- Contact-GraspNet снижает effective pose representation, привязывая grasp к observed contact [R14].
- AnyDexGrasp показывает важность разнообразия локальных геометрий, а не только числа object identities [R23].
- TOSC улучшает contact-relevant reconstruction относительно общего completion [R8].

Это поддерживает quotient hypothesis: supervision ближе к действию может быть статистически выгоднее общего surface loss.

### 10.4 Function-space generation технически правдоподобна

Functional Flow Matching строит generative models прямо в function space и превосходит рассмотренные function-space baselines в своих задачах [M1]. Flow-Matching Neural Processes и operator variants развивают arbitrary-query stochastic processes [M2–M5]. Это не evidence для robotics performance, но уменьшает algorithmic risk реализации.

---

## 11. Экспериментальная программа

### 11.1 Новый controlled benchmark: ShelfOcclusion-Grasp

Сцена содержит ровно target, shelf и один foreground occluder. Контролируемые оси:

- occluded target surface fraction: 0–20, 20–40, 40–60, 60–75, >75%;
- скрыта ли хотя бы одна потенциальная contact region;
- depth noise, missing-pixel rate, edge flying pixels;
- target–occluder separation;
- foreground geometry;
- camera viewpoint and small calibration error;
- object local-geometry novelty.

Важно разделять pixel occlusion и **decision occlusion**: долю oracle-feasible grasps, чьи критические contact areas невидимы. Второй показатель сильнее коррелирует со сложностью задачи.

### 11.2 Splits

- **Instance split:** unseen meshes.
- **Category split:** unseen semantic categories.
- **Local-geometry split:** held-out families of curvature/thickness/handle/cavity patterns; это особенно важно после наблюдений AnyDexGrasp [R23].
- **Occluder split:** unseen foreground shapes/materials.
- **Sensor shift:** synthetic → real and one camera → another.

Один mesh не должен попадать в разные splits через near-duplicate assets.

### 11.3 Baselines

Минимально необходимы:

1. AnyGrasp/GSNet direct detector [R15, R16].
2. Contact-GraspNet [R14].
3. GIGA и/или NeuGraspNet [R11, R12].
4. deterministic per-query margin regressor с той же encoder capacity.
5. heteroscedastic/evidential per-query distributional critic.
6. independent conditional flow per grasp без query interactions.
7. full shape completion + тот же grasp oracle/scorer [R1–R4, R9].
8. MC completion robust planning в духе Lundell [R2].
9. UNCLE-Grasp-style LCB/abstention [R7].
10. vMF-Contact либо максимально близкая parallel-jaw probabilistic реализация [R26].
11. oracle full-shape selector и oracle feasible candidate — верхние границы.

Все методы должны получать одинаковые masks, candidates и observed-collision filter там, где это возможно.

### 11.4 Главные метрики

- top-1 analytic/physics success;
- real robot lift success;
- success **при фиксированном attempt coverage**;
- risk–coverage curve и area under it;
- calibration of $P(r>0)$, lower-CVaR calibration и Brier/NLL;
- worst-bin success по decision occlusion;
- candidate-regret к full-shape oracle;
- function posterior quality: energy score / sliced Wasserstein на joint query vectors;
- pairwise feasibility correlation error;
- marginal consistency для вложенных query sets;
- inference latency, peak GPU memory и training-label cost.

Нельзя сравнивать 80% attempted success одного метода с 100% attempt rate другого без полной risk–coverage curve.

### 11.5 Диагностические задачи до большого симулятора

1. **2D silhouette toy:** один visible contour совместим с двумя hidden shapes, у которых disjoint feasible intervals. Проверить, воспроизводит ли модель две границы, а не усредняет их.
2. **Thin-contact counterexample:** Chamfer почти нулевой, но исчезновение маленькой hidden patch меняет grasp feasibility.
3. **Correlated alternatives:** две hidden hypotheses делают группы grasps взаимоисключающими; проверить joint samples.
4. **Nested query test:** marginal одного grasp не должен зависеть от того, какие дополнительные queries были поданы.

Если модель проваливает эти задачи, реальный робот преждевременен.

### 11.6 Ablations

- binary feasibility vs signed margin;
- independent outputs vs coherent process;
- Gaussian latent NP vs flow process;
- mean/LCB/quantile/lower-CVaR selection;
- без ray-shadow tags;
- без Lipschitz/boundary loss;
- без observed-collision hard constraints;
- число flow steps и posterior samples;
- query count/density;
- без RGB, только depth;
- noisy masks;
- optional reconstruction auxiliary head — только как анализ, не основная модель;
- conformal calibration on/off;
- ensemble OOD on/off.

### 11.7 Реальный робот

Минимально убедительный дизайн:

- 20–30 unseen household targets;
- несколько foreground obstacles, не встречавшихся в simulation;
- не менее 20 randomized trials на target–occluder family, либо power analysis до сбора;
- заранее зафиксированные occlusion bins и attempt policy;
- парные сцены для методов, randomized execution order;
- failure taxonomy: perception/mask, no proposal, selector, terminal collision, slip;
- Wilson/Jeffreys intervals и paired significance test;
- отчёт по всем trials, включая abstentions.

Малый lift должен быть одинаковым readout для всех методов. Reach planner фиксирован и не входит в claim.

---

## 12. Go/no-go milestones

### M0 — label geometry (1 неделя)

- Построить stable approximate margin на action neighborhoods.
- Проверить, что margin воспроизводим при повторном physics simulation.
- **No-go**, если nearest opposite-label distance в основном отражает шум oracle, а не robustness.

### M1 — ambiguity toy (1 неделя)

- Flow process восстанавливает multimodal correlated boundaries.
- **No-go**, если independent mixture critic не хуже по joint и decision metrics.

### M2 — quotient advantage (2–3 недели)

- На controlled 3D benchmark direct margin posterior превосходит completion pipelines при сопоставимом compute или меньшем числе labels.
- **No-go**, если преимущество исчезает при одинаковом encoder/candidate set.

### M3 — value of coherence (1–2 недели)

- Coherent process даёт значимый выигрыш над per-query distributional critic по sample efficiency, calibration after selection или backup-set decisions.
- **Pivot**, если top-1 одинаков: сделать paper про structured posterior/calibration только при сильном общем результате; иначе упростить до scalar method и отказаться от ICLR-level novelty claim.

### M4 — real transfer (2–4 недели)

- Улучшение success at matched coverage в тяжёлых occlusion bins, без ухудшения latency до непригодной.
- **No-go**, если выигрыш в simulation объясняется недостоверным grasp oracle и не переносится.

---

## 13. Проверка по критериям ICLR 2027

Официальный reviewer guide просит ответить на четыре вопроса: конкретная задача, мотивация и место в литературе, поддержка claims, значимость нового знания; SOTA сам по себе не обязателен [I1].

### 13.1 Конкретная задача

**Да.** Learn the conditional law of the local grasp-feasible set induced by a single occluded RGB-D observation, without reconstructing the object, then make a risk-sensitive one-step selection.

### 13.2 Хорошо ли мотивировано и помещено в литературу

**Потенциально да.** Есть три независимые линии evidence: completion helps, uncertainty helps, task-local representations help. Related-work section обязан включить не только robotics, но random sets, neural/functional processes и loss-calibrated decisions.

### 13.3 Поддержаны ли claims

**Пока нет — это research proposal.** Нужны proofs для sufficiency/regret/coverage claims, controlled counterexample, strong baselines, real robot и честные confidence intervals.

### 13.4 Значимость для широкой ML-аудитории

**Условно да.** DQPL — общая альтернатива latent-state reconstruction для partial-observation decisions. Если работа останется только новой grasp architecture, значимость для ICLR будет пограничной.

### 13.5 Новизна

По выполненному поиску до 25.08.2026 не найдена работа, одновременно делающая следующее:

1. single noisy RGB-D с foreground occlusion;
2. conditional posterior над continuous feasible-grasp set/margin field;
3. direct action-space learning без object reconstruction;
4. coherent arbitrary-query stochastic-process output;
5. risk-sensitive selection/abstention из этого posterior.

Это **не доказательство отсутствия** работы. Самые близкие угрозы: UNCLE-Grasp [R7], TOSC [R8], vMF-Contact [R26], All Feasible Actions [R28], grasp function [R29], GNP [R31], GDF [R32], Functional Flow Matching [M1] и Flow-Matching Neural Processes [M2]. Их нужно обсуждать прямо, а не прятать.

### 13.6 Реалистичность дедлайна

ICLR 2027 требует abstract до 18 сентября и full paper до 25 сентября 2026, основной текст — не более 9 страниц [I2]. На дату этого журнала остаётся около месяца. Полная новая simulation suite + достаточные реальные trials за этот срок — высокий execution risk. Если данных/robot protocol ещё нет, разумнее целиться в следующую конференцию, чем ослаблять доказательность.

---

## 14. Claims, которые можно и нельзя писать

### Допустимые до результатов

- «Мы формулируем hidden-geometry grasping как conditional random feasible-set prediction».
- «Мы предлагаем action-space decision quotient вместо object-space reconstruction».
- «Мы выводим sufficiency и posterior-to-decision regret bounds при явных предположениях».
- «Модель не имеет reconstruction output/loss и query-ит только candidate grasps».

### Только после подтверждения

- «Эффективнее completion» — нужны matched latency, memory и label-budget curves.
- «Лучше calibrated» — нужны proper scoring rules и held-out calibration.
- «Надёжнее при окклюзии» — нужна success-at-equal-coverage curve и real data.
- «General framework» — нужна хотя бы вторая non-grasp toy/domain или убедительная theorem-level generality.

### Нельзя заявлять

- «Первый uncertainty-aware grasp selector».
- «Первый grasp field».
- «Первый neural process for grasping».
- «Conditional coverage guarantee» при обычном split conformal.
- «Epistemic uncertainty» только из generative samples.
- «Minimal sufficient representation» для фиксированного expected-success decision.
- «No reconstruction» при наличии hidden occupancy/SDF auxiliary decoder в основной модели.

---

## 15. Основные риски и способы фальсификации

### Риск 1: field скрыто реконструирует всю форму

Latent encoder теоретически может хранить shape. Запрет должен относиться к вычислительному/supervision interface, а не к недоказуемому содержимому нейронов. Проверка: probing decoder на occupancy; information bottleneck; сравнение label/sample complexity. Claim — «не требует explicit reconstruction», не «не содержит shape information».

### Риск 2: margin labels слишком дорогие

Boundary-focused active sampling offline, local perturbations, cached oracle, binary-to-distance transform по local graph. Отчитывать число oracle calls, не только число meshes.

### Риск 3: posterior неидентифицируем из обычного supervised dataset

Если на каждый $x$ есть ровно один $S$, conditional generative model может игнорировать noise. Нужны repeated occlusion renderings, deliberate ambiguous families и population-level conditional variation. В synthetic benchmark полезны парные shapes, совпадающие на visible surface.

### Риск 4: action correlations не нужны для top-1

Это принципиальный риск, не minor ablation. Сравнение с equal-capacity independent critic должно быть центральным. Если gain только в backup-set или simultaneous calibration, paper должен честно сузить claim.

### Риск 5: proposal recall доминирует selector

Отдельно report proposal recall under full-shape oracle, selector regret conditional on a feasible candidate и end-to-end result. Иначе failure причинно не локализован.

### Риск 6: sim-to-real gap маскируется abstention

Сравнивать при одинаковом coverage, показывать coverage itself, фиксировать thresholds до robot trials, включать camera/mask shift.

### Риск 7: foreground obstacle требует motion planning

В benchmark occluder влияет на видимость и terminal collision. Все дальние approach constraints отданы фиксированному planner и не являются объектом обучения. Сцены, где ни один предложенный terminal grasp недостижим, исключаются или помечаются как planner failure до сравнения selectors.

---

## 16. Рекомендуемая структура статьи

1. **Introduction:** reconstruction solves a harder problem than the decision needs; occlusion makes feasible set random.
2. **Problem:** latent state, observation, decision quotient, random feasible set, signed margin.
3. **Theory:** sufficiency, regret transfer, query coverage; exact assumptions.
4. **Method:** ray-aware observation encoder, conditional function flow, structure losses, risk selection.
5. **Controlled evidence:** ambiguous toy and Chamfer counterexample.
6. **ShelfOcclusion-Grasp:** simulation, baselines, calibration, scaling.
7. **Real robot:** matched-coverage results and failure taxonomy.
8. **Limitations:** single view, in-distribution ambiguity, fixed candidate generator, no approach/lift dynamics.

Главный рисунок должен показывать одно observed RGB-D, две разные совместимые hidden shapes только как *explanatory ground truth*, два sampled feasible sets в grasp action space и risk-selected common robust grasp. Не показывать reconstructed meshes как output модели.

---

## 17. Приоритетный следующий эксперимент

Не начинать с humanoid. Сначала построить минимальную проверку центрального тезиса:

1. Сгенерировать 2D/2.5D парные объекты с идентичным visible contour и различной hidden backside.
2. Определить 2D parallel-jaw action space и точный feasible set.
3. Обучить: deterministic field, independent mixture critic, latent NP, flow process.
4. Сравнить joint posterior, lower-CVaR decision, calibration after max-selection и sample efficiency.
5. Проверить, что sampled fields соответствуют целым plausible feasible sets, а не pointwise salt-and-pepper patterns.

Этот эксперимент дешёвый и прямо отвечает на самый опасный вопрос: нужна ли coherent random-function model, или достаточно независимых scores.

---

## 18. Литература: robotic grasping и ближайшие угрозы

- **[R1]** Varley et al., *Shape Completion Enabled Robotic Grasping* (2016): https://arxiv.org/abs/1609.08546
- **[R2]** Lundell et al., *Robust Grasp Planning Over Uncertain Shape Completions* (IROS 2019): https://arxiv.org/abs/1903.00645
- **[R3]** *3DSGrasp: 3D Shape-Completion for Robotic Grasp* (2023): https://arxiv.org/abs/2301.00866
- **[R4]** *PCF-Grasp: Using Point Cloud Completion Features for Fast and Efficient Grasping* (2025): https://arxiv.org/abs/2504.16320
- **[R5]** *Measuring Uncertainty in Shape Completion to Improve Grasp Quality* (2025): https://arxiv.org/abs/2504.16183
- **[R6]** *Shape Completion with Prediction of Uncertain Regions* (2023): https://arxiv.org/abs/2308.00377
- **[R7]** *UNCLE-Grasp: Uncertainty-Aware Target-Level Grasping under Occlusion*, v3 (2026): https://arxiv.org/abs/2601.14492
- **[R8]** *TOSC: Task-Oriented Shape Completion for Robotic Grasping* (2026): https://arxiv.org/abs/2601.05499
- **[R9]** *Single-View Shape Completion for Robotic Grasping in Clutter* (2025): https://arxiv.org/abs/2512.16449
- **[R10]** *ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping* (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.html
- **[R11]** Jiang et al., *GIGA: Synergies Between Affordance and Geometry* (RSS 2021): https://roboticsproceedings.org/rss17/p024.pdf
- **[R12]** *NeuGraspNet: Learning Any-View 6DoF Robotic Grasping in Cluttered Scenes*: https://openreview.net/forum?id=Fdu33eoZas
- **[R13]** Breyer et al., *Volumetric Grasping Network* (2021): https://arxiv.org/abs/2101.01132
- **[R14]** Sundermeyer et al., *Contact-GraspNet* (2021): https://arxiv.org/abs/2103.14127
- **[R15]** Fang et al., *AnyGrasp* (2022/2023): https://arxiv.org/abs/2212.08333
- **[R16]** Wang et al., *Graspness Discovery in Clutters* (ICCV 2021): https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.html
- **[R17]** Fang et al., *GraspNet-1Billion* (CVPR 2020): https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html
- **[R18]** Mahler et al., *Dex-Net 2.0* (2017): https://arxiv.org/abs/1703.09312
- **[R19]** Mousavian et al., *6-DOF GraspNet* (2019): https://arxiv.org/abs/1905.10520
- **[R20]** *GoalGrasp: Targeted Grasping in Partially Occluded Scenes* (2024): https://arxiv.org/abs/2405.04783
- **[R21]** *OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics* (2024): https://arxiv.org/abs/2401.12202
- **[R22]** *A Cross-view Fusion Framework for Robust 6-DoF Grasp Pose Estimation* (CVPR 2026; дополнительный view, не наша постановка): https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_A_Cross-view_Fusion_Framework_for_Robust_6-DoF_Grasp_Pose_Estimation_CVPR_2026_paper.html
- **[R23]** *AnyDexGrasp* (ICLR 2025): https://graspnet.net/anydexgrasp/assets/files/AnyDexGrasp.pdf
- **[R24]** *SpringGrasp: An Optimization Pipeline for Robust and Compliant Dexterous Pre-Grasp Synthesis* (2024): https://arxiv.org/abs/2404.13532
- **[R25]** *FFHFlow: A Flow-Based Variational Approach for Multi-Fingered Grasp Synthesis* (CoRL 2025): https://arxiv.org/abs/2407.15161
- **[R26]** *vMF-Contact: Uncertainty-Aware Contact-Based Grasping* (2024/2025): https://arxiv.org/abs/2411.03591
- **[R27]** *Efficient End-to-End Detection of 6-DoF Grasps with the Power Spherical Distribution*: https://openreview.net/forum?id=f82nippU5C
- **[R28]** *Learning to Generate All Feasible Actions* (2023): https://arxiv.org/abs/2301.11461
- **[R29]** Johns et al., *Deep Learning a Grasp Function for Grasping under Gripper Pose Uncertainty* (2016): https://arxiv.org/abs/1608.02239
- **[R30]** *Dealing with Ambiguity in Robotic Grasping via Multiple Predictions* (2018): https://arxiv.org/abs/1811.00793
- **[R31]** Noseworthy et al., *Amortized Inference for Efficient Grasp Model Adaptation* / Grasping Neural Process (ICRA 2024): https://groups.csail.mit.edu/rrg/papers/noseworthy_shaw_icra24.pdf
- **[R32]** Enwerem et al., *Grasp Execution Without a Planner: Configuration-Space Grasp Distance Fields with Certified Safety & Guaranteed Quality* (preprint, Aug. 2026): https://arxiv.org/abs/2608.00600
- **[R33]** Yan et al., *Learning Probabilistic Multi-Modal Actor Models for Vision-Based Robotic Grasping* (2019): https://arxiv.org/abs/1904.07319

## 19. Литература: общая ML/математика, из которой выводится идея

- **[M1]** Kerrigan et al., *Functional Flow Matching* (AISTATS 2024): https://proceedings.mlr.press/v238/kerrigan24a.html
- **[M2]** *Flow Matching Neural Processes* (2025): https://arxiv.org/abs/2512.23853
- **[M3]** Garnelo et al., *Conditional Neural Processes* (2018): https://arxiv.org/abs/1807.01613
- **[M4]** *The Functional Neural Process* (2019): https://arxiv.org/abs/1906.08324
- **[M5]** *Operator Flow Matching for Learning Stochastic Processes* (2025): https://arxiv.org/abs/2501.04126
- **[M6]** Lacoste-Julien et al., *Loss-Calibrated Approximate Inference in Bayesian Neural Networks* / decision-aware Bayesian inference line (2011): https://proceedings.mlr.press/v15/lacoste_julien11a.html
- **[M7]** Kiyani et al., *Decision Theoretic Foundations for Conformal Prediction* (ICML 2025): https://proceedings.mlr.press/v267/kiyani25a.html
- **[M8]** Patel et al., *Conformal Contextual Robust Optimization* (AISTATS 2024): https://proceedings.mlr.press/v238/patel24a.html
- **[M9]** *Action-Conditional Conformal Prediction* (preprint, 2026): https://arxiv.org/abs/2606.05551
- **[M10]** *Policy-Coupled Conformal Prediction* (preprint, 2026): https://arxiv.org/abs/2607.02206
- **[M11]** Molchanov, *Theory of Random Sets* (2005): https://www.nzdr.ru/data/media/biblio/kolxoz/M/MD/Molchanov%20I.%20Theory%20of%20Random%20Sets%20%28ISBN%20185233892X%29%28Springer%2C%202005%29%28501s%29_MD_.pdf
- **[M12]** Capacity-functional characterization of random closed sets (математический источник/обзор): https://arxiv.org/abs/1106.2993
- **[M13]** *Random-Set Neural Networks* (2023): https://arxiv.org/abs/2307.05772
- **[M14]** Fong et al., *Martingale Posterior Distributions* (2021): https://arxiv.org/abs/2103.15671

## 20. Критерии конференции

- **[I1]** ICLR 2027 Reviewer Guide: https://iclr.cc/Conferences/2027/ReviewerGuidelines
- **[I2]** ICLR 2027 Author Guidelines: https://iclr.cc/Conferences/2027/AuthorGuidelines

---

## 21. Финальная оценка

**Сильная научная формулировка:** не «как угадать невидимую форму», а «какой минимальный posterior нужен для класса решений, когда латентный мир наблюдается частично». Для grasping этим объектом становится случайное множество feasible grasps или его signed-margin embedding.

**Главная новизна:** pushforward posterior в decision quotient + queryable coherent stochastic process over action margins. Flow matching, random-set theory, CVaR и conformal calibration — инструменты, а не отдельные claims новизны.

**Главный эмпирический вопрос:** даёт ли structured joint posterior что-либо сверх хорошо обученного per-grasp distributional critic при одинаковых данных, candidates и compute? Если нет — направление следует закрыть, даже если визуализации sampled fields красивы.

**Главный шанс на ICLR:** показать общий принцип «predict uncertainty after quotienting by downstream decisions» с ясной теорией и grasping как трудной, физически проверяемой демонстрацией. Это потенциально новое знание для ML-аудитории, а не композиция существующих robotics modules.

---

# Addendum 25.08.2026: от posterior поля к proper elicitation случайного множества

## 22. Новый основной вердикт

После второго круга поиска я **не рекомендую** делать главным contribution conditional functional flow над action-space SDF. Интерпретация DQPL остаётся полезной: форма действительно должна быть факторизована по downstream decisions. Но конкретная CRFSP-реализация имеет три уязвимости.

1. Для выбора одного grasp по ожидаемому бинарному успеху достаточно скаляра

   $$
   p_x(g)=\mathbb P(g\in F_S\mid x).
   $$

   Полный совместный posterior над полем не является минимально достаточным объектом для этой узкой decision rule.

2. FlowSDF уже учит условное распределение signed distance transforms сегментационных масок через flow matching [N4]. Перенос того же механизма в grasp action space математически содержателен, но визуально слишком похож на «FlowSDF for grasps».

3. Functional flow matching и neural processes уже дают общий аппарат для distributions over functions [M1–M5]. Reviewer может принять random-set interpretation, но всё равно посчитать алгоритмический вклад недостаточным.

**Новый лучший кандидат:**

- общая идея: **Fell-Event Elicitation for Random Feasible-Action Sets (FELLAS)**;
- learning objective: **Choquet Query Score (CQS)**;
- архитектура: **Choquet Excursion Network (CEN)**;
- роботическая инстанциация: условный закон множества локально допустимых parallel-jaw grasps под single-view occlusion;
- decision rule: максимизация posterior-вероятности того, что *всё калиброванное множество малых ошибок исполнения* остаётся допустимым.

Ключевое отличие от CRFSP: модель не учит density/velocity в пространстве функций и не получает pointwise likelihood. Она учится отвечать на случайные **событийные запросы к множеству** и по конструкции задаёт корректную Choquet capacity. Это одновременно новый target, proper loss и sampling-free probabilistic head.

Предварительная оценка: **умеренно сильный ICLR-кандидат**, если выполнятся четыре условия:

1. доказана strict propriety на конечном candidate bank и аккуратное расширение к random closed sets;
2. CEN превосходит direct query network и latent neural process при одинаковом encoder/compute;
3. multi-point query supervision даёт выигрыш именно на robust-set decision, а не только красивую calibration metric;
4. ToleranceNet-подобный baseline не закрывает весь практический выигрыш.

Если оставить только ball queries вокруг одного grasp, идея рискует превратиться в probabilistic ToleranceNet. Поэтому общий event-scoring result и эксперименты с неодносвязными/анизотропными query sets обязательны.

---

## 23. Дополнительный журнал отбраковки

### Ветка G — posterior erosion profile / distribution of tolerance

Для полной формы можно определить

$$
R_S(g)=\sup\{\rho:B_\rho(g)\subseteq F_S\}
$$

и учить survival function

$$
H_x(g,\rho)=\mathbb P(R_S(g)\ge \rho\mid x).
$$

Это давало красивый monotone survival objective и непосредственно решало robust grasp selection. Однако GraspNet-1Billion уже содержит **Tolerance Network / grasp affinity fields**, где target — максимальное perturbation, которое выдерживает grasp, и этот prediction используется для reranking [N2]. Наш posterior под окклюзией сильнее deterministic tolerance, но headline выглядел бы как «ToleranceNet + conditional uncertainty». Ветка остаётся baseline/special case, не основной идеей.

### Ветка H — martingale consistency по уровням раскрытия

Для вложенных наблюдений $\mathcal F_1\subset\mathcal F_2$ Bayes-предикторы удовлетворяют

$$
\mathbb E[m_{\mathcal F_2}\mid\mathcal F_1]=m_{\mathcal F_1}.
$$

Это привлекательнее обычной augmentation invariance: fine-view prediction может законно измениться, но не должен иметь систематический drift до раскрытия информации. В мае 2026 вышла работа **Martingale-Consistent Self-Supervised Learning**, которая формулирует ровно этот тезис, предлагает prediction/latent-space penalties и unbiased two-sample Monte Carlo estimator [N1]. Совпадение полное; ветка отвергнута как headline.

### Ветка I — minimax regret по observational fiber

Можно выбирать действие по worst-case regret среди скрытых форм, совместимых с $x$. Но robust decision-focused learning уже учит uncertainty sets и минимизирует worst-case regret [N10], а learning feasible regions через inverse optimization занято отдельной линией [N9]. В grasping такая постановка также легко становится чрезмерно консервативной. Не основной вклад.

### Ветка J — random utility/ranking posterior

Совместное распределение попарных предпочтений между grasps потенциально достаточно для top-1/ranking decisions, но decision-focused learning через learning-to-rank уже существует [M6 и новые DFL-работы], а random-utility networks — самостоятельная линия. Без нового elicitable object это выглядит как смена decoder.

### Ветка K — conditional Laplace functional

Для бинарного feasible-vector $Y\in\{0,1\}^N$ закон задаётся

$$
L_x(t)=\mathbb E[\exp(-t^\top Y)\mid x],\qquad t\ge 0.
$$

Его можно учить MSE по случайным sparse $t$. Формально это корректно и может быть полезной relaxation CQS, но characteristic-function learning и distribution learning через transforms уже существуют [N14]. Поэтому Laplace functional стоит оставить как ablation, а не как novelty claim.

---

## 24. Где полный random-set posterior действительно нужен

Нужно прямо признать неприятный, но важный факт.

### 24.1 Обычный top-1 expected success

Если робот выполняет ровно выбранный $g$, utility равна $Y(S,g)$, а цель — максимизировать средний успех, то Bayes rule есть

$$
g^*(x)=\arg\max_g \mathbb P(Y(S,g)=1\mid x).
$$

Зависимости между соседними grasps и полный закон $F_S$ для этого не нужны. Любая статья, которая скрывает этот факт, уязвима для простого reviewer objection.

### 24.2 Новая, операционально оправданная задача

Пусть команда $g$ реализуется с неизвестной ошибкой $\delta$ из калиброванного компактного множества $E_g$ в локальном tangent space позы/раскрытия. Тогда сильное событие надёжности равно

$$
Z(S,g,E_g)=\mathbf 1\{g\oplus E_g\subseteq F_S\}.
$$

Его posterior-вероятность

$$
I_x(g\oplus E_g)=
\mathbb P(g\oplus E_g\subseteq F_S\mid x)
$$

гарантирует локальную grasp-formation feasibility для **любого** residual error внутри заявленного calibration envelope. Выбор

$$
\hat g(x)=\arg\max_g I_x(g\oplus E_g)
$$

уже не сводится к одному marginal score, потому что событие требует совместной допустимости целого набора соседних действий.

Это всё ещё узкая локальная задача:

- нет планирования approach trajectory;
- нет RL/VLA;
- нет active view;
- нет reconstruction;
- маленький lift остаётся только physical readout;
- $E_g$ описывает лишь calibration/control/depth-induced terminal-pose error.

Если реальная система не может обосновать компактный $E_g$ калибровочными измерениями, robust-set semantics становится искусственной. Тогда лучше вернуться к marginal success probability, а эту идею закрыть.

---

## 25. Новый learning objective: Choquet Query Score

### 25.1 Объект предсказания

Пусть $\mathcal G$ — компактное метрическое пространство локальных gripper configurations, а

$$
F_S=\{g:r_S(g)\ge 0\}
$$

— замкнутое feasible set, индуцированное полной формой и локальным grasp oracle. При наблюдении $x$ оно становится random closed set с условным законом $\Pi_x$.

Классическая capacity functional равна

$$
T_x(K)=\mathbb P(F_S\cap K\neq\varnothing\mid x),
\qquad K\subset\mathcal G\text{ compact}.
$$

По Choquet–Kendall–Matheron theorem значения $T_x(K)$ на всех компактных $K$ однозначно определяют закон random closed set [M11, M12, N7]. Для robust decision дополнительно удобна inclusion functional

$$
I_x(K)=\mathbb P(K\subseteq F_S\mid x).
$$

### 25.2 Supervision одним set-valued observation

Для каждого training scene полная форма используется только offline, чтобы получить один sample $F_S$. Затем случайно выбирается query set $K$, и образуются две дешёвые бинарные метки:

$$
h(F_S,K)=\mathbf 1\{F_S\cap K\neq\varnothing\},
$$

$$
i(F_S,K)=\mathbf 1\{K\subseteq F_S\}.
$$

Не нужны ни ground-truth posterior, ни несколько hidden shapes для каждого точного $x$, ни matching между generated и true modes.

### 25.3 Loss

Основной bounded loss:

$$
\mathcal L_{\mathrm{CQS}}(\theta)=
\mathbb E_{(x,F),K\sim\nu_x}
\left[
(T_\theta(x,K)-h(F,K))^2
+\lambda_I(I_\theta(x,K)-i(F,K))^2
\right].
$$

Здесь $\nu_x$ может зависеть от наблюдения и геометрии candidate bank, но **не должна зависеть от скрытой метки $F$**, если не используется явная importance correction. Иначе balanced query mining меняет elicited target.

Возможны log/BCE variants, но Brier выбран основным по трём причинам:

- bounded gradients при редких multi-point events;
- точное excess-risk разложение;
- устойчивость к слегка шумному grasp oracle.

### 25.4 Почему singleton loss недостаточен

При $K=\{g\}$

$$
T_x(K)=I_x(K)=\mathbb P(g\in F_S\mid x),
$$

то есть CQS превращается в обычный per-grasp BCE/Brier. Новая информация появляется только для $|K|\ge 2$, где hit/inclusion probes наблюдают зависимости между действиями.

### 25.5 Fell-event extension

Более общий запрос задаётся hit-regions $H_1,\dots,H_q$ и miss-region $K_0$:

$$
A_\zeta=\{F:F\cap H_j\neq\varnothing\ \forall j,\ F\cap K_0=\varnothing\}.
$$

Такие события образуют базис hit-or-miss/Fell topology. Brier score вероятностей $\mathbb P(F\in A_\zeta\mid x)$ даёт общий **Fell Event Score**. В первой реализации достаточно capacity + inclusion queries; полный cylinder score нужен как теоретическое расширение и evaluation metric.

---

## 26. Основные теоретические утверждения

### Proposition 1 — strict propriety CQS

Пусть $P_x$ — истинный условный закон random closed set, $Q_x$ — report, а $\nu_x$ имеет поддержку на countable convergence-determining family компактных запросов. Тогда capacity-часть CQS строго proper:

$$
\mathcal R(Q_x;P_x)-\mathcal R(P_x;P_x)
{}={}
\mathbb E_{K\sim\nu_x}
\left(T_{Q_x}(K)-T_{P_x}(K)\right)^2\ge 0.
$$

Равенство возможно только если capacities совпадают на определяющей семье; по uniqueness theorem тогда $Q_x=P_x$. Inclusion term не нужен для идентифицируемости, но напрямую усиливает signal на robust decision queries.

### Proposition 2 — конечный candidate bank

Для $N$ кандидатов обозначим $Y_j=\mathbf1\{g_j\in F\}$ и

$$
T(J)=\mathbb P\left(\bigvee_{j\in J}Y_j=1\right),\qquad J\subseteq[N].
$$

Все $2^N$ значения $T(J)$ однозначно задают joint law $Y$. Если у конкретного pattern множество нулей равно $Z$, а множество единиц — $O$, то

$$
\mathbb P(Y_Z=0,Y_O=1)
{}={}
\sum_{B\subseteq O}(-1)^{|B|}
\left[1-T(Z\cup B)\right].
$$

Это обычная inclusion–exclusion/Möbius inversion. Экспоненциально перечислять запросы не требуется: Monte Carlo по $J$ даёт unbiased stochastic objective. Но если обучать только singleton/pair queries, полный law не идентифицируется; это осознанный statistical-computational trade-off.

### Proposition 3 — regret robust-set decision

Пусть

$$
u_x(g)=I_x(g\oplus E_g),
\qquad
\hat u_x(g)=I_\theta(x,g\oplus E_g),
$$

и $\sup_g|\hat u_x(g)-u_x(g)|\le\varepsilon$. Тогда для $g^*=\arg\max u_x$ и $\hat g=\arg\max\hat u_x$

$$
u_x(g^*)-u_x(\hat g)\le 2\varepsilon.
$$

Это простой, но важный мост от calibration of set queries к downstream grasp regret.

### Proposition 4 — discretization query set

Если каждое learned field $f_m(x,\cdot)$ $L_f$-Lipschitz, $K_\delta$ — $\delta$-net для $K$, а threshold CDF $G$ имеет bounded density $\|G'\|_\infty$, то

$$
\left|G\!\left(\inf_{g\in K}f_m(g)\right)
-G\!\left(\min_{g\in K_\delta}f_m(g)\right)\right|
\le \|G'\|_\infty L_f\delta.
$$

Аналогично для capacity через supremum. Это даёт явный контроль ошибки finite perturbation stencil.

### Что здесь не следует заявлять без отдельного доказательства

- finite $M$ не восстанавливает произвольный posterior без approximation error;
- calibration in simulation не даёт distribution-free real-world guarantee;
- random queries низкого порядка не идентифицируют high-order topology;
- uniform capacity error нельзя выводить из малого empirical CQS без complexity/generalization bound;
- CQS не решает OOD hidden shapes.

---

## 27. Новая архитектура: Choquet Excursion Network

### 27.1 Генеративное определение, хотя training не требует sampling

Encoder строит representation $h=E_\theta(x)$. Head выдаёт $M$ mode weights

$$
\pi_m(x)\ge 0,\qquad \sum_{m=1}^M\pi_m(x)=1,
$$

и $M$ непрерывных action fields

$$
f_m(x,g)=D_\theta(h,z_m,g).
$$

Сэмпл случайного feasible set определяется так:

1. $m\sim\mathrm{Categorical}(\pi(x))$;
2. $U\sim G$, где в минимальной версии $G$ — fixed standard logistic CDF;
3. 

   $$
   F_{m,U}(x)=\{g\in\mathcal G:f_m(x,g)\ge U\}.
   $$

Если $f_m$ непрерывен, каждый sample — замкнутое excursion set. Один общий threshold $U$ связывает все action queries; поэтому модель не создаёт независимый salt-and-pepper posterior.

### 27.2 Capacity и inclusion без Monte Carlo

Для конечного query set $K=\{g_1,\dots,g_k\}$:

$$
T_\theta(x,K)
=\sum_{m=1}^M\pi_m(x)
G\!\left(\max_{g\in K}f_m(x,g)\right),
$$

$$
I_\theta(x,K)
=\sum_{m=1}^M\pi_m(x)
G\!\left(\min_{g\in K}f_m(x,g)\right).
$$

Для пустого запроса используются set-theoretic conventions $T_\theta(x,\varnothing)=0$ и $I_\theta(x,\varnothing)=1$; max/min formulas выше относятся к непустому $K$.

Цена запроса — $O(M|K|)$; ODE solve, posterior sampling и dense voxel field не нужны.

### 27.3 Общий hit-or-miss event

Для hit-regions $H_1,\dots,H_q$ и miss-set $K_0$ внутри одного mode:

$$
a_m=\min_j\sup_{g\in H_j}f_m(x,g),
\qquad
b_m=\sup_{g\in K_0}f_m(x,g).
$$

Тогда

$$
\mathbb P(F\cap H_j\neq\varnothing\ \forall j,
F\cap K_0=\varnothing\mid x,m)
{}={}
\left[G(a_m)-G(b_m)\right]_+.
$$

Это аналитический query operator над распределением множеств.

### 27.4 Почему capacity корректна по конструкции

CEN сначала задаёт настоящий probability law через mixture случайных excursion sets, а уже затем вычисляет $T$. Поэтому автоматически выполняются:

- $T(\varnothing)=0$;
- монотонность по $K$;
- complete alternation;
- projective consistency между разными конечными наборами action queries.

Direct Set Transformer, который независимо регрессирует $T(x,K)$, этих свойств не гарантирует и должен быть отдельным baseline.

### 27.5 Approximation capacity

На конечном bank любая distribution на $\{0,1\}^N$ аппроксимируется CEN с не более чем $2^N$ modes: каждому binary pattern соответствует field с большими положительными значениями на feasible actions и отрицательными на остальных, а mixture weight равен вероятности pattern.

Для компактного непрерывного $\mathcal G$:

1. finite-support probability measures плотны в пространстве probability measures на random closed sets;
2. каждый deterministic closed set $A$ можно задать level set функции типа signed distance;
3. neural field аппроксимирует эту функцию;
4. при увеличении масштаба field влияние logistic threshold сжимается к deterministic level set.

Полная theorem потребует аккуратной Fell-topology формулировки и regularity assumptions. В основном тексте лучше дать finite-bank theorem, а continuous density result — в appendix.

### 27.6 Concrete encoder

Новый contribution находится в head, поэтому scene encoder нужно сделать сильным, но не экзотическим:

- target points, foreground points и shelf points имеют разные type embeddings;
- каждый point хранит 3D coordinate, RGB, depth confidence и camera ray;
- grasp query кодируется не только quaternion, а небольшим point cloud gripper geometry в позе $g$, что поддержано свежими данными о пользе explicit gripper geometry [N17];
- shared cross-attention/equivariant point encoder формирует action feature;
- $M$ learned mode tokens дают $f_m(x,g)$ и $\pi_m(x)$.

SE(3)-equivariance желательна, но не должна быть novelty claim: OrbitGrasp уже показывает сильный continuous/equivariant grasp field [N18].

---

## 28. Практический training recipe

### 28.1 Candidate bank

Для каждого наблюдения frozen proposal mechanism создаёт $N$ terminal grasps $g_{1:N}$. Он одинаков для всех методов. Для каждого полного training mesh offline oracle выдаёт $Y_{1:N}$.

Нужно обязательно публиковать **oracle proposal recall**: если bank не содержит надёжного grasp, posterior head не может исправить ошибку proposal stage.

### 28.2 Query sampler

Начальная смесь, которую нужно аблировать:

- 30% singleton queries — marginal calibration и warm start;
- 20% случайные pairs/quadruples — зависимости дальних действий;
- 35% local perturbation stencils вокруг nominal grasp — целевая robust inclusion;
- 15% union-of-regions / anisotropic stencils — не дать задаче схлопнуться к scalar tolerance.

Cardinality и геометрия $K$ выбираются по observed candidate bank, не по $Y$. Для локального $E_g$ использовать центр, $\pm$ principal axes и небольшое число boundary points в tangent coordinates; одинаковые физические единицы задаются через calibrated metric на translation/rotation/width.

### 28.3 Labels и loss в batch

Для индексов $J\subset[N]$:

$$
h_J=\max_{j\in J}Y_j,
\qquad
i_J=\min_{j\in J}Y_j.
$$

CEN оценивает соответствующие max/min своих mode fields. На один scene можно дешёво сэмплировать десятки $J$, переиспользуя action features.

### 28.4 Регуляризация

Допустимы только вспомогательные terms, чья роль проверяется ablation:

- entropy floor для $\pi$ в начале обучения против mode collapse;
- mild diversity по mode fields на multi-point queries;
- Lipschitz/spectral control action decoder для стабильной discretization;
- temperature annealing softmax/softmin только как numerical approximation.

Не добавлять reconstruction loss: он разрушит чистоту тезиса и создаст очевидный shortcut через hidden geometry.

### 28.5 Калибровка скорости закрытия gripper для label oracle

Цель — не минимальная скорость сама по себе, а **quasi-static regime, в котором label практически не меняется при дальнейшем замедлении**. Для фиксированного набора сцен и grasps выполнить sweep $v_1>v_2>\ldots>v_k$ и измерить

$$
\Delta_i=\Pr\left[Y_{v_i}(S,g)\neq Y_{v_{i+1}}(S,g)\right].
$$

Выбрать максимальную, то есть наиболее быструю, скорость $v^*$, после которой disagreement остаётся ниже заранее заданного $\varepsilon$ для всех меньших скоростей. Дополнительно проверить стабильность contact count, penetration/impulse и positive-label rate. Затем зафиксировать $v^*$ для всей генерации dataset; высокие скорости вне plateau не использовать, поскольку они превращают oracle label в артефакт динамики.

---

## 29. Inference и связь с физическим grasp

1. Из одного noisy RGB-D строятся target/occluder/shelf point sets.
2. Frozen proposer даёт $g_{1:N}$.
3. Для каждого nominal $g_j$ строится calibrated perturbation stencil $K_j\approx g_j\oplus E_{g_j}$.
4. CEN вычисляет

   $$
   s_j=I_\theta(x,K_j).
   $$

5. Выбирается $j^*=\arg\max_j s_j$.
6. Если $\max_j s_j<\eta$, система abstains.
7. Иначе выполняются только terminal placement, короткое close и небольшой lift.

Дополнительно доступны без переобучения:

- $T(\{g\})$: обычная вероятность feasible grasp;
- $I(K)$: uniform robustness;
- $T(K)$: вероятность, что хотя бы один action в локальном/семантическом bundle допустим;
- coherent set samples для диагностики;
- hit-or-miss probabilities для сложных action patterns.

---

## 30. Карта ближайших работ и точная граница novelty

| Линия | Что предсказывает/оптимизирует | Почему не совпадает с FELLAS/CEN |
|---|---|---|
| Johns et al. [N3] | deterministic dense grasp function; convolution с pose-noise distribution | ожидаемый успех под известным stochastic noise; нет conditional random-set law и all-actions-feasible event |
| GraspNet ToleranceNet [N2] | maximum perturbation, которое выдерживает данный grasp | scalar tolerance per grasp; нет posterior под occlusion, proper capacity score или multi-region queries |
| UNCLE-Grasp/TOSC [R7, R8] | posterior/uncertainty через shape completions или contact-region completion | output остаётся в object space; CEN никогда не строит shape |
| Probabilistic U-Net [N5] | conditional distribution of segmentation masks через latent VAE | sample-based mask posterior; нет event-proper capacity objective и analytic valid query head |
| FlowSDF [N4] | flow matching conditional SDF distribution для masks | самый близкий аргумент против CRFSP; CEN не учит flow/SDF distribution и не решает ODE |
| Functional Flow Matching / Neural Processes [M1–M5] | distributions over functions | общий stochastic-process apparatus; CEN задаёт law над level sets и обучается proper set-event queries |
| Random-Set Neural Networks [M13, N6] | belief masses на фиксированных subsets классов для epistemic/OOD classification | конечное class label space, hand-budgeted focal class sets, belief/credal semantics; не условный закон observed set-valued target в continuous action domain |
| Inverse feasible-region learning [N9] | feasible region, согласованная с observed optimization decisions | не учит random feasible set posterior из full set-valued supervision и не отвечает на capacity queries |
| Martingale-consistent SSL [N1] | tower consistency между coarse/refined observations | точное совпадение с отвергнутой веткой H; другая структурная property |

### Что можно честно заявлять новым

1. **Learning objective:** integrated proper score по random hit/inclusion queries, который elicits conditional law random closed set без set likelihood.
2. **Architecture:** mixture of learned neural excursion sets с общим random threshold и аналитическими capacity/inclusion queries.
3. **Decision formalization:** reliable grasp как posterior inclusion probability калиброванного execution set, а не uncertainty одного pose.
4. **Theory:** strict propriety, finite-bank identification, capacity validity, approximation и decision regret.

### Что нельзя заявлять новым

- random sets, capacities, excursion sets и Brier score сами по себе;
- implicit neural fields;
- mixture models;
- robust grasping или tolerance prediction вообще;
- probabilistic segmentation;
- learning feasible regions;
- SE(3)-equivariant grasp encoders.

Таргетированный поиск на 25.08.2026 не нашёл работу, объединяющую conditional random-set capacity elicitation, event-proper training и analytic neural excursion mixture. Это **не доказательство отсутствия**: перед submission нужен отдельный Google Scholar/Semantic Scholar citation chase по терминам *capacity functional estimation*, *random closed set regression*, *excursion-set mixture*, *set-distribution scoring rule*.

---

## 31. Эксперименты, которые могут опровергнуть центральный тезис

### Experiment 0 — exact finite-bank sanity check

Сначала не использовать robot data.

- $N=10$–16 actions, чтобы exact joint law можно было перечислить.
- Hidden mode создаёт несколько сильно коррелированных feasible patterns с одинаковым partial observation.
- Сравнить independent Bernoulli, mixture Bernoulli, latent NP, CRFSP и CEN.
- Метрики: exact TV/KL, all-subset capacity error, pair/high-order correlation, calibration after selection.

**Kill criterion:** CEN не восстанавливает high-order event probabilities лучше более простой mixture Bernoulli при matched parameters.

### Experiment 1 — occlusion twins в 2D/2.5D

- Пары объектов имеют один и тот же visible contour/depth, но разные hidden backsides.
- Parallel-jaw action bank и feasible sets вычисляются точно.
- Train/test split по visible families и по hidden variants.
- Query sets включают singletons, local perturbation bodies и disconnected bundles.

Главная картинка: один partial observation, несколько истинных compatible feasible sets, CEN mode fields и откалиброванные $T/I$ queries. Никаких reconstructed shapes как model output.

**Kill criterion:** multi-point CQS не улучшает robust-set Brier/regret относительно singleton-trained CEN.

### Experiment 2 — full 3D simulation

Контролируемые факторы:

- occlusion ratio;
- depth holes/outliers/edge noise;
- hidden-shape ambiguity при одинаковой видимой части;
- perturbation radius и anisotropy;
- seen/unseen shape families;
- candidate-bank recall.

Основные metrics:

1. single-grasp success;
2. robust-set success $\mathbf1\{K_g\subseteq F\}$ на dense offline stencil;
3. CQS/Brier по held-out query families и cardinalities;
4. risk–coverage/selective curve;
5. regret к oracle $\max_g I_x(K_g)$;
6. calibration after argmax selection;
7. inference latency и memory;
8. capacity-axiom violation для unconstrained direct-query baseline.

### Experiment 3 — real humanoid

- Single wrist RGB-D.
- Target on shelf, один foreground occluder.
- Несколько заранее измеренных calibration envelopes $E_g$.
- Один nominal grasp на trial; close + 2–3 cm lift.
- Blind randomization методов, одинаковые candidates, threshold и retry policy.
- Отдельно сообщать perception failure, terminal collision, no-contact, slip, unstable lift.

Matched-coverage сравнение обязательно: success rate CEN и baselines нужно сравнивать при одинаковой доле abstentions.

### Required baselines

1. deterministic per-grasp BCE critic;
2. critic + Johns-style convolution;
3. deterministic ToleranceNet analogue;
4. distributional tolerance/survival head;
5. deep ensemble / MC dropout;
6. latent neural process или mixture-of-Bernoulli field;
7. CRFSP/functional-flow baseline;
8. posterior shape completion + LCB/CVaR в стиле UNCLE-Grasp;
9. direct Set Transformer $T(x,K)$ без capacity-valid architecture;
10. CEN singleton-only;
11. CEN без mixture и CEN без inclusion term.

### Главные ablations

- query cardinality curriculum;
- random threshold против independent point noise;
- число modes $M$;
- fixed против learned threshold CDF;
- local-only против mixed query geometry;
- explicit gripper point query против pose vector;
- mode diversity term;
- soft vs exact max/min на inference.

### Does CEN actually recover modes?

На synthetic finite-bank задаче, где истинные latent modes и их posterior weights известны, нужно отдельно проверить, восстанавливает ли CEN мультимодальную структуру, а не только усреднённые event probabilities.

- сопоставить learned и true modes через Hungarian matching по induced feasible patterns;
- измерить per-mode Hamming/IoU, ошибку posterior weights и долю collapsed/duplicate modes;
- одновременно измерить TV joint law и capacity/inclusion error, поскольку нумерация и конкретная mixture-декомпозиция мод не идентифицируемы;
- повторить для correctly specified, under-specified и over-specified числа modes $M$, нескольких seeds и разных query distributions.

Главным критерием остаётся восстановление закона random feasible set и downstream robust regret. Семантическое совпадение отдельных $m$ — диагностическая метрика: разные наборы latent modes могут задавать один и тот же наблюдаемый закон.

---

## 32. Наиболее опасные reviewer objections

### «Это просто probabilistic segmentation в action space»

Ответ будет убедителен только если основной вклад — theorem/objective/architecture для random-set laws, а не применение. FlowSDF и Probabilistic U-Net должны быть прямыми baselines на synthetic random-set task.

### «Capacity score — это MMD/kernel mean embedding с indicator features»

Отчасти верно. Если определить

$$
\phi_K(F)=\mathbf1\{F\cap K\neq\varnothing\},
$$

то интегрированный Brier excess risk равен squared distance между mean embeddings этих features. Новизна должна быть сформулирована как **выбор canonical convergence-determining features random-set space + tractable capacity-valid conditional architecture**, а не как изобретение distribution embeddings. Characteristic-kernel literature [N8] нужно цитировать открыто.

### «Для grasp selection достаточно marginal probability»

Да, для expected success ровно выбранной pose. Поэтому paper task обязан быть robust inclusion под bounded execution error. Нельзя размывать это различие словами «reliability».

### «Это ToleranceNet с posterior»

Если все queries — isotropic balls, objection справедлив. Нужны general multi-region queries, общий random-set theorem и демонстрация решений, которые scalar radius не выражает.

### «Mixture of thresholded fields слишком ограничена»

Нужны finite-bank universality theorem, approximation ablation по $M$ и сравнение с flow/latent NP. При малом $M$ честно обсуждать bias.

### «Нет повторных одинаковых x, posterior неидентифицируем»

Как и любая conditional density estimation, модель опирается на sharing across nearby observations. Synthetic occlusion twins должны специально создавать группы с одинаковой видимой частью и разными hidden variants. Без такого benchmark joint-posterior claims трудно проверить.

### «Query sampler задаёт метрику и может скрыть ошибки»

Публиковать результаты по нескольким held-out $\nu$, cardinalities и geometries; train/test query distributions должны различаться. Directly report worst-family error, не только среднее CQS.

---

## 33. ICLR 2027 audit

Официальный reviewer guide просит ответить на четыре вопроса и подчёркивает, что SOTA не обязателен, если работа даёт новое, релевантное и значимое знание [I1]. Для FELLAS ответы пока такие.

### Какой конкретный вопрос решает paper?

Как обучать условный закон случайного множества, когда label сам является множеством, likelihood на hyperspace недоступен, а downstream решения задаются hit/inclusion events?

### Хорошо ли мотивирован метод и расположен ли он в литературе?

Потенциально да: random-set theory даёт canonical capacity; proper scoring rules — elicitation; excursion sets — valid generative family; grasp occlusion — физическая демонстрация. Но необходима открытая связь с RS-NN, stochastic segmentation, kernel mean embeddings, FlowSDF, ToleranceNet и inverse feasible-region learning.

### Поддерживают ли claims теория и experiments?

Пока нет — это исследовательская спецификация. Минимальный пакет доказательств:

- strict propriety;
- validity/universality CEN на finite bank;
- exact synthetic posterior recovery;
- matched-compute simulation;
- calibrated real execution envelope и real robot lift trials.

### В чём significance для широкой ML-аудитории?

Если сработает, вклад шире grasping: uncertainty-aware segmentation, safe regions, feasible design sets и spatial risk maps можно учить через event queries без density на space of sets. Если general random-set experiment отсутствует, paper будет выглядеть как специализированная robotics method.

### Предварительная оценка

- **Novelty:** 7.5/10 после targeted search; выше CRFSP, но citation chase ещё не завершён.
- **Technical depth:** 8/10 при строгих proofs; 5/10 без них.
- **Empirical risk:** высокий — marginal/tolerance baselines могут оказаться достаточными.
- **Clarity of thesis:** высокая, если paper начинается с proper random-set elicitation, а grasping появляется как demanding instance.
- **ICLR probability:** умеренная, не высокая; центральный synthetic result должен быть готов до масштабирования robotics.

---

## 34. Приоритетный следующий эксперимент и stop/go gate

Первым реализовать только finite-bank CQS/CEN на exact synthetic data.

### Минимальный протокол на 2–3 дня

1. $N=12$, четыре latent feasible-set modes, два из них имеют одинаковые singleton marginals, но разные correlations.
2. Partial observation скрывает mode частично; exact conditional law известен.
3. CEN: $M=4$, маленький encoder и action decoder.
4. Baselines: independent Bernoulli, mixture Bernoulli, direct set-query MLP.
5. Train singleton-only и mixed-cardinality CQS.
6. Evaluate all $2^{12}$ capacities, pattern TV и robust-cluster regret.

### Go

- mixed CQS восстанавливает high-order capacities;
- CEN сохраняет capacity validity и выигрывает robust regret;
- direct query MLP нарушает coherence или хуже переносится на unseen query cardinality;
- результат устойчив к $M$, seed и query distribution.

### Stop / pivot

- singleton/marginal baseline имеет тот же robust regret;
- mixture Bernoulli полностью доминирует CEN;
- CQS хорошо fit’ится только на seen query geometry;
- posterior recovery требует почти $2^N$ modes;
- калиброванный $E_g$ в реальном роботе слишком мал, чтобы joint set event влияло на выбор.

Только после прохождения gate стоит строить 3D RGB-D dataset и humanoid experiment.

---

## 35. Новые источники, проверенные во втором круге

- **[N1]** Gögl, Xing, Yau, *Martingale-Consistent Self-Supervised Learning* (2026): https://arxiv.org/abs/2605.11846
- **[N2]** Fang et al., *GraspNet-1Billion* — Tolerance Network / grasp affinity fields, Sec. 4.4 (CVPR 2020): https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html
- **[N3]** Johns et al., *Deep Learning a Grasp Function for Grasping under Gripper Pose Uncertainty* (2016): https://arxiv.org/abs/1608.02239
- **[N4]** Bogensperger et al., *FlowSDF: Flow Matching for Medical Image Segmentation Using Distance Transforms* (IJCV 2025): https://www.research-collection.ethz.ch/entities/publication/59b396af-29f4-47bf-8e2b-2170331f811e
- **[N5]** Kohl et al., *A Probabilistic U-Net for Segmentation of Ambiguous Images* (NeurIPS 2018): https://papers.neurips.cc/paper/7928-a-probabilistic-u-net-for-segmentation-of-ambiguous-images.pdf
- **[N6]** Manchingal et al., *Random-Set Neural Networks* (ICLR 2025): https://arxiv.org/abs/2307.05772
- **[N7]** Cenzer et al., *Algorithmic Randomness and Capacity of Closed Sets* — effective Choquet characterization (2011): https://arxiv.org/abs/1106.2993
- **[N8]** Simon-Gabriel and Schölkopf, *Kernel Distribution Embeddings: Universal Kernels, Characteristic Kernels and Kernel Metrics on Distributions* (JMLR 2018): https://www.jmlr.org/papers/v19/16-291.html
- **[N9]** Ren et al., *Inverse Optimization via Learning Feasible Regions* (ICML 2025): https://proceedings.mlr.press/v267/ren25d.html
- **[N10]** Yamao et al., *Robust Decision-Focused Learning via Worst-Case Regret Minimization* (UAI 2026): https://proceedings.mlr.press/v337/yamao26a.html
- **[N11]** Ye et al., *Learning Decision-Sufficient Representations for Linear Optimization* (COLT 2026): https://proceedings.mlr.press/v336/ye26a.html
- **[N12]** Rindt et al., *Survival Regression with Proper Scoring Rules and Monotonic Neural Networks* (AISTATS 2022): https://proceedings.mlr.press/v151/rindt22a.html
- **[N13]** Kratz and Nagel, *On the Capacity Functional of Excursion Sets of Gaussian Random Fields* (2014/2017): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2541528
- **[N14]** Li et al., *Neural Characteristic Function Learning for Conditional Image Generation* (ICCV 2023): https://openaccess.thecvf.com/content/ICCV2023/html/Li_Neural_Characteristic_Function_Learning_for_Conditional_Image_Generation_ICCV_2023_paper.html
- **[N15]** Fissler and Molchanov, *Set-valued Conditional Functionals of Random Sets* (2025): https://arxiv.org/abs/2504.13620
- **[N16]** Sartor et al., *Advancing Constrained Monotonic Neural Networks* (ICML 2025): https://proceedings.mlr.press/v267/sartor25a.html
- **[N17]** *Supermarket-6DoF: A Large-Scale Real-World Dataset for Category-Level 6D Object Pose Estimation and Grasping* (2025): https://arxiv.org/abs/2502.16311
- **[N18]** Hu et al., *OrbitGrasp: SE(3)-Equivariant Grasp Learning* (CoRL 2024 proceedings): https://proceedings.mlr.press/v270/hu25b.html
- **[N19]** Li et al., *PONG: Probabilistic Object Normals for Grasping* (2023): https://arxiv.org/abs/2309.16930
- **[N20]** *FIRMGrasp: Friction-Invariant Risk-Minimizing Grasping* (2026): https://arxiv.org/abs/2607.25049

---

## 36. Итог одной фразой после второго круга

**Не предсказывать скрытый объект и даже не предсказывать posterior его grasp-score field; учить через proper set-event queries условный закон того, какие области пространства действий целиком или хотя бы частично остаются допустимыми, а корректность этого закона зашить в архитектуру как смесь случайных excursion sets.**

---

# Addendum 25.08.2026: повторная сравнительная оценка после добавления LiMON

## 37. Условия оценки

Ниже оценивается не текст proposal сам по себе, а ожидаемая полная статья при следующем одинаковом для всех идей уровне исполнения:

- центральный falsification gate дал положительный, но не чудесный результат;
- fixed candidate pool, matched encoder/compute и современные direct/completion/uncertainty baselines;
- крупная controlled simulation, category-held-out split и воспроизводимый real shelf experiment;
- заявленные теоремы корректны, но не считаются глубокими, если это прямые следствия классической теории;
- нет скрытого выигрыша от candidate generator, abstention rate или дополнительной reconstruction supervision.

Используется стандартная 10-балльная шкала OpenReview: 5 — marginally below threshold, 6 — marginally above threshold, 7 — good paper/accept. ICLR 2027 требует прежде всего конкретный вопрос, корректное положение в литературе, поддержку claims и significant new knowledge; SOTA сам по себе не обязателен: https://iclr.cc/Conferences/2027/ReviewerGuidelines.

Вероятности принятия ниже — субъективные интервалы, а не статистически калиброванные прогнозы. Если центральный gate идеи не проходит, её ожидаемый review score уменьшается минимум на 1–2 пункта.

## 38. Новая угроза LiMON

LiMON не может позиционироваться как первая moment-based contact representation для parallel-jaw grasping. Уже существует:

- Adjigble et al., *Model-free and learning-free grasping by Local Contact Moment matching*, IROS 2018: https://doi.org/10.1109/IROS.2018.8594226. LoCoMo вычисляет локальные contact-moment descriptors между поверхностью объекта и пальцами parallel-jaw gripper по partial point cloud;
- SpectGRASP, 2021: https://arxiv.org/abs/2107.12492. Он использует spectral correlation и затем LoCoMo для ранжирования контактов;
- NeuGraspNet, RSS 2024: https://arxiv.org/abs/2306.07392. Он уже формулирует grasping как local neural surface rendering и query-local interaction representation;
- TOSC, AAAI 2026: https://arxiv.org/abs/2601.05499. Он уже занимает тезис, что contact-relevant geometry полезнее полной generic completion.

LiMON всё ещё отличается: он предсказывает скрытые gripper-indexed line measures, использует truncated Fourier moment cone, hard data/refinement consistency и не декодирует поверхность. Но reviewer summary становится опасно простым: «learned hidden LoCoMo/TOSC representation with Fourier moments and a PSD projection». Кроме того, conditional mean sketch теряет мультимодальность именно в самых неоднозначных occlusion twins. Поэтому LiMON имеет высокий practical upside, но не самый высокий expected ICLR score.

## 39. Обновлённый рейтинг

| Rank | Финальная идея файла | Защищаемое ядро | Novelty | ICLR breadth | Lab fit | Ожидаемый review profile | Ожидаемое принятие |
|---:|---|---|---:|---:|---:|---|---:|
| 1 | **FiRe** | response image/polytope observation-consistency set, support loss и evidence-contractive witnesses | 7.3/10 | 8.2/10 | 7.5/10 | `6, 6, 5, 7`, mean 6.0 | 50–60% |
| 2 | **FELLAS/CEN** | proper elicitation conditional random feasible sets через hit/inclusion queries | 7.6/10 | 8.3/10 | 6.8/10 | `6, 6, 5, 6`, mean 5.75 | 45–55% |
| 3 | **LiMON** | constrained non-invertible interaction transform из line-measure moments | 6.6/10 | 7.7/10 | 8.7/10 | `6, 5, 5, 6`, mean 5.5 | 35–45% |
| 4 | **CapGrasp** | conditional capacity whole gripper-region hit events с coherent joint circuit | 7.0/10 | 7.6/10 | 6.7/10 | `6, 5, 5, 6`, mean 5.5 | 35–45% |
| 5 | **AvoGrasp** | avoidance probability failure set для calibrated pose packet | 6.7/10 | 7.5/10 | 8.1/10 | `6, 5, 5, 5`, mean 5.25 | 30–40% |
| 6 | **Grasp Metamers / MetaContact** | sensor-equivalent mechanics-conflicting groups и joint bi-contact posterior | 5.8/10 | 7.1/10 | 8.5/10 | `6, 5, 5, 5`, mean 5.25 | 25–35% |
| 7 | **FiberGrasp** | necessary/possible action sets над observation fibers | 6.3/10 | 7.2/10 | 6.4/10 | `5, 5, 5, 6`, mean 5.25 | 25–35% |
| 8 | **Grasp-Certificate Process / RJPN** | posterior push-forward action-indexed certificate function | 5.7/10 | 7.2/10 | 7.2/10 | `5, 5, 5, 5`, mean 5.0 | 20–30% |
| 9 | **FiGO / OC-GOP** | outcome process плюс Blackwell/tower consistency под nested occlusion | 5.4/10 | 7.4/10 | 6.3/10 | `5, 5, 5, 5`, mean 5.0 | 15–25% |

`MetaContact.md` и `MetaContact-2.md` идентичны; они считаются одной темой.

## 40. Почему FiRe выходит на первое место после понижения MetaContact

FiRe имеет самый сильный единый paper claim:

> Under a coarsened observation, learn the convex image of the observation-consistency set through the downstream response operator; train it by support queries whose error controls robust decision regret, and force the response set to contract when valid evidence is added.

Это не сводится к очевидному «разные hidden shapes дают разные grasps». Claim определяет:

1. минимальный для класса linear robust decisions объект — response image, а не hidden shape;
2. objective — support-function regression с прямым decision bound;
3. архитектурное ограничение — monotone contraction по information refinement;
4. новую измеряемую величину — irreducible Occlusion Ambiguity Tax;
5. benchmark, который отдельно измеряет information ambiguity и model error.

Однако нельзя заявлять generic novelty «learning uncertainty sets for decisions»: Wang et al., *Learning Decision-Focused Uncertainty Sets in Robust Optimization* уже учит uncertainty sets через downstream robust optimization: https://arxiv.org/abs/2305.19225. Защищаемая дистанция FiRe — именно **coarsening-fiber response image + whole-function witnesses + exact evidence filtration + occlusion-twin ambiguity decomposition**. Если убрать хотя бы filtration или twin/OAT experiment, FiRe падает примерно до expected score 5–5.5 как contextual robust optimization applied to grasping.

## 41. Решение для совместной цели ICLR + lab pipeline + Applied Robotics portfolio

Выбирать **FiRe**, а не LiMON и не MetaContact.

- Для ICLR FiRe лучше превращается в general-ML paper: representation, loss, decision bound, information order и benchmark образуют одну причинно связанную историю.
- Для лаборатории output непосредственно ранжирует существующий candidate bank; не требуется менять wrist RGB-D stack, строить full completion или внедрять RL/VLA.
- Для Applied Robotics portfolio FiRe демонстрирует не только новую network head, но полный цикл research engineering: controlled ambiguity benchmark, physically meaningful oracle labels, decision-theoretic method, efficient selector и real humanoid validation.
- LiMON оставить как самостоятельный дешёвый Gate-0 competitor: если компактные line moments почти полностью сохраняют oracle ranking, это сильный deterministic baseline или будущая отдельная работа. Не смешивать LiMON внутрь FiRe до получения независимых результатов, иначе central claim размоется.

Решающий ранний gate FiRe: на 100–500 occlusion-twin families response polytopes должны одновременно (i) иметь малую effective witness complexity, (ii) давать minimax-regret advantage над coordinate intervals/direct critic и (iii) содержать usable common grasps при severe occlusion. Если H1 или H2 не проходит, основной выбор следует переключить на LiMON; если H5 не проходит, robust selector не имеет operational value и постановку надо менять, а не маскировать abstention.

---

## 42. Рекомендуемое распределение $K$-queries

Training distribution по $K$ должно сочетать три масштаба информации:

| Тип $K$ | Доля всех queries | Рекомендуемый размер | Назначение |
|---|---:|---:|---|
| Singleton $K=\{g\}$ | **20%** | $|K|=1$ | Marginal feasibility и warm start discriminator |
| Local perturbation $K_g=g\oplus E_g$ | **45%** | **7–20 grasпов** | Robust inclusion и локальная граница feasible region |
| Global cross-anchor subset | **35%** | преимущественно **2–4 граспа/anchor-а** | Зависимости между удалёнными grasp-регионами и согласованность latent modes |

Последние 35% следует разделить на:

- **25 процентных пунктов** — геометрически различные distant pairs, triples или quadruples;
- **10 процентных пунктов** — disconnected unions $K_{g_1}\cup K_{g_2}$ вокруг двух удалённых anchors для наиболее сложной проверки mode stitching.

Local stencil не должен быть сверхплотным. Практический состав: центр, $\pm$ главные направления translation, $\pm$ наиболее важные rotation directions и несколько случайных boundary points. Stencil следует пересэмплировать между эпохами.

Большие случайные global subsets не использовать: уже при $|K|\approx10$ inclusion почти всегда равен нулю, а hit — единице, поэтому targets насыщаются. Cross-anchor grasps нужно выбирать по геометрическому разнообразию и observed candidate bank, но **не по скрытым меткам $Y$**, иначе изменяется elicited target.

Обязательная ablation: `local-only` против `global-subset-only` против `mixed`; отдельно измерять robust top-1 regret и Brier на high-order/disconnected queries.

---

## 43. Центральный beyond-ToleranceNet experiment

Создать две группы сцен с одинаковыми per-grasp marginals и всеми local-tolerance statistics вокруг двух удалённых grasпов $g_A,g_B$, но разными дальними зависимостями:

$$
P_1:\ 0.5(1,1)+0.5(0,0),
\qquad
P_2:\ 0.5(1,0)+0.5(0,1).
$$

Обе группы неразличимы для scalar critic, ToleranceNet и local-only FELLAS, однако

$$
I(\{g_A,g_B\})=0.5\ \text{vs}\ 0,
\qquad
T(\{g_A,g_B\})=0.5\ \text{vs}\ 1.
$$

Downstream-задача: выбрать пару backup grasps, максимизирующую $T(\{g_1,g_2\})$, то есть вероятность успеха хотя бы одного grasp. Главный результат — FELLAS выбирает комплементарную пару и повышает empirical backup success относительно ToleranceNet/scalar/local-only baselines; high-order Brier является вторичной диагностикой.

---

## 44. Топ-2 способа кодировать наблюдение для FELLAS

### 44.1 Boundary-Ray Query Transformer — основной вариант

**Мотивация.** Стоит кодировать не полную форму occluder-а, а границу того, что observation сообщает о target: видимую поверхность, depth discontinuities и censored camera rays. Это переносит идею occupied + neighboring empty evidence из [ME-PCN (ICCV 2021)](https://arxiv.org/abs/2108.08187) в query-conditioned grasp encoder и добавляет сочетание local query geometry с global context, мотивированное [Local Occupancy-Enhanced Grasping](https://arxiv.org/abs/2407.15771). Знак depth discontinuity как cue порядка foreground/background используется в [RGB-D Edge Detection](https://ece.umn.edu/~cchoi/pub/Choi13iros_edge.pdf).

**Стартовый численный бюджет:**

| Компонент | Количество |
|---|---:|
| Visible target points | $N_t=1024$ |
| Boundary anchors | $N_b=128$–$256$ |
| Rays на anchor | $L=4$ |
| Всего ray tokens | $N_r=L N_b=512$–$1024$ |
| Всего входных tokens | $N_t+N_r=1536$–$2048$ |
| Tokens после encoder/compression | $256$–$512$ |
| Cross-attention blocks | $2$–$4$ |
| Canonical gripper points | $N_g=16$–$32$ |

Для target point $q_i$:

$$
t_i=
[q_i,n_i,\phi_i^{RGB},c_i^D,c_i^M,d_i^{\partial M}]
\in\mathbb R^{d_\phi+9}.
$$

Для boundary pixel $b$ с outward normal $n_b^{2D}$ и масштабов $s\in\{2,4,8\}$ pixels:

$$
z_{\mathrm{in}}^{(s)}=D(b-sn_b^{2D}),
\qquad
z_{\mathrm{out}}^{(s)}=D(b+sn_b^{2D}),
$$

$$
\Delta z_s=z_{\mathrm{out}}^{(s)}-z_{\mathrm{in}}^{(s)}.
$$

Один ray token можно кодировать как

$$
r_{b\ell}=
[v_{b\ell},z_{\mathrm{in}}^{(2)},z_{\mathrm{out}}^{(2)},
\Delta z_2,\Delta z_4,\Delta z_8,
d_{bq},n_b^{2D},c_D,c_M]
\in\mathbb R^{13}.
$$

Здесь $v_{b\ell}\in\mathbb R^3$ — camera-ray direction, а $d_{bq}$ — расстояние до ближайшего visible target point. Все $\Delta z$ передаются непрерывно; hard foreground/background rule не используется.

Grasp кодируется геометрией gripper-а, а не только pose-vector:

$$
P_g=\{R_gp_j+t_g\}_{j=1}^{N_g},
\qquad N_g=16\text{--}32,
$$

где canonical points лежат на finger pads, fingertips, closing corridor и palm. Полный расчёт:

$$
H_x=\operatorname{Enc}(\{t_i\}_{i=1}^{1024}\cup\{r_j\}_{j=1}^{N_r}),
\qquad
h_{\mathrm{global}}=\operatorname{Pool}(H_x),
$$

$$
\pi(x)=\operatorname{softmax}(W_\pi h_{\mathrm{global}}),
$$

$$
a_g=\operatorname{CrossAttn}(\operatorname{Enc}_g(P_g),H_x),
\qquad
f_m(x,g)=D_\theta(a_g,h_{\mathrm{global}},z_m).
$$

Так $\pi_m(x)$ получает глобальную информацию о hidden modes, а $f_m(x,g)$ выбирает локальные target/ray evidence, релевантные конкретному grasp.

### 44.2 ME-PCN-style two-branch encoder — безопасный baseline

**Мотивация.** Это более простой вариант с опубликованной [реализацией ME-PCN](https://github.com/Wenri/ME-PCN): отдельно кодировать observed target points и ближайшие informative empty rays, затем глобально слить признаки. Он дешевле в реализации, но хуже сохраняет query-local evidence для конкретного $g$.

Для controlled comparison использовать тот же бюджет: $1024$ target points и $512$–$1024$ ray features. ME-PCN-style ray feature:

$$
e_{ik}=[p^e_{ik},D_{ik},v_{ik}]\in\mathbb R^9,
$$

где $p^e_{ik}\in\mathbb R^3$ — empty point, $D_{ik}\in\mathbb R^3$ — vector offset к target point, $v_{ik}\in\mathbb R^3$ — ray direction. Выбирать преимущественно четыре ближайших informative rays для каждого из $128$–$256$ boundary-adjacent target anchors:

$$
E_Q=\operatorname{Enc}_Q(Q),
\qquad
E_R=\operatorname{Enc}_R(R^*_{\mathrm{ray}}),
$$

$$
h=\operatorname{Fuse}(\operatorname{Pool}(E_Q),
\operatorname{Pool}(E_R)),
\qquad
(h,P_g,z_m)\mapsto f_m(x,g).
$$

Числа выше — стартовая конфигурация FELLAS, а не заявленные hyperparameters оригинального ME-PCN.

### Общие ограничения и решающая ablation

Полный observed scene depth не подавать в stochastic encoder, но использовать как deterministic collision gate:

$$
\text{target + boundary/rays}\rightarrow\text{FELLAS},
\qquad
\text{full observed PCD}\rightarrow\text{collision filter}.
$$

Не требуются full occluder mask, full occluder PCD или voxel grid. Сравнение при одинаковом CEN head и token budget:

$$
\text{target only}
\;\text{vs}\;
+\Delta z\text{ boundary}
\;\text{vs}\;
+\text{ray tokens}
\;\text{vs}\;
+\text{full scene PCD}.
$$

Главная проверка мотивации: `boundary + ray` должен быть не хуже `full scene PCD` на unseen occluder shapes при меньшем числе nuisance scene features.

---

## 45. Откуда семплировать candidates и global $K$

Training должен повторять inference pipeline:

$$
S\rightarrow x_{\mathrm{occ}}
\xrightarrow{\text{frozen GraspGen}}
C(x_{\mathrm{occ}})=\{g_1,\ldots,g_N\},
\qquad
K\sim\nu(\cdot\mid C(x_{\mathrm{occ}})).
$$

**GraspGen видит только occluded observation $x$; GT/full mesh $S$ используется исключительно label oracle-ом:**

$$
Y_j=Y(S,g_j).
$$

Нельзя выбирать candidates или $K$ по $Y/F$ либо из full-shape grasp distribution: это создаёт selection bias и меняет elicited target CQS.

Global $K$ выбирать label-free по геометрическому разнообразию в candidate bank — distant pairs/quadruples, farthest-point или stratified sampling — используя

$$
d(g_i,g_j)^2=
\frac{\|t_i-t_j\|^2}{\sigma_t^2}+
\frac{d_{SO(3)}(R_i,R_j)^2}{\sigma_R^2}+
\frac{|w_i-w_j|^2}{\sigma_w^2}.
$$

Если GraspGen даёт слишком узкий support, для training anchors использовать label-free смесь

$$
q_{\mathrm{train}}(g\mid x)=
0.8q_{\mathrm{GraspGen}}(g\mid x)+
0.2q_{\mathrm{broad}}(g\mid x),
$$

где $q_{\mathrm{broad}}$ семплирует вокруг видимого target/bounding volume. На inference остаются только GraspGen candidates. Обязательно отдельно измерять `oracle success@GraspGen pool`: FELLAS не может выбрать хороший grasp, которого нет в candidate bank.

---

## 46. Headline для fixed-SKU warehouse-задачи

Основной источник uncertainty формулировать как

$$
\boxed{\textbf{observation-equivalent rigid worlds induced primarily by pose/view ambiguity}.}
$$

Для известного SKU $S_0$ определить sensor-equivalent pose fiber

$$
\mathcal T_\varepsilon(x)=
\{T\in SE(3):d_x(\mathcal R(TS_0),x)<\varepsilon\}.
$$

Разные $T_k\in\mathcal T_\varepsilon(x)$ дают почти одинаковое RGB-D observation, но разные hidden poses детали вроде ручки и поэтому разные feasible sets:

$$
S_k=T_kS_0,
\qquad
d_x(x_i,x_j)<\varepsilon,
\qquad
F_{S_i}\neq F_{S_j}.
$$

Dataset должен содержать группы `same/near-same observation × different plausible poses × different feasible sets`; метрики $d_x$ считать по visible depth, contour и boundary/ray evidence с sensor tolerance. Pose ambiguity одного SKU — главный эксперимент, instance ambiguity и combined instance+pose — вторичные.

Это естественная non-identifiability single-view perception для partially symmetric rigid objects, а не искусственная генерация разных hidden backsides; прямой precedent — [Humt et al., Shape Completion with Prediction of Uncertain Regions (IROS 2023)](https://elib.dlr.de/195724/1/Humt_etal_IROS23.pdf).

Обязательный сильный baseline — **pose-posterior / symmetry-aware CAD** с несколькими pose hypotheses: если SKU известен, можно хранить его CAD вместо обучения random feasible set. FELLAS должна либо превзойти этот baseline по compute/data efficiency при сопоставимом decision quality, либо показать преимущество generalization на unknown/new objects, для которых CAD недоступен.
