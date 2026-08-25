# AvoGrasp: conditional avoidance fields вместо shape completion для grasping под окклюзией

**Статус:** research proposal после первичного novelty-аудита

**Дата среза литературы:** 25 августа 2026

**Целевой venue:** ICLR

**Сценарий:** один target на полке, один foreground-occluder, один шумный RGB-D кадр с wrist camera, parallel-jaw gripper, короткое закрытие и подъём на малую высоту

**Вне scope:** RL, VLA, активное получение нового вида, удаление препятствия, clutter-removal, планирование всей траектории руки, полная 3D-реконструкция сцены или объекта

## 0. Решение в одном абзаце

Предлагается заменить стандартный learning target «восстановить скрытую форму» и стандартный target «предсказать success одного grasp» на новый объект:

> **условный avoidance functional случайного множества неуспешных parallel-jaw действий в пространстве grasp-поз.**

Полная геометрия объекта во время обучения индуцирует множество неуспешных grasp-поз $\mathcal B(S)$. После частичного наблюдения $X=x$ это множество неизвестно и становится random closed set. Для компактного pose packet $K$, например малой окрестности номинального grasp, модель предсказывает

$$
A^\star(K\mid x)
=
\Pr\!\left(\mathcal B(S)\cap K=\varnothing\mid X=x\right).
$$

Это вероятность того, что **ни одна** поза из заданной окрестности не является неуспешной при истинной, скрытой форме. Singleton $K=\{g\}$ даёт обычную вероятность успеха grasp; непустая окрестность является принципиально более сильным запросом. Архитектура AvoGrasp представляет условный закон $\mathcal B(S)\mid X=x$ как малую смесь латентных implicit sets непосредственно в action space, а не как смесь 3D shapes. Обучение выполняется новым integrated avoidance-Brier objective на compact-set queries. На population level этот objective является proper для avoidance functional; при достаточно богатом семействе запросов capacity functional однозначно задаёт закон random closed set. На inference не строятся mesh, occupancy, SDF или completed point cloud.

Главная формула статьи:

$$
g^\star
=
\arg\max_{g\in\mathcal C(x)}
\widehat A_\theta(K_{\rho_{\rm hw}}(g)\mid x),
$$

где $\mathcal C(x)$ — candidate pool, а $K_{\rho_{\rm hw}}(g)$ — калиброванная окрестность допустимой ошибки позы gripper.

## 1. Почему это не «ещё один grasp scorer»

Обычный discriminative grasp network учит

$$
p_{\rm point}(g,x)=\Pr(Y(S,g)=1\mid X=x).
$$

Он отвечает только на вопрос о единственной математически точной позе. Две позы могут иметь одинаковый point success, но одна может лежать на узком пике: малое смещение приводит к провалу. Вторая может лежать внутри широкой области успешных действий. При шумном RGB-D и неоднозначной скрытой форме это различие существенно.

AvoGrasp учит событие более высокого порядка:

$$
Z_K(S)=
\mathbf 1\{\forall h\in K:\;Y(S,h)=1\}.
$$

Соответственно:

$$
A^\star(K\mid x)=\mathbb E[Z_K(S)\mid X=x].
$$

Это не свёртка point score с известным pose-noise distribution. Свёртка оценивает средний успех случайно выбранной perturbation:

$$
\mathbb E_{\Delta}[Y(S,g\circ\Delta)].
$$

Avoidance query оценивает вероятность того, что **весь заданный uncertainty set** не пересекает множество неуспеха. Поэтому он различает широкую безопасную область и набор разрозненных успешных точек с тем же средним score.

Также это не «causal failure modes». $\mathcal B(S)$ — только математическое множество action configurations, не прошедших один стандартизованный grasp test. Модель не строит причинную таксономию, не прогнозирует последовательность отказов и не оценивает весь manipulation cycle.

## 2. Точная постановка

### 2.1 Скрытое состояние и наблюдение

Пусть $S$ содержит полную геометрию и физические параметры target, необходимые только simulator/oracle во время обучения. Камера формирует

$$
X=\Pi_{\omega}(S,O)+\varepsilon,
$$

где $O$ — foreground-occluder, $\Pi_{\omega}$ — физический RGB-D rendering с depth ordering, $\omega$ — camera/occluder configuration, а $\varepsilon$ — sensor noise.

На inference доступны:

- RGB-D;
- target mask или target ID плюс segmentation module;
- видимые target points;
- видимые obstacle points;
- camera rays и метки observed/free/unknown;
- calibration gripper-camera.

Предсказание amodal mask или полной формы не требуется.

### 2.2 Action space

Действие

$$
g=(T,w)\in
\mathcal G
\subset
SE(3)\times[w_{\min},w_{\max}]
$$

задаёт final parallel-jaw frame и commanded opening width. Рассматривается компактная рабочая область $\mathcal G$, ограниченная shelf ROI и параметрами gripper.

Метрика задаётся в hardware-normalized координатах:

$$
d_{\mathcal G}(g,h)^2
=
\frac{\|t_g-t_h\|_2^2}{\sigma_t^2}
+
\frac{\|\log(R_g^\top R_h)\|_2^2}{\sigma_R^2}
+
\frac{(w_g-w_h)^2}{\sigma_w^2}.
$$

$\sigma_t,\sigma_R,\sigma_w$ измеряются по calibration/repeatability робота, а не подбираются для улучшения test score.

### 2.3 Что считается успехом

$Y(S,g)=1$, если при известной полной target geometry:

1. gripper, уже находясь в final/pre-contact configuration, может закрыться;
2. формируется устойчивый parallel-jaw contact;
3. target удерживается при фиксированном коротком подъёме $\epsilon_{\rm lift}$.

В этот label не входят global reachability, движение humanoid к полке, motion planning от текущей конфигурации, оценка длинной траектории или полное поднятие. Коллизия открытого gripper с **видимой** частью obstacle в final configuration проверяется отдельным консервативным geometric filter и не превращается в изучение всего manipulation cycle.

### 2.4 Random closed set неуспешных действий

Для полной сцены введём oracle margin $q(S,g)$: положительное значение означает успешный standardized grasp test, ноль — границу, отрицательное — отказ. Тогда:

$$
\mathcal B(S)
=
\{g\in\mathcal G:q(S,g)\le 0\}.
$$

При непрерывном $q$ это замкнутое множество, причём $Y(S,g)=\mathbf 1\{q(S,g)>0\}$: boundary points консервативно считаются отказом. Если simulator возвращает только binary labels, практическая аппроксимация $\mathcal B(S)$ строится как замыкание отрицательных samples с заранее фиксированным local interpolation/certification rule. В таком случае точный объект эксперимента — конечный pose packet, а не заявленная без доказательства continuous guarantee.

После наблюдения $X=x$ форма неизвестна, поэтому

$$
\mathcal B_x\sim P(\mathcal B(S)\mid X=x).
$$

Для compact $K\subset\mathcal G$ определим capacity и avoidance:

$$
T^\star(K\mid x)
=
\Pr(\mathcal B_x\cap K\ne\varnothing\mid x),
$$

$$
A^\star(K\mid x)
=
1-T^\star(K\mid x)
=
\Pr(\mathcal B_x\cap K=\varnothing\mid x).
$$

Выражение $A^\star(K\mid x)$ и есть новый learning target.

### 2.5 Какие $K$ нужны

Основной deployment query:

$$
K_\rho(g)=\{h:d_{\mathcal G}(g,h)\le\rho\}.
$$

На практике непрерывный шар аппроксимируется детерминированным pose packet:

$$
\widetilde K_{\rho,L}(g)
=
\{g\circ\Delta_\ell\}_{\ell=1}^{L},
$$

где perturbations покрывают translation, rotation и width boundary. Для richer supervision используются:

- singleton packets $K=\{g\}$;
- nested packets $K_{\rho_1}(g)\subset K_{\rho_2}(g)$;
- anisotropic packets из эмпирической hardware error model;
- небольшая доля finite unions разнесённых packets, чтобы учить зависимости между action regions, а не только point marginals.

## 3. Новый learning objective

### 3.1 Integrated avoidance-Brier score

Для training tuple $(S,x,K)$ oracle label:

$$
Z_K(S)=
\mathbf 1\{\mathcal B(S)\cap K=\varnothing\}.
$$

Модель возвращает $\widehat A_\theta(K\mid x)\in[0,1]$. Основной objective:

$$
\mathcal L_{\rm avoid}(\theta)
=
\mathbb E_{S,X}
\mathbb E_{K\sim\nu}
\left[
\left(
\widehat A_\theta(K\mid X)-Z_K(S)
\right)^2
\right].
$$

$\nu$ — заранее определённое распределение compact queries. Оно должно покрывать singleton, local robustness packets и finite unions. Баланс достигается sampling queries, а не label-dependent weighting: последнее разрушило бы простой properness argument.

### 3.2 Proposition 1: propriety

Для фиксированных $x,K$:

$$
\mathbb E[(a-Z_K)^2\mid x,K]
=
(a-A^\star(K\mid x))^2
+
A^\star(K\mid x)(1-A^\star(K\mid x)).
$$

Следовательно, единственный population minimizer:

$$
a=A^\star(K\mid x).
$$

То есть objective честно elicит вероятность avoidance event, а не произвольный ranking score.

### 3.3 Proposition 2: идентифицируемость random set

Классическая random-set theory утверждает: закон random closed set однозначно задаётся capacity functional на compact sets. Пусть $\mathcal K_0$ — **счётная convergence-determining family** compact sets для выбранной topology на $\mathcal G$. При нужных regularity assumptions её можно построить из finite unions balls с центрами и радиусами из счётных плотных множеств. Тогда, если:

1. action ROI является compact metric space;
2. $\nu(\{K\})>0$ для каждого $K\in\mathcal K_0$, а не только имеет topological support около этих queries;
3. model class достаточно выразителен;

то population minimizer integrated Brier risk совпадает с $A^\star$ на каждом $K\in\mathcal K_0$. Равенство на convergence-determining family идентифицирует условный закон $\mathcal B(S)\mid X=x$, а не только независимые pointwise probabilities. Здесь положительная масса каждого элемента важна: одно лишь dense continuous sampling без дополнительных continuity assumptions не даёт этот вывод автоматически.

Практическая модель не должна заявлять exact recovery по конечным данным. Теорема задаёт корректный population object и объясняет, почему packets/unions содержат больше информации, чем singleton BCE.

### 3.4 Proposition 3: Bayes decision

Для utility

$$
U_K(S,g)=
\mathbf 1\{\mathcal B(S)\cap K_\rho(g)=\varnothing\},
$$

решение

$$
g^\star=\arg\max_g A^\star(K_\rho(g)\mid x)
$$

является Bayes-optimal. Это соответствие между training target и deployment decision. Completion metrics вроде Chamfer distance в objective не нужны.

### 3.5 Почему не BCE на perturbed grasps

Независимый BCE для каждого $h\in K$ учит только marginals

$$
\Pr(Y(S,h)=1\mid x).
$$

Из них нельзя восстановить

$$
\Pr(\forall h\in K:Y(S,h)=1\mid x),
$$

потому что успехи соседних poses зависимы через одну и ту же скрытую форму. Packet-level label обучает именно joint event.

## 4. Новая архитектура: AvoGrasp Conditional Avoidance Field

### 4.1 Архитектурный объект

AvoGrasp не выводит shape samples. Он выводит небольшую условную смесь implicit sets в action space:

$$
\widehat{\mathcal B}_x
\sim
\sum_{m=1}^{M}\pi_m(x)\,
\delta_{\mathcal B_{\theta,m}(x)}.
$$

Каждый atom:

$$
\mathcal B_{\theta,m}(x)
=
\{g:s_{\theta,m}(x,g)\le0\},
$$

где $s_{\theta,m}$ — непрерывное implicit action-margin field. Латентный atom не обязан соответствовать какой-либо 3D shape; он кодирует только один совместно согласованный вариант множества grasp outcomes.

Это важное отличие от shape-completion ensemble: latent capacity тратится на decision boundary в низкоразмерном action space, а не на grasp-irrelevant back surface.

### 4.2 Observation encoder

Encoder получает sparse token set трёх типов:

1. **target-visible tokens:** 3D point, RGB feature, estimated normal, depth-noise scale;
2. **obstacle-visible tokens:** 3D point и occluder flag;
3. **visibility-ray tokens:** camera origin/direction, first observed depth, state observed-free/unknown.

Ray tokens не образуют dense SDF. Они нужны, чтобы сеть различала «пространство измерено как пустое» и «пространство скрыто obstacle».

Backbone должен быть $SE(3)$-equivariant или по меньшей мере использовать gripper-relative coordinates. Практический первый вариант:

- sparse point/ray transformer;
- equivariant vector features для geometry;
- global target token для mixture weights;
- local multi-scale tokens для query decoder.

### 4.3 Latent set atoms

Global encoder output создаёт:

$$
\{(\pi_m,z_m)\}_{m=1}^{M},
\qquad
\pi_m\ge0,\quad\sum_m\pi_m=1.
$$

$M=8$ или $16$ — стартовый диапазон. $z_m$ не является shape code. Он параметризует коррелированную карту успешности по многим grasp queries.

Mixture atoms нужны не ради красивой визуализации. Без shared latent atom модель легко сведётся к независимым point probabilities и потеряет информацию о том, какие action regions появляются или исчезают совместно при разных скрытых shapes.

### 4.4 Gripper-chart query decoder

Для candidate $g$:

1. target, obstacle и ray tokens переводятся в gripper frame;
2. cross-attention queries соответствуют левой jaw sweep, правой jaw sweep и центральному grasp corridor;
3. relative features агрегируются только в ограниченных gripper-centric neighborhoods;
4. decoder получает $z_m$, local features и $g$, после чего выдаёт $s_{\theta,m}(x,g)$.

Таким образом computation зависит от числа queried grasps, но не от resolution полного 3D volume.

Можно обеспечить equivariance:

$$
s_{\theta,m}(Tx,Tg)=s_{\theta,m}(x,g)
$$

для совместного rigid transform $T$, если все geometric interactions строятся из gripper-relative invariants/equivariants.

### 4.5 Capacity-valid compact-set decoder

Для дискретного packet $K=\{g_\ell\}_{\ell=1}^{L}$:

$$
r_m(K,x)
=
\min_{\ell}s_{\theta,m}(x,g_\ell).
$$

Hard avoidance prediction:

$$
\widehat A_\theta^{\rm hard}(K\mid x)
=
\sum_{m=1}^{M}
\pi_m(x)\,
\mathbf 1\{r_m(K,x)>0\}.
$$

Это **точный avoidance functional** finite-mixture random closed set. Если $K_1\subseteq K_2$, то:

$$
\widehat A_\theta^{\rm hard}(K_2\mid x)
\le
\widehat A_\theta^{\rm hard}(K_1\mid x).
$$

Для обучения:

$$
\operatorname{softmin}_{\tau_q}\{s_\ell\}
=
-\tau_q\log\sum_{\ell=1}^{L}\exp(-s_\ell/\tau_q),
$$

$$
\widehat A_\theta(K\mid x)
=
\sum_m\pi_m(x)\,
\sigma\!\left(
\operatorname{softmin}_{\tau_q}
\{s_{\theta,m}(x,g_\ell)\}/\tau_s
\right).
$$

Ненормированный softmin выбран намеренно: добавление query points не увеличивает его, поэтому antitonicity сохраняется и в smooth approximation.

Теоретический claim о точном avoidance functional относится к hard decoder. Smooth expression — дифференцируемый training surrogate: antitonicity у него сохраняется, но полный набор Choquet-capacity axioms автоматически не следует. После обучения primary calibrated prediction следует считать hard mixture; smooth prediction допустима как отдельная practical approximation только с явным отчётом discrepancy и capacity-violation tests.

Если continuous field $s_{\theta,m}$ является $L_s$-Lipschitz, а packet — $\epsilon$-net множества $K$, то:

$$
\left|
\min_{g\in K}s_{\theta,m}(x,g)
-
\min_{g\in\widetilde K}s_{\theta,m}(x,g)
\right|
\le L_s\epsilon.
$$

Это даёт понятный trade-off между packet density и approximation error.

### 4.6 Candidate generation

Главный научный claim относится к selection, поэтому proposal mechanism должен быть отделён:

- visible-contact proposals от Contact-GraspNet/OrbitGrasp-like generator;
- coarse object-centric $SE(3)$ seeds в ROI;
- дополнительные camera-ray seeds внутри occlusion cone;
- 5–10 gradient-ascent steps по $\log \widehat A_\theta(K_{\rho_{\rm hw}}(g)\mid x)$.

В selector-only benchmark все методы получают один candidate pool. В full-system benchmark AvoGrasp использует одинаковый strong proposal generator с ближайшими discriminative baselines. Это не позволит приписать selector преимущество более богатому sampler.

### 4.7 Inference

1. Получить RGB-D и target mask.
2. Построить sparse point/ray tokens.
3. Сгенерировать $N$ candidate grasps.
4. Отфильтровать candidates, чья final open-gripper geometry пересекает inflated visible obstacle points.
5. Для каждого $g_i$ сформировать hardware packet $K_{\rho_{\rm hw}}(g_i)$.
6. Одним batched pass получить $\widehat A_\theta(K_{\rho_{\rm hw}}(g_i)\mid x)$.
7. Выбрать максимум или abstain, если максимум ниже $1-\alpha$.

Если abstention в продукте запрещён, модель всё равно выбирает argmax и возвращает calibrated risk. В статье нужно показывать и forced-attempt, и selective режимы.

## 5. Как получить supervision без полной реконструкции на inference

### 5.1 Training scenes

Training-time meshes допустимы: постановка говорит, что скрытая grasp-relevant geometry известна через training distribution. Для каждого target mesh:

- разместить target на shelf;
- физически разместить ровно один foreground obstacle;
- выбрать wrist-camera pose;
- отрендерить RGB-D с реалистичным depth ordering;
- добавить measured RealSense-like noise, edge dropout, multipath/outlier model;
- сохранить full geometry только для oracle labeling.

Random point dropout не должен заменять физическую окклюзию: он не воспроизводит contiguous unknown regions и связь obstacle boundary с hidden target.

### 5.2 Oracle grasp test

Для каждого nominal candidate $g$:

- вычислить быстрый analytic antipodal/collision proxy;
- запустить batched closing simulation;
- выполнить фиксированный малый lift;
- получить $Y(S,g)$.

Для packet $K$ выполнить oracle test для всех его pose samples:

$$
Z_K(S)=\prod_{h\in K}Y(S,h).
$$

Continuous-ball claim допустим только после adaptive refinement packet и empirical Lipschitz audit. До этого корректное название target — **finite pose-packet avoidance**.

### 5.3 Query sampling curriculum

Наивные random packets почти всегда полностью good или полностью bad и дают мало boundary information. Нужны:

1. $30\%$ singleton queries;
2. $40\%$ nested local packets вокруг успешных/пограничных grasps;
3. $20\%$ hard packets, где рядом есть positive и negative poses;
4. $10\%$ finite unions удалённых packets для joint-dependence supervision.

Проценты являются стартовой гипотезой и должны аблироваться.

### 5.4 Несколько occlusions одной формы

Один и тот же $S$ нужно рендерить при нескольких obstacle positions и visibility levels. Это даёт:

- одинаковый oracle action set при разных объёмах наблюдаемой информации;
- прямую проверку calibration versus occlusion;
- возможность проверить tower/coherence property без требования инвариантных predictions.

Опциональная абляция может добавить martingale-consistency regularizer между nested observations, но он **не должен быть core novelty**: в мае 2026 уже появился general-ML preprint Martingale-Consistent Self-Supervised Learning. Кроме того, unbiased conditional-refinement sampling здесь нетривиален. Основной AvoGrasp objective самодостаточен.

### 5.5 Полная training loss

Минимальный вариант:

$$
\mathcal L
=
\mathcal L_{\rm avoid}
+
\lambda_{\rm eq}\mathcal L_{\rm equiv}
+
\lambda_{\rm lip}\mathcal L_{\rm local\text{-}Lip}.
$$

Где:

- $\mathcal L_{\rm avoid}$ — единственный probabilistic target;
- $\mathcal L_{\rm equiv}$ проверяет joint rigid-transform consistency;
- $\mathcal L_{\rm local\text{-}Lip}$ ограничивает чрезмерно рваное action field и делает packet approximation контролируемым.

Не следует добавлять reconstruction, Chamfer или occupancy loss в main model. Иначе paper потеряет центральный тезис «predict the decision set, not the hidden state».

## 6. Последовательный поиск идеи и причины отбраковки альтернатив

### Шаг 1: deterministic full completion

Идея: complete target point cloud, затем применить grasp detector.

Почему отброшено:

- это уже центральная архитектура TARGO-Net;
- более ранние работы делают shape completion for grasping;
- новая работа Single-View Shape Completion for Robotic Grasping in Clutter использует diffusion completion;
- TARGO сам сообщает, что shape completion хуже переносится на real noise;
- completion optimizes огромное число grasp-irrelevant surface variables.

Вывод: направление важно как baseline, но не имеет требуемой conceptual novelty.

### Шаг 2: posterior over complete shapes + LCB/CVaR

Идея: семплировать несколько plausible shapes и выбирать grasp по expected quality или lower confidence bound.

Почему отброшено:

- Robust Grasp Planning Over Uncertain Shape Completions делает MC-dropout shape samples с 2019 года;
- UNCLE-Grasp в версии от августа 2026 агрегирует force-closure variability across completion hypotheses и использует conservative LCB/abstention;
- стоимость растёт как number of shapes $\times$ number of grasps;
- uncertainty остаётся привязана к качеству full reconstruction.

Вывод: uncertainty-aware selection эффективна, но полный shape posterior не нов.

### Шаг 3: completion только contact region

Идея: reconstruct лишь потенциальные contact patches.

Почему отброшено:

- TOSC (AAAI 2026) прямо формулирует task-oriented shape completion, фокусируясь на potential contact regions;
- NeuGraspNet интерпретирует grasping как local neural surface rendering;
- всё ещё требуется выбрать геометрическое представление скрытого contact patch, хотя для решения нужен outcome set.

Вывод: task relevance правильна, но contact geometry всё ещё лишний intermediate target.

### Шаг 4: deterministic implicit grasp distance field

Идея: учить distance from query $g$ to valid grasp manifold.

Почему отброшено:

- NGDF уже учит distance до valid grasp level set;
- свежий Configuration-Space Grasp Distance Fields (август 2026) строит smooth distance to finite grasp configurations для control/safety;
- deterministic distance не выражает multimodal uncertainty скрытой формы;
- distance to nearest annotated grasp и probability that a whole query packet avoids failure — разные объекты, но первый уже занят.

Вывод: action-space field полезен как architectural evidence, но novelty должна быть probabilistic set law + compact queries.

### Шаг 5: direct scalar success probability

Идея: учить $\Pr(Y=1\mid x,g)$, возможно с quantile/LCB.

Почему отброшено:

- это стандартный grasp evaluator;
- не кодирует joint outcome соседних poses;
- не различает широкий plateau и хрупкий peak при одинаковом point score;
- не даёт закон feasible/failure set.

### Шаг 6: martingale-consistent occlusion loss

Идея: сделать predictions coherent при nested occlusions.

Почему не выбран как core:

- общая формализация уже опубликована как arXiv preprint в мае 2026;
- conditional refinement sampling для скрытых shapes сложен;
- consistency — regularizer, но не новый output object.

Вывод: допустима абляция, но не центральный contribution.

### Шаг 7: random closed action set + avoidance functional

Почему оставлен:

- ни одна найденная grasping work не учит conditional capacity/avoidance functional множества неуспешных grasp actions;
- объект прямо соответствует selection under latent shape ambiguity;
- singleton строго включает обычный classifier как special case;
- architecture может гарантировать set-functional coherence;
- loss имеет population properness и связь с Choquet random-set theory;
- inference не материализует скрытую 3D форму.

## 7. Карта ближайшей robotics-литературы

| Работа | Что делает | Почему не закрывает AvoGrasp |
|---|---|---|
| S4G, CoRL 2020 | Прямо регрессирует amodal SE(3) grasps из single-view point cloud | Deterministic per-point proposal/quality; нет conditional random action set |
| Contact-GraspNet, ICRA 2021 | Генерирует 6-DoF grasps из raw depth, anchored at visible contacts | Может быть proposal backbone; скрытые action-region dependencies не моделируются |
| VGN, CoRL 2020 | Dense TSDF-to-grasp field, 10 ms | Требует volumetric TSDF; point score, не avoidance of compact pose sets |
| GIGA, RSS 2021 | Joint implicit reconstruction and affordance | Geometry auxiliary target остаётся центральным |
| TARGO/TARGO-Net, 2024–2026 | Benchmark по occlusion; target completion + target-scene transformer | Deterministic full target completion; стандартные grasp outputs |
| NeuGraspNet, RSS 2024 | Global implicit reconstruction + local surface rendering per grasp | Deterministic local geometry and scalar quality |
| Robust Grasp Planning Over Uncertain Shape Completions, IROS 2019 | MC shape completions и robust metric | Материализует full shapes; нет learned compact-set probability |
| UNCLE-Grasp, arXiv v3 Aug 2026 | Completion samples, force-closure variability, LCB abstention | Domain-specific strawberry pipeline; full completion; object-level LCB |
| PUGS | Пропускает MVS occupancy uncertainty в grasp selection | Multi-view reconstruction uncertainty, не single-view latent-shape action-set law |
| Johns et al., 2016 | Сглаживает grasp function известной pose uncertainty | Expected smoothed point quality, не avoidance probability скрытого failure set |
| NGDF, 2022 | Unsigned distance to valid-grasp manifold | Deterministic nearest-manifold cost; не partial-occlusion posterior |
| TOSC, AAAI 2026 | Completion potential contact regions для task-oriented dexterous grasp | Всё ещё completion; другая hand/task постановка |
| Configuration-Space GDF, Aug 2026 | Distance field + CBF/CLF controller для execution | Known finite grasp configurations, motion/control, не learning under hidden shape |

**Novelty claim, который можно защищать после дополнительного search:** первая conditional random-closed-set formulation для grasp selection, первый compact-set avoidance objective в grasp learning и первая capacity-valid action-space architecture, которая marginalizes hidden shape without reconstructing it.

Формулировка «первая» пока является рабочей гипотезой, а не установленным фактом. Перед submission обязателен отдельный Semantic Scholar/Google Scholar citation chase по терминам random feasible set, stochastic viability set, excursion-set prediction, grasp success region и chance-constrained grasping.

## 8. Косвенные свидетельства, что направление может сработать

### 8.1 Hidden geometry действительно нужна

TARGO показывает, что performance текущих моделей падает с occlusion. В ablation удаление shape completion у TARGO-Net даёт до $18\%$ падения на high occlusion. Значит, raw visible geometry недостаточна и training distribution shapes несёт полезный prior.

### 8.2 Но deterministic completion — плохое bottleneck

TARGO отдельно отмечает более сильное падение в real setting и связывает discrepancy с noise, затрудняющим shape completion. Это прямо поддерживает замену geometry reconstruction на decision-space marginalization.

### 8.3 Учет uncertainty улучшает реальные grasps

IROS 2019 Robust Grasp Planning Over Uncertain Shape Completions сообщает статистически значимое улучшение относительно deterministic completion. UNCLE-Grasp v3 при примерно $87\%$ synthetic occlusion на physical robot сообщает success among attempted grasps $0.800$ против $0.483$ у strongest completed baseline, хотя с меньшим attempt rate. Это сильное свидетельство полезности uncertainty, но также показывает необходимость честно строить risk-coverage curve.

### 8.4 Непрерывная структура action set полезна

NGDF сообщает большой прирост execution success по сравнению с discrete-grasp baselines, а Johns et al. показывают пользу выбора pose, окружённой хорошей областью grasp function. Это не доказывает AvoGrasp, но поддерживает тезис, что topology/width успешного action region важнее single peak.

### 8.5 Proper set prediction practically learnable

В general ML Object Detection as Probabilistic Set Prediction показывает, что random-set representation и proper scoring rule могут улучшать probabilistic modeling без потери основного task metric. AvoGrasp переносит не конкретную detector architecture, а более общий принцип: set-valued outcome должен прогнозироваться как единый stochastic object.

### 8.6 Математическая экономия

Full shape completion прогнозирует $10^4$–$10^6$ geometric degrees of freedom. AvoGrasp прогнозирует $M$ low-dimensional implicit action fields и вычисляет их только в queried poses. Если downstream utility зависит от geometry только через $Y(S,g)$, action-set law является decision-sufficient, тогда как shape posterior содержит нерелевантные детали.

## 9. Главные риски и способы фальсифицировать идею

### Риск 1: avoidance event слишком консервативен

Для большого $K$ событие «все poses успешны» почти всегда ложно.

**Mitigation:** hardware-calibrated малый packet; curves по $\rho$; альтернативный $q$-content target только как отдельная версия:

$$
\mathbf 1\left\{
\Pr_{\Delta\sim q_\rho}
[Y(S,g\circ\Delta)=1]\ge1-\beta
\right\}.
$$

Core paper лучше начать с finite packets, где semantics точна.

### Риск 2: label cost

Packet из $L$ poses увеличивает simulation cost в $L$ раз.

**Mitigation:** analytic prefilter, batched simulation, nested packets с reuse labels, adaptive refinement только у boundary grasps. Сначала выполнить oracle study на уже размеченном dense grasp set.

### Риск 3: mixture atoms collapse

Модель может использовать все atoms одинаково и фактически стать point classifier.

**Диагностика:** pairwise field disagreement, effective mixture size, packet calibration, performance на finite-union queries. Diversity penalty разрешён только после проверки, что он не ухудшает probabilistic calibration.

### Риск 4: candidate recall

Лучший robust grasp может отсутствовать в candidate pool.

**Mitigation:** отдельно измерять oracle top-$N$ recall; selector-only и full-system результаты; camera-ray seeds; differentiable refinement.

### Риск 5: conditional prior shift

Если test object вне training shape distribution, calibrated probability может стать overconfident.

**Mitigation:** instance-disjoint и category-disjoint splits; OOD score по encoder; abstention; небольшая real calibration split. Нельзя заявлять distribution-free guarantee.

### Риск 6: target segmentation leakage

GT mask может искусственно упростить задачу.

**Mitigation:** основной scientific comparison с common GT/predicted masks, затем end-to-end result с одним frozen segmenter. Не смешивать segmentation improvement с grasp objective.

### Риск 7: capacity theorem окажется декоративным

Если experiments используют только singleton и один radius, reviewer справедливо назовёт random-set theory переименованием robustness score.

**Mitigation:** обязательны multi-radius calibration, finite-union queries, architectural monotonicity audit и ablation independent marginals versus shared latent sets.

## 10. Falsification gates до большого проекта

### Gate 0: расширенный novelty search

**Go**, если после citation chase не находится работа, которая одновременно:

1. моделирует feasible/failure grasps как conditional random closed set;
2. учит hit/avoidance probabilities compact pose sets;
3. делает это из partial RGB-D без shape completion.

Иначе proposal нужно пересобрать.

### Gate 1: oracle value of information

Имея full shape и dense success labels, сравнить:

- oracle point-success selection;
- oracle expected perturbed success;
- oracle packet-avoidance selection.

**Go**, если packet oracle улучшает high-occlusion, pose-perturbed short-lift success хотя бы на практически значимую величину и не уничтожает coverage. Если oracle advantage отсутствует, learning не спасёт objective.

### Gate 2: learnability на synthetic

Сравнить одинаковый backbone:

- singleton BCE;
- independently smoothed BCE;
- AvoGrasp $M=1$;
- AvoGrasp $M>1$ + union packets.

**Go**, если AvoGrasp лучше по packet Brier, risk-coverage и selected-grasp success на unseen instances.

### Gate 3: no-completion versus completion

Сравнить TARGO-Net, completion ensemble и AvoGrasp при равном candidate pool.

**Go**, если AvoGrasp одновременно:

- не хуже по clean/low-occlusion;
- лучше на high occlusion + noise;
- быстрее или существенно дешевле completion ensemble.

### Gate 4: real transfer

**Go для ICLR**, если advantage сохраняется на physical RGB-D и real robot, а calibration degradation измерена, не скрыта.

## 11. Экспериментальный дизайн

### 11.1 Benchmark A: controlled single-occluder shelf

Факторы:

- visibility bins: $0.2$–$1.0$;
- object instance/category;
- target minimum dimension;
- obstacle width/position;
- camera yaw/pitch;
- depth noise levels;
- pose perturbation radii;
- sim and real sensor.

Сцены не должны становиться cluttered: один target, один obstacle, shelf surfaces.

### 11.2 Benchmark B: TARGO-compatible transfer

Хотя TARGO cluttered, его occlusion-balanced data позволяет проверить, сохраняется ли benefit за пределами узкой постановки. Это secondary result, не основной benchmark.

### 11.3 Baselines

Обязательные:

1. Contact-GraspNet или OrbitGrasp direct proposal/score;
2. VGN/GIGA-like implicit affordance;
3. TARGO-Net;
4. TARGO-Net without shape completion;
5. MC completion + mean score;
6. MC completion + LCB/CVaR;
7. direct scalar $p(Y=1\mid x,g)$ с тем же AvoGrasp encoder;
8. pose-noise smoothed scalar grasp function;
9. partial-observation NGDF adaptation;
10. oracle full-shape selector upper bound.

### 11.4 Primary metrics

- **Selected Grasp Success Rate** после фиксированного short lift;
- **Packet Success Rate** under calibrated perturbations;
- **risk-coverage curve** и area under risk-coverage;
- **high-occlusion success** в заранее заданных bins;
- **integrated Brier score** по query radii;
- **ECE/calibration slope** по occlusion bins;
- candidate recall;
- latency, peak memory, number of geometry queries.

Chamfer/IoU completion quality можно показывать только для completion baselines как diagnostic, но не как main metric.

### 11.5 Critical ablations

- singleton versus packets;
- independent point marginals versus shared latent set atoms;
- $M=1,4,8,16$;
- balls only versus balls + finite unions;
- no ray tokens;
- random dropout versus physical occluders;
- no hard boundary mining;
- no equivariance;
- softmin temperature;
- packet density $L$;
- no gradient refinement;
- GT versus predicted mask;
- optional martingale regularizer.

### 11.6 Statistical protocol

- splits только по object identity, дополнительный category-disjoint split;
- одинаковые scenes и candidate pools для selector comparison;
- минимум 5 training seeds для synthetic;
- paired bootstrap confidence intervals по scenes;
- randomized-block physical trials по object/occlusion;
- заранее указать primary endpoint и high-occlusion range;
- публиковать forced-attempt и selective metrics;
- не выбирать $\rho_{\rm hw}$ на test set.

## 12. Expected paper claims

Claim 1:

> Partial-observation grasping naturally induces a conditional random feasible-action set; point scores and shape completions are unnecessarily restrictive representations of this object.

Claim 2:

> Integrated avoidance scoring is proper for compact-set safety events and, over a convergence-determining query family, identifies the conditional random-set law.

Claim 3:

> A capacity-valid mixture of implicit action fields can learn this object without reconstructing hidden geometry.

Claim 4:

> Under severe occlusion and sensor noise, direct action-set marginalization yields better robust grasp selection and calibration than deterministic completion, completion ensembles and scalar grasp scorers at lower inference cost.

Claim 4 нельзя писать до реальных результатов.

## 13. ICLR acceptance audit

ICLR 2026 формулирует основной вопрос review как «принесёт ли работа sufficient value и new knowledge», требует well-motivated placement, support for claims, scientific rigor и significance; SOTA сам по себе не обязателен.

### Originality

**Потенциал: высокий.** Новый output object, proper objective и capacity-valid architecture образуют одну концептуальную линию. Это сильнее, чем добавить uncertainty head к TARGO.

**Угроза:** reviewer может увидеть «robust classification over perturbations». Защита требует finite-union queries, set-law theorem и shared-latent ablation.

### Technical quality

**Потенциал: высокий.** Proposition 1 элементарна и точна; random-set identification опирается на классическую теорию; architecture гарантирует antitonicity by construction.

**Угроза:** continuous-set claims при finite packet. Нужно чётко разделить exact finite-packet model и asymptotic $\epsilon$-net approximation.

### Significance

**Потенциал: высокий при general framing.** «Predict conditional feasible sets instead of hidden states» применимо к manipulation, motion primitives, design feasibility и medical treatment sets.

**Угроза:** один robot setup будет выглядеть как narrow application. Нужен минимум один non-grasp synthetic benchmark random feasible-set learning либо убедительная general theorem/algorithm section.

### Empirical rigor

**Сейчас: главный риск.** Для ICLR недостаточно показать 20 robot grasps. Нужны controlled synthetic study, strong completion/uncertainty baselines, category splits, calibration и physical validation.

### Reproducibility

Нужно выпустить:

- scene generator;
- fixed mesh splits;
- pose-packet definitions;
- oracle labels или label generator;
- candidate pools для fair selector benchmark;
- code и checkpoints;
- exact hardware calibration metric.
- disclosure существенного использования LLM в ideation/writing, если оно попадает под требования Author Guide; сам факт использования не заменяет проверку формул, источников и экспериментальных claims авторами.

### Итоговая честная оценка

Идея имеет ICLR-level conceptual core, но пока не ICLR-level evidence. Paper становится сильным только если Gate 1 доказывает ценность avoidance objective ещё до learning, а Gate 3 показывает преимущество над completion ensemble. Без этих двух результатов математическая упаковка не компенсирует слабый empirical effect.

## 14. Минимальный implementation roadmap

### Phase 1: oracle notebook, 1–2 недели

- взять 200–500 meshes;
- получить dense grasp labels;
- построить finite packets вокруг grasps;
- сравнить oracle selection objectives;
- построить success versus $\rho$ curves.

Результат: решение Gate 1 без training нового network.

### Phase 2: scalar and $M=1$ prototype, 2–3 недели

- common point/ray encoder;
- continuous gripper query field;
- singleton BCE baseline;
- $M=1$ packet avoidance;
- calibration and boundary mining.

### Phase 3: latent random-set mixture, 2–4 недели

- $M>1$ atoms;
- finite-union query training;
- capacity monotonicity tests;
- category-disjoint evaluation.

### Phase 4: completion baselines and real sensor, 3–5 недель

- TARGO-Net/MC completion;
- measured sensor noise;
- predicted masks;
- runtime profiling.

### Phase 5: physical robot, 2–4 недели

- calibrate pose-error metric;
- randomized blocked trials;
- risk-coverage and forced-attempt results.

## 15. Unit tests для научной реализации

1. **Set inclusion:** если $K_1\subseteq K_2$, hard и soft predictions не должны давать $A(K_2)>A(K_1)+\epsilon$.
2. **Singleton reduction:** $K=\{g\}$ должен совпадать с mixture point-success prediction.
3. **Permutation invariance:** порядок poses внутри packet не влияет на output.
4. **Joint rigid equivariance:** совместный transform scene и grasp не меняет score.
5. **Packet refinement:** prediction стабилизируется при увеличении packet density.
6. **Atom coherence:** один atom задаёт совместные outcomes для всех queried grasps.
7. **No geometry leakage:** inference graph не читает full mesh/completed point cloud.
8. **Calibration:** synthetic oracle frequency совпадает с predicted avoidance в bins.
9. **Noise monotonicity audit:** это не hard constraint, но severe noise не должен систематически повышать confidence.
10. **Candidate fairness:** сравниваемые selectors получают byte-identical pose pools.

## 16. Возможное название и abstract

### Название

**AvoGrasp: Learning Conditional Avoidance Fields for Reliable Grasping under Occlusion**

Альтернатива с более general-ML framing:

**Learning Random Feasible Sets from Partial Observations: An Application to Occlusion-Robust Grasping**

Второе название лучше для ICLR, если будет general benchmark.

### Draft abstract

Reliable decisions from partial observations need not require reconstructing the hidden state. We study single-view parallel-jaw grasping when a foreground object occludes grasp-relevant target geometry. Existing methods either predict grasps from visible points, reconstruct one complete shape, or sample complete shapes and aggregate grasp scores. We instead model the feasible grasp region itself as a conditional random closed set in action space. For any compact packet of nearby grasp poses, our model predicts its avoidance probability: the probability that the packet does not intersect the latent failure set. We introduce an integrated avoidance score that is proper for these compact-set events and identify conditions under which it recovers the conditional random-set law. Our architecture represents this law as a capacity-valid mixture of implicit action fields and answers grasp-conditioned queries without producing meshes, occupancy grids, or completed point clouds. [Experiments placeholder: controlled simulation, category-disjoint objects, noisy real RGB-D, physical humanoid.] The method should be claimed to improve robust grasp success and calibration under severe occlusion only after the corresponding experiments are complete.

## 17. Что нельзя заявлять

- «Certified safe grasp» без formal coverage assumptions и exact continuous verification.
- «Distribution-free» без conformal layer и exchangeability assumptions.
- «Первая uncertainty-aware grasp model» — это неверно.
- «Первая action-space grasp field» — NGDF и другие поля уже существуют.
- «Не использует shape prior» — использует training distribution shapes.
- «Восстанавливает скрытую геометрию implicit» — это противоречит тезису.
- «SOTA» только по собственному узкому benchmark без TARGO/completion/uncertainty baselines.
- «Continuous packet guarantee», если labels и inference используют лишь sparse finite samples.

Корректная формулировка:

> AvoGrasp directly learns a calibrated conditional law over grasp-feasibility sets induced by hidden geometry, and evaluates finite pose-packet avoidance without reconstructing that geometry.

## 18. Литература и источники

### Occlusion and grasping

1. [TARGO: Benchmarking Target-driven Object Grasping under Occlusions](https://arxiv.org/abs/2407.06168), 2024; expanded TARGO-Net accepted in IJCV 2026.
2. [TARGO project and current benchmark results](https://targo-benchmark.github.io/).
3. [S4G: Amodal Single-view Single-Shot SE(3) Grasp Detection in Cluttered Scenes](https://proceedings.mlr.press/v100/qin20a.html), CoRL 2020.
4. [Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes](https://arxiv.org/abs/2103.14127), ICRA 2021.
5. [Volumetric Grasping Network](https://arxiv.org/abs/2101.01132), CoRL 2020.
6. [GIGA: Synergies Between Affordance and Geometry](https://arxiv.org/abs/2104.01542), RSS 2021.
7. [NeuGraspNet](https://www.roboticsproceedings.org/rss20/p046.pdf), RSS 2024.
8. [Shape Completion Enabled Robotic Grasping](https://arxiv.org/abs/1609.08546), IROS 2017.
9. [Single-View Shape Completion for Robotic Grasping in Clutter](https://arxiv.org/abs/2512.16449), 2025.
10. [TOSC: Task-Oriented Shape Completion](https://ojs.aaai.org/index.php/AAAI/article/view/38053), AAAI 2026.

### Uncertainty and robust grasp selection

11. [Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645), IROS 2019.
12. [UNCLE-Grasp v3](https://arxiv.org/abs/2601.14492), revised 23 August 2026.
13. [PUGS: Perceptual Uncertainty for Grasp Selection](https://onurbagoren.github.io/PUGS/).
14. [Deep Learning a Grasp Function for Grasping under Gripper Pose Uncertainty](https://arxiv.org/abs/1608.02239), 2016.
15. [Robotic Pick-and-Place With Uncertain Object Instance Segmentation and Shape Completion](https://pmc.ncbi.nlm.nih.gov/articles/PMC8022832/), IEEE T-ASE 2021.

### Action-space fields

16. [Neural Grasp Distance Fields for Robot Manipulation](https://arxiv.org/abs/2211.02647), 2022.
17. [Grasp Execution Without a Planner: Configuration-Space Grasp Distance Fields with Certified Safety & Guaranteed Quality](https://arxiv.org/abs/2608.00600), submitted 1 August 2026.
18. [OrbitGrasp: SE(3)-Equivariant Grasp Learning](https://arxiv.org/abs/2407.03531), CoRL 2024.

### Random sets, proper scoring and partial observation

19. [Theory of Random Sets, Molchanov](https://www.nzdr.ru/data/media/biblio/kolxoz/M/MD/Molchanov%20I.%20Theory%20of%20Random%20Sets%20%28ISBN%20185233892X%29%28Springer%2C%202005%29%28501s%29_MD_.pdf), Springer, 2005.
20. [Introduction to Stochastic Geometry: random closed sets and Choquet capacity](https://www.cmm.minesparis.psl.eu/~figliuzzi/Stochastic_Geometry.pdf).
21. [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf), JASA 2007.
22. [Object Detection as Probabilistic Set Prediction](https://arxiv.org/abs/2203.07980), 2022.
23. [Martingale-Consistent Self-Supervised Learning](https://arxiv.org/abs/2605.11846), May 2026.

### Venue criteria

24. [ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide).
25. [ICLR 2026 Author Guide](https://iclr.cc/Conferences/2026/AuthorGuide).

## 19. Финальный research verdict

**Оставить и проверять экспериментально.**

AvoGrasp отличается от плохого шаблона «собрать несколько robotics pipelines» тем, что сначала меняет математический объект задачи:

$$
\text{hidden shape}
\longrightarrow
\text{random feasible action set}
\longrightarrow
\text{avoidance queries}.
$$

Robotics-работы здесь используются для проверки gap и косвенной эмпирической поддержки, а не как источник архитектурной композиции. Источник формализации — random closed sets, capacity functionals и proper scoring.

Самая сильная часть идеи — не mixture network сама по себе, а согласованная тройка:

1. новый target $A^\star(K\mid x)$;
2. proper compact-set objective;
3. architecture, которая по построению является conditional random set и сохраняет antitonicity.

Самая опасная часть — возможность, что oracle packet-avoidance selection не даст выигрыша над хорошо откалиброванным expected-success scorer. Поэтому Gate 1 является обязательным и должен быть выполнен до масштабной разработки.
