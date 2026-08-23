# Sensor-Null Action Spectra for Parallel-Jaw Grasping

## Итоговый вердикт

Самая сильная из рассмотренных постановок — не ещё один grasp scorer и не
восстановление скрытой формы, а изучение **размерности неидентифицируемости
действий** при одном RGB-D наблюдении.

Рабочее название paper:

> **What Can a Gripper Know from One View? Sensor-Null Action Spectra for
> Occluded Grasping**

Одно предложение, содержащее всю идею:

> Две полные геометрии могут быть неразличимы для wrist RGB-D, но давать разные
> функции успешности grasp во всём пространстве действий; мы предлагаем измерять
> сингулярный спектр этого сенсорно-невидимого изменения, проверяем гипотезу о его
> низкой эффективной размерности при реальном execution noise и учим компактный
> эллипсоид целых utility-функций, не восстанавливая скрытую форму.

Это **conditional go**, а не обещание accept. Постановка имеет ICLR-потенциал
только в том случае, если предварительный эксперимент обнаружит сильный и
неочевидный spectral law. Без этого результата оставшаяся модель будет выглядеть
как conditional low-rank regression с robust argmax и должна быть отвергнута.

## 1. Точная задача и сознательно исключённый scope

Дано одно RGB-D наблюдение $o$ target object на полке с wrist camera. Перед
объектом может находиться один фронтальный occluder. Сцена не cluttered. Облако
точек может содержать пропуски и реалистичный depth noise. Считается, что target
mask или crop уже получен существующим perception stack.

Модель должна выбрать parallel-jaw grasp $g$ из компактной рабочей области

$$
\mathcal G\subset (SE(3)/C_2)\times[w_{\min},w_{\max}],
$$

где $C_2$ учитывает симметрию параллельных губок, а $w$ — commanded opening.
Оценивается только terminal experiment:

1. открытый gripper уже помещён в заданную grasp pose;
2. губки закрываются с фиксированными speed/force limits;
3. объект поднимается на заранее фиксированные 1–2 cm;
4. успех означает, что объект удержан после этого малого lift.

Полный approach trajectory, global reachability, whole-cycle feasibility и
последующее manipulation не являются label и не входят в claim. Кандидаты можно
предварительно фильтровать существующим motion stack; paper изучает качество
terminal grasp при частичном наблюдении. RL и VLA не используются. На вход модели
не подаются mesh, voxel grid или scene SDF.

## 2. Почему обычный deterministic grasp target структурно неверен

Пусть $z\in\mathcal Z$ — полное локальное физическое состояние: невидимая
геометрия target и фиксированная геометрия полки/occluder. Для изоляции явления
friction, density и controller держатся фиксированными; они не превращаются в
длинный список дополнительных latent variables. Пусть

$$
R:\mathcal Z\to\mathcal O
$$

— RGB-D rendering/sensing map. Operational grasp utility определяется сразу как
вероятность terminal success при измеренной ошибке исполнения $\xi$:

$$
q_z(g)=\Pr_{\xi}\{\text{close-and-lift succeeds for }g\circ\xi\mid z\}.
$$

Таким образом, $Q:z\mapsto q_z(\cdot)$ отображает полное состояние не в один
score, а в функцию из пространства действий в $[0,1]$.

Для наблюдения $o$ множество допустимых полных состояний есть sensor fiber

$$
\mathcal C_\varepsilon(o)
=\{z:d_{\mathcal O}(R(z),o)\le\varepsilon,\ z\text{ удовлетворяет объявленному
hidden-shape family}\}.
$$

Ключевой объект — не posterior над shape, а его pushforward в action space:

$$
\mathcal U(o)=\{q_z(\cdot):z\in\mathcal C_\varepsilon(o)\}.
$$

Если существуют $z_0,z_1$ с одинаковым наблюдением, но
$q_{z_0}\ne q_{z_1}$, то не существует единственной физически правильной
функции $f(o,g)$, которую мог бы восстановить deterministic learner. Большая
сеть не решает эту проблему: target сам не является функцией наблюдения.

При известном достоверном prior по hidden variants можно максимизировать mean или
CVaR по $\mathcal U(o)$. При отсутствии такого prior основной decision rule —
maximin

$$
g^\star(o)\in\arg\max_{g\in\mathcal G}
L(o,g),\qquad
L(o,g)=\inf_{q\in\mathcal U(o)}q(g),
$$

с возможностью abstain, если $\max_g L(o,g)$ ниже порога. Научный объект
$\mathcal U(o)$ не зависит от выбора mean/CVaR/maximin; maximin используется как
строгий тест качества выученного множества.

## 3. Новый объект: sensor-null action spectrum

### 3.1 Локальная версия

На одном гладком visibility stratum предположим дифференцируемость $R$ и $Q$.
Tangent directions, невидимые сенсору в первом порядке, образуют

$$
\mathcal N_z=\ker DR_z.
$$

Определим **sensor-null action operator**

$$
A_z=DQ_z\big|_{\mathcal N_z}:
\mathcal N_z\longrightarrow L^2(\mathcal G,\nu_G).
$$

Его образ содержит все первые порядки изменения *целой grasp-utility функции*,
которые не проявляются в RGB-D. Сингулярные значения

$$
s_1(A_z)\ge s_2(A_z)\ge\cdots
$$

называются **sensor-null action spectrum**, а число значений выше operational
tolerance $\delta$ — **action ambiguity dimension**.

Это не intrinsic dimension полной формы. Скрытая поверхность может иметь сотни
степеней свободы, но, возможно, менять функции grasp success только по нескольким
согласованным patterns: например, одновременно портить grasps с одной стороной
закрытия или переносить устойчивость между двумя families контактов.

Спектр зависит от объявленной нормы допустимых shape perturbations. Поэтому
нельзя называть его абсолютным свойством камеры. В экспериментах hidden
deformations whitened относительно явно заданного high-dimensional random-field
measure; более invariant глобальный объект ниже зависит только от множества
utility functions.

Для noisy RGB-D hard nullspace можно заменить soft invisibility operator

$$
A_z^{(\tau)}=
DQ_z\left(I+\tau^{-2}DR_z^*\Sigma_o^{-1}DR_z\right)^{-1/2},
$$

где $\Sigma_o$ — измеренная sensor covariance. Видимые направления подавляются,
а направления ниже noise floor сохраняются. Это extension для анализа, а не
дополнительный tensor на входе grasp network.

### 3.2 Конечная нелинейная версия

Для произвольного, возможно disconnected fiber измерим centered Kolmogorov width

$$
d_r(\mathcal U(o))=
\inf_{\mu,\,\dim V\le r}
\sup_{q\in\mathcal U(o)}
\inf_{v\in V}\|q-\mu-v\|.
$$

Главная эмпирическая гипотеза paper:

> При реалистичном hidden-shape family и при smoothing, точно соответствующем
> измеренному execution noise робота, $d_r(\mathcal U(o))$ быстро падает, хотя
> dimension скрытой геометрии велик; при стремлении execution noise к нулю это
> сжатие заметно ухудшается.

Эта гипотеза falsifiable. Она не должна быть записана в abstract как факт до
pilot study.

## 4. Теоретическое ядро

### 4.1 Impossibility: observation-only point prediction не может быть правильным

Рассмотрим два latent states с observation distributions $P_0,P_1$. Обозначим
statewise regret через
$r_i(g)=\max_h q_i(h)-q_i(g)$ и потребуем нетривиальную pairwise decision
separation

$$
\Delta=\inf_g\bigl(r_0(g)+r_1(g)\bigr)>0.
$$

Тогда ни один компромиссный третий grasp не является хорошим сразу в обоих
states, и стандартный overlap/Le Cam argument даёт для любого observation-only
selector при равном prior

$$
\mathbb E[\operatorname{regret}]
\ge \frac{\Delta}{2}\bigl(1-\operatorname{TV}(P_0,P_1)\bigr).
$$

Для exact sensor twins $P_0=P_1$, поэтому bound равен $\Delta/2$. Это не
generalization error и не нехватка model capacity, а неидентифицируемость.

Дополнительный factorization result делает связь точной. Если $R$ — submersion,
его fibers connected и $DQ_g\ker DR=0$ в каждой точке, то $Q_g$ постоянно на
fibers и существует $\bar Q_g$, для которой

$$
Q_g=\bar Q_g\circ R.
$$

Нулевой sensor-null spectrum означает, что обычный deterministic grasp target
локально корректен; ненулевой spectrum измеряет нарушение этого допущения.

### 4.2 Local nonlinear compression

Пусть $\psi:B_{\mathcal H}(c)\to\mathcal C(o)$ — chart гладкого fiber,
$F=Q\circ\psi$, $A=DF(0)$, а $\|D^2F\|\le K$. Для пространства $V_r$,
натянутого на первые $r$ left singular functions $A$, Taylor theorem даёт

$$
\sup_{\|h\|\le c}
\operatorname{dist}(F(h)-F(0),V_r)
\le c\,s_{r+1}(A)+\frac{Kc^2}{2}.
$$

Следовательно, spectrum контролирует nonlinear width в локальном fiber с явно
отделённым curvature term.

### 4.3 Почему execution noise может физически создавать low rank

Пусть $Y_z(g)$ — sharp success field, а $K_\xi$ — Markov operator measured
pose-noise kernel. Тогда operational utility

$$
q_z=K_\xi Y_z.
$$

Если $B_z=DY_z|_{\ker DR_z}$ bounded, то

$$
A_z=K_\xi B_z,
\qquad
s_j(A_z)\le \|B_z\|\,s_j(K_\xi).
$$

Любой compact smoothing kernel тем самым ограничивает эффективный action rank.
Для isotropic diffusion на компактном action manifold
$K_\xi=H_t=e^{-t\Delta_{\mathcal G}}$, поэтому

$$
s_j(A_z)\le\|B_z\|e^{-t\lambda_j}.
$$

Weyl law $\lambda_j\asymp j^{2/d}$ даёт decay вида
$\exp(-c t j^{2/d})$. Это не обещает rank 8 в полном 6–7D action space; именно
поэтому spectral pilot является обязательным. Для uniform action selection
имеется bound

$$
\|(I-P_r)H_tf\|_\infty
\le
\sup_g K_t(g,g)^{1/2}
e^{-t\lambda_{r+1}/2}\|f\|_2.
$$

Anisotropic measured covariance соответствует anisotropic diffusion, а не
обязательной isotropic Gaussian approximation.

### 4.4 Spectrum-to-decision identity

На общем конечном candidate set пусть $e_m$ выбирает action $g_m$. Для
линейного fiber ball

$$
q(h)=q_0+Ah,\qquad \|h\|_2\le c,
$$

точная robust utility имеет форму

$$
\begin{aligned}
L_m
&=\inf_{\|h\|\le c}e_m^T(q_0+Ah)\\
&=q_{0,m}-c\|A^Te_m\|_2\\
&=q_{0,m}-c\sqrt{\sum_j s_j^2u_j(m)^2}.
\end{aligned}
$$

Это ключевой мост между spectrum и моделью. Если оставить $r$ modes, то
выброшенная row sensitivity не превосходит $cs_{r+1}$ на дискретном candidate
set. После вычитания $cs_{r+1}+Kc^2/2$ truncated score остаётся conservative
при указанных assumptions.

### 4.5 Stability learned set

Если true и predicted sets utility functions находятся на Hausdorff distance не
больше $\delta$ в $L^\infty$, то их lower envelopes отличаются не больше чем
на $\delta$. Если $\hat g$ максимизирует learned lower envelope, а $g^\star$
— true, то

$$
L(g^\star)-L(\hat g)\le2\delta.
$$

Таким образом, uniform set approximation — ровно та learning objective, которая
контролирует downstream decision. Average PCA reconstruction такой гарантии не
даёт.

## 5. Efficiently learnable formalization: SNAE

### 5.1 Представление

Назовём модель **Sensor-Null Action Ellipsoid (SNAE)**. Frozen или jointly
trained high-recall candidate generator получает только $o$ и возвращает

$$
G_o=\{g_1,\ldots,g_M\}.
$$

Один point/ray encoder обрабатывает noisy RGB-D. Query decoder для каждого grasp
возвращает

$$
\mu_m=\mu_\theta(o,g_m),\qquad
b_m=b_\theta(o,g_m)\in\mathbb R^r,
$$

а group head возвращает residual radius $\rho(o)\ge0$. Матрица
$B=[b_1^T;\ldots;b_M^T]$ задаёт ellipsoid

$$
\widehat{\mathcal U}_r(o)=
\{\mu+Ba+e:\|a\|_2\le1,\ \|e\|_\infty\le\rho\}.
$$

SVD $B=U\operatorname{diag}(d)V^T$ даёт learned action modes и их ordered
amplitudes. Поскольку Euclidean ball invariant к $V$, right rotation не меняет
множество. Это позволяет сети выдавать непрерывную mode field $b(o,g)$, а
orthogonalization выполнять только для анализа spectrum. Никакая hard QR,
зависящая от состава candidate set, не требуется.

Lower utility имеет closed form

$$
\widehat L(o,g_m)=
\operatorname{clip}_{[0,1]}
\bigl(\mu_m-\|b_m\|_2-\rho\bigr).
$$

Inference cost после RGB-D encoder равен $O(Mr)$; получение ordered spectrum
из малого Gram matrix $B^TWB$ стоит $O(Mr^2+r^3)$. При $M=512,r=8$ это
несколько тысяч scalars вместо сотен тысяч occupancy/SDF queries.

### 5.2 Grouped learning objective

Training example — не одна сцена, а observation group

$$
\mathcal D_b=(o_b,G_b,\{q_{b,s}\}_{s=1}^{S_b}),
$$

где все hidden twins имеют один и тот же observation и один и тот же candidate
set. Training-only code $a_{b,s}$ решает

$$
a_{b,s}^*=\arg\min_{\|a\|_2\le1}
\operatorname{LSE}_{m\in\Omega_{b,s}}
\left|q_{b,s,m}-\mu_{b,m}-b_{b,m}^Ta\right|.
$$

Коэффициент не оценивается на роботе: observation принципиально не содержит
информации, какой twin присутствует. Он нужен только для fitting образа fiber.

Outer loss непосредственно учит enclosing set:

$$
\begin{aligned}
\mathcal L_b={}&
\operatorname{LSE}_{s,m}
\left[
|q_{b,s,m}-\mu_{b,m}-b_{b,m}^Ta_{b,s}^*|-\rho_b
\right]_+\\
&+\lambda_\rho\rho_b
+\lambda_w\frac1M\sum_m\|b_{b,m}\|_2
+\lambda_{\rm rank}\sum_j d_{b,j}.
\end{aligned}
$$

Первая строка штрафует нарушение enclosure в uniform norm. Вторая минимизирует
residual, среднюю decision-relevant half-width и мягко удаляет лишние modes.
Log-sum-exp используется только для gradients; hard worst residual регулярно
добавляется обратно через mining. Split group-level calibration может прибавить
один held-out residual quantile к $\rho$, но conformal calibration не заявляется
как novelty.

### 5.3 Sparse supervision

Для каждого twin можно симулировать только случайное
$\Omega_{b,s}\subset[M]$, а остальные entries использовать как held-out. Low-rank
factor связывает изменения разных grasпов; direct lower-score baseline такой
связи не имеет. Однако sample-efficiency — эмпирический claim, не следствие слова
«low-rank». Нужно строить curves по числу simulated twin-action pairs и сравнивать
при строго одинаковом label budget.

### 5.4 Greedy sensor-twin mining

После каждой стадии обучения следует искать новый hidden variant, максимизирующий
не уже покрытую моделью компоненту:

$$
z^*=\arg\max_{z:\,d(R(z),o)\le\varepsilon}
\inf_{\|a\|\le1}
\|Q(z,G_o)-\mu-Ba\|_\infty.
$$

Он добавляется в group, и fitting повторяется. Это reduced-basis greedy algorithm
в task space. На inference shape completion не выполняется. Начинать следует с
procedural high-dimensional hidden surfaces; differentiable contact/render
optimization — только дополнительный stress test.

## 6. Sensor-Twin benchmark

### 6.1 Synthetic twins

Для каждого visible shell создаётся shadow volume, задаваемый camera rays и
фронтальным obstacle. Внутри него изменяется только невидимая часть target:

- гладкие random-field displacements с 32–64+ whitened degrees of freedom;
- локальные concavity/convexity и thickness changes;
- held-out procedural programs с holes, lips или rear appendages;
- изменения, влияющие на второй jaw contact и удержание, но не на видимый z-buffer.

Visible vertices, silhouette, texture, camera, obstacle и shelf фиксируются.
Render пары отклоняется, если RGB или depth различимы выше sensor tolerance. Для
exact synthetic twins один и тот же sampled noise realization накладывается после
rendering; отдельно тестируется независимый реалистичный noise.

Низкий action rank нельзя получать за счёт low-dimensional generator. Поэтому
обязательно публикуются dimension, covariance spectrum и held-out deformation
programs самого hidden generator.

Candidate generator запускается ровно один раз на $o_b$; его output копируется
всем twins. Генерировать candidates с full hidden mesh запрещено, иначе benchmark
утекает latent information.

### 6.2 Utility labels

Для каждого $(z_{b,s},g_m)$ выполняются common-random-number Monte Carlo rollouts
при measured gripper pose covariance. Label — beta-binomial/shrinkage estimate
close-and-1–2 cm-lift success probability. Один и тот же набор perturbations для
всех twins снижает variance их utility differences.

Simulator использует full geometry только для получения ground truth. Learned
model её не получает. Отдельно репортятся:

- candidate recall с full-mesh oracle;
- exact-pose binary field;
- smoothed utility при нескольких multiples реальной covariance;
- Monte Carlo confidence interval каждого reported spectral error.

### 6.3 Physical twins

Решающая real-world конструкция — 3D-printed object families с общей
camera-facing cap и сменными rear/side modules. Front obstacle закрывает стык и
скрытые modules. Camera pose, light, texture и visible surface не меняются.

До grasp tests проверяется именно observation equivalence:

1. pixelwise depth/RGB discrepancy относительно repeated-scan noise;
2. двухвыборочный test между observation distributions;
3. high-capacity RGB-D classifier hidden-module ID с confidence interval;
4. classifier с full mesh как positive control.

RGB-D classifier должен быть статистически неотличим от chance. Иначе это обычная
visual generalization задача, а не sensor-fiber ambiguity.

## 7. Обязательный дешёвый falsification pilot

До обучения большой модели:

- 50 visible-shell families;
- 64–128 sensor twins на family;
- 256–512 shared candidates;
- 32–64 common pose perturbations на twin-action pair;
- не меньше 32–64 независимых hidden deformation coordinates;
- exact field и utility fields при 0.5×, 1× и 2× measured covariance.

Для centered twin-by-action matrix $X_b$ считаются singular spectra, но только
Frobenius energy недостаточно. Дополнительно решается oracle rank $r$
$L^\infty$ enclosing-factor problem и измеряется downstream robust selection.

Проект продолжать только при одновременном выполнении условий:

1. $r\le8$ объясняет не меньше 90% centered energy минимум в 70% held-out
   shell groups.
2. В тех же groups 90-й percentile absolute entry error меньше 0.05 и uniform
   residual не доминирует ellipsoid width.
3. Low-rank oracle maximin уступает full utility-matrix oracle не более 5
   percentage points.
4. По **worst-twin utility** mean-score selector уступает full maximin не меньше
   10 points в заранее объявленном moderate/high-occlusion regime; его expected
   utility репортится отдельно и не обязана быть хуже.
5. Хотя бы в половине таких scenes существует common grasp с worst-twin utility
   выше 0.7; иначе benchmark демонстрирует только невозможность.
6. Action rank заметно растёт при уменьшении execution noise; иначе предложенный
   physical mechanism не подтверждается.
7. RGB-D twin discriminator остаётся на chance.

Любой из первых трёх failures убивает spectral model. Failure пунктов 4–5 убивает
robust-grasping use case. Failure пункта 6 не обязательно убивает empirical law,
но требует удалить heat-smoothing causal explanation.

## 8. Полная экспериментальная программа

### 8.1 Splits

- unseen visible-shell geometry;
- unseen object categories;
- unseen hidden random-field seeds;
- unseen deformation programs/topology;
- unseen obstacle width/height while retaining a single occluder;
- synthetic-to-real printed twins;
- PCD dropout, multipath-like depth outliers и calibration noise.

Train/test split по hidden program обязателен: split только по object instance
проверяет interpolation внутри одного ambiguity generator.

### 8.2 Baselines с одинаковыми входом и label budget

1. Deterministic expected-success grasp scorer.
2. Direct worst-twin/lower-envelope scalar head — главный destructive baseline.
3. Independent per-grasp quantile или evidential head.
4. Deep ensemble scalar scorers.
5. Conditional PCA/SVD basis и ICLR-2026-style subspace regression.
6. Conditional scenario critics с $K$ целыми utility vectors.
7. Deterministic shape completion + grasp evaluator.
8. Multiple stochastic shape completions + mean/CVaR/maximin evaluation.
9. TARGO-style completion/occlusion-aware model, адаптированный к single-object
   setup.
10. Oracle full twin utility matrix, oracle full mesh и candidate-recall oracle.

FFHFlow является важным uncertainty-aware conceptual baseline, но работает с
dexterous pose distribution; его нельзя выдавать за прямой parallel-jaw
implementation baseline без корректной адаптации.

### 8.3 Metrics

- spectrum/width curves и learned-vs-oracle principal angles;
- held-out $L^2$, $L^\infty$ и Hausdorff-set proxy error;
- lower-envelope calibration, coverage и sharpness;
- expected, CVaR и worst-twin utility выбранного grasp;
- maximin regret относительно full-matrix oracle;
- physical grasp success по каждому hidden module, не только aggregate mean;
- abstention-risk curve;
- simulation label efficiency;
- inference latency/memory относительно multiple completions;
- candidate recall отдельно от reranking quality.

### 8.4 Ablations

- rank $r\in\{0,1,2,4,8,16,32\}$;
- exact utility против smoothed utility и несколько noise scales;
- ellipsoid против box, $K$-scenario set и direct lower head;
- average reconstruction loss против uniform enclosure loss;
- без hard twin mining;
- без group identity / со случайно перемешанными twins;
- 8, 16, 32, 64, 128 twins — демонстрация finite-sample rank bias;
- sparse label fraction;
- continuous query decoder против фиксированной candidate table;
- predicted residual $\rho$ против held-out calibrated residual.

## 9. Novelty boundary после adversarial literature audit

| Направление | Что уже существует | Чего оно не даёт |
|---|---|---|
| Direct partial-PCD grasping | Grasp score/poses непосредственно из partial point cloud | Не диагностирует, является ли score функцией observation; нет observation-equivalent utility sets |
| Robust shape completion | Несколько full-shape samples и aggregate grasp quality | Моделирует latent geometry, а не low-dimensional image ambiguity в action-function space |
| TARGO | Single-view target grasping под varying occlusion, geometry completion, benchmark по visibility | Не удерживает RGB-D фиксированным при изменении hidden geometry и не измеряет irreducible action ambiguity |
| Uncertainty-aware grasp generators | Distribution/likelihood grasp poses, OOD/view uncertainty | Не дают sensor-null operator или spectrum целых counterfactual utility fields |
| Partial identification / common actions | Utility bounds и actions, feasible для observation-consistent states | Нет learned action-function spectrum, spectral law или exact grasp twins |
| Goal-oriented inverse problems | Low-rank uncertainty QoI в основном для linear-Gaussian inverse problems | Не рассматривают sensor-null image в непрерывной action-utility function и physical grasp decision |
| Adaptive reduced bases / Grassmann regression | Neural prediction parameter-dependent subspaces | Conditional basis сам по себе уже занят; нет ambiguity set, sensor fibers или spectrum-to-grasp theorem |
| SNAE | Sensor-null operator, exact twins, action spectrum, enclosing functional ellipsoid, terminal grasp test | Новый claim, который всё ещё должен быть подтверждён pilot и финальным search |

Поэтому **не являются novelty**: SVD, ellipsoid, QR/SVD layer, robust argmax,
conformal residual, neural operator или low-rank factorization по отдельности.

Защищаемая contribution stack:

1. новая измеримая величина — sensor-null action spectrum;
2. impossibility–compressibility–decision chain теорем;
3. exact/indistinguishable sensor-twin benchmark;
4. первый убедительный empirical law о task-space dimension hidden grasp
   ambiguity;
5. SNAE как минимальная amortized реализация этого объекта;
6. physical sensor twins, показывающие, что явление не является renderer trick.

Прямого совпадения с этим stack в проведённом поиске не найдено. Это не позволяет
честно написать «first» без повторного поиска по Google Scholar, Semantic Scholar,
DBLP и свежим arXiv/venue proceedings непосредственно перед submission.
Финальный exact-phrase audit по сочетаниям *sensor-null action*, *nullspace to
action-value function*, *observation-equivalent grasp utility* и *sensor twins
for hidden grasp geometry* также не обнаружил совпадающей постановки; найденные
«physical twin» работы используют twin как повторяемый surrogate объекта, а не
как пару сенсорно-неразличимых геометрий с разными action utilities.

## 10. Mock review по официальным критериям ICLR

ICLR просит установить: конкретность вопроса, мотивированность и placement в
literature, достаточность доказательств и significance/new knowledge; SOTA не
обязателен. В текущем виде proposal отвечает этим критериям следующим образом.

### Вероятные strengths

- Вопрос точен: какая часть grasp utility не идентифицируется RGB-D и имеет ли
  эта часть малую action dimension?
- Problem existence доказывается sensor twins, а не выводится из общей фразы
  «occlusion is hard».
- Теория связывает impossibility, smoothing, spectrum и decision rule.
- Benchmark разрывает обычно сцепленные hidden geometry и visible observation.
- Модель компактна и operationally соответствует реальному gripper noise.
- Результат может обобщиться на другие partially observed physical actions.

### Вероятные причины weak reject

1. «Это conditional PCA plus robust grasping; SVD стандартна».
2. Exact twins выглядят искусственно, а admissible hidden family выбран авторами.
3. Low rank возникает потому, что deformation generator сам low-dimensional.
4. $L^2$ heat theorem не гарантирует малый rank в 6–7D и не контролирует
   optimizer-sensitive $L^\infty$ error.
5. Direct lower head получает тот же robust action дешевле.
6. Worst-case selection слишком conservative и не соответствует test prior.
7. Simulator utility и real utility плохо коррелируют.
8. Candidate generator не предлагает common grasp; reranker получает unfair blame.
9. Real RGB-D всё-таки выдаёт module через noise, lighting или calibration.
10. Claims локального differential operator смешаны с finite/disconnected twins.

Экспериментальная программа выше специально отвечает на каждый пункт. Особенно
важны high-dimensional/held-out generators, uniform error, direct-lower baseline,
candidate oracle и physical discriminator.

### Честная оценка acceptance potential

- **Сейчас, без pilot:** weak reject. Mathematical framing интересен, но central
  empirical law неизвестен.
- **После успешного spectral pilot, но без real twins:** borderline; reviewer
  может считать эффект synthetic artifact.
- **После всех go/no-go gates, сильного direct-lower comparison и physical
  twins:** правдоподобный ICLR accept-level submission, потому что paper сообщает
  новое и удивительное знание, а не только улучшает leaderboard.
- **Если SNAE не превосходит direct lower head:** архитектурный claim удалить.
  Возможен paper о spectrum/benchmark, только если empirical finding сам по себе
  силён и воспроизводим.

## 11. Ограничения, которые нужно объявить заранее

1. Любой robust set относителен к объявленному hidden-shape family; arbitrary
   unseen geometry делает lower utility тривиально нулевой.
2. Spectrum зависит от scale/metric hidden perturbations; глобальные utility
   widths и whitened deformation measure должны репортиться вместе.
3. Один ellipsoid плохо описывает сильно disconnected or asymmetric utility set.
   Большой residual $\rho$ должен честно показать failure, а не скрываться mixture
   model после просмотра результатов.
4. В zero-noise, point-contact limit action rank может быть большим.
5. Model не решает approach planning, reachability, clutter и long-horizon lift.
6. Worst-twin robustness может снизить average success при достоверном prior;
   поэтому mean/CVaR curves должны быть показаны наряду с maximin.
7. Guarantee относится к utility-set approximation и candidate set, а не к
   неограниченной физической безопасности.

## 12. Порядок исполнения проекта

1. Измерить wrist-camera depth noise и gripper pose covariance.
2. Реализовать exact shadow-volume twin generator с 64+ hidden coordinates.
3. Зафиксировать один high-recall observation-only candidate generator.
4. Провести pilot spectrum/width study без neural model.
5. Принять go/no-go решение по семи условиям раздела 7.
6. Только после go реализовать SNAE и destructive baselines.
7. Добавить greedy hidden-twin mining и sparse-label curves.
8. Напечатать modular physical twins и проверить observation equivalence до
   grasp trials.
9. Повторить novelty search непосредственно перед writing.
10. Строить paper вокруг empirical law и scientific object; architecture оставить
    средством проверки, а не источником завышенного novelty claim.

## 13. Основные источники

- Официальные критерии оценки ICLR 2027:
  https://iclr.cc/Conferences/2027/ReviewerGuidelines
- Обзор 6-DoF grasp synthesis: https://arxiv.org/abs/2207.02556
- Pose-noise convolution learned grasp function:
  https://arxiv.org/abs/1608.02239
- Robust grasp planning over uncertain completions:
  https://arxiv.org/abs/1903.00645
- TARGO benchmark/model: https://targo-benchmark.github.io/
- Shape completion with uncertain regions:
  https://hummat.github.io/2023-iros-uncertain/
- FFHFlow uncertainty-aware grasp generation:
  https://proceedings.mlr.press/v305/feng25a.html
- Variational neural belief grasping:
  https://arxiv.org/abs/2604.25897
- Goal-oriented low-rank inverse problems:
  https://arxiv.org/abs/1607.01881
- Adaptive learned parameter-dependent reduced basis:
  https://arxiv.org/abs/2105.14633
- Reduced-order modeling with Grassmann layers:
  https://proceedings.mlr.press/v145/bollinger22a.html
- ICLR 2026, *Deep Learning for Subspace Regression*:
  https://openreview.net/pdf?id=HF60Lu1Maj
- Conditional robust optimization:
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/3df874367ce2c43891aab1ab23ae6959-Abstract.html
- Functional prediction sets/zonotopes:
  https://openreview.net/pdf?id=TMg1JYoR4u
- Partial-identification lower-bound learner:
  https://proceedings.mlr.press/v202/oprescu23a.html
- Individualized decisions under partial identification:
  https://arxiv.org/abs/2110.10961

Полный журнал рассмотренных и отвергнутых направлений находится в
reports/ICLR_GRASP_IDEA_RESEARCH.md.
