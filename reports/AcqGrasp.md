# AcqGrasp: Acquisition-Equivalent Contact Learning for Parallel-Jaw Grasping

## Итоговый вердикт

Самая сильная новая постановка этого search pass — учить не функцию от конечного
point cloud, а **контактное решение на классе эквивалентных RGB-D acquisitions
одной физической поверхности**.

Рабочее название paper:

> **AcqGrasp: Learning Contact Decisions, Not RGB-D Sampling Laws**

Одно предложение, содержащее идею:

> Point-cloud grasp predictor может быть invariant к перестановке точек и при
> этом менять выбранный grasp при изменении pixel resolution, FPS budget,
> sampling density или depth-noise той же поверхности; мы формулируем
> acquisition-equivalent geometric decision learning, строим локальный
> deconvolved-quadrature слой для finite-scale contact functionals и проверяем
> его на paired acquisitions с неизменными physics labels.

Это **conditional go**, а не готовый acceptance claim. Идея имеет ICLR-потенциал
только при одновременном выполнении трёх условий:

1. обычные сильные grasp models действительно показывают material decision
   drift между реалистичными acquisitions одной поверхности;
2. PointConv или Monte Carlo Convolution вместе с сильной paired augmentation
   не устраняют этот drift;
3. аналитическая correction улучшает не только score consistency, но и
   downstream grasp regret и repeated real small-lift success.

Если хотя бы одно условие не выполняется, оставшаяся конструкция выглядит как
специализированный robust point-cloud layer и недостаточна для ICLR.

## 1. Точная задача и сознательно исключённый scope

Дано одно RGB-D наблюдение $o$ target object на полке с wrist camera. Перед
объектом может быть один фронтальный obstacle. Clutter из нескольких
взаимодействующих объектов не рассматривается. Point cloud может иметь
неравномерную плотность, пропуски, edge artifacts и реалистичный depth noise.
Target mask или crop считается полученным существующим perception stack.

Parallel-jaw grasp задаётся

$$
g=(R_g,t_g,w)
\in \mathcal G
\subset (SE(3)/C_2)\times[w_{\min},w_{\max}],
$$

где $C_2$ учитывает физическую симметрию губок, а $w$ — commanded opening.

Label соответствует стандартизованному terminal experiment:

1. внешний planner уже привёл открытый gripper в terminal pre-grasp pose;
2. губки закрываются с фиксированными speed и force limits;
3. объект поднимается на заранее заданные несколько миллиметров или 1–2 cm;
4. успех означает удержание после этого малого lift.

В paper не входят:

- global approach trajectory и reachability;
- whole-cycle feasibility от исходной позы humanoid;
- long-horizon lift или последующая manipulation;
- RL, VLA и language conditioning;
- scene-level SDF, dense voxel field или полная mesh reconstruction;
- causal taxonomy отдельных failure modes;
- active next-best-view planning.

Основная подзадача — **acquisition-stable candidate evaluation**. Один и тот же
observation-only candidate generator выдаёт общий набор

$$
G_o=\{g_1,\ldots,g_M\}.
$$

Все сравниваемые evaluators получают одинаковые candidates. Дополнительный
end-to-end experiment разрешён, но не может заменять controlled reranking test.

## 2. Почему permutation invariance недостаточно

Для point set $P=\{x_1,\ldots,x_n\}$ permutation-invariant model удовлетворяет

$$
f(\{x_1,\ldots,x_n\})
=f(\{x_{\pi(1)},\ldots,x_{\pi(n)}\}).
$$

Это утверждение ничего не говорит о следующих парах:

- $P$ и cloud с удвоенным числом samples на одной стороне объекта;
- native depth cloud и его FPS/voxel subsample;
- одна и та же depth surface при 320x240 и 1280x960;
- одна поверхность при axial noise 0.5 mm и 2 mm;
- одинаковый expected depth при разных edge dropout laws;
- clouds разных RGB-D устройств с разным pixel footprint.

Если samples распределены по поверхности с intensity $\lambda(s)$, обычное
unweighted pooling асимптотически интегрирует относительно
$\lambda(s)\,d\mu(s)$, а не относительно объявленной физической меры
$d\mu(s)$. Поэтому увеличение плотности одной области меняет representation,
хотя объект не меняется.

Для parallel jaws проблема особенно остра. Многие contact cues используют
локальные extrema или почти-extrema вдоль closing direction. При Gaussian
measurement noise raw maximum не только имеет variance, но и систематически
растёт с числом точек. Следовательно, более плотный cloud может сделать один и
тот же объект визуально «толще» для contact head.

Научный вопрос paper:

> Можно ли определить и эффективно выучить grasp decision, зависящий от
> латентной видимой поверхности и физического contact scale, но согласованный
> между различными объявленными sampling/noise laws этой поверхности?

## 3. Acquisition law как математический nuisance

### 3.1 Латентная поверхность и reference measure

Пусть $S\subset\mathbb R^3$ — видимая часть target surface из фиксированного
camera pose. Обозначим через $\mu_S$ заранее объявленную reference measure.

Два естественных варианта:

1. physical surface-area measure;
2. pushforward uniform image-plane measure через noiseless depth map.

Выбор нельзя скрывать. Термин «density invariant» бессодержателен без меры,
относительно которой должна считаться density. Surface-area measure лучше
соответствует физической площади pad contact, но требует Jacobian и local normal.
Ray-domain measure проще и стабильнее для одного RGB-D устройства. Обе версии
нужно сравнить в ablation.

Target и frontal obstacle можно задать двумя marked measures

$$
\mu=\mu_{\mathrm{target}}\oplus\mu_{\mathrm{obstacle}}.
$$

Это не scene SDF. Модель видит только конечные samples и локально запрашивает
малое число functionals для конкретного grasp.

### 3.2 Marked point-process observation

Acquisition $a$ создаёт

$$
O_a=\{(x_i,\omega_i,\Sigma_i,r_i)\}_{i=1}^{N_a},
$$

где

- $x_i\in\mathbb R^3$ — измеренная точка;
- $\omega_i>0$ — quadrature или inverse-propensity weight;
- $\Sigma_i\succeq0$ — covariance measurement error в 3-D;
- $r_i$ — RGB/semantic mark target или obstacle.

Удобная theoretical model:

$$
s_i\sim n_a\lambda_a(s)\,\mu_S(ds),
\qquad
x_i=s_i+\varepsilon_i,
\qquad
\varepsilon_i\mid s_i\sim\mathcal N(0,\Sigma_i).
$$

Notation $s_i\sim n_a\lambda_a\mu_S$ означает inhomogeneous Poisson process с
этой intensity measure. Для ideal Horvitz--Thompson correction

$$
\omega_i=\frac{1}{n_a\lambda_a(s_i)}.
$$

В реальном pipeline weight не обязательно оценивать по уже прореженному cloud.
Лучше сохранить source-pixel area, voxel occupancy count и sampling probability
через crop, masking и FPS. Noise covariance берётся из device calibration и
проецируется вдоль camera ray; learned residual допускается только как небольшая
calibration correction.

### 3.3 Эквивалентность не означает одинаковые finite samples

Два observations $O_a$ и $O_b$ называются acquisition-equivalent не потому,
что совпадают их точки. Они получены из одного $S$ через разные объявленные
sampling intensities и noise kernels.

Для sequence refinements $O_{a,n}$ predictor acquisition consistent, если для
каждого допустимого acquisition law $a$

$$
\sup_{g\in\mathcal G}
|q_\theta(O_{a,n},g)-q^*(S,g)|
\xrightarrow[n\to\infty]{P}0,
$$

и limit $q^*(S,\cdot)$ не зависит от $a$.

Finite clouds не обязаны давать строго одинаковые outputs. Требуется
контролируемый finite-sample error и общий limit. Exact invariance к двум
независимым noisy samples была бы невозможным и ненужным требованием.

Разные viewpoints не считаются эквивалентными: новый viewpoint меняет visible
surface и несёт новую информацию. Primary benchmark меняет acquisition fixed
visible surface, а viewpoint generalization измеряется отдельно.

## 4. Почему hard contact extrema — неверный target

Пусть $u_g$ — closing direction в world frame. Наивный support cue

$$
\widehat h_{\max}(u_g)=\max_i u_g^T x_i.
$$

Если проекция measurement error имеет fixed variance

$$
u_g^T\varepsilon_i\sim\mathcal N(0,\sigma^2),
$$

то для независимых errors

$$
\frac{\max_{i\le n}u_g^T\varepsilon_i}
{\sigma\sqrt{2\log n}}
\xrightarrow[n\to\infty]{P}1.
$$

Даже при bounded true surface observed maximum расходится. На практических $n$
это проявляется как sample-count-dependent positive bias. Robust quantile
уменьшает проблему, но меняет target в зависимости от local sampling density.

Более фундаментально, exact support recovery при fixed Gaussian error является
severely ill-posed deconvolution problem с очень медленными rates. Поэтому paper
не должен обещать восстановление идеальных point contacts или infinitesimal
surface support.

Вместо этого вводится **finite contact scale** $\tau>0$. Для smooth local gate
$\psi_{g,k}$ и direction $u_{g,k}$ зададим entropic support

$$
H_{k,\tau}(S,g)
=\tau\log
\frac{
\int_S\psi_{g,k}(s)
\exp(u_{g,k}^Ts/\tau)\,d\mu_S(s)
}{
\int_S\psi_{g,k}(s)\,d\mu_S(s)
}.
$$

При $\tau\downarrow0$ functional приближается к local support при обычных
regularity assumptions. При finite $\tau$ он:

- игнорирует sub-resolution spikes;
- отражает конечный pad/contact neighborhood;
- допускает statistical estimation;
- плавно меняется при малом pose noise;
- может быть вычислен для нескольких scales.

Используются

$$
\tau_1>\tau_2>\cdots>\tau_L\ge\tau_{\min}(\sigma,n_{\mathrm{eff}}).
$$

Scale не должен становиться свободным способом замаскировать noise. Нижняя
граница выводится из variance amplification deconvolution и проверяется по
measured pad/noise scales.

## 5. Deconvolved quadrature contact layer

### 5.1 Contact probes

Пусть $\kappa_k:\mathbb R^3\to\mathbb R$ — smooth kernel в gripper frame.
Для grasp $g$ соответствующий surface functional

$$
Z_k(S,g)
=\int_S\kappa_k(g^{-1}s)\,d\mu_S(s).
$$

Небольшой basis включает:

- soft mass внутри left и right pad neighborhoods;
- exponentials вдоль $+u_g$ и $-u_g$;
- smooth probes около fingertip и pad boundaries;
- bilateral gap features;
- target и obstacle probes с раздельными marks;
- несколько physically resolvable spatial scales.

Это не полный hand-crafted grasp metric. Kernels дают compact acquisition-stable
geometric evidence, а связь с success учит query head.

### 5.2 Gaussian measurement correction

Convolution функции с Gaussian error соответствует heat semigroup. Для smooth
resolvable kernel введём inverse correction

$$
\mathcal D_\Sigma
=\exp\left(-\frac12\Sigma:\nabla^2\right).
$$

Тогда acquisition-corrected estimator

$$
\widehat Z_k(O_a,g)
=\sum_{i=1}^{N_a}\omega_i
\left[
\mathcal D_{R_g^T\Sigma_iR_g}\kappa_k
\right](g^{-1}x_i).
$$

Для exponential probe

$$
\kappa(x)=\exp(tu^Tx)
$$

correction имеет closed form

$$
[\mathcal D_\Sigma\kappa](x)
=\exp\left(
tu^Tx-\frac12t^2u^T\Sigma u
\right).
$$

Именно эта формула устраняет Gaussian moment inflation. Для Gaussian-windowed
exponentials также возможна analytic correction, если window bandwidth шире
measurement blur. Если это условие нарушено, inverse heat flow усиливает noise и
kernel должен быть запрещён.

В implementation basis parametrized так, чтобы covariance каждого spatial
window удовлетворяла resolvability constraint

$$
B_k-c\Sigma_i\succeq0
$$

для заранее выбранного safety factor $c>1$ в соответствующем local region.
Можно использовать softplus parametrization eigenvalues выше noise floor.

### 5.3 Self-normalized functionals

Если абсолютная normalization reference measure неизвестна, используется
ratio

$$
\widehat{\bar Z}_k
=\frac{\widehat Z_k}{\widehat Z_0+\epsilon},
$$

где $\kappa_0$ — local mass kernel. Ratio не является exactly unbiased, но
consistent при стандартных assumptions. В theory отдельно доказываются
unbiased unnormalized estimator и finite-sample bias ratio estimator.

### 5.4 Non-Gaussian и correlated noise

Gaussian model — первый tractable case, а не описание всех depth artifacts.
Paper обязан разделить:

1. calibrated axial Gaussian core;
2. bounded covariance misspecification;
3. sparse outliers;
4. correlated edge/multipath artifacts.

Для sparse outliers можно добавить bounded-influence clipping только после
analytical moment correction. Для correlated noise effective sample size
оценивается по ray blocks или repeated-frame calibration. Нельзя выдавать
номинальное число points за число независимых measurements.

Если простой calibrated Gaussian core не переносится на real camera даже после
этих corrections, основной method claim должен быть снят.

## 6. Efficiently learnable formalization: AcqGrasp

### 6.1 Вход и representation

Input tensor содержит только retained local samples

$$
(x_i,\omega_i,\Sigma_i,r_i).
$$

Light RGB-D encoder возвращает per-point appearance/semantic feature $f_i$.
Точные geometric kernels вычисляются непосредственно по coordinates и declared
marks; encoder не может незаметно заменить их произвольным black-box feature.

Spatial hash, radius graph или ball query один раз строит neighborhoods каждого
candidate. Для кандидата извлекаются только точки, пересекающие terminal open
jaw, closing slab и pad neighborhoods. Approach corridor не оценивается.

### 6.2 Compact contact sketch

Для каждого $g_m$ слой возвращает

$$
z_m=
[
\widehat Z_{m,1},\ldots,\widehat Z_{m,K},
\widehat H_{m,1},\ldots,\widehat H_{m,L},
\bar f_m,
w_m,p_{\mathrm{pad}}
].
$$

$\bar f_m$ — density-corrected pooled RGB/semantic feature, а
$p_{\mathrm{pad}}$ содержит только короткий список известных gripper parameters:
pad width, height и commanded force class. Полная robot state или длинный набор
scene variables не подаётся.

Shared query head

$$
q_\theta(O,g_m)=\sigma(\rho_\theta(z_m))
$$

возвращает terminal success probability. Optional residual head выдаёт малую
pose correction $\Delta g_m$ в tangent coordinates, но main experiment должен
работать и без refinement.

### 6.3 Training objective

Для supervised label $y_{b,m}\in\{0,1\}$

$$
\mathcal L_{\mathrm{sup}}
=-
\sum_{b,m}
\left[
y_{b,m}\log q_{b,m}
+(1-y_{b,m})\log(1-q_{b,m})
\right].
$$

Для одной latent scene $b$ доступны acquisitions
$a\in\mathcal A_b$ и общий candidate set. Paired consistency loss

$$
\mathcal L_{\mathrm{acq}}
=\frac{1}{|\mathcal P_b|M}
\sum_{(a,a')\in\mathcal P_b}
\sum_{m=1}^{M}
\left(q_{b,a,m}-q_{b,a',m}\right)^2.
$$

Decision-sensitive pairwise ranking loss

$$
\mathcal L_{\mathrm{rank}}
=\sum_{(m,j):y_m>y_j}
\log\left(1+\exp[-\beta(q_m-q_j)]\right).
$$

Итог:

$$
\mathcal L
=\mathcal L_{\mathrm{sup}}
+\lambda_a\mathcal L_{\mathrm{acq}}
+\lambda_r\mathcal L_{\mathrm{rank}}
+\lambda_b\mathcal L_{\mathrm{bandwidth}}.
$$

Paired consistency loss не является novelty. Он нужен для finite-sample
residuals после analytical correction. Destructive baseline получает тот же
loss и те же paired data без proposed layer.

### 6.4 Inference

Выбранный grasp

$$
\hat g=\arg\max_{g_m\in G_o}q_\theta(O,g_m).
$$

Если требуется conservative deployment, можно использовать calibrated lower
score $q_m-c_m$, но uncertainty/calibration не заявляются основной новизной.

При $M$ candidates, $k$ local points, $K$ kernels и $L$ scales query cost

$$
O(MkKL).
$$

Первый practical configuration:

- $M=256$--512;
- $k=48$--96;
- $K=8$--16;
- $L=3$--4.

Все probes реализуются segmented reductions. Нет dense volume, iterative shape
completion, diffusion sampling или simulator rollouts на inference.

## 7. Обязательное теоретическое ядро

Следующие результаты являются theorem targets. До доказательства их нельзя
писать в paper как установленные факты.

### 7.1 Acquisition unbiasedness

При marked Poisson model, exact $\lambda_a$, known Gaussian covariance и
admissible kernel доказать

$$
\mathbb E[
\widehat Z_k(O_a,g)
\mid S]
=Z_k(S,g)
$$

для любого $a$ в объявленном acquisition family.

Этот result сам по себе элементарен и недостаточен для ICLR. Его роль — сделать
точным base layer, на котором строятся более сильные statements.

### 7.2 Uniform consistency over action queries

Для compact $\mathcal G$, bounded kernel envelope и bounded importance weights
получить bound вида

$$
\Pr\left{
\sup_{g\in\mathcal G}
|\widehat Z_k(S,g)-Z_k(S,g)|
>
C B w_{\max}
\sqrt{
\frac{d_G\log n+\log(K/\delta)}{n_{\mathrm{eff}}}
}
\right}
\le\delta.
$$

Exact constants и entropy term должны следовать из выбранного kernel/action
class. Важно, чтобы bound зависел от effective sample size и weight degeneracy,
а не только от nominal $N$.

### 7.3 No-go result для raw extrema

Формализовать sample-count drift hard max при unbounded additive noise и
показать, что permutation-invariant max pooling не является acquisition
consistent estimator физического support.

Result должен включать finite-$n$ consequence в миллиметрах для measured camera
noise, иначе asymptotic theorem останется нерелевантным robotics experiment.

### 7.4 Bias--variance law contact scale

Для entropic support получить decomposition

$$
|\widehat H_{\tau}-h|
\le
\underbrace{|H_\tau-h|}_{\text{soft-support bias}}
+
\underbrace{|\widehat H_\tau-H_\tau|}_{\text{sampling/deconvolution error}}.
$$

При Gaussian moment correction variance растёт примерно экспоненциально с
$\sigma^2/\tau^2$. Это должно дать nonzero resolvable scale

$$
\tau_{\min}=F(\sigma,n_{\mathrm{eff}},\delta,\text{local mass}).
$$

Новый scientific insight состоит именно в невозможности одновременно иметь
infinitesimal contact resolution и stable finite-sample decision при fixed
measurement noise.

### 7.5 Sketch-to-decision transfer

Если query head $L_q$-Lipschitz по contact sketch и

$$
\sup_{g\in\mathcal G}
\|\widehat z(O_a,g)-z(S,g)\|
\le\epsilon_z,
$$

то

$$
\sup_g
|\widehat q(O_a,g)-q^*(S,g)|
\le L_q\epsilon_z+\epsilon_{\mathrm{model}}.
$$

Для

$$
g^*=\arg\max_g q^*(S,g),
\qquad
\hat g=\arg\max_g\widehat q(O_a,g),
$$

стандартный decision bound

$$
q^*(S,g^*)-q^*(S,\hat g)
\le
2(L_q\epsilon_z+\epsilon_{\mathrm{model}}).
$$

При дополнительном quadratic margin можно получить distance-to-optimal-set
bound. Именно этот result связывает acquisition correction с grasp selection,
а не только с reconstruction error.

### 7.6 Misspecified acquisition law

Если используются $\widehat\omega_i$ и $\widehat\Sigma_i$, доказать perturbation
bound через

$$
\max_i|\widehat\omega_i-\omega_i|
\quad\text{и}\quad
\max_i\|\widehat\Sigma_i-\Sigma_i\|.
$$

Без такого result ideal sensor model мало связан с real deployment.

## 8. Paired-Acquisition Grasp Benchmark

### 8.1 Основной принцип

Benchmark unit — не отдельный point cloud, а latent scene с набором acquisitions

$$
\mathcal B_b=(S_b,G_b,y_b,\{O_{b,a}\}_{a\in\mathcal A_b}).
$$

Object, pose, visible surface, candidate set и physics label фиксированы. Меняется
только acquisition law.

Physics label нельзя пересчитывать отдельно после каждого sensor rendering.
Он один раз определяется по latent mesh и standardized close-and-lift simulator
или по physical repeated trials. Иначе sensor sensitivity смешивается с label
noise.

### 8.2 Synthetic acquisition matrix

Для каждой сцены генерируются:

- 4 native image resolutions или controlled pixel binnings;
- point budgets, отличающиеся минимум на порядок;
- uniform random thinning;
- FPS;
- voxel downsampling;
- deliberately non-uniform local thinning;
- 3--4 heteroscedastic axial noise levels;
- edge-dependent noise;
- small outlier fractions;
- missing-depth masks;
- correct и perturbed covariance/weight marks.

Scene families:

- isolated object на shelf;
- тот же object с одним frontal occluder;
- visibility strata от low до high occlusion;
- smooth, edged, thin и locally concave shapes;
- material-independent synthetic depth для clean mechanism study;
- отдельный non-Gaussian sensor stress split.

Viewpoint держится фиксированным внутри acquisition group. Другие viewpoints
образуют новые groups.

### 8.3 Real paired data

Нужны два разных real tests.

**Pure discretization test.** Один native RGB-D frame повторно обрабатывается
разными FPS, voxel, crop и point budgets. Он изолирует software acquisition
pipeline.

**Measurement test.** Камера и object фиксируются jig-ом, записываются repeated
frames. Из них строятся independent clouds с measured temporal noise. Edge и
multipath regions помечаются отдельно.

Для physical grasp trials object возвращается в jig с контролируемой pose.
Сравниваются candidates, выбранные разными acquisitions одной сцены. Каждый
candidate исполняется несколько раз с randomized measured pose perturbations.

### 8.4 Metrics

- paired score deviation;
- top-1 agreement;
- top-$k$ overlap;
- Kendall или Spearman ranking stability;
- label regret относительно fixed candidate oracle;
- AP/AUC и calibration;
- expected terminal small-lift success;
- error против $n_{\mathrm{eff}}$;
- sensitivity к weight/covariance misspecification;
- latency и memory;
- clean-condition degradation;
- результаты по visibility, geometry и obstacle strata.

Score consistency не должна быть единственным headline metric. Constant model
идеально consistent и бесполезна. Главные metrics — latent-label regret и real
selected-grasp success.

## 9. Обязательный дешёвый falsification pilot

До обучения нового backbone:

- 20--30 CAD objects;
- 5--10 fixed shelf poses на object;
- версии без obstacle и с одним frontal obstacle;
- 256 shared candidates на scene;
- минимум 12 acquisitions на scene;
- четыре point budgets;
- три realistic noise levels;
- один clean latent-physics label vector.

Реализовать только четыре evaluators:

1. сильный существующий point-cloud grasp scorer;
2. тот же scorer с paired acquisition augmentation;
3. PointConv или Monte Carlo density-corrected version;
4. fixed deconvolved contact sketch плюс маленький MLP.

Проект продолжать только при одновременном выполнении gates:

1. Conditional regression baseline score на $\log N$ после учёта scene и grasp
   показывает material acquisition effect.
2. Между realistic acquisition extremes имеется не меньше 5 percentage points
   latent-label regret gap **или** не меньше 20 points потери top-10 overlap.
3. Proposed fixed sketch уменьшает paired consistency error минимум вдвое.
4. Он улучшает regret на unseen acquisition law, а не только на train rendering
   menu.
5. Clean/native accuracy ухудшается не больше чем на 2 points.
6. Gain остаётся при 10--30% ошибке в $\Sigma$ и weights.
7. Density-corrected PointConv с тем же paired data не достигает того же
   результата.
8. На repeated real frames ranking drift существует и уменьшается proposed
   correction.

Failure пунктов 1–2 означает, что paper выдумывает практически несущественную
проблему. Failure пунктов 3–7 убивает method novelty. Failure пункта 8 убивает
real-camera claim.

Самый важный destructive outcome:

> Fixed-budget FPS плюс обычная strong noise augmentation уже делает grasp
> ranking стабильным.

При таком результате нельзя спасать идею большей сетью.

## 10. Полная экспериментальная программа

### 10.1 Splits

- unseen object geometry;
- unseen object categories;
- unseen acquisition resolution;
- unseen point-budget range;
- unseen thinning process;
- unseen noise variance;
- unseen non-Gaussian artifact mixture;
- unseen RGB-D device;
- unseen obstacle height/width;
- synthetic-to-real;
- calibrated и misspecified sensor marks.

Главный split должен держать acquisition law целиком вне train. Random split
finite clouds из одного rendering menu недостаточен.

### 10.2 Baselines при одинаковых data и compute

1. PointNet++ candidate evaluator.
2. Sparse point/voxel grasp evaluator.
3. PointConv.
4. Monte Carlo Convolution.
5. Quadrature-aware GINO/PCNO-style neural operator.
6. Original scorer плюс point-budget/noise augmentation.
7. Paired Siamese consistency training.
8. Score-based или supervised point-cloud denoising перед scorer.
9. R2S-style depth repair.
10. Raw max/min contact features.
11. Robust quantile contact features.
12. Unweighted log-sum-exp support.
13. Density-only correction.
14. Noise-only correction.
15. Full AcqGrasp.
16. Clean-mesh contact-feature oracle.
17. Latent physics candidate oracle.

Все learned baselines получают один encoder capacity, candidate set, labels и
paired acquisitions. Иначе improvement можно объяснить большим backbone или
дополнительными renders.

### 10.3 Ablations

- без $\omega_i$;
- без $\Sigma_i$ correction;
- без paired loss;
- hard max против entropic support;
- один $\tau$ против multiscale;
- fixed против learned bandwidth выше noise floor;
- surface-area против ray-domain reference measure;
- exact против 10%, 20%, 30%, 50% misspecified covariance;
- Gaussian-only против outlier-robust extension;
- target-only против target-plus-obstacle marks;
- 4, 8, 16, 32 contact probes;
- local points 32, 64, 128;
- fixed candidates против end-to-end generator;
- одинаковый frame resampling против independent repeated frames.

### 10.4 Real hardware protocol

1. Оценить axial depth error по distance, incidence angle и edge proximity.
2. Измерить temporal correlation repeated frames.
3. Сохранить quadrature metadata через preprocessing.
4. Зафиксировать object pose jig-ом.
5. Получить paired acquisitions до физического контакта.
6. Выбрать balanced set stable и acquisition-sensitive candidates.
7. Исполнить standardized close-and-small-lift несколько раз на candidate.
8. Randomize только measured execution perturbation.
9. Report selection success, а не только score agreement.
10. Повторить с одним frontal obstacle и без него.

## 11. Novelty boundary после adversarial literature audit

| Направление | Что уже существует | Чего оно не даёт |
|---|---|---|
| Permutation-invariant point networks | Invariance к order input points | Нет invariance или consistency к sampling intensity и measurement law |
| PointConv / Monte Carlo Convolution | Inverse-density quadrature для continuous convolution | Нет noisy near-extreme contact target, action-query theory и paired decision benchmark |
| Neural operators | Discretization-consistent function-space maps | Обычно предполагают доступные point evaluations/quadrature; не решают noisy inverse observation для contact extrema |
| Point-cloud denoising | Восстановление clean coordinates | Строит общий repaired cloud и не гарантирует acquisition-stable action decision |
| R2SGrasp | Learned real-to-sim depth/feature repair | Нет declared acquisition quotient или decision consistency theorem |
| Certified point-cloud robustness | Worst-case certificates вокруг finite cloud | Не consistency stochastic acquisitions одного latent surface |
| Noisy support estimation | Statistical recovery convex support under additive noise | Восстанавливает hard global set с медленными rates, а не несколько regularized action-conditioned functionals |
| Acquisition-invariant MRI representations | Scanner/protocol-invariant learned features | Термин уже занят; нет point-process contact geometry или grasp decision transfer |
| AcqGrasp | Marked sensor law, finite contact scale, deconvolved quadrature, paired fixed-label benchmark | Новый stack, который ещё должен пройти pilot |

Поэтому **не являются novelty по отдельности**:

- inverse-density weighting;
- Horvitz--Thompson estimation;
- Gaussian moment correction;
- inverse heat notation;
- log-sum-exp;
- paired consistency loss;
- neural operator;
- grasp MLP.

Защищаемая contribution stack:

1. formal learning problem на acquisition-law equivalence classes;
2. contact-specific obstruction для hard noisy extrema;
3. finite-scale action-conditioned target с resolvability law;
4. bandwidth-constrained deconvolved quadrature layer;
5. uniform acquisition-to-decision theory;
6. paired-acquisition grasp benchmark с fixed physics labels;
7. real evidence на ranking и small-lift selection.

Если paper не содержит весь stack, reviewer может справедливо описать его как
композицию известных estimators.

## 12. Mock review по критериям ICLR

### Вероятные strengths

- Problem statement отделяет permutation invariance от sensor-law consistency.
- Есть конкретный paradox: больше noisy points может ухудшать hard contact cue.
- Mathematical obstruction определяет finite-scale model design.
- Model локальный, быстрый и не требует hidden shape reconstruction.
- Paired benchmark меняет acquisition при fixed physics target.
- Theory и experiments измеряют один и тот же decision-level claim.
- Идея потенциально переносится на insertion, collision и sampled-geometry
  decisions.

### Вероятные причины weak reject

1. «Это PointConv плюс Gaussian correction и grasp head».
2. Современные fixed-budget pipelines уже достаточно стабильны.
3. RealSense noise не Gaussian, correlated и material dependent.
4. Выбор reference measure произволен.
5. Entropic support не равен реальному contact.
6. Unbiasedness theorem слишком elementary.
7. Paired consistency можно получить augmentation без physics layer.
8. Improvement существует только в synthetic extreme resolutions.
9. Один grasp benchmark недостаточен для broad ICLR claim.
10. Candidate generator, а не evaluator, является реальным источником drift.

Experimental program выше отвечает на эти objections следующим образом:

- PointConv и augmentation — обязательные destructive baselines;
- pilot измеряет phenomenon до новой сети;
- covariance misspecification и non-Gaussian splits обязательны;
- reference measure ablated;
- $\tau$ связан с noise и pad scale;
- headline theorem — uniform scale/decision law, не unbiasedness;
- common candidate set изолирует evaluator;
- real repeated-frame and hardware tests обязательны;
- дополнительный sampled-geometry task желателен для broad claim.

### Честная acceptance estimate

- **Сейчас, без pilot:** weak reject. Формализация новая и логична, но magnitude
  явления неизвестен.
- **После positive synthetic pilot:** borderline; reviewer всё ещё может считать
  effect artificial.
- **После unseen-acquisition transfer и repeated real frames:** borderline to
  weak accept при сильной theory.
- **После real small-lift improvement и второго geometric-decision task:**
  правдоподобный ICLR accept-level paper.
- **Если PointConv + augmentation равен AcqGrasp:** method paper следует закрыть.

Нельзя честно использовать формулировку «objectively strong novelty» до
успешного destructive pilot. На текущем этапе объективно сильны постановка и
falsifiability; empirical significance ещё не установлена.

## 13. Ограничения, которые нужно объявить заранее

1. Acquisition equivalence относится к одной declared visible surface; другой
   viewpoint не эквивалентен.
2. Reference measure является частью task definition.
3. Gaussian correction покрывает только calibrated core sensor noise.
4. Correlated edge/multipath artifacts снижают effective sample size.
5. Inverse heat correction стабильна только выше noise-dependent bandwidth.
6. Finite-scale support — regularized evidence, а не exact contact point.
7. Model не восстанавливает hidden rear geometry.
8. Occlusion остаётся epistemic limitation; proposal исправляет acquisition
   drift, а не делает unseen surface observable.
9. Guarantee относится к candidate set, а не ко всему непрерывному $SE(3)$ без
   coverage assumptions.
10. Motion planning, global collisions и long lift остаются внешними.
11. Семантическая segmentation error не является главным target.
12. Другой gripper или soft pad требует новой kernel family и scale calibration.

## 14. Порядок исполнения проекта

1. Зафиксировать terminal grasp protocol и candidate generator.
2. Откалибровать camera axial covariance и temporal correlations.
3. Сохранить sampling/quadrature metadata через point-cloud preprocessing.
4. Создать paired synthetic acquisition generator с fixed physics labels.
5. Измерить baseline decision drift без proposed model.
6. Выполнить go/no-go gates раздела 9.
7. Реализовать fixed deconvolved sketch без нового backbone.
8. Сравнить с PointConv, augmentation, quantiles и denoising.
9. После положительного результата доказать uniform и scale/decision bounds.
10. Добавить learnable kernels только поверх validated fixed basis.
11. Собрать repeated-frame real benchmark.
12. Провести balanced hardware small-lift study.
13. Добавить второй cheap sampled-geometry decision task.
14. Повторить novelty search перед submission.
15. Строить paper вокруг learning problem и empirical law, а не вокруг названия
    слоя.

## 15. Минимальный defensible paper

Минимальная работа должна внести все четыре contributions:

1. определить acquisition-law-equivalent geometric decision learning и
   доказать obstruction для hard noisy contact features;
2. предложить resolvability-constrained deconvolved quadrature contact layer с
   uniform и decision guarantees;
3. выпустить paired-acquisition parallel-jaw benchmark, где physics labels и
   candidates фиксированы при изменении sensor law;
4. показать на реальном robot, что acquisition-stable ranking улучшает
   repeated small-lift selection при point-budget/noise shifts.

Target abstract:

> Point-cloud grasp predictors are invariant to point order but need not be
> invariant to how a physical surface was acquired. This distinction is acute
> for contact decisions: under fixed depth noise, naive point extrema become
> sample-count dependent. We formulate acquisition-equivalent geometric
> decision learning, introduce a deconvolved quadrature layer for finite-scale
> action-conditioned contact functionals, and evaluate it on paired RGB-D
> acquisitions with fixed physical grasp labels. The resulting model preserves
> grasp rankings across unseen point budgets and noise laws and improves
> small-lift selection without reconstructing a scene field.

Последнее предложение является target claim, а не текущим результатом.

## 16. Основные источники

- PointConv:
  https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_PointConv_Deep_Convolutional_Networks_on_3D_Point_Clouds_CVPR_2019_paper.pdf
- Monte Carlo Convolution:
  https://arxiv.org/abs/1806.01759
- GINO:
  https://arxiv.org/abs/2309.00583
- Обзор discretization-consistent neural operators:
  https://www.nature.com/articles/s42256-026-01267-z
- Estimation of convex supports from noisy measurements:
  https://arxiv.org/abs/1804.09879
- PointGuard:
  https://arxiv.org/abs/2103.03046
- 3DeformRS / 3-D certification context:
  https://arxiv.org/abs/2103.16652
- R2SGrasp:
  https://isee-laboratory.github.io/R2SGrasp/
- Generalizing 6-DoF Grasp Detection via Domain Prior Knowledge:
  https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Generalizing_6-DoF_Grasp_Detection_via_Domain_Prior_Knowledge_CVPR_2024_paper.pdf
- TARGO:
  https://targo-benchmark.github.io/
- Acquisition-invariant MRI representation learning:
  https://arxiv.org/abs/1810.07430
- Официальные критерии ICLR 2027:
  https://iclr.cc/Conferences/2027/ReviewerGuidelines

Полный журнал search cycles и rejected directions находится в
`reports/EdgeFlux.md`.
