# Do not complete what you cannot see: Grasp-Certificate Processes под foreground-окклюзией

Дата исследования: 2026-08-25  
Статус: новая самостоятельная идея; это не продолжение и не пересказ прежних файлов репозитория  
Целевая площадка: ICLR 2027  
Ограничения: single noisy RGB-D, parallel-jaw gripper, один foreground-окклюдер, без RL/VLA, без полной реконструкции объекта и без оценки всего manipulation cycle

## Краткий итог

Предлагается сменить сам предсказываемый объект. Вместо одной наиболее вероятной скрытой формы, occupancy/SDF либо одного усреднённого grasp score модель должна учить **условное распределение над целой функцией физического grasp-сертификата**:

$$
\Pi_y
{}={}
\mathrm{Law}\!\left(C_X(\cdot)\mid Y=y\right),
\qquad
C_X:\mathcal G\to\mathbb R,
$$

где $X$ — неизвестная полная форма target-а, $Y$ — одно частично окклюдированное RGB-D наблюдение, $g\in\mathcal G\subset SE(3)\times\mathbb R_+$ — parallel-jaw grasp, а $C_X(g)$ — signed local certificate: положительное значение означает одновременно допустимую ширину, отсутствие запрещённого пересечения gripper-а и робастный antipodal/force-closure margin; отрицательное — величину нарушения. 

Это **push-forward posterior**: распределение полной формы $P(X\mid Y)$ проталкивается отображением $X\mapsto C_X(\cdot)$, но сама форма не восстанавливается и не декодируется. Один sample модели — не mesh, а согласованный grasp-quality landscape по всем query grasps. Поэтому две скрытые формы, различающиеся в нерелевантных для grasp областях, считаются эквивалентными; формы, одинаковые в видимой части, но дающие разные контакты за окклюдером, должны порождать несколько modes процесса.

Для этого предлагаются:

1. **Новый learning objective:** random-design, tail-sensitive Energy–Variogram score над векторами сертификатов сразу для набора grasp queries. В отличие от BCE по каждому grasp он учит условный совместный закон и не поощряет collapse к среднему; в отличие от likelihood не требует явной плотности.
2. **Новая architecture — Ray–Jaw Process Network (RJPN):** camera-ray tokens, sparse incidence attention между лучами RGB-D и closing/swept rays конкретного gripper-а, один общий conditional latent на скрытую форму и query-wise decoder. Shared latent делает ответы для разных grasпов согласованными и projective by construction, но не создаёт SDF/voxel grid/mesh.
3. **Решение:** выбрать grasp с максимальным нижним posterior-quantile $q_\alpha(C_X(g)\mid y)$, а при отсутствии положительного квантили вернуть `no certified grasp`. На отдельном calibration split можно контролировать false-safe risk conformal risk control.

Наиболее сильная paper claim звучит так:

> При частичной наблюдаемости не надо решать более трудную inverse problem $Y\to X$, если downstream loss зависит от $X$ только через action-indexed certificate $C_X(\cdot)$. Надо непосредственно учить posterior push-forward в пространстве decision functions и оптимизировать собственный proper score для его finite-dimensional marginals.

Это существенно шире grasping и годится для любой задачи «частичное наблюдение → выбрать query/action → физический scalar certificate», что и создаёт ICLR, а не только robotics, позиционирование.

## 1. Что именно является научным gap

### 1.1. Три существующие линии

Первая линия — **direct grasp prediction по видимым данным**. GPD уже формулировал grasp detection непосредственно по noisy, partially occluded RGB-D/point cloud без известного CAD model. S4G делает single-view amodal proposals и скаляризует робастность к малым pose perturbations. GraspNet/GSNet, Contact-GraspNet, AnyGrasp и другие современные детекторы также в основном выдают poses и point scores. Это сильные методы, но скрытая shape ambiguity обычно поглощается одним discriminative score; модель не обязана выдавать согласованное распределение того, как один и тот же неизвестный hidden shape одновременно меняет качества нескольких grasp candidates. [GPD](https://arxiv.org/abs/1706.09911), [S4G](https://proceedings.mlr.press/v100/qin20a.html), [Graspness/GSNet](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.html).

Вторая линия — **completion first / completion jointly with grasping**. Классический Shape Completion Enabled Robotic Grasping восстанавливает скрытые regions, затем планирует grasps. NeuGraspNet строит implicit occupancy, глобально ray-march-ит реконструированную поверхность и локально рендерит geometry для score; его собственный runtime table сообщает 865 ms total против 193–280 ms у ряда discriminative/generative baselines. ZeroGrasp использует octree-CVAE, совместно реконструирует shape и grasp poses и вводит 3D occlusion fields. CenterGrasp также совместно учит shape reconstruction и grasp estimation. Уже в 2026 году TOSC сузил completion до task-relevant contact regions, но всё ещё явно генерирует и выбирает completed geometry. [Shape Completion Enabled Grasping](https://arxiv.org/abs/1609.08546), [NeuGraspNet](https://www.roboticsproceedings.org/rss20/p046.pdf), [ZeroGrasp](https://openaccess.thecvf.com/content/CVPR2025/papers/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.pdf), [CenterGrasp](https://centergrasp.cs.uni-freiburg.de/), [TOSC](https://ojs.aaai.org/index.php/AAAI/article/download/38053/42015).

Третья линия — **uncertainty-aware grasp mechanics**. Dex-Net 2.0 уже принципиально важен: он не обязан восстанавливать форму, а учит probability of grasp success из synthetic depth, интегрируя uncertainties своего graphical model; следовательно, утверждать «никто не учил решение напрямую» нельзя. PSSNet генерирует несколько plausible full-shape completions. SpringGrasp использует GP implicit surface uncertainty для compliant dexterous pregrasp optimization. FIRMGrasp строит CVaR-based certificate по uncertainty коэффициента трения при заданных contacts/object model. Однако это соответственно scalar Bernoulli quality, posterior над полной формой, dexterous compliant planning или uncertainty физического параметра при известной контактной геометрии — не posterior над action-indexed certificate function из severe foreground-occluded observation. [Dex-Net 2.0](https://arxiv.org/abs/1703.09312), [PSSNet](https://proceedings.mlr.press/v155/saund21a.html), [SpringGrasp](https://www.roboticsproceedings.org/rss20/p042.html), [FIRMGrasp](https://arxiv.org/abs/2607.25049).

### 1.2. Сохраняющийся разрыв

В рассмотренной литературе не найден метод, одновременно удовлетворяющий следующим условиям:

- одно noisy RGB-D наблюдение;
- foreground obstacle создаёт явно известную visibility censoring geometry;
- hidden target geometry берётся только из training shape distribution;
- сеть не выдаёт ни complete shape, ни hidden occupancy/SDF;
- предсказывается не независимый scalar mean/probability для каждого grasp, а условный **stochastic process** физических margins по grasp space;
- learning objective является proper multivariate score по случайным finite query sets и тем самым учит не только marginals, но и dependence/ranking structure;
- inference query-local и использует геометрию взаимного расположения camera rays и jaw rays.

Это более узкий и проверяемый novelty claim, чем «первый uncertainty-aware grasp method». Абсолютно гарантировать novelty по web search невозможно; перед submission всё равно требуется отдельный systematic Scholar/Semantic Scholar/OpenReview sweep и citation chasing. Но по состоянию на 2026-08-25 ближайшие работы расходятся с proposed object, objective или architecture как минимум по двум главным осям.

## 2. Почему несколько более очевидных идей были отброшены

### 2.1. Полная probabilistic shape completion

Отброшено как центральная идея. PSSNet, ZeroGrasp, NeuGraspNet и CenterGrasp уже занимают пространство deterministic/probabilistic/joint implicit completion. Добавить новый diffusion/flow decoder скрытой формы — архитектурная замена, а не новый learning problem. Кроме того, reconstruction loss расходует capacity на backside regions, не влияющие ни на contact, ни на clearance.

### 2.2. Completion только contact regions

До 2026 года это могло выглядеть сильным gap, но TOSC уже явно формулирует task-oriented completion потенциальных contact regions. Перенос с dexterous на parallel-jaw и отказ от language не дают достаточной ICLR novelty. Наша идея делает ещё один шаг: **не реконструировать даже contact surface**, а предсказывать только push-forward её физического эффекта.

### 2.3. Один Bayesian/BCE grasp-success score

Отброшено как недостаточно новое: Dex-Net 2.0 уже учит probability of robust success из synthetic point clouds, а большая часть discriminative grasp literature аппроксимирует $P(S=1\mid y,g)$. Ensemble, MC-dropout или heteroscedastic head не меняют статистический объект.

Кроме novelty есть техническая проблема: binary label уничтожает расстояние до failure boundary. Grasp с margin (+0.001) и (+1.0) имеет одинаковую метку; модель хуже переносит изменение gripper width, calibration noise и выбранного risk level. Signed certificate сохраняет эту информацию.

### 2.4. Независимая quantile regression для каждого grasp

Это сильный baseline, но не финальная идея. Она может правильно учить marginal $q_\alpha(C(g)\mid y)$, однако sampled answers для соседних $g$ не обязаны соответствовать хоть одной общей hidden-shape hypothesis. Получается «Frankenstein posterior»: левый grasp оценивается как при одной скрытой форме, правый — как при другой. Для forced one-shot argmax marginal theoretically sufficient; поэтому преимущество совместного процесса является эмпирической гипотезой, а не аксиомой. Оно должно проявиться в sample efficiency, smooth risk landscape, calibration после selection и ranking under ambiguous fibers — иначе process contribution надо честно снять.

### 2.5. Worst case по всем shapes, согласованным с visible points

Отброшено. Без сильного distributional restriction почти для каждого hidden-contact grasp найдётся completion, делающий его плохим; lower envelope становится тривиальным. Явная оптимизация Wasserstein/likelihood ball вокруг shape posterior снова требует моделировать огромный object space. Lower quantile learned push-forward сохраняет risk sensitivity, но остаётся non-vacuous благодаря training distribution.

### 2.6. Martingale consistency по progressively revealed views

Идея математически красива: $\mathbb E[C\mid\mathcal F_k]$ должен быть martingale по вложенным information sets. Но в мае 2026 появился общий preprint **Martingale-Consistent Self-Supervised Learning**, который прямо вводит coarse/refined-view consistency и two-sample stochastic refinement. Поэтому tower-property regularizer можно оставить только как optional ablation, но нельзя строить на нём core novelty. [Martingale-Consistent SSL](https://www.alphaxiv.org/abs/2605.11846).

### 2.7. Разложение на causal failure modes или весь подход–захват–подъём

Отброшено по постановке. Proposed certificate относится только к локальной grasp geometry и малым execution perturbations. Reachability всего humanoid arm, долгий collision-free approach и динамика полного подъёма остаются external filters/evaluation, а не prediction target.

## 3. Новая постановка: posterior над grasp-certificate process

### 3.1. Переменные

Пусть:

- $X\sim P_X$ — полная rigid target geometry из shape distribution;
- $A$ — foreground occluder и shelf geometry;
- $M$ — camera pose/intrinsics и visibility operator;
- $\eta$ — RGB-D noise;
- $Y=\mathcal R_M(X,A)+\eta$ — single RGB-D observation с target/occluder masks;
- $g=(R,t,w)\in\mathcal G\subset SE(3)\times[0,w_{max}]$ — candidate parallel-jaw grasp;
- $\delta\in\Delta\subset\mathfrak{se}(3)$ — малая bounded pose/calibration perturbation.

Target mask допустимо считать данным от upstream segmenter. Это изолирует научный вопрос hidden grasp geometry. RGB features можно добавить, но основная постановка должна работать по depth + visibility types, чтобы результат нельзя было объяснить category recognition.

### 3.2. Не binary success, а signed local certificate

На полном training mesh офлайн вычисляется

$$
C_X(g)=
\min_{\delta\in\Delta}
\min\left\{
\frac{\epsilon_{\mathrm{fc}}(X,g\oplus\delta)}{s_\epsilon},
\frac{d_{\mathrm{body}}(X,g\oplus\delta)}{s_d},
\frac{w_{max}-w_X(g\oplus\delta)}{s_w}
\right\}.
$$

Здесь:

- $\epsilon_{\mathrm{fc}}$ — signed antipodal/Ferrari–Canny-style closure margin для двух contacts;
- $d_{\mathrm{body}}$ — signed clearance non-contact частей fingers/palm от target; observed obstacle collision проверяется отдельным deterministic gate;
- $w_X(g)$ — required closing width;
- $s_\epsilon,s_d,s_w$ — фиксированные physical scales, чтобы минимум был dimensionless;
- bounded minimum по $\delta$ даёт локальную робастность, но не моделирует trajectory или lift.

Условие $C_X(g)>0$ — проверяемый sufficient local certificate. Можно начать с более дешёвого label из ACRONYM/GraspNet simulator и позднее заменить на differentiable exact margin; framework от конкретной формулы не зависит. Критично, что это **один scalar certificate**, а не длинный список causal failure heads.

Полная geometry создаёт функцию

$$
C_X:\mathcal G\to\mathbb R.
$$

Из-за окклюзии $X$ неизвестен, поэтому для одного $y$ эта функция случайна. Целевой объект обучения:

$$
\Pi_y = (X\mapsto C_X(\cdot))_{\mathrm{push}} P(X\mid Y=y),
$$

то есть posterior push-forward на function space.

### 3.3. Decision rule

Для risk level $\alpha\in(0,0.5)$:

$$
g^*(y)
{}={}
\arg\max_{g\in\mathcal C(y)}
Q_\alpha\!\left[C_X(g)\mid Y=y\right],
$$

где $\mathcal C(y)$ — candidate set после exact rejection по observed shelf/occluder points. Если лучший calibrated lower quantile не положителен, система возвращает `no certified grasp`; можно отдельно сообщить лучший forced-choice grasp для benchmark, где abstention запрещён.

При точном conditional law и непрерывном распределении

$$
Q_\alpha[C_X(g)\mid y]>0
\Longrightarrow
P(C_X(g)>0\mid y)\ge 1-\alpha.
$$

У learned model это только model-based statement. Distribution-free conditional guarantee невозможна без дополнительных assumptions; поэтому paper не должен называть raw quantile «гарантией». Реальный false-safe risk калибруется на held-out scenes, а под distribution shift показывается degradation.

### 3.4. Почему push-forward sufficient

Для любого local downstream loss вида

$$
L(g,X)=\widetilde L(g,C_X(g))
$$

conditional Bayes risk равен

$$
\mathbb E[L(g,X)\mid Y=y]
{}={}
\int \widetilde L(g,c(g))\,d\Pi_y(c).
$$

Следовательно, два posteriors над geometry, имеющие одинаковый push-forward $\Pi_y$, decision-equivalent для всей семьи таких losses. Реконструировать различия между ними статистически и вычислительно избыточно.

Это не утверждает, что $\Pi_y$ всегда low-dimensional. Выигрыш возникает из queryability: сеть хранит небольшой latent и вычисляет только $B$ нужных значений $C(g_j)$, вместо dense field по всей сцене. Это честное различие между функциональным математическим объектом и огромной materialized SDF variable.

## 4. Новый objective: Random-Design Tail Energy–Variogram Score

### 4.1. Почему loss должен видеть несколько grasps одной формы

Один hidden shape одновременно определяет качества всех grasпов. Training item поэтому должен быть не $(y,g,c)$, а

$$
\left(y,G,\mathbf c_X(G)\right),
\quad
G=(g_1,\ldots,g_B),
\quad
\mathbf c_X(G)=(C_X(g_1),\ldots,C_X(g_B)).
$$

Design $G\sim\nu(\cdot\mid y)$ рандомизируется каждый step и смешивает:

- 40% hard candidates около $C=0$;
- 25% visible-contact proposals;
- 25% candidates, whose jaw interaction tube пересекает target occlusion cone;
- 10% явно плохих/collision negatives.

Sampling weights должны быть зафиксированы до test evaluation. Иначе можно получить улучшение только от более удачного proposal distribution.

### 4.2. Generative prediction

RJPN берёт shared noise $e^{(s)}\sim\mathcal N(0,I)$ и выдаёт sample общей функции:

$$
\widehat{\mathbf c}^{(s)}
{}={}
\left(
f_\theta(y,g_1,e^{(s)}),\ldots,
f_\theta(y,g_B,e^{(s)})
\right),
\qquad s=1,\ldots,S.
$$

Один и тот же $e^{(s)}$ используется для всех $g_j$. Independent noise per grasp запрещён в основной модели, потому что разрушает common-hidden-shape semantics.

### 4.3. Invertible tail transform

Чтобы отрицательные margins и область около нуля влияли сильнее без потери propriety, вводится монотонная биекция

$$
h_\gamma(c)
{}={}
c-\gamma\tau\mathrm{softplus}(-c/\tau),
\qquad \gamma>0,\;\tau>0.
$$

Её производная $1+\gamma\sigma(-c/\tau)>0$, значит преобразование invertible и не смешивает разные distributions. Оно растягивает adverse $c<0$ tail, а не меняет target law на эвристически reweighted outcome distribution.

Обозначим $H(\mathbf c)$ component-wise применение $h_\gamma$.

### 4.4. Energy component

Monte Carlo training loss:

$$
\mathcal L_{\mathrm{ES}}
{}={}
\frac1S\sum_{s=1}^{S}
\left\|H(\widehat{\mathbf c}^{(s)})-H(\mathbf c)\right\|_2
{}-{}
\frac{1}{2S(S-1)}
\sum_{s\ne r}
\left\|H(\widehat{\mathbf c}^{(s)})-H(\widehat{\mathbf c}^{(r)})\right\|_2.
$$

Первый член требует accuracy, второй вознаграждает только diversity, подтверждаемую данными, и препятствует collapse. Energy score — multivariate strictly proper scoring rule при finite first moment; он применим к implicit ensemble без tractable density. Биективный $H$ сохраняет идентифицируемость distribution. [Gneiting & Raftery, proper scoring rules](https://stat.uw.edu/research/tech-reports/strictly-proper-scoring-rules-prediction-and-estimation-revised), [Scoring-rule training of generative networks](https://arxiv.org/abs/2112.08217).

### 4.5. Variogram component для ranking dependence

Energy score может быть относительно нечувствителен к ошибочной correlation structure. Поэтому добавляется proper variogram score:

$$
\mathcal L_{\mathrm{VS}}
{}={}
\sum_{j<k} a_{jk}
\left(
|h_\gamma(c_j)-h_\gamma(c_k)|^p
{}-{}
\frac1S\sum_s
|h_\gamma(\hat c_j^{(s)})-h_\gamma(\hat c_k^{(s)})|^p
\right)^2,
$$

где $0<p\le2$, а $a_{jk}=\exp[-d_{\mathcal G}(g_j,g_k)^2/\ell^2]$ сильнее связывает соседние grasps. Variogram score известен большей чувствительностью к dependencies; сумма с strictly proper ES остаётся strictly proper из-за ES term. [Variogram scoring rules](https://journals.ametsoc.org/abstract/journals/mwre/143/4/mwr-d-14-00269.1.xml).

Итог:

$$
\boxed{
\mathcal L_{\mathrm{process}}
{}={}
\mathbb E_{X,Y,G}
\left[
\mathcal L_{\mathrm{ES}}
+\lambda_v\mathcal L_{\mathrm{VS}}
\right]
}
$$

Никакого reconstruction, Chamfer, occupancy или hidden-point loss в основной модели нет.

### 4.6. Теоретический результат, достойный paper

Нужно формально доказать следующую proposition.

**Proposition (random-design identification).** Пусть:

1. conditional certificate process имеет separable sample paths и finite first moments;
2. distribution $\nu$ имеет support на всех open subsets compact query domain $\mathcal G_0$;
3. model family projectively consistent;
4. $H$ — биекция, а ES metric имеет strong negative type.

Тогда population minimizer ожидаемого random-design ES совпадает с истинными finite-dimensional conditional laws

$$
\mathrm{Law}\left((C_X(g_1),\ldots,C_X(g_B))\mid Y=y\right)
$$

для $\nu$-почти всех finite designs; при continuity это идентифицирует law процесса на $\mathcal G_0$.

Proof skeleton:

1. Условиться на $Y=y,G$.
2. Strict propriety ES идентифицирует transformed vector law.
3. Invertibility $H$ возвращает original vector law.
4. Интегрирование по full-support random designs и continuity расширяют equality с almost-everywhere designs.
5. Projective consistency + Kolmogorov extension связывают finite-dimensional laws с process law.

Это не новая теорема о proper scores сама по себе; novelty — random-query formulation для прямого обучения decision-function push-forward и её реализация без latent-state reconstruction. Нельзя продавать proof как фундаментальную новую probability theory.

## 5. Новая architecture: Ray–Jaw Process Network

### 5.1. Почему обычный point encoder слаб для этой задачи

Occlusion — не просто missing points. RGB-D сообщает три разных типа constraints:

1. measured target surface point;
2. known free space перед depth return;
3. censored space за foreground obstacle, где отсутствие target points не является evidence of emptiness.

Обычная point cloud сеть видит главным образом пункт 1 и может спутать «пусто» с «не наблюдалось». 3D occlusion fields ZeroGrasp подтверждают пользу явного различения self/inter-object occlusion, хотя сама работа использует это для reconstruction. RJPN кодирует ту же observation semantics без volumetric field. [ZeroGrasp](https://openaccess.thecvf.com/content/CVPR2025/papers/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.pdf).

### 5.2. Camera-ray tokens

Для subsampled depth pixels создаются tokens

$$
r_i=(o_c,d_i,z_i,\ell_i,f_i),
$$

где $o_c,d_i$ задают camera ray, $z_i$ — terminal depth, $\ell_i\in\{\text{target},\text{occluder},\text{background}\}$, $f_i$ — local RGB/depth feature. Дополнительно задаются free interval $[0,z_i)$ и, для occluder pixels, censored interval внутри coarse target ROI behind $z_i$.

Линию удобно кодировать Plücker coordinates $(d_i,m_i=o_c\times d_i)$. Они не заменяют depth, а дают компактные invariant incidence features. В light-field/neural-rendering literature ray/epipolar attention и target-ray canonicalization уже показали, что geometrically constrained attention может generalize лучше unconstrained aggregation и обходиться без explicit volumetric reconstruction. Это источник general-ML inspiration, не robotics pipeline. [GNT, ICLR 2023](https://openreview.net/pdf?id=xE-LtsE-xx), [Generalizable Patch-Based Neural Rendering](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920156.pdf), [Light Field Neural Rendering](https://openaccess.thecvf.com/content/CVPR2022/papers/Suhail_Light_Field_Neural_Rendering_CVPR_2022_paper.pdf).

### 5.3. Jaw-query geometry

Каждый $g$ индуцирует малый фиксированный набор geometric primitives:

- $J_L(g),J_R(g)$: pad closing rays/strips;
- $V_{close}(g)$: volume между fingers;
- $V_{body}(g)$: non-contact finger/palm volume;
- $V_{approach}^{short}(g)$: только короткая pre-contact retraction, не весь arm path.

Для каждой jaw primitive $j$ вычисляются relative features с camera ray $r_i$:

$$
\phi(r_i,j)=
\left[
d_i^\top d_j,
\mathrm{dist}(r_i,j),
\mathrm{signedDepth}_{g}(z_i),
\mathbf 1[\text{ray tube overlap}],
\ell_i
\right].
$$

Все points/rays одновременно переводятся в gripper frame $g^{-1}$. Поэтому при общей rigid transform сцены и grasp query эти relative features неизменны. Получается exact query-conditioned SE(3) invariance без тяжёлого dense equivariant volume. Общая польза equivariance для rotation robustness и data efficiency подтверждается Vector Neurons, SE(3)-Transformers и complete-local-frame GNNs; в grasping OrbitGrasp уже показывает силу continuous equivariant quality functions, но остаётся deterministic. [Vector Neurons](https://openaccess.thecvf.com/content/ICCV2021/html/Deng_Vector_Neurons_A_General_Framework_for_SO3-Equivariant_Networks_ICCV_2021_paper.html), [SE(3)-Transformer](https://papers.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html), [Complete Local Frames](https://proceedings.mlr.press/v162/du22e.html), [OrbitGrasp](https://arxiv.org/abs/2407.03531).

### 5.4. Sparse ray–jaw incidence attention

Для query $g$ берутся только $k$ camera rays, чьи free/censored segments пересекают enlarged interaction tube

$$
T(g)=V_{close}(g)\cup V_{body}(g)\cup V_{approach}^{short}(g).
$$

Jaw tokens cross-attend к этим ray tokens с attention bias $b(\phi(r_i,j))$. Это отличает сеть и от global PointNet/Transformer, и от local surface rendering NeuGraspNet:

- сеть не спрашивает «какая поверхность находится в 3D point $p$?»;
- сеть спрашивает «какое observed/free/censored ray evidence релевантно контактному и collision tube этого grasp query?»;
- hidden surface никогда не materialize-ится.

Global target token из всех visible target points добавляется отдельно, чтобы shape prior не стал чисто локальным.

### 5.5. Shared latent process

Scene encoder выдаёт context $e_y$. Conditional normalizing flow либо lightweight diffusion-free transport генерирует

$$
z=T_\theta(\epsilon;e_y),
\qquad \epsilon\sim\mathcal N(0,I),
\qquad z\in\mathbb R^{d_z},\;d_z\approx16\text{--}32.
$$

Затем для любого query

$$
\widehat C(g;z,y)
{}={}
D_\theta\bigl(e_y,e_{inc}(g,y),e_g,z\bigr).
$$

Ключевое ограничение: decoder одного $g$ не принимает другие members candidate set. Поэтому добавление, удаление или permutation queries не меняет уже полученные samples; один shared $z$ определяет одну функцию. Это обеспечивает:

- permutation equivariance по queries;
- projective consistency finite query distributions;
- common-hidden-hypothesis coupling;
- возможность cache $e_y$ и независимо batch-query тысячи grasps.

Conditional Neural Processes и Functional Neural Processes показывают, что нейросети могут масштабируемо задавать distributions over functions; RJPN отличается task push-forward target, SE(3) query space, ray–jaw incidence и proper random-design loss. [Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a/garnelo18a.pdf), [Functional Neural Process](https://arxiv.org/abs/1906.08324), [dependent neural-process predictions](https://wessel.ai/assets/publications/Markou%2C%202022%2C%20Practical%20Conditional%20Neural%20Processes%20Via%20Tractable%20Dependent%20Predictions.pdf).

### 5.6. Deterministic visible-space gate

Не надо заставлять stochastic network учить очевидную геометрию. Из measured points/free space аналитически вычисляются:

$$
c_{obs}(y,g)=
\min\{d(g,V_{shelf}),d(g,P_{occluder}),d(g,P_{visible\ target\ noncontact})\}.
$$

Итоговый sample:

$$
\widehat C_{final}(g;z,y)
{}={}
\min\{\widehat C_{hidden}(g;z,y),c_{obs}(y,g)/s_d\}.
$$

Так сеть отвечает только за distributional ambiguity скрытой target geometry. Это снижает sample complexity и делает failure analysis физически понятным, не превращая модель в causal failure-mode system.

### 5.7. Candidate generation и optimization

На первой версии paper candidate generator должен быть frozen и одинаков для всех rankers. Можно объединить:

1. visible-surface proposals стандартного GPG/Contact-GraspNet/GSNet;
2. coarse SE(3) seeds в target ROI, включая occlusion cone;
3. local gradient refinement нижнего empirical quantile RJPN.

Для $S$ shared latent samples:

$$
\widehat q_\alpha(g)
{}={}
\mathrm{Quantile}_\alpha
\{\widehat C(g;z_s,y)\}_{s=1}^{S}.
$$

Так paper проверяет именно selection under ambiguity, а не скрывает gain в proposal heuristic. Отдельной extension может быть end-to-end proposal head.

### 5.8. Вычислительная гипотеза

При $N$ input rays, $B$ candidates, $k\ll N$ relevant rays/query и $S$ process samples:

$$
\text{cost}\approx O(\mathrm{Enc}(N))+O(Bk)+O(SB),
$$

без $O(R^3)$ voxel grid, marching cubes или multi-camera ray marching. Это пока hypothesis, а не доказанный speedup. Design target: $N=2048$, $B=512$, $k=48$, $S=16$, менее 100 ms ranking после candidate generation на современной GPU. В paper надо сообщить end-to-end wall time, peak memory и energy, а не только decoder time.

## 6. Как сделать distribution learnable без full reconstruction labels

### 6.1. Данные уже достаточно велики

ACRONYM содержит 17.7M physics-labeled parallel-jaw grasps для 8,872 объектов из 262 категорий; это естественная база для full-mesh certificate generation. GraspNet-1Billion содержит 190 real RGB-D scenes, 97,280 images с двух cameras и более 1.1B annotated grasps, но его обычные cluttered scenes не изолируют требуемую foreground-occlusion ambiguity. [ACRONYM](https://research.nvidia.com/publication/2021-05_acronym-large-scale-grasp-dataset-based-simulation), [GraspNet-1Billion](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf).

### 6.2. Synthetic observation generation

Для каждого $X$:

1. Поместить один target на shelf, без clutter.
2. Поместить один foreground occluder между wrist camera и target.
3. Render RGB-D, instance masks, camera rays и ground-truth visibility.
4. Добавить RealSense-like quantization, missing-depth edges, Gaussian/axial noise и extrinsic jitter.
5. Bin-ировать severity по visible target surface ratio: 70–100%, 50–70%, 30–50%, 10–30%.
6. Вычислить $C_X(g)$ только offline по full mesh; не подавать hidden points в model input или auxiliary target.

Train/val/test split обязателен по object instance и предпочтительно по category family. Random view split одного mesh создаст leakage shape prior.

### 6.3. Occlusion-equivalence families — решающий dataset element

Обычные random occlusions не доказывают ambiguity: сеть может распознать category по RGB и почти детерминированно угадать backside. Поэтому нужен paired controlled split.

Строятся families $\{X_1,\ldots,X_K\}$, для которых при одном camera/occluder:

$$
d_{vis}(Y(X_i),Y(X_j))<\varepsilon,
\quad
\text{но}
\quad
\|\mathbf C_{X_i}(G)-\mathbf C_{X_j}(G)\|>\Delta.
$$

Два способа:

- **natural fibers:** retrieval и alignment CAD shapes с близкими visible partial clouds, но разными hidden grasp labels;
- **controlled fibers:** локальные mesh/SDF деформации только в region, закрытом obstacle rays и не нарушающем measured free-space constraints; видимые pixels остаются буквально одинаковыми.

Controlled variants нужны прежде всего для falsification benchmark, а не для заявления photorealism. В real test используются обычные предметы.

При minibatch training полезно помещать несколько members одной fiber family с одинаковым $y$, чтобы proper score видел реальные multi-modal conditional samples, а не надеялся восстановить modes только из smoothness across unrelated observations.

### 6.4. Почему labels дешевле reconstruction

Для одного mesh можно один раз вычислить dense candidate certificates и многократно render-ить разные masks/noise. Не нужны watertight occupancy samples по всей сцене и Chamfer supervision. ACRONYM уже показывает практическую ценность миллионов simulation-labeled grasps и существенное улучшение при росте dataset diversity. Это косвенное, не прямое доказательство learnability proposed process. [ACRONYM paper](https://arxiv.org/abs/2011.09584).

## 7. Calibration и безопасное abstention

### 7.1. Почему raw posterior недостаточен

Даже strictly proper training не даёт finite-sample guarantee при model misspecification или sim-to-real shift. Более того, $g^*(y)$ выбирается после max по многим candidates, поэтому naive per-candidate calibration может ломаться от winner's curse.

### 7.2. Policy-level calibration

После freeze RJPN и candidate generator на calibration scenes исполняется вся selection policy. Вводится nested family:

$$
\pi_\lambda(y)=
\begin{cases}
g^*(y), & \widehat q_\alpha(g^*)\ge\lambda,\\
\texttt{abstain}, & \text{иначе}.
\end{cases}
$$

Loss, например,

$$
\ell_\lambda(y,X)
{}={}
\mathbf 1[\pi_\lambda(y)\ne\texttt{abstain}]
\mathbf 1[C_X(\pi_\lambda(y))\le0].
$$

С увеличением $\lambda$ система становится более conservative. Conformal Risk Control умеет контролировать ожидаемое значение monotone loss и tight до $O(1/n)$ при exchangeability. Здесь корректный claim — population false-safe risk среди deployment draws/с учётом выбранной normalisation, а не per-scene conditional guarantee. [Conformal Risk Control, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf), [risk-controlling prediction sets](https://arxiv.org/abs/2101.02703).

Нужно отчётливо показывать risk–coverage curve: lowering false-safe rate ценой частых abstentions само по себе не является SOTA grasping.

## 8. Экспериментальный protocol, который действительно проверяет claim

### 8.1. Главные research questions

**RQ1.** Даёт ли process posterior более высокий top-1 success и меньший false-safe rate, чем deterministic probability/quantile predictors при одинаковых candidates и backbone capacity?

**RQ2.** Возникает ли преимущество именно при severe foreground occlusion и occlusion-equivalent shapes, а не на easy visible grasps?

**RQ3.** Нужны ли joint dependencies: превосходит ли shared-latent Energy–Variogram objective independent marginal CRPS/pinball?

**RQ4.** Даёт ли ray–jaw incidence architecture выигрыш сверх обычного PointTransformer/SE(3) encoder?

**RQ5.** Действительно ли отказ от reconstruction улучшает task accuracy/compute, а не только меняет output representation?

### 8.2. Baselines

Минимальный набор:

1. **Direct Bernoulli:** тот же encoder/decoder, BCE по $\mathbf 1[C>0]$; Dex-Net-like scientific baseline.
2. **Deterministic margin:** Huber/MSE по $C$.
3. **Independent quantiles:** pinball/CRPS per grasp без shared latent.
4. **Deep ensemble:** 5 deterministic margin networks.
5. **Completion + same certificate evaluator:** PSSNet/ZeroGrasp-like multiple completions, затем analytic $C$.
6. **NeuGraspNet:** implicit completion + quality, retrained on the same isolated-occluder data.
7. **S4G/GSNet/Contact-GraspNet/AnyGrasp:** standard single-view detectors where technically compatible.
8. **OrbitGrasp-style deterministic continuous quality field:** closest function-valued, but non-stochastic baseline.
9. **RJPN marginal-only:** architecture control without process loss.

Для ranking experiment всем implicit rankers даётся один candidate pool. Отдельно можно сравнить native end-to-end systems, но эти цифры нельзя смешивать.

### 8.3. Метрики

Не ограничиваться GraspNet AP.

**Decision:**

- top-1 certificate success $P[C>0]$;
- physics-sim lift success на короткую фиксированную высоту как external validation;
- real-robot success;
- regret $\max_g C_X(g)-C_X(g^*)$;
- success vs visible ratio.

**Risk/calibration:**

- false-safe rate $P[C\le0\mid\widehat q_\alpha>0]$;
- empirical coverage каждого lower quantile;
- ECE/Brier для event $C>0$;
- risk–coverage/AURC при abstention;
- calibration after candidate max, не до него.

**Distribution/process:**

- held-out Energy Score;
- marginal CRPS;
- variogram score;
- pairwise rank correlation of sampled certificate landscapes;
- mode coverage на occlusion-equivalence families;
- smoothness/local Lipschitz diagnostics в $SE(3)$, не как target metric.

**Efficiency:**

- ms scene encoding, candidate scoring и total;
- peak VRAM;
- throughput grasps/s;
- number of geometry queries;
- parameter count.

### 8.4. Обязательные ablations

- shared $z$ vs independent $z_g$;
- ES vs ES+VS;
- identity $H$ vs tail transform;
- ray tokens vs terminal 3D points only;
- incidence sparse attention vs global cross-attention;
- gripper-frame canonicalization vs augmentation only;
- no observed-space analytic gate;
- latent dimensions 8/16/32/64;
- samples $S=2/4/8/16/32$;
- random occlusions only vs fiber batches;
- certificate margin vs binary label;
- with vs without reconstruction auxiliary loss.

Последняя ablation особенно важна: если occupancy auxiliary supervision заметно помогает без materialized reconstruction на inference, честный вывод может быть «task push-forward benefits from geometric auxiliary training», а не догма «geometry supervision вредна».

### 8.5. Real-robot design

Минимально убедительный вариант:

- 30–40 unseen rigid household objects;
- 5 foreground occluders различной ширины/texture;
- 3 severity bins;
- 3 repeats на object–occluder condition;
- fixed wrist RGB-D pose и одинаковый candidate generator;
- randomized method order;
- заранее прописанные exclusion criteria;
- Wilson confidence intervals и paired statistical test.

Главный endpoint — successful grasp and tiny lift. Approach path planner одинаков для методов и не входит в learned score. Failures reachability/arm collision считаются отдельно и не используются для подтверждения certificate claim.

## 9. Самые быстрые kill tests до большого проекта

### Kill test A: 2D ambiguous fibers

Сгенерировать 2D silhouettes с одинаковой видимой передней дугой и разной скрытой задней дугой. Grasp — пара parallel lines, certificate — signed antipodal/clearance margin. Сравнить BCE, independent quantiles и process ES.

Проект следует остановить или существенно упростить, если RJPN-like shared process не улучшает одновременно:

- lower-tail calibration после argmax;
- regret выбранного grasp;
- recovery двух modes hidden geometry.

### Kill test B: exact same observation, two hidden shapes

Дать буквально одинаковый $y$ с двумя/четырьмя equally likely certificate landscapes. Правильная модель должна sample-ить эти landscapes, а не их pointwise average. Это unit test objective, не benchmark performance.

### Kill test C: candidate dependence necessity

Если independent marginal quantile network совпадает с process model по top-1 success, calibration, sample efficiency и compute на 3D fibers, joint-process story для one-shot grasp selection не оправдана. Тогда paper надо переформулировать вокруг direct certificate quantiles и ray–jaw architecture, но ICLR novelty станет заметно слабее.

### Kill test D: completion comparison

При равном compute/data сравнить multiple-completion posterior + exact certificate. Если он устойчиво лучше RJPN в severe occlusion и не медленнее, тезис о достаточности direct push-forward остаётся математически верным, но practical/SOTA motivation рушится.

## 10. Novelty audit относительно ближайших работ

| Работа | Что уже сделано | Почему proposed contribution не сводится к ней | Что обязательно сравнить |
|---|---|---|---|
| Dex-Net 2.0 | Direct probability of robust grasp success из synthetic depth | Один scalar Bernoulli predictor; planar setup; нет process law и foreground-fiber benchmark | BCE/direct probability baseline |
| S4G | Single-view amodal 6-DoF proposals, robustness к pose perturbations | Deterministic proposal/score, не posterior hidden-shape certificate process | Same candidate robustness labels |
| PSSNet | Diverse plausible full completions | Декодирует full geometry | Multiple completions + exact scorer |
| NeuGraspNet | Single-view implicit occupancy, global/local surface rendering, implicit SE(3) score | Reconstruction является core; BCE scalar quality | Accuracy/runtime/memory; same candidates |
| ZeroGrasp | Octree CVAE joint shape+grasp; explicit 3D occlusion fields | Генерирует reconstruction/grasp octrees; multi-object clutter | Severe occlusion split, no-completion ablation |
| TOSC | Completion только task-relevant contact regions | Всё ещё completion; dexterous/task-language setting | Contact-only completion baseline if code/data permit |
| SpringGrasp | GPIS uncertainty и compliant dexterous planning | Surface uncertainty from GPIS, known point cloud; dynamic compliant multi-finger metric | Не основной baseline, но related uncertainty |
| FIRMGrasp | CVaR friction-aware probabilistic closure certificate | Randomness — friction при заданной geometry/contact set, не perceptual hidden shape | Use risk theorem carefully; no priority claim |
| OrbitGrasp | Continuous equivariant grasp quality function over sphere | Deterministic function, не conditional stochastic process | Continuous-field baseline |
| Conditional/Functional Neural Processes | General distributions over functions | Нет task push-forward, occlusion geometry или grasp objective | Cite as mathematical architecture ancestry |
| Goal-oriented Bayesian inverse problems | Posterior QoI без полного parameter posterior | В основном linear/Gaussian inverse problems, не learned action-indexed process | Cite as conceptual justification, не method baseline |

Goal-oriented inverse-problem literature прямо утверждает, что если интересует quantity of interest, posterior этого QoI можно аппроксимировать без вычисления full parameter posterior, и выводит optimal low-rank approximations в linear-Gaussian case. Это важное косвенное подтверждение central direction, но не доказательство эффективности RJPN на grasping. [Goal-oriented optimal approximations](https://kiwi.oden.utexas.edu/papers/goal-oriented-bayesian-inverse-problem.pdf).

## 11. ICLR acceptance potential: строгая оценка

Официальный ICLR 2027 reviewer guide просит ответить, хорошо ли мотивирован метод и расположен в литературе, подтверждены ли claims, какова significance и создаёт ли submission новое знание; SOTA сам по себе не обязателен. [ICLR 2027 Reviewer Guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines).

### 11.1. Почему это может быть ICLR paper

**Новый ML object.** Не «новый grasp detector», а conditional stochastic process, являющийся task push-forward latent-world posterior.

**Новый objective.** Proper random-design multivariate score по action queries с tail bijection и dependence-sensitive variogram component.

**Новая architecture.** Ray–jaw incidence operator + common latent process + projectivity, без reconstruction decoder.

**Теория.** Sufficiency/equivalence statement, process identification proposition, точное разделение model-based quantile и conformal population risk.

**Новый benchmark phenomenon.** Occlusion-equivalence fibers проверяют irreducible ambiguity, которую обычный AP скрывает.

**Практический эффект.** Потенциально лучше false-safe/top-1 performance и существенно меньше geometry compute, особенно при severe occlusion.

### 11.2. Что сделает submission лишь RSS/CoRL-level

- только robot success table без general process formulation;
- только замена BCE на energy score;
- ray attention без доказательства irreducible ambiguity;
- сравнение с устаревшими baselines;
- отсутствие calibrated uncertainty metrics;
- выигрыш из нового candidate generator вместо selector;
- reconstruction secretly used as training target without ablation;
- 10–15 hand-picked objects без confidence intervals.

### 11.3. Вероятные reviewer objections

**«Для argmax достаточно $P(success\mid y,g)$; зачем joint process?»**  
Правильный ответ не теоретический bluff. Да, при fixed binary utility и точных marginals joint law не нужен. Hypothesis paper-а: shared process структурирует scarce conditional samples, сохраняет continuous margins, даёт coherent candidate rankings и лучше calibrates selected extreme. Это должно быть доказано ablation/learning curves; иначе objection побеждает.

**«Это neural process applied to grasping.»**  
Ответом должны быть random-design identification result, physical push-forward formulation, ray–jaw operator, fiber benchmark и large empirical gap. Одного названия недостаточно.

**«Latent process всё равно неявно reconstructs shape.»**  
Latent может содержать shape information, и запрещать это бессмысленно. Проверяемое отличие: ни один decoder/output/loss не определяет occupancy or surface; эквивалентные по certificate shapes неразличимы для objective. Bottleneck/latent probing могут показать, что irrelevant backside geometry восстанавливается хуже, чем certificate.

**«Energy score плохо видит correlations.»**  
Именно поэтому добавлен variogram term; нужны synthetic correlation stress tests и ES-only ablation.

**«Conformal guarantee ломается under sim-to-real.»**  
Верно. Claim только exchangeable calibration regime; отдельно нужны corruption/shift curves и, возможно, небольшая real calibration set.

**«Signed certificate не равен actual lift success.»**  
Верно. Paper должен показать две оси: accuracy по certificate (проверка learning claim) и короткий lift outcome (external validity). Нельзя менять label post hoc, чтобы совпасть с robot trials.

### 11.4. Текущая субъективная оценка

- conceptual novelty: **8/10**;
- mathematical elegance: **8/10**;
- architecture novelty: **7.5/10**;
- efficient learnability: **7/10**, пока не пройден 3D kill test;
- potential practical impact: **8/10**;
- ICLR acceptance potential сейчас, без результатов: **6/10**;
- после process-identification theorem, fiber benchmark, modern baselines и убедительного real robot study: **8/10**.

Главная неопределённость — даст ли joint process измеримый one-shot selection gain сверх хорошо откалиброванной marginal quantile network. Это центральный scientific risk, его нельзя прятать.

## 12. План реализации по стадиям

### Stage 0 — 2–3 недели: falsify objective

1. 2D fiber simulator.
2. Shared-latent MLP process.
3. ES/VS implementation и unit tests на known Gaussian mixtures.
4. BCE, pinball, independent latent baselines.
5. Проверка process mode recovery и post-selection calibration.

Go criterion: заметное улучшение хотя бы по двум из трёх — regret, false-safe AURC, sample efficiency — без deterioration top-1 mean.

### Stage 1 — 4–6 недель: 3D offline feasibility

1. ACRONYM subset 500–1,000 shapes.
2. Один shelf/occluder renderer и RGB-D noise.
3. Cached signed certificates для 256–1,024 candidates/scene.
4. Point baseline и minimal ray–jaw incidence encoder.
5. Natural + controlled fiber split.

Kill criterion: independent quantile baseline неотличим от RJPN во всех severe bins при сопоставимом compute.

### Stage 2 — 6–10 недель: full model

1. Sparse incidence indexing CUDA/PyTorch.
2. Conditional latent transport.
3. Full random-design ES+VS.
4. Same-pool comparisons с NeuGraspNet-like completion и deterministic methods.
5. Calibration split, corruption suite, category OOD.

### Stage 3 — 4–6 недель: real robot

1. Frozen perception/candidate stack.
2. Preregistered object/occluder conditions.
3. Paired trials и reporting всех non-grasp failures.
4. Runtime/memory profiling на deployment GPU.

### Stage 4 — paper shaping

Основной текст на девять страниц должен держать одну линию:

1. irreducible ambiguity example;
2. task-pushforward process;
3. proper random-design objective;
4. ray–jaw process architecture;
5. fiber benchmark + real outcomes;
6. calibrated risk and compute.

Не добавлять VLM, active vision, tactile feedback, long-horizon planning или second gripper: они размоют claim.

## 13. Минимальный pseudocode

```text
TRAIN STEP
input: full mesh X, rendered occluded observation y, query set G={g_j}_{j=1..B}
target: c = [certificate(X, g_j)]_{j=1..B}

ray_tokens = encode_camera_rays(y)
scene_context = scene_encoder(ray_tokens)

for s in 1..S:
    eps_s ~ Normal(0, I)
    z_s = conditional_transport(eps_s, scene_context)
    for g_j in G:
        local_rays = sparse_incidence_lookup(ray_tokens, jaw_tube(g_j))
        q_j = ray_jaw_attention(g_j, local_rays, scene_context)
        c_hat[s,j] = decoder(q_j, z_s)
        c_hat[s,j] = min(c_hat[s,j], observed_clearance(y, g_j))

loss = tail_energy_score(c_hat, c)
     + lambda_v * variogram_score(c_hat, c, grasp_distances(G))
update(theta, loss)

INFERENCE
encode y once
generate common latents z_1..z_S
score a common candidate pool G
q_alpha[g] = empirical_lower_quantile({c_hat[s,g]}_s)
g_star = argmax_g q_alpha[g]
execute iff calibrated q_alpha[g_star] >= lambda
```

## 14. Falsifiable headline claims для будущего abstract

До получения результатов это hypotheses, не факты:

1. **Decision sufficiency:** direct certificate-process learning matches or exceeds multi-completion planning while eliminating any geometry decoder.
2. **Tail reliability:** RJPN lowers false-safe rate after candidate selection at matched execution coverage.
3. **Ambiguity recovery:** на identical-visible/different-hidden fibers модель восстанавливает multi-modal certificate landscapes, тогда как deterministic score и single completion усредняют или выбирают один mode.
4. **Process benefit:** shared-latent ES+VS превосходит independent marginal quantiles в severe occlusion, especially low-data regime.
5. **Efficiency:** sparse ray–jaw querying reduces ranking latency/memory relative to implicit occupancy + global/local rendering.
6. **Transfer:** gains сохраняются на real noisy RGB-D и unseen categories без real fine-tuning либо после небольшой calibration-only set.

Если claim 4 не подтверждается, core paper теряет наиболее принципиально новую часть. Если подтверждаются только 1/2/5, остаётся сильный applied architecture paper, но не исходная ICLR-level process-learning история.

## 15. Рекомендованный окончательный paper pitch

**Working title:** *Do Not Complete What You Cannot See: Learning Grasp-Certificate Processes under Occlusion*.

**One-sentence problem:** Из одного censored RGB-D observation выбрать parallel-jaw grasp, надёжный относительно training-distribution uncertainty скрытой target geometry, не восстанавливая эту geometry.

**One-sentence method:** Учить conditional stochastic process signed physical margins по $SE(3)$ с proper random-query score и ray–jaw incidence network, где shared latent sample кодирует согласованную hidden grasp hypothesis, но не shape.

**One-sentence result, который должен быть достигнут:** На occlusion-equivalent synthetic shapes и real foreground-occluded objects метод даёт лучший top-1/risk–coverage trade-off при меньшем runtime, чем direct scalar uncertainty и probabilistic completion baselines.

**Три contributions, не больше:**

1. task-pushforward process formulation + identification result;
2. tail-sensitive random-design proper objective + RJPN architecture;
3. occlusion-fiber benchmark и strong synthetic/real evidence.

## 16. Основные источники

### Robotics — только карта сделанного и gap

- Mahler et al. [Dex-Net 2.0](https://arxiv.org/abs/1703.09312).
- ten Pas et al. [Grasp Pose Detection in Point Clouds](https://arxiv.org/abs/1706.09911).
- Qin et al. [S4G](https://proceedings.mlr.press/v100/qin20a.html).
- Fang et al. [GraspNet-1Billion](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf).
- Eppner et al. [ACRONYM](https://arxiv.org/abs/2011.09584).
- Saund & Berenson [Diverse Plausible Shape Completions](https://proceedings.mlr.press/v155/saund21a.html).
- Jauhri et al. [NeuGraspNet](https://www.roboticsproceedings.org/rss20/p046.pdf).
- Chen et al. [SpringGrasp](https://www.roboticsproceedings.org/rss20/p042.html).
- Iwase et al. [ZeroGrasp](https://openaccess.thecvf.com/content/CVPR2025/papers/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.pdf).
- Wu et al. [TOSC](https://ojs.aaai.org/index.php/AAAI/article/download/38053/42015).
- Hu et al. [OrbitGrasp](https://arxiv.org/abs/2407.03531).
- Enwerem et al. [FIRMGrasp](https://arxiv.org/abs/2607.25049).

### General ML, statistics и inverse problems — источники формализации

- Spantini et al. [Goal-Oriented Optimal Approximations of Bayesian Linear Inverse Problems](https://arxiv.org/abs/1607.01881).
- Gneiting & Raftery [Strictly Proper Scoring Rules, Prediction, and Estimation](https://stat.uw.edu/research/tech-reports/strictly-proper-scoring-rules-prediction-and-estimation-revised).
- Scheuerer & Hamill [Variogram-Based Proper Scoring Rules](https://journals.ametsoc.org/abstract/journals/mwre/143/4/mwr-d-14-00269.1.xml).
- Pacchiardi et al. [Probabilistic Forecasting with Generative Networks via Scoring Rule Minimization](https://arxiv.org/abs/2112.08217).
- Garnelo et al. [Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a/garnelo18a.pdf).
- Louizos et al. [Functional Neural Processes](https://arxiv.org/abs/1906.08324).
- Varma et al. [GNT](https://openreview.net/pdf?id=xE-LtsE-xx).
- Suhail et al. [Generalizable Patch-Based Neural Rendering](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920156.pdf).
- Fuchs et al. [SE(3)-Transformers](https://papers.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html).
- Angelopoulos et al. [Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf).

## Заключение

Наиболее перспективный gap — не «лучше достроить скрытый объект», а **научиться не достраивать его вообще**. Partial observation задаёт distribution не над одним правильным grasp и не обязательно над одной plausible shape, а над целым grasp-certificate landscape. Прямое обучение push-forward этого landscape сохраняет ровно ту uncertainty, которая нужна для выбора действия, и выбрасывает нерелевантную geometry.

Сильная сторона идеи — согласованность problem, objective и architecture:

- problem требует reasoning об irreducible hidden geometry;
- objective учит joint conditional law certificate queries proper score-ом;
- architecture связывает queries общим latent shape hypothesis и маршрутизирует только camera rays, геометрически относящиеся к jaw interaction tube;
- decision использует lower tail, а deployment risk калибруется отдельно.

Слабое место тоже ясно: для forced one-shot selection точные marginals достаточны, поэтому joint-process advantage не гарантирован математикой. Именно это делает проект хорошей научной гипотезой, а не набором модулей. Первые 2D/3D fiber kill tests должны решить, существует ли реальный выигрыш, до затрат на большой simulator и robot study.
