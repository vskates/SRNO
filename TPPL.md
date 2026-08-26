# Task-Pushforward Grasp Processes: выбор parallel-jaw grasp без реконструкции скрытой формы

**Research freeze:** 25 августа 2026.  
**Статус:** исследовательская гипотеза и проверяемый план, а не утверждение уже достигнутого SOTA.  
**Область:** supervised/general ML + uncertainty quantification для single-view RGB-D parallel-jaw grasp selection. RL и VLA не используются.

## 0. Короткий ответ

Предлагаемая идея — **Task-Pushforward Process Learning (TPPL)**. Вместо восстановления скрытой формы объекта или выдачи независимого score для каждого grasp модель учит условное распределение **целых grasp-response functions**

$$
\Pi_x = \mathcal L\big(R_S(\cdot)\mid X=x\big),
\qquad R_S:\mathcal G\to[0,1],
$$

где $X$ — единственное шумное RGB-D наблюдение с foreground-окклюзией, $S$ — неизвестная полная форма, $\mathcal G$ — множество допустимых parallel-jaw grasp-кандидатов, а $R_S(g)$ — скалярная локальная робастность grasp $g$ на полной форме $S$. Это pushforward распределения скрытых форм через grasp-oracle, но сама форма никогда не декодируется.

Grasp-инстанциация называется **Occlusion-Conditional Grasp Response Process (OC-GRP)**. Она использует:

1. sparse RGB-D/visibility encoder без voxel grid, SDF или mesh;
2. небольшой набор глобальных posterior slots, каждый из которых задаёт один согласованный возможный response field по всем grasp-запросам;
3. grasp-centric query decoder;
4. новый **Action-Panel Energy Score (APES)** — sample-only strictly proper objective на панелях grasp-кандидатов, дополнительно чувствительный к их относительному ранжированию;
5. risk-aware selector, который выбирает grasp по lower-tail quality и tail-регрету относительно лучшего grasp в каждом возможном скрытом мире.

Центральная гипотеза: для выбора grasp нужна не скрытая поверхность как таковая, а только её образ в пространстве grasp-outcome functions. Этот объект существенно меньше полной геометрии, но богаче детерминированного $p(\text{success}\mid x,g)$: он сохраняет совместную неопределённость между альтернативными grasp-кандидатами, необходимую для tail-risk и shape-wise regret.

Самая сильная формулировка вклада для ICLR:

> **Учить не posterior скрытого мира, а его минимальный task-pushforward posterior на queryable response functions, используя proper scoring rule на случайных action panels.**

Это broad general-ML постановка для решений при частичной наблюдаемости; grasping — физически содержательная и жёсткая проверка.

---

## 1. Точный scope и что сознательно исключено

### 1.1. Вход и сцена

- один кадр RGB-D с wrist camera;
- известны calibration и поза камеры;
- один целевой объект на полке и ровно одно foreground-препятствие; heap/clutter не рассматривается;
- предполагается target mask или target identity от внешнего perception-модуля;
- depth может иметь axial noise, missing pixels и boundary artifacts;
- препятствие и полка наблюдаемы и используются для обычного collision/reachability filtering.

### 1.2. Выход

Не генерировать всю траекторию и не оценивать весь цикл «approach → close → большой lift». Модель **только ранжирует конечный набор кинематически и по наблюдаемому окружению допустимых parallel-jaw grasp poses**. Motion planner остаётся внешним и одинаковым для всех сравниваемых методов.

### 1.3. Что является learned target

Для полной training shape $S$ и terminal grasp $g$ offline-oracle возвращает один скаляр

$$
R_S(g)=
\Pr_{\delta\sim\mathcal P_{\rm local}}
\left[
\begin{array}{l}
\text{collision-free terminal closure,}\\
\text{two-contact/force-closure criterion,}\\
\text{retention under a 2 cm quasi-static lift}
\end{array}
\right]\in[0,1].
$$

$\delta$ интегрирует небольшие perturbations позы gripper, depth/calibration error и диапазон коэффициента трения. Это **локальная grasp robustness**, а не learned feasibility всего движения. На inference все nuisance variables уже свернуты в один response; модель не предсказывает длинный набор физических переменных.

### 1.4. Запреты, соблюдённые в дизайне

- нет RL;
- нет VLA/LLM policy;
- нет полной или локальной surface reconstruction как auxiliary task;
- нет scene SDF, TSDF, occupancy octree или decoded point cloud;
- нет causal failure-mode decomposition;
- нет learned approach/lift trajectory feasibility;
- нет композиции нескольких готовых robotics pipelines как основного вклада.

---

## 2. Что литература уже умеет и где именно остаётся gap

### 2.1. Карта robotics prior art

| Направление | Репрезентация/learning target | Что закрыто | Что остаётся открытым для нашей задачи |
|---|---|---|---|
| [S4G, CoRL 2020](https://proceedings.mlr.press/v100/qin20a.html) | single-shot point-wise grasp pose + scalar quality; robust annotation под pose perturbations | single-view, noisy/partial point cloud, parallel jaw | детерминированный score; нет posterior по скрытой форме и нет joint uncertainty между grasp-кандидатами |
| [6-DOF GraspNet, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Mousavian_6-DOF_GraspNet_Variational_Grasp_Generation_for_Object_Manipulation_ICCV_2019_paper.html) | VAE генерирует grasp poses; отдельный evaluator даёт $P(S\mid g,X)$ | multimodal grasp generation и refinement | latent моделирует множество хороших действий, а не неоднозначность outcome field скрытого объекта; evaluator pointwise |
| [VGN, CoRL 2021](https://proceedings.mlr.press/v155/breyer21a.html) | dense volumetric grasp map по TSDF | эффективный parallel-jaw grasping в clutter | требует volumetric geometry; сама статья отмечает point estimate ориентации как упрощение |
| [NeuGraspNet, RSS 2024](https://www.roboticsproceedings.org/rss20/p046.html) | single-view TSDF → implicit occupancy reconstruction → global/local surface rendering → BCE grasp quality | сильный single-view 6-DoF evaluator; local gripper/object features важны | reconstruction является core component; $L_{qual}$ — pointwise BCE; нет distribution over response functions |
| [ZeroGrasp, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.html) | octree-CVAE совместно декодирует occupancy/SDF/normals/grasps; 3D occlusion fields | SOTA на GraspNet-1B, 5 FPS, 1M images/11.3B grasps/12K objects | решает неоднозначность через full reconstruction; reconstructive loss и octree — именно то, что здесь требуется избежать |
| [Robust Grasp Planning over Uncertain Shape Completions, IROS 2019](https://arxiv.org/abs/1903.00645) | MC-dropout shape samples → mesh каждого sample → grasp evaluation на всех meshes | прямое свидетельство, что shape uncertainty улучшает robust grasping | дорогой detour через voxel/mesh; uncertainty существует в geometry space, не в минимальном task space |
| [Measuring Uncertainty in Shape Completion, 2025](https://arxiv.org/abs/2504.16183) | 60 MC-dropout completions; heuristic penalty по pointwise std | +7 п.п. к completion baseline и +23 п.п. к partial GPD; реальный Robotiq 2F-85 | 4 s completion + 2 s scoring; std не задаёт multimodal posterior и heuristic weight не proper |
| [SpringGrasp, RSS 2024](https://www.roboticsproceedings.org/rss20/p042.html) | differentiable compliant grasp metric under surface uncertainty | uncertainty-aware planning даёт не менее +18 п.п. к force-closure planner | dexterous/compliant grasp, не наш parallel-jaw selector; опирается на uncertain surface |
| [Cross-view Fusion, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_A_Cross-view_Fusion_Framework_for_Robust_6-DoF_Grasp_Pose_Estimation_CVPR_2026_paper.html) | вторая view + cross-view features | occlusion robustness без task-agnostic reconstruction | дополнительная камера/view запрещена нашей постановкой; работа косвенно подтверждает, что missing geometry остаётся bottleneck |

Две эмпирические линии особенно важны.

Первая: completion-aware методы действительно исправляют collisions и grasp ranking. Значит, игнорировать скрытую геометрию нельзя. Вторая: uncertainty поверх completion дополнительно улучшает grasping, но дорого и грубо. Следовательно, gap — не «ещё один direct grasp detector», а **амортизированное сохранение task-relevant hidden-shape uncertainty без decoding geometry**.

### 2.2. Почему обычного binary cross-entropy недостаточно — и когда оно достаточно

Пусть успех бинарный и robot выбирает один grasp для максимизации только ожидаемой вероятности успеха. Тогда

$$
g^*_{\rm mean}=\arg\max_g \Pr(Y_g=1\mid X=x)
$$

и идеально обученный pointwise BCE действительно достаточен. Нельзя честно продавать functional posterior как необходимый для этой узкой risk-neutral цели.

Новый объект становится необходимым при более сильном и здесь осмысленном определении надёжности:

- максимизировать lower-tail локальной robust quality, а не только её mean;
- ограничить вероятность быть существенно хуже shape-specific oracle grasp;
- иметь возможность менять risk tolerance без retraining;
- различать два случая с одинаковыми marginal means, но разной совместной структурой: «один grasp умеренно хорош на всех plausible shapes» и «он идеален на половине shapes и катастрофичен на другой половине»;
- корректно оценивать avoidable regret среди сотен адаптивно сравниваемых grasp-кандидатов.

Именно поэтому target должен быть распределением функций, а не набором независимых marginal Bernoulli heads.

### 2.3. General-ML опоры, но не готовое решение

- [Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a.html) формализуют scalable conditional distributions over functions и дают $O(n+m)$ inference по context/target points.
- [Functional Neural Processes](https://proceedings.neurips.cc/paper_files/paper/2019/hash/db182d2552835bec774847e06406bfa2-Abstract.html) показывают, что function-space uncertainty можно учить масштабируемо и получать более robust uncertainty estimates.
- [Strictly Proper Scoring Rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) дают теорию energy score: sample-based loss может честно elicitate целое распределение без tractable likelihood.
- [A Spectral Energy Distance, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/9873eaad153c6c960616c89e54fe155a-Abstract.html) практически показывает, что domain-structured energy distance может обучать implicit generator стабильно, unbiased minibatch-оценкой и с consistency guarantee.
- [Decision-Focused Learning through Learning to Rank, ICML 2022](https://proceedings.mlr.press/v162/mandi22a.html) подтверждает, что downstream decision quality часто требует pointwise/pairwise/listwise ranking losses, а не обычной predictive accuracy.
- [Decision-Theoretic Foundations for Conformal Prediction, ICML 2025](https://proceedings.mlr.press/v267/kiyani25a.html) доказывает связь uncertainty sets, Value-at-Risk и max-min решений и эмпирически улучшает safety–utility trade-off. Это поддержка risk-aware decision object, но не наш training objective.

Ни одна из этих работ не задаёт нужный robotics method. Inspiration — в математическом объекте: conditional stochastic process + proper sample score + decision geometry.

---

## 3. Итерации по идеям и причины отбраковки

### Кандидат A: simultaneous conformal lower band по всем grasps

Идея: учить $\hat r(x,g)$, калибровать score $s_i=\sup_g(\hat r-r)/\sigma$, выбирать grasp по simultaneous lower bound. Это решает adaptive-selection/optimizer’s-curse и даёт finite-sample marginal coverage.

**Почему не top-1:** это сильный evaluation/calibration layer, но как основная ICLR идея выглядит применением conformal prediction. Guarantee marginal по exchangeable test scenes, плохо переносит synthetic-to-real shift и не моделирует multimodal hidden-shape posterior. Оставить как optional calibration baseline, не как paper core.

### Кандидат B: K hidden-geometry witnesses без mesh decoding

Идея: сеть выдаёт K adversarial latent witnesses, покрывающих worst plausible contact geometries; выбрать grasp по worst witness.

**Почему отброшено:** без proper objective witnesses легко становятся произвольными adversarial modes; с geometric supervision это reconstruction in disguise. Неясно, какой loss гарантирует, что finite witnesses сохраняют нужную conditional distribution.

### Кандидат C: pairwise dominance graph над grasps

Идея: предсказывать $P(R(g_i)>R(g_j)\mid x)$, затем находить Condorcet/robust winner.

**Почему отброшено:** $O(M^2)$, возможны не-транзитивные циклы, теряется absolute quality. Listwise DFL уже близок концептуально; novelty недостаточна.

### Кандидат D: information-bottleneck latent скрытой формы

Идея: full-shape teacher сжимает объект в минимальный latent, сохраняющий grasp outcomes; partial-view student предсказывает latent distribution.

**Почему не выбран отдельно:** полезная идея, но bottleneck objective трудно интерпретировать и валидировать; latent может неявно запомнить shape. Она поглощается более чистой постановкой: response function сама определяет task-equivalence class, отдельный learned shape code не нужен.

### Кандидат E: pointwise distributional grasp evaluator

Идея: для каждого $(x,g)$ предсказывать Beta/quantile distribution robust quality.

**Почему недостаточно:** marginals не обязаны быть совместимы с одним скрытым объектом. Независимые samples для двух grasp-кандидатов могут соответствовать разным невозможным shapes, поэтому shape-wise regret и dominance неверны.

### Top-1: task-pushforward posterior over coherent response functions

Этот вариант сохраняет ровно тот объект, через который hidden shape влияет на решение; proper objective не требует likelihood; один global latent/slot обеспечивает совместимость всех queries; полная геометрия не появляется ни в output, ни в auxiliary loss.

---

## 4. Формализация TPPL

### 4.1. Скрытый мир, наблюдение и task map

Пусть:

- $S\in\mathcal S$ — полная target shape и локальные физические свойства, нужные grasp oracle;
- $O$ — известная геометрия shelf/foreground obstacle;
- $X=\mathcal O_{cam}(S,O,\eta)$ — single-view RGB-D observation с noise $\eta$;
- $\mathcal G(X)$ — finite candidate set после обычного reachability и observed-obstacle collision filtering;
- $T(S)=R_S(\cdot)\in\mathcal F$, где $\mathcal F\subset C(\mathcal G,[0,1])$.

Определим task equivalence:

$$
S\sim_T S' \quad\Longleftrightarrow\quad
R_S(g)=R_{S'}(g)\quad \forall g\in\mathcal G.
$$

Геометрически разные объекты в одном equivalence class неразличимы для текущей grasping task. Искомый объект:

$$
\Pi_x := T_{\mathrm{push}} p(S\mid X=x),
$$

то есть pushforward conditional shape distribution в quotient/function space. TPPL напрямую аппроксимирует $\Pi_x$, не аппроксимируя $p(S\mid x)$.

### 4.2. Почему это не «неявная реконструкция» в тривиальном смысле

Decoder разрешено спросить только $R(g)$ в конечном grasp pose. Нельзя запросить occupancy, normal, SDF или point coordinate в произвольной 3D точке. Две формы, имеющие одинаковое $R(\cdot)$, обязаны иметь одинаковое представление для loss. Следовательно, модель идентифицирует форму лишь с точностью до task equivalence; information about geometry в null-space $T$ не supervision и не требуется.

### 4.3. Минимальная достаточность: proposition sketch

**Proposition 1 (task sufficiency).** Для любого decision rule $a(X)\in\mathcal G(X)$ и любой bounded loss $L(a,R_S)$, зависящей от мира только через grasp responses, conditional Bayes risk полностью определяется $\Pi_X$:

$$
\mathbb E[L(a(X),R_S)\mid X=x]
=\int_{\mathcal F} L(a(x),f)\,d\Pi_x(f).
$$

Если класс losses содержит все bounded measurable probes of $R$, любая другая representation, сохраняющая Bayes risk для всех этих losses, должна различать все $T$-неэквивалентные миры. Значит, posterior на $\mathcal S/\!\sim_T$ является минимальным достаточным probabilistic object для данной task family.

Это не глубокая новая теорема само по себе; ICLR novelty должна быть в **learnable construction, APES consistency, finite-panel approximation и физических результатах**, а не в переименовании pushforward measure.

---

## 5. OC-GRP architecture

### 5.1. Sparse visibility encoder

Input tokens, без dense volume:

1. visible target points: $(p_i, rgb_i, n_i, c_i, v_i)$, где $c_i$ — depth confidence, $v_i$ — camera ray direction;
2. observed obstacle/shelf points с отдельным type embedding;
3. sparse censor tokens на rays, прерванных foreground obstacle внутри object-centric crop; token хранит ray direction и observed stopping depth, но не утверждает, что за ним находится occupancy.

SE(3)-equivariant sparse point transformer или vector-neuron encoder получает один scene context $c_X$. Camera rays сохраняют viewpoint/visibility information, поэтому строгая глобальная SE(3)-инвариантность не навязывается там, где она физически неверна.

### 5.2. Posterior scenario slots

Slot head выдаёт $K$ пар

$$
(u_k,w_k)_{k=1}^K,
\qquad w_k\ge0,\quad \sum_k w_k=1.
$$

Каждый $u_k$ — **не shape code**, а глобальный response-scenario token. Он используется неизменно для всех $g$ в одной action panel. Поэтому

$$
\hat R_k(g)=D_\theta(c_X,u_k,\gamma(X,g))
$$

является одной согласованной возможной функцией. Sampling нового $u$ независимо для каждого $g$ запрещён: это уничтожило бы joint uncertainty.

Практический старт: $K=8$, $d_u=64$. Более выразительный вариант — continuous latent generator $u=h_\theta(c_X,\epsilon)$; finite slots предпочтительнее для первого paper из-за single-pass inference и прямой визуализации modes.

### 5.3. Grasp-centric query decoder

$\gamma(X,g)$ строится только из observed points в фиксированной окрестности gripper, преобразованных в frame grasp $g$, плюс analytic embeddings трёх простых gripper volumes: left finger, right finger и closing corridor. Cross-attention к $c_X$ добавляет global shape prior.

Decoder возвращает один scalar $\hat R_k(g)\in[0,1]$. Никаких normals скрытой поверхности, contact points или occupancy он не выдаёт.

### 5.4. Candidate set

Core paper — **selection**, не proposal generation. Всем методам даётся один и тот же candidate bank:

- candidates от сильного frozen single-view proposal model;
- object-centric Sobol/SE(3) proposals в ограниченном диапазоне gripper width;
- одинаковый deterministic filter по observed shelf/obstacle collision и robot IK.

Так исключается ложная победа за счёт другого sampler. Отдельно измеряется oracle coverage candidate bank; если oracle grasp отсутствует, selector не наказывается как perception failure.

---

## 6. Новый learning objective: Action-Panel Energy Score

### 6.1. Action panels и process-level score

На training step выбирается panel

$$
G=(g_1,\ldots,g_M),\qquad g_m\sim\nu_X,
$$

где $\nu_X$ — смесь proposal distribution, uniform object-centric exploration и hard grasps около текущей decision boundary. Offline full-shape oracle даёт

$$
y_G=[R_S(g_1),\ldots,R_S(g_M)]^\top.
$$

Slot $k$ выдаёт $\hat y_G^{(k)}$ тем же одним $u_k$ для всех $M$ queries.

APES определяется не для одного фиксированного набора координат, а как интеграл finite-dimensional energy scores:

$$
\mathcal S_{\rm APES}(Q,R)
=\mathbb E_{G\sim\nu_0}
\left[
\mathcal S_E\!\left(
(\mathrm{ev}_G)_{\mathrm{push}} Q,\,
\mathrm{ev}_G R
\right)
\right],
$$

где $\mathrm{ev}_G(R)=[R(g_1),\ldots,R(g_M)]$, а в population construction base distribution $\nu_0$ имеет full support по расположению и конечным размерам panels. Это отличает objective от суммы независимых per-grasp scores: один panel оценивает joint finite-dimensional law, а интеграл по panels — stochastic process.

### 6.2. Decision-sensitive embedding

Пусть $B_G$ — incidence matrix sparse graph на grasp-кандидатах: edges соединяют близкие poses и top-vs-hard-negative pairs. Определим

$$
\Phi_G(y)=
\begin{bmatrix}
\sqrt{\lambda_{abs}}\,y\\
\sqrt{\lambda_{rank}}\,B_Gy\\
\sqrt{\lambda_{thr}}\,\sigma((y-\tau)/T)
\end{bmatrix},
\qquad
d_G(y,y')=\|\Phi_G(y)-\Phi_G(y')\|_2.
$$

- raw $y$ сохраняет absolute quality;
- $B_Gy$ делает loss чувствительным к pairwise ordering и decision regret;
- smooth threshold channel концентрирует capacity около practically acceptable quality $\tau$;
- $\lambda_{abs}>0$ делает embedding injective, поэтому rank term не теряет абсолютный масштаб.

### 6.3. Weighted finite-sample APES

Предсказываемая мера $Q_\theta(\cdot\mid X)=\sum_k w_k\delta_{\hat R_k}$. Loss одного scene:

$$
\boxed{
\mathcal L_{APES}
{}={}
\sum_{k=1}^{K} w_k d_G(\hat y_G^{(k)},y_G)
-\frac12\sum_{k=1}^{K}\sum_{\ell=1}^{K}
w_kw_\ell d_G(\hat y_G^{(k)},\hat y_G^{(\ell)})
}
$$

Первый член требует accuracy; второй не является ad-hoc diversity bonus. Это repulsive term energy score: он препятствует collapse только тогда, когда diversity нужна для совпадения с target distribution. Loss не требует density, KL posterior, adversary или decoded geometry.

### 6.4. Propriety: proposition sketch

**Proposition 2 (finite-panel Fisher consistency).** Для фиксированного $G$, Euclidean distance $d_G$ имеет strong negative type. Поэтому population energy score

$$
\mathbb E_{Y\sim P_G,\hat Y\sim Q_G}d_G(\hat Y,Y)
-\tfrac12\mathbb E_{\hat Y,\hat Y'\sim Q_G}d_G(\hat Y,\hat Y')
$$

минимизируется при $Q_G=P_G$; при injective $\Phi_G$ минимум единственен. Это прямое следствие теории energy scores, но предложенный action/rank embedding и conditional response-process use являются новыми частями.

Если $\nu_0$ даёт ненулевую вероятность panels любого конечного размера на dense subset $\mathcal G$, а $R$ continuous, совпадение всех finite-dimensional laws определяет law stochastic process. Практический training с максимальным размером $M$ идентифицирует только зависимости до порядка $M$, если не наложить дополнительную finite-latent/regularity assumption; это ограничение должно быть явно в theorem. Finite $K$ даёт оптимальную energy quantization внутри model class; approximation улучшается с $K$ и capacity decoder.

### 6.5. Adaptive panels без потери propriety

Uniform panels тратят много queries на очевидно плохие grasps. Однако sampling только около текущего predicted optimum меняет целевое распределение score и может сделать self-confirming ошибку. Поэтому использовать смесь

$$
q_\theta(G\mid X)
=(1-\rho)\nu_0(G\mid X)
+\rho\,\nu^{\rm hard}_\theta(G\mid X),
$$

где hard proposal выбирает predicted top grasps, near-ties и uncertain grasps. Каждый sampled panel получает detached importance weight

$$
\omega_\theta(G,X)
=\frac{\nu_0(G\mid X)}
{q_\theta(G\mid X)}.
$$

Тогда

$$
\mathbb E_{G\sim q_\theta}
[\omega_\theta(G,X)\,\mathcal S_{E,G}]
=\mathbb E_{G\sim\nu_0}[\mathcal S_{E,G}],
$$

то есть population target и strict propriety интегрального APES сохраняются, а compute концентрируется у decision boundary. Для discrete candidate bank все sampling probabilities известны. Обязательная ablation — adaptive panels с correction, без correction и uniform panels.

### 6.6. Decision-regret bound: корректная ограниченная версия

Energy distance индуцирует MMD для distance kernel. Для любого downstream utility $h_g(R)$ из соответствующего RKHS с $\|h_g\|_\mathcal H\le B$:

$$
|\mathbb E_{\Pi_x}h_g-\mathbb E_{Q_x}h_g|
\le B\,\mathrm{MMD}(\Pi_x,Q_x).
$$

Если $g_Q$ максимизирует predicted expected utility, а $g_P$ — true utility, стандартное telescoping даёт

$$
J_P(g_P)-J_P(g_Q)
\le 2B\,\mathrm{MMD}(\Pi_x,Q_x).
$$

Для CVaR аналогичный bound следует из Wasserstein approximation при Lipschitz quality, но **не следует автоматически только из малого empirical APES**. Поэтому paper должен либо доказать связь APES с нужной risk functional при дополнительных regularity assumptions, либо честно ограничить theorem expected smooth utilities и CVaR проверить эмпирически.

### 6.7. Один training step

1. Sample $(X,S)$; $S$ доступна только data loader/oracle, не network.
2. Один раз вычислить $c_X=E_\theta(X)$ и slots $(u_k,w_k)$.
3. Sample panel $G\sim q_\theta(\cdot\mid X)$; sampling probabilities сохранить, sampling path и importance weight detach.
4. Из precomputed table получить $y_G=\mathrm{ev}_G R_S$.
5. Одним batched query call получить все $\hat y_G^{(k)}\in\mathbb R^M$.
6. Вычислить $\omega_\theta(G,X)\mathcal L_{\rm APES}$ и backprop только через encoder, slots и query decoder.
7. Периодически обновлять hard-panel proposal по detached predicted top/uncertain grasps.

Ни teacher shape encoder, ни posterior network $q(u\mid S)$, ни reconstruction decoder не нужны. Это снижает риск, что метод окажется обычным CVAE с переименованным latent.

---

## 7. Inference и новый reliability objective

Для каждого candidate $g$ доступны согласованные samples $\{\hat R_k(g),w_k\}$. Определим scenario regret

$$
\Delta_k(g)=\max_{g'\in\mathcal G(X)}\hat R_k(g')-\hat R_k(g).
$$

Основной selector:

$$
g^*=\arg\max_{g\in\mathcal G(X)}
\left[
\mathrm{LCVaR}_{\alpha}(\hat R(g))
-\lambda\mathrm{UCVaR}_{\beta}(\Delta(g))
\right].
$$

Интерпретация:

- lower-CVaR запрещает grasp, катастрофичный для небольшой, но правдоподобной группы hidden shapes;
- upper-CVaR regret предпочитает grasp, близкий к лучшему доступному выбору в каждом plausible world;
- $\lambda=0,\alpha=1$ восстанавливает risk-neutral mean selector;
- risk parameters можно менять после обучения.

Более консервативная constrained версия:

$$
\min_g\mathrm{UCVaR}_{\beta}(\Delta(g))
\quad\text{s.t.}\quad
\Pr_{Q_x}[R(g)\ge\tau]\ge1-\epsilon.
$$

Primary physical metric всё равно top-1 success. Tail-regret — learning/diagnostic metric, а не замена реального эксперимента.

---

## 8. Как получить learnable conditional ambiguity

### 8.1. Training examples

Для каждой full CAD shape:

1. stable pose на shelf;
2. один foreground occluder с random geometry/pose;
3. controlled target occlusion ratio $0,20,40,60,80\%$;
4. RGB-D rendering с calibrated RealSense-like noise;
5. $M_{offline}$ grasp candidates и dense scalar $R_S(g)$ от local oracle;
6. full mesh после label generation удаляется из network input.

ZeroGrasp показывает, что synthetic scale порядка 12K objects и 11.3B grasp annotations возможен; наш response-only target может переиспользовать dense grasp annotation logic без high-resolution reconstruction targets.

### 8.2. Почему одного full shape на уникальный $x$ статистически достаточно

Conditional proper scoring rules Fisher-consistent в expectation по $(X,R)$; repeated identical $x$ формально не обязательны, как и в conditional generative modeling. Но finite data может позволить posterior collapse. Поэтому нужен специальный stress subset.

### 8.3. Occlusion-equivalent families

Создать пары/семейства сцен, у которых видимая RGB-D геометрия совпадает в пределах sensor tolerance, а скрытая grasp-response geometry различается. Без искусственного mesh splicing:

- category-aligned CAD nearest neighbours по visible Chamfer/RGB distance;
- разные back-side handles, cavities или thickness при одинаковом front silhouette;
- одинаковый occluder и camera pose;
- отдельный procedural set с аналитически контролируемым front profile и разными hidden backs.

Эти families дают прямой тест: deterministic head должен усредниться; OC-GRP должен выдать несколько coherent response modes и назначить им корректную массу.

### 8.4. Sim-to-real

- domain randomization depth noise, missing edges, intrinsics, lighting, material;
- небольшой calibration set реальных paired trials;
- APES fine-tuning только по executed/local-oracle grasp outcomes, без real full-shape scans;
- optional split-conformal scalar calibration top-1 risk после обучения, но не часть core objective.

---

## 9. Вычислительная эффективность

Пусть $N$ observed sparse tokens, $M$ grasp candidates, $K$ posterior scenarios.

- scene encoding делается один раз: sparse point attention, без $R^3$ volume;
- query decoding полностью parallel: примерно $O(KM d)$ после local-neighbour lookup;
- APES repulsion $O(K^2M)$; при $K=8$ это мало относительно encoder;
- нет marching cubes, virtual-camera ray marching, 60 completion passes или per-shape collision evaluation;
- memory зависит от $(N+KM)d$, а не от volumetric resolution.

Целевой engineering budget: $N\approx4096$, $M=512$, $K=8$, $d=128$, один forward pass <100 ms на desktop GPU. Это target, не предварительно доказанный результат. ZeroGrasp с 5 FPS — обязательный runtime baseline; если OC-GRP медленнее при худшем top-1 success, efficiency claim не проходит.

---

## 10. Эксперимент, который может подтвердить SOTA-потенциал

### 10.1. Новый benchmark slice

Публичные clutter benchmarks недостаточно изолируют нужную причину ошибки. Нужен paired single-target benchmark:

- target и exactly one foreground occluder;
- occluder не касается target;
- фиксированные bins occlusion ratio и depth-noise severity;
- seen-category/unseen-instance и unseen-category splits;
- shared candidate bank;
- oracle candidate coverage reported отдельно;
- real shelf set минимум 30 unseen objects × 5 occlusion layouts × 3 noise/view repeats.

### 10.2. Primary metrics

1. **real top-1 grasp-and-2cm-lift success**;
2. simulated top-1 success по occlusion bins;
3. hidden-geometry collision/contact failure rate;
4. oracle-normalized regret $R_S(g^*_S)-R_S(\hat g)$;
5. worst-20% success/CVaR across shape families;
6. calibration of $P(R(g)\ge\tau)$: Brier/NLL/ECE только как diagnostics;
7. APES/energy distance на unseen panels;
8. latency, peak memory, throughput candidates/s.

### 10.3. Baselines

Все получают одинаковые observations и candidates, насколько метод допускает evaluator-only use.

- PointNetGPD/6-DOF GraspNet-style direct pointwise evaluator;
- S4G-style deterministic scalar head;
- VGN/GIGA-style dense deterministic quality;
- NeuGraspNet;
- ZeroGrasp;
- uncertain completion + joint evaluation (Lundell et al.);
- completion uncertainty penalty (Duarte et al.);
- deep ensemble/Beta/quantile pointwise evaluator;
- latent neural process с independent Gaussian likelihood;
- OC-GRP $K=1$;
- OC-GRP без rank channel $B_Gy$;
- OC-GRP без global shared slot, с independent per-grasp latent — ожидаемый negative control;
- APES model с mean-only selection и с tail-regret selection.

### 10.4. Критические ablations

| Ablation | Проверяемая причинная гипотеза |
|---|---|
| full APES vs first accuracy term only | repulsive term учит conditional multimodality, а не просто noisy mean |
| shared global slot vs per-query noise | coherent world samples нужны для правильного shape-wise regret |
| raw response channel vs rank-only | absolute quality предотвращает выбор «наименее плохого», но физически плохого grasp |
| rank channel on/off | decision geometry ускоряет/улучшает top-1 selection при том же marginal error |
| uniform panels vs adaptive APES с/без importance correction | focus на decision boundary экономит labels, а correction предотвращает self-confirming bias |
| censor ray tokens on/off | explicit unknown-vs-free distinction помогает именно при foreground occlusion |
| $K=1,2,4,8,16$ | gain действительно связан с posterior capacity и достигает saturation |
| natural data vs +occlusion-equivalent families | controlled ambiguity предотвращает collapse |
| mean vs lower-CVaR vs regret-constrained | distribution используется decision rule, а не только даёт красивую uncertainty visualization |

### 10.5. Статистика

- paired trials: один и тот же object/layout/candidate bank для всех selectors;
- bootstrap CI по объектам, а не по миллионам коррелированных grasps;
- McNemar или paired hierarchical logistic model для real success;
- preregister primary occlusion bin $40\!-\!70\%$;
- report calibration и success отдельно на in-distribution и unseen-category;
- минимум три seeds для training, но physical unit анализа — объект/layout.

---

## 11. Косвенные доказательства, что идея может сработать

Это не прямое доказательство SOTA, а evidence chain.

1. **Скрытая geometry полезна.** NeuGraspNet делает reconstruction core component и показывает сильные результаты при hard views; ZeroGrasp достигает SOTA, явно используя reconstruction для contact/collision refinement. Значит, direct partial-only representation теряет важный сигнал.
2. **Но shape uncertainty важнее одной completion.** Lundell et al. показывают statistically significant improvement при joint evaluation по MC shape samples. Duarte et al. получают +7 п.п. относительно deterministic completion и +23 п.п. относительно partial GPD, хотя используют лишь MC-dropout std heuristic.
3. **Risk-aware physical modeling даёт большой эффект.** SpringGrasp сообщает не менее +18 п.п. к force-closure baseline under shape uncertainty. Хотя embodiment другой, направление эффекта поддерживает modeling uncertainty, а не только mean surface.
4. **Direct evaluator может быть сильнее generative geometry/planner.** [Get a Grip, CoRL 2024/2025 proceedings](https://proceedings.mlr.press/v270/lum25b.html) сообщает, что discriminative vision-based evaluators, при достаточном количестве positive и negative examples, превосходят analytic/generative baselines; статья прямо отмечает успех этого paradigm для parallel-jaw grasping.
5. **Dense supervision доступна.** ZeroGrasp-11B содержит 11.3B physically valid grasp annotations. APES требует много outcome queries, но не новых mesh reconstruction labels сверх тех CAD assets, на которых grasps уже вычисляются.
6. **Function-space uncertainty — рабочий ML объект.** Neural Processes обеспечивают scalable conditional stochastic functions; Functional Neural Processes показывают improved uncertainty. APES устраняет необходимость tractable likelihood и, подобно spectral energy distance, допускает stable sample-only training.
7. **Дополнительная view помогает именно из-за occlusion.** CVPR 2026 cross-view fusion улучшает hard/corner views. Это поддерживает постановку information deficit, но одновременно подчёркивает ценность метода, который остаётся single-view.

Почему TPPL потенциально лучше completion-based SOTA:

- capacity не тратится на hidden geometry, не влияющую ни на один candidate grasp;
- training metric совпадает с downstream response и ranking geometry;
- uncertainty не сводится к pointwise variance;
- inference амортизирует «sample shape → evaluate every grasp» в один conditional process forward;
- generalization может быть лучше, потому что разные shapes с одинаковым grasp behavior совместно обучают один equivalence class.

Это сильная, но проверяемая гипотеза. Возможен обратный результат: reconstruction auxiliary supervision может давать настолько полезный geometric inductive bias, что response-only модель проиграет на unseen categories.

---

## 12. Novelty audit на дату research freeze

### 12.1. Search-supported negative claim

Поиск по single-view/occluded grasping, uncertain shape completion, grasp quality fields, neural processes, proper scoring, decision-focused learning и функциональным posterior не обнаружил работы, которая одновременно:

1. определяет conditional law на **полных grasp-response functions** как pushforward скрытой shape distribution;
2. не reconstructs geometry даже auxiliary;
3. использует один shared latent world по всем grasp queries;
4. обучает implicit conditional law strictly proper energy score на random action panels;
5. включает decision/rank differences внутрь injective energy embedding;
6. применяет joint law к shape-wise tail regret при single-view foreground occlusion.

Это не математическое доказательство отсутствия статьи. Перед submission нужен повторный systematic search по Semantic Scholar/OpenAlex/Google Scholar, citation chasing от ZeroGrasp, NeuGraspNet, Lundell, Neural Processes и всех ICLR/NeurIPS 2026 papers.

### 12.2. Самые близкие угрозы novelty

| Близкая идея | Почему не совпадает | Что нельзя заявлять |
|---|---|---|
| Neural/Functional Processes | distributions over functions уже известны | нельзя заявлять изобретение stochastic function model или global latent |
| energy score | proper sample-based distribution learning известно | нельзя заявлять изобретение energy score; новый объект — action-panel decision embedding и task-pushforward use |
| decision-focused/listwise learning | ranking по solution set известен | нельзя заявлять, что pairwise/listwise differences сами по себе новые |
| uncertain shape completion | joint performance по geometry samples известно | нельзя заявлять первое uncertainty-aware grasp planning |
| ZeroGrasp | probabilistic latent + grasp/shape совместно уже есть | novelty — отказ от geometry decoding и posterior в response-function space, не просто «CVAE для occlusion» |
| exact [“Decision Quotient” terminology](https://arxiv.org/abs/2603.14689) | в 2026 появилась теоретическая работа с таким названием | не использовать “Decision Quotient” как название paper; говорить task-pushforward/equivalence и аккуратно цитировать adjacent theory при финальном related work |

### 12.3. Что действительно должно быть новым contribution

- постановка **latent-world posterior distillation into a response stochastic process**;
- APES: интегральный proper finite-panel objective с injective absolute + rank + threshold embedding и unbiased decision-adaptive panel sampling;
- consistency от finite action panels к continuous query process;
- risk/regret guarantee в подходящем function class;
- OC-GRP sparse architecture и controlled ambiguity dataset;
- эмпирическое доказательство, что task-pushforward posterior превосходит и pointwise evaluators, и geometry-posterior pipelines по success–risk–compute frontier.

Если из paper убрать theorem/APES и оставить «latent neural process для grasp scores», novelty будет недостаточной для ICLR.

---

## 13. ICLR acceptance audit

[ICLR 2027 Reviewer Guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines) формулирует главный вопрос как наличие достаточной ценности и нового знания, отдельно проверяет motivation/literature placement, correctness/rigor и significance и прямо говорит, что отсутствие SOTA само по себе не является основанием для reject. [ICLR 2027 Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers) явно включает probabilistic methods, UQ, structured prediction, robotics и general ML и призывает к ambitious, complete “slow science”.

### 13.1. Предварительная scorecard

| Критерий | Сейчас | Что нужно для сильной submission |
|---|---:|---|
| конкретность вопроса | 9/10 | сохранить scope selector-only и single target/occluder |
| conceptual novelty | 7.5/10 | доказать process identification из random panels и пользу importance-corrected decision-adaptive sampling; одной feature map недостаточно |
| technical correctness | 7/10 | строгие assumptions для process consistency и regret bound; не overclaim CVaR theorem |
| significance/general ML | 8/10 | показать TPPL как общий latent inverse-decision principle, не только robotics trick |
| empirical rigor potential | 9/10 | fair candidates, strong reconstruction baselines, real paired trials, controlled ambiguity |
| clarity | 8/10 | одна центральная схема: hidden world → task map → response posterior → risk selector |
| SOTA potential | 8/10 | выигрыш особенно вероятен при 40–70% occlusion и tight runtime budget; не гарантирован |

### 13.2. Вероятные reviewer objections и обязательные ответы

**“Pointwise BCE already learns success probability.”**  
Согласиться для risk-neutral binary objective; показать constructed pairs с одинаковыми marginals и разным joint regret, затем empirical gain tail-regret selector. Не прятать этот special case.

**“This is just a Neural Process application.”**  
Вынести APES theorem, task-pushforward minimality, action-panel sampling theory и comparison с generic latent NP. Желательно добавить второй non-robotic partial-observation decision benchmark или убедительный synthetic theorem experiment.

**“Response slots secretly reconstruct shape.”**  
Показать probes: из slots плохо восстанавливается Chamfer/occupancy, но хорошо responses; shape pairs с разной geometry и одинаковым response collapse together, а visually identical shapes with different response separate probabilistically.

**“Finite K misses continuous uncertainty.”**  
Дать $K$-scaling, continuous-latent ablation и saturation. Claim — efficient posterior coreset, не точный Bayesian posterior.

**“Synthetic oracle does not transfer.”**  
Real calibration set, perturbation randomization, physical top-1 trials и direct outcome fine-tuning without meshes.

**“Candidate generation leaks full shape.”**  
Candidate bank строится только из partial observation и одинаков для всех methods; отдельно report oracle coverage.

**“Foreground obstacle is only a renderer trick.”**  
Observed obstacle участвует и в visibility censor tokens, и в terminal collision filter; evaluation stratified by inter-object occlusion, а self-occlusion reported отдельно.

### 13.3. Go/no-go gates

Не подавать как ICLR main-track method, если выполняется любое:

1. APES не превосходит $K=1$ BCE/listwise baseline хотя бы на 3 п.п. top-1 success в primary 40–70% occlusion bin при matched runtime;
2. posterior modes не восстанавливаются на occlusion-equivalent families (mode coverage/calibration не лучше deep ensemble);
3. ZeroGrasp/NeuGraspNet имеют одновременно выше success и ниже latency;
4. gain исчезает на unseen-category или real shelf split;
5. rank channel улучшает только training metric, но не top-1/regret;
6. теоретический результат остаётся тривиальной переформулировкой известного energy score без нового finite-panel/decision implication.

Условие сильной подачи: improvement не только average success, но и **Pareto dominance по worst-tail success и latency/memory**, плюс один theorem, который не зависит от grasp-specific деталей.

---

## 14. Minimal viable implementation

### Phase 1 — falsify core statistical claim

- procedural 2.5D/3D shapes с одинаковым visible front и 2–4 hidden backs;
- 64–128 grasp queries;
- сравнить BCE, independent quantiles, global-latent NP и APES slots;
- измерить recovery conditional modes, energy distance и tail regret.

Если global coherence не даёт преимущества, остановиться до robot stack.

### Phase 2 — synthetic parallel-jaw benchmark

- 2–5K training shapes, затем scale;
- exact one-occluder scenes;
- local robust oracle;
- frozen candidate bank;
- $K=8, M=256/512$;
- strongest direct and completion baselines.

### Phase 3 — real shelf

- Robotiq-like gripper, wrist RGB-D;
- target segmentation supplied;
- 30+ novel objects;
- paired layouts and randomized method order;
- only 2 cm lift outcome;
- report failure taxonomy только как evaluation annotation, не как learned causal target.

### Phase 4 — ICLR generalization

Один небольшой второй domain, где latent object влияет на целую action-response function под partial observation. Например, choose-one support placement на частично наблюдаемом deformable height field. Использовать ту же TPPL/APES без robotics-specific architecture, чтобы отделить general objective от grasp engineering.

---

## 15. Paper claim set, который можно будет защищать

Только после успешных экспериментов:

1. **Conceptual:** task-pushforward posterior — более экономный uncertainty object для partially observed decisions, чем posterior полного latent world.
2. **Method:** APES учит conditional stochastic response processes по random action panels без likelihood и latent-world reconstruction.
3. **Theory:** APES proper на finite panels; dense random panels идентифицируют continuous response process; малый process discrepancy ограничивает regret для заданного utility class.
4. **Architecture:** shared scenario slots + grasp-centric decoder дают coherent, queryable uncertainty при single-pass inference.
5. **Robotics:** на single-view foreground-occluded parallel-jaw selection OC-GRP улучшает top-1 и worst-tail success при меньших compute/memory, чем uncertain completion.

Нельзя заранее утверждать “SOTA”. Правильная текущая формулировка: **framework имеет правдоподобный путь к SOTA на специально изолированной subproblem и содержит general-ML contribution, но статус зависит от head-to-head с ZeroGrasp/NeuGraspNet и completion-UQ baselines.**

---

## 16. Итоговое решение

Наиболее перспективное направление — не предсказывать hidden contacts, не восстанавливать surface и не добавлять uncertainty penalty к готовому grasp score. Следует изменить сам статистический target:

$$
\text{hidden shape posterior}
\xrightarrow{\text{pushforward through local grasp oracle}}
\text{posterior over grasp-response functions}.
$$

OC-GRP — архитектура для этого target; APES — learnable proper objective; tail quality + shape-wise regret — причина, по которой нужен joint function posterior. Этот набор образует одну цельную идею, а не pipeline из robotics modules.

Главная научная ценность будет не в лозунге «без reconstruction», а в доказанном принципе:

> При частичной наблюдаемости следует моделировать неопределённость в минимальном пространстве функций, на которых фактически принимается решение, и обучать это распределение proper score, определённым на совместных панелях действий.

Если принцип выдержит controlled ambiguity, strong reconstruction baselines и real shelf trials, он достаточно broad, falsifiable и technically substantive для ICLR.

---

## Основные источники

### Robotic grasping

- Qin et al. [S4G: Amodal Single-view Single-Shot SE(3) Grasp Detection in Cluttered Scenes](https://proceedings.mlr.press/v100/qin20a.html), CoRL 2020.
- Mousavian et al. [6-DOF GraspNet: Variational Grasp Generation for Object Manipulation](https://openaccess.thecvf.com/content_ICCV_2019/html/Mousavian_6-DOF_GraspNet_Variational_Grasp_Generation_for_Object_Manipulation_ICCV_2019_paper.html), ICCV 2019.
- Breyer et al. [Volumetric Grasping Network](https://proceedings.mlr.press/v155/breyer21a.html), CoRL 2021 proceedings.
- Weng et al. [NeuGraspNet](https://www.roboticsproceedings.org/rss20/p046.html), RSS 2024.
- Iwase et al. [ZeroGrasp](https://openaccess.thecvf.com/content/CVPR2025/html/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.html), CVPR 2025.
- Lundell et al. [Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645), IROS 2019.
- Duarte et al. [Measuring Uncertainty in Shape Completion to Improve Grasp Quality](https://arxiv.org/abs/2504.16183), 2025.
- Chen et al. [SpringGrasp](https://www.roboticsproceedings.org/rss20/p042.html), RSS 2024.
- Lum et al. [Get a Grip](https://proceedings.mlr.press/v270/lum25b.html), CoRL proceedings 2025.
- Zhu et al. [Cross-view Fusion for Robust 6-DoF Grasp Pose Estimation](https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_A_Cross-view_Fusion_Framework_for_Robust_6-DoF_Grasp_Pose_Estimation_CVPR_2026_paper.html), CVPR 2026.

### General ML and mathematics

- Garnelo et al. [Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a.html), ICML 2018.
- Louizos et al. [The Functional Neural Process](https://proceedings.neurips.cc/paper_files/paper/2019/hash/db182d2552835bec774847e06406bfa2-Abstract.html), NeurIPS 2019.
- Gneiting & Raftery [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf), JASA 2007.
- Gritsenko et al. [A Spectral Energy Distance for Parallel Speech Synthesis](https://proceedings.neurips.cc/paper/2020/hash/9873eaad153c6c960616c89e54fe155a-Abstract.html), NeurIPS 2020.
- Mandi et al. [Decision-Focused Learning: Through the Lens of Learning to Rank](https://proceedings.mlr.press/v162/mandi22a.html), ICML 2022.
- Kiyani et al. [Decision Theoretic Foundations for Conformal Prediction](https://proceedings.mlr.press/v267/kiyani25a.html), ICML 2025.
- ICLR [2027 Reviewer Guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines) and [Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers).
