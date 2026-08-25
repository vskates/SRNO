# Beyond Shape Completion: Decision-Quotient Process Learning for Occluded Parallel-Jaw Grasping

**Research memo, 25 August 2026.**  Статус: новая проверяемая гипотеза и design документа, а не заявление о достигнутом SOTA. По прямому указанию существующий репозиторий, `reports/EdgeFlux.md` и другие внутренние Markdown-идеи не читались. Поэтому novelty-аудит ниже проведён по внешней литературе; логически несовместимое требование «гарантировать непересечение с непрочитанными внутренними заметками» проверить невозможно.

## 0. Короткий вердикт

Предлагаемая top-1 идея — **Decision-Quotient Process Learning (DQPL)**, а её grasp-инстанцирование — **Visibility-Gated Grasp Certificate Process (VGCP)**.

Вместо восстановления скрытой формы объекта или выдачи одного confidence для каждого grasp модель учит условный stochastic process

$$
Q_x^* = \mathcal L\!\left(M_S(\cdot)\mid X=x\right),
$$

где $S$ — неизвестная полная форма только во время обучения, $X$ — одно шумное RGB-D наблюдение, а $M_S(g)$ — **один статический grasp-certificate margin** для parallel-jaw grasp $g$. То есть обучается не posterior по форме, а его pushforward в пространство функций «grasp → margin».

Ключевой объект — quotient скрытых форм:

$$
S\sim_{\mathcal G} S' \quad\Longleftrightarrow\quad
M_S(g)=M_{S'}(g)\ \text{для всех допустимых }g.
$$

Геометрические различия внутри одного класса эквивалентности не могут изменить ни один рассматриваемый grasp и принципиально не должны тратить capacity модели. Это переносит goal-oriented inference из Bayesian inverse problems в conditional generative learning и robotic grasp selection.

Главные contributions, которые образуют цельную ICLR-paper, а не сборку robotic modules:

1. **Новая learning target:** conditional law целой action-certificate function — decision-sufficient quotient posterior — без decoded shape, occupancy или SDF.
2. **Новая setwise learning objective:** Random-Restriction Decision Kernel Score (RR-DKS), strictly proper на конечных ограничениях процесса и специально чувствительный к относительному regret соседних grasp-кандидатов.
3. **Новая архитектурная индукция:** один scene-level latent задаёт согласованную гипотезу скрытой формы для всех grasp queries; depth-ray visibility gate локализует stochastic capacity только там, где grasp зависит от невидимой геометрии.
4. **Новая decision rule:** absolute lower-tail certificate плюс upper-tail **shape-wise opportunity regret**, который невозможно корректно вычислить из независимых per-grasp uncertainties.
5. **Обобщаемое знание:** если downstream использует латентное состояние только через family of action certificates, full latent reconstruction достаточна, но не необходима; posterior на quotient является минимальным для всех certificate-measurable решений.

Это явно не RL и не VLA. Модель не оценивает весь цикл approach–close–lift и не строит SDF сцены.

---

## 1. Точная постановка и границы

### 1.1 Лабораторный режим

- humanoid/wrist RGB-D camera;
- один целевой rigid object на полке;
- одно foreground-препятствие, частично закрывающее target; dense clutter исключён;
- parallel-jaw gripper;
- одно наблюдение, без next-best-view и без удаления obstacle;
- цель физической проверки — закрыть gripper, схватить и поднять на малую фиксированную высоту;
- target point cloud может иметь axial noise, edge dropout и outliers;
- во время теста нет CAD model или полной формы; shape prior присутствует только в обучающем распределении.

Предполагаются target mask и калибровка камеры. Ошибки segmentation должны тестироваться отдельно, но не являются основной научной задачей.

### 1.2 Что именно решает paper

Входом selector служат $X=(P_t,P_o,C,\Sigma_d)$: partial target cloud, наблюдаемые точки obstacle/shelf, camera rays и depth-noise descriptor. High-recall proposer предоставляет $K$ parallel-jaw candidates $\mathcal G(X)$. Contribution ранжирует и при необходимости локально уточняет эти candidates.

Таким образом, главный объект исследования — **reliable grasp selection under hidden target geometry**, а не универсальный end-to-end manipulation stack. Candidate recall обязательно измеряется отдельно, иначе плохой proposer ошибочно будет выглядеть как плохой selector.

### 1.3 Явные non-goals

- RL, offline RL, distributional RL;
- VLA/VLM reasoning;
- active perception, tactile exploration;
- full object/scene completion;
- long-horizon planning, regrasp, place;
- learned feasibility всего approach-to-lift trajectory;
- causal taxonomy/failure modes;
- dense clutter removal;
- large scene SDF/TSDF как вход или prediction target.

Obstacle и shelf входят только в быстрый conservative collision filter конечной gripper pose: observed point cloud дилатируется на sensor-error radius. Это стандартная наблюдаемая constraint, а не новый learned outcome.

---

## 2. Что уже существует и где остаётся gap

### 2.1 Direct grasp prediction по partial cloud

[Contact-GraspNet](https://arxiv.org/abs/2103.14127) привязывает 6-DoF pose к наблюдаемому contact point, снижает effective output до 4-DoF и сообщает более 90% success в structured clutter. [AnyGrasp](https://doi.org/10.1109/TRO.2023.3281153) даёт dense parallel-jaw grasps, устойчив к сильному depth noise и сообщает 93.3% bin-clearing success. [GraspGen](https://github.com/NVlabs/GraspGen) использует diffusion proposal generator и on-generator discriminator; авторы сообщают 17% gain на FetchBench, 21× меньшую память и 20 Hz.

Эти результаты подтверждают, что partial-cloud proposer и отдельный learned scorer могут быть быстрыми и сильными. Но их scalar quality/confidence не является условным distribution по скрытой grasp-relevant геометрии и не моделирует совместную зависимость scores разных candidates через одну и ту же неизвестную форму.

### 2.2 Joint geometry + affordance

[GIGA](https://roboticsproceedings.org/rss17/p024.pdf) совместно учит occupancy и continuous grasp affordance; geometry branch особенно помогал в packed scenes с occlusion. [ICGNet](https://arxiv.org/abs/2401.09939) объединяет instance-centric reconstruction и grasp detection. [ZeroGrasp](https://openaccess.thecvf.com/content/CVPR2025/html/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.html) одновременно реконструирует объекты и предсказывает grasps примерно при 5 FPS, обучаясь на 1M images и 11.3B grasp annotations.

[TARGO/TARGO-Net](https://targo-benchmark.github.io/) — наиболее близкий benchmark: single-view target-driven grasping при occlusion до 90%. TARGO-Net завершает target shape и затем предсказывает grasp. На synthetic data shape completion дала до 18 percentage points при высокой occlusion; в real experiment GIGA упал с 70% до 30%, TARGO-Net — с 80% до 66.7%. Но сами авторы формулируют важное ограничение: **shape completion is beneficial but does not generalize to real-world environments** ([paper](https://arxiv.org/html/2407.06168v1)).

Следовательно, скрытая геометрия полезна, но декодировать её целиком — не единственный и не обязательно лучший statistical target.

### 2.3 Shape uncertainty

- [Robust Grasp Planning Over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645) генерирует MC-dropout voxel completions и оценивает grasp по всем shape samples; uncertainty-aware planning статистически превосходит single completion.
- [Gualtieri & Platt](https://pmc.ncbi.nlm.nih.gov/articles/PMC8022832/) сравнивают MC completion, contact uncertainty и direct success prediction. Их neural success predictor быстрее MC и сильнее uncertainty-unaware costs, но получает **completed local point cloud**, выдаёт одну binary probability и не учит joint action process.
- [vMF-Contact](https://arxiv.org/abs/2411.03591), ICRA 2025, параметризует directional aleatoric/epistemic uncertainty через evidential vMF и сообщает 39% relative clearance-rate improvement; hidden geometry всё ещё входит лишь косвенно, включая auxiliary partial reconstruction.
- [Measuring Uncertainty in Shape Completion to Improve Grasp Quality](https://arxiv.org/abs/2504.16183) штрафует candidates за uncertainty completed cloud.
- [UNCLE-Grasp](https://arxiv.org/abs/2601.14492) особенно опасен для слабой novelty claim: MC-dropout completions, force-closure metrics, lower confidence bound и abstention уже объединены. На максимальной synthetic occlusion conditional success вырос с 0.78 до 0.87, но inference занимает около **57.53 s** против 2.48 s у completed CGNet, и paper явно не оценивает aleatoric posterior всех форм, совместимых с observation.

Значит, «несколько completions + LCB/CVaR» уже не ново. Gap уже: **learn the distribution of the decision-relevant functional directly, coherently across actions, without producing shape samples**.

### 2.4 General-ML и mathematical support

- Goal-oriented Bayesian inverse problems давно различают posterior параметров и posterior quantity of interest. [Spantini et al.](https://arxiv.org/abs/1607.01881) показывают, что можно оптимально аппроксимировать QoI posterior, не вычисляя полный parameter posterior, фокусируясь только на informed and goal-relevant directions.
- [Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a) и [Neural Diffusion Processes](https://proceedings.mlr.press/v202/dutordoir23a) показывают learnable distributions over functions и inference через finite marginals. DQPL не заявляет neural processes как своё изобретение; новое — choice of quotient target, visibility-gated grasp process и decision-proper objective.
- [Pacchiardi et al., JMLR 2024](https://www.jmlr.org/papers/volume25/23-0038/23-0038.pdf) показывают, что implicit conditional generators можно обучать adversarial-free proper kernel scores; population optimum восстанавливает conditional distribution, хотя на каждый continuous condition имеется один realised target.
- Decision-focused learning показывает несовпадение pointwise prediction error и downstream regret; listwise losses могут быть эффективнее ([Mandi et al., ICML 2022](https://proceedings.mlr.press/v162/mandi22a.html)). DQPL отличается тем, что сохраняет calibrated conditional law, а не оптимизирует один фиксированный downstream solver.

---

## 3. Итерации, которые следует отвергнуть

| Вариант | Почему сначала правдоподобен | Почему отвергнут |
|---|---|---|
| Full shape completion + grasp head | TARGO, GIGA и ZeroGrasp показывают gains | Запрещён full reconstruction; crowded literature; reconstructive loss тратит capacity на grasp-irrelevant backside; TARGO фиксирует real-domain weakness |
| Completion только contact region | Task-relevant QoI вместо full mesh | [TOSC, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/download/38053/42015) уже формулирует task-oriented completion потенциальных contact regions для dexterous grasp; всё равно выдаётся geometry |
| Stochastic completion + robust aggregation | Правильно сохраняет ambiguity | Lundell 2019 и UNCLE-Grasp 2026 уже делают completion sampling + robust scoring; 57.53 s у UNCLE демонстрирует compute problem |
| Evidential per-contact uncertainty | One-pass и сильные real results | vMF-Contact уже существует; per-grasp marginal confidence не задаёт correlations между взаимоисключающими candidates |
| Independent quantile heads $q_\alpha(g\mid x)$ | Очень дёшево и напрямую risk-aware | Для top-1 lower-tail ranking полезно, но не даёт joint hidden-shape hypotheses, shape-wise regret, coherent refinement или top-k fallback; novelty слишком мала |
| Conformal LCB как main contribution | Finite-sample coverage | Post-hoc wrapper; не использует training shape distribution и не решает representation problem. Допустим только как optional calibration |
| Whole-cycle success predictor | Максимально близко к robot outcome | Явно нежелательная постановка; смешивает approach planning, control и lift с target hidden geometry, затрудняя scientific attribution |

DQPL остаётся после этих отбрасываний, потому что меняет **statistical object being learned**, а не заменяет backbone или добавляет uncertainty penalty.

---

## 4. Formalization: latent shape → decision quotient

### 4.1 Random variables

Пусть

- $S\sim p_{\rm shape}$ — complete target geometry из training distribution;
- $O$ — foreground obstacle и shelf context;
- $X=\mathcal R(S,O,V)+\varepsilon$ — одно RGB-D observation при viewpoint $V$ и sensor noise $\varepsilon$;
- $g=(R,t,w)\in\mathcal A\subset SE(3)\times[0,w_{\max}]$ — final parallel-jaw configuration;
- $\mathcal A_X$ — candidates, прошедшие reachability и conservative collision с **наблюдаемым** obstacle/shelf.

### 4.2 Единственный learned target: static certificate margin

На complete training mesh offline вычисляется bounded margin $M_S(g)\in[-1,1]$. Он агрегирует только geometry at closure:

1. angular slack до нарушения antipodal friction cones;
2. slack допустимого jaw separation;
3. non-contact gripper/target clearance at the final pose;
4. normalized force-closure or gravity-wrench margin для фиксированного friction interval.

Практически используется smooth minimum этих четырёх нормированных margins. $M_S(g)>0$ означает certified static grasp; $M_S(g)<0$ — величину нарушения. Это **один scalar label**, а не vector state, trajectory outcome или scene SDF. Physics simulator lift используется только как independent evaluation label, не как prediction target.

Obstacle clearance $C_O(X,g)$ вычисляется отдельно из dilated observed cloud; модель не должна угадывать весь approach path.

### 4.3 Certificate process

Каждая complete shape индуцирует функцию

$$
T:S\mapsto F_S,\qquad F_S(g)=M_S(g).
$$

Истинный conditional object:

$$
Q_x^* = T_{\mathrm{push}}p(S\mid X=x),
$$

то есть pushforward shape posterior в function space $\mathcal F=C(\mathcal A_X,[-1,1])$. Model output — samples $F^{(\ell)}\sim Q_\theta(\cdot\mid x)$, но **никогда** samples формы.

### 4.4 Proposition 1: decision sufficiency

Для любой decision rule, loss и risk functional, измеримых только через finite restrictions $\{M_S(g_j)\}_{j=1}^K$, условный shape posterior $p(S\mid x)$ можно заменить на $Q_x^*$ без изменения Bayes action или Bayes risk.

**Proof sketch.** Любая такая величина имеет вид $h(T(S))$. По change of variables

$$
\mathbb E_{S\mid x}[h(T(S))]=\mathbb E_{F\sim T_{\mathrm{push}}p(S\mid x)}[h(F)].
$$

### 4.5 Proposition 2: minimality и information contraction

Если posterior representation $R(x)$ позволяет вычислить $\mathbb E[\phi(F_S)\mid x]$ для всех bounded continuous $\phi$, то $R(x)$ определяет $Q_x^*$ почти всюду. Следовательно, $Q_x^*$ минимален с точностью до a.s. invertible reparameterization для всего класса certificate-measurable decisions.

Кроме того, для любого f-divergence выполняется data processing:

$$
D_f(T_{\mathrm{push}}P\,\|\,T_{\mathrm{push}}\widehat P)\le D_f(P\,\|\,\widehat P).
$$

Full shape modeling therefore solves a strictly stronger problem: его точность достаточна для grasp decisions, но обратное неверно. Это не доказывает автоматически меньшую finite-sample error neural network, но даёт точную причину ожидать более благоприятный bias–variance/compute trade-off.

**Важная оговорка.** Для purely top-1 expected-margin selection достаточно per-action means. Joint process становится необходимым для tail risk относительно shape-wise best alternative, coherent local optimization, simultaneous candidate calibration и fallback sets. Поэтому paper обязан показать gain именно от joint process, а не только от stochastic scalar head.

---

## 5. Architecture: Visibility-Gated Grasp Certificate Process (VGCP)

### 5.1 Input encoder

1. Sparse point/point-transformer encoder обрабатывает $P_t$ и $P_o$ один раз. Point features: xyz, RGB, estimated normal, depth confidence, target/obstacle flag, camera ray.
2. Для каждого candidate $g$ локальные points переводятся в gripper frame и cross-attend к grasp query token. Это даёт translation/rotation handling без dense 3D grid.
3. Gravity и shelf normal задаются как explicit vectors; parallel-jaw symmetry обеспечивается weight sharing или averaging для $g$ и $gR_\pi$.

### 5.2 Deterministic ray-shadow gate

Для $R_p$ фиксированных probe points в closing region gripper:

- probe проецируется в depth image;
- сравнение probe depth с measured depth делит пространство на observed-free, observed-surface и behind-first-surface;
- $s(X,g)\in[0,1]$ — доля grasp-relevant probes в visibility shadow, объединённая с sensor-noise scale.

Это не reconstructed occupancy. Gate говорит только, **должна ли** prediction данного grasp зависеть от latent hidden-shape hypothesis.

### 5.3 Common latent, continuous query decoder

Берём $z\sim\mathcal N(0,I_r)$ один раз на scene/sample и задаём

$$
F_\theta(x,z;g)
=\operatorname{clip}\left(
\mu_\theta(x,g)+
s(x,g)\,\Delta_\theta(x,g,z),-1,1
\right).
$$

Один и тот же $z$ используется для всех $g_j$. Поэтому sample $\ell$ — одна согласованная grasp landscape, а не набор независимых uncertainty bars. Нелинейный decoder по $z$ позволяет multimodal certificate distributions без normalizing flow или diffusion chain.

Gate не должен жёстко обнулять residual: sensor noise оставляет малый floor $s_{\min}$. Ablation сравнивает no gate, learned gate и deterministic+learned correction.

### 5.4 Почему architecture efficiently learnable

- нет voxel grid, marching cubes или decoded point cloud;
- encoder cost $O(Nd)$ платится один раз;
- $K$ queries и $L$ posterior samples требуют $O(KLd)$ маленьких decoder evaluations;
- один global latent автоматически задаёт projectively consistent stochastic process: restriction sample на subset candidates совпадает с restriction того же full sample;
- practical target: $K=256,L=16$, менее 100 ms selector latency на одной современной GPU. Это experimental target, не предварительно заявленный result.

---

## 6. New learning objective: Random-Restriction Decision Kernel Score

### 6.1 Почему BCE/MSE недостаточны

Обычный BCE учит $\Pr(M>0\mid x,g)$, а MSE — conditional mean. Independent quantile regression восстанавливает marginal tails, но ни одна из этих losses не идентифицирует joint law

$$
(M_S(g_1),\ldots,M_S(g_K))\mid X=x,
$$

необходимый для shape-wise relative regret.

### 6.2 Random finite restriction

На каждом training scene выбирается query set

$$
G=(g_1,\ldots,g_K),\qquad
y^*=[M_S(g_1),\ldots,M_S(g_K)].
$$

Queries смешивают:

- candidates текущего high-recall proposer;
- near-boundary grasps $M\approx0$;
- **shadow pairs** с одним видимым contact anchor, но разными hidden-side jaw offsets/widths;
- local SE(3) perturbations успешных и неуспешных grasps.

### 6.3 Grasp graph kernel

Строится sparse kNN graph candidates по gripper-aware metric на $SE(3)\times\mathbb R$. Пусть $B_G$ — его normalized incidence matrix, а

$$
A_G=\begin{bmatrix}I_K\\ \beta B_G\end{bmatrix}.
$$

Первая часть сохраняет absolute margins; вторая усиливает различия $M(g_i)-M(g_j)$, то есть локальные ranking/regret contrasts. Используется fixed multiscale characteristic kernel

$$
k_G(u,v)=\sum_{b=1}^{B}\eta_b
\exp\!\left[-\frac{\|A_G(u-v)\|_2^2}{2\sigma_b^2}\right].
$$

Поскольку $A_G$ injective из-за блока $I_K$, Gaussian kernel остаётся characteristic на $\mathbb R^K$; contrast block меняет finite-sample sensitivity, не жертвуя propriety.

### 6.4 RR-DKS loss

Для $L$ common-latent samples $y^{(1)},\ldots,y^{(L)}$ минимизируется Monte-Carlo kernel score

$$
\mathcal L_{\rm RR-DKS}=
\mathbb E_{X,S,G}\left[
\frac{1}{L(L-1)}\sum_{\ell\ne m}k_G(y^{(\ell)},y^{(m)})
-\frac{2}{L}\sum_{\ell=1}^{L}k_G(y^{(\ell)},y^*)
\right].
$$

Это adversarial-free objective: первый term не даёт samples collapse, второй притягивает forecast law к observed function restriction.

### 6.5 Proposition 3: propriety

Для фиксированного $G$ population RR-DKS uniquely minimized истинным conditional finite-dimensional law $Q_x^{*,G}$, если kernel characteristic и model well specified. Если distribution random query sets имеет full support и sizes неограниченны в population idealization, равенство всех finite restrictions определяет process law. Практическое $K\le K_{\max}$ учит только необходимые finite restrictions; нельзя честно заявлять exact recovery бесконечного процесса.

**Proof sketch.** Expected kernel score отличается от $\mathrm{MMD}_{k_G}^2(Q_\theta^G,Q_*^G)$ только константой, не зависящей от $\theta$. Characteristic $k_G$ даёт ноль iff distributions equal. Рандомизация $G$ переносит утверждение на almost every queried restriction.

Warm-up mean loss допустим первые несколько epochs, но его weight должен обнулиться; иначе optimum уже не обязан быть истинным posterior.

---

## 7. Process-aware grasp selection

Для samples $m_{\ell j}=F_\theta(x,z_\ell;g_j)$ определим:

1. absolute lower-tail robustness
   $$
   A_j=\operatorname{LCVaR}_\alpha(m_{\cdot j});
   $$
2. shape-wise opportunity regret
   $$
   r_{\ell j}=\max_h m_{\ell h}-m_{\ell j},\qquad
   R_j=\operatorname{UCVaR}_\alpha(r_{\cdot j});
   $$
3. final score
   $$
   J_j=A_j-\lambda R_j+\eta\,\mathbb E[m_{\cdot j}],
   $$
   subject to $C_O(X,g_j)\ge\gamma$.

Исполняется $g^*=\arg\max_j J_j$. Если все $A_j<0$, система всё равно может вернуть best-effort grasp и отдельно выставить risk flag; abstention не является обязательной частью задачи.

Почему regret не дублирует LCVaR: $A_j$ штрафует absolute bad outcomes. $R_j$ спрашивает, был ли этот плохой outcome **избежим**, то есть существовал ли на той же hidden-shape hypothesis существенно лучший candidate. Общий latent $z_\ell$ критичен: независимые per-grasp samples разрушили бы shape-wise comparison.

### Proposition 4: decision stability

Пусть process paths bounded, metric — sup norm, а $W_1(Q_\theta,Q_*)\le\epsilon$. Lower/upper CVaR с tail mass $\alpha$ Lipschitz, evaluation $F\mapsto F(g)$ — 1-Lipschitz, а regret map $F\mapsto\max_hF(h)-F(g)$ — 2-Lipschitz. Тогда

$$
\sup_g|J_\theta(g)-J_*(g)|
\le \frac{1+2\lambda}{\alpha}\epsilon+\eta\epsilon,
$$

и plug-in selected grasp имеет не более удвоенного score gap. Это связывает process-distribution error с downstream selection, не обещая физический success без валидности certificate.

Поскольку decoder differentiable по $g$, после discrete ranking можно выполнить 5–10 projected gradient steps по smooth empirical $J(g)$, сохраняя obstacle constraint. Это optional refinement, не trajectory planning.

---

## 8. Training data без reconstruction target

### 8.1 Base assets

- ACRONYM meshes/grasps дают тысячи varied objects и миллионы analytic grasps;
- TARGO protocol используется для controlled occlusion bins и внешнего сравнения, но основной setup упрощается до target + one obstacle + shelf;
- Objaverse-LVIS subset допустим для long-tail shapes после watertight/scale filtering.

Full meshes нужны только offline для renderer и $M_S(g)$; ни mesh, ни point completion не являются network output.

### 8.2 Scene generator

Для каждого target:

1. sample pose on shelf and wrist-camera pose;
2. place one foreground primitive или household obstacle так, чтобы occlusion попала в bins 0–20, 20–40, 40–60, 60–80%;
3. render RGB-D;
4. add RealSense-like axial noise, depth-dependent variance, correlated noise, edge dropout, quantization и 0–2% outliers;
5. generate $K$ candidates only from partial observation;
6. evaluate scalar certificate on complete target mesh and visible-context collision analytically.

### 8.3 Occlusion twins: controlled identifiability test

Natural datasets плохо доказывают, что model сохранила conditional ambiguity. Нужен отдельный diagnostic benchmark:

- несколько complete shapes имеют идентичный front component;
- различаются только backside width, concavity или hidden support ridge;
- foreground screen гарантирует pixel-identical noiseless RGB-D target fragment;
- разные variants меняют ordering shadow-pair grasps.

Истинный discrete posterior и joint certificate law здесь известны. Это позволяет проверить mode coverage, pairwise correlations и regret calibration, а не только final success. Затем результат подтверждается на natural meshes и real objects; synthetic twins сами по себе недостаточны для paper.

### 8.4 Optional simultaneous calibration

После freezing модели split-conformal residual

$$
e_i=\max_{g\in\mathcal G_i}\left(\widehat q_\delta(x_i,g)-M_{S_i}(g)\right)
$$

может дать one-sided simultaneous correction для конечного candidate set при exchangeability. Это secondary safety layer. Он не исправляет distribution shift и не является core contribution.

---

## 9. Experimental program, способный подтвердить или опровергнуть идею

### 9.1 Три уровня проверки

**Level A — known-posterior occlusion twins.** Проверить recovery joint law, а не robot success.

**Level B — natural synthetic shelf scenes.** Novel-instance и category-held-out splits; controlled occlusion/noise; physics execution с малым lift.

**Level C — real wrist RGB-D.** Один target, один foreground obstacle, не clutter. Household objects плюс 3D-printed twins; фиксированные bins и минимум 300–600 attempts, randomized method order, Wilson/bootstrap 95% CI.

### 9.2 Candidate-controlled baselines

Всем selectors передаётся одинаковый candidate union и одинаковый obstacle filter:

1. raw Contact-GraspNet/AnyGrasp/GraspGen score;
2. deterministic VGCP: BCE или Huber mean;
3. independent quantile regression;
4. deep ensemble / MC dropout scalar scorer при matched FLOPs;
5. Gualtieri-style direct success predictor;
6. stochastic shape completion + multi-shape LCB/CVaR, Lundell/UNCLE-style;
7. TARGO-Net completion features + same candidate set;
8. ZeroGrasp reconstruction-based ranking where license/code permits;
9. vMF-Contact uncertainty score;
10. oracle complete-mesh certificate upper bound.

Отдельная end-to-end table допускает native candidates каждого метода, но не заменяет candidate-controlled table.

### 9.3 Metrics

**Primary:** top-1 physical/sim small-lift success by occlusion bin.

**Decision:** oracle-normalized certificate regret; lower-tail regret; success–coverage curve; failure rate at fixed coverage; candidate recall.

**Distribution:** held-out RR-DKS/energy score; marginal CRPS; pairwise sign accuracy $P[M(g_i)>M(g_j)]$; covariance error на twins; PIT/quantile coverage by occlusion and noise.

**Systems:** selector latency, peak memory, samples/sec, label-generation cost. Completion runtime должен включать mesh/point decoding и evaluation всех samples.

### 9.4 Critical ablations

- common latent vs independent latent per grasp;
- RR-DKS vs BCE, energy score, kernel score без $B_G$;
- absolute LCVaR only vs regret only vs combined $J$;
- visibility gate vs no gate;
- RGB-D vs depth only;
- shadow-pair sampling vs uniform candidates;
- scalar continuous margin vs binary success;
- latent dimension $r\in\{8,16,32,64\}$, $L\in\{2,4,8,16,32\}$, $K\in\{32,64,128,256\}$;
- learned posterior vs stochastic completion under matched wall-clock and memory;
- synthetic noise families and real sensor transfer.

### 9.5 Go/no-go thresholds

До заявления SOTA должны одновременно выполниться:

1. не менее **+5 percentage points absolute** top-1 success против strongest candidate-controlled baseline в combined 40–80% occlusion; 95% CI разности не пересекает 0;
2. statistically significant gain в worst occlusion bin и при high noise;
3. joint-process variant превосходит independent quantiles; иначе main novelty не нужна;
4. RR-DKS улучшает pairwise/covariance calibration на twins и real-like synthetic, не только average score;
5. selector $K=256,L=16$ укладывается в целевой <100 ms и существенно быстрее multi-completion baseline;
6. gain сохраняется на category-held-out shapes и real camera;
7. candidate recall не объясняет improvement.

Если пункты 1, 3 или 6 не выполнены, позиционирование как ICLR main paper следует остановить или радикально изменить.

---

## 10. Novelty audit по closest work

| Работа | Latent/output | Uncertainty | Joint law across grasp candidates | Full reconstruction at inference | Отличие DQPL |
|---|---|---|---|---|---|
| Dex-Net 2.0 | expected robustness $R(g,y)$ | marginalized training noise | нет | нет | DQPL учит full conditional certificate law и setwise contrasts |
| Contact-GraspNet / AnyGrasp / GraspGen discriminator | grasp pose + point score | deterministic или proposal diversity | нет decision posterior | нет | process samples share a hidden-shape latent |
| GIGA / ICGNet / TARGO-Net / ZeroGrasp | geometry + affordance | главным образом point estimate | нет calibrated process | да | quotient certificate replaces geometry output |
| Lundell 2019 | MC completed shapes | dropout shape samples | да, через одинаковые meshes | да, многократно | DQPL distills pushforward directly |
| Gualtieri & Platt 2021 SP | binary grasp success from completed local cloud | one probability | нет | да, one completion | no completion; continuous joint margin law; proper setwise score |
| vMF-Contact 2025 | directional contact distributions | evidential aleatoric/epistemic | limited hierarchical pose uncertainty | auxiliary partial reconstruction | hidden-shape certificate process, not orientation posterior |
| UNCLE-Grasp 2026 | multiple completed clouds + force closure | MC variance + LCB | shared completion samples | да, около 57.53 s | no shape samples; aleatoric conditional process; candidate-relative regret |
| DQPL/VGCP | $M_S(\cdot)\mid X$ | conditional hidden-geometry + sensor ambiguity | **да** | **нет** | decision quotient + RR-DKS + visibility gate |

Осторожная novelty claim после поиска:

> To the best of our external-literature search as of 25 August 2026, no prior grasping method directly learns a calibrated conditional stochastic process of a static grasp certificate over continuous parallel-jaw actions as the pushforward of hidden-shape uncertainty, without producing a full or contact-region geometry, and trains it with a proper setwise score that preserves shape-wise candidate contrasts.

Нельзя заявлять как новые: uncertainty-aware grasping, risk-sensitive selection, neural processes, kernel scores, task-aware inference или direct grasp scoring по отдельности.

---

## 11. Косвенные основания ожидать SOTA potential

Это evidence chain, не доказательство будущего результата.

1. **Hidden geometry materially matters.** TARGO-Net без shape completion теряет до 18 points на high occlusion; GIGA geometry branch особенно помогает в packed/occluded scenes.
2. **Single completion недостаточна.** Lundell et al. показывают statistically significant benefit uncertainty-aware multi-shape evaluation; UNCLE-Grasp улучшает severe-occlusion conditional success с 0.78 до 0.87.
3. **Но sampling shapes дорог.** UNCLE-Grasp сообщает около 57.53 s/object, главным образом из-за repeated completions/evaluations.
4. **Direct success surrogates могут быть быстрее MC и сильнее uncertainty-unaware costs.** Это эмпирически показано Gualtieri & Platt, хотя их target и input не дают joint posterior.
5. **Continuous action scorers работают.** GIGA implicit affordance, Contact-GraspNet contact rooting и GraspGen on-generator discriminator показывают эффективность query-based grasp fields/scorers.
6. **Function-distribution learners feasible.** Neural Diffusion Processes восстанавливают rich function laws через finite marginals; простой common-latent decoder выбран здесь ради меньшей latency.
7. **Proper kernel scores пригодны для implicit conditional generators.** JMLR 2024 даёт population consistency и adversarial-free training evidence.
8. **Goal-oriented posterior может быть существенно проще full inverse.** Теория goal-oriented Bayesian inverse problems прямо поддерживает inference QoI без parameter posterior.

Комбинация этих фактов делает superiority правдоподобным именно в high-occlusion/high-ambiguity regime, но слабая или нулевая occlusion, вероятно, останется областью паритета с deterministic scorers.

---

## 12. ICLR 2027 acceptance audit

Официальный [ICLR 2027 Reviewer Guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines) просит ответить: ясна ли конкретная проблема, мотивирован ли метод литературой, поддержаны ли claims rigorously, и создаёт ли работа significant new knowledge; SOTA сам по себе не обязателен.

### Originality

**Потенциал высокий**, если центральным contribution остаётся decision-quotient posterior + proper process objective. Только «новый grasping architecture» будет выглядеть incremental robotics.

### Significance to general ML

Paper следует подавать как новый принцип для **partial-observation inverse-to-decision problems**:

> Learn the posterior of an action-indexed quantity of interest, not the posterior of the latent world.

Grasping — demanding testbed. В appendix/secondary synthetic task желательно показать перенос RR-DKS на ещё один inverse problem (например, choose support location from occluded 2D shape), без robotics pipeline.

### Technical quality

Нужны полные proofs Propositions 1–4, точные assumptions для function-space kernel/restrictions, calibration diagnostics, matched-compute baselines и reproducible data generator.

### Empirical rigor

Главный reviewer attack: improvement вызван более крупной сетью, candidate proposer или synthetic prior. Поэтому mandatory candidate-controlled, matched-FLOPs и category-held-out experiments.

### Clarity

В paper нельзя смешивать три uncertainty:

- posterior ambiguity hidden geometry given noisy observation — core;
- sensor aleatoric noise — входит в conditional law;
- neural epistemic/OOD uncertainty — **не решена полностью** single implicit generator и должна называться limitation.

### Предварительная субъективная оценка

- идея без experiments: borderline, потому что direct grasp success prediction уже существует;
- с proofs + twins + strong sim only: plausible workshop/CoRL, всё ещё рискованно для ICLR;
- с joint-process ablation, category OOD, real robot и compute win над stochastic completion: **credible ICLR submission**;
- без demonstrated value joint correlations: reject/reframe.

---

## 13. Главные риски и способы фальсификации

1. **Certificate mismatch.** Static margin может плохо предсказывать small-lift success из-за friction/COM/control. Проверить rank correlation и calibrate только scalar margin; не превращать модель в whole-cycle predictor.
2. **Posterior collapse.** Common latent игнорируется. Twins, sample diversity, kernel score и conditional covariance metrics обнаруживают это напрямую.
3. **Training prior mismatch.** Model может быть уверенно неверной на unseen shapes. Category-held-out, procedural hidden perturbations, real objects и explicit risk flag обязательны.
4. **Candidate insufficiency.** Если partial proposer не предлагает grasp, selector бессилен. Report GT-certificate recall; добавить visible-contact shadow sweeps и differentiable local refinement.
5. **Gate leakage.** RGB semantics могут почти идентифицировать training object. Instance-disjoint splits, texture randomization и depth-only ablation.
6. **Setwise kernel scalability.** RBF concentration в больших $K$. Multiscale bandwidths, sparse contrast graph и random subsets; сравнить energy/variogram/conditional-CRPS alternatives.
7. **Process overkill for top-1.** Независимые quantiles могут быть столь же сильны. Это решающий ablation; отрицательный result уничтожает main process claim.
8. **Conservative score sacrifices easy grasps.** Report performance vs $\alpha,\lambda$, not one cherry-picked point; low-occlusion parity is acceptable.
9. **Novelty collision with very recent work.** Повторить literature search непосредственно перед submission, включая papers после July 2026, хотя ICLR policy считает самые свежие peer-reviewed works contemporaneous.

---

## 14. Минимальный implementation path

### Phase 0 — математический toy (1–2 недели)

- 2D parallel jaws и occlusion twins с exact posterior;
- deterministic mean, independent quantiles, common-latent RR-DKS;
- доказать recovery covariance/order и tail-regret gain.

**Stop**, если RR-DKS не восстанавливает joint law или regret selection не сильнее independent marginals.

### Phase 1 — 3D offline selector (3–5 недель)

- ACRONYM shapes, shelf + one obstacle renderer;
- fixed high-recall Contact-GraspNet/GraspGen candidate cache;
- analytic continuous certificate;
- VGCP $r=32,K=64,L=4$ training, $L=16$ inference;
- candidate-controlled sim evaluation.

### Phase 2 — strongest baselines (3–4 недели)

- matched deterministic/quantile/ensemble heads;
- stochastic completion + LCB;
- TARGO/ZeroGrasp where runnable;
- latency/memory and calibration tables.

### Phase 3 — real robot (4–6 недель)

- 3D-printed twins first, then household objects;
- randomized 300–600 attempts;
- exact failure logging but no causal failure-mode model;
- release scene geometry, RGB-D, candidates, certificates and result protocol.

### Phase 4 — paper hardening

- full proofs and assumptions;
- secondary non-robot inverse-decision toy for generality;
- preregister primary metrics/go-no-go threshold;
- claim only measured SOTA, not “potentially SOTA” as fact.

---

## 15. One-paragraph paper pitch

> Inferring an occluded object’s full geometry is a stronger problem than selecting a grasp. We formalize this mismatch through the decision quotient of latent shapes: two shapes are equivalent when they induce the same static certificate over all parallel-jaw actions. We learn the conditional law of this certificate function directly from one noisy RGB-D view, without decoding geometry. A visibility-gated neural process shares one latent hidden-shape hypothesis across grasp queries, while a strictly proper random-restriction kernel score fits both absolute certificate margins and local action-regret contrasts. The resulting process supports lower-tail and shape-wise regret selection at query cost rather than repeated shape completion. Controlled occlusion twins test posterior identifiability; synthetic shelf scenes and real wrist-camera experiments test whether decision-quotient inference is faster and more reliable than deterministic grasp scorers and stochastic completion baselines.

## 16. Suggested title / acronym alternatives

Primary:

**Beyond Shape Completion: Decision-Quotient Process Learning for Reliable Grasping under Occlusion**

Alternatives:

- **Learn the Grasp Consequences, Not the Hidden Shape**
- **VGCP: Visibility-Gated Certificate Processes for Occluded Grasping**
- **Action-Pushforward Inference for Decisions under Partial Observation**

Последний вариант лучше для broad ICLR framing; первый яснее для reviewers.
