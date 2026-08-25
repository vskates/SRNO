# FiRe: Filtration-Consistent Response Polytopes for Grasp Selection under Occlusion

**Статус:** исследовательская гипотеза и проект paper; литература проверена по состоянию на **25 августа 2026**.  
**Целевой venue:** ICLR, primary area — general machine learning / uncertainty and structured prediction; robotic grasping — главная реализация и проверка.  
**Ограничение novelty claim:** поиск не обнаружил работы с тем же learning target, objective и contractive architecture, но это не является доказательством абсолютной novelty. Перед submission нужен повторный systematic search и author-level review.

## 0. Итоговый выбор

Предлагается не восстанавливать скрытую форму объекта и не предсказывать один scalar grasp score. Новый объект обучения — **множество совместно возможных response-функций**: какие utilities дали бы все candidate grasps в каждом полном мире, совместимом с одним частичным наблюдением. Для конечного набора grasps это множество векторов в `[0,1]^N`; для непрерывного grasp space — множество функций `g -> utility`.

Ключевой переход:

> learn neither the hidden state nor its posterior density; learn the image of the observation-consistency set through the downstream utility operator.

Практическая модель **FiRe** (Filtration-Consistent Response Polytopes) предсказывает компактный convex polytope из `K` witness utility fields. По мере добавления наблюдаемой геометрии polytope может только сжиматься. Его вершины не обязаны быть реконструкциями или даже физически интерпретируемыми shapes: это support witnesses, достаточные для надёжного выбора действия.

Главные contributions, если гипотезы подтвердятся:

1. **Новый general-ML learning target:** coarsening-induced response polytope вместо point label, conditional mean, full hidden state или posterior над hidden state.
2. **Новый objective:** action-directed support-function matching, обучающий целое множество совместных utility vectors в метрике, непосредственно ограничивающей downstream robust regret.
3. **Новая architecture:** Contractive Witness Field, где successive evidence updates являются column-stochastic mixing предыдущих functional witnesses; вложенность uncertainty sets обеспечивается архитектурно.
4. **Новое измеримое знание:** decomposition ошибки на irreducible **Occlusion Ambiguity Tax** и avoidable model error; controlled Occlusion-Twin benchmark отделяет information-theoretic ambiguity от обычной generalization error.
5. **Grasping result:** надёжный выбор parallel-jaw grasp по одному noisy RGB-D при foreground occlusion, без полной реконструкции, RL и VLA.

Это сильнее, чем идея «ещё одна uncertainty-aware grasp model»: научный вопрос — **как обучать decision-relevant set-valued representations при coarsened observations так, чтобы uncertainty согласованно сжималась с информацией**.

## 1. Точная область задачи

На входе:

- одно RGB-D наблюдение wrist camera;
- target mask или target point labels считаются доступными; качество segmentation тестируется отдельно;
- foreground obstacle частично закрывает target;
- полка и obstacle наблюдаемы с шумом;
- cluttered-bin reasoning, active perception и multi-view accumulation не входят в core setting.

На выходе:

- ranking или selection одного 6-DoF parallel-jaw grasp `g = (t, R, w)`;
- опционально `abstain`, если ни один grasp не имеет достаточного certified lower utility;
- auxiliary candidate proposal head допустим, но core contribution — **selection under hidden grasp-relevant geometry**.

Utility не оценивает весь длинный цикл approach-to-lift. Она локальна:

\[
q(s,g)=\Pr_{\zeta}\{\text{collision-free local closure, stable contacts, retain object after a 1 cm lift}\},
\]

где `s` — полная target geometry и локальные физические параметры, а `ζ` — небольшие perturbations pose, friction и execution. Коллизии с наблюдаемой полкой/препятствием лучше сначала использовать как hard filter. Это не RL, не VLA и не causal failure-mode model.

## 2. Что уже занято литературой

### 2.1 Robotics ML

| Линия | Репрезентативные работы | Что уже сделано | Почему это не наша novelty |
|---|---|---|---|
| Direct partial-cloud grasp scoring | [Contact-GraspNet](https://arxiv.org/abs/2103.14127), [VGN](https://corlconf.github.io/corl2020/paper_359/), [GraspNet-1Billion](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html) | Dense grasp quality/pose prediction непосредственно из point cloud/TSDF | Обычно point estimate; скрытая conditional ambiguity не является обучаемым structured object |
| Implicit geometry + grasp | [GIGA](https://doi.org/10.15607/RSS.2021.XVII.024), [NeuGraspNet](https://www.roboticsproceedings.org/rss20/p046.pdf) | Joint/shared implicit geometric representation и grasp function; NeuGraspNet работает с single random view | Всё ещё опирается на geometric scene representation или rendering; нет множества mutually compatible utility fields |
| Local completion | [Local Occupancy-Enhanced Grasping](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09354.pdf) | Completion только grasp-related local region улучшает GraspNet metrics | Это более узкая реконструкция, но learning target остаётся occupancy |
| Explicit occlusion benchmark + completion | [TARGO / TARGO-Net](https://targo-benchmark.github.io/) | Single-view target grasping, balanced occlusion levels; completed target + scene reasoning; качество падает с occlusion | Закрывает obvious paper «occlusion benchmark + transformer + completion» |
| Large-scale joint reconstruction | [ZeroGrasp, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.html) | High-resolution reconstruction + grasp prediction; 11.3B grasp annotations; SOTA claim на GraspNet | Ещё один reconstruction-based метод будет трудно защитить как принципиально новый |
| Generative uncertainty | [FFHFlow, CoRL 2025](https://proceedings.mlr.press/v305/feng25a.html) | Flow likelihood, perceptual uncertainty и uncertainty-aware ranking для dexterous hands | Уже закрывает broad claim «generative uncertainty makes partial-observation grasping safer» |
| Similarity/retrieval | [SuperGrasp](https://arxiv.org/abs/2603.29254) | Single-view retrieval по superquadrics + evaluation/refinement | Retrieval of full-shape primitives — другой predict-then-transfer pipeline |

Особенно важно: TARGO непосредственно показывает unresolved degradation с ростом occlusion, но его решение — shape completion. ZeroGrasp сообщает сильный результат именно от reconstruction-aware grasping. Значит, чтобы претендовать на ICLR-level novelty, нельзя ограничиться «completion не нужна»: требуется новый learning problem и знание, объясняющее, **какая информация вместо completion достаточна для решений**.

### 2.2 General ML и decision making

| Соседняя идея | Ближайшая работа | Перекрытие | Оставшийся gap |
|---|---|---|---|
| Distribution over functions | [Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a.html), [Transformer Neural Processes](https://proceedings.mlr.press/v162/nguyen22b.html) | Условные stochastic functions и scalable function prediction | Likelihood/distribution fitting, не support recovery в action metric; нет information-order nesting |
| Multiple hypotheses | [Learning in an Uncertain World](https://openaccess.thecvf.com/content_iccv_2017/html/Rupprecht_Learning_in_an_ICCV_2017_paper.html) | Winner-takes-all multiple outputs вместо conditional mean | Independent hypotheses не образуют coherent action-response set и не дают robust-decision bound |
| Learned uncertainty sets | [Learning Decision-Focused Uncertainty Sets](https://arxiv.org/abs/2305.19225), [End-to-end Conditional Robust Optimization](https://proceedings.mlr.press/v244/chenreddy24a.html) | Data-driven/contextual uncertainty sets, downstream-aware optimization | Не изучают image consistency set, observation coarsening, response-function image и pathwise contraction with evidence |
| Robust regret DFL | [Robust Decision-Focused Learning via Worst-Case Regret](https://proceedings.mlr.press/v337/yamao26a.html) | Worst-case regret при measurement error и shift | Uncertainty задаётся вокруг coefficients/distributions; не обучается latent-world response set из partial observations |
| Risk-aware prediction sets | [Decision-Theoretic Foundations for Conformal Prediction](https://arxiv.org/abs/2502.02561), [action-conditional extension](https://arxiv.org/abs/2606.05551), [policy-coupled counterfactual sets](https://arxiv.org/abs/2607.02206) | Prediction sets и max-min decisions имеют сильное decision-theoretic обоснование | Предсказывают outcome/label sets с coverage; здесь нужен low-dimensional image высокоразмерной hidden geometry в joint utility-function space и consistency across observation refinements |
| Exact decision abstractions | [Decision Quotient, 2026](https://arxiv.org/abs/2603.14689) | Формализует optimizer quotient и сложность exact relevance certification | Теория exact coordinate hiding; не approximate set learning, не coarsening filtration, не response-polytope architecture/objective |
| Nested uncertainty | [Early-Exit Networks with Nested Prediction Sets](https://proceedings.mlr.press/v244/jazbec24a.html) | Показывает, что standard Bayesian/conformal UQ может быть inconsistent между exits | Nesting относится к computation exits и label sets, не к partial-order наблюдений и utility fields |

Следовательно, нельзя заявлять новизну для max-min rule, convex uncertainty sets, support functions или optimizer quotient по отдельности. Novel combination должна быть уже:

> **conditional response-set learning under an observation filtration, with action-directed support loss and architectural contraction.**

## 3. Проверенные и отброшенные направления

### 3.1 Full или local shape completion

Отброшено как main idea. TARGO-Net, ZeroGrasp, GIGA, NeuGraspNet и local occupancy уже дают плотное prior art. Улучшение decoder, voxelization или diffusion completion будет robotics/CV engineering, если нет отдельного general-ML result.

### 3.2 Posterior over completions + CVaR/minimax

Отброшено. Это дорогой high-dimensional latent variable, оптимизирующий геометрию, большая часть которой не меняет grasp ordering. FFHFlow уже соединяет partial-observation uncertainty и risk-aware ranking; decision-focused robust optimization уже изучает learned uncertainty sets. Кроме того, likelihood в shape space не обязан быть calibrated для редких, но decision-critical hidden contacts.

### 3.3 Независимые intervals/quantiles для каждого grasp

Отброшено как core. Marginals теряют корреляцию между действиями. Модель может одновременно считать возможными нижние/верхние значения разных grasps, хотя ни один физический hidden world не создаёт такую комбинацию. Это даёт «Frankenstein uncertainty» и слишком консервативный selection.

### 3.4 Deep ensemble или epistemic confidence

Отброшено. Disagreement параметров модели не тождественен irreducible ambiguity скрытой формы. При большом количестве данных ensemble может уверенно предсказывать conditional mean для genuinely ambiguous observation.

### 3.5 Только minimax-regret objective

Отброшено как самостоятельная novelty из-за robust DFL literature. Regret остаётся полезным decision functional и metric, но не определяет новый representation target.

### 3.6 Masked/invariant visible representation

Отброшено. Occlusion invariance может уничтожить именно то различие, которое должно расширить uncertainty. На паре одинаково видимых, но grasp-несовместимых shapes любой deterministic invariant embedding обязан collapse; проблема не в regularization, а в неидентифицируемости.

## 4. Новый general-ML problem: response-set learning under coarsening

### 4.1 Observation consistency

Пусть `s ∈ S` — полный hidden state. Наблюдение `x` задаёт набор допустимых sensor constraints. Вместо posterior density определим distribution-supported consistency set

\[
\mathcal C_\epsilon(x)=\{s\in \operatorname{supp}(P_{\rm train}):d_{\rm obs}(O(s),x)\le \epsilon\}.
\]

`ε` включает tolerances depth, calibration и segmentation. Это не множество всех геометрически мыслимых тел, а множество worlds, правдоподобных под обучающим shape distribution.

Создадим filtration наблюдений

\[
x_0\preceq x_1\preceq\dots\preceq x_L,
\]

где каждый следующий уровень добавляет ray/point constraints. Тогда по определению

\[
\mathcal C(x_{\ell+1})\subseteq \mathcal C(x_\ell).
\]

Это set-based, а не Bayesian statement: даже с bounded sensor noise добавление constraints реализуется пересечением и не может расширить consistency set.

### 4.2 Response image вместо hidden state

Для candidate grasps `G={g_1,...,g_N}` полный state создаёт joint response vector

\[
U_G(s)=[q(s,g_1),\dots,q(s,g_N)]\in[0,1]^N.
\]

Target uncertainty object:

\[
\mathcal Y_G(x)=\{U_G(s):s\in\mathcal C(x)\}.
\]

FiRe учит его closed convex hull

\[
\mathcal P_G(x)=\operatorname{cl\,conv}\mathcal Y_G(x).
\]

Convexification — не удобная hallucination. Для любого randomized one-shot decision `p ∈ Δ_N`

\[
\min_{u\in\mathcal Y_G(x)}p^\top u
=
\min_{u\in\mathcal P_G(x)}p^\top u.
\]

То есть convex hull lossless для worst-case linear utility. Он также сохраняет deterministic max-min и worst-case regret. Геометрически разные shapes с одинаковым `U_G` collapse автоматически; shape details, не влияющие ни на один candidate grasp, не требуют capacity.

Для continuous grasps модель реализует `K` функций `f_k(x,g)`; конечный vector возникает только при query. Полная SDF/occupancy никогда не декодируется.

### 4.3 Две величины, которые standard grasp success смешивает

**Robust Success Floor**

\[
F(x)=\max_n\min_{u\in\mathcal P_G(x)}u_n.
\]

Это лучший guaranteed utility среди candidates.

**Occlusion Ambiguity Tax**

\[
\operatorname{OAT}(x)
=\min_n\max_{u\in\mathcal P_G(x)}
\left[\max_j u_j-u_n\right].
\]

OAT — минимальный worst-case regret, неизбежный для любого deterministic selector, видящего только `x`. Если две hidden geometries дают одно RGB-D, но требуют несовместимых grasps, OAT положителен даже для бесконечно большой модели.

Для selector `a_θ(x)`:

\[
\operatorname{EAR}_\theta(x)
=\max_{u\in\mathcal P_G(x)}[\max_j u_j-u_{a_\theta(x)}]
-\operatorname{OAT}(x)\ge 0.
\]

**Excess Ambiguity Regret (EAR)** измеряет только avoidable model/optimization error. Это более научная diagnosis, чем raw grasp success: она отделяет «наблюдение принципиально не содержит ответа» от «модель плохо использовала доступную информацию».

## 5. Новый learning objective: Action-Directed Response Support Matching

### 5.1 Empirical target polytope

Для training observation `x_i^ℓ` строится ambiguity bag `B_i^ℓ` полных worlds, совместимых с ним. Physics/analytic labeling даёт matrix

\[
Y_i^\ell=[U_G(s_1),\dots,U_G(s_M)]^\top\in[0,1]^{M\times N}.
\]

Target polytope — `conv(rows(Y_i^ℓ))`. Model outputs `K` witness vectors/functions `V_i^ℓ` и polytope `hat P_i^ℓ=conv(rows(V_i^ℓ))`.

### 5.2 Support-function loss

Для compact convex set `P`, support function

\[
h_P(z)=\max_{u\in P}z^\top u.
\]

Для vertex set она вычисляется одним `max`. Under `ℓ∞` Hausdorff geometry:

\[
d_H^{\infty}(P,Q)
=\sup_{\lVert z\rVert_1\le1}|h_P(z)-h_Q(z)|.
\]

Поэтому основной objective:

\[
\mathcal L_{\rm RSM}
=\sum_{\ell=0}^{L}
\mathbb E_{z\sim\nu_G}
\rho\!\left(
\operatorname{LSE}_{k}(z^\top V_{ik}^{\ell})-
\operatorname{LSE}_{m}(z^\top Y_{im}^{\ell})
\right).
\]

`ρ` — Huber loss, LSE — smooth approximation maximum. Direction distribution `ν_G` не isotropic-only, а action-directed:

- `±e_n` — upper/lower utility каждого grasp;
- `(e_j-e_n)/2` — pairwise preference и minimax-regret directions;
- sparse Rademacher directions с `||z||_1=1` — joint correlations;
- hard directions, найденные projected ascent по текущему support discrepancy.

Это не обычный Chamfer/WTA loss. Support matching:

- permutation-invariant к witness slots;
- не штрафует redundant interior vertices;
- учит именно convex set, а не arbitrary ordering particles;
- имеет прямой downstream guarantee;
- сохраняет correlation across grasps, в отличие от independent quantiles.

Полный training objective:

\[
\begin{aligned}
\mathcal L_{\rm FiRe}
&=\mathcal L_{\rm RSM}\\
&\quad+\lambda_{\rm sel}\mathcal L_{\rm EAR}\\
&\quad+\lambda_{\rm prop}\mathcal L_{\rm proposal}.
\end{aligned}
\]

`L_EAR` — differentiable listwise surrogate, сопоставляющий predicted robust-regret ranking с empirical bag oracle. Это вспомогательный term; главный novelty test обязан показать, что `L_RSM` даёт выигрыш сам по себе.

## 6. Новая architecture: Contractive Witness Field

```text
noisy RGB-D + target/obstacle roles
              |
      sparse ray/point tokens
              |
  x0 ⊂ x1 ⊂ ... ⊂ xL  (evidence filtration)
              |
    equivariant set encoder
              |
 K base witness utility fields f^0_k(g)
              |
   A1       A2                AL
 f^1 = A1ᵀf^0 -> f^2 = A2ᵀf^1 -> ... -> f^L
              |
  nested response polytopes P0 ⊇ P1 ⊇ ... ⊇ PL
              |
 query candidate grasps -> max-min / minimax-regret selector
```

### 6.1 Sparse observation tokens

Для каждого sampled pixel/point:

- 3D point, RGB, estimated normal и depth confidence;
- camera ray direction;
- semantic role: target / obstacle / shelf;
- free-space segment до measured depth и unknown-shadow indicator после obstacle.

Это sparse evidence, не scene SDF. Point/ray encoder должен быть SE(3)-equivariant или использовать строго object-centric relative coordinates; bilateral jaw symmetry закладывается в grasp query representation.

### 6.2 Functional witnesses

Coarsest level `x_0` порождает `K` latent witness slots. Shared action decoder получает slot, relative grasp pose и local visible features:

\[
f_k^0(g)=\sigma(D_\theta(z_k^0,\phi(x_0,g)))\in[0,1].
\]

Один slot задаёт coherent function по **всем** grasps. Он не является независимым quantile для каждого action.

### 6.3 Constructive contraction

Новые evidence tokens не создают произвольные новые functions. Они предсказывают matrix

\[
A_\ell\in\mathbb R_+^{K\times K},\qquad
\mathbf 1^\top A_\ell=\mathbf 1^\top,
\]

то есть каждый column лежит на simplex. Update для любого continuous grasp query:

\[
f^{\ell}(g)=A_\ell^\top f^{\ell-1}(g).
\]

Каждый новый witness — convex combination прежних functions, поэтому **для всех grasps одновременно**

\[
\widehat{\mathcal P}(x_\ell)\subseteq
\widehat{\mathcal P}(x_{\ell-1}).
\]

Это exact architectural property, а не soft penalty. Standard uncertainty heads могут стать более широкими после дополнительной информации; здесь это невозможно по конструкции.

### 6.4 Selection

Primary reliable rule:

\[
\hat g_{\rm floor}=\arg\max_g\min_k f_k^L(g).
\]

Diagnostic/minimax-regret rule:

\[
\hat g_{\rm regret}=\arg\min_g\max_k
\left[\max_{g'}f_k^L(g')-f_k^L(g)\right].
\]

Для лабораторного reliability deployment предпочтителен max-min floor с abstention threshold. Minimax regret нужен для OAT/EAR и как secondary policy; он не должен незаметно подменять absolute success reliability.

### 6.5 Candidate generation

Core paper должен сначала изолировать selection:

- фиксированный high-recall candidate bank, одинаковый для всех methods;
- continuous FiRe field query для каждого candidate;
- observed collision hard filtering перед ranking.

Secondary end-to-end experiment может добавить lightweight equivariant proposal head. Нельзя позволить proposal recall скрыть качество нового objective: обязательно report `oracle@candidate-set` и ranking conditional on candidate recall.

## 7. Теоретические утверждения, которые реально можно доказать

### Proposition 1: response convexification is decision-lossless

Для любого `p ∈ Δ_N` минимум linear utility на `Y` равен минимуму на `conv(Y)`. Следовательно, hidden state reconstruction является избыточной для worst-case one-step linear decision, если response set известен.

### Proposition 2: information monotonicity

Если `x^- ≼ x^+`, то

\[
\mathcal P(x^+)\subseteq\mathcal P(x^-),\quad
F(x^+)\ge F(x^-),\quad
\operatorname{OAT}(x^+)\le\operatorname{OAT}(x^-),
\]

при фиксированном action set. Больше valid evidence не может ухудшить oracle robust floor и не может увеличить irreducible ambiguity. В направлении усиления occlusion inequalities обращаются.

### Proposition 3: support error gives selection bound

Если

\[
\sup_{\|z\|_1\le1}|h_{\widehat P}(z)-h_P(z)|\le\varepsilon,
\]

то coordinate robust floors отличаются не более чем на `ε`. Grasp, выбранный max-min на `hat P`, имеет true robust-floor suboptimality не более `2ε`.

Для regret важен scale. Если тот же `ε`-bound выполнен на всём `ℓ₁` unit ball, то `\|e_j-e_n\|_1=2`: ошибка worst-case regret каждого фиксированного action не больше `2ε`, а suboptimality action, выбранного по `hat P`, не больше `4ε`. Эквивалентно, при обучении normalized directions `(e_j-e_n)/2` сначала получается bound для половины regret, после rescaling — те же `2ε/4ε`. LSE в practical loss добавляет стандартную temperature-dependent approximation error; теорема относится к exact support maximum. Это делает support loss не просто surrogate «по интуиции», а objective с явным decision guarantee и явно указанной аппроксимационной поправкой.

### Proposition 4: OAT is an observation-level lower bound

Любой deterministic selector, зависящий только от `x`, несёт worst-case regret как минимум `OAT(x)`. Exact response polytope и minimax-regret selector достигают этой границы на finite candidate set. Таким образом, benchmark может отделять irreducible ambiguity от excess model error.

### Что нельзя заявлять без дополнительной работы

- `hat P` не имеет distribution-free conditional coverage автоматически.
- Empirical ambiguity bag не равен истинному support distribution.
- Architectural nesting не гарантирует, что true hidden world содержится в predicted polytope.
- Conformal calibration может дать marginal coverage для sampled utility vectors, но не simultaneous coverage всех compatible shapes и не conditional guarantee для каждого occlusion level.

Эти границы надо написать явно. Опциональный split-conformal radius по score `min_k ||U(s)-V_k(x)||∞` можно использовать как calibration baseline/layer, но не выдавать за core novelty.

## 8. Как сделать efficiently learnable targets без reconstruction at inference

### 8.1 Counterfactual Occlusion Bags

Для каждого full training object:

1. Render noisy RGB-D, target mask, foreground obstacle и shelf.
2. Создать master pool full states той же coarse category/scale.
3. Зарегистрировать кандидаты по **видимой** target geometry.
4. Оставить shapes, чьи rendered visible depth, silhouette, normals и RGB лежат в sensor tolerances исходного observation.
5. Разнообразить только hidden region bounded cage deformations/part grafts, не меняя ни одного visible ray; plausibility фильтровать category prior и mesh validity.
6. Всегда сохранять исходный full state.
7. Для nested observations фильтровать один master pool последовательно, чтобы `B^{ℓ+1} ⊆ B^ℓ` выполнялось буквально.

Основной benchmark должен использовать retrieval of real held-out meshes; procedural hidden grafts — controlled stress test. Иначе reviewer справедливо скажет, что результат определяется произвольным shape generator.

### 8.2 Utility labels

Для каждого `(state, grasp)`:

- быстрый analytic antipodal/force-closure filter;
- signed clearance в локальном gripper closing/swept volume, вычисляемый offline по full mesh;
- несколько short-horizon physics perturbations для empirical success probability;
- perturbations friction, object pose и jaw calibration;
- локальный 1 cm lift, не весь motion pipeline.

Storage — matrix `M × N` utilities на bag, а не dense SDF. Можно использовать low-rank compression utilities после offline labeling, но не до проверки effective rank.

### 8.3 Вычислительная сложность

При `K=16`, `N=256` output содержит 4096 scalar responses. Inference decoder complexity примерно `O(KNd)`; contraction — `O(LK²)`. Это существенно меньше dense 3D reconstruction и не зависит от voxel resolution. Реальное сравнение обязано report:

- parameters, FLOPs, peak GPU memory;
- latency encoder / candidate query / selection;
- storage и offline label cost;
- success per joule или success-latency Pareto.

## 9. Occlusion-Twin benchmark: необходимый эксперимент на существование проблемы

Обычные occlusion levels показывают degradation, но не доказывают conditional ambiguity. Нужны группы objects/scenes, которые дают одинаковое наблюдение в пределах sensor noise, но разные utility vectors.

### 9.1 Synthetic twins

- Одинаковая visible front shell и texture.
- Hidden variants: толщина, back-side concavity, скрытый flange/handle, asymmetric mass/contact surface, cavity в closing region.
- Foreground obstacle гарантирует invisibility различающей части.
- Pose, shelf и visible obstacle идентичны.
- Candidate set общий для всей twin group.

### 9.2 Physical twins

3D-print 12–24 families с identical visible face и сменными скрытыми backs/inserts. RGB texture стандартизировать. Для каждого family провести несколько placements и occlusion widths. Это маленький, но решающий real-world proof: point predictor обязан дать один ответ на одинаковый input, тогда как FiRe должен расширить response polytope, найти common robust grasp или abstain.

### 9.3 Новые metrics

- `OAT` и `EAR` по twin groups;
- robust success floor gap к bag oracle;
- support error на held-out directions;
- refinement violation rate;
- empirical coverage of the true utility vector;
- selective risk–coverage curve;
- standard grasp success and TARGO-style performance by occlusion band;
- candidate recall и conditional ranking accuracy отдельно.

## 10. Экспериментальная программа уровня ICLR

### RQ1. Существует ли отдельная conditional ambiguity problem?

Сравнить visually matched twins. Проверить, что direct BCE scorer и ensemble имеют низкий EAR только при маленьком OAT, а при высоком OAT остаются overconfident. Report mutual indistinguishability input pairs и oracle ambiguity.

### RQ2. Нужен ли joint response polytope?

Baselines:

1. scalar BCE/quality regression;
2. heteroscedastic Gaussian / beta head;
3. independent quantiles per grasp;
4. deep ensemble и MC dropout;
5. MHP/WTA utility particles;
6. Conditional/Transformer Neural Process over utility field;
7. learned box/ellipsoid uncertainty set;
8. decision-focused contextual robust-optimization head;
9. probabilistic shape completion -> `K` completions -> grasp utilities;
10. TARGO-Net/ZeroGrasp-style completion baseline, если код и лицензия позволяют;
11. empirical bag oracle и full-geometry oracle.

Ключевое сравнение: FiRe polytope против coordinate box с теми же marginal lower/upper bounds. Выигрыш докажет ценность cross-action coherence, а не просто conservative thresholding.

### RQ3. Важна ли exact filtration consistency?

Сравнить:

- unconstrained witness transformer;
- soft nesting penalty;
- post-hoc intersection;
- proposed column-stochastic contraction.

Измерять set error, violation rate, performance после случайного удаления/добавления points и monotonicity robust floor.

### RQ4. Нужна ли reconstruction?

Сравнить fixed compute/memory:

- local occupancy completion;
- full/sparse reconstruction;
- FiRe без geometry decoder;
- FiRe + auxiliary reconstruction head.

Если auxiliary reconstruction стабильно улучшает held-out EAR при том же compute, тезис «reconstruction unnecessary» следует ослабить до «explicit reconstruction output unnecessary».

### RQ5. Переносится ли эффект на real noisy shelf setup?

Стратифицировать trials по visibility, depth noise, target family и obstacle clearance. Основной statistic — paired difference на одинаковых scenes; confidence intervals bootstrap по object family, не по отдельным почти зависимым grasps.

### Минимальный масштаб

- 8k–12k training meshes из ACRONYM/Objaverse-LVIS-подобного источника с legal split;
- category-held-out и instance-held-out tests;
- 500+ synthetic twin families;
- `M=16–32` worlds в bag, `K=8/16/32` witnesses, `N=256–512` candidates;
- минимум 5 random seeds для central simulation comparisons;
- real trials, распределённые по object family, а не только cherry-picked household objects.

## 11. Обязательные ablations

1. `K = 1, 4, 8, 16, 32, 64`.
2. Только axis directions; axis + pairwise; + sparse random; + hard direction mining.
3. Support loss против Chamfer, Hausdorff-on-vertices и WTA.
4. Convex polytope против independent box и Gaussian covariance ellipsoid.
5. Coherent global mixing `A_ℓ` против action-wise mixing; последнее должно показать Frankenstein inconsistency.
6. Exact contraction против no contraction / soft regularization.
7. Retrieval-only bags против hidden graft augmentation.
8. Bag size и visible-match tolerance.
9. RGB-D против depth-only; ray tokens против point-only.
10. Max-min, minimax regret, conditional mean и calibrated lower-confidence selection.
11. С hard observed-collision filter и без него.
12. Ground-truth mask против noisy predicted mask.
13. Candidate oracle, fixed candidate bank и optional learned proposal head.

## 12. Falsifiable hypotheses и kill criteria

### H1: decision response sets имеют низкую effective complexity

Utility matrices ambiguity bags должны иметь низкий approximate vertex/covering complexity. Проверить singular spectrum, archetypal reconstruction error и support error versus `K`.

**Kill:** для приемлемого `ε` требуется `K > 64` или memory/latency сравнимы с sparse reconstruction.

### H2: joint structure важнее marginals

При одинаковых coordinate lower/upper errors FiRe должен снижать EAR и улучшать selective success относительно box baseline.

**Kill:** independent intervals статистически не хуже на twins и TARGO-style scenes.

### H3: contraction улучшает generalization

Exact nesting должна снизить support error на unseen occlusion levels и обеспечить zero architectural violations.

**Kill:** unconstrained model устойчиво лучше по EAR/success, а violations не коррелируют с failures.

### H4: ambiguity bags отражают реальные hidden variation

Model, trained on synthetic/retrieved bags, должен лучше ранжировать grasps на 3D-printed twins и real objects.

**Kill:** эффект исчезает вне того же generator/retrieval engine.

### H5: есть common robust actions

На значимой доле high-occlusion scenes robust floor bag oracle должен быть выше operational threshold.

**Kill:** почти все hard scenes имеют `F(x)≈0`; тогда правильный результат — abstention/impossibility benchmark, но не SOTA grasp selector.

### H6: objective, а не больший capacity, создаёт выигрыш

FiRe сравнивается с parameter/FLOP-matched MHP, CNP и ensemble.

**Kill:** преимущество исчезает после matching compute и candidate recall.

## 13. Честный novelty audit

### Что потенциально новое

- Learning the **response image of a consistency set** rather than a state/output distribution.
- Action-directed support-function regression of a conditional response polytope.
- Evidence-monotone architecture that contracts a polytope of whole utility functions through shared stochastic mixing.
- OAT/EAR decomposition and exact/noise-equivalent occlusion twins for one-step decision ambiguity.
- First application, насколько показал поиск, к single-view parallel-jaw grasp selection without shape reconstruction.

### Что не новое и не должно так называться

- convex hull, support function и Hausdorff identity;
- robust/max-min optimization;
- multiple hypotheses;
- learned uncertainty sets;
- decision-focused learning;
- conformal calibration;
- optimizer/decision quotient;
- SE(3)-equivariant point encoding;
- occlusion-aware grasping и shape completion.

### Наиболее опасные reviewer objections

1. **«Это contextual robust optimization в grasping».** Ответ должен опираться на controlled coarsening filtration, functional response image, support-target supervision и exact contraction; без сильных ablations objection победит.
2. **«Convex hull invents impossible worlds».** Для заявленных linear worst-case и regret functionals доказать losslessness; не распространять claim на arbitrary nonlinear risks.
3. **«Ambiguity bags arbitrary».** Retrieval-only held-out evaluation и physical twins обязательны.
4. **«Same result gives completion samples + robust scoring».** Сравнить с equal-compute completion sampling и показать меньший decision support error / latency.
5. **«Only a robotics application».** Добавить минимум два non-robotics coarsened-observation benchmarks: masked contextual decision toy с exact set и partial medical/portfolio-like tabular benchmark без причинных claims. Они должны проверять тот же objective и monotonicity, а не быть декоративными.
6. **«No statistical guarantee».** Чётко отделить deterministic set-approximation theorem от coverage. Опционально добавить conformal outer inflation, но не размывать main paper.

## 14. Почему это может быть ICLR paper

Официальный [ICLR 2027 Reviewer Guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines) просит оценивать technical correctness, rigor, reproducibility, novel findings и value/new knowledge, причём работа не обязана просто выигрывать established leaderboard. [Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers) явно включает general ML, uncertainty, structured prediction и robotics.

Проект соответствует этому только при следующей форме paper:

- сначала общий learning problem и controlled ambiguity example;
- затем propositions о convexification, information monotonicity, support-to-decision bound и irreducible tax;
- затем architecture, конструктивно удовлетворяющая новой структуре;
- затем synthetic exact-ground-truth benchmarks;
- только после этого крупный grasping result и real robot.

Одна робототехническая таблица success rate недостаточна. Напротив, новый benchmark + теория без убедительного real grasping также недостаточны. ICLR potential появляется в их соединении.

**Оценка до экспериментов:** novelty potential — высокий; technical risk — высокий; acceptance potential — medium/high только если H1–H4 проходят и method выигрывает у equal-compute completion/uncertainty baselines. Заявлять будущий SOTA сейчас научно нельзя.

## 15. Самый дешёвый de-risking sequence

### Gate A — exact toy, 2–3 дня

Сделать 2D hidden-shape toy с одинаковыми visible rays и конечными actions. Точные response polytopes известны enumeration. Проверить support loss, `2ε` bound, OAT/EAR и contraction.

### Gate B — mesh twins, 1–2 недели

100 procedural twin families, 64 candidates, analytic utilities. Сравнить BCE, intervals, WTA и FiRe. Не строить robot pipeline до проверки H1/H2.

### Gate C — retrieved ambiguity, 2–3 недели

Сделать visible-geometry retrieval на реальных mesh collections; измерить bag purity, diversity и effective `K`. Если response sets не compressible, остановить architecture line.

### Gate D — integration

Frozen candidate generator, TARGO-like scenes, physics labels, full baseline suite. Только после positive ranking result добавлять proposal head.

### Gate E — physical twins и shelf robot

Сначала repeatable 3D-printed twins, затем heterogeneous household objects. Failure taxonomy: candidate miss, visible collision error, hidden-geometry ambiguity, execution noise, segmentation.

## 16. Предлагаемая структура будущей статьи

**Title:** *Learning Response Polytopes under Coarsened Observations: Filtration-Consistent Grasp Selection without Shape Completion*

**Abstract skeleton:**

> Partial observations generally identify a set of latent worlds rather than a single state. Existing predictors either average their outcomes, reconstruct every latent detail, or attach marginal uncertainty to individual actions. We propose response-polytope learning: predicting the convex image of the observation-consistency set under a downstream utility operator. We introduce an action-directed support loss and a contractive witness architecture whose predicted uncertainty sets provably shrink as evidence is added. The support error upper-bounds robust decision suboptimality, while a new ambiguity tax separates irreducible information loss from model error. Instantiated for single-view parallel-jaw grasp selection, the method predicts coherent grasp-utility fields without decoding hidden geometry. Controlled occlusion twins, large-scale simulation, and real-robot experiments test whether this representation improves reliability, calibration, and efficiency over completion, ensemble, and decision-focused uncertainty baselines.

**Claim hierarchy:**

1. General theorem/objective.
2. Structural architecture.
3. New ambiguity measurement/benchmark.
4. Grasping SOTA or Pareto improvement only if actually observed.

## 17. Bottom line

Лучший найденный direction — не «догадаться о скрытой геометрии лучше», а **научиться представлять ровно множество downstream consequences, которое остаётся неидентифицированным после наблюдения**. Occlusion становится не data augmentation severity, а order over information sets. Это даёт новый learning target, objective, architecture, theorem и benchmark metric в одном coherent story.

Самая ценная проверяемая научная мысль:

> Under coarsening, reliable learning should be set-valued in response space and antitone in information; reconstructing the latent world is sufficient but generally not necessary.

Если polytope оказывается компактным и переносится на physical twins, это действительно substantial general-ML knowledge. Если нет, kill criteria быстро покажут, что идея либо требует completion, либо сама задача при сильной occlusion неразрешима без abstention/active sensing.

## Основные источники

- [TARGO: Benchmarking Target-driven Object Grasping under Occlusions](https://arxiv.org/abs/2407.06168)
- [TARGO project / IJCV 2026 version](https://targo-benchmark.github.io/)
- [ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping](https://openaccess.thecvf.com/content/CVPR2025/html/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.html)
- [Local Occupancy-Enhanced Object Grasping](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09354.pdf)
- [NeuGraspNet](https://www.roboticsproceedings.org/rss20/p046.pdf)
- [VGN](https://corlconf.github.io/corl2020/paper_359/)
- [Contact-GraspNet](https://arxiv.org/abs/2103.14127)
- [GraspNet-1Billion](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html)
- [FFHFlow](https://proceedings.mlr.press/v305/feng25a.html)
- [Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a.html)
- [Multiple Hypothesis Prediction](https://openaccess.thecvf.com/content_iccv_2017/html/Rupprecht_Learning_in_an_ICCV_2017_paper.html)
- [Early-Exit Neural Networks with Nested Prediction Sets](https://proceedings.mlr.press/v244/jazbec24a.html)
- [End-to-end Conditional Robust Optimization](https://proceedings.mlr.press/v244/chenreddy24a.html)
- [Learning Decision-Focused Uncertainty Sets](https://arxiv.org/abs/2305.19225)
- [Robust Decision-Focused Learning via Worst-Case Regret](https://proceedings.mlr.press/v337/yamao26a.html)
- [Decision-Theoretic Foundations for Conformal Prediction](https://arxiv.org/abs/2502.02561)
- [Conformal Risk-Averse Decisions with Action-Conditional Guarantees](https://arxiv.org/abs/2606.05551)
- [Prediction Sets for Counterfactual Decisions](https://arxiv.org/abs/2607.02206)
- [Decision Quotient: exact relevance certification](https://arxiv.org/abs/2603.14689)
- [ICLR 2027 Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines)
- [ICLR 2027 Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers)
