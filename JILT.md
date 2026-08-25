# Learn the interaction transform, not the object: JILT for grasping through occlusion

**Research proposal / independent search pass**

**Date of literature cut:** 25 August 2026

**Target venue:** ICLR 2027

**Status:** conditional go; the idea has a defensible new learning target and architecture, but its central sufficiency premise must be killed or validated before large-scale training

## 0. Executive decision

The selected direction is **JILT — Jaw-Interaction Line Transform**; its learnable implementation is **JILT-Net**.

The central proposal is not to reconstruct the hidden object, sample hidden shapes, predict a posterior over grasp outcomes, learn a random feasible set, or estimate another scalar confidence. Instead, for each parallel-jaw candidate, the full training geometry is reduced to a compact **gripper-interaction line measure**: how target occupancy and oriented surface mass are distributed along a small, hardware-fixed bundle of jaw-closing lines and short body-clearance lines. The model predicts a few Fourier moments of those positive measures directly from one occluded RGB-D observation.

The new general-ML question is:

> In a partially observed inverse decision problem, can one predict a non-invertible, task-specific forward transform of the latent state directly, while forcing all predicted transform queries to remain in the range of physically valid positive measures, and thereby make better decisions than either latent reconstruction or unconstrained end-task regression?

The grasping instance is unusually clean because a parallel jaw does not interact with every voxel of an object. It interrogates the object along a small family of closing and clearance lines at a finite pad and sensor resolution. A complete object can therefore supervise a compact transform without becoming the inference output.

The new learning objective is the **Random-Bundle Cone Score (RBCS)**. It scores truncated Toeplitz moment matrices on random gripper-line bundles and enforces exact refinement and reparameterization identities. A small mechanics decoder is trained separately, with stop-gradient at the predicted sketch, to test decision sufficiency without changing the estimand. Unlike an arbitrary vector regression head, every predicted moment sequence is projected into a truncated trigonometric moment cone, so it corresponds to at least one non-negative measure on the physical jaw interval.

The new architecture is JILT:

1. an analytic visible-surface anchor and free-space support lift from the segmented RGB-D point set;
2. a sparse neural operator on oriented line bundles, not on a dense 3-D volume;
3. several unrolled learned-update / additive anchor-and-support consistency / moment-cone-projection blocks;
4. a deterministic finite-pad mechanics decoder and a frozen candidate generator.

The output is a conditional mean **interaction sketch**, not a conditional distribution. There is no shared latent shape sample, CVaR, random set, capacity functional, response polytope, Blackwell/tower regularizer, observation-fiber lower/upper set, metamer group, or ray-to-jaw attention process. That is the deliberate non-intersection with the occlusion ideas written on 25 August 2026.

The strongest indirect case for potential superiority is a conjunction of four established observations:

- TARGO shows that current 6-DoF grasp systems lose about 20 percentage points or more at extreme occlusion, while targeted occlusion-aware training helps materially.
- TOSC reports that restricting completion to contact-relevant regions improves both completion and downstream grasp metrics over generic completion, supporting the premise that task geometry is smaller than complete geometry.
- direct measurement-domain inference in computational imaging can match or beat reconstruction-first pipelines while using less computation, supporting the broader “do not invert what the task does not need” principle;
- range/data consistency and physically valid moment constraints reduce the hypothesis space and cannot increase Euclidean projection error to a valid target.

None of these results proves that JILT will outperform TARGO-Net, ZeroGrasp, NeuGraspNet, GraspGen, Contact-GraspNet, or a strong direct critic. The proposal becomes an ICLR paper only if the interaction sketch is empirically sufficient at small bandwidth and if cone/range structure gives an independent gain at equal encoder, labels, candidates, and compute.

## 1. Exact task contract

### 1.1 Included

- one rigid target object on a shelf;
- one wrist-camera RGB-D frame;
- noisy and non-uniform target point cloud;
- ordinary self-occlusion plus at most one foreground blocker or shelf lip;
- a supplied target mask or target ID plus an upstream segmenter;
- a fixed parallel-jaw gripper;
- selection of a terminal 6-DoF grasp pose and commanded opening;
- fixed closure controller and a millimetric or centimetric vertical lift;
- full geometry and offline grasp/contact oracle available during synthetic training only;
- unseen test instances and a deliberately shifted object-family split.

### 1.2 Excluded

- reinforcement learning;
- VLA or language-conditioned policies;
- next-best view, obstacle removal, pushing, or active exploration;
- tactile or motor-current feedback;
- full approach-to-lift trajectory feasibility;
- long-horizon manipulation;
- causal decomposition into failure modes;
- dense scene SDF, completed mesh, completed point cloud, occupancy volume, or neural radiance field as model output;
- sampling multiple completed objects at inference;
- generic clutter reasoning.

Observed shelf and foreground-obstacle geometry is handled by a deterministic collision gate over the terminal gripper and a short pre-contact retraction. The learned object concerns only hidden target geometry relevant to local closure and the tiny lift.

### 1.3 Two evaluation regimes must remain separate

**Information-only regime.** The foreground blocker is present during imaging and removed without moving target or camera before execution. This isolates the value of hidden target geometry.

**Combined shelf regime.** The blocker remains during execution. Its observed geometry is used by the same deterministic collision filter for every method.

The main scientific claim must be established in the information-only regime. Otherwise an apparent gain can come merely from a better obstacle-collision detector.

## 2. Explicit non-intersection with today's Markdown ideas

The following exclusion map was built from the Markdown files modified on 25 August 2026. It is based on estimand, objective, and architecture, not names.

| Existing direction | Its scientific object | Why JILT is different |
|---|---|---|
| FiGO / OC-GOP | posterior over a grasp-outcome function plus Blackwell/tower coherence | JILT predicts one conditional mean line-transform sketch; there is no function posterior or filtration loss |
| FiberGrasp | necessary and possible grasp sets over an observation fiber | JILT predicts neither lower/upper action sets nor worst-case membership |
| Grasp Metamers / MetaContact | sensor-equivalent shape groups and a likelihood-weighted paired-contact mixture | JILT uses ordinary per-shape transform labels; it constructs no exact metamer groups and no contact mixture |
| DQPL / CRFSP | posterior over a random feasible-action field | JILT's latent target lives on physical interaction lines, not in action-feasibility function space |
| FELLAS / CEN | random closed feasible set, Choquet capacity queries, proper event scoring | RBCS is a squared score for conditional mean moment matrices, not a hit/inclusion probability |
| grasp-certificate process / RJPN | stochastic certificate process, energy/variogram score, ray-jaw incidence attention | JILT is deterministic, uses no shared noise and no attention from camera rays to jaw rays; it performs constrained completion in oriented-line transform space |
| AvoGrasp | avoidance functional of a random set of failed actions | no action-set avoidance event or robust pose packet is predicted |
| FiRe | response polytope, support functions, filtration-consistent contraction, witness field | no utility polytope, support-function loss, or information-contractive witness architecture is used |
| CapGrasp | capacity operator for Boolean gripper events | JILT predicts positive geometric line measures and their moments, not probabilities of unions or avoidance events |

JILT also does **not** reopen the earlier rejected Fourier embedding of the successful-grasp set. That method embeds a measure over actions. JILT takes Fourier moments of target occupancy along a few physical lines inside a particular gripper frame. Orientation harmonics of a grasp-quality field and frequency moments of a jaw-line measure are different domains and different targets.

The closest older repository idea is AcqGrasp, which estimates smooth visible-surface integrals invariantly across RGB-D sampling laws. JILT is not acquisition correction: it predicts unobserved line measures under external occlusion and imposes a truncated moment/range geometry across candidate queries. Nevertheless, AcqGrasp is an important internal red-team threat because both methods use compact physical probes. The paper must show that the gain is from hidden-measure completion plus range constraints, not merely better quadrature of visible points.

### 2.1 Late workspace collision audit

During the final audit, `FELLAS.md` acquired an addendum that **compares** the prototype under its earlier working name “LiMON” with FiRe and CEN. That addendum does not make the method a FiRe/CEN component: it explicitly keeps it as an independent deterministic competitor and recommends not mixing the methods. The present report adopts the collision-free name JILT and incorporates the addendum's valid LoCoMo criticism below. The scientific separation is by estimand, loss, and architecture, not by renaming.

## 3. What the current literature occupies

### 3.1 External occlusion is already a measured problem

[TARGO / TARGO-Net](https://targo-benchmark.github.io/) directly studies target-driven grasping from one RGB-D view over target visibility levels from 0 to 0.9. Its public results show a large degradation for VGN, GIGA, EdgeGraspNet, and ICGNet at extreme occlusion; TARGO-Net reduces but does not eliminate the drop. TARGO-Net segments and completes the target, then fuses completed target and scene features with a transformer. Its single-scene augmentation using occlusion-induced failure labels improves baselines by at least about 5% and TARGO-Net by about 10%.

Consequences:

- “occlusion hurts grasping” is not a novel finding;
- a new benchmark with only severity bins is insufficient;
- TARGO-Net is a mandatory direct baseline;
- the new work must isolate informational occlusion from physical obstruction;
- equal-candidate reranking and equal-compute comparisons are required.

### 3.2 Full or local reconstruction is occupied

- [ZeroGrasp, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.pdf) jointly predicts an octree reconstruction, normals, SDF-related geometry, grasp quantities, and explicit inter/self-occlusion fields.
- [PSSNet](https://proceedings.mlr.press/v155/saund21a.html) generates diverse plausible shapes from ambiguous depth and demonstrates robot grasping under occlusion.
- [TOSC, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38053) completes only potential contact regions and reports 16.17% better grasp displacement and 55.26% better Chamfer distance than its selected state of the art.
- [NeuGraspNet, RSS 2024](https://roboticsproceedings.org/rss20/p046.html) reinterprets grasping as neural surface rendering from an implicit feature volume and couples local surface rendering with grasp prediction.

Therefore “complete less geometry” is not the novelty. JILT must be presented as **range-constrained forward-transform prediction** and must demonstrate that its finite sketch is non-invertible and smaller than even contact-region reconstruction.

### 3.3 Direct grasp and contact representations are strong

- [Contact-GraspNet](https://arxiv.org/abs/2103.14127) roots a grasp in an observed contact point, reduces pose dimensionality, and achieves strong real grasping without full CAD reconstruction.
- [SpaHybGen](https://www.nature.com/articles/s42256-026-01292-y) predicts hardware-agnostic spatial contact features from noisy depth and optimizes grasps for seven hands, reporting 94.3–98.0% success in semi-cluttered scenes.
- [GraspGen](https://arxiv.org/abs/2507.13097) uses a diffusion transformer plus an on-generator discriminator, trains with more than 53 million simulated grasps, and reports state-of-the-art FetchBench performance.
- [LoCoMo](https://doi.org/10.1109/IROS.2018.8594226) matches zero-moment-shift descriptors of observed local object and gripper surface patches and ranks parallel-jaw grasps without learning or hidden-shape inference.
- [SpectGRASP](https://arxiv.org/abs/2107.12492) uses spherical-harmonic correlation of gripper and object normal signals for efficient parallel-jaw proposal generation, then uses LoCoMo for ranking.

These works prevent claims such as “first contact representation,” “first moment-based contact representation,” “first spectral grasp method,” or “first reconstruction-free grasp method.” JILT is not learned LoCoMo: LoCoMo summarizes and matches **visible local surface curvature**, whereas JILT predicts **unobserved positive occupancy/normal-flux measures along action-indexed closing tubes**, supervises a frequency sequence rather than a zero-moment-shift descriptor, and projects it into a supported moment cone. Even so, “LoCoMo plus hidden completion plus PSD” is a credible hostile reviewer summary. JILT's narrow claim is therefore the conditional line-measure target, random-bundle elicitation with exact bundle identities, and the additive support/range-consistent architecture—not moments alone.

### 3.4 General inverse-problem precedents

The broad idea of tailoring inference to a downstream task is established. [Task-adapted reconstruction](https://doi.org/10.1088/1361-6420/ac28ec) jointly optimizes reconstruction and post-processing. Direct measurement-domain classification without reconstruction also exists in CT and optical imaging; [Machine Friendly Machine Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC6820559/) explicitly argues that all reconstructed information is already in raw measurements and demonstrates sinogram-space interpretation.

Range and data consistency are also established:

- [Data-consistent neural networks for nonlinear inverse problems](https://doi.org/10.3934/ipi.2022037) construct networks whose output reproduces observed data under the forward operator.
- [Deep Null Space Learning](https://arxiv.org/abs/1806.06137) builds data preservation into inverse-problem networks and supplies convergence analysis.
- Helgason–Ludwig conditions characterize physically consistent Radon data and have been used for projection completion; see the [Fourier-domain consistency formulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC5541827/).
- [Unsupervised learning from incomplete measurements](https://arxiv.org/abs/2201.12151) proves that a fixed incomplete operator cannot generally identify missing information, while multiple operators can make the signal model learnable.

Thus “data consistency” and “measurement-domain learning” are not individually novel. The novelty hypothesis is their new synthesis around **action-indexed positive line measures**, a **truncated moment cone**, and **random gripper-bundle supervision** without reconstructing the source object.

## 4. Sequential search and rejection log

### Branch A — shape-prior group DRO

Idea: train a worst-group grasp selector across object-family priors so hidden-geometry predictions remain safe when the shape prior changes.

Rejection: group DRO and robust decision-focused learning already provide the general machinery. For binary one-shot success, the method reduces to a conservative scalar critic, often chooses visible-only grasps, and does not explain a new structure of the occlusion problem. It also approaches the earlier minimax-regret and necessary-set branches.

### Branch B — task-aware partial functional maps

Idea: align an occluded target to complete training archetypes with a mechanics-weighted partial functional map, then transfer grasp annotations without completing geometry.

Rejection: [functional-map grasp transfer](https://arxiv.org/abs/2203.00776) already transfers grasps across deformable shapes. Extending it with partial correspondence and an archetype bank is a plausible robotics system but looks like a composition of shape correspondence and grasp transfer. It also depends strongly on topology/category and has no clean advantage for novel rigid household shapes.

### Branch C — physically adversarial blocker curriculum

Idea: optimize a differentiable 3-D foreground blocker to maximize top-one grasp regret, then train the selector in a minimax visibility game.

Rejection as the main paper: TARGO already shows the benefit of occlusion-induced augmentation, while adversarial physical masks and sensor-aware 3-D adversarial augmentation are established. The contribution would be a better training distribution. Worse, a worst-case blocker can remove all decision-relevant evidence, so the objective can become an impossible-information game rather than better inference of hidden geometry.

### Branch D — occlusion-semigroup equivariance

Idea: represent nested masks as non-invertible semigroup elements and force the representation to transform through learned idempotent contractions instead of becoming invariant.

Rejection: this is too close to the existing Blackwell/filtration and contractive-witness directions in today's files. It would violate the non-intersection requirement even if the algebraic language changed.

### Branch E — redundant evidence / min-cut certificates

Idea: select grasps supported by several disjoint visible evidence paths, using a differentiable max-flow/min-cut layer so one connected blocker cannot erase all support.

Rejection: this helps only when grasp-relevant information is redundantly visible. The hard case explicitly has a hidden antipodal region. It also approaches evidence coresets and proof-carrying grasps, whose coverage is poor exactly where external occlusion matters.

### Branch F — direct interaction-transform prediction

Initial idea: use a Radon or X-ray transform as a compact substitute for object completion.

First correction: a full dense X-ray transform is injective and is therefore merely another complete shape representation. It would violate the no-reconstruction requirement.

Second correction: a parallel jaw needs only a finite, query-local bundle of lines at pad resolution and only a small number of low-order frequency moments along each line. The union of queried bundles is deliberately incomplete and has a large null space. It cannot recover the full object.

Third correction: independent predicted line features can be physically contradictory. The correct learning space is the cone of truncated moment sequences of positive measures, with exact identities between reparameterized and refined lines.

This corrected branch survives as JILT.

## 5. General ML formulation: task-transform learning under coarsening

Let $Z\in\mathcal Z$ be a latent physical state, $X=M_\omega(Z)+\varepsilon$ a coarsened/noisy observation, $a\in\mathcal A$ an action, and $U(Z,a)$ its downstream utility.

Reconstruction-first learning estimates $\hat Z(X)$ and evaluates $U(\hat Z,a)$. Direct decision learning estimates $q(X,a)$ with no structured intermediate object. JILT studies an intermediate regime.

Assume there is an action-indexed forward transform

$$
\Phi_a:\mathcal Z\rightarrow\mathcal K_a
$$

and a small decoder $D_a$ such that, on the declared physical regime,

$$
U(Z,a)\approx D_a(\Phi_a(Z)).
$$

The family $\Phi(Z)=\{\Phi_a(Z):a\in\mathcal A\}$ is highly redundant and obeys known range constraints

$$
\Phi(Z)\in\mathfrak R\subsetneq\prod_a\mathcal K_a.
$$

The proposed problem is to learn

$$
F_\theta:X\mapsto \widehat\Phi_X
$$

subject to $\widehat\Phi_X\in\mathfrak R_Q$, where $\mathfrak R_Q$ is a finite-query relaxation of the true transform range, while never constructing $\hat Z$.

This is not generic end-to-end task-adapted reconstruction because the latent state is never decoded. It is not a scalar critic because the intermediate transform has explicit algebra, exact consistency tests, a controllable bandwidth, and transfer across candidate sets.

The broad scientific hypothesis is:

> Predicting a constrained, non-invertible forward transform can dominate both source reconstruction and unconstrained decision regression when the downstream interaction is low-bandwidth but the latent state is high-dimensional and partially unidentifiable.

## 6. Grasp Interaction Line Measure

### 6.1 Geometry and action

Let $S\subset\mathbb R^3$ be the closed rigid target solid with boundary $\partial S$. A grasp is

$$
g=(R_g,t_g,w)\in (SE(3)/C_2)\times [w_{\min},w_{\max}],
$$

where $C_2$ quotients the exchange of the two identical jaws. In the gripper frame let

- $e_c$: closing axis;
- $e_a$: approach axis;
- $e_h=e_c\times e_a$: pad-height axis.

The hardware defines a fixed line bundle $\mathcal L(g)$:

- a $3\times3$ or $4\times4$ grid of lines parallel to $e_c$ through the two pad footprints;
- a few lines through finger-body and palm clearance regions;
- optional lines along a short pre-contact retraction, not the whole arm path.

The bundle size is hardware-fixed and does not grow with scene resolution.

### 6.2 Positive measures along closing lines

For a canonical line

$$
\ell_{g,j}(s)=t_g+R_g(\xi_j+s e_c),
\qquad s\in I=[-W/2,W/2],
$$

define a soft tube kernel $k_\sigma$ with radius tied to pad resolution. The occupancy line measure is

$$
d\mu^{\mathrm{occ}}_{S,g,j}(s)
=
\left[
\int_{e_c^\perp}
k_\sigma(y-\xi_j)
\mathbf 1_S\!\left(t_g+R_g(y+s e_c)\right)dy
\right]ds.
$$

It is a finite non-negative measure on the physical aperture interval $I$.

For contact orientation, define positive left/right normal-flux measures on the surface:

$$
d\mu^{\pm}_{S,g,j}(s)
=
\int_{\partial S}
k_\sigma(P_\perp R_g^\top(x-t_g)-\xi_j)
[\pm n(x)^\top R_g e_c]_+
\delta\!\left(s-e_c^\top R_g^\top(x-t_g)\right)dA(x).
$$

These measures preserve only pad-scale mass and orientation evidence along the closing line. They do not retain texture, backside geometry outside the interaction tubes, or a globally renderable object.

### 6.3 Truncated Fourier moments

For frequencies $\omega_m=2\pi m/W$, $m=-M,\ldots,M$, define

$$
z^{c}_{S,g,j,m}
=
\int_I e^{-i\omega_m s}\,d\mu^{c}_{S,g,j}(s),
\qquad
c\in\{\mathrm{occ},+,-\}.
$$

The per-line target is a small complex vector $z_{-M:M}$, with $M=4,8,16$ tested explicitly. The zero-frequency term is total tube mass. Low frequencies encode center, spread, and coarse multi-interval structure; higher frequencies recover sharper entry/exit structure but are less stable under depth noise.

This representation connects naturally to the finite trigonometric moment problem. For every non-negative measure,

$$
z_{-m}=\overline{z_m},
$$

and the Toeplitz moment matrix

$$
T_M(z)_{pq}=z_{p-q},
\qquad p,q=0,\ldots,M,
$$

is positive semidefinite. With an additional interval-localizing matrix, the representing measure is constrained to the known physical aperture $I$.

An arbitrary neural vector does not satisfy these conditions. JILT does by construction.

### 6.4 What is observed, free, and censored

The RGB-D target mask supplies a visible surface point measure. Camera free space supplies inequality information before each depth return. A foreground-obstacle return censors the target ray behind the obstacle; it is not evidence of empty target space.

JILT uses these signals as follows:

- visible target surface contributes an analytic, non-negative anchor measure to the normal-flux channels;
- the occupancy channel receives no fictitious observed volume: it is predicted subject to support constraints, while an optional calibrated surface-shell anchor is kept separate;
- known free space removes impossible support intervals from every residual measure;
- observed obstacle points are never imputed as target and go only to the deterministic collision gate;
- censored intervals remain learnable unknowns conditioned on the global visible target.

Crucially, the observed surface contribution is not treated as a known coefficient of the complete transform. Every Fourier coefficient mixes visible and hidden mass. The consistent parameterization is additive:

$$
\widehat z^c=z^c_{\mathrm{anchor}}+\widehat z^c_{\mathrm{res}},
\qquad
\widehat\mu^c_{\mathrm{res}}\ge 0,
\qquad
\mathrm{supp}\widehat\mu^c_{\mathrm{res}}\subseteq I^c_{\mathrm{admissible}}.
$$

For occupancy, $z^{\mathrm{occ}}_{\mathrm{anchor}}=0$ in the conservative main model. For normal flux, the anchor is the splatted visible surface measure. This preserves observed evidence without pretending that RGB-D observes hidden volume.

Unlike RJPN, the network does not cross-attend each jaw primitive to individual camera rays. The camera observation is first lifted to sparse coefficients and constraints in the same oriented-line coordinate system; the neural operator then acts within that transform space.

### 6.5 Why this is not full reconstruction

Let $Q(X)=\{g_1,\ldots,g_K\}$ be the frozen candidate bank and let

$$
\Phi_Q(S)=\{z_{S,g_k,j,m}:k\le K,j\le J,|m|\le M\}.
$$

For finite $K,J,M$, $\Phi_Q$ has an infinite-dimensional null space: changes to $S$ outside all interaction tubes leave every coefficient unchanged. Even within a tube, distinct high-frequency occupancy patterns share the same truncated moments. Therefore $\Phi_Q$ is non-injective by design.

The paper must demonstrate this empirically:

- train a strong reconstruction probe from JILT sketches;
- show poor full-shape Chamfer/occupancy recovery relative to a completion latent;
- simultaneously show high grasp-margin recovery;
- report sketch dimension and queried spatial support.

If a dense candidate bank and large $M$ make full-shape recovery easy, the claimed non-reconstruction advantage has disappeared.

### 6.6 Critical conditional-mean limitation

The mean interaction sketch is not automatically sufficient for Bayes-optimal physical success. For a nonlinear mechanics decoder,

$$
D_g\!\left(\mathbb E[Z_g\mid X]\right)
\ne
\mathbb E[D_g(Z_g)\mid X]
$$

in general. A posterior method can retain distinctions that a conditional mean destroys. JILT deliberately does not solve this by adding a stochastic latent, because that would return to today's posterior/process family.

The project therefore makes a narrower, falsifiable structural hypothesis: after finite pad integration, execution smoothing, and restriction to a local short-lift score, the low-order interaction sketch is either approximately sufficient or supports an accurately learned deterministic Bayes-score decoder. This must be tested on repeated coarsened-observation groups, not assumed from per-shape reconstruction accuracy.

Three outcomes are possible:

1. a linear decoder of the conditional mean works — the strongest and cleanest JILT result;
2. a small nonlinear decoder works — still useful, but only an empirical sufficiency result;
3. posterior sampling materially wins on ambiguous groups — JILT is the wrong main method.

The paper must report which regime the data supports. It may not call a conditional mean a “belief” or imply lower-tail reliability.

## 7. New learning objective: Random-Bundle Cone Score

### 7.1 Training item

One training item contains

$$
(S,X,G,\mathcal P, Z),
$$

where

- $S$ is available only to the offline label generator;
- $X$ is an occluded noisy RGB-D observation;
- $G=(g_1,\ldots,g_B)$ is a random bundle of candidate grasps;
- $\mathcal P$ contains random pad-line refinement relations;
- $Z=\Phi_G(S)$ contains exact or high-resolution numerical moment targets.

Grasp bundles mix:

- 40% candidates near the exact oracle decision boundary;
- 25% visible-contact proposals;
- 25% candidates whose interaction tubes cross the target's occlusion cone;
- 10% clear negatives.

Sampling probabilities are logged and reused by every baseline. Otherwise a better query distribution can masquerade as a better objective.

### 7.2 Cone score

Let $\widehat z_\theta(X,g,j,c)$ be the predicted moment sequence after projection to the feasible truncated-moment cone $\mathcal K_{M,I}$. The base score is

$$
\mathcal L_{\mathrm{cone}}
=
\mathbb E
\left[
\sum_{g\in G}\sum_{j,c}
\left\|
W_{g,j,c}^{1/2}
\left(
T_M(\widehat z)-T_M(z)
\right)
W_{g,j,c}^{1/2}
\right\|_F^2
\right].
$$

Weights emphasize frequencies resolvable at the measured sensor/pad scale and candidates close to a mechanics boundary. They must be fixed from training/calibration data, not tuned on test success.

Squared Toeplitz-matrix error is a weighted squared error on the moments. Its population minimizer is the conditional mean moment matrix. Because the PSD moment cone is convex, the conditional mean of valid moment matrices remains valid. RBCS therefore elicits a coherent conditional mean interaction sketch without inventing a posterior.

### 7.3 Projection residual

The learned update produces an unconstrained sequence $\widetilde z$; the architecture applies

$$
\widehat z=\Pi_{\mathcal K_{M,I}}(\widetilde z).
$$

Training includes

$$
\mathcal L_{\mathrm{proj}}
=
\|\widetilde z-\widehat z\|_2^2,
$$

so the learned operator approaches the cone rather than relying on a large corrective projection at every layer. A PSD Toeplitz projection can be implemented by alternating affine Toeplitz projection, eigenvalue clipping, Hermitian symmetry, and an aperture localizing LMI. Exact differentiability is not required at the first prototype; implicit differentiation or a fixed number of differentiable projection steps is sufficient.

### 7.4 Refinement consistency

If a pad tube is partitioned into children with kernels summing to the parent kernel, the measures and all their moments add exactly:

$$
z_{\mathrm{parent}}=\sum_{r=1}^{R}z_{\mathrm{child},r}.
$$

For random refinement tree $\mathcal P$,

$$
\mathcal L_{\mathrm{ref}}
=
\sum_{(p,\mathrm{ch}(p))\in\mathcal P}
\left\|
\widehat z_p-
\sum_{r\in\mathrm{ch}(p)}\widehat z_r
\right\|_2^2.
$$

This prevents a coarse pad from predicting high occupancy while every sub-pad predicts empty space.

### 7.5 Reparameterization consistency

The same physical line can appear in multiple nearby candidates or with reversed jaw orientation. Canonical line hashing during training supplies equivalence pairs. For line reversal,

$$
z_m(-e_c)=e^{-i\omega_m\Delta}\overline{z_m(e_c)},
$$

where $\Delta$ is the known origin shift. Let $P_{\ell\to\ell'}$ denote this analytic phase/conjugation map. Then

$$
\mathcal L_{\mathrm{same}}
=
\sum_{\ell\sim\ell'}
\|\widehat z_{\ell'}-P_{\ell\to\ell'}\widehat z_\ell\|_2^2.
$$

This is a transform identity, not pairwise equality of grasp scores under different occlusion levels.

### 7.6 Decision term

A separately trained differentiable decoder $D_\psi$ maps the line sketches of one candidate to a local mechanics score:

$$
\widehat q_g=D_\psi(\mathrm{sg}(\widehat Z_g)),
$$

where $\mathrm{sg}$ is stop-gradient in the clean two-stage formulation. This keeps the moment operator an estimator of the conditional mean rather than allowing decision gradients to deform the purported moment prediction. Joint fine-tuning without stop-gradient is allowed only as a separately named ablation; in that version the conditional-mean elicitation theorem no longer describes the final network.

The first version should use a small monotone MLP plus analytic features:

- estimated left/right entry and exit depth;
- pad overlap mass;
- opposing normal flux;
- finger/body clearance mass;
- commanded width slack.

The decoder does not model the full approach, causal failure types, object dynamics, or long lift. It is trained against the same standardized local oracle used by all methods.

A listwise decision loss is

$$
\mathcal L_{\mathrm{dec}}
=
-\sum_{g\in G}
\pi_g^\star
\log
\frac{\exp(\widehat q_g/\tau_q)}
{\sum_{h\in G}\exp(\widehat q_h/\tau_q)},
\qquad
\pi_g^\star
=
\frac{\exp(q_g^\star/\tau_y)}
{\sum_h\exp(q_h^\star/\tau_y)}.
$$

The decision term is deliberately secondary. If an unconstrained direct critic with the same encoder and this listwise loss matches JILT, the transform contribution is falsified.

### 7.7 Complete objective

$$
\boxed{
\mathcal L_{\mathrm{RBCS}}
=
\mathcal L_{\mathrm{cone}}
+\lambda_p\mathcal L_{\mathrm{proj}}
+\lambda_r\mathcal L_{\mathrm{ref}}
+\lambda_s\mathcal L_{\mathrm{same}}
}
$$

The decoder is trained in a second optimization

$$
\min_\psi\mathcal L_{\mathrm{dec}}
\left(D_\psi(\mathrm{sg}(\widehat Z)),q^\star\right).
$$

An implementation may alternate the two optimizers, but $\mathcal L_{\mathrm{dec}}$ must not update the moment operator in the theorem-bearing model.

What is new is not squared loss, PSD projection, or listwise ranking in isolation. The proposed learning object is a random action-indexed family of truncated positive-measure moments, and RBCS elicits it while testing exact range identities across gripper queries. The separately trained decoder tests whether this object is decision-sufficient.

### 7.8 Exact scope of the range claim

The finite constraint set guarantees that every line/channel has a non-negative supported representing measure and that explicitly linked refinement/reparameterization queries agree. It does **not** guarantee that all measures in a large arbitrary bundle are projections of one common three-dimensional solid. Enforcing that global condition would approach inverse rendering or object reconstruction and is neither claimed nor required here.

Accordingly, the paper should call $\mathfrak R_Q$ a **finite-bundle moment-consistency relaxation**, not “the exact 3-D transform range.” A stronger cross-line realizability constraint is allowed only if it remains non-injective, cheap, and independently ablated.

## 8. New architecture: JILT

### 8.1 Overview

JILT alternates learned completion in line space with nonlearned physical projections:

$$
X
\xrightarrow{\text{analytic lift}}
Z^{(0)}_{\mathrm{obs}},M_{\mathrm{known}}
\xrightarrow{K\text{ operator blocks}}
\widehat Z
\xrightarrow{D_\psi}
\widehat q(G).
$$

No block outputs a voxel grid, point completion, mesh, SDF, object latent sample, random action set, or utility posterior.

### 8.2 Sparse observation encoder

The input encoder receives:

- visible target points with normals and RGB/depth features;
- camera pose and intrinsics;
- target/occluder/background mark;
- source-pixel footprint and depth confidence if available;
- shelf plane and target crop bounds.

A lightweight SE(3)-equivariant point encoder produces a global target code and point features. This encoder is not claimed as novel and must be shared with direct-critic and local-completion ablations.

### 8.3 Analytic visible-measure lift

For every queried physical line, NUFT-style kernel accumulation computes the contribution of visible target surface to the normal-flux moments. Known-free ray segments are intersected analytically with each line tube and produce a binary or soft support mask over the aperture.

The lift yields:

- visible normal-flux anchor moments $z^\pm_{\mathrm{anchor}}$;
- an occupancy support constraint, not fabricated visible-volume moments;
- intervals excluded from residual-measure support by camera free space;
- censored intervals behind target self-occlusion or the foreground blocker;
- confidence weights from point footprint and depth noise.

This is not a learned jaw-ray attention module. It is a fixed linear transform from measured marked surface samples into partial line-moment data.

### 8.4 Oriented-Line Operator block

Each token represents a physical line tube, not an RGB-D pixel or point. Its coordinates are

$$
(u,p,\rho),
\qquad
u\in\mathbb{RP}^2,
\quad p\in u^\perp,
$$

where $u$ is unoriented closing direction, $p$ the closest point of the line to the target-frame origin, and $\rho$ the tube/pad scale.

The learned update is a sparse neural operator

$$
\widetilde Z^{(k+1)}
=
Z^{(k)}
+\mathcal O_{\theta_k}
\left(
Z^{(k)},
E_X,
\Gamma_Q
\right),
$$

where $\Gamma_Q$ is a graph with three analytic edge types:

1. same-line / reversed-line equivalence;
2. parent-child pad refinement;
3. nearby parallel lines whose tubes overlap.

Messages use relative Plücker/incidence invariants between **predicted line tokens**, not cross-attention to camera rays. The global target code supplies the learned shape prior.

The operator is permutation-equivariant over the queried candidate set. Adding a candidate creates additional line tokens but does not change the coordinate meaning of existing tokens.

### 8.5 Additive data- and support-consistency layer

The learned state is the residual moment sequence, not the total sequence. After each learned update, project the residual onto the positive-measure cone supported only on intervals not ruled out by calibrated free space:

$$
Z_{\mathrm{res}}^{(k+1/3)}
=
\Pi_{\mathcal K_{M,I_{\mathrm{admissible}}}}
\!\left(\widetilde Z_{\mathrm{res}}^{(k+1)}\right),
\qquad
Z^{(k+1/3)}
=Z_{\mathrm{anchor}}+Z_{\mathrm{res}}^{(k+1/3)}.
$$

The support cone is implemented with interval-localizing constraints; for a union of admissible intervals, use a sum of interval-supported residual measures. Sensor uncertainty expands the admissible intervals rather than asserting exact empty space. The network cannot subtract visible target flux because the learned residual is non-negative, and it cannot put residual target mass into calibrated free space. No complete-transform coefficient is incorrectly treated as directly observed.

### 8.6 Moment-cone projection layer

For every channel and line, construct $T_M(z)$, then alternate:

1. Hermitian/Toeplitz averaging;
2. PSD eigenvalue clipping;
3. frequency-selective aperture support projection;
4. restoration of hard observed constraints;
5. a small number of Dykstra or Douglas–Rachford iterations.

This gives $Z^{(k+2/3)}\in\mathcal K_{M,I}$. Parent-child and same-line affine constraints are then projected jointly over each small bundle to obtain $Z^{(k+1)}$.

The projection is cheap because each moment matrix is only $5\times5$, $9\times9$, or $17\times17$, and bundles are embarrassingly parallel.

### 8.7 Decoder and selection

The decoder converts valid moments to pad-scale features. It can use:

- a stable maximum-entropy measure consistent with the moments;
- a small frequency-selective Vandermonde decomposition;
- or moments directly, which is preferable if reconstruction of even a 1-D profile adds no value.

The main ablation must compare all three. Calling the first two “1-D reconstruction” is acceptable; the prohibited operation is full object reconstruction. The primary model should use moments directly if it matches performance.

Candidate score:

$$
\widehat q_g
=
D_\psi
\left(
\widehat Z_g,
w_g,
\text{visible obstacle clearance}(g)
\right).
$$

Selection is ordinary

$$
\widehat g=\arg\max_{g\in Q(X):\,c_{\mathrm{obs}}(g)>0}\widehat q_g.
$$

There is no risk functional or abstention novelty claim. Selective prediction can be reported only as an optional calibrated operating point shared by all baselines.

### 8.8 Candidate generator

To isolate the proposed contribution, use one frozen candidate generator for all rankers:

1. visible-surface proposals from Contact-GraspNet/GSNet-style geometry;
2. a small set of coarse target-frame proposals that cross the occlusion cone;
3. the same local refinement budget for every evaluator.

End-to-end generation is a later experiment, not required to validate RBCS or JILT.

### 8.9 Complexity target

For $B=128$ candidates, $J=16$ pad/body lines, $M=8$, and $K=4$ unrolled blocks:

- moment storage is on the order of $B\times J\times(2M+1)$, not a $128^3$ or octree volume;
- cone projections act on matrices of size $M+1=9$;
- visible lift is a sparse segmented reduction over points near queried tubes;
- the observation encoder is cached once;
- no diffusion, mesh extraction, marching cubes, or simulator rollout is used at inference.

The paper should target less than one third of the latency and peak memory of an equal-accuracy stochastic/full-completion baseline, while staying within 1.5 times the latency of the shared direct critic.

## 9. Theory package

All statements below are theorem targets, not established results of this proposal.

### Proposition 1 — cone validity

For a finite Hermitian moment sequence $z_{-M:M}$, PSD of its Toeplitz matrix plus the interval-localizing constraint is necessary and sufficient for existence of a non-negative representing measure supported on the declared jaw interval, under the selected trigonometric K-moment formulation.

Consequence: every JILT line output has at least one physically non-negative 1-D occupancy interpretation. An unconstrained vector head does not.

### Proposition 2 — conditional-mean closure

Let $Z\mid X=x$ be a random valid moment matrix with finite second moment. Under squared Frobenius RBCS, the population minimizer is

$$
\widehat T^*(x)=\mathbb E[T(Z)\mid X=x].
$$

Because the PSD cone and affine Toeplitz/refinement constraints are convex, $\widehat T^*(x)$ remains feasible.

This establishes coherence only for the conditional mean sketch. It does not recover or claim the conditional law of hidden shapes.

### Proposition 3 — projection cannot worsen valid-target error

If $\mathfrak C$ is the closed convex finite-bundle constraint set and the true target $Z\in\mathfrak C$, then Euclidean projection satisfies

$$
\|\Pi_{\mathfrak C}(\widetilde Z)-Z\|_2
\le
\|\widetilde Z-Z\|_2.
$$

This is the cleanest indirect mathematical reason the range layer can help: for the same unconstrained prediction, projection cannot increase squared error to a valid label. It says nothing by itself about neural optimization or grasp success.

### Proposition 4 — finite-moment approximation for simple line sections

Assume that, at pad resolution, occupancy along each queried closing line is a union of at most $K_0$ intervals separated by at least $\delta$, and depth noise is below a declared scale. A finite set of Fourier samples or moments can identify/approximate the interval endpoints with an error bound depending on $M,K_0,\delta$, and noise.

This proposition connects moment bandwidth to actual contact geometry. It must be stated only for the explicit simple-section class. Household handles and deep concavities violate it and become stress tests.

### Proposition 5 — decoder stability

If the local mechanics decoder $D_g$ is $L_D$-Lipschitz on the feasible moment set, then

$$
|D_g(\widehat Z_g)-D_g(Z_g)|
\le
L_D\|\widehat Z_g-Z_g\|.
$$

If the oracle best candidate has margin

$$
\gamma
=
q_{g^*}-\max_{g\ne g^*}q_g,
$$

then uniform score error below $\gamma/2$ preserves the selected action. Combining this with the previous proposition gives a bandwidth/noise condition for correct selection.

### Proposition 6 — deliberate non-injectivity

For any finite queried bundle union $U_Q$, construct two solids $S_1,S_2$ that agree inside $U_Q$ but differ on a positive-volume subset outside $U_Q$. Then

$$
\Phi_Q(S_1)=\Phi_Q(S_2)
$$

while $S_1\ne S_2$. With truncated moments, additional within-tube non-identifiability exists.

This theorem prevents the paper from quietly claiming both “not reconstruction” and “all geometry is recoverable.”

### Proposition 7 — random-bundle identification on the queried domain

If the candidate-bundle design has full support on a compact deployment query domain, the model is continuous in line coordinates, and population RBCS is minimized, then the conditional mean moment function is identified almost everywhere under the bundle design and everywhere by continuity.

This is an identification result for the transform mean, not for a shape, feasible set, or stochastic process.

## 10. Why JILT could outperform current approaches

### 10.1 Against full completion

Full completion spends capacity and compute on geometry outside all queried interaction tubes. More importantly, a small global Chamfer error can coexist with a catastrophic error at one hidden contact. JILT allocates every supervised output dimension to pad/contact/clearance scale.

TOSC supplies direct indirect evidence for task restriction: contact-region completion improves both its contact-relevant geometry and downstream grasp generation relative to generic completion. JILT takes the restriction one step further, from regions to non-invertible measurements, but must prove that this extra compression does not remove essential mechanics.

### 10.2 Against direct scalar critics

A scalar critic can fit incompatible local evidence without any cross-query physical test. JILT shares line tokens between nearby candidates, enforces additive pad refinements, preserves visible data, and makes each frequency sequence a valid positive measure.

The projection theorem guarantees no larger label-space squared error after exact projection. This is not a guarantee of lower grasp regret, but it provides a concrete mechanism rather than a generic “physics helps” slogan.

### 10.3 Against stochastic completion or action posterior methods

For one-shot expected local quality, a coherent conditional mean transform may be enough; no sampling is required. JILT therefore avoids multiple shape decodes, multiple grasp evaluations, energy-score ensembles, and tail estimation.

This is also a risk: if ambiguity and lower-tail behavior are essential, stochastic methods should win. The experiments must include ambiguity-stratified results and cannot hide this failure by evaluating only average success.

### 10.4 Against local contact completion

TOSC predicts contact-region geometry that is still spatially renderable. JILT predicts only a finite set of line moments at current candidate/pad scale. A successful result would establish a new point on the information-compute frontier: less than local geometry, more structure than a score.

### 10.5 Against purely observed-contact methods

Contact-GraspNet and related methods exploit visible contacts efficiently. Under a front camera, however, the opposing parallel-jaw contact is frequently hidden even without an external blocker. JILT is supervised to predict the hidden contribution along the closing line, so it can retain candidates that visible-only methods must reject.

### 10.6 Expected superiority claim, stated honestly

The defensible pre-experiment hypothesis is:

> At equal candidate recall and encoder budget, a range-constrained conditional line-moment operator will improve high-occlusion top-one success and hidden-contact calibration over direct critics, while matching task-focused/full completion at substantially lower latency and memory.

“Will improve” is a hypothesis. It becomes a paper claim only after confidence intervals, equal-compute controls, and real execution.

## 11. Experimental program

### 11.1 Gate 0 — oracle sketch sufficiency before any neural model

This is the decisive first experiment.

For complete meshes and a fixed candidate bank:

1. compute exact local grasp margin $q_S(g)$;
2. compute JILT moments at $M\in\{2,4,8,16,32\}$ and pad grids $2^2,3^2,4^2,6^2$;
3. train only the small decoder $D_\psi$ with full moments;
4. compare against a local voxel/contact crop of equal byte size;
5. measure score $R^2$, top-one regret, sign accuracy near the boundary, and hidden-contact failure recall;
6. stratify by convexity, handles/holes, number of line intervals, pad size, and execution noise.

Go criterion:

- $M\le8$, at most $4\times4$ pad lines, and less than 256 real scalars per candidate retain at least 95% of the oracle top-one performance of the local high-resolution crop;
- sign accuracy within the physically important margin band is at least 90%;
- failures on multi-interval sections are localized and predictable.

If this gate fails, JILT is dead. Do not compensate by increasing bandwidth until the transform becomes a hidden voxel reconstruction.

### 11.2 Gate 0b — conditional-mean decision sufficiency

Gate 0 tests whether the sketch preserves mechanics for a known full object. It does **not** test whether the conditional mean sketch survives observational ambiguity. A second no-large-network experiment is mandatory.

1. Construct small groups of different complete shapes rendered to the same quantized/noisy partial RGB-D observation within a calibrated tolerance. These groups are an evaluation device only; JILT is not trained with a metamer/fiber objective.
2. For every common candidate, compute the group mean sketch $\bar Z_g$, the mean oracle utility $\bar q_g$, and the utility of the decoder applied to the mean sketch $D_\psi(\bar Z_g)$.
3. Compare the action selected from $D_\psi(\bar Z_g)$ with the Bayes action from $\bar q_g$.
4. Compare linear, small nonlinear, direct scalar, and stochastic-completion decoders.
5. Stratify by within-group hidden-contact disagreement.

Go criterion:

- mean-sketch selection retains at least 95% of the Bayes value on moderate-ambiguity groups;
- the loss relative to stochastic completion is below 2 percentage points at equal candidate bank;
- failures rise smoothly with a declared ambiguity statistic rather than appearing unpredictably.

If this gate fails, the conditional mean is not the right decision object. The project should stop or explicitly reopen a posterior direction, which would violate the present non-intersection requirement and therefore constitute a different project.

### 11.3 Gate 1 — cone and range value without occlusion inference

Given incomplete/noisy moment labels directly, compare:

- unconstrained MLP completion;
- PSD moment cone only;
- additive anchor/support consistency only;
- refinement/same-line constraints only;
- full unrolled projection architecture.

This synthetic experiment tests whether constraint projection improves moment error and mechanics regret independently of the RGB-D encoder.

Go criterion: full projection reduces boundary-band decision regret by at least 20% relative to the same learned update without projections and never produces negative-mass/invalid moment sequences.

### 11.4 Gate 2 — observation-to-sketch learning

Render single targets on a shelf with one ray-consistent foreground blocker. Training variables:

- object mesh and pose;
- camera pose within the wrist calibration envelope;
- blocker shape, depth, and connected image footprint;
- target visibility from 0.2 to 1.0;
- depth noise, quantization, missing returns, and edge artifacts;
- segmentation boundary corruption.

Split by object instance and shape family. Never split different views of one mesh across train and test.

### 11.5 Benchmarks

1. **TARGO-Synthetic** for direct comparison to the strongest external-occlusion benchmark.
2. **TARGO-Real** where compatible annotations are available.
3. A controlled **ShelfLine-OCC** set with full meshes and explicit information-only renders.
4. A physical shelf set of 30–50 unseen household objects, including bottles, boxes, bowls, mugs, tools, handled objects, thin objects, and deliberately non-line-convex shapes.

### 11.6 Required baselines

- TARGO-Net;
- ZeroGrasp;
- NeuGraspNet;
- Contact-GraspNet or GSNet-style direct predictor;
- GraspGen if its input/output contract can be matched;
- deterministic local contact completion;
- PSSNet or another stochastic full-shape completion plus equal-budget grasp evaluation;
- same JILT encoder with a direct scalar BCE/regression head;
- same encoder with unconstrained moment regression;
- same architecture with moments randomly mixed across frequencies, to test whether transform algebra matters;
- LoCoMo and SpectGRASP-style visible contact descriptors on the same candidate bank;
- visible-only analytic contact baseline;
- AcqGrasp-style visible probe baseline if its implementation is available.

### 11.7 Fairness controls

- one frozen candidate bank for the principal evaluation;
- identical training meshes, observations, and physical labels;
- equal encoder parameters;
- both equal-label and equal-wall-clock training curves;
- latency and peak memory on the same GPU;
- no completion baseline is penalized by an artificially weak grasp planner;
- no JILT result can use target mesh information at inference;
- hyperparameters selected without physical test outcomes.

### 11.8 Metrics

Primary:

- top-one simulated and physical short-lift success;
- top-one oracle regret on a common candidate bank;
- success versus target visibility;
- worst visibility-bin success for visibility $\le0.35$;
- hidden-contact subset success;
- latency, peak memory, and energy if measurable.

Transform diagnostics:

- weighted moment RMSE;
- invalid Toeplitz/localizing matrix rate;
- visible-anchor subtraction and free-support violation;
- refinement and same-line residual;
- boundary-band mechanics sign accuracy;
- moment bandwidth versus performance.

Non-reconstruction diagnostics:

- full-shape probe Chamfer/IoU from the sketch;
- fraction of object volume covered by queried tubes;
- sketch dimension versus local/full representation dimension;
- examples of different full shapes with the same finite sketch but equal queried mechanics.

### 11.9 Ablations

1. no cone projection;
2. no additive anchor/support consistency;
3. no refinement graph;
4. no same-line canonicalization;
5. no global target code;
6. no visible analytic lift;
7. $M=2,4,8,16,32$;
8. pad grid resolution;
9. moment-direct versus maximum-entropy versus Vandermonde decoder;
10. RGB off / depth only;
11. occluder marks off;
12. learned line operator replaced by independent query MLP;
13. full RBCS replaced by direct listwise loss;
14. candidate bundle distribution changed at test;
15. exact versus approximate cone projection iterations.

### 11.10 Real-robot protocol

- calibrate camera, gripper pose repeatability, pad footprint, and depth noise before model selection;
- 30–50 held-out objects, at least 5 placements, 3 blocker positions, and repeated execution;
- preregister forced-choice and optional abstention operating points;
- randomize method order;
- report Wilson or bootstrap confidence intervals and paired tests by scene;
- record target mask, candidate bank, selected grasp, moment validity, collision gate, and outcome;
- separate perception failure, deterministic collision rejection, and physical grasp failure only for diagnosis, not as a multi-head causal target.

## 12. Falsification and kill criteria

### Kill 1 — the sketch is not sufficient

If Gate 0 requires $M>16$, more than a $4\times4$ line grid, or a near-dense candidate cover to match local geometry, the method has lost its compact advantage.

### Kill 2 — cone projection is decorative

If unconstrained moment regression has the same moment error, boundary regret, and real success, remove the cone claim. Without an independent range-structure gain, the paper becomes a new feature representation only.

### Kill 3 — direct critic is enough

If the shared-encoder direct critic matches JILT on high-occlusion and hidden-contact subsets, the structured target is not justified.

### Kill 4 — ambiguity requires a posterior

If stochastic completion or a proper stochastic action-process model consistently wins on ambiguous views at acceptable compute, a conditional mean sketch is the wrong estimand. Do not conceal this by changing the evaluation to easy shapes.

### Kill 5 — no efficiency advantage

If JILT approaches full/local completion latency or memory at matched success, “no reconstruction” has no operational value.

### Kill 6 — range constraints fight real sensor error

If exact visible/free-space projections become inconsistent under ordinary RealSense noise or segmentation errors, soften the calibrated constraints. If the softened system loses its gain, stop.

### Kill 7 — only the candidate generator improves

If gains disappear under a fixed common candidate bank, JILT is not the cause.

### Kill 8 — only simulation improves

If the gain is absent on paired real shelf scenes, the learned hidden line prior did not transfer.

### Kill 9 — the method reconstructs the object implicitly

If a lightweight probe reconstructs the full object nearly as well as a completion latent, reduce queries/bandwidth. If performance then collapses, the scientific claim is false: full geometry was necessary.

### Kill 10 — nearest prior owns the idea

Before implementation freeze, perform citation chasing from TOSC, NeuGraspNet, SpaHybGen, SpectGRASP, task-adapted inverse problems, sinogram completion, trigonometric-moment networks, and any paper citing them after this literature cut. Stop or rename claims if an existing work already predicts action-local X-ray/moment data with range-constrained learning.

## 13. Adversarial novelty audit

### 13.1 Strongest reviewer summary against the paper

> “This is learned hidden LoCoMo/TOSC expressed with Fourier features and a PSD layer. LoCoMo already uses local contact moments, TOSC already completes contact regions, SpectGRASP already uses spectral correlation, inverse-problem networks already enforce data consistency, and a direct critic is simpler.”

The paper survives this objection only if experiments establish all of the following:

1. the finite line sketch is measurably non-invertible and much smaller than a contact-region representation;
2. hidden-measure prediction materially outperforms a visible-only LoCoMo/SpectGRASP-style descriptor under matched candidates;
3. the same sketch and decoder transfer across candidate banks/resolutions;
4. cone/range projection improves high-occlusion decision regret over an unconstrained head;
5. the gain remains after sharing encoder, labels, candidate generator, and listwise loss;
6. the method matches or beats task/full completion with substantially less inference cost.

### 13.2 Exact novelty sentence

The narrow defensible sentence is:

> We introduce direct prediction of gripper-indexed truncated Fourier moments of positive interaction-line measures from coarsened 3-D observations, together with a random-bundle cone score and an unrolled data/range-consistent line operator, enabling parallel-jaw grasp selection without decoding either the latent object or a stochastic action-space object.

Do not claim:

- first spectral grasping;
- first task-aware geometry;
- first reconstruction-free grasping;
- first measurement-domain learning;
- first data-consistent network;
- guaranteed grasp safety;
- recovered hidden geometry;
- uncertainty quantification.

### 13.3 Why this can be general ML

The transferable contribution is the middle layer between reconstruction and scalar prediction:

$$
\text{partial measurement}
\rightarrow
\text{constrained non-invertible task transform}
\rightarrow
\text{decision}.
$$

Other potential applications include collision queries from sparse scans, non-destructive-testing decisions from incomplete projections, tool-surface interaction, and direct measurement-domain diagnosis. The paper should demonstrate at least one small non-robotic synthetic inverse-decision task where range-constrained task-transform prediction beats source reconstruction and scalar regression. This is important for ICLR breadth and should not be a cosmetic appendix.

## 14. ICLR 2027 audit

The [ICLR 2027 reviewer guide](https://iclr.cc/Conferences/2027/ReviewerGuidelines) asks whether a paper addresses a specific question, is well motivated in the literature, supports its claims rigorously, and contributes significant new knowledge or value; state-of-the-art performance is explicitly not required. The [call for papers](https://www.iclr.cc/Conferences/2027/CallForPapers) asks for ambitious, complete “slow science.”

### Specific question

Clear: can a range-constrained, non-invertible interaction transform replace both hidden-shape reconstruction and unstructured score regression under occlusion?

### Motivation and placement

Strong if the paper directly compares to TARGO, TOSC, ZeroGrasp, NeuGraspNet, direct contact models, task-adapted inverse problems, and measurement-domain inference. Weak if it discusses only grasping or only tomography.

### Support for claims

Requires:

- exact Gate 0 sufficiency curve;
- cone/range ablations;
- equal-resource comparisons;
- real paired scenes;
- non-reconstruction probe;
- proofs for validity, projection, non-injectivity, and selection stability;
- an independent general-ML toy/application.

### Significant new knowledge

Potentially strong: the paper can establish when direct constrained task transforms are a better learning object than either inverse reconstruction or end-task prediction. This is broader than a new grasp module.

### Current acceptance estimate

- **Idea alone:** weak reject / borderline; too many components are individually known.
- **With positive Gates 0, 0b, 1, and 2 plus clean theory and a decisive LoCoMo control:** borderline accept.
- **With real high-occlusion gain, equal-compute completion parity, and general inverse-decision transfer:** plausible accept.
- **Without a range-constraint ablation or non-reconstruction test:** reject.

## 15. Claims and preregistered thresholds

### Claims allowed before experiments

- JILT defines a finite non-injective line-moment target.
- Cone projection guarantees finite moment validity under the declared formulation.
- The architecture does not output full geometry or an action posterior.
- TARGO and related work establish the practical occlusion gap.
- Existing work indirectly supports task restriction, measurement-domain inference, and data/range consistency.

### Claims allowed only after evidence

- superior high-occlusion success;
- equal or better accuracy than completion at lower compute;
- better sample efficiency;
- better real transfer;
- moment representation is sufficient at $M\le8$;
- cone structure, rather than encoder capacity, creates the gain.

### Suggested go thresholds

- at least 5 percentage points absolute gain over the strongest shared-candidate direct baseline in visibility $\le0.35$, with a paired 95% interval excluding zero;
- parity or better than the strongest completion baseline with at most one third of its latency and peak memory;
- at least 20% reduction in boundary-band decision regret from the full projection stack versus unconstrained moment regression;
- less than 2% observed/free-space constraint violation after measured sensor corruption;
- full-shape probe substantially worse than contact/local mechanics prediction;
- real-robot gain of at least 5 points on the preregistered hidden-contact subset.

These are project management thresholds, not universal definitions of significance.

## 16. Minimum implementation roadmap

### Phase 0 — no-learning oracle study, 1–2 weeks

- implement exact line/tube moment labels from meshes;
- cache grasp margins and moment bandwidths;
- run Gate 0 across object families;
- run Gate 0b on small calibrated coarsened-observation groups;
- stop immediately if compact sufficiency fails.

### Phase 1 — moment cone and decoder, 1 week

- implement Hermitian Toeplitz construction;
- implement PSD plus interval support projection;
- unit-test non-negative measure recovery;
- train the small mechanics decoder on full sketches.

### Phase 2 — synthetic missing-transform task, 1–2 weeks

- mask line intervals directly;
- train independent MLP and unrolled projection operator;
- test range/refinement gains without RGB-D confounds.

### Phase 3 — RGB-D lift and JILT, 3–5 weeks

- integrate target/occluder segmentation;
- implement visible/free/censored transform lift;
- train shared encoder and line operator;
- compare to direct critic and local completion.

### Phase 4 — TARGO and full baseline suite, 3–5 weeks

- adapt common candidates and labels;
- run equal-compute scaling curves;
- complete novelty citation chase.

### Phase 5 — real shelf experiment, 2–4 weeks

- calibrate pad/sensor scale;
- collect preregistered paired blocker scenes;
- execute forced-choice short lifts;
- publish all failure strata and confidence intervals.

## 17. Scientific unit tests

1. Zero measure gives an all-zero PSD Toeplitz matrix.
2. A single interval produces conjugate-symmetric moments and a PSD matrix.
3. Reversing a line matches the analytic conjugation/phase rule.
4. Splitting a tube kernel makes parent moments equal the sum of children.
5. Cone projection never increases Euclidean error to a valid synthetic target.
6. Additive consistency preserves visible anchor mass and residual support projection excludes calibrated free space.
7. Free-space constraints forbid recovered mass in calibrated empty intervals.
8. Candidate permutation leaves every output unchanged up to the same permutation.
9. Duplicating an identical physical line across candidates gives identical predictions.
10. Changing geometry outside all queried tubes leaves labels unchanged.
11. Two different interval configurations with the same truncated moments demonstrate non-identifiability.
12. Increasing $M$ cannot reduce oracle information, although learned performance may worsen from variance.
13. Turning off the global shape code makes hidden residual prediction collapse while observed contributions remain unchanged.
14. The same visible lift evaluated at two point sampling densities converges to the same coefficients.
15. The deterministic obstacle gate alone rejects a known collision even if JILT predicts a high target score.

## 18. Draft paper pitch

### Possible title

**Learn the Interaction Transform, Not the Object: Range-Constrained Line Moments for Grasping Through Occlusion**

### Draft abstract

Single-view grasp detectors fail when a foreground object hides a target's opposing contact surface. Existing remedies either complete a full or contact-localized shape, or directly regress grasp scores without constraining whether predictions across nearby actions could arise from any physical object. We introduce task-transform learning under coarsening: instead of inverting the latent object, predict only a non-invertible forward transform sufficient for the downstream interaction and keep that prediction in the transform's physical range. For parallel-jaw grasping, we represent each candidate by truncated Fourier moments of positive occupancy and normal-flux measures along a small hardware-fixed bundle of closing and clearance lines. We propose the Random-Bundle Cone Score and JILT, an unrolled line-space neural operator alternating learned completion with exact visible-data, refinement, reparameterization, and truncated-moment-cone projections. JILT never decodes a mesh, voxel field, completed point cloud, or stochastic action object. Theory characterizes moment validity, projection improvement, deliberate non-injectivity, and action stability under finite sketch error. Controlled synthetic, TARGO, and real shelf experiments test whether constrained interaction sketches preserve grasp mechanics at substantially lower cost than reconstruction and improve hidden-contact selection over matched direct critics.

## 19. Final verdict

JILT is the strongest surviving non-overlapping direction from this search, but it is intentionally not declared a finished breakthrough.

Its value is not “Fourier features for grasps.” The potentially new contribution is a specific middle learning object: **a constrained, non-invertible interaction transform predicted directly from a coarsened observation**. Parallel-jaw grasping supplies exact local line measures, finite moment cones, and hard physical consistency identities, so the idea can generate both theory and a falsifiable architecture rather than only a metaphor.

The first action should not be implementation of the full RGB-D network. It should be the oracle sketch-sufficiency experiment. If eight low-order moments on a small pad bundle do not preserve the local mechanics of the best candidate, the elegant range theory is irrelevant and the project should stop. If they do, JILT offers a credible ICLR path that is structurally separate from today's posterior, random-set, capacity, response-polytope, metamer, avoidance, and ray-jaw process proposals.

## 20. Primary sources

### Grasping and occlusion

- Xia et al., [TARGO and TARGO-Net: Benchmarking Target-driven Object Grasping under Occlusions](https://targo-benchmark.github.io/), IJCV 2026.
- Iwase et al., [ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping](https://openaccess.thecvf.com/content/CVPR2025/papers/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.pdf), CVPR 2025.
- Saund and Berenson, [Diverse Plausible Shape Completions from Ambiguous Depth Images](https://proceedings.mlr.press/v155/saund21a.html), CoRL 2020 / PMLR 2021.
- Wu et al., [TOSC: Task-Oriented Shape Completion for Open-World Dexterous Grasp Generation from Partial Point Clouds](https://ojs.aaai.org/index.php/AAAI/article/view/38053), AAAI 2026.
- Jauhri et al., [Learning Any-View 6DoF Robotic Grasping in Cluttered Scenes via Neural Surface Rendering](https://roboticsproceedings.org/rss20/p046.html), RSS 2024.
- Sundermeyer et al., [Contact-GraspNet](https://arxiv.org/abs/2103.14127), ICRA 2021.
- Wang et al., [Learning contact representations in real-world clutter for universal robotic grasping / SpaHybGen](https://www.nature.com/articles/s42256-026-01292-y), Nature Machine Intelligence 2026.
- Murali et al., [GraspGen: A Diffusion-based Framework for 6-DOF Grasping with On-Generator Training](https://arxiv.org/abs/2507.13097), 2025.
- Adjigble et al., [SpectGRASP: Robotic Grasping by Spectral Correlation](https://arxiv.org/abs/2107.12492), 2021.
- Adjigble et al., [Model-free and learning-free grasping by Local Contact Moment matching (LoCoMo)](https://doi.org/10.1109/IROS.2018.8594226), IROS 2018.
- Ma et al., [Generalizing 6-DoF Grasp Detection via Domain Prior Knowledge](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Generalizing_6-DoF_Grasp_Detection_via_Domain_Prior_Knowledge_CVPR_2024_paper.pdf), CVPR 2024.

### Inverse problems, range learning, and moments

- Adler et al., [Task Adapted Reconstruction for Inverse Problems](https://doi.org/10.1088/1361-6420/ac28ec), Inverse Problems 2022.
- Ge et al., [Machine Friendly Machine Learning: Interpretation of Computed Tomography Without Image Reconstruction](https://pmc.ncbi.nlm.nih.gov/articles/PMC6820559/), Scientific Reports 2019.
- Schwab et al., [Deep Null Space Learning for Inverse Problems](https://arxiv.org/abs/1806.06137), 2018.
- Arndt et al., [Data-consistent neural networks for solving nonlinear inverse problems](https://doi.org/10.3934/ipi.2022037), Inverse Problems and Imaging 2023.
- Tachella et al., [Unsupervised Learning From Incomplete Measurements for Inverse Problems](https://arxiv.org/abs/2201.12151), NeurIPS 2022.
- Natterer, [The Mathematics of Computerized Tomography, Chapter 2: The Radon Transform and Related Transforms](https://doi.org/10.1137/1.9780898719284.ch2), SIAM; classical background for transform-range thinking.
- Arcadu et al., [An Improved Extrapolation Scheme for Truncated CT Data Using 2D Fourier-Based Helgason–Ludwig Consistency Conditions](https://pmc.ncbi.nlm.nih.gov/articles/PMC5541827/), 2017.
- Jiang et al., [Convolutional Neural Networks on Non-uniform Geometrical Signals Using Euclidean Spectral Transformation (NUFT)](https://openreview.net/pdf?id=B1G5ViAqFm), ICLR 2019.
- Yang and Xie, [Frequency-selective Vandermonde decomposition of Toeplitz matrices with applications](https://doi.org/10.1016/j.sigpro.2017.05.028), Signal Processing 2018.
- Krein and Nudelman, *The Markov Moment Problem and Extremal Problems*, classical source for truncated moment problems.
- Unser, [A Unifying Representer Theorem for Inverse Problems and Machine Learning](https://link.springer.com/article/10.1007/s10208-020-09472-x), Foundations of Computational Mathematics 2021.

### Venue criteria

- [ICLR 2027 Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines).
- [ICLR 2027 Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers).
