# Research log: an ICLR-level grasping problem

Started: 2026-08-24

Last updated: 2026-08-25

## Executive outcome after 106 research cycles

The recommended research programme is **EdgeFlux: learning grasp decisions
from vanishing-support sensor responses**.  Before selecting one parallel-jaw
grasp, the wrist makes one fixed, safe lateral RGB-D micro-motion.  EdgeFlux
does not treat the second image as merely another dense view.  It registers the
pair, isolates target surface born by disocclusion, divides its surface measure
by the probe displacement, and integrates the resulting weak visibility
response against compact jaw-, pad-, and approach-local kernels for every
continuous grasp query.

The broad ML problem is learning a decision from a controlled observation in
which the informative stratum has probability mass $O(\epsilon)$, although
its displacement-normalized weak derivative remains $O(1)$.  The precise
novelty claim is the conjunction of an action-generated RGB-D birth stratum, a
displacement-normalized measure queried by continuous grasp actions, and an
equal-token-budget grasp-regret separation.  It is not a claim of first active
view selection, first two-view fusion, or information superiority over the raw
pair.

The efficient-learnability mechanism is explicit: with $K$ unstratified
auxiliary tokens, distinguishing two hidden grasp cases requires
$K=\Omega(1/\epsilon)$ under the sparse-mixture model, whereas conditioning
the same learned token budget on the action-registered birth stratum removes
that dilution.  The proposed learned component is candidate-local and compact;
registration and visibility comparison cost $O(n)$, and $m$ kernel queries
over $K$ retained tokens cost $O(mK)$.

The credible performance target is Pareto-SOTA in physical top-one success
versus camera motion, latency, learned FLOPs, and memory at matched candidates
and sensing action--not unrestricted multi-view accuracy.  This is a
**research-ready but experimentally conditional** proposal.  It must be
rejected if any of four tests fail: there is no safe resolvable parallax/noise
window; oracle birth evidence rarely changes the best grasp; learned saliency
matches it at equal cost; or the strongest local cross-view baseline remains
better on the success--latency frontier.  The complete formulation, theorem
sketches, baselines, evidence, and adversarial novelty audit are consolidated
in the final sections below.

## Scope and rejection criteria

- Parallel-jaw, 6-DoF grasping from one wrist RGB-D view.
- A target object is on a shelf and may be visually occluded by a frontal obstacle.
- No RL or VLA.
- The learned object must not be whole-cycle reach-to-lift feasibility.
- The input must not require a complete scene SDF or another information-heavy scene state.
- A viable contribution must be a new learning object or problem, not a richer head attached to an established grasp pipeline.
- The intended physical test is terminal placement of an already-reached open gripper, closure, and a very small lift/load test. Arm reachability and full approach planning remain out of scope.

## Literature-derived boundary

The main occupied directions are:

1. direct grasp generation/scoring from partial point clouds (S4G, Contact-GraspNet, PointNetGPD, REGNet, GSNet, AnyGrasp);
2. joint shape reconstruction and grasping (GIGA, NeuGraspNet, CenterGrasp, ZeroGrasp);
3. uncertain or multiple shape completions followed by robust grasp evaluation;
4. contact-region-only completion (TOSC) and camera-ray shell prediction (ShellGrasp-Net);
5. continuous grasp manifolds/distance fields (NGDF) and SE(3) generative models;
6. scalar pose-perturbation tolerance (GraspNet ToleranceNet) and success smoothing under pose uncertainty;
7. classical capture regions, funnels, passive self-alignment, and grasp basins of attraction.

Important sources:

- Grasp synthesis review: https://arxiv.org/abs/2207.02556
- GraspNet evaluation and pose-NMS: https://graspnet.net/evaluation.html
- Neural Grasp Distance Fields: https://arxiv.org/abs/2211.02647
- Robust grasping over uncertain shape completions: https://arxiv.org/abs/1903.00645
- vMF-Contact uncertainty model: https://arxiv.org/abs/2411.03591
- TOSC: https://ojs.aaai.org/index.php/AAAI/article/view/38053
- Diverse plausible shape completion: https://arxiv.org/abs/2011.09390
- Goal-oriented inverse problems: https://arxiv.org/abs/1607.01881
- Pullback information geometry: https://proceedings.mlr.press/v151/arvanitidis22b.html

## Rejected cycles

### 1. Action-conditioned minimal contact geometry

Proposal: replace complete SDF completion by bilateral contact-depth/support fields local to a queried grasp.

Rejection: ShellGrasp-Net already predicts entry/exit shell depths, and TOSC already completes only task-relevant contact regions. Adding uncertainty would be a modification of an existing representation.

### 2. Random closed set of valid grasps

Proposal: learn the conditional hitting functional of the feasible grasp set, with a submodular mixture model and greedy candidate coverage.

Rejection: contextual submodular grasp libraries already optimize the probability that a sequence/set contains a successful grasp. More importantly, the laboratory executes one grasp without receiving a new observation between candidates; joint set coverage does not improve the Bayes-optimal single action beyond its marginal score.

### 3. Contact-induced grasp geometry and directional tolerance cells

Proposal: learn a quotient/stratified metric on grasp pose space from contact responses, later strengthened to an asymmetric local tolerance body in the tangent space.

Rejection: NeuralGrasps already learns contact-map similarity, NGDF already models a continuous grasp manifold, and GraspNet already predicts a scalar perturbation tolerance. A 6D anisotropic tolerance body is mathematically richer but would likely be reviewed as a richer ToleranceNet head.

### 4. Passive uncertainty contraction during closure

Proposal: rank grasps by how strongly jaw closure contracts initial object-pose uncertainty, using a learned closure map or its Jacobian.

Rejection: capture regions, funnels, self-alignment, and basins of attraction are classical grasp/manipulation objects. The learned map would be a new estimator of an old object. In a pick-and-tiny-lift task, final pose contraction is also only a proxy for success, and its contact-derivative sim-to-real gap is severe.

### 5. Observation-fiber utility processes

Proposal: learn a joint conditional law/support of grasp utilities over
sensor-indistinguishable hidden shapes, represented by shared latent utility
critics, and use the induced directed regret geometry for selection.

Rejection as the primary paper thesis: conditional multivariate scenario
generation, proper energy-score training, scenario predict-then-optimize, and
worst-case-regret decision-focused learning are already established general
templates. The remaining new elements would be an occlusion-twin dataset and a
grasping instantiation; that is not a sufficiently defensible algorithmic or
formal novelty claim for the target bar. In particular, a one-shot Bayes action
under expected success only needs the vector of marginal expected utilities, so
the joint process is useful only after imposing an additional risk/robustness
criterion.

Close general precedents found during the audit:

- multivariate conditional quantile functions trained with the energy score:
  https://proceedings.mlr.press/v151/kan22a.html
- scenario predict-then-optimize:
  https://arxiv.org/abs/2401.17787
- robust decision-focused learning via worst-case regret:
  https://proceedings.mlr.press/v337/yamao26a.html
- distribution-free robust functional predict-then-optimize:
  https://arxiv.org/abs/2602.08215

The formulation is retained below because two components may be reusable:
observation-preserving physical twins and exact supervision of otherwise
unidentifiable hidden counterfactuals.

### 6. Grasp observability / fiberwise utility variation

Proposal: predict the sharp utility interval or vertical sensitivity of a grasp
along the null directions of the RGB-D rendering map,

$$
\omega(o,g)=\sup_{z,z'\in R^{-1}(o)}|Q(z,g)-Q(z',g)|,
\qquad
\|P_{\ker DR(z)}\nabla_z Q(z,g)\|.
$$

Rejection: this is a local/nonlinear restatement of goal-oriented inverse-problem
uncertainty and partial identification. Selecting a grasp by its lower endpoint
returns to established robust grasp planning over shape uncertainty; using the
variation only as an auxiliary head would not create a new learning problem with
enough independent value.

### 7. Heat-flow robustness on grasp pose space

Proposal: treat perturbation-smoothed binary grasp success as heat evolution on
the gripper-symmetry quotient of SE(3), and enforce scale/semigroup consistency.

Rejection: pose-uncertainty convolution of a learned grasp function was already
the central method of Johns et al. (2016), and later work repeatedly samples and
smooths neighboring 6-DoF grasps. A Lie-group heat equation would add formal
structure, but not a sufficiently new task or empirical capability.

Source: https://arxiv.org/abs/1608.02239

### 8. Post-search conformal grasp fields

Proposal: construct a simultaneous lower confidence band over all candidate
grasps (or an epsilon-net of continuous SE(3)) so that maximizing a learned score
does not invalidate calibration through optimizer's curse.

Rejection: conformal predict-then-optimize, selection-conditional conformal
inference, and conformal safety calibration have mature general treatments; a
2022 robotics paper already includes a parallel-jaw grasping demonstration. If
the full selection map is frozen before calibration, calibrating only its chosen
action also makes the basic guarantee nearly immediate.

Sources:

- https://wafr2022.github.io/proceedings/WAFR_2022_Final_24.pdf
- https://arxiv.org/abs/2310.10003
- https://www.jmlr.org/beta/papers/v26/24-0452.html

### 9. Learned primal-dual lift certificates

Proposal: predict contact forces proving gravity-wrench feasibility, or a dual
separating twist proving failure, rather than a scalar grasp score.

Rejection: grasp analysis through convex programs and wrench-space certificates
is classical, while GraspQP already embeds differentiable force-closure QPs into
grasp synthesis. Under partial observation, a certificate for predicted hidden
contacts is not physically verifiable; restricting it to visible contacts reduces
the method to an analytic contact-based grasp detector.

Sources:

- https://www.cs.cmu.edu/~lihan/Research/LMI_icra.html
- https://graspqp.github.io/

### 10. Randomized grasp policies against hidden twins

Proposal: because one deterministic grasp may be defeated by a hidden completion,
learn a mixed distribution over grasps that maximizes worst-case success over
sensor-indistinguishable shapes.

Rejection: mixed strategies for robust optimization are an established general
construction, including efficient learning formulations. More importantly, under
the laboratory's natural one-shot expected-success objective, a Bayes-optimal pure
action is sufficient: randomization helps only after imposing an adversarial
worst-case game. Thus the formulation creates its own need rather than exposing a
necessary grasping subproblem.

Source: https://proceedings.mlr.press/v108/sessa20a.html

### 11. Observation-conditioned guaranteed wrench core

Proposal: for latent scene $z$, grasp $g$, and feasible grasp-wrench body
$W(z,g)\subset\mathbb R^6$, learn

$$
K(o,g)=\bigcap_{z:R(z)\simeq o} W(z,g).
$$

A compact implementation would predict the gauge of this convex body. The exact
identity

$$
p_{K(o,g)}(w)=\sup_{z:R(z)\simeq o}p_{W(z,g)}(w)
$$

would permit supervision with scalar wrench queries and avoid full meshes/SDFs.

Rejection: the central mathematical object is already the *Reachable Wrench Space
under Uncertainties* (RWSU), introduced in 2014 as all wrenches that a grasp can
apply regardless of contact-location and contact-orientation uncertainty, with
algorithms for lower and upper approximation. Recent work also constructs
risk-adjusted GWS under friction uncertainty and aggregates force-closure quality
over uncertain shape completions. Amortizing RWSU with a neural convex gauge is an
interesting efficiency improvement, but it is not a new task or principle; the
likely review summary would be "learned approximation of a classical robust grasp
metric."

Sources:

- https://doi.org/10.1109/AIM.2014.6878251
- https://arxiv.org/abs/2607.25049
- https://arxiv.org/abs/1903.00645

### 12. Parallel-jaw first-contact operator / contact-mode atlas

Proposal: closing a jaw selects the earliest surface intersection, hence the
contact map is a switching argmin (a lower envelope of smooth contact-time
sheets), not an ordinary smooth regression target. Learn the competing sheets,
their active contact modes, and the gap to a contact-switch discriminant in a
local grasp-pose neighborhood.

Rejection: predicting action-conditioned tactile/contact outcomes is already an
active line, and probabilistic contact grasping explicitly models directional
ambiguity. The new active-mode margin is mechanically interpretable, but a
contact switch does not itself imply grasp failure. Under hidden geometry the
method also needs a set/distribution of atlases, returning to the already weak
utility-process formulation with a more elaborate intermediate variable.

Sources:

- https://www.roboticsproceedings.org/rss18/p070.pdf
- https://arxiv.org/abs/2411.03591

### 13. Learned antipodal support jets

Proposal: for a convex object, the support function $h(u)$, its spherical
gradient, and its Hessian at $u$ and $-u$ determine antipodal contact
locations and local curvatures. Predict these compact support jets from a
partial observation instead of reconstructing an SDF.

Rejection: the representation is exact only for convex bodies. Useful
parallel-jaw grasps on non-convex household objects frequently exploit local
width minima; extending the representation to localized cross-sections produces
an action-space aperture field close to existing dense grasp maps. The elegant
convex geometry would therefore purchase novelty by excluding central cases.

### 14. Persistent topology of the feasible-grasp set

Proposal: learn the persistent homology of superlevel sets of a grasp field on
$SE(3)/\mathbb Z_2$, preserving distinct stable grasp components and returning
at least one candidate from each persistent component.

Rejection: grasp moduli spaces and persistent-homology-based grasp/cage
analysis already provide close mathematical context. More importantly, component
coverage becomes necessary only if later, unknown IK or trajectory constraints
may delete whole components. For the specified one-shot small-lift subproblem,
it is an expensive surrogate for top-K candidate recall rather than an inherent
objective.

Sources:

- https://roboticsproceedings.org/rss09/p36.pdf
- https://berkeleyautomation.github.io/caging/

### 15. Critical-friction / capability-boundary regression

Proposal: predict the minimum friction coefficient or maximum supported load for
each candidate, yielding a monotone physical phase boundary rather than a binary
success score.

Rejection: minimum required friction has been used as a grasp-quality measure
for decades, and maximum load is a direct wrench-space query. Learning these
classical scalar metrics from RGB-D changes the label, not the problem.

### 16. Cross-view grasp-field synchronization

Proposal: use known camera transforms to align view-specific grasp functions and
enforce agreement on overlapping reliable domains, potentially interpreted as
gluing local sections of a grasp-field sheaf. Train from wrist-camera sweeps and
deploy from one view.

Rejection: without an anchor the consistency loss collapses; with a fused
multi-view teacher it becomes standard multi-view self-distillation. The sheaf
language does not create a new capability beyond multi-view fusion and
equivariant grasp learning.

Sources:

- https://orbitgrasp.github.io/
- https://pku-epic.github.io/GraspNeRF/

## Candidate C: information-isotonic affordance bounds (rejected)

Represent an RGB-D observation as a set $E$ of occupied/free-space ray
constraints. Define refinement by reverse inclusion of the compatible latent
worlds:

$$
E_1\preceq E_2 \quad\Longleftrightarrow\quad
\mathcal C(E_2)\subseteq\mathcal C(E_1).
$$

For a grasp $g$, the sharp utility endpoints

$$
L(E,g)=\inf_{z\in\mathcal C(E)}Q(z,g),\qquad
U(E,g)=\sup_{z\in\mathcal C(E)}Q(z,g)
$$

obey an information-order law:

$$
E_1\preceq E_2\Rightarrow
L(E_1,g)\le L(E_2,g)\le U(E_2,g)\le U(E_1,g).
$$

Thus additional consistent sensor evidence can only contract an affordance
interval. Ordinary expected-success or evidential grasp scores have no such
property and can become arbitrarily more pessimistic after receiving valid
additional evidence.

A compact model would query only rays intersecting a candidate-local gripper
influence volume, and use a permutation-invariant set architecture that is
monotone under evidence inclusion by construction. It predicts two scalars per
candidate, not a completion. Training uses nested observations and grouped
occlusion twins; the endpoints are the min and max closure/small-lift utility
among twins consistent with each evidence prefix.

Current advantages:

1. The target follows from an exact order-theoretic property, not a chosen loss.
2. It yields an anytime consistency law across point dropout, denoising, and
   extra views.
3. Inference is candidate-local and linear in the selected ray tokens.
4. The idea generalizes to affordance bounds in other non-injective embodied
   inverse problems.

Current risks:

1. Monotone set networks are established; the architecture alone is not novel.
2. The endpoints remain partial-identification intervals, a direction already
   judged too close to goal-oriented uncertainty quantification.
3. With an overly broad admissible hidden-shape class, $L$ is identically zero
   for most single-view grasps.
4. A learned monotone output is not automatically a valid physical bound;
   coverage still depends on the twin family and calibration.
5. The laboratory setting supplies one view, so the refinement axiom must improve
   single-view training/robustness rather than rely on active multi-view sensing.

Relevant general architecture work:

- https://proceedings.mlr.press/v202/runje23a.html

Rejection after the direct general-ML audit: sharp conditional endpoints are a
standard partial-identification object, and learned estimators of sharp bounds
already appear at NeurIPS/ICML. Set-inclusion-monotone networks are also an
independent established topic. Their combination would give a clean grasping
application, but not the requested objectively strong conceptual novelty.

Closest sources found during the rejection audit:

- https://proceedings.mlr.press/v202/oprescu23a.html
- https://proceedings.neurips.cc/paper_files/paper/2024/hash/bdabb5d4262bcfb6a1d529d690a6c82b-Abstract-Conference.html
- https://proceedings.mlr.press/v213/padh23a.html

### 17. Low-rank bilateral contact-compatibility operator

Proposal: represent the intrinsically two-sided parallel-jaw decision by a
pairwise kernel between left and right contact evidence. Factor a rank-$r$
kernel so that all contact-pair scores can be computed in $O(Nr)$ rather than
$O(N^2)$.

Rejection: this is an efficient architecture, not a new learning problem.
PhyGrasp already predicts per-point pair embeddings and an explicit grasp-pair
match classifier; VCPD and earlier geometric methods also generate and classify
contact pairs. Low-rank evaluation alone is incremental.

Sources:

- https://arxiv.org/html/2402.16836v1
- https://proceedings.mlr.press/v205/cai23a.html
- https://haojhuang.github.io/edge_grasp_page/

### 18. Learned admissible branch-and-bound over $SE(3)$

Proposal: learn a supremum oracle $U(o,B)$ for cells $B$ of continuous grasp
space and use best-first subdivision until the returned grasp is
$\epsilon$-optimal.

Rejection: without a physically sound upper bound the guarantee concerns only a
possibly wrong learned score. Making the bound physically valid requires neural
verification plus uniform calibration, reducing the idea to a composition of
established tools. Without that guarantee it is another coarse-to-fine candidate
generator, a populated grasping direction.

Sources:

- https://proceedings.mlr.press/v155/jeng21a.html
- https://arxiv.org/abs/2204.01131
- https://proceedings.neurips.cc/paper_files/paper/2021/file/fac7fead96dafceaf80c1daffeae82a4-Paper.pdf

### 19. Difference-body / neural width geometry

Proposal: for a convex body $K$, learn only its difference body
$D=K+(-K)$. Its support function is the directional width
$h_D(n)=h_K(n)+h_K(-n)$, and a support point of $D$ is the chord between the
two first parallel-jaw contacts. This is a translation-free, embodiment-specific
quotient of shape.

Rejection: the exact result covers only global support contacts on convex bodies.
Local grasps of non-convex objects require a windowed, multi-valued chord field,
which is effectively the full contact-pair/grasp field again. Differentiable
support functions are also already learned as compact contact geometry. The
mathematics is attractive but the broad empirical claim would be false.

Sources:

- https://inrol.github.io/inrol-mdsf/
- https://research.google/pubs/cvxnet-learnable-convex-decomposition/
- https://motion.cs.illinois.edu/RoboticSystems/AdvancedTopicsInPlanning.html

### 20. Action-identifiability intervals and sensor twins

Proposal: construct exact or sensor-indistinguishable latent twins with different
hidden backside geometry and quantify, for every candidate, the range of utility
on the observation fiber. Use common feasible actions or a learned lower endpoint.

Status: the *benchmark axis* survives, but the scalar model does not. No grasping
benchmark was found that holds RGB-D fixed while changing only hidden geometry;
TARGO varies the amount of occlusion and then completes shape. However,
perceptual aliasing, common-action existence under partial observability, sharp
bounds, and robust lower-score learning are all known general objects. Sensor
twins therefore become supervision and evaluation for a stronger model below,
not the complete paper.

Sources:

- https://targo-benchmark.github.io/
- https://www.mdpi.com/2076-3417/16/10/4917
- https://cdn.aaai.org/AAAI/1992/AAAI92-029.pdf

## Candidate D: sensor-null action spectra

### Broad scientific hypothesis

Partial observation leaves a high- or infinite-dimensional set of hidden shapes,
but the *effect* of those shapes on a perturbation-smoothed parallel-jaw grasp
field can be much lower-dimensional. A learner should model the pushforward of
sensor ambiguity into action-utility space, rather than reconstruct hidden shape
or predict unrelated scalar uncertainty for every action.

Let $R:\mathcal Z\to\mathcal O$ be the RGB-D sensor map and let
$Q:\mathcal Z\to L^2(\mathcal G)$ map a latent scene to its closure-and-small-
lift utility function. On a differentiable visibility stratum, define the
**sensor-null action operator**

$$
A_z = DQ_z\big|_{\ker DR_z}:\ker DR_z\longrightarrow L^2(\mathcal G).
$$

Its image contains exactly the first-order changes in the entire grasp field that
the sensor cannot observe. Its singular values form the **sensor-null action
spectrum**, and the numerical rank at tolerance $\delta$ is the action
ambiguity dimension. This is not the dimension of hidden shape and not the
uncertainty of one selected grasp.

The finite, nonlinear counterpart for observation $o$ is

$$
\mathcal U(o)=\{Q(z,\cdot):z\in\mathcal C_\varepsilon(o)\},
$$

with centered Kolmogorov width

$$
d_r(\mathcal U(o))=
\inf_{\mu,\,\dim V\le r}\;
\sup_{q\in\mathcal U(o)}\inf_{v\in V}
\|q-\mu-v\|.
$$

The central empirical claim is not assumed for free: measure whether these widths
decay rapidly for realistic occluded-object families, and reject the project if
they do not.

### Why smoothing can make the hypothesis true

The operational target should be success probability under the robot's actual
small pose perturbation $\xi$,

$$
Q_z^\sigma(g)=\mathbb E_\xi[Y_z(g\circ\xi)],
$$

not an infinitely sharp binary contact indicator. On a compact action manifold,
isotropic perturbation is heat-kernel smoothing. If
$Y_z=\sum_j c_{zj}e_j$ in Laplace--Beltrami eigenfunctions with eigenvalues
$\lambda_j$, then

$$
Q_z^\sigma=\sum_j e^{-\sigma^2\lambda_j/2}c_{zj}e_j.
$$

Consequently, uniformly $L^2$-bounded raw success fields admit the spectral
truncation bound

$$
\|Q_z^\sigma-P_rQ_z^\sigma\|_2
\le e^{-\sigma^2\lambda_{r+1}/2}\|Y_z\|_2.
$$

Weyl growth $\lambda_r\asymp r^{2/d}$ gives exponentially decreasing error in
$r^{2/d}$. This does not prove that $r$ is tiny in practice, but it supplies a
falsifiable mechanism: finite pads and execution noise remove action frequencies
that no robot can exploit reliably.

### Efficient model: a sensor-null action ellipsoid

The model should follow the operator rather than merely attach a low-rank head
to a grasp network.  A radius-$c$ ball of infinitesimal sensor-null
perturbations is mapped by $A_z$ to an ellipsoid in action-function space.  Its
truncated SVD is

$$
A_z h\simeq \sum_{j=1}^r s_j u_j\langle v_j,h\rangle .
$$

This motivates an observation-conditioned **sensor-null action ellipsoid
(SNAE)**, not independent uncertainty intervals.  For a candidate set
$G_o=\{g_m\}_{m=1}^M$ generated solely from the shared observation, a point/ray
encoder and gripper-query decoder output

$$
\mu_m=\mu_\theta(o,g_m),\qquad
U_{mj}=u_{\theta,j}(o,g_m),\qquad d_j\ge 0,\qquad \rho\ge0.
$$

A weighted thin QR or polar layer makes $U^TWU=I_r$.  The radii $d_j$ are
kept separate from the directions, fixing the scale ambiguity.  The predicted
functional set is

$$
\widehat{\mathcal U}_r(o)=
\left\{\mu+U\mathrm{diag}(d)a+e:
\|a\|_2\le1,\ \|e\|_\infty\le\rho\right\}.
$$

The coefficient $a$ indexes an observation-preserving hidden perturbation and
is never inferred at deployment.  The robust lower score is the support
function of the ellipsoid and is therefore closed-form:

$$
\widehat L_m
=\inf_{q\in\widehat{\mathcal U}_r(o)}q_m
=\mu_m-
\sqrt{\sum_{j=1}^r d_j^2U_{mj}^2}-\rho .
$$

After clipping to $[0,1]$, select $\arg\max_m\widehat L_m$, or abstain when
its value is below a declared threshold.  Inference is $O(Mr+Mr^2)$, returns
only $r+2$ scalars per queried grasp, and never predicts a mesh, voxel grid,
or scene SDF.

For training group $b$, let $q_{b,s}\in[0,1]^M$ be the perturbation-smoothed
utility vector of hidden twin $s$.  The exact finite-sample target is a
minimum-width rank-$r$ enclosing ellipsoid with an $L^\infty$ residual:

$$
\begin{aligned}
\min_{\mu,U,d,\rho,\{a_s\}}\quad &
 \rho+\lambda_w\frac1M\sum_{m=1}^M
 \sqrt{\sum_{j=1}^r d_j^2U_{mj}^2}
 +\lambda_v\sum_{j=1}^r\log(d_j+\epsilon)\\
\text{s.t.}\quad &U^TWU=I_r,\quad \|a_s\|_2\le1,\\
&\|q_{b,s}-\mu-U\mathrm{diag}(d)a_s\|_\infty\le\rho
\quad\forall s.
\end{aligned}
$$

The first two terms directly minimize the uniform error and actionwise robust
half-width used by the downstream selector.  The log-volume term is only a
tightness regularizer; an ablation must show that it does not manufacture
overconfidence.  In implementation, projected inner updates for $a_s$, a
log-sum-exp approximation of the largest residual, and alternating
network/coefficient steps make the program differentiable and minibatchable.
Hard residual mining restores the worst twin/candidate pairs lost by softening.

Sparse utility labels $\Omega_{b,s}\subset[M]$ can be used in the inner fit,
with disjoint held-out candidate entries controlling reconstruction.  This is a
testable sample-efficiency claim, not a guarantee obtained merely from the
factorization.  The required baseline is a direct lower-envelope head with the
same encoder, candidate generator, twin groups, and label budget.  SNAE is useful
only if it improves uniform field completion, off-grid candidate querying, or
robust selection; matching a conditional-PCA reconstruction score is
insufficient.

The architecture itself is not claimed as novel.  Parameter-dependent reduced
bases, Grassmann regression, and ellipsoidal uncertainty sets already exist.
The contributions would be the sensor-null action operator/spectrum, the
grasp-specific compressibility law, exact sensor-twin supervision, and learning
an enclosing action-function image whose support function is the decision rule.

### Greedy twin mining as reduced-basis construction

Given the current basis, search only inside the RGB-D shadow volume for a hidden
deformation whose utility field has the largest residual outside the span:

$$
z^*=\arg\max_{z:\,d(R(z),o)\le\varepsilon}
\|(I-P_\Phi)(Q(z,G_o)-\mu)\|.
$$

Add this counterexample to the twin group and repeat. This is a greedy reduced-
basis construction in task space, not shape completion at inference. In practice
the inner search can use procedural hidden surfaces first and differentiable
render/contact surrogates only as a secondary stress test.

### Immediate theory targets

1. **Sensor-twin lower bound.** For two equally likely latent scenes with
   observation laws $P_0,P_1$, suppose choosing the other scene's optimal
   action costs at least $\Delta$. Every observation-only (even randomized)
   selector then has expected regret at least

   $$
   \frac{\Delta}{2}\big(1-\mathrm{TV}(P_0,P_1)\big),
   $$

   by reduction to binary testing and Le Cam's bound. Exact twins have
   $P_0=P_1$, so more model capacity cannot remove the gap.
2. **Factorization criterion.** If fibers are connected and
   $DQ_g\ker DR=0$ everywhere, then $Q_g=\bar Q_g\circ R$; zero action rank
   is exactly local-to-global action identifiability.
3. **Local nonlinear width bound.** Let $\psi:B_{\mathcal H}(c)\to \mathcal C(o)$ be a chart of one smooth sensor fiber, set
   $F=Q^\sigma\circ\psi$, and assume $\|D^2F\|\le K$.  If $V_r$ is
   spanned by the first $r$ left singular functions of $A=DF(0)$, Taylor's
   theorem and SVD optimality give

   $$
   \sup_{\|h\|\le c}
   \mathrm{dist}\big(F(h)-F(0),V_r\big)
   \le c\,s_{r+1}(A)+\tfrac12Kc^2.
   $$

   The spectrum therefore controls local nonlinear ambiguity up to an explicit
   curvature term; it is not only a visualization of a Jacobian.
4. **Spectrum-to-decision identity.** On a finite shared candidate set, let
   $e_m$ evaluate candidate $m$.  For the linearized fiber ball
   $q(h)=q_0+Ah,\ \|h\|\le c$, the exact robust utility is

   $$
   \inf_{\|h\|\le c}e_m^Tq(h)
   =q_{0,m}-c\|A^Te_m\|_2
   =q_{0,m}-c\sqrt{\sum_j s_j^2u_j(m)^2}.
   $$

   Retaining $r$ modes and subtracting
   $cs_{r+1}+Kc^2/2$ gives a conservative lower score under the preceding
   curvature assumption.  This identity is the reason for the row-norm in SNAE;
   the ellipsoid is not a decorative uncertainty head.
5. **Heat-compressibility bound.** If the raw sensor-null derivative is a bounded
   operator $B_z$ and operational utility applies heat smoothing $H_t$, then
   $A_z=H_tB_z$ is compact and

   $$
   s_j(A_z)\le \|B_z\|e^{-t\lambda_j}.
   $$

   Thus measured execution precision supplies an upper bound on effective
   action-space rank; the claim should be tested across noise scales.  A uniform
   version is available for action selection: with $K_t(g,g)$ the heat-kernel
   diagonal,

   $$
   \|(I-P_r)H_tf\|_\infty
   \le \sup_g K_t(g,g)^{1/2}
   e^{-t\lambda_{r+1}/2}\|f\|_2.
   $$

   This separates the $L^2$ spectral diagnostic from the uniform error needed
   by a maximization over actions.
6. **Robust-selection stability.** If the learned and true functional sets are
   within $\delta$ in Hausdorff $L^\infty$ distance, their lower envelopes
   differ by at most $\delta$, and maximizing the learned lower envelope has at
   most $2\delta$ maximin regret.

Together, items 1 and 4 form an impossibility--compressibility sandwich:
non-injective sensing creates irreducible decision loss, but the resulting family
of operational utility functions can still admit an efficient task-space model.

### Current novelty threats

1. Goal-oriented inverse problems already compress posterior uncertainty in a
   quantity of interest; here the claimed new object is the *sensor-null image in
   an action-function space*, learned conditionally from raw RGB-D.
2. Functional prediction sets, conditional uncertainty sets, neural operators,
   zonotopes, and reduced-basis methods all exist. None of those components is a
   contribution by itself.
   In particular, adaptive parameter-dependent bases are learned in
   https://arxiv.org/abs/2105.14633, and the ICLR 2026 paper *Deep Learning for
   Subspace Regression* directly learns parameter-indexed Grassmann points:
   https://openreview.net/pdf?id=HF60Lu1Maj.  Consequently, neither conditional
   basis prediction nor an orthogonality layer may appear in the contribution
   list.
3. If the paper only shows a conditional PCA plus robust argmax, it is not ICLR
   level. It needs the operator/spectrum result, exact-twin benchmark, a measured
   low-rank law, and a clear advantage over direct lower-bound and scenario heads.
4. Arbitrary local hidden bumps can make the unsmoothed field high-rank. The
   project must specify the perturbation scale and plausible hidden-shape family;
   universal worst-case geometry would make the claim false or vacuous.
5. The benchmark must contain enough common robust grasps. Otherwise it measures
   impossibility but cannot demonstrate a better selector.
6. FFHFlow already uses flow likelihoods to introspect uncertainty of grasps from
   partial point clouds. It models a distribution of dexterous grasp poses and
   OOD/view likelihood, not an observation-fiber set of utility functions or its
   sensor-null spectrum, but it is an essential uncertainty-aware baseline and a
   warning against broad claims such as "first uncertainty-aware grasp model."

### Cheap falsification study before model development

Generate a pilot with about 50 held-out visible shells, **64--128** exact hidden
variants per shell, 256--512 shared visible-input-generated candidates, and Monte
Carlo execution perturbations at the measured robot covariance.  Eight or
sixteen twins are not enough: their centered utility matrix has rank at most
seven or fifteen by construction, making a claimed rank of eight nearly
tautological.  The hidden generator must itself have at least 32--64 independent
degrees of freedom, combine smooth random fields with held-out discrete topology
programs, and report its own effective rank.  Before training a neural model:

1. Compute centered SVD spectra of every twin-by-candidate utility matrix for raw
   binary outcomes and for operationally smoothed success probabilities.
2. Require rank $r\le8$ to explain at least 90% of centered energy and to keep
   the 90th-percentile per-candidate error below 0.05 in at least 70% of held-out
   visible-shell groups.  Also solve the uniform enclosing-factor objective: a
   favorable Frobenius SVD alone does not establish the $L^\infty$ accuracy
   needed by action maximization.
3. Compare oracle maximin selection on the full matrix with selection after
   rank-$r$ truncation. A loss larger than 5 percentage points kills the
   reduced-basis route even if average reconstruction looks good.
4. Require a nontrivial regime: ordinary mean-score selection should trail the
   oracle common-grasp selector by at least about 10 points, while an oracle
   common grasp with worst-twin utility above 0.7 should exist in at least half
   of benchmark scenes.
5. Train a high-capacity hidden-variant discriminator on RGB-D. Its accuracy must
   be statistically indistinguishable from chance, while an oracle with the full
   latent mesh must distinguish utility fields. Otherwise the data measure
   ordinary visual generalization, not sensor-fiber ambiguity.

Only after these tests pass is the amortized model justified.

Closest general sources for the audit:

- https://arxiv.org/abs/1607.01881
- https://proceedings.neurips.cc/paper_files/paper/2022/hash/3df874367ce2c43891aab1ab23ae6959-Abstract.html
- https://www.jmlr.org/papers/v26/25-1161.html
- https://arxiv.org/abs/2602.08215
- https://proceedings.mlr.press/v145/bollinger22a.html
- https://proceedings.mlr.press/v305/feng25a.html

## Superseded substrate from cycle 5: observation-fiber utility processes

### Scientific claim

The inverse problem caused by single-view occlusion is unnecessarily hard because grasping does not require the hidden shape itself. It requires the image of the hidden-shape ambiguity under the grasp utility functional.

Let $z\in\mathcal Z$ be a complete latent object/scene, $R(z)$ its RGB-D rendering, and $Q(z,g)\in[0,1]$ the closure-and-small-load utility of grasp $g$. For an observation $o$, its sensor fiber is

$$
\mathcal C_\varepsilon(o)=\{z:d(R(z),o)\leq\varepsilon\}.
$$

Instead of reconstructing $z$, learn the observation-conditioned random utility function

$$
U_o:g\mapsto Q(Z,g),\qquad Z\sim p(\cdot\mid o),
$$

or, without a trusted probability prior, its support over $Z\in\mathcal C_\varepsilon(o)$.

This is a goal-oriented inverse problem whose output is a distribution/set of scalar fields on grasp space, not a distribution of 3D shapes.

### Why marginal grasp uncertainty is insufficient

For candidates $G=\{g_1,\ldots,g_M\}$, the relevant pushforward is the joint utility vector

$$
\nu_{o,G}=\big(Q(Z,g_1),\ldots,Q(Z,g_M)\big)_{\mathrm{push}}p(Z\mid o).
$$

Independent confidence intervals for each grasp discard correlation across candidates. The hidden completion that is bad for one grasp can be good for another. A decision-aware representation must preserve entire hypothetical utility profiles.

### Compact model

Predict $K$ latent utility critics and optional weights:

$$
f_{\theta,k}(o,g)\in[0,1],\quad \pi_{\theta,k}(o)\geq0,
\quad\sum_k\pi_{\theta,k}=1.
$$

For any candidate set, row $k$,

$$
\big(f_{\theta,k}(o,g_1),\ldots,f_{\theta,k}(o,g_M)\big),
$$

is one hypothetical utility world. A point-cloud encoder supplies observation tokens; a gripper-centric query encoder supplies a feature for $g$; $K$ shared scenario tokens produce utilities consistently across every queried grasp. Inference is $O(KM)$ and does not generate a mesh, voxel grid, or SDF.

Train the weighted version with a proper multivariate score such as the energy score on utility vectors. Train a prior-free support version with a soft Hausdorff/set loss.

### Occlusion-twin supervision

Ordinary datasets provide one latent scene per observation and cannot identify counterfactual hidden-shape ambiguity. Construct grouped scenes

$$
\{z_{b,1},\ldots,z_{b,S}\},\qquad R(z_{b,s})=o_b,
$$

by deforming geometry only inside the camera/obstacle shadow volume while fixing visible surfaces, silhouette, texture, camera, and depth-noise realization. Generate the candidate set from $o_b$, so it is identical for every twin. Simulate all twin/candidate pairs to obtain an $S\times M$ utility matrix.

A decisive real experiment uses 3D-printed twin families with the same camera-facing shell and statistically indistinguishable wrist RGB-D, but different hidden backside/contact geometry.

### Directed regret geometry

For the support formulation define

$$
d_o(g,h)=\left[\sup_{z\in\mathcal C_\varepsilon(o)}
\big(Q(z,h)-Q(z,g)\big)\right]_+.
$$

This is a directed pseudometric: $d_o(g,g)=0$ and

$$
d_o(g,k)\leq d_o(g,h)+d_o(h,k).
$$

It measures the worst supported disadvantage of choosing $g$ instead of $h$. Worst-case oracle regret is the directed eccentricity

$$
R_o(g)=\max_h d_o(g,h),
$$

and minimax-regret selection is a directed 1-center. The critic model gives a property-preserving approximation

$$
\widehat d_o(g,h)=
\max_k\big[f_{\theta,k}(o,h)-f_{\theta,k}(o,g)\big]_+.
$$

This form is related to quasimetric embeddings, but here the quasimetric is derived from observation-fiber decision regret rather than imposed as a generic metric-learning device.

Strict minimax regret can sacrifice absolute worst-case success. It should therefore be reported as one decision rule, alongside posterior expected utility, CVaR, and maximin utility; a safety threshold on lower utility can precede regret minimization.

### Candidate theory

1. **Non-identifiability lower bound.** If two latent scenes have the same observation and different optimal grasps separated by a utility gap, every deterministic point predictor incurs positive regret on at least one scene.
2. **Decision sufficiency.** For any one-shot loss depending on the latent scene only through $Q(z,\cdot)$, the pushforward utility process is sufficient; the full posterior over geometry contains no additional decision-relevant information.
3. **Support stability.** If the predicted and true utility supports are within Hausdorff distance $\delta$ in $\ell_\infty$, every pairwise regret distance is within $2\delta$, yielding a bounded excess robust-selection regret.
4. **Distributional stability.** Wasserstein error of the learned joint utility law bounds error in expectations of Lipschitz decision losses.
5. **Finite candidate approximation.** For perturbation-smoothed grasp utility that is Lipschitz on the gripper-symmetry quotient of $SE(3)$, an $\epsilon$-net of grasps induces controlled decision regret.

### Essential baselines

- deterministic grasp score;
- deep ensemble/evidential uncertainty over individual grasp scores;
- scalar GraspNet-style tolerance;
- deterministic shape completion then grasping;
- multiple shape completions followed by mean, CVaR, maximin, or regret selection;
- independent per-grasp quantile regression with the same compute/data;
- oracle twin utility matrix and candidate-recall upper bounds.

### Essential metrics

- multivariate energy distance/calibration of the joint utility law;
- marginal calibration and sharpness;
- expected, CVaR, worst-twin, and oracle regret of the selected grasp;
- physical grasp success on randomized hidden twin variants;
- inference time and memory versus multiple full completions;
- robustness versus occlusion, PCD noise, unseen visible geometry, and unseen hidden deformation families.

### Current novelty/acceptance risks

1. Goal-oriented uncertainty quantification and minimax regret are established general ideas; novelty must come from the sensor-fiber learning problem, joint utility-process target, property-preserving critic model, and occlusion-twin benchmark together.
2. Bounds are always relative to a specified latent support/prior. Distribution-free claims over arbitrary hidden geometry would be vacuous.
3. Exact observation fibers are easy synthetically but require careful verification on real RGB-D. The physical-twin benchmark must show that a discriminator cannot identify the hidden variant better than chance.
4. If candidate recall is poor, no re-ranker can recover a robust grasp; this must be isolated experimentally.
5. To be ICLR-level rather than only a robotics systems paper, the paper should emphasize the general learning object and theory, with grasping as a stringent non-injective physical inverse problem.

## Second independent search pass: 2026 boundary and new cycles

This pass deliberately did **not** elaborate Candidate D or any of cycles
1--20.  It restarted from the task constraints and used robotics papers only
to mark occupied territory or to falsify novelty.  The mathematical
inspirations considered after that audit were sparse witnesses, stochastic
orders, mathematical morphology, variational inequalities, hybrid-system
jets, homogenization, dimensional analysis, and set convergence in numerical
analysis.

### Updated occupancy map

Several developments narrow the space more than the older survey alone
suggests:

1. TARGO now directly benchmarks target-conditioned grasping from a single
   RGB-D view under target occlusion.  Its model segments the visible target,
   completes missing target geometry, and fuses target and scene features.
   Therefore, "target-aware grasping under occlusion" is a setting, not a new
   problem statement: https://targo-benchmark.github.io/
2. ICGNet already makes instance-centric implicit geometry simultaneously
   serve reconstruction, segmentation, and grasp detection from a partial
   point cloud.  A new proposal cannot claim novelty merely from avoiding a
   monolithic scene field or from composing per-object fields:
   https://icgraspnet.github.io/
3. NeuGraspNet already couples local neural surface rendering and grasp
   evaluation from arbitrary single views:
   https://openreview.net/forum?id=Fdu33eoZas
4. A 2026 hybrid energy-based/ICP method explicitly treats partial-observation
   6-DoF grasp generation, while a separate 2026 study reports that modular
   pose-and-shape estimation followed by classical antipodal planning can
   outperform end-to-end alternatives.  Thus "use a better implicit
   representation" and "restore a modular geometry stage" are both occupied
   methodological choices, not open scientific objects:
   https://arxiv.org/abs/2606.18053 and
   https://arxiv.org/abs/2605.26944
5. ICLR 2026 work on learning grasps from procedurally random primitive
   assemblies shows that data generation itself can carry broad ML novelty,
   but also raises the bar for any proposal whose only contribution is a new
   synthetic object distribution:
   https://proceedings.iclr.cc/paper_files/paper/2026/hash/4b2a917e30e1bb1aff055b4d8c6c081c-Abstract-Conference.html

These sources leave a real empirical gap -- robust grasping of a partially
occluded target remains unsolved -- but they do not by themselves imply a new
learning problem.

### 21. Continuous amodal co-rigidity

Proposal: predict a continuous kernel

$$
K_o(x,y)=\Pr[x\text{ and }y\text{ belong to the same rigid target}\mid o]
$$

and use it to choose two jaw contacts without completing a mesh.

Rejection: this is an instance-centric amodal representation.  ICGNet already
learns an instance-centric implicit field jointly for reconstruction and
grasping.  A random-partition-consistent kernel would be an elegant structural
constraint, but it neither establishes contact stability nor supplies a new
capability.  The likely paper would be reviewed as a different head for
amodal instance completion.

### 22. Grasp-evidence coresets

Proposal: learn a small subset of RGB-D points or camera rays that preserves
the ranking or argmax of all candidate grasps.

Rejection: differentiable point-cloud sampling is already the purpose of
SampleNet, L2G learns task-aware point downsampling, and Graspness learns which
scene points deserve expensive grasp processing.  Replacing their downstream
loss by grasp-ranking distortion would be a useful efficiency paper, but not a
new grasping object.  A formal coreset guarantee would also be relative to a
fixed scorer, rather than to physical grasp success.

Sources:

- https://arxiv.org/abs/1912.03663
- https://arxiv.org/abs/2203.05585
- https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.pdf

### 23. Sensor-grounded proof-carrying grasps

Proposal: output a grasp together with a sparse subset of raw RGB-D rays whose
depth intervals prove visible antipodality and collision-free finger occupancy
under bounded sensor noise.  A nonlearned interval verifier would accept or
reject the proof.

Rejection as the top direction: the distinction between a prediction and a
raw-sensor witness is conceptually clean, but coverage is structurally poor in
the stated setting.  The second contact or first-contact surface is often in
the camera/obstacle shadow, so a sound visible-ray certificate must abstain
precisely on many interesting grasps.  If hidden surfaces are supplied by a
network, the proof is no longer sensor-verifiable.  Certified Grasping also
already establishes grasp certificates for known planar geometry; the
remaining novelty would mainly be a 3-D partial-observation verifier and a
selective-prediction protocol.

Source: https://arxiv.org/abs/1909.03985

This remains a potentially publishable narrow safety project, but not the best
match to a broad ICLR thesis.

### 24. Universal grasp-dominance partial order

Proposal: learn

$$
g\succeq_o h
\quad\Longleftrightarrow\quad
Q(z,g)\ge Q(z,h)\quad
\text{for every }z\text{ compatible with }o,
$$

instead of learning scalar scores.

Rejection: dominance only removes actions; it need not leave one executable
maximal action, and empirically most candidates may be incomparable.  More
fundamentally, stochastic-dominance and partial-order learning are established
general decision frameworks.  Specializing them to hidden grasp geometry
returns to the utility-world and robust-selection constructions already
rejected in cycles 5, 6, and 10.

Closest general source: https://arxiv.org/abs/2402.02698

### 25. Mathematical-morphology grasp transform

Proposal: express jaw closing and collision exclusion as a learned hit-or-miss
transform or max-plus convolution between the gripper and observed target.

Rejection: generalized grasp planning has already used voxel
cross-correlation between gripper and object, including collision constraints,
and SpectGRASP uses spectral surface correlation.  Morphological notation
would expose useful algebra but would not change the learned problem.

Sources:

- https://arxiv.org/abs/2006.12676
- https://arxiv.org/abs/2107.12492

### 26. Monotone-operator closure surrogate

Proposal: write terminal jaw closure as a variational inequality and learn its
resolvent with a firmly nonexpansive network, giving a stable differentiable
map from a local contact description to final contacts.

Rejection: learned solution operators for variational inequalities already
cover contact problems, while differentiable grasp simulators already optimize
through contact.  For this project, the operational object is also the learned
closure map rejected in cycle 4.  Monotonicity would be a valuable inductive
bias, not an independently new subproblem.

Sources:

- https://doi.org/10.1007/s40687-022-00327-1
- https://arxiv.org/abs/2208.12250

### 27. Thermodynamic friction-potential grasping

Proposal: predict a convex local dissipation potential from RGB-D and derive
contact forces as its gradient, guaranteeing passive and physically consistent
friction.

Rejection: physically consistent learned friction laws and dissipation
potentials are established in general mechanics, and visual or visuo-haptic
friction estimation has already been used to guide grasp planning.  More
importantly, a single RGB-D image does not identify the contact-pair-specific
friction of an unseen object and gripper pad.  The method would either learn
dataset material priors or require an additional tactile/material sensor.

Sources:

- https://proceedings.mlr.press/v242/dai24a.html
- https://arxiv.org/abs/2010.08277

### 28. Union-to-intersection obstacle semilattice

Proposal: enforce the exact compositional law that adding obstacles intersects
the feasible-grasp set, using per-obstacle factors combined by a minimum.

Rejection: analytic collision filters already have this compositionality, and
ICGNet explicitly motivates instance-level composability.  In the non-cluttered
laboratory setting there is usually only one frontal obstacle, so the proposed
algebra has little room to demonstrate extrapolation.  It is an architecture
constraint around an old collision subproblem.

### 29. Infinitesimal support-release jet

Proposal: for an object initially supported by the shelf, learn the one-sided
derivative of the quasi-static contact equilibrium as the gripper begins a
millimetric lift.  A compact output would include object twist, support-force
unloading rate, and a strict-complementarity margin.

Rejection after a deeper mechanics audit: this is not merely close to
whole-cycle feasibility; the underlying contact-transition object is old.
Early squeeze-grasp mechanics explicitly partition configurations into those
that slide out, jam, or lift from the support.  Shared grasping subsequently
formalized robust environmental contact modes, and modern quasi-static
formulations cover pushing, grasping, and jamming with complementarity
problems.  Learning a KKT derivative would be technically new, but the
scientific problem would remain learned evaluation of a known support-to-hand
contact transition.

Sources:

- https://arxiv.org/abs/1902.03487
- https://arxiv.org/abs/2006.02996
- https://roboticsproceedings.org/rss08/p23.pdf

### 30. Mechanical-similarity equivariant grasp fields

Proposal: use Buckingham-$\Pi$ groups for object scale, jaw stiffness,
closing force, mass, and gravity, and require the grasp field to transform
according to mechanical similarity rather than merely $SE(3)$ equivariance.

Rejection: it is a promising inductive bias for cross-scale transfer, but the
required dimensionless groups contain mass, stiffness, and friction that the
given RGB-D observation does not identify.  With those quantities provided,
the main contribution becomes dimensional-analysis feature engineering; with
them latent, the equivariance is not valid across visually identical objects.
Classical scale-invariant grasp design also prevents a broad "first
scale-invariant grasping" claim:
https://publications.ri.cmu.edu/grasp-invariance-2

### 31. Homogenized contact affordances

Proposal: treat unresolved surface microgeometry as a fast scale and learn the
effective friction/compliance law seen by a finite-area jaw pad, rather than
completing microscopic shape.

Rejection: homogenized friction is mathematically legitimate -- effective
friction need not equal spatially averaged friction -- but the requisite
microtopography is below ordinary wrist RGB-D resolution and material
properties remain unobserved.  Recent work already learns
mechanics-derived rough-surface engagement descriptors from measured
topography.  In the available setup this would be a contact-law surrogate
trained from missing inputs, not a reliable grasp perception problem.

Sources:

- https://arxiv.org/abs/2110.12762
- https://doi.org/10.1016/j.triboint.2026.112411

## Candidate E: LimitGrasp -- learning action-set limits from approximate contact oracles

### Recommendation in one sentence

Treat simulator supervision for contact-rich decisions as a **converging family
of action sets**, not as ground-truth scalar labels: learn the inner limit of
grasps that remain successful under numerical refinement and the outer limit
of grasps that can remain successful, directly from partial RGB-D.

This is the strongest new direction found in the second pass, but it is a
**conditional recommendation**, not a paper-ready claim.  The cheap
multi-resolution simulation audit below has to demonstrate a substantial,
structured label-instability phenomenon before model development.

### The scientific gap

Large grasp datasets routinely turn a simulator run or an analytic metric into
a label.  This silently identifies three different objects:

1. physical success under a fixed terminal close-and-load protocol;
2. the solution of a chosen continuous contact model;
3. the output of one discretization and one numerical contact solver.

For smooth forward problems, treating item 3 as an approximation of item 2 is
often harmless.  Parallel-jaw contact is not smooth: contact creation,
friction-cone activation, edge contact, and slip create switching surfaces in
grasp space.  A timestep, mesh, contact stiffness, collision margin, or solver
tolerance can move those surfaces.  A network trained on one resolution can
therefore learn a *numerical decision boundary* with high validation accuracy
against the same oracle.

The problem is documented on the simulator side but is not made the supervised
learning target:

- comparative contact-model work shows that numerical relaxations can
  severely affect downstream robotics applications:
  https://arxiv.org/abs/2304.06372
- SimBenchmark separates solver error from time-discretization error and
  measures the speed--accuracy curve:
  https://leggedrobotics.github.io/SimBenchmark/
- IPC-GraspSim obtains better physical grasp prediction by modeling compliant
  parallel jaws with a much more accurate contact method, at substantial
  computational cost:
  https://arxiv.org/abs/2111.01391
- Get a Grip reports that merely spawning the hand at a pre-grasp rather than
  already in contact removed nonphysical Isaac Gym behavior and unstable
  simulation labels.  This is direct evidence that the data-generation
  protocol can alter the supervised target:
  https://arxiv.org/abs/2410.23701
- ICLR 2026 accepted work shows that hard-contact settings can yield erroneous
  gradients and that adaptive integration materially changes the result:
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/44039e59aaf6a41b16f1fc5b27bcd409-Abstract-Conference.html

The unoccupied question found in this audit is:

> What should a perception model learn when its physical training label is
> available only through a family of non-uniformly converging numerical
> oracles?

Grasping supplies a consequential, discontinuous instance, but the question is
broader than grasping and distinct from learning a faster simulator.

### Precise scope

- The arm has already placed the open parallel jaw gripper at a queried
  terminal pose.  The operational test is close, hold, and lift/load by a few
  millimetres.
- Approach reachability and full-cycle execution are excluded.
- Input is a target-masked local RGB-D/ray point set, a compact shelf-plane
  descriptor, and a grasp query.  No complete mesh or scene SDF is an input.
- A separate frozen generator proposes grasps.  LimitGrasp evaluates the new
  scientific target; candidate generation is not claimed as a contribution.
- Physical perturbations such as measured pose error are held to one declared
  distribution.  Numerical refinement variables are not randomized and
  averaged as though they were physical properties.

### 1. A refinement family, not a simulator ensemble

Let $z$ be a complete scene used only by the offline simulator, $o=R(z)$
the RGB-D observation, and

$$
\mathcal G = SE(3)/C_2
$$

the parallel-jaw pose space after quotienting the finger-swap symmetry.  Width
can either be included in $g$ or deterministically obtained by closure.

Fix:

- one continuous compliant-contact model $\mathcal M$;
- one terminal execution protocol;
- one distribution $\Xi$ of *physical* pose/control perturbations;
- a collection $\Gamma$ of numerical refinement paths that are intended to
  approximate $\mathcal M$.

A path $\gamma\in\Gamma$ specifies meshes, timesteps, collision
discretizations, tolerances, and solver iterations at levels $k=1,2,\ldots$.
Let $\eta_{\gamma,k}\downarrow0$ denote its resolution scale.  The
operationally smoothed simulator utility is

$$
q_{\gamma,k}(z,g)
=
\Pr_{\xi\sim\Xi}
\left[
Y_{\mathcal M,\gamma,k}(z,g,\xi)=1
\right].
$$

This probability is estimated using matched perturbation seeds across
resolutions.  Matching is important: otherwise ordinary Monte Carlo noise is
mistaken for numerical disagreement.

At a success threshold $\tau$, every level induces a closed action set

$$
S_{\gamma,k}(z;\tau)
=
\overline{\{g\in\mathcal G:q_{\gamma,k}(z,g)\ge\tau\}}.
$$

### 2. The new target: inner and outer action-set limits

For a metric $d_{\mathcal G}$ on the grasp quotient, define the
Painlevé--Kuratowski limits along a refinement path:

$$
\mathrm{Li} S_{\gamma,k}
=
\left\{
g:\limsup_{k\to\infty}
d_{\mathcal G}(g,S_{\gamma,k})=0
\right\},
$$

$$
\mathrm{Ls} S_{\gamma,k}
=
\left\{
g:\liminf_{k\to\infty}
d_{\mathcal G}(g,S_{\gamma,k})=0
\right\}.
$$

The first contains grasps approximable by successful grasps at every
sufficiently fine level.  The second also contains actions that are approached
by success along only a subsequence.  Aggregate declared consistent
refinement paths as

$$
S_{\mathrm{core}}(z;\tau)
=
\bigcap_{\gamma\in\Gamma}
\mathrm{Li}S_{\gamma,k}(z;\tau),
$$

$$
S_{\mathrm{possible}}(z;\tau)
=
\overline{
\bigcup_{\gamma\in\Gamma}
\mathrm{Ls}S_{\gamma,k}(z;\tau)
}.
$$

The difference

$$
B_{\mathrm{num}}(z;\tau)
=
S_{\mathrm{possible}}(z;\tau)
\setminus
S_{\mathrm{core}}(z;\tau)
$$

is the numerically unresolved action band.  If all consistent schemes converge
to the same success set, core and possible sets coincide except at the
decision boundary.  If they do not, returning one scalar "ground truth" hides
an ill-posed label.

This construction is deliberately invariant to every finite prefix of a
refinement sequence.  A very inaccurate coarse simulation can save compute,
but cannot define the scientific target.

### 3. What the RGB-D model predicts

The observation-conditioned fields are

$$
L(o,g)
=
\Pr_{Z\mid o}
\left[g\in S_{\mathrm{core}}(Z;\tau)\right],
\qquad
U(o,g)
=
\Pr_{Z\mid o}
\left[g\in S_{\mathrm{possible}}(Z;\tau)\right].
$$

$L$ is the probability that a queried grasp belongs to the stable success
core; $U-L$ is the probability that it belongs to the numerical ambiguity
band.  Partial visibility and sensor noise remain ordinary input uncertainty;
the target specifically records whether the *same complete scene and grasp*
changes label under numerical refinement.

The decision rule is simply

$$
\hat g(o)=\arg\max_{g\in G(o)}L_\theta(o,g),
$$

possibly with $U_\theta-L_\theta\le\beta$ as a coverage constraint.  No
trajectory, full scene representation, or causal failure taxonomy is
predicted.

### 4. A compact nested-envelope network

Use a point/ray encoder on a fixed physical-radius crop around the queried
gripper closing region.  Cross-attention with a compact gripper query produces
six nonnegative or unconstrained scalars:

$$
a_\theta,\quad b^-_\theta,b^+_\theta,\quad
c^-_\theta,c^+_\theta,\quad p^-_\theta,p^+_\theta.
$$

One parsimonious refinement model is

$$
\ell_\theta(o,g,\eta)
=
\sigma\!\left(
a_\theta-b^-_\theta-c^-_\theta\eta^{p^-_\theta}
\right),
$$

$$
u_\theta(o,g,\eta)
=
\sigma\!\left(
a_\theta+b^+_\theta+c^+_\theta\eta^{p^+_\theta}
\right),
$$

with $b^\pm,c^\pm\ge0$ and $p^\pm$ constrained to a plausible positive
range.  As resolution improves, the lower envelope can only rise and the
upper envelope can only fall.  At the formal limit,

$$
L_\theta(o,g)=\sigma(a_\theta-b^-_\theta),
\qquad
U_\theta(o,g)=\sigma(a_\theta+b^+_\theta).
$$

Here $c^\pm$ describe removable numerical uncertainty, while $b^\pm$
permit a residual gap across refinement paths.  A single scalar
$\eta$ is acceptable only for a prescribed path.  For independent timestep,
mesh, and tolerance coordinates, replace the power law by a small monotone
lattice; do not pretend incomparable discretizations have a canonical scalar
order.

The model remains a candidate-query field with $O(M)$ inference for $M$
grasps.  It does not run a simulator or reconstruct geometry at deployment.

### 5. Finite supervision

For each fixed $(z,g)$, run matched perturbation trials at levels
$(\gamma,k)$.  From the binomial observations form finite-sample lower and
upper confidence bounds

$$
\mathrm{LCB}_{\gamma,k}(z,g),
\qquad
\mathrm{UCB}_{\gamma,k}(z,g).
$$

At refinement level $k$, empirical tail envelopes are

$$
\widehat\ell_k(z,g)
=
\min_{\gamma,\;j\ge k}
\mathrm{LCB}_{\gamma,j}(z,g),
$$

$$
\widehat u_k(z,g)
=
\max_{\gamma,\;j\ge k}
\mathrm{UCB}_{\gamma,j}(z,g).
$$

By construction, $\widehat\ell_k$ is nondecreasing and
$\widehat u_k$ nonincreasing with $k$.  Train the nested-envelope field on
all levels with a weighted proper binomial loss plus envelope regression.  A
width penalty may be applied only subject to held-out finer-level coverage;
otherwise the network can win by becoming confidently narrow.

Most labels need not use the finest solver:

1. evaluate every scene/grasp with one cheap level;
2. refine a stratified subset spanning score, object geometry, occlusion, and
   observed solver margin;
3. allocate the most expensive level to items whose predicted tail interval
   crosses $\tau$ or whose observed labels fail to stabilize;
4. keep a uniformly sampled finest-level subset for unbiased evaluation of the
   allocation rule.

This is offline experimental design, not RL.

### 6. The necessary mathematical honesty

No finite collection of approximate labels identifies an infinite-resolution
limit without assumptions.  This should be a theorem in the paper, not hidden
in limitations:

> For any finite observed prefix of simulator outputs, there exist two
> continuation sequences agreeing on that prefix but having different
> Kuratowski limits.

Consequently, LimitGrasp must declare and test a convergence class.  A
reasonable local assumption away from contact-mode intersections is an
asymptotic expansion of the signed distance to the success set,

$$
d_{\mathcal G}(g,S_{\gamma,k})_{\mathrm{signed}}
=
d_{\gamma,\infty}(g)
+c_\gamma(g)\eta_{\gamma,k}^{p_\gamma(g)}
+o(\eta_{\gamma,k}^{p_\gamma(g)}).
$$

At mode intersections this expansion may fail; that is exactly where the
inner/outer set formulation is preferable to extrapolating a scalar label.
Held-out levels finer than every training level are therefore mandatory.

Potential formal results are:

1. **Finite-prefix invariance.**  Inner and outer limits are unchanged by any
   finite set of coarse levels, unlike majority vote or a fixed-resolution
   label.
2. **Set consistency.**  Under graph convergence of the numerical contact
   solutions, regularity of the threshold away from a null boundary, and
   uniform convergence of learned signed-distance envelopes, the predicted
   core/possible sets converge in Kuratowski distance; with compactness and
   stronger regularity this upgrades to Hausdorff convergence.
3. **Selection stability.**  If
   $\sup_g|\widehat L(o,g)-L(o,g)|\le\delta$, maximizing
   $\widehat L$ over a fixed candidate set has at most $2\delta$ excess
   stable-core regret.
4. **Refinement stopping.**  If a validated numerical error bound keeps a
   candidate's entire envelope on one side of $\tau$, finer simulation
   cannot change its core classification and can be skipped.
5. **Solver-selection dependence of ERM.**  When core and possible sets differ
   on positive measure, ordinary ERM trained from one solver converges to a
   solver-dependent target even with infinite data.

The first, third, and fifth statements should be elementary.  The second is
where genuine technical care is required; it must not be overstated for
frictional rigid contact without verified convergence hypotheses.

### 7. Why this is not an occupied neighboring method

| Neighbor | Its target | Difference in Candidate E |
|---|---|---|
| IPC-GraspSim / improved contact solvers | make one simulator more physically accurate | learn the stable action-set limit exposed by refinement, without proposing a solver |
| contact-engine benchmarks | compare physical and computational errors on prescribed mechanics tests | predict perception-conditioned grasp action sets and test selection on real objects |
| DiffMJX / differentiable simulation | improve gradients through hard contact | forward success-label convergence, not gradient usefulness |
| multi-fidelity regression | predict a designated highest-fidelity output cheaply | no finite fidelity is declared truth; learn an inner/outer asymptotic set |
| discretization-invariant neural operators | make a learned operator commute with input/output resampling | study discretization of the **label oracle** and a discontinuous decision set |
| domain randomization | average performance over a chosen physical nuisance distribution | numerical settings are approximation errors to remove or expose, not deployment randomness |
| simulator ensembles | reduce variance or hedge over engines | refinement paths must share a declared continuous model; arbitrary engines are not votes |
| robust grasp planning | hedge over pose, shape, friction, or load uncertainty | hold physical uncertainty fixed and isolate numerical-label uncertainty |

Relevant general boundaries:

- multi-fidelity active learning:
  https://proceedings.mlr.press/v202/wu23p.html
- discretization invariance in operator learning:
  https://iclr-blogposts.github.io/2026/blog/2026/discretisation-invariance/

The distinction will be credible only if experiments separately vary physical
nuisance parameters and numerical approximation parameters.

### 8. Benchmark and experimental protocol

#### Simulation matrix

Start with 200--500 rigid objects, including ordinary household shapes and
adversarial contact geometry: thin rims, rounded edges, small chamfers,
near-parallel faces, narrow necks, and shallow concavities.  Render a single
wrist RGB-D view with:

- no occlusion, mild frontal occlusion, and severe but non-total frontal
  occlusion;
- measured depth quantization, dropout, and pose noise;
- an explicit target mask and shelf plane;
- no clutter beyond the one occluder.

Generate 256--512 grasps from the same frozen observation-only generator.
Every selected full-scene/grasp pair is then evaluated with matched physical
perturbation seeds over a refinement grid:

- timestep, including at least a $4\times$ ratio between adjacent levels;
- collision and compliant-pad mesh resolution;
- contact tolerance/collision margin;
- nonlinear/contact solver tolerance and iteration budget;
- at least two independently implemented *consistent* paths where model
  matching can be defended.

Do not mix rigid point-contact, penalty contact, and compliant-pad mechanics
inside one alleged numerical path.  Those are different continuous models.
A secondary experiment may compare their separate limit sets as model-form
uncertainty, but it is not the primary target.

#### Real protocol

Use the actual parallel jaw gripper with repeatable prepositioning.  For each
test:

1. move to a collision-free pre-grasp by an external planner;
2. start scoring only at the standardized terminal open pose;
3. close with fixed force/velocity control;
4. lift or load by the same few millimetres;
5. record retention and relative slip.

Select a balanced physical subset from numerically stable successes, stable
failures, and the ambiguity band.  Repeat pose perturbations rather than
testing each grasp once.  The decisive hypothesis is that ambiguity-band
membership predicts a default simulator's real errors beyond its scalar score,
geometry class, and analytic force-closure margin.

#### Metrics

- label flip rate versus each numerical coordinate and refinement depth;
- distance between successive success sets on the grasp quotient;
- size and geometry of the estimated ambiguity band;
- coverage of the predicted envelope on unseen finer levels;
- width of that envelope at fixed coverage;
- precision and recall of stable-core membership;
- real grasp success, slip, and selection regret;
- simulation calls and wall-clock cost;
- results stratified by occlusion, PCD noise, object geometry, and score
  margin.

Essential baselines:

- one default simulator/fidelity;
- the finest available level for every training item;
- mean and worst-case labels over an arbitrary simulator ensemble;
- physical-parameter domain randomization;
- deep ensemble on single-fidelity labels;
- a scalable multi-fidelity GP or deep multi-fidelity regressor;
- a direct model predicting the finest observed label;
- analytic force closure and a standard learned grasp scorer;
- an oracle using all refinement levels.

All learned baselines must share the RGB-D encoder, candidate set, and
simulation budget where applicable.

### 9. Cheap falsification before building the neural model

Run this audit on only 30--50 objects and roughly 5,000 geometrically diverse
grasps:

1. Use at least four nested timestep/tolerance levels and two mesh levels with
   matched perturbation seeds.
2. Verify that contact parameters describe the same intended continuous model;
   remove disagreements caused by unit errors, inconsistent friction
   conventions, or different execution controllers.
3. Measure the default-to-finest label flip rate.  If it is below 5% overall
   and below 10% even in low-margin strata, stop.
4. Require a structured ambiguity band: flips should concentrate near
   contact-mode or geometric margins and should be reproducible across random
   seeds.  Unstructured nondeterministic engine noise is not the proposed
   phenomenon.
5. Test a simple tail-envelope extrapolator on a withheld finer level.  At
   least 90% coverage with mean probability width below 0.20 on at least 70%
   of items is a reasonable proceed threshold.
6. Execute a small real balanced set.  After conditioning on default
   simulator score, ambiguity-band membership must still predict error.  If it
   does not, numerical convergence is scientifically interesting but not
   useful for grasp selection.
7. Compare equal-compute strategies.  If "run IPC-GraspSim once on fewer
   examples" or "use the finest solver only" matches the learned limit model,
   the amortization thesis is false.
8. Check that each image has at least one candidate with high stable-core
   probability.  Otherwise lower-limit selection is vacuously conservative.

Only a positive result on items 3, 5, 6, and 7 justifies a full paper.

### 10. ICLR acceptance audit

The current ICLR 2027 reviewer guide asks whether the paper studies a specific,
well-motivated problem, supports its claims rigorously, and creates significant
new knowledge or value; it explicitly asks reviewers to be open to surprising
work that may take the field in a new direction rather than requiring an
established leaderboard result:
https://iclr.cc/Conferences/2027/ReviewerGuidelines

#### Potential strengths

1. **Original learning object.**  The target is a limit of decision sets
   generated by approximate numerical oracles, not another grasp score,
   completion field, or trajectory feasibility head.
2. **Broad relevance.**  The same issue appears wherever learned decisions are
   supervised by nonsmooth simulators: legged contact, assembly, deformable
   manipulation, fracture, and collision.
3. **Tight theory--experiment link.**  Finite-prefix non-identifiability,
   set convergence, nested envelopes, and action-selection bounds directly
   determine the data collection and metrics.
4. **Practical efficiency.**  Expensive refined simulation is used only on a
   subset offline; deployment remains one local candidate-query evaluation.
5. **Clean compliance with project constraints.**  No RL, VLA, full-cycle
   feasibility, causal failure modes, or scene SDF is required.

#### Principal rejection risks

1. **Phenomenon risk.**  Carefully calibrated compliant simulation may already
   make action labels stable.  Then the paper manufactures a problem from bad
   simulator settings.
2. **Ground-truth risk.**  Different engines often encode different physical
   contact models, not different discretizations of one model.  Calling their
   intersection a continuum limit would be mathematically wrong.
3. **Method risk.**  If the result is only min/max labels plus a two-headed
   network, reviewers will call it robust multi-fidelity regression.  The
   set-limit target, unseen-refinement evaluation, adaptive data allocation,
   and theory must all matter empirically.
4. **Reality risk.**  Numerical stability need not imply physical accuracy.
   A consistently wrong solver is stable.  A real balanced ambiguity-band
   experiment is essential.
5. **Cost risk.**  Producing paired compliant-pad simulations over multiple
   refinements may be more expensive than collecting enough real grasps.
6. **Scope risk.**  A grasp-only benchmark is unlikely to establish the broad
   ML claim.  Add at least one inexpensive non-grasp contact benchmark, such
   as a frictional block/peg or planar complementarity system, where the
   continuous solution or a verified finest reference is known.
7. **Asymptotic-model risk.**  Power-law convergence can fail at contact-mode
   switches.  Report failure regions rather than forcing false extrapolation.

### 11. Minimum paper that would be defensible

A credible paper would make four contributions:

1. formulate **learning action-set limits from approximate oracles** and prove
   the non-identifiability and selection results;
2. introduce a nested-envelope estimator with budget-aware adaptive
   refinement and evaluation on unseen finer oracle levels;
3. release a paired multi-resolution parallel-jaw grasp benchmark from noisy,
   partially occluded RGB-D;
4. show on real hardware that selecting from the learned stable core improves
   small-lift reliability at equal simulation and inference cost.

A possible title is:

> **LimitGrasp: Learning Stable Action Sets from Non-Converged Contact
> Simulators**

The strongest honest abstract claim would be:

> Contact-rich learned decision systems are commonly supervised by one
> numerical simulator setting.  We show that near contact-mode boundaries the
> induced successful-action set need not be stable under refinement, making
> ordinary labels solver-dependent.  We formulate supervision through the
> inner and outer limits of refining action sets, propose an amortized nested
> estimator from partial observations, and demonstrate that its stable core
> transfers better to finer solvers and physical parallel-jaw grasping.

The claim must remain conditional until the falsification audit establishes
that the ambiguity band is nontrivial, learnable, and predictive of real
failure.

### 12. Submission-timing and AI-disclosure note

As of 2026-08-24, the ICLR 2027 abstract and paper deadlines are 2026-09-18
and 2026-09-25 AOE:
https://iclr.cc/Conferences/2027/AuthorGuidelines

Unless the laboratory already has a matched multi-resolution simulation
pipeline and real grasp data, Candidate E cannot be validated to the standard
claimed above in one month.  The scientifically responsible target is ICLR
2028 rather than a rushed ICLR 2027 submission.

ICLR 2027 also requires an AI-use statement.  This research-ideation process
is significant enough that it must be described precisely in that statement;
the human authors remain responsible for verifying every novelty, citation,
theorem, and empirical claim:
https://iclr.cc/Conferences/2027/AIPolicyForAuthors

A standalone Russian-language proposal with the full mathematical model,
benchmark, falsification gates, mock review, and execution order is available
in reports/LimitGrasp.md.

## Third independent search pass: acquisition law versus physical surface

This pass deliberately excluded Candidates D and E and all 31 earlier cycles.
It asked a different question: can the learning problem be defined on a
physical surface while the network is evaluated only on a finite, noisy RGB-D
acquisition of that surface?  This is not the usual permutation-invariance
question.  Reordering a point set does not change it, but changing pixel
resolution, point budget, range-dependent noise, or the sampling density does.

The most important literature boundary is the following.

- PointConv and Monte Carlo Convolution already treat point convolution as
  quadrature under non-uniform sampling and use inverse-density correction:
  https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_PointConv_Deep_Convolutional_Networks_on_3D_Point_Clouds_CVPR_2019_paper.pdf
  and https://arxiv.org/abs/1806.01759.  Therefore, merely adding density
  weights to a grasp network is not novel.
- Neural operators explicitly target discretization-consistent maps between
  function spaces.  GINO is one relevant construction:
  https://arxiv.org/abs/2309.00583.  A recent overview makes the role of
  quadrature weights and sampling structure explicit:
  https://www.nature.com/articles/s42256-026-01267-z.  Therefore, merely
  replacing PointNet with a neural operator is not a new problem.
- The phrase "acquisition-invariant representation" is not globally new;
  scanner/protocol invariance was already studied for MRI with Siamese
  representation learning: https://arxiv.org/abs/1810.07430.  Any novelty claim
  here must concern the declared point-process law, noisy contact functionals,
  and decision transfer--not the adjective "acquisition-invariant."
- Hard support recovery under fixed Gaussian measurement noise is a severely
  ill-posed statistical problem with only polylogarithmic rates in broad
  classes:
  https://arxiv.org/abs/1804.09879.  Thus a claim that a network recovers exact
  infinitesimal contacts from noisy samples would be mathematically suspect.
- Existing grasp systems recognize sensor noise, but primarily through data
  repair, fusion, augmentation, or ordinary uncertainty prediction.  For
  example, R2SGrasp repairs real depth before grasp detection:
  https://isee-laboratory.github.io/R2SGrasp/, while the CVPR 2024 domain-prior
  method combines contact priors and score optimization:
  https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Generalizing_6-DoF_Grasp_Detection_via_Domain_Prior_Knowledge_CVPR_2024_paper.pdf.
- The 2026 boundary is still dominated by stronger sensing/backbones and
  geometry-aware heads rather than a sensor-law quotient.  GA-Grasp, for
  example, uses RGB, depth, and normals for transparent and ordinary objects:
  https://openreview.net/pdf?id=8wm1HEfPss.
- Direct grasping under occlusion is already a named benchmark problem.
  TARGO explicitly uses a single RGB-D observation rather than next-best-view
  planning:
  https://targo-benchmark.github.io/.  The present pass therefore does not
  claim that partial target visibility itself is new.

### 32. Projectively coherent grasp fields

The first attempted formulation required predictions from nested point
subsets to form a martingale or a projectively consistent family.  It was
rejected.  Martingale posterior neural processes already establish the general
probabilistic mechanism (https://arxiv.org/abs/2304.09431), and recent work
explicitly identifies conditioning consistency as a general predictive-model
property (https://arxiv.org/abs/2604.19312).  In a single-frame grasping task,
the result would be a consistency regularizer with no uniquely grasp-specific
learning target.

### 33. One-bit persistent hand--eye calibration from grasp outcomes

The proposed model treated an unknown persistent $\Delta\in SE(3)$ as a
shared latent variable:

$$
 y_i\sim\mathrm{Bernoulli}
 \left(q_\theta(O_i,\exp(\delta)g_i)\right),
 \qquad \Delta=\exp(\delta),
$$

and used the Fisher information of the frozen grasp field to update a compact
posterior over $\delta$ from binary successes.  The idea is efficient and
would yield a useful system, but it is standard logistic M-estimation/system
identification on a Lie group.  Learned uncalibrated hand--eye coordination is
old enough to be a canonical reference
(https://journals.sagepub.com/doi/10.1177/0278364917710318), and modern
self-calibration and calibration-free camera-centric policies further weaken
the novelty claim.  A local asymptotic-normality or Fisher-rank theorem would
not make this an ICLR-level new learning problem.

### 34. A generic discretization-invariant grasp neural operator

This cycle represented the visible surface as a measure $\mu$, learned an
operator $T:\mu\mapsto q_\mu(\cdot)\in C(\mathcal G)$, and required the
argmax to converge as both input samples and output grasp queries were
refined.  It was rejected in this generic form.  Neural-operator
discretization invariance is mature, and the additional argmax statement
follows from standard uniform approximation plus a margin condition.  Without
a contact-specific statistical obstruction and a new estimator, reviewers
could accurately describe the work as GINO applied to grasp scoring.

### 35. One-step decision-value next-best view

A non-RL version selected one wrist-camera motion by expected downstream grasp
regret rather than reconstruction entropy.  It was rejected because the
robotics direction is already densely occupied.  ACE-NBV directly predicts
grasp affordance at imagined views
(https://proceedings.mlr.press/v229/zhang23i.html), closed-loop target grasping
with a wrist camera is established (https://arxiv.org/abs/2207.10543), and a
NeurIPS 2024 paper learns a neural grasp field for active perception:
https://papers.nips.cc/paper/2024/file/4364fef031fdf7bfd9d1c9c56b287084-Paper-Conference.pdf.
Changing information gain to expected decision value would be sensible, but
not enough to define a new area.

### 36. Ray--gripper incidence learning

This cycle represented each RGB-D pixel as a finite free-space ray plus a hit
and represented a gripper by a small family of swept line/tube primitives.
Pairwise Pluecker invariants would drive an $SE(3)$-invariant cross-attention
operator.  It was rejected as the main idea for three reasons.  First, for one
calibrated camera, endpoint plus camera origin already determines the ray, so
the representation does not add information.  Second, Pluecker ray
conditioning has just entered robot learning: RayViT reports camera-robust
manipulation using ray maps (https://arxiv.org/abs/2607.29622), and camera
conditioning with per-pixel Pluecker rays is already an ICRA 2026 result:
https://ripl.github.io/know_your_camera/.  Third, a ray--action attention block
would be an attractive inductive bias, but presently looks like an
architecture contribution rather than a new supervised target.

### 37. Proprioceptive caliper tomography

The terminal jaw width after closure is a cheap scalar measurement of a local
object caliper.  This suggested training a visual continuous caliper transform
from sparse real attempts.  It was rejected as a standalone formulation.  A
symmetric parallel-jaw mechanism usually reveals the sum of two contact
depths, not the individual left/right contact positions or their normals.
Missed unilateral contact is also censored.  Width and binary lift outcome can
be useful auxiliary supervision, but the inverse problem is too
underidentified to support the central scientific claim without tactile or
independent-finger sensing.  At that point it becomes a visual-haptic learning
paper, an already substantial neighboring field.

## Candidate F: AcqGrasp -- acquisition-equivalent contact learning

### Recommendation in one sentence

Study whether a contact decision can be learned as a function of the latent
visible surface rather than of the RGB-D sensor's finite sampling law, using a
deconvolved quadrature layer for finite-scale contact functionals and a
paired-acquisition benchmark that can falsify the premise before a large model
is trained.

This is the strongest new candidate from the third pass, but it remains a
conditional recommendation.  Its ICLR claim survives only if ordinary grasp
rankings materially change under physically irrelevant acquisition changes
and if PointConv plus strong augmentation does not remove the effect.

### 1. The broad open problem

Most point-cloud grasp models are permutation invariant.  Permutation
invariance says

$$
 f(\{x_1,\ldots,x_n\})=
 f(\{x_{\pi(1)},\ldots,x_{\pi(n)}\}),
$$

but it does not say that the result is unchanged when one surface region is
sampled four times more densely, when the FPS budget changes, or when the
depth-noise variance changes.  In fact, an unweighted sum converges to the
sensor-weighted measure $\lambda\mu$, not the declared physical reference
measure $\mu$.  A max over points is worse: with unbounded Gaussian depth
noise it grows with sample count even when the physical surface is fixed.

The proposed general problem is **learning decisions on acquisition-law
equivalence classes**.  Two finite observations are not required to be
identical pathwise.  They are equivalent in law when they arise from the same
latent visible surface through different declared sampling intensities and
measurement kernels.  A predictor is acquisition consistent if its output
converges to the same surface-level decision as either acquisition is
refined.

This is broader than grasping.  It applies to learned collision, insertion,
inspection, and other geometric decisions whose inputs are samples but whose
targets are properties of a physical object.  Parallel-jaw contact is a sharp
test because it depends on near-extreme local geometry, exactly where
sampling-density and noise effects are amplified.

### 2. Observation model and precise scope

Let $S\subset\mathbb R^3$ be the target's visible surface and let
$\mu_S$ be a declared reference measure.  Surface area is one choice; the
pushforward of uniform image-plane area is another.  The choice must be fixed
and reported, because "density invariant" is meaningless without a reference
measure.

An acquisition $a$ produces a marked point process

$$
 O_a=\{(x_i,\omega_i,\Sigma_i,r_i)\}_{i=1}^{N_a},
 \qquad x_i=s_i+\varepsilon_i,
 \qquad \varepsilon_i\mid s_i\sim
 \mathcal N(0,\Sigma_i).
$$

Here $s_i\in S$, $\omega_i$ is a quadrature/inverse-propensity weight,
$\Sigma_i$ is the calibrated depth uncertainty mapped into 3-D, and $r_i$
may hold RGB or a target/obstacle mark.  A convenient theoretical model is an
inhomogeneous Poisson process with intensity
$n_a\lambda_a(s)\mu_S(ds)$, for which
$\omega_i=(n_a\lambda_a(s_i))^{-1}$.  A real RGB-D implementation can carry
pixel-footprint weights through cropping and FPS instead of trying to recover
them after preprocessing.

The task is candidate-local.  Given a parallel-jaw pose
$g\in\mathcal G=SE(3)/H$, where $H$ is the finite physical gripper
symmetry, predict

$$
 q^*(S,g)=P(Y=1\mid S,g)
$$

for a standardized close and millimetre-scale lift.  Motion planning to the
pre-grasp is external.  The model does not estimate full approach-to-lift
executability, does not reconstruct a scene SDF, and does not model causal
failure modes.  Obstacle points can be marked and queried locally, but clutter
is outside the proposed benchmark.

### 3. Why hard contact geometry is the wrong statistical target

For closure direction $u_g$, a naive contact proxy is

$$
 \widehat h_{\max}(u_g)=\max_i u_g^\top x_i.
$$

If $u_g^\top\varepsilon_i\sim\mathcal N(0,\sigma^2)$, then its noise
contribution is of order $\sigma\sqrt{2\log N_a}$.  Increasing the point
budget can therefore make the estimated object larger.  Exact recovery of a
hard support under fixed Gaussian error is also known to have extremely slow
rates.  The proposal consequently does **not** claim to recover infinitesimal
contacts.

Instead define contact at a finite physical/statistical scale $\tau>0$.  For
a smooth gripper-local gate $\psi_{g,k}$ and direction $u_{g,k}$, use the
entropic support functional

$$
 H_{k,\tau}(S,g)
 =\tau\log
 \frac{\int_S \psi_{g,k}(s)
       \exp(u_{g,k}^\top s/\tau)\,d\mu_S(s)}
      {\int_S \psi_{g,k}(s)\,d\mu_S(s)}.
$$

As $\tau\downarrow0$, this approaches a local support value under ordinary
regularity conditions.  At finite $\tau$, it is a contact-scale statistic:
it ignores sub-resolution spikes, represents a finite pad rather than an
ideal point contact, and remains estimable.  Several scales
$\tau_1>\cdots>\tau_L$, bounded below by the calibrated noise level, expose
the useful geometry without pretending the inverse problem is well posed.

The central statistical hypothesis is not "more smoothing is robust."  It is
that the task-relevant contact decision factors through a small collection of
finite-scale, action-conditioned surface functionals that admit correction
for the acquisition process.

### 4. Deconvolved quadrature contact layer

Let $\kappa_{g,k}:\mathbb R^3\to\mathbb R$ be a smooth localized kernel in
the gripper frame.  It may be a Gaussian gate multiplied by an exponential
linear form, a soft jaw slab, or a soft pad-boundary probe.  Define

$$
 Z_k(S,g)=\int_S \kappa_{g,k}(s)\,d\mu_S(s).
$$

For Gaussian measurement error, convolution by the noise kernel is the heat
operator.  Denote the corresponding inverse, restricted to resolvable smooth
kernels, by

$$
 \mathcal D_{\Sigma}
 =\exp\!\left(-\tfrac12\Sigma:\nabla^2\right).
$$

Then the proposed acquisition-corrected estimator is

$$
 \widehat Z_k(O_a,g)
 =\sum_{i=1}^{N_a}\omega_i
   [\mathcal D_{R_g^\top\Sigma_iR_g}\kappa_k]
   (g^{-1}x_i).
$$

For the pure exponential kernel
$\kappa(x)=\exp(tu^\top x)$, this is the cheap closed-form correction

$$
 [\mathcal D_\Sigma\kappa](x)
 =\exp\!\left(tu^\top x-
              \tfrac12t^2u^\top\Sigma u\right).
$$

Thus the geometric core is a weighted reduction, not a reconstructed field or
an iterative inverse problem.  Gaussian-exponential bases also have analytic
corrections, provided their spatial bandwidth is larger than the measurement
blur.  The resolvable-bandwidth constraint is essential: unconstrained inverse
heat flow would amplify high-frequency noise and make the proposal invalid.

The layer produces a compact vector

$$
 z(O_a,g)=
 [\widehat Z_1,\ldots,\widehat Z_K,
  \widehat H_{1,\tau_1},\ldots,
  \widehat H_{L,\tau_L},m_g],
$$

where $m_g$ contains only small declared gripper parameters such as opening
and pad size.  Separate bases are tied to the two closure directions, pad
interiors, fingertip boundaries, and the target/obstacle mark.  A small MLP or
low-rank cross-query block maps $z$ to success probability and an optional
local pose residual.

### 5. Efficient learnable model

The proposed model, provisionally called **AcqGrasp**, has four parts.

1. A light RGB-D encoder attaches target/obstacle features and, if the camera
   does not expose a confidence model, a calibrated residual variance to each
   retained point.  The geometric correction uses the device model whenever
   possible; it must not hide all uncertainty inside a learned scalar.
2. A fixed-radius spatial index retrieves only points intersecting the open
   jaw and pad neighborhoods of each candidate.
3. The deconvolved quadrature contact layer evaluates $K$ smooth kernels at
   $L$ resolvable scales in the candidate frame.
4. A shared query head returns $q_\theta(O,g)$.  Candidate generation is held
   fixed across methods so the paper tests the new learning problem rather
   than a pipeline advantage.

With $M$ candidates, $k$ local points, and a small fixed number $KL$ of
kernels, the dominant query cost is $O(MkKL)$, parallelizable as segmented
reductions.  Reasonable first values are $k=64$, $K=8$--16, and
$L=3$--4.  No voxel grid, scene SDF, mesh completion, diffusion sampling, or
trajectory roll-out is required.

Training uses ordinary supervised labels plus paired acquisitions of the same
latent scene:

$$
 \mathcal L=
 \mathcal L_{\rm BCE}(y,q_\theta(O_a,g))
 +\lambda_{\rm acq}
  \mathbb E_{a,b,g}
  [q_\theta(O_a,g)-q_\theta(O_b,g)]^2
 +\lambda_{\rm rank}\mathcal L_{\rm rank}.
$$

The paired term is not the proposed contribution by itself.  Its role is to
penalize finite-sample residuals after the analytical correction.  Training
must include unseen-acquisition tests; otherwise augmentation can memorize the
finite rendering menu.

### 6. Theory that the paper must actually prove

The following are plausible theorem targets, not results that may be asserted
without proof.

1. **Acquisition unbiasedness.**  Under the marked Poisson model, correct
   inverse-intensity weights, a known Gaussian kernel, and an admissible
   $\kappa$, show
   $\mathbb E[\widehat Z_k\mid S]=Z_k(S,g)$ for every acquisition law in the
   declared family.
2. **Uniform query consistency.**  For a compact grasp set and a bounded
   Lipschitz kernel family, obtain a high-probability uniform error bound over
   $g$.  The effective sample size must expose weight degeneracy; a bound in
   nominal $N$ alone would be misleading.
3. **Hard-extreme obstruction.**  Formalize the divergence or sample-count
   bias of naive max contact features under fixed unbounded noise, and relate
   it to the slow support-estimation boundary rather than presenting only an
   empirical anecdote.
4. **Scale tradeoff.**  Bound soft-support approximation bias against the
   variance amplification of deconvolution.  The useful conclusion should be
   a lower resolvable $\tau(\sigma,n_{\rm eff})$, not an impossible promise
   that $\tau\to0$ cheaply.
5. **Decision transfer.**  If the learned head is Lipschitz in the sketches
   and the physical score has a stated margin around its optimal set, turn the
   uniform sketch error into grasp regret or distance-to-optimal-set bounds.
6. **Misspecified acquisition law.**  Give a perturbation bound in errors of
   $\omega_i$ and $\Sigma_i$.  Real cameras will violate the ideal model;
   robustness to moderate misspecification is necessary for the theory to
   mean anything.

The first and third statements are relatively elementary.  They are not
enough for ICLR.  The paper needs a nontrivial uniform/scale/decision result
that explains a successful method design choice.

### 7. Why this is not the neighboring occupied work

#### Versus PointConv and Monte Carlo Convolution

Those methods density-correct a continuous convolution.  AcqGrasp must add
three things that are absent from that contribution: an explicit marked sensor
law including measurement error, finite-scale near-extreme contact
functionals queried by actions, and a decision-level consistency target and
benchmark.  If density weighting alone matches the full method, this
candidate has no novelty.

#### Versus neural operators

Generic neural operators start from samples or quadrature of a function and
seek discretization-consistent function-space maps.  AcqGrasp focuses on an
inverse observation operator: non-uniform surface sampling followed by a
known noisy measurement kernel, with an ill-posed hard-contact limit.  The
output is a continuous action query, but merely using GINO is explicitly not
the contribution.

#### Versus depth denoising and R2S repair

A denoiser reconstructs a cleaner point cloud and then applies an ordinary
detector.  The proposed layer estimates only the small contact functionals
needed by the current action, corrects them in expectation, and never claims a
globally repaired surface.  Equal-compute denoising baselines remain
essential.

#### Versus certified point-cloud robustness

PointGuard (https://arxiv.org/abs/2103.03046) and 3DeformRS/3DCertify-style
work study certificates to bounded point changes or transformations.  The
present target is not a worst-case certificate around one finite cloud; it is
consistency across stochastic acquisitions of the same latent surface.

#### Versus noisy convex-support estimation

The statistical literature estimates an entire convex support from noisy
samples.  AcqGrasp deliberately avoids that hard target.  It estimates a few
regularized, local, action-conditioned functionals and learns the decision
that factors through them.  The older lower-rate results are an obstruction
and design guide, not the proposed method.

### 8. Paired-acquisition benchmark

The unit of evaluation must be a latent physical scene with multiple
acquisitions, not unrelated point clouds at several resolutions.

For each isolated target-on-shelf scene, with a separate optional frontal
obstacle, keep the object, pose, candidate set, and physics label fixed.  Render
or derive a factorial acquisition family over:

- native image resolution and pixel binning;
- uniform thinning, FPS, voxel sampling, and locally non-uniform thinning;
- point budgets spanning at least an order of magnitude;
- heteroscedastic axial depth noise and a small outlier component;
- missing-depth patterns and edge noise;
- calibrated versus moderately misspecified noise/weight marks.

Do not call different viewpoints equivalent: a new viewpoint reveals a new
visible surface and therefore adds information.  Viewpoint generalization can
be reported separately, but the primary paired test changes the acquisition
of the same visible surface.

Physics labels must be generated once from the latent mesh or collected once
for the physical scene and reused across its acquisitions.  Otherwise label
noise is confounded with sensor-law sensitivity.  Evaluation should use a
common, mesh-derived candidate set for all models when measuring acquisition
consistency; each model may additionally be evaluated end to end.

Core metrics are:

- acquisition consistency error of $q(O_a,g)$ over paired $a,b$;
- top-1 and top-$k$ grasp identity/ranking stability;
- regret relative to the fixed latent-physics candidate oracle;
- ordinary AP/AUC and small-lift success;
- calibration stratified by point budget and noise level;
- error versus effective, not nominal, sample size;
- latency and memory;
- results stratified by target visibility, obstacle presence, local curvature,
  and contact scale.

Essential baselines include PointNet++/sparse-point encoders, PointConv or
Monte Carlo Convolution, a quadrature-aware neural operator, noise
augmentation, point-cloud denoising, R2S-style repair, unweighted soft extrema,
density-only correction, noise-only correction, robust quantiles, and an
oracle with clean mesh contact features.

### 9. Cheap falsification before full development

The idea should be killed quickly unless the following pilot succeeds.

1. Use 20--30 CAD objects, exact fixed candidate sets, and simple analytic or
   high-confidence simulator labels.  Generate at least 12 acquisitions per
   fixed scene across four point budgets and three noise levels.
2. First measure the phenomenon without a new model.  Regress baseline scores
   on point count conditional on latent scene and grasp.  Require a material
   conditional acquisition effect, not just lower accuracy at low resolution.
3. Require either a greater than five percentage-point latent-label regret
   gap or a greater than twenty percentage-point top-10 overlap loss between
   realistic acquisition extremes.  If rankings are already stable, stop.
4. Implement only four evaluators: the original baseline, strong paired
   augmentation, PointConv/density correction, and the fixed analytical
   deconvolved contact sketch plus a small head.  Do not build a new backbone.
5. Proceed only if the contact sketch removes at least half of the paired
   consistency error and improves regret on an unseen acquisition law, while
   losing no more than two points on the clean/native condition.
6. Replace calibrated $\Sigma$ and weights by values perturbed by 10--30%.
   If gains vanish under mild misspecification, the real-camera proposal is
   too brittle.
7. Reprocess the same real RGB-D frames at several budgets to test pure
   discretization, then collect repeated frames of fixed scenes to test
   measurement noise.  Only after both pass should physical grasps be run.
8. On hardware, use a balanced set of candidates whose baseline ranking is
   acquisition-sensitive.  If the allegedly stable selection does not improve
   repeated millimetre-lift success, stop.

The most dangerous negative result is that FPS to a fixed budget plus ordinary
noise augmentation already removes the effect.  That outcome should terminate
the paper rather than motivate a larger network.

### 10. Adversarial ICLR audit

#### Strongest case for acceptance

1. The work defines a nuisance transformation that current point-cloud
   invariance language usually omits: a change in the sampling and measurement
   law, not a permutation or rigid transform.
2. It identifies a task-specific statistical obstruction: contact uses
   near-extreme geometry, for which more noisy points can make a naive estimate
   worse.
3. It proposes a compact differentiable estimator derived from quadrature and
   inverse heat flow, rather than a larger grasp architecture.
4. Theory, paired data, and decision metrics test the same claim.
5. The method is efficient and respects the laboratory scope: a local
   candidate query from noisy RGB-D, no RL/VLA, no full scene SDF, no full-cycle
   feasibility, and no causal failure taxonomy.
6. The acquisition-equivalence problem can plausibly transfer to insertion,
   collision, and other sampled-geometry decisions, giving it ICLR relevance
   beyond one gripper.

#### Strongest case for rejection

1. **Composition criticism.**  A reviewer can summarize the components as
   inverse-density Monte Carlo integration, classical Gaussian deconvolution,
   log-sum-exp, and a grasp MLP.  The new problem definition, scale theorem,
   and benchmark must create knowledge that is not reducible to this list.
2. **Phenomenon criticism.**  Modern pipelines often use FPS to a fixed count.
   Sensor-law sensitivity may be too small after standard preprocessing.
3. **Model criticism.**  Real depth errors are correlated, biased near edges,
   material dependent, and non-Gaussian.  A diagonal Gaussian correction can
   be less useful than learned denoising.
4. **Reference-measure criticism.**  There is no uniquely "correct" surface
   density for every contact task.  The paper must declare and ablate the
   measure rather than smuggle in a convenient one.
5. **Soft-contact criticism.**  Entropic support is a regularized statistic,
   not actual contact.  Its scales must correlate with pad size/noise and
   improve real decisions.
6. **Scope criticism.**  One grasp benchmark may look like specialized point
   processing.  A small second task involving a sampled geometric decision
   would materially strengthen the broad claim.
7. **Theory criticism.**  Unbiasedness of a Horvitz--Thompson estimator with a
   known Gaussian moment correction is elementary.  Without uniform,
   misspecification, and action-regret results, the theory is decorative.

Before the pilot, the honest ICLR assessment is **promising but not yet an
objectively strong acceptance claim**.  It is more original than a new grasp
backbone or denoising pipeline, but less secure than it sounds if the empirical
acquisition effect is small.  Passing the kill tests, releasing the paired
benchmark, and showing a nontrivial decision theorem would move it into a
credible ICLR submission; failing any of those reduces it to a robotics
robustness module.

### 11. Minimum defensible paper

A defensible paper would contribute all four of the following:

1. a formal definition of acquisition-law-equivalent geometric decision
   learning and an impossibility/bias result for hard noisy contact features;
2. a bandwidth-constrained deconvolved quadrature contact operator with
   uniform and decision-level guarantees under declared assumptions;
3. a paired-acquisition parallel-jaw benchmark in which physics and candidate
   labels are fixed while sampling intensity and noise law change;
4. real evidence that the acquisition-consistent ranking improves repeated
   small-lift selection under point-budget and sensor-noise shifts.

A possible title is:

> **AcqGrasp: Learning Contact Decisions, Not RGB-D Sampling Laws**

The strongest honest abstract claim would be:

> Point-cloud grasp predictors are invariant to point order but need not be
> invariant to how a physical surface was sampled.  This distinction is acute
> for contact decisions: under fixed depth noise, naive point extrema become
> sample-count dependent.  We formulate acquisition-equivalent geometric
> decision learning, introduce a deconvolved quadrature operator for
> finite-scale action-conditioned contact functionals, and evaluate it on
> paired RGB-D acquisitions with fixed physical grasp labels.  The resulting
> model preserves grasp rankings across unseen point budgets and noise laws and
> improves small-lift selection without reconstructing a scene field.

The last sentence is a target claim, not a conclusion, until the pilot and
hardware evidence exist.

## Fourth independent search pass: additional rejected directions

This pass was started after Candidate F had already appeared in the document.
Its purpose is therefore not to polish AcqGrasp, but to exclude adjacent ideas
that could otherwise be mistaken for an independent Candidate G.  Robotics
papers below are used only to audit occupancy and empirical need; the proposed
mathematical objects came from geometry, statistics, mechanics, or general ML.

### 38. Bilateral contact-germ teaching distributions

Proposal: regard each parallel-jaw interaction as a pair of local surface germs
plus a swept free-space corridor.  Select or procedurally synthesize a minimal
training object library whose induced contact germs form an
$\varepsilon$-net of this local interaction space.  This would replace
object-instance diversity by a contact-specific teaching dimension.

Rejection: the underlying empirical thesis is already occupied even if the
covering-number notation is new.  AnyDexGrasp explicitly studies local-geometry
coverage and reports that denser local geometric sampling per object is more
valuable than simply adding objects:
https://graspnet.net/anydexgrasp/assets/files/AnyDexGrasp.pdf.  The ICLR 2026
random-toy paper then makes procedural primitive assemblies themselves a major
grasp-generalization result:
https://proceedings.iclr.cc/paper_files/paper/2026/hash/4b2a917e30e1bb1aff055b4d8c6c081c-Abstract-Conference.html.
An $\varepsilon$-net selector could improve their data efficiency, but the
likely contribution would be a formalized dataset curriculum, not a new
learned object or capability.  This also collides with cycles 1 and 17.

### 39. Finite-pad stop-loss / contact-capacity curves

Proposal: for a candidate grasp, predict a compact monotone curve

$$
 C_g(a)=\int_{P_g}(a-h_g(u))_+\,du,
$$

where $h_g$ is the local height profile under a compliant jaw pad and $a$
is indentation.  Derivatives of $C_g$ encode contact area and pressure-growth
surrogates, so a small vector of spline coefficients could replace a scene SDF
and a binary grasp score.

Rejection after the mechanics audit: this is a learned approximation to an old
finite-contact constitutive object.  Winkler-type pressure profiles and grasp
stability have been analyzed directly
(https://doi.org/10.9746/sicetr.51.83); 6DLS integrates finite non-planar
contact surfaces and pressure into friction wrenches
(https://arxiv.org/abs/1909.06885); and IPC-GraspSim models compliant
parallel-jaw closure with much higher fidelity than point-contact metrics
(https://arxiv.org/abs/2111.01391).  Learned compliance-aware grasping is also
now explicit: https://arxiv.org/abs/2607.17541.  A vision network predicting
$C_g$ would therefore amortize classical contact mechanics.  For an occluded
opposing patch, the curve is also no more identifiable than the hidden shape.
The construction is elegant and compact, but it does not clear the novelty bar.

### 40. Visibility-boundary provenance as a relative surface current

Proposal: represent the observed target as a surface current with boundary and
decompose its boundary measure into a physical edge and a camera-induced
visibility cut.  A boundary-corrected geometric operator would use the former
as contact evidence and prevent the latter from masquerading as a thin graspable
edge.

Rejection: the mathematical language changes, but the learned subproblem is
border ownership / occlusion-boundary estimation.  Interactive occlusion
boundary estimation is already learned explicitly
(https://arxiv.org/abs/2408.15038), depth completion methods use boundary cues
for downstream grasping
(https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.1041702/full),
and TARGO directly addresses target grasping under occlusion with target shape
completion (https://arxiv.org/abs/2407.06168).  A current-valued loss neither
recovers the hidden opposing contact nor creates a new decision target.  In its
defensible scope it is a boundary-aware point-cloud head; outside that scope it
returns to amodal completion and cycles 21 and 32.

### 41. Proposal-covariant grasp scoring

Proposal: a grasp score should transform predictably when the upstream
candidate generator changes.  For generators $\pi_i(g\mid o)$, learn
pairwise log-density-ratio cocycles

$$
 r_{ij}(o,g)+r_{jk}(o,g)=r_{ik}(o,g)
$$

and factor physical utility from proposal density, so a discriminator trained
under one generator remains valid for another.

Rejection: the practical failure is real, but both sides of the proposed paper
are occupied.  GraspGen identifies discriminator distribution shift and uses
on-generator training to obtain its strongest results
(https://arxiv.org/abs/2507.13097), while GraRe reports that detector confidence
is poorly aligned with grasp quality and improves frozen detectors by reranking
(https://arxiv.org/abs/2608.00946).  In general ML, covariate-shift correction
(https://jmlr.csail.mit.edu/beta/papers/v10/bickel09a.html), noise-contrastive
estimation (https://proceedings.mlr.press/v9/gutmann10a.html), and conditional
NCE (https://arxiv.org/abs/1806.03664) already supply the density-ratio
machinery.  Cocycle consistency is useful regularization, but the result would
be a more portable grasp discriminator rather than a new grasping problem.

### 42. Reflected feasible-grasp flow

Proposal: generate grasps with a flow on $SE(3)$ whose trajectories are
reflected at learned collision and aperture boundaries, so probability mass
never leaves the currently feasible action domain.

Rejection: reflected flow matching on constrained domains is already a general
method (https://proceedings.mlr.press/v235/xie24k.html).  Grasp Diffusion
Network already combines manifold diffusion with collision cost guidance
(https://arxiv.org/abs/2412.08398), and constrained grasp diffusion already
makes feasibility guidance the central robotics mechanism
(https://constrained-grasp-diffusion.github.io/).  Replacing soft guidance by a
Skorokhod reflection is mathematically cleaner but is an almost direct method
transfer.  It also treats a predicted partial-view feasibility boundary as if
it were physical, so the strongest claimed guarantee would be false.

### 43. Active learning of contact-mode transition strata

Proposal: expensive high-fidelity grasp labels need not be sampled throughout
six-dimensional action space.  If contact mechanics induces smooth success
regions separated by a codimension-one transition set $\Sigma$, actively
query the simulator near $\Sigma$.  Under reach and margin assumptions, seek
a label complexity governed by boundary dimension rather than ambient
dimension.

Rejection: once stated precisely, the desired result is standard active level-
set estimation or active classification with a smooth decision boundary.
Sequential level-set query complexity is already studied
(https://proceedings.mlr.press/v130/bachoc21a.html), including Bernoulli
observations (https://proceedings.mlr.press/v151/letham22a.html) and adaptive
rates for smooth decision boundaries (https://arxiv.org/abs/1711.09294).
Contact simulation is indeed costly -- IPC-GraspSim reports a large cost gap
relative to analytic metrics, while GRIP accelerates IPC data generation by up
to 48x (https://arxiv.org/abs/2503.05020) -- but this establishes usefulness,
not conceptual novelty.  Contact physics merely supplies assumptions for an
existing active-learning problem, and real contact transitions need not form
one smooth boundary because mode intersections and simulator chatter violate
those assumptions.

### 44. Proposal-invariant valid-set generative learning

Proposal: positive grasps in large datasets are outputs of heuristic or
optimization samplers.  Rather than imitate their density, learn the
Haar-volume measure of the physical success set

$$
 \nu_o(dg)\propto \mathbf 1\{Q(o,g)\geq\tau\}\,\lambda_{SE(3)}(dg),
$$

or a noise-smoothed version, using inverse proposal weights in flow matching.
The intended broad problem was generative learning of feasible action sets that
is invariant to the annotation sampler.

Rejection: GraspDataGen makes the sampling process concrete -- it requests
fixed numbers of collision-free and colliding guesses before validation
(https://github.com/NVlabs/GraspDataGen/blob/main/docs/workflows/graspgen.md) --
so the bias is plausible.  However, importance-weighted generative learning
under sample-selection bias is established
(https://arxiv.org/abs/1806.02512), and even unbiased diffusion from biased
datasets is already an ICLR contribution
(https://proceedings.iclr.cc/paper_files/paper/2024/hash/89bd6217280d1417370c89ee493ba3c7-Abstract-Conference.html).
With known proposal propensities the grasp method is a direct specialization;
with unknown propensities the target is not identifiable from positive samples.
Moreover, there is no canonical relative scale between translation and rotation
in a Haar-volume notion of a “uniform grasp,” so the desired measure smuggles in
an arbitrary robustness metric.  This direction is distinct from Candidate F's
input-acquisition law, but not strong enough for Candidate G.

### 45. Bilateral first-contact survival kernels

Proposal: avoid full hidden-shape completion by predicting the joint law of the
two first-contact stopping times encountered as the jaws close.  Each grasp
would query two one-dimensional cumulative hazards plus a small copula, from
which missed contact, terminal aperture, and contact-order uncertainty could be
computed efficiently.

Rejection: the output is compact, but it repackages the action-conditioned
entry/exit geometry rejected in cycle 1 and the switching first-contact map
rejected in cycle 12.  ShellGrasp-Net already predicts shell entry/exit depths,
and general 3-D learning now includes a line-segment field that predicts whether
and where a query segment intersects a surface:
https://proceedings.iclr.cc/paper_files/paper/2025/file/ce02914eb52e15bbbb4b70c2a994dfeb-Paper-Conference.pdf.
A copula does not repair single-view nonidentifiability and brings the proposal
close to cycles 17 and 37.  “Survival analysis” would therefore be new notation
around an occupied contact representation.

### 46. Bounded-influence grasp operators on contaminated point measures

Proposal: define the grasp predictor as a statistical functional of the RGB-D
empirical measure and require a bounded influence function under point
addition, deletion, and ray-direction displacement.  A clipped local operator
or median-of-means aggregation could then give a certified action-ranking
stability radius under Huber contamination.

Rejection: this is a legitimate robustness target, but not independent of
Candidate F's acquisition-law problem, and its general machinery is classical
robust statistics.  PointGuard already certifies point-cloud predictions under
modified, added, and deleted points (https://arxiv.org/abs/2103.03046), while
AdvGrasp studies physically meaningful adversarial attacks on robotic grasping
(https://www.ijcai.org/proceedings/2025/62).  A bounded-influence backbone would
be a useful defense, but the guarantee would concern stability of the learned
ranking rather than correctness of physical contact.  Strong outliers can also
be actual thin object features, so clipping conflicts with the contact signal
the model must preserve.

At this point no direction in cycles 38--46 is promoted.  The next search must
not merely replace a scalar head, add a robustness certificate, debias a
proposal distribution, or rename local hidden-contact prediction.  Candidate G
still requires a distinct estimand whose necessity follows from the physics and
whose estimator is not an off-the-shelf general-ML correction.

### 47. Mixed-dimensional contact-stratum diffusion

Proposal: model successful grasps as a stratified measure

$$
 \mu_o=\sum_k \alpha_k(o)\rho_k(o,\cdot)
       \mathcal H^{d_k}\!\restriction M_k(o)
$$

on a union of contact-mode manifolds of different intrinsic dimensions.  A
dimension-aware diffusion would renormalize the small-noise score separately on
each stratum instead of treating all labels as an ambient $SE(3)$ density.

Rejection: a 2026 general-ML paper now addresses almost exactly deep generative
learning on stratified spaces, including varying stratum dimensions, score
geometry, dimension estimation, and convergence rates:
https://arxiv.org/abs/2604.10650.  More importantly, the physical premise is
unclear.  Canonicalized exact-contact labels can be singular, but a genuinely
robust executable grasp normally has an open capture/tolerance neighborhood in
command space.  Treating simulator label manifolds as physics would preserve a
dataset artifact; expanding them to capture basins returns to cycles 3, 4, and
7.  NGDF, OrbitGrasp, and modern manifold grasp flows already make the robotics
side crowded.  This is therefore neither an independent estimand nor a safe
method transfer.

### 48. Topological obstruction to a deterministic equivariant grasp selector

Proposal: for an observation with stabilizer $H\subset SE(3)$, any
equivariant deterministic selector $f$ must satisfy $h f(o)=f(o)$ for every
$h\in H$.  Symmetric objects often have no single grasp fixed by all of $H$,
so a continuous equivariant top-1 selector cannot exist.  Predict instead a
stabilizer-compatible bundle-valued section or an orbit measure.

Rejection: the obstruction is correct but not new enough.  General ML already
proves impossibility results for continuous canonicalization and constructs
continuous weighted equivariant frames
(https://proceedings.mlr.press/v235/dym24a.html); representation learning with
nontrivial stabilizers is also explicit (https://arxiv.org/abs/2301.05231).
On the robotics side, EquiGraspFlow already predicts an equivariant conditional
distribution rather than a deterministic canonical pose
(https://equigraspflow.github.io/), and OrbitGrasp represents continuous grasp
quality on orientation orbits (https://orbitgrasp.github.io/).  A
grasp-specific stabilizer theorem would explain why these outputs are sensible,
but a weighted-frame or orbit-measure model would be an application of existing
theory.  Near-symmetry instability is likewise handled by probabilistic frames;
it does not create a new physical target.

### 49. Shape-derivative supervision for grasp transfer

Proposal: learn not only grasp utility $Q(S,g)$, but its directional shape
derivative with respect to a surface deformation field $V$,

$$
 D_SQ(S,g)[V].
$$

Simulator-provided derivatives could give Sobolev supervision, transport grasp
fields between nearby object shapes, and quantify which local geometry controls
the decision.

Rejection: derivative supervision for neural networks is the established
Sobolev-training construction (https://arxiv.org/abs/1706.04859), and
shape-derivative-informed neural operators on varying geometries now exist
(https://arxiv.org/abs/2603.03211).  Contact mechanics makes the proposal less,
not more, secure: frictional unilateral contact is generally not shape
differentiable without additional conditions and must use directional
derivatives (https://doi.org/10.1137/19M125813X).  Differentiable grasp metrics
and optimizers already exploit grasp sensitivities, including a method reporting
a 22% physical success improvement
(https://www.roboticsproceedings.org/rss16/p066.pdf).  The proposed model would
thus combine an existing training signal with an unreliable simulator
derivative.  If smoothed until differentiable, it becomes another local
robustness/tolerance head; if kept nonsmooth, it is too unstable for the claimed
sample-efficiency advantage.

### 50. Learned oriented-matroid contact types

Proposal: discard fragile metric contact coordinates and predict the
combinatorial sign pattern of the relative orientations among the jaw axis,
contact normals, friction-cone generators, and gravity wrench.  Such sign
patterns can be organized as an oriented matroid; continuous margins would then
refine only the predicted combinatorial cell.

Rejection: the discrete type is neither sufficient for success nor a new
supervision target.  It records which side of a collection of hyperplanes a
configuration occupies, but not jaw width, collision clearance, lever arms,
finite pad area, or the magnitude of the friction margin.  Adding all required
metric residuals reconstructs an ordinary contact-quality vector.  Classical
grasp analysis already polyhedralizes friction cones and reasons over contact
mode cells, while modern methods explicitly learn paired contact embeddings and
match classifiers (for example PhyGrasp: https://arxiv.org/abs/2402.16836).
Oriented-matroid language could organize a solver, but it would not create a
physically sufficient compact estimand and is too close to cycles 12, 17, and
26.

### 51. Learned medial-axis / local-feature-size grasp fields

Proposal: predict a sparse medial-ball field from the partial RGB-D observation.
Each ball supplies a center, local thickness, and two or more boundary witnesses;
parallel-jaw candidates could then be generated directly from its bilateral
geometry without reconstructing a dense SDF.

Rejection: this is an old grasp representation, not an unoccupied gap.  A 2011
planner already used a grid of medial spheres to identify local symmetry and
gripper-compatible thickness
(https://h2t.iar.kit.edu/pdf/Przybylski2011.pdf), followed by a complete
humanoid grasping system based on the medial-axis transform
(https://h2t.iar.kit.edu/pdf/Przybylski2012.pdf).  Learning the transform would
speed up and partially complete a classical skeleton planner.  Under occlusion,
the medial axis is also globally unstable and depends on the missing backside;
making it probabilistic returns to hidden-shape completion, while pruning it for
noise discards thin features important to grasping.  The proposal therefore
fails both novelty and identifiability tests.

### 52. Factorized physical-constraint supervision

Proposal: replace an opaque success label by a small collection of locally
checkable predicates -- bilateral contact, friction-cone inclusion, terminal
collision exclusion, and gravity-wrench support -- and train a compositional
constraint network whose conjunction defines a grasp.

Rejection: the factorization is attractive for data efficiency, but it is not a
new learning problem.  Physics-informed hard constraints and differentiable
constraint layers are a mature general-ML construction; scalable hard-constraint
learning was already an ICLR 2024 contribution
(https://proceedings.iclr.cc/paper_files/paper/2024/file/9aeda582add763c41c7b39691ce19ab0-Paper-Conference.pdf).
In grasping, analytic filters, differentiable contact optimization, and GraspQP
(https://graspqp.github.io/) already combine essentially these predicates.  The
conjunction is not sufficient under compliant contact and partial hidden
geometry, while expanding it to cover every failure produces exactly the long
causal failure-mode vector excluded by the project brief.  In the compact case
it is a proof-carrying grasp variant of cycles 9 and 23; in the complete case it
becomes a simulator.

### 53. Multi-fidelity grasp-oracle label fusion

Proposal: treat antipodal metrics, rigid-body simulation, compliant-contact
simulation, and a small number of physical lifts as correlated annotators of a
latent grasp-success field.  A hierarchical label model would estimate each
oracle's object- and contact-dependent confusion process and provide soft labels
to a visual grasp model.

Rejection: multi-fidelity binary classification already has direct surrogate
models, including autoregressive Gaussian-process classifiers for expensive
simulations
(https://www.sciencedirect.com/science/article/pii/S0045782519304785).
Without enough physical labels, correlated oracle errors are not identifiable;
with enough labels, an ordinary supervised residual or domain-adaptation model
is simpler.  More importantly, the proposed hierarchy does not define a new
physical target -- it estimates the same binary success field from several
imperfect sources.  Candidate E already investigates the stronger question of
what should be learned when numerical contact oracles change under refinement.
This weaker label-fusion version cannot be Candidate G.

### 54. Equilibrium-tomographic center-of-mass sets

Proposal: use the fact that the target is already at rest on the shelf as a
weak physical observation of its hidden mass distribution.  If the same rigid
object is observed in resting orientations $R_j$ with planar support polygons
$S_j$, its object-frame center of mass $c$ must lie in the convex identified
set

$$
 C_J= B\cap\bigcap_j
 \{c:\Pi_{g^\perp}(R_jc+t_j)\in S_j\},
$$

where $B$ is a coarse object bound.  Multiple nonparallel gravity directions
can shrink the intersection of support prisms in all three coordinates.  One
could collect these inequality labels by repeated passive placements and train
a compact single-view RGB-D predictor of $C_J$, without a force/torque sensor,
then use the current shelf support as one additional test-time constraint.

Rejection after the identifiability audit: a single stable rest supplies only
membership of the gravity projection in the support polygon; it neither
localizes the point inside that polygon nor identifies mass magnitude.  For an
opaque object, visually identical shells with different internal ballast remain
indistinguishable even after imposing the same stable pose.  Repeated poses make
an interesting offline metrology protocol, but using their intersection as a
weak label does not make the latent CoM predictable from one novel object's
RGB-D.  With a learned object prior this becomes ordinary censored/set-valued
regression; without such a prior the honest output is usually the broad analytic
intersection itself.  Estimating CoM for grasping is also occupied: tactile-
visual slip/regrasp data already produced a learned CoM-aware planner and a
reported 31-point success gain, while a 2025 vision method directly targets CoG
localization for uneven-mass unknown objects.  The equilibrium inequality is
classical, so the remaining novelty would be a data-collection protocol rather
than a new grasp-learning object.

Sources:

- https://motion.cs.illinois.edu/RoboticSystems/AdvancedTopicsInPlanning.html
- https://arxiv.org/abs/2006.00906
- https://arxiv.org/abs/2507.19242
- https://www.jstage.jst.go.jp/article/jrsj1983/17/5/17_5_728/_article/-char/ja/

### 55. Learned support exchange by wrench-set containment

Proposal: treat picking as replacement of the environmental support by the two
jaws.  The current shelf contact proves that the unknown normalized gravity
wrench belongs to an environment-compatible load set $L(o)$; a queried grasp
produces a bounded wrench-capacity body $K(o,g)$.  Learn their support
functions in a shared low-dimensional gravity-wrench plane and score the robust
exchange by

$$
 m(o,g)=\inf_{u\in\mathbb S^1}
 \big[h_{K(o,g)}(u)-h_{L(o)}(u)\big].
$$

Convex containment is equivalent to $m\ge0$, uniform support-function error
directly bounds margin error, and inference over a fixed directional grid is
cheap.  A more ambitious benchmark would hold out combinations of current
supports and new jaw contacts to test compositional transfer through the shared
wrench coordinates.

Rejection: Task Wrench Space containment in Grasp Wrench Space is already the
classical definition of a task-suitable grasp; the gravity-generated Mass Wrench
Space and support-function/convex-hull computation are established mechanics,
not a new formalization.  The only apparently new input is inferring $L$ from
the observed rest.  But, by cycle 54, rest certifies only that the actual load is
*one unknown member* of a usually broad support-polygon set.  Treating the whole
set as the task wrench space is often so conservative that it returns the usual
central grasp; learning a posterior over its members again reduces to uncertain
CoM estimation.  A direct outcome model trained on the same distribution remains
Bayes-optimal for average success, so factorizing it into two convex heads would
need an artificial held-out-pair protocol to show an advantage.  This is an
elegant physical composition and could be useful engineering, but it composes
inverse statics with a decades-old grasp metric and does not meet the requested
ICLR novelty bar.

Sources:

- https://www.robotic.dlr.de/fileadmin/robotic/borst/Borst-ICRA2004-TaskWrenchSpace.pdf
- https://graspnetapi.readthedocs.io/en/latest/graspnetAPI.utils.dexnet.grasping.html
- https://arxiv.org/abs/2006.02996
- https://arxiv.org/abs/1806.01384

### 56. Grasp-aperture viability windows

Proposal: for a fixed contact frame and approach direction, learn the interval
$[a_{\min},a_{\max}]$ of admissible parallel-jaw openings instead of regressing
one grasp width.  Capturing the target imposes a lower bound on aperture, while
the shelf or frontal obstacle imposes an upper bound; the interval width is a
direct clearance/capture margin.  An isotonic decoder over aperture could learn
the two transition points with censored supervision, using only two scalars per
pose.  Extending the interval over insertion depth would give a small viability
strip for a prescribed straight approach.

Rejection: width is already part of the standard learned parallel-jaw grasp
parameterization, and classical grasp validators explicitly trace the open jaws
along their lines of action and prescribed approach to test contact and
collision.  Contact-GraspNet reduces its pose by rooting pose and width in the
observed cloud; GraspNet/Dex-Net code exposes open width, close width, and
approach collision; practical shelf planners often use a conservative maximum
opening during collision checks.  Predicting two transition points could be a
useful shelf-specific head, but it is still a richer parameterization of the
same geometric validation problem.  Once depth-dependent, it also becomes a
partial trajectory-viability estimator, moving toward the explicitly excluded
full approach-to-lift feasibility setting.  It therefore lacks both broad ML
novelty and scope compatibility.

Sources:

- https://arxiv.org/abs/2103.14127
- https://github.com/graspnet/graspnetAPI/blob/master/graspnetAPI/utils/dexnet/grasping/grasp.py
- https://journals.sagepub.com/doi/10.1177/17298814211040632

### 57. Positive--unlabeled valid-grasp-set learning

Proposal: regard a finite list of annotated successful grasps as samples from,
not an exhaustive description of, the valid-grasp set.  Train a support or
density estimator without declaring every unmatched action negative, using a
PU/one-class risk and a candidate-measure correction.  This would address the
false negatives created by sparse grasp annotations and avoid penalizing novel
valid modes.

Rejection: the grasp-specific sparse-label problem and its false negatives were
already stated directly in *Learn to Grasp with Less Supervision*; its maximum-
likelihood grasp-sampling loss reports equivalent 90.7% physical success with
two labels per image rather than sixteen.  Positive-only generative grasp models
also estimate a conditional grasp distribution without exhaustive negative
annotation, while GraspNet's evaluation machinery analytically labels arbitrary
queried poses instead of treating unlisted poses as false.  For modern simulated
6-DoF data the dominant discriminator problem is sampled/on-generator covariate
shift, already covered by GraspGen and cycle 41, not a missing PU formalization.
This route is therefore occupied supervision engineering, not Candidate G.

Sources:

- https://arxiv.org/abs/2110.01379
- https://arxiv.org/abs/1912.13470
- https://github.com/NVlabs/GraspDataGen/blob/main/docs/workflows/graspgen.md

### 58. Censored time-to-slip grasp fields

Proposal: replace binary lift success at an arbitrary horizon by a survival
function $S(t\mid o,g)$ or hazard of first slip under a standardized small
lift.  Right-censored trials that remain stable until the experiment ends would
still contribute information, and different deployment horizons could query the
same compact monotone curve.

Rejection: this changes the label resolution, not the grasping information
available before contact.  Slip onset is governed by weight, friction, surface
texture, modulus, pressure distribution, and realized contacts; recent tactile
work explicitly argues that these variables require physical interaction.
Existing slip prediction and stability work consequently relies on tactile or
force/pressure time series, including pre-lift tactile snapshots and predictive
forward models.  From only wrist RGB-D, $S(t\mid o,g)$ is no more identifiable
than the binary outcome and would learn dataset material priors.  Survival loss
and censoring are mature statistical tools, so even with tactile hardware the
new contribution would mainly be a richer stability label.  It also drifts from
pre-contact grasp selection toward post-contact failure prediction, contrary to
the requested scope.

Sources:

- https://www.mdpi.com/2218-6581/8/4/85
- https://robot-learning.cs.utah.edu/_media/project/veiga-toh2018-slip-prediction.pdf
- https://www.nature.com/articles/s42256-025-01062-2
- https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202501051

### 59. Learned minimum escape-energy barriers

Proposal: replace a local force-closure margin by the minimum mechanical work
needed for the object to leave the grasp's metastable basin.  The saddle height
of an effective contact-energy landscape could distinguish a locally balanced
but easily rolling/slipping grasp from one with a finite nonlinear barrier.  A
compact model might predict the barrier and lowest-escape mode for each queried
parallel-jaw pose from RGB-D.

Rejection: minimum escape energy and energy-bounded caging are already explicit
robotic grasp objects, and potential-energy indices for elastic fingertips have
long defined grasp stability by the minimum energy that causes slip.  Modern
work also extends caging to time-varying geometric and energy spaces.  Learning
the same scalar or saddle direction would amortize a classical analysis, not
define a new problem.  Moreover, the landscape depends on complete object
geometry, CoM, friction, pad elasticity, preload, and dissipation; the stated
single RGB-D input does not identify these quantities.  A faithful label would
require the expensive contact simulation and hidden state already rejected in
cycles 26, 27, 39, and Candidate E, while a simplified label would not support
the claimed nonlinear advantage.

Sources:

- https://escholarship.org/content/qt21f0t7pd/qt21f0t7pd_noSplash_bc6738a447c0692953c7fd95461bfed5.pdf
- https://arxiv.org/abs/2410.16481
- https://arxiv.org/abs/1905.00134

### 60. Bi-equivariant object--gripper grasp operators

Proposal: condition grasp generation on both the observed object measure and an
explicit geometric/kinematic representation of the gripper.  The output is a
relative-pose distribution and should transform under independent object and
gripper frame changes by a left--right group action, rather than only being
equivariant to a common scene transform.  A bi-equivariant neural operator
could in principle generalize to unseen parallel-jaw morphologies and amortize
training over a family of hands.

Rejection: cross-embodiment grasping is now a strongly occupied capability.
UniGrasp already conditions on object geometry and gripper attributes and
reports real zero-shot success on unseen two- and multi-finger grippers.
GraspGen-X (CVPR 2026) explicitly conditions a 6-DoF diffusion generator on a
gripper swept-volume representation, trains on procedural grippers and hundreds
of millions of grasps, and evaluates novel morphologies and physical grasp
processes.  A cleaner left--right equivariance law would be an architectural
improvement to that established problem, not a new scientific target.  The
laboratory also has one fixed parallel-jaw gripper, so embodiment diversity does
not address its dominant partial-observation and shelf-occlusion errors; a SOTA
claim would require a much broader benchmark than the intended setting.

Sources:

- https://arxiv.org/abs/1910.10900
- https://graspgenx.github.io/
- https://openaccess.thecvf.com/content/CVPR2026/papers/Han_GraspGen-X_Cross-Embodiment_6-DOF_Diffusion-based_Grasping_CVPR_2026_paper.pdf

### 61. External-field-equivariant grasp learning

Proposal: distinguish geometric covariance from physical symmetry.  A grasp
field on a shelf is not invariant when only the object is rotated while gravity
and the shelf remain fixed.  The correct law jointly transforms the point cloud,
grasp, gravity vector, shelf normal, and possibly approach field,

$$
 q(RP,Rg;R\gamma,Rn_s)=q(P,g;\gamma,n_s),
$$

while a fixed external field leaves only its stabilizer subgroup.  A
field-conditioned equivariant generator could prevent an $SE(3)$-equivariant
geometry model from assigning equal physical quality to co-rotated grasps that
have different gravity rejection.

Rejection as the main idea: the distinction is correct but its implementation
is standard.  An $E(3)$/$SE(3)$-equivariant network can accept gravity and
support normals as vector features; conditioning on a preferred approach
already leads to subgroup-equivariant grasp samplers such as CAPGrasp's
$\mathbb R^3\times SO(2)$ formulation.  Gravity-aware grasp generation also
already trains on a gravity-rejection score and reports its largest benefit on
heavy objects.  Thus the proposed covariance law is an important correctness
condition and benchmark ablation for any future model, but not a new learning
object.  Claiming that existing equivariant generators are simply “wrong” would
also overreach: many learn geometric grasp distributions, for which co-rotation
is intended, and defer gravity-aware selection to another score.

Sources:

- https://equigraspflow.github.io/
- https://wengzehang.github.io/CAPGrasp/
- https://arxiv.org/abs/2312.11804

### 62. Pretraining grasp support from passive stable placements

Proposal: view placement and grasping as two instances of supporting a rigid
body against an external wrench.  Use abundant stable/unstable placement data
to learn a one-sided supportability operator for a surface patch and load
direction, then compose two oppositely directed support predictions to score a
parallel-jaw contact pair.  This could appear to provide grasp pretraining from
passive drops rather than expensive grasp simulation or robot trials.

Rejection: the apparent duality is too weak for the intended transfer.
Placement on a plane teaches unilateral support under one gravity load and is
strongly coupled to the object's CoM and broad base.  A jaw grasp depends on
bilateral preload, the compatibility of two finite patches, frictional moments,
closure, and bounded squeeze.  Treating it as two independent placements assumes
the separability already rejected in cycle 17; retaining the joint mechanics
requires grasp-like paired data again.  Grasping/fixturing geometric duality is
classical, and learned placement has already been explicitly inserted into a
grasp/regrasp system in *Learning to Regrasp by Learning to Place*.  The likely
result is auxiliary-task pretraining with a severe contact-distribution shift,
not a new efficiently sufficient supervision source.

Sources:

- https://geometry.stanford.edu/paper/cms-lrlp-21/cms-lrlp-21.pdf
- https://ics.uci.edu/~eppstein/gina/grasp.html
- https://motion.cs.illinois.edu/RoboticSystems/AdvancedTopicsInPlanning.html

### 63. Persistent gripper-state adaptation from past lifts

Proposal: separate episodic scene uncertainty from a low-dimensional latent
hardware state $\eta$ shared across many grasps, such as closing-force scale,
pad stiffness, wear, or a friction multiplier.  A hierarchical likelihood

$$
 y_i\sim\mathrm{Bernoulli}
 \big(q_\theta(o_i,g_i;\eta)\big)
$$

would update a posterior over $\eta$ from historical binary lifts and adapt all
future single-view predictions without interacting with the current object or
using RL.

Rejection: binary success does not isolate a persistent pad factor from
object-specific material, roughness, mass, contact geometry, execution error,
and changing candidate selection.  The likelihood can fit these confounders by
moving $\eta$, so identifiability requires calibration objects or richer force/
tactile measurements.  With those measurements the problem is ordinary online
system identification or Bayesian simulator calibration; recent systems already
estimate friction online from tactile signals and experimental platforms fit
friction and elastic parameters by Bayesian calibration.  With only grasp
outcomes, the proposal repeats the standard latent calibration structure of
cycle 33 with a less identifiable parameter.  It is a sensible maintenance
module, not an ICLR-level grasp formalization.

Sources:

- https://arxiv.org/abs/2602.02026
- https://www.nature.com/articles/s44182-026-00092-1
- https://arxiv.org/abs/2111.01391

### 64. Jointly learned pad design and grasp selection

Proposal: augment the grasp action $g$ with a low-dimensional replaceable-pad
design $d$, for example a thickness profile, groove basis, or spatial
stiffness field, and optimize

$$
 (d^\star,\pi^\star)
 \in\arg\max_{d,\pi}\;
 \mathbb E_{(o,P)}
 \left[u\!\left(o,\pi(P;d),d\right)\right].
$$

The intended benefit would be mechanical regularization: a pad family could
increase the capture range of a visual grasp predictor under partial and noisy
point clouds, rather than demanding an ever more accurate perception model.

Rejection: both halves and their joint optimization are already explicit
research directions.  *Parallel-Jaw Gripper and Grasp Co-Optimization for Sets
of Planar Objects* jointly optimizes finger shape, contacts, and object-specific
gripper poses.  *Fit2Form* learns a generative model of interchangeable
parallel-jaw finger geometries using a differentiable learned fitness proxy.
Most decisively, *Co-Design of Soft Gripper with Neural Physics* jointly
optimizes a gripper's spatial stiffness distribution and grasp pose through a
fast differentiable surrogate, then fabricates the designs and measures
physical success.

Adding a partial RGB-D encoder to this objective would therefore be an input
modification, not a new estimand.  Moreover, these methods obtain their design
signal from complete object geometry and contact simulation; under the present
single-view setup the hidden shape that determines a custom pad is precisely
unobserved.  Robustifying the design over all sensor-consistent hidden shapes
would return to the sensor-null ambiguity and capture-basin ideas already
excluded in Candidate D and cycle 4.  The fixed laboratory end effector and
fabrication-heavy evaluation also make this a hardware detour rather than a
broad, efficiently learnable grasp formulation.

Sources:

- https://arxiv.org/abs/2310.18425
- https://proceedings.mlr.press/v155/ha21b.html
- https://proceedings.mlr.press/v305/yi25a.html

### 65. Top-one-regret-calibrated continuous grasp learning

Proposal: replace pointwise binary classification with a structured surrogate
whose population minimizer is calibrated for the actual decision

$$
 g^\star(P)=\arg\max_{g\in\mathcal G(P)}p(y=1\mid P,g),
\qquad
 \mathcal R_{\mathrm{top1}}
 =\mathbb E\!\left[
 p(y=1\mid P,g^\star)-p(y=1\mid P,\hat g)
 \right].
$$

One could use listwise normalization over sampled $SE(3)$ candidates,
hard-negative mining near the current maximum, and quotient-aware distances
for the parallel-jaw symmetry, then prove a calibration bound from surrogate
excess risk to top-one physical regret.

Rejection: with adequate action support, any strictly proper pointwise loss
already recovers the conditional success probability, and its argmax is the
Bayes-optimal grasp.  A listwise loss changes finite-sample emphasis but does
not define a new physical quantity.  If candidates come from a learned
proposal, the important inconsistency is missing action support and
proposal-dependent negatives, which is exactly cycle 41 rather than a ranking
theorem.  On the robotics side, GraRe already isolates candidate ordering as a
task, trains a separate quality re-ranker for frozen detectors, and reports
large gains; older grasp work also explicitly upweights the highest-quality
candidates.  Thus a calibrated structured loss could be a useful technical
component, but the main claim would be general top-$k$ calibration applied to
grasping, with no new estimand or evidence that it can beat stronger geometry
and generator models.

Sources:

- https://arxiv.org/abs/2608.00946
- https://proceedings.mlr.press/v229/yang23a.html
- https://openreview.net/forum?id=HyIqztkDM

### 66. Layered target--occluder order as a minimal scene representation

Proposal: avoid a full scene SDF by retaining only a labelled depth-order
relation along camera rays.  The observation would be a small directed
occlusion graph plus visible target and obstacle depth layers; a grasp operator
would predict which approach and finger corridors preserve the order needed to
reach the target.  In spirit, the representation tries to separate “what hides
what” from metric surface completion.

Rejection: a per-ray order is sufficient to explain the current image, but not
to certify a three-dimensional gripper corridor.  Two scenes can have exactly
the same target--occluder order and visible depths while differing arbitrarily
in lateral obstacle extent or the hidden opposing target surface, producing
different valid grasps.  Adding the missing metric extent turns the
representation into ordinary layered occupancy/amodal geometry; leaving it out
makes the action target non-identifiable.  Occlusion-order graphs and explicit
visible/amodal/occluder masks are already established in amodal perception, and
TARGO already fuses completed target and scene features for single-view direct
target grasping under occlusion.  Therefore the order complex would be a
compressed obstacle-aware feature, not an efficiently sufficient new grasp
object.  It also overlaps cycle 28's obstacle composition and cycle 40's
visibility-boundary provenance.

Sources:

- https://arxiv.org/abs/2109.11103
- https://openaccess.thecvf.com/content/CVPR2022/html/Mohan_Amodal_Panoptic_Segmentation_CVPR_2022_paper.html
- https://targo-benchmark.github.io/

### 67. Learning on the space of oriented jaw slabs

Proposal: represent a parallel-jaw action first by two oriented parallel contact
planes, equivalently an oriented slab with center $c$, normal $b$, and width
$w$, and only secondarily attach an approach direction $a\perp b$.  A
network on affine-plane incidence could predict a measure over slabs, making jaw
exchange a built-in $\mathbb Z_2$ quotient and antipodal contact a native
event rather than a property decoded from $SE(3)$.

Rejection: once the approach vector and finger depth are supplied, the variables
$(c,b,a,w)$ are smoothly equivalent to the standard parallel-jaw pose and
width, modulo the same jaw symmetry.  If $a$ is omitted, the representation
cannot distinguish a collision-free approach from one that intersects the
shelf or frontal obstacle.  The supposed simplification therefore either loses
an action degree of freedom or merely changes coordinates.  Contact-GraspNet
already anchors a 6-DoF grasp at a candidate contact and predicts its orthogonal
baseline, approach, and width; contact-pair and antipodal representations are
classical.  An affine-incidence kernel may be an elegant architecture, but it
does not alter supervision, identifiability, or Bayes-optimal selection, and
thus cannot carry the paper's scientific claim.

Sources:

- https://arxiv.org/abs/2103.14127
- https://proceedings.mlr.press/v155/jeng21a.html
- https://motion.cs.illinois.edu/RoboticSystems/AdvancedTopicsInPlanning.html

### 68. Appearance-conditioned local contact-law prediction

Proposal: use RGB not merely as another geometric feature but to predict a
compact distribution over friction/compliance parameters at the two prospective
jaw patches.  The grasp score would marginalize a finite-contact model,

$$
 q(P,I,g)
 =\int q_{\mathrm{mech}}(P,g;\mu_1,\mu_2,k_1,k_2)\,
 p_\theta(d\mu_1\,d\mu_2\,dk_1\,dk_2\mid I,P,g),
$$

so visually inferred material priors could change pose selection even when two
candidates have similar depth geometry.

Rejection: a static RGB crop does not identify the required contact law.
Coatings with the same appearance can have different friction, internal
structure controls compliance and mass, and lighting can change appearance
without changing mechanics.  At most the network learns category correlations,
whose benefit disappears on the mechanically matched but visually ambiguous
objects needed for a convincing test.  Existing visual--tactile benchmarks
already show that touch complements vision for grasp-stability prediction, and
recent physical-property-aware grasping explicitly uses tactile motion to infer
mass, stiffness, and friction because they are hidden from static vision.
DeliGrasp instead imports semantic priors from an LLM, which is outside scope.
With measured material parameters, the proposal reduces to classical
friction/compliance-aware scoring; without them, it is a material classifier
feeding that score.  It therefore fails identifiability and novelty, and also
reopens the hidden mechanical variables rejected in cycles 30 and 31.

Sources:

- https://www.objectfolder.org/benchmark-manipulation
- https://vitacphys.github.io/ViTacPhys/
- https://deligrasp.github.io/

### 69. Instance-specific grasp-field stabilizer discovery

Proposal: learn the subgroup that leaves an individual object's grasp utility
field invariant,

$$
 H_Q(z)=\{h:\;Q(z,hg)=Q(z,g)\ \text{for all }g\},
$$

rather than impose only a global $SE(3)$ covariance law or infer geometric
symmetry.  Because mechanically irrelevant texture and shape details need not
break $H_Q$, discovering this task-induced stabilizer could quotient duplicate
grasp modes and share supervision over its orbits.

Rejection: estimating $H_Q$ requires comparing the same grasp field over many
transformed actions, so it is not cheaper than learning that field directly.
For a partial view, an apparent rotational or reflectional symmetry may be
broken by hidden geometry; gravity, the shelf, and the frontal obstacle further
reduce the physical stabilizer even when the isolated shape is symmetric.
Enforcing the inferred group therefore creates confident invalid grasps in
exactly the difficult cases.  More importantly, *SymmetryGrasp* already detects
3-D symmetry from a single RGB-D view and uses it to improve antipodal grasp
detection, while symmetry-aware object pose and equivariant grasp generation
are mature neighboring lines.  Replacing geometric symmetry by the stabilizer
of the label function is conceptually cleaner but operationally becomes
adaptive symmetry regularization, not a new decision target or physical
capability.

Sources:

- https://doi.org/10.1109/LRA.2022.3214785
- https://arxiv.org/abs/2405.11257
- https://arxiv.org/abs/2608.03295

### 70. Few-shot instance adaptation from past grasp outcomes

Proposal: when the same physical object recurs, condition the visual grasp field
on a small unordered context

$$
 D_k=\{(g_i,y_i)\}_{i=1}^k,\qquad
 q_\theta(y\mid P,g,D_k),
$$

using a conditional neural process or equivariant set encoder.  The latent
context could absorb object-specific mass distribution, friction, or hidden
geometry without explicitly estimating a long physical state, while each update
would remain a single feed-forward pass rather than RL.

Rejection: this exact learning object is already occupied by *Amortized
Inference for Efficient Grasp Model Adaptation*.  Its Grasping Neural Process
encodes a set of labelled grasps, infers a posterior over unobserved
object-level properties, and predicts feasibility of new actions; it is
explicitly evaluated for adapting novel objects from a few interactions and
for force-aware grasp selection.  More fundamentally, the present laboratory
brief does not promise repeated attempts on the same object before the required
pickup.  Adding an adaptation phase changes a one-shot shelf grasp into an
exploratory-interaction task and creates a strong comparison against that ICRA
2024 method.  A new equivariant encoder or RGB-D front end would be an extension
of the same neural-process formulation, not a novel direction.

Sources:

- https://groups.csail.mit.edu/rrg/papers/noseworthy_shaw_icra24.pdf
- https://www.csail.mit.edu/news/helping-robots-grasp-unpredictable

### 71. Bounded-order compositional grasp fields for rigid assemblies

Proposal: exploit the fact that a parallel-jaw closure selects one left and one
right supporting component of an object assembled as
$K=\bigcup_{i=1}^m K_i$.  For a fixed grasp, per-component first-contact
times compose by a min-plus reduction, collision evidence by a max veto, global
mass moments add, and the terminal contact mode appears to require only a
pairwise interaction between the two selected components.  This suggests a
bounded many-body expansion

$$
 Q(K,g)\approx
 \rho_g\!\left(
 \bigoplus_i u_g(K_i),\;
 \bigoplus_{i,j}v_g(K_i,K_j)
 \right)
$$

whose interaction order is tied to the number of jaws rather than the number
of object parts.  A soft part encoder and a max/min-plus pairwise network could
then be trained on procedural assemblies and evaluated on real objects with
more parts than any training example.

Rejection: the exactness premise fails in the relevant mechanics.  During
closure the object may translate or rotate on the shelf, so the two terminal
contacts are not independent extrema of the fixed union; whole-object CoM,
connection rigidity, and collisions also couple all parts.  Restoring those
terms removes the bounded-order theorem.  Without provided part labels, the
implementation reduces to a pairwise kernel over surface/contact tokens, which
is the bilateral compatibility model rejected in cycle 17 and is already
represented by explicit grasp-pair networks such as PhyGrasp.  Its empirical
motivation is likewise occupied by cycle 38: the ICLR 2026 random-toy work
trains on rigid assemblies of up to five primitives and already makes their
zero-shot compositional generalization the central result.  The proposed paper
would therefore combine a pairwise contact architecture with an existing
primitive-assembly curriculum, precisely the kind of recomposition that this
search must avoid.

Sources:

- https://arxiv.org/html/2402.16836v1
- https://arxiv.org/abs/2510.12866
- https://proceedings.mlr.press/v205/cai23a.html

## Decision-theoretic checkpoint after cycle 71

The current constraint box has a simple but important consequence.  Let
$O$ be the one-shot RGB-D observation, $G$ a grasp, and $Y\in\{0,1\}$
the standardized small-lift outcome.  For the stated utility, every
Bayes-optimal selector is determined by

$$
 \eta(o,g)=\Pr(Y=1\mid O=o,G=g),\qquad
 \pi^\star(o)\in\arg\max_g\eta(o,g).
$$

Predicting a contact certificate, a physical parameter, a valid set, an
uncertainty object, or a geometric latent can improve statistical or
computational efficiency only if it is a more learnable sufficient statistic
for this same conditional law.  Renaming such a latent is not a new task.  A
genuinely different paper must therefore change at least one of the following
or establish a previously unknown and empirically material structure of
$\eta$:

1. the information available at decision time;
2. the action or execution contract;
3. the utility/risk functional;
4. the supervision and distribution-shift regime;
5. the computational access model for continuous grasp search.

The search has now audited each category.  Candidates D and F and cycles 6,
16, 20, 32--37, 40, 45--46, 54, 61, 63, 66, and 68 cover information and
identifiability.  Cycles 4, 12, 15, 25--31, 39, 42, 47, 51, 55--56, 58--60,
64, and 67 cover alternative action/mechanical objects.  Cycles 3, 7--11, 14,
23--24, 48, 59, and 65 cover robustness, risk, certificates, topology, and
selection.  Candidates E and F and cycles 18, 22, 34, 38, 41, 43--44, 49,
52--53, 57, 62, 69--71 cover computation, data, generalization, and structural
inductive biases.

The latest physically validated datasets do not expose an independent
mismatch.  GraspIT replaces force-closure-only labels by simulated
trajectory/slip validation, which returns to oracle fidelity and richer outcome
labels already covered by Candidate E and cycles 53 and 58.  FIRMGrasp makes
friction risk explicit, which confirms that cycle 15's capability boundary and
risk-aware wrench metrics are occupied.  This is evidence against forcing a
Candidate G from the current ingredients, not evidence that grasping is solved.

The next candidate must pass a stronger entrance test before receiving a
numbered cycle:

- state the new estimand in one line without “better representation of grasp
  success”;
- name the new information, action, utility, supervision, or access assumption
  that makes it non-equivalent to $\eta$;
- show that this assumption is present in the laboratory rather than added for
  novelty;
- identify a measurable failure curve of current SOTA that the new estimand
  predicts;
- survive comparison to all 71 cycles before a model is designed.

Sources:

- https://arxiv.org/abs/2607.05869
- https://arxiv.org/abs/2607.25049

### 72. Anytime prefix-optimal grasp portfolios

Proposal: replace a conditional grasp distribution by an ordered sequence
$(g_1,\ldots,g_K)$ whose every prefix is optimized for the available
candidate budget.  For oracle utility $Q$, a prefix-regret objective could be

$$
 \mathcal L_{\mathrm{prefix}}
 =\sum_{k\in\{1,2,4,\ldots,K\}}w_k
 \left[
 \max_{g\in\mathcal G}Q(o,g)
 -\max_{i\le k}Q(o,g_i)
 \right].
$$

An extensible low-discrepancy latent sequence transported to the parallel-jaw
quotient, or a jointly decoded ordered set with conditional repulsion, could
avoid redundant diffusion samples and expose a grasp-quality-versus-latency
curve rather than one result at an arbitrary sample count.

Rejection: under the stated one-shot utility, only the first executed action
has value.  If the model can estimate $Q(o,g)$, the Bayes solution is one
argmax; a portfolio becomes necessary only because search is approximate or
because a downstream IK/collision/planning oracle may reject candidates.  The
first case is an amortized-optimization implementation detail and returns to
cycles 8, 18, and 65.  The second changes the target to the whole-cycle
pipeline that the brief excludes.  Multiple Choice Learning has optimized
oracle loss of multiple structured outputs since 2012, GPNet directly predicts
a diverse set of 6-DoF parallel-jaw grasps under sampling cost, and MISO now
learns multiple diverse initial solutions for strict-runtime optimization.
Weighting their set loss over all prefixes is useful engineering, but it does
not create a new physical or statistical estimand.  A low-discrepancy latent
sequence also has no coverage guarantee after a learned transport unless the
high-utility preimage has strong regularity, at which point direct argmax
learning is simpler.

Sources:

- https://proceedings.neurips.cc/paper/4549-multiple-choice-learning-learning-to-produce-multiple-structured-outputs.pdf
- https://proceedings.neurips.cc/paper/2020/file/994d1cad9132e48c993d58b492f71fc1-Paper.pdf
- https://esharony.me/projects/miso/

### 73. Grasp-conditioned partial-information decomposition of RGB-D

Proposal: for each grasp query $g$, decompose the information that RGB $R$
and depth $D$ carry about lift outcome $Y$ into unique, redundant, and
synergistic terms,

$$
 I((R,D);Y\mid g)
 =U_R(g)+U_D(g)+R_{RD}(g)+S_{RD}(g).
$$

A query-local mixture of experts could route contact geometry to the
depth-unique branch, appearance or boundary evidence to the RGB-unique branch,
and cross-modal consistency to a synergy branch.  Training with paired modality
dropout and an information-decomposition regularizer might appear to offer a
principled response to noisy or missing depth instead of another arbitrary
fusion block.

Rejection: partial-information decomposition has no generally agreed unique
redundancy functional, and estimating its conditional atoms for
high-dimensional continuous RGB-D observations is harder than estimating the
grasp outcome itself.  Whatever decomposition is chosen, the Bayes decision
still depends only on $\eta(R,D,g)$; the atoms are explanatory auxiliaries,
not a new decision target.  The robotics mechanism is also occupied:
depth-guided cross-modal attention explicitly handles unequal RGB/depth
quality, bilateral fusion networks learn adaptive modality weights, and newer
adaptive asymmetric branches target background and depth ambiguity.  The
proposal would therefore contribute a general multimodal regularizer plus an
RGB-D grasp application.  It predicts no physical failure curve beyond
ordinary modality ablations and overlaps Candidate F's calibrated sensor marks
when depth reliability is made explicit.

Sources:

- https://arxiv.org/abs/2302.14264
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10057080/
- https://www.sciencedirect.com/science/article/pii/S0893608026006118

## Scope decision required after cycle 73

No Candidate G can be defended inside the currently fixed contract without
violating at least one explicit instruction.  This is not a claim that all
parallel-jaw grasp research is exhausted.  It is a statement about the
intersection of the following requirements:

- one-shot wrist RGB-D is the only decision-time evidence;
- the action is an ordinary fixed-hardware parallel-jaw grasp;
- utility is success of the standardized millimetric pickup;
- RL, VLA, full approach-to-lift feasibility, causal failure vectors, and
  whole-scene latent geometry are excluded;
- all ideas already recorded in this ledger may not be revisited;
- the result must introduce a genuinely new estimand and plausibly beat current
  SOTA, rather than attach a general-ML regularizer to a grasp pipeline.

Under these conditions, the decision-theoretic checkpoint shows that a new
auxiliary output is only a different parameterization of
$\eta(o,g)=\Pr(Y=1\mid o,g)$ unless it exploits new information, a new action,
a new utility, a new supervision/access regime, or a demonstrably new structure
of $\eta$.  Cycles 1--73 have now tested those structures across geometry,
mechanics, uncertainty, topology, symmetry, multimodality, data, simulation,
optimization, and finite candidate access.  Continuing to rename the same
conditional law would directly contradict the requested novelty standard.

Research can resume only after at least one scope choice is supplied.  Minimal
choices are:

1. permit a deployment-time signal beyond the initial RGB-D, such as separate
   finger encoders, motor current, force, or tactile data during closure;
2. permit a short passive RGB-D sequence obtained during unavoidable wrist
   motion, without adding a separate next-best-view action;
3. permit previous outcomes for a recurring object or scene;
4. permit a new controllable grasp variable such as closing force, compliance,
   or replaceable pad state;
5. permit a different operational utility or a downstream constraint that
   gives a grasp portfolio genuine value;
6. permit reopening and substantially revising one of Candidates D, E, or F.

Choosing one item does not make a paper novel by itself; it only opens a
nonempty search region in which a new gap can be sought.  Until then, assigning
a Candidate G would be scientifically misleading.

## Fifth independent search pass: action-frame and endogenous-query effects

This pass did not reopen Candidates D, E, or F.  It tested whether the
eye-in-hand embodiment or the fact that grasp candidates are themselves
computed from noisy RGB-D creates a new learning object while keeping the
one-shot information, action, and utility contract unchanged.

Two new 2026 boundary results make an ordinary representation proposal still
less defensible.  SpaHybGen already learns hardware-agnostic spatial contact
features from noisy depth and combines them with differentiable grasp
optimization across seven different hands:
https://www.nature.com/articles/s42256-026-01292-y.  FPTE already rescored the
terminal configurations produced by motion planning rather than the nominal
pre-planning grasps and reported a large real-world confidence difference:
https://martinmatak.github.io/fpte/.  Thus neither a universal contact interface
nor evaluation of the realized planner output is an open estimand.

The official ICLR 2027 reviewer guide also sharpens the acceptance test used in
this ledger.  It asks whether a paper supports a specific well-motivated
question, creates significant new knowledge or value, and may take the field in
a new direction; SOTA is explicitly not required by the venue:
https://iclr.cc/Conferences/2027/ReviewerGuidelines.  The call for papers asks
authors for ambitious "slow science" rather than work inflated merely because
it can now be completed faster:
https://iclr.cc/Conferences/2027/CallForPapers.  Therefore a claimed SOTA gain is
treated here as necessary for the laboratory ambition but not as a substitute
for a new scientific object.

### 74. Self-observed action-frame grasping

Proposal: when part of the gripper is visible in the wrist RGB-D image, infer a
per-frame camera-to-gripper action frame from the gripper pixels and express all
grasp queries relative to that visually observed frame.  A joint equivariant
model could then make the predicted grasp invariant to persistent mounting
error and responsive to camera or wrist deflection, without a calibration
target or extra deployment sensor.

Rejection after the dedicated calibration audit: learning eye-in-hand
calibration from a single image of the visible end effector is already the
exact perception problem studied by Valassakis et al.; their alternatives
include direct pose regression, learned 2-D/3-D correspondences followed by
PnP, and gripper depth/segmentation followed by registration:
https://proceedings.mlr.press/v164/valassakis22a.html.  Earlier large-scale
grasp learning also explicitly learned hand--eye coordination without camera
calibration or the current robot pose:
https://arxiv.org/abs/1603.02199.  Coupling either estimator to a modern 6-DoF
grasp detector would be valuable system engineering, but it would compose an
occupied calibration task with an occupied grasp task.  If the visible gripper
is removed from the input, the proposal collapses to persistent latent
calibration from outcomes, already rejected in cycle 33.  If it remains, the
new information is a calibration object, so this path also cannot claim to
solve the unchanged RGB-D grasp problem by a new structure of
$\eta(o,g)$.

### 75. Endogenous acquisition--candidate coupling

Proposal: model the fact that deployment candidates are random functions of the
same noisy observation that the evaluator receives.  If
$O_\varepsilon$ is a noisy acquisition and
$g_\varepsilon=\Gamma(O_\varepsilon)$, then the deployed evaluator is queried
on the graph

$$
  \{(O_\varepsilon,\Gamma(O_\varepsilon))\},
$$

not on independently perturbed point clouds and fixed ground-truth grasps.  A
candidate idea was to learn a graph-tangent correction involving
$(I,D\Gamma)$ and matched sensor perturbations, so that common-mode changes in
the crop, anchor, and grasp pose are not mistaken for independent execution
noise.

Rejection: once $O$ and the actually queried $g$ are both inputs, a proper
supervised loss on their deployment joint distribution still estimates the
ordinary conditional law $\eta(O,g)$.  Generating candidates after every
training augmentation, or simply training on-generator, reproduces the
endogenous joint distribution without a new target or theorem.  GraspGen
already identifies discriminator distribution shift and obtains its strongest
results with on-generator training: https://arxiv.org/abs/2507.13097.  Changes
between different generators are cycle 41; acquisition-law changes with fixed
physics are Candidate F; independent execution perturbations are cycles 3 and
7.  A tangent correction may reduce data cost for one frozen generator, but it
is an errors-in-variables regularizer rather than a new grasping problem and
cannot support a broad SOTA claim.

### Result of the fifth pass

The wrist embodiment supplies a useful calibration cue only when the gripper is
visible, which changes the available information and is already occupied.
Candidate generation from noisy RGB-D does create a coupled covariate law, but
ordinary on-generator ERM is statistically sufficient for it.  Neither route
passes the post-cycle-73 entrance test.  The scope-decision requirement at the
end of the previous pass therefore remains active.

## Sixth independent search pass: rare reliability events

This pass tested a supervision/access hypothesis not used by the preceding
cycles: perhaps the important label for a selected high-quality grasp is a very
small failure probability, for which ordinary Monte Carlo simulation produces
almost no negative examples.  The mathematical inspiration came from large
deviations, subset simulation, and importance sampling; robotics work was used
only to check whether the regime is operationally material.

### 76. Large-deviation grasp reliability

Proposal: let $\xi\in\mathbb R^d$ collect only declared execution
perturbations, with $\xi\sim\mathcal N(0,\sigma^2\Sigma)$, and let
$F_{z,g}=\{\xi:Y(z,g,\xi)=0\}$ be the failure set of the standardized terminal
close-and-small-lift protocol.  Instead of learning a binary score from crude
Monte Carlo labels, learn the reliability exponent

$$
 I(z,g)=\inf_{\xi\in F_{z,g}}
 \tfrac12\xi^T\Sigma^{-1}\xi .
$$

Under regularity assumptions, the Laplace principle suggests

$$
 -\sigma^2\log P_\sigma(F_{z,g})\longrightarrow I(z,g)
 \qquad(\sigma\downarrow0).
$$

An initially attractive training scheme was an observation- and
grasp-conditioned proposal $r_\phi(\xi\mid o,g)$, represented by a small
normalizing flow.  It would be trained across many scene--grasp contexts to
oversample dominating failure perturbations while retaining an unbiased
importance-weighted estimate

$$
 \widehat p_f(o,g)=\frac1N\sum_{i=1}^N
 \mathbf1\{Y_i=0\}
 \frac{p_\sigma(\xi_i)}{r_\phi(\xi_i\mid o,g)},
 \qquad \xi_i\sim r_\phi(\cdot\mid o,g).
$$

The visual deployment model would predict either $p_f$, $I$, or a compact
mixture of dominating points, and rank grasps by estimated reliability.  This
would be inference-efficient; the expensive sampling is entirely offline.

Rejection after the mathematical and empirical audits:

1. The exponent $I$ is the squared Mahalanobis distance from the nominal
   execution to the failure set.  For a locally smooth failure boundary it is
   exactly an anisotropic perturbation tolerance, already rejected in cycle 3;
   smoothing its indicator over $\xi$ is cycle 7.  Multiple dominating points
   enrich the description of that same tolerance boundary but do not define a
   different grasping estimand.
2. The sampling method is an application of a mature general construction.
   Large-deviation-initialized adaptive importance sampling already targets
   expensive high-dimensional failure probabilities and identifies an
   informative low-dimensional subspace:
   https://arxiv.org/abs/2209.06278.  Normalizing-flow importance samplers
   already learn nested rare-event proposals:
   https://arxiv.org/abs/2310.19167, and certifiable deep importance sampling
   for black-box systems already supplies statistical efficiency guarantees:
   https://arxiv.org/abs/2111.02204.  Conditioning or amortizing the flow over
   RGB-D/grasp contexts is standard conditional amortization, not a new
   rare-event principle.
3. The laboratory's useful error regime is not established as rare.  Current
   real grasp systems and the intended occluded/noisy setting commonly have
   failure rates of order $10^{-1}$, where hundreds rather than millions of
   matched simulations estimate a Bernoulli probability adequately.  Rare-event
   machinery becomes decisive near $10^{-4}$ or below, but a contact
   simulator is not physically calibrated to four decimal places.  It would
   produce a precise probability for the wrong physical law.
4. Robotics already validates robustness by repeated dynamics and pose
   perturbations.  Physically based evaluation under pose uncertainty reports
   that adding dynamics and uncertainty improves agreement with real grasp
   success:
   https://publications.ri.cmu.edu/physically-based-grasp-quality-evaluation-under-pose-uncertainty.
   QD Grasp explicitly labels sim-to-real transferability through robustness to
   randomized joint states: https://qdgrasp.github.io/sim2real_labelling/.
   These establish the value of perturbation labels but not an unmet need for
   extreme-tail estimation.
5. A cheap pilot would likely expose the mismatch immediately: measure the
   selected-grasp failure distribution with 200--500 matched perturbation trials
   per grasp.  If failures are frequent, standard stratified Monte Carlo wins;
   if no failures appear, a small real trial set cannot validate the claimed
   tail ordering.  There is no credible middle regime in which a learned
   $10^{-4}$ simulator tail can be shown to improve one-shot real grasping.

Consequently, rare-event simulation could reduce the offline cost of an
industrial reliability study, but the paper would be summarized as established
importance sampling applied to an established grasp-tolerance target.  It fails
both the independent-estimand and strong-indirect-evidence requirements and is
not promoted.

## Seventh independent search pass: selection exposure and supervision gauges

Date: 2026-08-25.

This pass did not reopen any of Candidates D--F or cycles 1--76.  It tested a
different possibility: perhaps the missing scientific object is created not by
contact geometry itself, but by the way a stochastic grasp oracle is queried,
compared, and then maximized.  The mathematical starting points were extreme-
value selection, semiparametric transformation models, common-random-number
experiments, critical-point geometry, and behavioral quotients.  Robotics work
was used only to test whether the corresponding phenomenon is already exposed
or whether the formulation changes the laboratory contract.

Two recent empirical results make this pass worth performing but do not by
themselves establish a new problem.

1. GraRe reports that, for fixed candidates, re-ranking raises detector success
   at rank one from 49.35% to 59.97%, while an oracle ordering reaches 98.95%.
   It also reports large real-robot gains while changing only the candidate
   order.  Thus candidate ranking contains substantial headroom, but re-ranking
   is now itself an explicit 6-DoF grasping task:
   https://arxiv.org/abs/2608.00946.
2. GraspGen explicitly identifies discriminator distribution shift and obtains
   a strong result with on-generator discriminator training.  Consequently,
   exposing a scorer to candidates produced by its actual generator is an
   occupied remedy, not a new principle:
   https://arxiv.org/abs/2507.13097.
3. TARGO shows that direct single-view target grasping still deteriorates with
   occlusion and that training on occlusion-induced failures is useful.  This
   establishes continuing empirical need, but its completion-and-fusion model
   already makes ordinary target-aware grasp scoring under occlusion an occupied
   problem: https://targo-benchmark.github.io/.

### 77. Search-budget-stable grasp evaluation

Proposal: a learned evaluator is not merely queried at a typical grasp.  Given
independent candidates $G_1,\ldots,G_N\sim\pi(\cdot\mid o)$, deployment selects

$$
 \widehat G_N\in\arg\max_i s_\theta(o,G_i).
$$

If the conditional score CDF under the proposal is $F_{\theta,o}$, the
selected-action law is, ignoring ties,

$$
 P_{\theta,N}(dg\mid o)
 =N\,\pi(dg\mid o)
   F_{\theta,o}(s_\theta(o,g))^{N-1}.
$$

Thus increasing $N$ concentrates evaluation on an extreme tail where small
systematic score errors can dominate.  A candidate formulation was to learn a
score whose *physical* selected-grasp utility is nondecreasing over a declared
budget range, using selection-law-weighted proper losses and a budget-uniform
pessimistic correction.  The benchmark would report true success versus
candidate budget rather than AP at one arbitrary $N$.

Rejection: the phenomenon is the general Best-of-$N$ reward-overoptimization
problem.  ICLR 2026 work already analyzes smoothing and regret for Best-of-$N$,
and separate accepted work applies uncertainty-pessimistic reward estimates to
prevent reward hacking under larger search:
https://openreview.net/forum?id=tCv1D3M7Lb and
https://openreview.net/pdf?id=EZn2TmBBfF.  In grasping, on-generator training
already attacks the proposal/evaluator shift, while post-selection calibration,
proposal-density correction, and top-one losses were separately excluded in
earlier cycles.  More fundamentally, with the true

$$
 \eta(o,g)=P(Y=1\mid o,g),
$$

best-of-$N$ cannot hurt in expectation: the problem is estimation error in
$\eta$, not a new physical estimand.  A selection-weighted loss or lower
confidence correction may be useful, and the budget curve is a worthwhile
diagnostic, but the paper would be a grasping instance of reward-model
overoptimization rather than a new grasp-learning object.

### 78. Scene-wise ordinal grasp fields under unknown oracle gauges

Proposal: analytic metrics, different simulators, and robot platforms need not
share a meaningful absolute quality scale, even if each approximately preserves
which grasp is better within one scene.  Suppose oracle $a$ exposes

$$
 M_{a,o}(g)=h_{a,o}(r(o,g))+\epsilon,
$$

where $h_{a,o}$ is an unknown strictly increasing link.  The physically
relevant latent $r(o,g)$ would be identifiable only up to a scene-wise
monotone gauge, but its argmax would remain invariant.  A gauge-invariant model
could train solely from within-scene order constraints, pool heterogeneous
oracles without pretending their numbers are calibrated, and learn one
continuous field on the parallel-jaw action quotient.

Rejection: rank-based estimation under unknown monotone transformations is a
mature semiparametric construction.  Maximum-rank-correlation estimators,
transformation models for ranking, and order-based multivariate regression
already provide invariance to unknown monotone links and consistency results:
https://jmlr.csail.mit.edu/beta/papers/v12/vanbelle11a.html and
https://doi.org/10.1016/j.jmva.2017.01.012.  The grasp-specific assumption is
also too strong.  A cheap force-closure metric and a compliant physical trial
can reverse, not merely rescale, candidate order because they omit different
mechanics.  If order is preserved, ordinary pairwise/listwise learning suffices;
if it is not, the gauge model is misspecified.  The proposal therefore offers
neither a new estimator nor credible fusion of inconsistent physical oracles.

### 79. Matched-perturbation preference supervision

Proposal: simulator labels for two grasps are normally estimated with
independent execution perturbations.  Instead, apply the same pose, control,
and load perturbation seed $\xi$ to every candidate in a scene and supervise
pairwise differences

$$
 \Delta_{g,h}(o)
 =E_\xi[Y(o,g,\xi)-Y(o,h,\xi)].
$$

When outcomes for nearby candidates are positively correlated under the same
$\xi$, common random numbers can dramatically reduce the variance of the difference
relative to two independent Monte Carlo estimates.  A skew-symmetric comparator
$d_\theta(o,g,h)=-d_\theta(o,h,g)$ with cycle-consistency could then train a
tournament or recover a scalar potential, concentrating simulation on close
ranking decisions rather than estimating every marginal to the same accuracy.

Rejection: common-random-number ranking and selection is a classical simulation-
optimization tool, including sequential Bayesian procedures for dependent
replications and multiple-comparison guarantees:
https://arxiv.org/abs/1410.6782 and
https://doi.org/10.1007/s10479-015-2019-x.  The matched design can reduce offline
label cost, but it does not change the population identity

$$
 \Delta_{g,h}=\eta(o,g)-\eta(o,h).
$$

Recovering a potential returns to an ordinary scalar grasp field; allowing
intransitive pairwise outputs contradicts that identity.  Exact common
perturbations are also easy in simulation but not in separate physical grasps,
where the object must be reset.  This is a sound data-efficiency ablation, not
an independent ICLR thesis.

### 80. Morse field of bilateral normal chords

Proposal: for a smooth visible or completed surface $S$, ideal parallel-jaw
antipodal contacts are stationary points of the squared pair distance on
$S\times S$.  For $x,y\in S$, define

$$
 F_S(x,y)=
 \left(P_{T_xS}(y-x),\;P_{T_yS}(x-y)\right).
$$

$F_S(x,y)=0$ means the connecting chord is normal to both tangent planes.
The Hessian of the pair-distance function classifies whether the chord is a
stable local width minimum, a saddle, or a maximum.  An initially attractive
model would predict this integrable vector field and its Morse index from a
partial point cloud, use root finding to obtain a sparse set of bilateral
contacts, and attach approach/collision variables only afterward.  This is more
structured than classifying arbitrary point pairs and could make candidate
generation scale with critical chords rather than all pairs.

Rejection: the mathematical re-expression does not change the grasping object.
Parallel-jaw antipodal planning already seeks opposing surface points whose
connecting line agrees with both normals, and planners already prefer local
width minima because perturbations increase squeeze rather than release it:
https://motion.cs.illinois.edu/RoboticSystems/AdvancedTopicsInPlanning.html.
Learning the zero set of $F_S$ is therefore another estimator of classical
antipodal pairs.  Under single-view occlusion, the opposing tangent plane is
often missing; predicting it returns to hidden contact completion.  On noisy
point clouds, estimating a Hessian of pair distance is less stable than the
contact-pair embeddings already used by learned grasp models.  Morse language
would add an elegant candidate-generation prior but no new estimand, and
approach/collision scoring would restore the ordinary pipeline.

### 81. Controller-induced behavioral quotient of grasp commands

Proposal: two commanded poses can be geometrically different yet become
indistinguishable under a compliant position/force controller.  Let
$K(\cdot\mid o,g)$ be the distribution of terminal contact state and binary
small-lift outcome produced by the fixed closure controller.  Define

$$
 g\sim_o h
 \quad\Longleftrightarrow\quad
 K(\cdot\mid o,g)=K(\cdot\mid o,h).
$$

Learning the quotient $\mathcal G/{\sim_o}$ could remove redundant commands,
place one representative in every mechanically distinct class, and make grasp
search depend on controller behavior rather than an arbitrary Euclidean metric
on $SE(3)$.

Rejection: if the kernel contains only the terminal success bit, equivalence is
just equality of $\eta$ values and collapses physically unrelated grasps.  If it
contains the full terminal contact state, estimating it requires the closure
rollout and hidden mechanics excluded from the compact one-view contract.
Conceptually, the quotient is a behavioral-equivalence restatement of capture
regions, controller funnels, and grasp basins.  For the one-shot Bayes decision,
removing redundant equal-value actions cannot improve the best achievable
success; it can only reduce search cost.  A learned bisimulation metric would
therefore be an expensive parameterization of the old controller-induced basin
object, not a new grasping task.

### 82. Grasp selection for post-closure verifiability

Proposal: a standard electric parallel gripper usually exposes terminal width,
joint state, or motor current during closure.  One could select not only a grasp
likely to succeed, but one whose cheap post-closure telemetry makes secure
bilateral capture distinguishable from a miss before the millimetric lift.  If
$S$ is the trace and $Y$ physical retention, the action-dependent experiment
is $p(S,Y\mid o,g)$.  A compact selector could trade expected success against
the Bayes error or conditional entropy of verifying $Y$ from $S$, followed
by an abstain/release rule.

Status: not admitted under the current scope, and not promoted.  This proposal
genuinely changes decision-time information, which is why it escapes the scalar
$\eta$ checkpoint, but the current contract declares one-shot RGB-D as the evidence
for grasp choice and does not authorize an extra verification objective.
Moreover, proprioception-based grasping with joint position/torque sensing is
established (https://arxiv.org/abs/1803.09674), informative sensor-based grasp
planning has long optimized information acquired during grasping
(https://doi.org/10.1016/j.robot.2013.09.009), and current-as-touch methods now
learn contact feedback from motor current and joint state
(https://cat.chenyangma.com/).  A new paper would need a much sharper
information-theoretic result than "choose an informative grasp" and explicit
permission to use closure telemetry.  Without both, this is neither in scope nor
objectively novel.

## Result of the seventh pass

No direction in cycles 77--82 passes the post-cycle-73 entrance test.

- Search exposure, ordinal gauges, and matched comparisons alter estimation or
  data efficiency but retain the same Bayes target $\eta$.
- The normal-chord and behavioral-quotient constructions are mathematically
  structured restatements of antipodal pairs and capture basins.
- Post-closure verifiability introduces genuinely new information, but that is
  a scope change and its broad robotics principle is already established.

This conclusion is consistent with the official ICLR 2027 criterion: reviewers
are asked whether the work creates significant new knowledge or value and are
explicitly told that SOTA is not required.  A large leaderboard gain cannot
substitute for a distinct, well-supported scientific question:
https://iclr.cc/Conferences/2027/ReviewerGuidelines.

The scope-decision requirement after cycle 73 therefore remains active.  Under
the fixed one-view, fixed-action, one-shot-success contract, promoting a new
candidate from this pass would knowingly relabel an established ranking,
simulation, or antipodal-grasp construction.  Research should resume only after
one real laboratory resource not represented in the contract is confirmed
(for example, an unavoidable passive RGB-D sequence or closure telemetry), or
after permission is given to reopen and materially revise a prior candidate.

## Eighth independent search pass: action semantics and decision rate

This pass again left cycles 1--82 closed.  It tested two high-level hypotheses
that do not begin from another contact representation:

1. perhaps a commanded pose is not the action at all unless its closure
   controller is specified, so the correct learning object is a
   controller-indexed family;
2. perhaps efficient grasp learning should be formalized through the minimum
   information rate needed for low decision regret rather than through point
   count, latent size, or reconstruction error.

Both directions are mathematically coherent.  Neither survives as Candidate G
under the actual laboratory contract and novelty threshold.

### 83. Controller-indexed grasp semantics

Proposal: a pose and width do not uniquely define physical execution.  Let
$c\in\mathcal C$ denote a compact closure-controller description: commanded
force or stiffness, closing velocity, position/force mode, and a small number
of saturation parameters.  Define

$$
 Q(o,g;c)=P(Y=1\mid O=o,G=g,C=c).
$$

Instead of fitting a separate scorer for each controller, learn the conditional
operator

$$
 \mathfrak Q_o:\mathcal C\longrightarrow C(\mathcal G),\qquad
 c\longmapsto Q(o,\cdot;c).
$$

An attractive compact model would encode the RGB-D observation once, represent
$c$ by dimensionless controller groups, and use a low-rank separable decoder

$$
 Q_\theta(o,g;c)
 =\sigma\!\left(
 b_\theta(o,g)+
 \sum_{j=1}^{r}
 u_{\theta,j}(o,g)v_{\theta,j}(c)
 \right).
$$

The empirical claim would be that controller variation changes the grasp field
through a low-dimensional response subspace.  If true, a small factorial
dataset over controllers could adapt a fixed-hardware grasp evaluator without
retraining its geometric encoder.  This is an operator over action semantics,
not a prediction of the full closure trajectory.

Rejection: the current setup has one fixed gripper and controller.  Adding
$c$ creates variation solely to justify the new model; it is not an
unrepresented laboratory nuisance known to vary at deployment.  With a fixed
$c_0$, the operator collapses exactly to the ordinary scalar field
$Q(o,g;c_0)=\eta(o,g)$.  If force or compliance is made a controllable action,
the required outcome depends on material, mass, friction, and deformation that
single-view RGB-D does not identify reliably.  Force-regulated grasping
accordingly uses tactile feedback, and a recent force-controlled parallel-jaw
system reports that tactile feedback is essential:
https://arxiv.org/abs/2602.10013.  Cross-hand and cross-gripper learning already
conditions policies or evaluators on embodiment/controller interfaces, while
commercial-gripper control itself is a mature field:
https://arxiv.org/abs/2404.09150 and
https://doi.org/10.3390/act12080332.

The low-rank conditional operator would therefore be an efficient
multi-controller surrogate, not a new physical problem.  Its strongest
experimental benefit would require either a variable controller absent from the
brief or extra tactile/force information.  It is not promoted.

### 84. Grasp rate--regret function

Proposal: replace informal claims such as “the model needs only a local crop”
with an information-theoretic object.  Let an encoder produce a stochastic
finite representation $Z\sim p_\phi(z\mid O)$, and let
$\pi:\mathcal Z\to\mathcal G$ select one grasp.  Define decision distortion

$$
 d(o,z)
 =
 \max_{g\in\mathcal G}\eta(o,g)
 -
 \eta(o,\pi(z)).
$$

The **grasp rate--regret function** would be

$$
 R_{\mathrm{grasp}}(D)
 =
 \inf_{\substack{p(z\mid o),\,\pi\\
                  E[d(O,Z)]\le D}}
 I(O;Z).
$$

It asks how many bits of a noisy RGB-D observation are necessary to choose a
grasp within expected physical regret $D$, without reconstructing shape.  A
practical estimator could use a vector-quantized ray tokenizer, an entropy
model, and a Lagrangian

$$
 \mathcal L_{\beta}
 =
 E[-Y\log q_\theta(Z,G)-(1-Y)\log(1-q_\theta(Z,G))]
 +\beta\,E[-\log p_\psi(Z)],
$$

augmented by a differentiable top-one regret surrogate on shared candidate
sets.  Reporting the entire bits--regret frontier would be more honest than
choosing one latent width, and could reveal whether contact decisions truly
need less information than amodal geometry.

Rejection: task-oriented rate--distortion, semantic compression, and
information-bottleneck learning already formalize the minimum information
needed for downstream inference or decisions.  In particular, rate--distortion
has already been used to define what an action-selecting learner should learn
under regret:
https://proceedings.mlr.press/v139/arumugam21a.html.  Semantic
rate--distortion with task variables and side information is also explicit:
https://arxiv.org/abs/2208.06094.  The proposed neural implementation is a
standard variational/VQ information bottleneck with a grasp loss.

The grasping-specific frontier could be a valuable measurement paper, but it
does not itself improve the uncompressed Bayes action and cannot support the
requested physical SOTA claim.  If “SOTA” is redefined as a memory/bitrate
Pareto frontier, the project objective has been changed from reliable grasping
to compression.  If the rate penalty is removed to compete on grasp success,
the new formulation disappears.  This is therefore an informative systems
diagnostic, not Candidate G.

## Result of the eighth pass

The action-semantics direction requires controller variation or contact sensing
that is not in the fixed input/action contract.  The rate--regret direction
supplies a rigorous definition of efficiency but imports a mature general
framework and has no mechanism for higher physical success at full information.
Neither creates the simultaneously new, necessary, and potentially SOTA grasp
estimand required by the brief.

## Research-status audit after cycle 84

This is a status audit against the original deliverable, not another candidate
and not a claim that robotic grasping has no open problems.

| Required property | Current evidence | Status |
|---|---|---|
| Parallel-jaw, noisy wrist RGB-D, shelf/one frontal occluder | Fixed throughout the ledger and checked against TARGO/CAPGrasp-style boundaries | satisfied as search scope |
| No RL, VLA, whole-cycle feasibility, causal failure taxonomy, or whole-scene SDF input | Enforced in every promoted/rejected formulation | satisfied as search scope |
| A scientific object not already present in this document | Cycles 1--84 cover every subsequently proposed object; the user explicitly excludes reusing them | **missing** |
| Necessity under the fixed one-shot contract | The Bayes checkpoint shows that auxiliary outputs reduce to estimating $\eta(o,g)$ unless information, action, utility, supervision, or access changes | **missing for every post-checkpoint proposal** |
| Efficiently learnable compact model | Several compact models were derived, but all belong to rejected or explicitly excluded ideas | not admissible |
| Objectively strong novelty after primary-source audit | No post-cycle-73 direction survives its closest general-ML/mathematics and robotics occupancy checks | **missing** |
| Credible mechanism for physical SOTA | Recent papers show empirical headroom under occlusion and ranking, but no unoccupied proposed mechanism explains how to capture it | **missing** |
| ICLR-level support | The official criteria require significant new knowledge/value and rigorous support; leaderboard improvement alone is insufficient | **missing without a surviving question** |

The blocker is therefore exact.  Under the current contract the only deployment
random variables are $O$, $G$, and $Y$, and the decision is

$$
 \pi^\star(o)\in\arg\max_g P(Y=1\mid O=o,G=g).
$$

After excluding all previously recorded structures of this conditional law, no
independent estimand remains.  Inventing another representation of
$P(Y=1\mid O,G)$ would violate the requested novelty standard; introducing a
controller, telemetry, a passive view sequence, previous outcomes, a new grasp
variable, or a different utility without confirmation would violate the
laboratory contract.

Research can be resumed without lowering the scientific bar once at least one
of the following facts is supplied:

1. a real additional signal that is already available in the setup and may be
   used by the formulation;
2. a real action/control degree of freedom that varies at deployment;
3. a utility or downstream requirement omitted from the current one-shot
   small-lift definition;
4. permission to reopen one prior candidate and replace, rather than merely
   decorate, its rejected core.

Until then, the correct research status is **blocked by an underdetermined scope
choice**, not “Candidate G found.”  This avoids converting mathematical
vocabulary into a false novelty claim.

## Scope reopening after cycle 84

On 2026-08-25 the user explicitly allowed all four previously listed ways to
change the contract: an additional signal, a deployment-time control variable,
a modified utility, and reopening Candidates D--F provided that their rejected
cores are replaced rather than decorated.  This removes the decision-theoretic
blocker above.  It does **not** establish that any particular signal or command
is implemented on the laboratory robot.  Every new candidate must therefore
state its hardware assumption as a falsifiable entrance condition.

The first reopened contract considered below is deliberately small:

$$
 O_0=R_x(Z),\qquad
 A_{\rm probe}=(v,\epsilon),\qquad
 O_1=R_{x+\epsilon v}(Z),\qquad
 G=g,\qquad Y\in\{0,1\}.
$$

Here $x$ is the calibrated wrist-camera pose, $v$ is one safe camera-twist
direction, and $\epsilon$ is a centimetre-scale or smaller probe amplitude.
The object and occluder remain stationary between $O_0$ and $O_1$.  The
terminal action is still one parallel-jaw close and millimetre-scale lift.
There is no RL, VLA, full approach-to-lift feasibility predictor, scene SDF, or
causal failure taxonomy.

**Hardware entrance condition.**  The wrist must be able to execute a
repeatable, collision-checked micro-translation or micro-rotation before grasp
selection and return two registered RGB-D frames.  If this is unavailable, the
candidate below is outside the real setup and must be rejected rather than
silently treating simulated camera motion as deployable.

## Ninth independent search pass: what the reopened contract actually buys

### Cycle 85: ordinary decision-focused next-best view

Proposal: predict a posterior over hidden geometry or grasp utilities and move
the wrist camera to the view with largest expected grasp-value improvement.

Rejection as a new scientific core: next-best-view grasping is already an
established robotics problem.  Recent examples explicitly couple view
selection to grasp affordance, information gain, or reconstruction:

- https://proceedings.mlr.press/v229/zhang23i/zhang23i.pdf
- https://papers.nips.cc/paper_files/paper/2024/file/4364fef031fdf7bfd9d1c9c56b287084-Paper-Conference.pdf
- https://arxiv.org/abs/2511.04199
- https://arxiv.org/abs/2606.19091

The general-ML analogue is also occupied.  Goal-oriented optimal experimental
design targets uncertainty in a downstream quantity of interest rather than
the latent parameter:
https://arxiv.org/abs/1802.06517.  Goal-driven Bayesian experimental design
now directly differentiates through a downstream decision layer:
https://arxiv.org/abs/2605.26093.  Replacing reconstruction entropy by grasp
regret is therefore not enough.  A reopened direction needs a new *measured
object*, not a new score for the familiar next-view pipeline.

### Cycle 86: a learned critical-force interval

Proposal: make commanded closing force $f$ part of the action and learn a
random lower holding threshold $F_{\min}(o,g)$, or a safe interval
$[F_{\min},F_{\max}]$ when damage is penalized.  Interval-censored attempts
would support survival-style learning and selection of a pose with a wide
force margin.

Rejection for the present target: without a damage or deformation utility,
larger available force makes the lower-threshold problem decision-trivial.
With that utility, the problem becomes gentle/force-regulated grasping, which
is already active and requires tactile or force instrumentation plus fragile
objects absent from the original task.  Relevant current systems already
learn force regulation and demonstrate the necessity of feedback:

- https://arxiv.org/abs/2409.10371
- https://arxiv.org/abs/2602.10013
- https://cat.chenyangma.com/

Threshold survival analysis would be an elegant parameterization, not an
unoccupied broad problem.

### Cycle 87: pre-lift stability from a small mechanical response

Proposal: close the fingers, apply a subcritical wrist or force dither, measure
motor current/joint response, and infer a frequency-domain stability margin
before committing to the lift.

Rejection as the main paper: grasp impedance, perturbation resistance, tactile
stability prediction, slip prediction, and proactive tactile safety margins
are already established:

- https://arxiv.org/abs/2208.02885
- https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202501051
- https://doi.org/10.1002/%28SICI%291097-4563%28199909%2916%3A9%3C509%3A%3AAID-ROB4%3E3.0.CO%3B2-K

A motor-current-only version might be inexpensive, but “identify a transfer
function, then classify stability” is system identification plus an occupied
grasp-verification task.  It also evaluates a grasp after it has already been
chosen, so a SOTA initial selector would require abort/regrasp logic and a much
larger execution contract.

### Cycle 88: replicated RGB-D cross-statistics

Proposal: collect a same-pose RGB-D burst and construct cross-frame
U-statistics whose expectation removes independent depth-noise bias from
pairwise geometric features.

Rejection: registered same-pose depth frames can first be averaged or robustly
filtered per pixel.  Unless the noise is irreducibly correlated or
material-dependent, cross-moment machinery does not reveal occluded geometry
and is unlikely to beat temporal denoising.  If camera motion is introduced,
the scientific object is no longer replicated noise but changing visibility,
which motivates cycle 90.

### Cycle 89: reopening Candidate E through $\Gamma$-convergence

Proposal: treat refining contact simulators as discrete variational
functionals $E_h$ and learn a decision map that is stable under
$\Gamma$-convergence of $E_h$ to $E$.

Rejection: this supplies stronger mathematical language but does not repair
Candidate E's ground-truth problem.  Different contact engines may encode
different compliance/friction laws rather than discretizations of one
functional, and convergence of minimizers does not imply a material,
physically predictive binary-grasp ambiguity band.  Multi-fidelity
classification already targets expensive failure boundaries:
https://arxiv.org/abs/1905.03406.  Without a positive matched-refinement pilot,
the proposal remains LimitGrasp with $\Gamma$-convergence decoration.

### Cycle 90: weak visibility response under controlled micro-motion

Proposal: do not reconstruct a second scene and do not score a large next
view.  Treat the calibrated change of RGB-D under a small camera motion as a
**measure-valued sensor response**.  Moving visibility boundaries create a
singular but finite information flux supported on occlusion contours.  Query
that flux directly with local grasp functionals.

This survives the first audit and is promoted below as a **conditional
Candidate G**.  It replaces Candidate D's unobserved low-rank utility
ellipsoid by a physically acquired sensor response, and it replaces Candidate
F's invariance to passive sampling laws by deliberate differentiation of the
observation law.  No low-rank hidden-shape assumption, robust maximin utility,
Gaussian deconvolution, or reference surface-density correction is retained.

## Candidate G: EdgeFlux -- decision learning from measure-valued visibility response

### 1. Broad scientific idea

Embodied sensors should sometimes learn not only from observations, but from
the **weak response of an observation to a calibrated infinitesimal action**.
This distinction matters when the sensor map is nonsmooth.  A small camera
motion transports most visible pixels smoothly, yet creates and destroys
surface area at occlusion boundaries.  The ordinary pixelwise derivative is
ill behaved precisely where new geometric information appears.

The hypothesis is:

> For partially occluded geometric decisions, an explicitly measure-valued
> representation of controlled visibility change is more sample- and
> resolution-efficient than treating two nearby frames as an ordinary tensor
> or reconstructing a complete scene.

Parallel-jaw grasping behind one frontal obstacle is a sharp instance.  A
small lateral wrist motion can reveal strips of the target immediately behind
the obstacle boundary.  Those strips may intersect a jaw pad, finger sweep, or
candidate contact neighborhood even when most hidden shape remains
unobserved.  The proposed learner represents only this new action-relevant
evidence.

The broad problem is **decision learning from singular sensor responses**.
Other possible instances include collision clearance behind an edge,
inspection of a partially covered part, and decisions driven by moving
interfaces.  These extensions are motivation, not experiments that may be
claimed without implementation.

### 2. The correct mathematical state space

Let $Z$ be a static labeled scene, $x\in SE(3)$ the wrist-camera pose, and

$$
 R_x(Z):\Omega\rightarrow\mathbb R^C
$$

the registered RGB, depth, confidence, and target/occluder-mark field on image
domain $\Omega$.  For a piecewise smooth scene, $R_x(Z)$ is naturally an
$SBV$ rather than a globally smooth field.  Its jump set
$\Gamma_x\subset\Omega$ contains depth and semantic occlusion boundaries.

For camera twist $v$ and small amplitude $\epsilon$, first warp persistent
visible surfaces from pose $x$ to $x+\epsilon v$ using calibrated
kinematics and measured depth.  The residual finite response is

$$
 \mu_{\epsilon,v}
 =
 \frac{R_{x+\epsilon v}(Z)
       -W_{\epsilon v}R_x(Z)}{\epsilon}\,\mathcal L^2.
$$

In the weak-* topology of finite Radon measures, the target limit has the
decomposition

$$
 \mu_v
 =
 a_v(u)\,\mathcal L^2
 +
 [R_x](u)\,V_n(u;v)\,
 \mathcal H^1\!\lfloor_{\Gamma_x}.
$$

The absolutely continuous part $a_v$ contains residual photometric, depth,
and registration change on persistent surfaces.  The singular part contains
the one-sided appearance/depth jump $[R_x]$ multiplied by the normal
velocity $V_n$ of the visibility boundary.  For this project the primary
object is the **target-birth component**: target surface that is visible at
$x+\epsilon v$ but was covered by the marked frontal obstacle at $x$.

With RGB-D, it is convenient to lift this component into 3-D as a positive
point measure

$$
 \nu_{\epsilon,v}
 =
 \frac{1}{\epsilon}
 \sum_{i\in\mathcal B_{\epsilon,v}}
 A_i\,c_i\,\delta_{X_i},
$$

where $\mathcal B_{\epsilon,v}$ is the registered target-disocclusion set,
$X_i$ is its 3-D point, $A_i$ is pixel footprint, and $c_i$ is depth
confidence.  The limit $\nu_v$ is a visibility-flux measure supported on the
one-sided target trace behind the occluder contour.  It is neither a completed
point cloud nor a posterior over hidden shape.

### 3. Why an ordinary finite difference is the wrong representation

The one-dimensional translation

$$
 f_t(s)=\mathbf 1\{s\ge t\}
$$

already shows the issue:

$$
 \frac{f_\epsilon-f_0}{\epsilon}
 =
 -\frac1\epsilon\mathbf 1_{[0,\epsilon)}.
$$

Its $L^2$ norm diverges as $\epsilon^{-1/2}$, but as a signed measure it
converges weakly to $-\delta_0$ with finite total variation.  For every
Lipschitz test function $\phi$,

$$
 \left|
 \int \phi(s)\frac{f_\epsilon-f_0}{\epsilon}\,ds+\phi(0)
 \right|
 \le \frac12\mathrm{Lip}(\phi)\epsilon.
$$

Thus concatenating or subtracting rasterized frames forces a model to resolve
an increasingly narrow and high-amplitude band, whereas querying the weak
measure remains stable.  This does not prove that a sufficiently large CNN or
two-view point transformer cannot learn the same decision.  It supplies the
specific approximation phenomenon that must be tested at unseen image
resolutions and probe amplitudes.

Differentiable rendering already derives boundary integrals and samples Dirac
terms caused by visibility:
https://cseweb.ucsd.edu/~tzli/diffrt/.  Classical dynamic-occlusion vision
already uses accretion/deletion to infer boundary ownership:
https://doi.org/10.1109/TPAMI.1985.4767638.  These facts validate the
mathematical decomposition but also prohibit claiming the boundary derivative
itself as new.  Candidate G's possible contribution is the *observed,
probe-normalized measure as a learnable decision representation*, its
grasp-query estimator, and the associated scale/decision theory.

### 4. Compact action-query model

Let $g\in SE(3)/H$ be a parallel-jaw pose modulo the gripper symmetry.
Choose $K$ smooth, local kernels in the gripper frame.  They cover the two
pad neighborhoods, finger-sweep boundaries, interior jaw volume, and a small
target/obstacle clearance band.  Query the visibility flux by

$$
 z_k(g;v)
 =
 \int \kappa_k(g^{-1}X)\,d\nu_v(X),
 \qquad k=1,\ldots,K.
$$

The deployment estimator is a weighted segmented reduction,

$$
 \widehat z_k(g;v,\epsilon)
 =
 \frac1\epsilon
 \sum_{i\in\mathcal B_{\epsilon,v}}
 A_i c_i\,\kappa_k(g^{-1}X_i).
$$

A lightweight base encoder supplies features
$b_\theta(O_{\cap},g)$ only from points local to the open gripper, where
$O_{\cap}$ contains persistent visible surface registered across the two
frames.  The same local registration/averaging is supplied to every
two-frame baseline, so ordinary denoising or extra sampling cannot be
misattributed to visibility flux.  A small query head predicts

$$
 q_\theta(O_{\cap},\nu_{\epsilon,v},g)
 =
 \sigma\!\left(
 h_\theta[
 b_\theta(O_{\cap},g),
 \widehat z(g;v,\epsilon),
 v,\epsilon,\widehat\sigma_D]
 \right).
$$

The probe representation contains $K$ numbers per candidate plus a small
set of boundary tokens; it does not retain a mesh, voxel grid, whole-scene
SDF, or sampled shape completion.  Candidate generation must be held fixed
in the representation ablation.  A secondary end-to-end result may regenerate
candidates after the probe, but it cannot substitute for that controlled
comparison.

The initial view can choose among a small safe set of probe twists without RL.
Let $\rho_\theta(u)$ be a learned density of how much revealing target
surface behind boundary point $u$ could change the current grasp ranking.
Use the analytically computable boundary speed to enumerate

$$
 v^*(O_0)
 =
 \arg\max_{v\in\mathcal V_{\rm safe}}
 \int_{\Gamma_{\rm target/occ}}
 \rho_\theta(u)[V_n(u;v)]_+\,d\mathcal H^1(u)
 -\lambda_{\rm move}c(v).
$$

This selector is not claimed as a new form of optimal experimental design.
Its role is only to make the new response measurable at low motion cost.
A fixed bidirectional lateral dither is the essential baseline; if learned
probe selection supplies the only gain, the paper risks collapsing back to
cycle 85.

### 5. Training data and objectives

Synthetic data should render the same isolated target and single frontal
occluder at a calibrated micro-baseline family.  Each latent scene has:

- one initial RGB-D frame;
- paired motions in four to eight safe tangent directions;
- amplitudes spanning sensor-noise-limited to ordinary next-view scales;
- exact target/occluder visibility-switch masks and boundary velocities;
- a shared, mesh-derived grasp candidate set and fixed small-lift labels.

Real pairs require robot-kinematic registration, repeated stationary scenes,
and an explicit test that the object did not move during sensing.  Target and
occluder marks may come from the laboratory perception stack, but results
must include predicted rather than oracle masks.

The primary loss is ordinary supervised grasp likelihood.  Synthetic meshes
also provide the limiting boundary integral $z_v^{\rm oracle}$, so an
auxiliary response loss can train target-birth detection and the weak query
directly:

$$
 \mathcal L
 =
 \mathcal L_{\rm BCE}(Y,q_\theta)
 +\lambda_{\rm birth}\mathcal L_{\rm birth}
 +\lambda_{\rm weak}
 \sum_{\epsilon\in\mathcal E_{\rm micro}}
 \|
 \widehat z_{\epsilon}
 -
 z_v^{\rm oracle}
 \|_1.
$$

Only amplitudes inside an empirically verified asymptotic regime belong to
$\mathcal E_{\rm micro}$.  Wider probes reveal genuinely different surface,
so forcing their responses to agree would be mathematically wrong.  Real data
without a latent mesh uses the outcome and birth-mask terms, not a fabricated
cross-scale target.  The auxiliary loss is not a contribution by itself.
The model earns its complexity only if explicit weak-measure queries
generalize better than equal-capacity raw-frame, fused-PCD, optical-flow, and
boundary-token baselines.

### 6. Theory targets

1. **Weak visibility derivative.**  Under piecewise $C^2$ surfaces,
   transverse visibility events, and calibrated camera motion, prove the
   $SBV$/Radon decomposition above and convergence of the lifted
   target-birth measure for smooth compactly supported 3-D queries.
2. **Raster-difference obstruction.**  Generalize the translated-step example
   to a moving image boundary and show why an $L^2$ finite-difference target
   has diverging norm while its total-variation measure remains bounded.  A
   meaningful approximation lower bound must specify the encoder class; a
   qualitative appeal to Dirac deltas is insufficient.
3. **Probe-scale bias--variance law.**  If the disoccluded strip has area
   $C\epsilon+O(\epsilon^2)$, pixel density is $n$, and retained marks have
   variance at most $\sigma^2$, a normalized smooth query should have
   squared bias $O(\epsilon^2)$ and variance
   $O(\sigma^2/(n\epsilon))$.  The resulting optimal physical scale

   $$
   \epsilon^*
   \asymp
   \left(\frac{\sigma^2}{n}\right)^{1/3}
   $$

   explains why the method uses a finite micro-motion rather than claiming
   that smaller is always better.  Correlated RealSense-style noise and
   registration error must appear in an extended bound.
4. **Uniform grasp-query consistency.**  For a compact candidate set and a
   uniformly bounded Lipschitz kernel family, bound

   $$
   \sup_g
   |\widehat z_k(g;v,\epsilon)-z_k(g;v)|
   $$

   by probe bias, effective pixel count, depth noise, mask error, and camera
   calibration error.
5. **Decision transfer.**  If the grasp head is $L$-Lipschitz in the flux
   sketch, uniform sketch error $\delta$ yields at most $2L\delta$
   additional top-one regret on a shared candidate set.  A margin larger than
   $2L\delta$ preserves the selected grasp.
6. **Response non-identifiability.**  If two latent scenes induce the same
   $(O_0,\nu_v)$ law for every allowed probe but have different optimal
   grasps, no policy under this contract can distinguish them.  This theorem
   states exactly which hidden geometry micro-motion cannot recover and
   prevents a false “solves occlusion” claim.

The ICLR 2023 result that linear-reconstruction neural operators can be
inefficient for moving discontinuities supplies indirect support for an
explicitly aligned nonlinear representation:
https://arxiv.org/abs/2210.01074.  It does not prove the proposed grasp result;
the representation ablation must do so.

### 7. Cheap falsification before model development

The first pilot needs no new backbone.

1. Take 30--50 TARGO-like isolated-object scenes with exactly one marked
   frontal occluder, 256--512 shared grasp candidates, and target visibility
   stratified from 10% to 70%.
2. Render paired RGB-D at lateral/vertical baselines of approximately
   2, 5, 10, and 20 mm, plus one ordinary next-view baseline.  Add calibrated
   depth noise and pose errors.
3. Measure the **oracle response value**: improvement of a classifier given
   the exact newly revealed target strip over the same classifier given
   $O_0$.  If the top-one success/regret improvement is below five
   percentage points in the heavily occluded stratum, stop; there is
   insufficient signal for any representation.
4. Measure how often newly revealed points intersect the local kernels of a
   grasp whose label/ranking was ambiguous from $O_0$.  If disocclusion is
   mostly far from candidate contacts and finger sweeps, the proposed
   task-local mechanism is false.
5. Compare five frozen-capacity inputs: $O_0$; raw concatenated frames;
   registered fused PCD; optical-flow/scene-flow plus frames; and the exact
   EdgeFlux kernel sketches.  EdgeFlux must improve unseen-amplitude and
   unseen-resolution regret by at least three points over the best two-view
   baseline, not merely beat the single-view model.
6. Replace exact switch masks with predicted masks and perturb camera
   extrinsics.  If half the gain disappears under realistic errors, stop.
7. Compare a fixed lateral dither, the cheap boundary-overlap selector, and a
   conventional large next-best view at equal sensing-plus-motion time.  The
   proposal needs a useful region of the success--latency Pareto curve; it
   need not dominate an unconstrained large view.
8. Only then run physical grasps.  Randomize object geometry, occlusion ratio,
   obstacle depth separation, texture, and probe direction.  Report both
   sensing failures and final millimetre-lift success.

The strongest cheap negative control is simply concatenating or fusing the
two frames.  If it matches EdgeFlux across probe scale, resolution, and noise,
the measure language has not bought learnability and Candidate G must be
rejected.

### 8. Novelty boundary

| Neighboring area | Already established | Candidate G must add |
|---|---|---|
| Differentiable rendering | visibility derivatives contain boundary/Dirac terms and can be edge-sampled | an observed, probe-normalized RGB-D response measure for learned downstream decisions |
| Dynamic occlusion/optical flow | accretion/deletion detects occlusion and depth ordering | calibrated 3-D target-birth flux queried by continuous grasp actions |
| Active grasp perception | next views optimize reconstruction, information gain, or grasp affordance; GCNGrasp-VP already uses one task-aware adjustment without reconstruction | finite micro-motion **and an explicit response representation** with a scale law tied to decision error; “one efficient view without reconstruction” is not novel |
| Multi-view point-cloud fusion | registers and aggregates all visible points | a compact singular-response representation with an equal-input generalization advantage |
| Event-based vision | camera motion produces sparse brightness-change events and has been used for grasping/occlusion-boundary detection | calibrated RGB-D target-birth mass, physical probe normalization, and continuous grasp queries; sparsity of temporal change alone is occupied |
| Neural operators with jumps | nonlinear reconstruction can beat linear bases for moving discontinuities | sensor-action response learning and physical decision evidence rather than a PDE solver |
| Candidate D | predicts an unobserved low-rank set of utility fields | acquires a new physical measurement; no low-rank or robust-ellipsoid hypothesis |
| Candidate F | removes dependence on passive point-sampling/noise law | exploits controlled visibility change; no inverse-density/Gaussian-deconvolution core |

The name, architecture, and table do not establish novelty.  Before a paper
claim, a citation-level search must still check active stereo, event-based
active vision, motion-boundary networks, and decision-focused computational
imaging.  The current search found classical ingredients but not their
combination as this learning object.

### 9. Adversarial ICLR audit

#### Strongest acceptance case

1. The paper defines a learnable object that ordinary frame models hide:
   sensor-action derivatives can be measures, not tensors in the same
   function space as observations.
2. The model follows the mathematics with explicit boundary mass and smooth
   action queries; it is compact rather than a larger grasp backbone.
3. The bias--variance result predicts a physical camera-motion scale and can
   be falsified across resolution and sensor noise.
4. The same object connects differentiable rendering, nonsmooth operator
   learning, active perception, and a real contact decision without copying a
   robotics pipeline.
5. Extra visual information attacks a documented failure regime: direct
   single-view target grasping degrades with occlusion, while active/multi-view
   grasping has shown that additional views can improve decisions:
   https://targo-benchmark.github.io/ and
   https://arxiv.org/abs/2003.06734.
6. Deployment remains a small number of local reductions plus a grasp-query
   head; no shape completion or whole-scene field is needed.

#### Strongest rejection case

1. **Rhetoric risk.**  A reviewer can call it “two RGB-D frames, an
   occlusion mask, and PointNet with BV terminology.”
2. **Classical-component risk.**  Boundary derivatives, accretion/deletion,
   weak convergence, kernel quadrature, and value of information are all
   established individually.
   Event cameras make the “motion produces sparse edge evidence” intuition
   especially non-new; the depth-lifted, scale-normalized decision object must
   supply the difference.
3. **Signal risk.**  Millimetre-scale motion may reveal too little target
   surface, especially when target and obstacle depths are close.
4. **Noise risk.**  Division by $\epsilon$ amplifies depth, segmentation,
   and calibration errors; the asymptotic scale may be larger than a cheap
   ordinary next view.
5. **Task-alignment risk.**  The revealed strip may establish depth ordering
   but not expose the two contact neighborhoods needed by a parallel jaw.
6. **Systems risk.**  A humanoid may spend more time planning and stabilizing
   the probe than it saves in perception.
7. **Benchmark risk.**  Synthetic exact visibility switches can overstate the
   advantage; real predicted masks and stationary-scene validation are
   mandatory.
8. **Generality risk.**  One grasp benchmark may be insufficient for the
   broad “singular sensor response” claim.  A synthetic decision-functional
   benchmark with moving discontinuities should directly test the
   resolution/sample-complexity theorem.

The official ICLR reviewer criterion remains significant new knowledge or
value supported rigorously, not merely a leaderboard gain:
https://iclr.cc/Conferences/2027/ReviewerGuidelines.  Candidate G is therefore
**promising and materially less blocked than D--F, but still conditional**.
It becomes paper-worthy only if the oracle-response pilot is positive and the
explicit measure representation beats raw two-view fusion under held-out
probe scales and resolutions.

### 10. Minimum defensible paper

A defensible paper would contribute:

1. a formulation of decision learning from measure-valued sensor responses,
   including the weak visibility decomposition and a non-identifiability
   boundary;
2. a probe-scale bias--variance theorem and uniform continuous-action query
   bound under calibrated but noisy RGB-D;
3. EdgeFlux, a compact boundary-measure/query layer that materially
   outperforms equal-capacity two-frame and fused-PCD models on held-out
   sensing scales;
4. a paired micro-motion occlusion benchmark with fixed grasp physics, plus
   real parallel-jaw small-lift evidence on a wrist-camera humanoid.

A possible title is:

> **EdgeFlux: Learning Grasp Decisions from Singular Visibility Response**

The strongest honest abstract claim, still contingent on the pilot, is:

> Small sensor motions do not change images smoothly: newly visible surface
> appears as boundary-supported mass.  We formulate embodied decision learning
> from this measure-valued response, derive the noise-dependent scale at which
> it is estimable, and introduce a compact action-query layer for target
> grasping behind an occluder.  Explicit visibility flux generalizes across
> camera baselines and RGB-D resolutions better than raw two-view fusion while
> improving physical parallel-jaw grasp selection without reconstructing
> hidden shape.

The last sentence is a target claim, not a conclusion.

## Post-promotion audit of Candidate G

### Cycle 91: data processing forbids an information-superiority claim

Let the full calibrated two-view input be

$$
 X=(O_0,O_1,x,v,\epsilon)
$$

and let $T(X)=(O_{\cap},\nu_{\epsilon,v})$ be the deterministic EdgeFlux
representation.  For a fixed candidate $g$, $T(X)$ cannot contain more
information about $Y$ than $X$:

$$
 I(Y;T(X)\mid g)\le I(Y;X\mid g).
$$

More generally, for any decision loss and unrestricted measurable predictors,
the Bayes risk satisfies

$$
 \mathcal R^*(X)\le \mathcal R^*(T(X)).
$$

This is not a minor caveat.  Candidate G may not claim that converting two
frames into a visibility measure creates information or asymptotically
dominates an unrestricted two-frame learner.  A sufficiently expressive raw
model can implement $T$ internally and retain any information that $T$
discards.

The defensible learnability claim is finite-resource and must name the resource
class.  For compute/parameter budget $B$, compare

$$
 \inf_{h\in\mathcal H^{\rm flux}_B}\mathcal R(h)-\mathcal R^*
 \quad\text{against}\quad
 \inf_{h\in\mathcal H^{\rm raw}_B}\mathcal R(h)-\mathcal R^*.
$$

The translated-step example can support a separation for declared linear,
band-limited, or fixed-resolution raw encoders, not for all neural networks.
The strongest theory route is therefore:

1. show that a $K$-query weak-measure estimator attains the minimax rate for
   a class of transverse moving-boundary functionals;
2. show a slower approximation rate for a clearly specified raw encoder class
   that does not align the discontinuity;
3. empirically extend the comparison to strong nonlinear two-frame
   transformers without pretending the theorem covers them.

The final model should retain the shared persistent local input and append the
flux sketch.  Deliberately discarding raw information to make a sufficiency
claim would weaken possible SOTA performance.  Candidate G survives this
audit only as a **structured finite-data/compute representation**, not as a
more informative sensor.

### Cycle 92: the parallax-resolution gate

The micro-motion premise can be tested analytically before rendering.  Consider
a pinhole cross-section with focal length $f_{\rm px}$, a foreground
occluder edge at depth $z_f$, target surface immediately behind it at depth
$z_b>z_f$, and lateral camera translation
$\epsilon_\perp$ normal to the projected edge.  A background point aligned
with the foreground edge before motion separates from it after motion by

$$
 \chi
 =
 f_{\rm px}|\epsilon_\perp|
 \left|\frac1{z_f}-\frac1{z_b}\right|
 \quad\text{pixels}.
$$

$\chi$ is the first-order width of the disoccluded target strip.  It is a
more meaningful experimental coordinate than millimetres of wrist motion:
the same baseline can be informative or invisible depending on focal length
and target--occluder depth separation.

For illustration only, with $f_{\rm px}=600$:

| $z_f$ | $z_b$ | $\chi$ at 5 mm | at 10 mm | at 20 mm | baseline for $\chi=3$ |
|---:|---:|---:|---:|---:|---:|
| 0.50 m | 0.55 m | 0.55 px | 1.09 px | 2.18 px | 27.5 mm |
| 0.50 m | 0.60 m | 1.00 px | 2.00 px | 4.00 px | 15.0 mm |
| 0.40 m | 0.60 m | 2.50 px | 5.00 px | 10.00 px | 6.0 mm |

This calculation corrects the earlier casual “2--20 mm” suggestion.  When
the obstacle is close to the target, even 20 mm may not produce three pixels
of new surface.  Conversely, a larger depth gap can make a 5--10 mm probe
useful.

If the target/occluder boundary has effective length $L_\Gamma$ pixels, the
number of switched target samples is approximately

$$
 N_{\rm birth}\asymp L_\Gamma\chi
$$

before mask and depth-confidence losses.  A necessary observability window is

$$
 \chi_{\min}\lesssim\chi
 \ll \ell_{\kappa,\rm px},
$$

where $\chi_{\min}$ is at least a few pixels under the actual segmentation
and registration noise, and $\ell_{\kappa,\rm px}$ is the projected scale
over which the grasp query is approximately smooth.  Below the window the
switch is unresolved; far above it the measurement is a useful additional
view but no longer a differential response.

The pilot must therefore stratify every result by $\chi$, not merely by
occlusion ratio or baseline.  A positive average produced only by large
$\chi$ values would support ordinary multi-view grasping, not EdgeFlux's
small-response thesis.

### Cycle 93: exact consequence of GCNGrasp-VP

The full paper
https://arxiv.org/html/2606.19091v1 already:

- predicts a per-point affordance field from a partial point cloud;
- selects one task-relevant camera adjustment without global reconstruction;
- reports 0.04 s view planning excluding 0.85 s preprocessing;
- improves real success after one view, for example from $4/28$ to
  $20/28$ on its brush task;
- identifies severe-occlusion affordance error as a remaining limitation.

Its task is semantic task-oriented grasping over four object--task pairs,
whereas Candidate G targets task-agnostic physical success for a target behind
one external obstacle.  The tasks and representations are different, but the
paper decisively occupies the broad claims “one task-aware extra view,”
“without reconstruction,” and “millisecond active grasp perception.”
ActiveNGF at NeurIPS 2024 additionally models a multi-view graspness field and
selects views from graspness inconsistency:
https://proceedings.neurips.cc/paper_files/paper/2024/file/4364fef031fdf7bfd9d1c9c56b287084-Paper-Conference.pdf.

Consequently, the required active baselines are not generic information gain
alone.  They include:

1. ActiveNGF or its strongest feasible adaptation;
2. GCNGrasp-VP-style affordance/view planning adapted to task-agnostic grasp
   success;
3. a fixed motion with registered local PCD fusion;
4. the same view selector and backbone with and without the flux layer.

If EdgeFlux gains come from choosing a better view rather than representing
the measured response, Candidate G is occupied by cycle 85.  If it does not
improve the success--latency or data--accuracy frontier over these methods,
there is no credible SOTA mechanism.

## Revised status after cycles 91--93

Candidate G remains alive, but its claim is now narrower and more rigorous:

> Under a resolvable parallax window, explicitly integrating the
> boundary-supported RGB-D birth measure against continuous grasp queries can
> provide a better finite-data/compute/resolution frontier than learning the
> same switching geometry implicitly from two rasterized views.

Three results are mandatory before calling it the final idea:

1. an oracle experiment showing that target-birth strips overlap
   decision-relevant grasp neighborhoods and improve top-one selection;
2. a representation result against strong two-view fusion at matched view,
   candidates, data, parameters, and compute;
3. a hardware check showing a nonempty safe motion interval whose parallax
   $\chi$ is resolvable but still local.

Without all three, the measure formulation is mathematically coherent but not
an objectively strong ICLR/SOTA proposal.

### Cycle 94: a 2026 two-view SOTA removes the local-fusion claim

The contemporaneous CVPR 2026 paper *A Cross-view Fusion Framework for Robust
6-DoF Grasp Pose Estimation* is closer than the active-view papers considered
above:

https://arxiv.org/html/2606.06878

It uses exactly the hardware opportunity available here: a calibrated auxiliary
observation from a wrist RGB-D camera, with the relative transform supplied by
forward kinematics.  It avoids full-scene reconstruction, groups both views in a
candidate-centred grasp cylinder, and performs cross-view attention only on that
local context.  Its ablation reports only $+0.32$ AP from direct point-cloud
pre-fusion but $+6.41$ AP from the complete post-fusion model; in selected
corner views it reports an average $+28.11$ AP.  Its real clutter-removal
success is $96\%$, versus $82\%$ for GSNet, with a total pipeline time of
$4.6$ s rather than $3.2$ s for the single-view baseline.  The paper samples
an auxiliary view rather than learning a next view, and the physical auxiliary
position is much farther away than the proposed micro-motion.  Those
differences leave an experimental niche, but not the broad methodological
claim.

The independent 2025 work on synthetic-aperture sensing by robot ``peering''
also establishes that a controlled side-to-side camera motion can remove
occlusion efficiently without ordinary feature-based 3-D reconstruction:

https://arxiv.org/abs/2511.16262

Therefore the following Candidate-G claims are **withdrawn**:

1. first efficient use of an auxiliary wrist view for occluded grasping;
2. first grasp-local cross-view fusion without full reconstruction;
3. first use of controlled camera motion as a compact anti-occlusion signal.

Candidate G survives only if the *renormalized singular response* gives a
specific learnability or compute advantage over the CVPR 2026 local-fusion
baseline.  A different name for two-frame differencing would fail novelty.

### Cycle 95: the stronger core is vanishing-support decision learning

The useful part of a differential view is not merely sparse; its mass vanishes
with probe amplitude.  After registration, write a candidate-local auxiliary
point distribution as

$$
 P_{h,\epsilon}
 = (1-\alpha_\epsilon)Q+\alpha_\epsilon P_h,
 \qquad
 \alpha_\epsilon=c\epsilon+o(\epsilon).
$$

$Q$ contains persistent geometry shared by two latent alternatives
$h\in\{0,1\}$.  Only the distribution $P_h$ of newly exposed target points
distinguishes the alternatives.  This produces a precise **rare-stratum
dilution** result for compute-limited fusion.

Suppose a local query network may read $K$ auxiliary tokens sampled without
the birth label.  By coupling and tensorization,

$$
 \mathrm{TV}
 \left(P_{0,\epsilon}^{\otimes K},
       P_{1,\epsilon}^{\otimes K}\right)
 \le
 K\alpha_\epsilon
 \mathrm{TV}(P_0,P_1).
$$

Le Cam's testing bound then gives

$$
 \inf_{\widehat h}
 \Pr(\widehat h\ne h)
 \ge
 \frac12
 \left[1-K\alpha_\epsilon
 \mathrm{TV}(P_0,P_1)\right].
$$

For fixed $K$, the raw local-token experiment becomes uninformative as
$\epsilon\to0$, even when the conditional birth distributions remain well
separated.  A birth-conditioned query instead reads from
$P_h^{\otimes K}$; its testing error is independent of $\epsilon$.  This is
an **equal learned-token-budget separation**, not an information-superiority
claim.  A full model allowed to process every raw pixel can recover the same
information, and an unrestricted model that analytically discovers the birth
stratum has effectively rediscovered the proposed representation.

At raster density $n$, a practical feature must retain both birth intensity
and conditional birth marks:

$$
 \widehat z_k(g;v)
 =
 \frac{1}{n\epsilon}
 \sum_{i=1}^{n}
 B_i\,\kappa_k(g^{-1}X_i),
 \qquad
 \mathrm{Var}(\widehat z_k)
 =O((n\epsilon)^{-1}).
$$

Here $B_i$ is the registered target-birth indicator.  Merely averaging the
birth points would discard the exposed-area rate $N_B/(n\epsilon)$; merely
pooling all auxiliary points would shrink their contribution by
$O(\epsilon)$.  The renormalized statistic preserves an $O(1)$ grasp-query
signal while requiring only a small number of learned birth tokens after an
$O(n)$ geometric warp and visibility comparison.

This changes the broad ML statement of Candidate G:

> Learn decisions from controlled observations whose useful evidence lies on
> a support of vanishing mass.  Use the known intervention and registration to
> expose that stratum, then learn from its finite measure derivative rather
> than asking a generic finite-token network to discover and amplify a rare
> component.

The mathematical ingredients -- sparse-mixture testing, importance sampling,
weak derivatives, and visibility boundary terms -- are individually classical.
The possible contribution is their new conjunction as a *learning experiment*:
an action-generated, mixed-dimensional, vanishing-support observation with a
continuous downstream decision query.  The theorem above is meaningful only
if the empirical model beats learned saliency and the 2026 cross-view method at
matched end-to-end cost.

### Cycle 96: normalization creates a new hard hardware gate

The $1/\epsilon$ normalization amplifies systematic errors as well as useful
birth mass.  If a registration, segmentation, or depth-noise process produces
a false-birth fraction $\beta_\epsilon$, its normalized contamination is

$$
 \frac{\beta_\epsilon}{\epsilon}.
$$

Consistency requires $\beta_\epsilon=o(\epsilon)$; bounded contamination at
least requires $\beta_\epsilon=O(\epsilon)$.  This is substantially stronger
than ordinary two-view fusion and is especially relevant because the laboratory
PCD is noisy.  No neural head can repair a systematic $1/\epsilon$ blow-up
without abandoning the derivative interpretation.

The cheap pilot must therefore estimate, on a static calibration object:

1. false-birth mass after kinematic registration and depth-uncertainty gating;
2. true target-birth mass behind the actual shelf obstacle;
3. their ratio across baseline, depth gap, edge orientation, and repeated
   frames;
4. the interval in which $n\epsilon$ is large enough for low variance while
   $\beta_\epsilon/\epsilon$ remains controlled.

The required end-to-end baselines are now exact:

1. the CVPR 2026 cross-view model or a faithful local-cylinder adaptation;
2. full raw two-view encoding;
3. equal-$K$ uniform auxiliary tokens;
4. equal-$K$ learned-saliency auxiliary tokens;
5. oracle birth tokens without $1/\epsilon$ normalization;
6. the proposed intensity-preserving normalized birth measure;
7. the same initial view, candidate generator, and fixed micro-motion for all
   representation comparisons.

### Status after cycles 94--96

Candidate G is still the strongest surviving direction, but it is not yet
promoted to the requested final framework.  Its defensible central hypothesis
is now narrower and stronger:

> In the resolvable small-parallax regime, grasp-relevant disocclusion is a
> rare component whose probability mass scales with camera displacement.
> Action-registered stratification plus measure normalization should dominate
> generic two-view fusion on the grasp-success/data/learned-token/latency
> frontier, while retaining the original view so no raw evidence is discarded.

There are now four independent kill conditions:

1. no nonempty safe interval satisfies both the parallax and false-birth gates;
2. oracle birth evidence does not change the best grasp often enough;
3. learned saliency matches the normalized measure at equal cost;
4. the full cross-view SOTA remains better on the success--latency frontier.

The token separation provides a real mechanism for efficient learnability and
does not violate data processing.  It is not indirect evidence of physical
SOTA by itself.  Until the four conditions are tested, claiming objective ICLR
novelty or likely SOTA would still be premature.

### Cycle 97: available workspace data cannot test the physical premise

A read-only search of the repository found object textures, meshes, and initial
pose arrays, but no recorded RGB-D/PCD sequence with calibrated camera
micro-motion.  Synthetic rendering can test the algebra but cannot establish
the decisive false-birth scaling under the laboratory's noisy sensor and
hand--eye calibration.  It would be misleading to replace this missing
measurement with a simulated positive result.

The minimum real-data collection is small:

1. one static target, one removable front obstacle, and an otherwise empty
   shelf;
2. target--obstacle depth gaps covering at least a near, middle, and far case;
3. lateral wrist translations normal to the main projected obstacle edge, for
   example $0,5,10,20,30,40$ mm when kinematically safe;
4. at least ten synchronized RGB-D frames per pose, plus ten repeated captures
   with no commanded motion;
5. camera intrinsics, depth scale/confidence if exposed by the sensor,
   timestamped camera-to-base transforms, and target/obstacle masks or a
   high-baseline oracle view.

Two controls are mandatory: obstacle plus shelf with the target removed, which
measures false births, and the complete target without the obstacle, which
checks registration error on an ordinary visible surface.  For each edge
segment and baseline, report predicted and measured $\chi$, true and false
birth counts, $N_{\rm true}/(N_{\rm false}+1)$, and the fitted log--log slopes
of birth mass versus $\epsilon$.  The candidate survives only if at least one
safe interval has:

- measured $\chi$ above the sensor-specific resolution threshold;
- enough target births for stable grasp-kernel estimates;
- false-birth contamination that does not grow after $1/\epsilon$
  normalization;
- repeatability across captures and at least two target--obstacle depth gaps.

This collection is the next rational action.  More literature search or a
larger network cannot decide this physical gate.

## Tenth independent search pass: non-visual probes and modified acquisition physics

This pass deliberately avoided further variants of EdgeFlux.  It asked whether
the newly allowed signals or utilities create a genuinely different learning
problem.  Robotics papers were used only to mark occupied territory after each
mathematical proposal was formed.

### Cycle 98: first-contact saltation as a pre-grasp correction signal

**Proposal.**  A rigid contact is a hybrid event rather than a smooth tactile
sample.  If a guard $h(x)=0$ is crossed at first contact and the state resets
through $R$, the local event response is described by the saltation matrix

$$
 S
 =
 D R+
 \frac{(f^+-D R f^-)\nabla h^\top}
      {\nabla h^\top f^-}.
$$

The first-contact time, wrist impulse, and jaw encoder discontinuity could be
treated as a compact singular observation.  A supervised model could infer a
small pose correction before high-force closure, without reconstructing the
object or learning a long policy.

**Rejection.**  The probe is available only after the gripper has moved one
chosen candidate into contact.  It therefore cannot improve the original
one-shot ranking over all candidates; it defines a recovery controller for a
possibly bad committed candidate.  With one symmetric actuator, aggregate
current and terminal width do not even identify which jaw or pad region touched
first.  Independent tactile arrays or fingers fix that ambiguity, but tactile
contact detection, grasp correction, and tactile regrasping are already broad
research areas.  The saltation matrix itself is a state-linearization object,
not directly measured from one event, so estimating it would require repeated
local impacts or a contact simulator.  This route is a compact hybrid-control
module, not a new pre-contact grasp-learning problem.

Relevant occupied territory includes tactile grasp correction and contact
reconstruction:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC10111055/
- https://escholarship.org/uc/item/2z77k26t

### Cycle 99: motor-current compliance spectroscopy

**Proposal.**  Apply a low-force closure dither and learn a quotient of the
force--displacement curve

$$
 F_g(\delta)
 =
 k_{\rm drive}i(\delta)-F_{\rm friction}(\dot\delta)
$$

that is invariant to closing speed and actuator gain.  Its first-contact
location, local slope, and hysteresis area could parameterize a small
grasp-specific mechanical response instead of a long causal failure vector.

**Rejection.**  A 2026 RA-L gripper already explicitly reconfigures between
sensing and grasping modes, estimates fingertip force from motor current, and
reports about $90\%$ accuracy for size/stiffness-based fruit-ripeness
classification:

https://doi.org/10.1109/LRA.2026.3662656

Thus current-based size/stiffness sensing is occupied hardware capability.  For
a standard coupled parallel gripper, the trace still aggregates drive friction,
transmission compliance, two contacts, and object motion.  Disentangling those
terms needs calibration or per-jaw sensing.  More importantly, as in cycle 98,
the measurement is obtained only after committing to the candidate.  A learned
speed-invariant curve would be useful system identification, not an ICLR-level
replacement for grasp selection.

### Cycle 100: active photometric contact-normal jets

**Proposal.**  Keep the camera fixed and vary a small set of known illumination
directions.  A photometric response

$$
 I_j(u)\simeq \rho(u)\,[\ell_j^\top n(u)]_+
$$

can estimate a visible surface normal and curvature jet even where RGB-D depth
is noisy.  Query only the two candidate jaw neighborhoods and fuse their
photometric jets with the coarse depth points.

**Rejection.**  This is a particularly clean sensor augmentation, but it does
not reveal the target surface hidden behind the frontal obstacle.  Its benefit
is limited to denoising already visible contact normals.  More decisively,
photometric stereo was used to determine legal configurations for a
parallel-jaw gripper more than two decades ago, and recent robotic manipulation
work again uses controlled illumination for grasp-relevant shape estimation:

- https://people.csail.mit.edu/bkph/papers/Determining_Grasp_Configuration.pdf
- https://arxiv.org/abs/2304.11824

A modern query network would improve robustness and material coverage, but the
scientific core would remain photometric stereo followed by grasp detection.

### Cycle 101: rolling-shutter or coded-exposure micro-baselines

**Proposal.**  Intentionally move the wrist during one exposure.  If row $r$
is captured at pose $x(t_0+\tau r)$, a single rolling-shutter image contains a
small family of calibrated viewpoints.  A row-aware decoder might recover
disocclusion evidence with no second-frame latency.

**Rejection.**  The physical scale fails before the learning problem becomes
interesting.  A $10$ ms readout and a gentle $0.05$ m/s wrist motion produce
only $0.5$ mm of synthetic baseline; the parallax audit in cycle 92 often
requires $10$--$30$ mm.  Producing $20$ mm during the same readout requires
$2$ m/s motion near the shelf.  RGB and depth channels may also use different
shutter mechanisms and synchronization, so the supposed RGB-D experiment is
camera-specific.  Rolling-shutter geometry, coded exposure, motion deblurring,
and occlusion-aware rectification are already established computational
photography problems:

- https://openaccess.thecvf.com/content_cvpr_2018/papers/Vasu_Occlusion-Aware_Rolling_Shutter_CVPR_2018_paper.pdf
- https://ics.uci.edu/~majumder/COMPPC/papers/fluttershutter.pdf

This direction purchases latency with unsafe motion and hardware assumptions;
it is not a credible laboratory or broad-ML candidate.

### Cycle 102: reopening Candidate D through Blackwell sensor order

**Proposal.**  Replace Candidate D's rejected robust utility ellipsoid by an
order over sensing experiments.  A sensing action $a$ induces a channel
$E_a:Z\mapsto O_a$.  If

$$
 E_b=K\circ E_a
$$

for a stochastic garbling kernel $K$, then $a$ Blackwell-dominates $b$
for *every* downstream utility, including every parallel-jaw grasp utility.
Approximate Le Cam deficiency,

$$
 \delta(E_b,E_a)
 =
 \inf_K\sup_z
 \mathrm{TV}\!\left(E_b(\cdot\mid z),
 K E_a(\cdot\mid z)\right),
$$

could yield a task-family-universal view selector learned from paired simulated
scenes.

**Rejection.**  The order is deliberately universal and therefore too coarse:
most views reveal different, incomparable surfaces and neither is a garbling of
the other.  Estimating the deficiency requires paired access to the latent
scene and rich conditional generators, while the laboratory needs only one
specific grasp decision.  Weakening the criterion to the given grasp-utility
family returns exactly to goal-oriented experimental design and cycle 85's
occupied decision-focused next-best-view problem.  Blackwell ordering and
channel deficiency are established decision theory, so the remaining
contribution would be an expensive active-grasp instantiation.

General boundary:

- https://arxiv.org/abs/1701.07602
- https://arxiv.org/abs/2005.06673

### Cycle 103: support-assisted closure as a learned sweeping process

**Proposal.**  Treat the shelf or front obstacle as a useful passive contact
rather than only a collision.  Under a slowly moving gripper constraint
$C_g(t)$, quasistatic object motion can be represented by a Moreau-type
sweeping inclusion

$$
 -\dot x(t)\in N_{C_g(t)}(x(t)).
$$

A model could learn the terminal capture basin or a low-dimensional correction
to this analytic sweeping process, enabling the parallel jaws to funnel an
otherwise occluded target before the small lift.

**Rejection.**  This is precisely extrinsic dexterity and environment-assisted
occluded grasping.  It changes the task from grasp prediction to a contact-rich
push/rotate/grasp sequence and therefore violates the requested exclusion of
whole-cycle feasibility and capture-basin formulations.  The robotics territory
is also directly occupied, including a CoRL system with $78\%$ real success
and a 2025 method specifically for grasp-constraining walls:

- https://proceedings.mlr.press/v205/zhou23a.html
- https://arxiv.org/abs/2507.14721

A non-RL sweeping-process implementation would replace their controller, but
not create a new scientific task.

### Cycle 104: path signatures of the natural wrist approach

**Proposal.**  Use RGB-D frames already produced while the humanoid approaches
the shelf.  After registration, regard the observation as a measure-valued path
$M_{0:T}$ and encode a truncated rough-path signature.  Signatures are
invariant to monotone time reparameterization, so the same model could consume
different approach speeds and frame rates without a recurrent policy.

**Rejection.**  The input now includes the approach trajectory that the brief
explicitly asked not to evaluate as part of the grasp cycle.  More technically,
the signature is an established sequence representation; it neither decides
which views are informative nor solves registration.  With registered frames
it is a sophisticated multi-view fusion operator, and without them it mixes
camera motion with geometry.  The approach data could be free in one deployment
but not under the stated observation contract, so this is neither an
independent subproblem nor a defensible novelty claim.

## Result of the tenth pass

No Candidate H from contact sensing, acquisition hardware, universal sensor
ordering, or environmental contact passes all constraints.  The failures are
structurally different:

1. contact/current probes arrive after committing to a grasp;
2. illumination improves visible normals but not external occlusion;
3. rolling-shutter baselines are too small at safe wrist speeds;
4. universal experiment orders are too coarse and task-specific versions are
   ordinary active view planning;
5. support-assisted and approach-path methods expand into the forbidden grasp
   cycle.

This negative result strengthens Candidate G for a specific reason, not merely
by elimination.  A registered camera micro-motion is the only examined
additional action that is simultaneously non-contact, pre-decision, compact,
and directly changes the hidden geometry observable behind the obstacle.
EdgeFlux's remaining novelty must still come from vanishing-support
renormalization and its efficient-learning separation, not from the existence
of that action itself.

### Cycle 105: direct rare-stratum lower bound for grasp regret

Cycle 95 stated a binary testing separation.  It becomes decision-relevant
without an informal appeal to classification.  Let two equally likely latent
worlds $H\in\{0,1\}$ share all persistent local geometry but have different
birth distributions $P_0,P_1$.  Restrict the action reduction to two grasps
$\{g_0,g_1\}$, where $g_h$ is optimal in world $h$, and assume the
wrong-grasp gap

$$
 Q_h(g_h)-Q_h(g_{1-h})\ge \Delta>0.
$$

An unstratified $K$-token auxiliary observation has law

$$
 \mathbb P_{h,\epsilon}^{K}
 =
 \left[(1-\alpha_\epsilon)Q+
       \alpha_\epsilon P_h\right]^{\otimes K},
 \qquad
 \alpha_\epsilon=c\epsilon+o(\epsilon).
$$

Any selector $\widehat g$ from these tokens induces a test of $H$.
Therefore

$$
\begin{aligned}
 \inf_{\widehat g}
 \mathbb E\,\mathrm{Reg}(\widehat g)
 &\ge
 \frac{\Delta}{2}
 \left[
  1-\mathrm{TV}
  \left(\mathbb P_{0,\epsilon}^{K},
        \mathbb P_{1,\epsilon}^{K}\right)
 \right] \\
 &\ge
 \frac{\Delta}{2}
 \left[
  1-K\alpha_\epsilon
  \mathrm{TV}(P_0,P_1)
 \right].
\end{aligned}
$$

If $\mathrm{TV}(P_0,P_1)=\rho>0$, making this lower bound smaller than
$\Delta/4$ requires

$$
 K=\Omega((\rho\epsilon)^{-1}).
$$

By contrast, after an oracle or sufficiently accurate action-registered birth
stratification, $K$ learned tokens have law $P_h^{\otimes K}$, whose
testing difficulty does not deteriorate with $\epsilon$.  This is the direct
efficient-grasp statement: a generic local sampler needs a token budget that
grows as the informative support shrinks, while a known-stratum query does not.

Three qualifications prevent misuse of the theorem:

1. The sensor must still acquire enough pixels/points to contain those births:
   $n\alpha_\epsilon\gtrsim K$.  EdgeFlux reduces *learned representation*
   cost, not the optical resolution required by physics.
2. The bound does not apply to an unrestricted full-view network or to a
   saliency front-end allowed to inspect all pixels at uncharged cost.  Such a
   model can discover the birth stratum itself.  End-to-end FLOPs, memory, and
   latency must therefore be measured, not inferred from the theorem.
3. A noisy birth label replaces $P_h$ by a contaminated conditional law.
   Cycle 96's $\beta_\epsilon/\epsilon$ gate is necessary for the apparent
   constant-$K$ advantage to survive.

These limitations are strengths of the formulation: they isolate the exact
resource being improved and avoid an impossible claim of gaining information
through deterministic preprocessing.

## Recommended framework after 105 cycles

### Name and broad question

**EdgeFlux: learning grasp decisions from vanishing-support sensor
responses.**

The broad question is:

> How should a learner represent a controlled observation change when the
> decision-relevant evidence has vanishing spatial mass but a nonzero weak
> derivative?

Parallel-jaw grasping behind an external obstacle is the primary testbed, not
the definition of the general problem.  Related regimes can occur whenever a
small sensing action reveals a lower-dimensional stratum, but the paper should
make no broad empirical claim beyond the evaluated grasp setting.

### Exact laboratory contract

1. The scene is static and contains one target, shelf geometry, and at most one
   frontal obstacle; dense clutter is excluded.
2. Initial input $O_0$ is wrist RGB-D, optionally with a target mask supplied
   identically to every method.
3. Before final grasp selection, the wrist executes one collision-checked,
   pre-approved lateral translation $a=(v,\epsilon)$ and records $O_1$.
   This is a sensing action, not RL and not a learned manipulation policy.
4. The output is one parallel-jaw pose $g$.  The label is closure plus a
   standardized tiny lift, not feasibility of the full approach-to-carry cycle.
5. No scene SDF, full mesh, NeRF, or long vector of causal failure variables is
   predicted.

The main representation experiment should use a fixed safe dither.  A learned
view selector is optional and cannot appear in the novelty claim because
decision-focused active grasp viewing is occupied.

### Mathematical observation object

Let $\mathsf M_x$ be the registered visible target surface measure at camera
pose $x$.  Let $\mathcal B_{\epsilon,v}$ be points visible at
$x+\epsilon v$ that have no visibility-preserving antecedent at $x$.
Their positive birth measure is

$$
 \nu_{\epsilon,v}^{+}
 =
 \frac{1}{\epsilon}
 \mathsf M_{x+\epsilon v}
 \lfloor_{\mathcal B_{\epsilon,v}}.
$$

For piecewise-smooth scenes away from visibility topology changes,

$$
 \nu_{\epsilon,v}^{+}
 \overset{*}{\rightharpoonup}
 \nu_v^{+},
$$

where $\nu_v^{+}$ is supported on the target-side visibility fold.  An
ordinary $L^2$ image derivative is the wrong state space because the strip
narrows while its amplitude grows; weak convergence against continuous test
functions remains finite.

For compact candidate-local kernels $\kappa_k$, define

$$
 z_k(g;v)
 =
 \int
 \kappa_k(g^{-1}X)\,d\nu_v^{+}(X).
$$

These are not full-scene variables.  They ask how fast newly visible target
surface enters a small collection of jaw, pad, and approach neighborhoods of a
queried grasp.

### Efficient learnable model

The finite estimator is

$$
 \widehat z_k(g;v)
 =
 \frac{1}{n\epsilon}
 \sum_{i=1}^{n}
 w_i B_i\,
 \kappa_k(g^{-1}X_i),
$$

where $B_i$ is the registered birth indicator and $w_i$ is a sensor
confidence/area weight.  The query model is

$$
 \widehat q_\theta(g)
 =
 h_\theta\!\left(
   e_\theta(O_0,g),\;
   r_\theta(O_0,O_1,g),\;
   \widehat z_1,\ldots,\widehat z_m,\;
   \epsilon,\widehat\Sigma_{\rm reg}
 \right).
$$

$e_\theta$ is any strong single-view grasp-local encoder.
$r_\theta$ receives a matched small budget of raw registered local tokens so
that EdgeFlux augments rather than artificially replaces useful evidence.
$\widehat\Sigma_{\rm reg}$ exposes calibration uncertainty to the gate.  The
new layer has $O(n)$ deterministic registration/visibility work and
$O(mK)$ candidate-query work; it does not run a second dense learned
backbone unless an ablation shows that one is necessary.

Candidate generation must depend only on $O_0$ and be shared by every
representation baseline.  Otherwise better candidate recall would be
incorrectly attributed to EdgeFlux.

### Training

Synthetic paired renders provide exact visibility provenance and arbitrary
queried grasp labels.  Real static pairs provide sensor-noise and registration
adaptation.  A minimal objective is

$$
 \mathcal L
 =
 \mathrm{BCE}(Y,\widehat q_\theta(g))
 +\lambda_{\rm birth}\mathcal L_{\rm birth}
 +\lambda_{\rm weak}
  \sum_{\epsilon\in\mathcal E_{\rm local}}
  \|\widehat z_\epsilon-z^{\rm oracle}_v\|_1.
$$

The weak-response loss is applied only inside the empirically verified local
window.  Wider views expose genuinely different finite surfaces and must not be
forced to have the same normalized feature.  Training samples should be
balanced in parallax coordinate

$$
 \chi=f_{\rm px}|\epsilon_\perp|
 \left|z_f^{-1}-z_b^{-1}\right|,
$$

not merely in millimetres or global occlusion ratio.

### Theory package required for an ICLR paper

1. **Weak visibility derivative:** existence and boundary support under an
   explicit piecewise-smooth camera model.
2. **Finite-resolution estimator:** bias
   $O(\epsilon)$, variance $O((n\epsilon)^{-1})$, and the resulting local
   scale trade-off.
3. **Rare-stratum grasp-regret bound:** cycle 105's
   $K=\Omega(1/\epsilon)$ separation for unstratified local tokens.
4. **Uniform decision transfer:** if every candidate score is within
   $\delta$, maximizing the predicted score incurs at most $2\delta$
   top-one regret.
5. **Noise impossibility:** false-birth mass not scaling as
   $O(\epsilon)$ destroys the normalized limit.

Only items 1 and 3 are plausible core theory contributions.  Items 2, 4, and 5
make the claims operationally honest.

### Exact novelty claim

The paper must **not** claim first active grasp viewing, first two-view grasp
fusion, first visibility-boundary derivative, first importance sampling, or
first weak-measure network.  All are false or too broad.

The defensible claim is:

> First formulation and evaluation of grasp learning from an
> action-generated, vanishing-support RGB-D stratum, represented by a
> displacement-normalized visibility-birth measure and queried directly by
> continuous parallel-jaw actions, with a token-budget grasp-regret separation.

The closest components remain separated across fields:

- visibility boundary terms in differentiable rendering:
  https://cseweb.ucsd.edu/~tzli/diffrt/
- dynamic occlusion/accretion--deletion as a strong depth cue:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4521857/
- rare-event importance sampling reducing learning variance:
  https://arxiv.org/abs/2008.06334
- robot peering as synthetic-aperture anti-occlusion sensing:
  https://arxiv.org/abs/2511.16262
- local two-view grasp fusion:
  https://arxiv.org/html/2606.06878

No searched work was found that uses the observed displacement-normalized
target birth measure as a continuous grasp-query representation or proves the
vanishing-support decision separation.  This is evidence for novelty, not a
guarantee that an unindexed concurrent manuscript does not exist.

### SOTA mechanism and evaluation claim

Absolute multi-view accuracy with unrestricted camera motion is the wrong and
probably unwinnable claim: a far auxiliary view contains more information.
The plausible SOTA claim is the **success--motion--latency--learned-compute
frontier** under a fixed small sensing budget.

The evidence is indirect but nontrivial:

1. The 2026 cross-view grasp paper reports that one auxiliary view and
   grasp-local post-fusion improve AP by $6.41$ over its baseline, by
   $28.11$ on selected corner views, and real success from $82\%$ to
   $96\%$.  This supports the value of localized cross-view evidence.
2. Robot peering demonstrates that controlled lateral motion can efficiently
   overcome partial occlusion without ordinary feature-based reconstruction.
3. Dynamic-occlusion studies show that accretion/deletion contributes powerful
   depth-order evidence.
4. The regret theorem shows why explicitly stratifying the newly visible
   evidence can beat fixed-token unstratified local fusion.

The benchmark must compare at the same initial view, probe, candidates, labels,
and hardware:

1. single-view TARGO/GSNet-style detectors;
2. direct registered PCD fusion;
3. full raw two-view encoding;
4. the CVPR 2026 cross-view local-cylinder model;
5. ActiveNGF and a GCNGrasp-VP-style adaptation where feasible;
6. equal-$K$ uniform and learned-saliency auxiliary tokens;
7. EdgeFlux with and without normalization, raw residual tokens, confidence,
   and weak-response supervision.

Report physical top-one success and AP against $\chi$, target visibility,
motion distance, end-to-end latency, learned FLOPs, peak memory, calibration
error, depth noise, and unseen object geometry.  EdgeFlux supports a SOTA claim
only if it is Pareto-undominated and improves physical success by a practically
meaningful margin over the strongest equal-motion baseline.

### Final research verdict

After 105 analyzed cycles, EdgeFlux is the only direction that currently has
all of:

1. a new general learning question rather than only a new grasp head;
2. a compact model compatible with noisy wrist RGB-D and parallel jaws;
3. no RL, VLA, full-scene SDF, or full-cycle feasibility estimator;
4. a mathematical mechanism for efficient learnability;
5. close prior work that indirectly supports the physical opportunity while
   not already containing the proposed learning object.

It is therefore the **recommended framework**, conditional on the cycle-97
hardware gate and the oracle/matched-compute pilots.  Failure of any one of the
four kill conditions in cycles 94--96 rejects the project rather than merely
calling for a larger model.

### Cycle 106: final exact-core collision search and ICLR red-team

The final search used the core conjunction rather than broad active-grasp
keywords:

- "visibility birth measure" robot grasp;
- "displacement-normalized" disocclusion learning;
- "vanishing-support" "active perception";
- "boundary-supported" grasp measure camera motion.

As of 25 August 2026, it found no paper matching the proposed observed
birth-measure target, displacement normalization, continuous grasp query, and
rare-stratum regret result.  The search did return unrelated uses of
displacement normalization and established rolling-shutter/active-perception
work.  Together with cycles 90--105, this supports the narrow novelty claim.
It does not prove absence from unindexed submissions or differently worded
concurrent work.

The adversarial reviewer summaries and required answers are:

1. **“This is engineered frame differencing.”**
   The paper fails if its contribution is a mask concatenated to RGB-D.  The
   answer must be the weak-measure limit, the $1/\epsilon$ intensity
   preservation, the grasp-regret separation, and empirical scale transfer.
2. **“A sufficiently large two-view network can learn it.”**
   Correct in principle.  The claim is a finite learned-compute/data frontier,
   not Bayes information superiority.  Full raw fusion and learned saliency are
   mandatory baselines.
3. **“The theorem assumes the answer through an oracle birth mask.”**
   The theorem isolates the value of stratum side information.  The actual
   paper must measure the cost and error of obtaining that side information
   from known camera action and depth registration.  Oracle-only results are
   insufficient.
4. **“Micro-motion reveals too little geometry.”**
   The $\chi$ and $n\epsilon$ gates make this falsifiable.  A result outside
   the local window supports ordinary two-view grasping, not EdgeFlux.
5. **“Noise explodes under normalization.”**
   Correct unless $\beta_\epsilon=O(\epsilon)$.  The static-camera and
   target-removed controls in cycle 97 must be reported before model results.
6. **“The active-view problem is already solved.”**
   The selector is held fixed in the main experiment.  The claim concerns the
   representation of the observed response, with ActiveNGF, GCNGrasp-VP, and
   Cross-view Grasp as baselines.
7. **“The result depends on perfect target masks.”**
   Report both oracle-mask and detector-mask settings and attribute mask error
   separately.  Target segmentation is neither hidden inside the method nor
   claimed as a contribution.

Under the official ICLR 2027 criteria, SOTA is not formally required, but the
paper must create significant new knowledge or value:

https://iclr.cc/Conferences/2027/ReviewerGuidelines

EdgeFlux can meet that bar only as a general vanishing-support learning problem
with a nontrivial physical demonstration.  Without the weak-limit theorem,
regret separation, real noise scaling, and matched cross-view results, the
likely and accurate verdict is “interesting robotics feature engineering.”
With all four, the work has a defensible ICLR identity: it connects a new
statistical experiment, an efficient continuous-action representation, and a
physical regime where the distinction matters.

## Independent extension: externally occluded partial-PCD grasping

**Date:** 25 August 2026

**Full report:** [ICLR_GRASP_OCCLSION_IDEA_RESEARCH_1IDEA.md](ICLR_GRASP_OCCLSION_IDEA_RESEARCH_1IDEA.md)

This extension answers a new request without using Candidates A--G or cycles
1--106 above as idea sources.  It studies a single parallel-jaw grasp from one
wrist RGB-D view when exactly one foreground object additionally censors the
target.  RL, VLA, dense clutter, full-shape reconstruction, and whole-cycle
reach-to-carry feasibility remain excluded.

The literature audit first rejects external-occlusion benchmarking,
occlusion augmentation, deterministic/probabilistic shape completion, direct
partial-PCD diffusion, and uncertainty scoring as main contributions.  It
also rejects a newly developed grasp-conditioned hidden-contact certificate
after finding the contemporaneous PartialBiGrasp paper, which already queries
hidden local occupancy in gripper-aligned control regions.

The surviving conditional proposal is **CapGrasp**, a conditional capacity
operator on gripper-induced compact sets.  Rather than outputting pointwise
occupancy or a completed target, it predicts a coherent joint distribution of
hit/miss events for contact slabs, nested distance probes, and collision
volumes.  A non-negative tensor-train probability circuit induces a valid
finite capacity by construction.  Inclusion--exclusion then computes
correlated events such as “outer shell clear and both contacts present.”  The
central separation is that identical pointwise occupancy marginals can imply
different bilateral-contact probabilities; a joint capacity retains this
information without decoding a global shape.

The proposal is not yet a SOTA claim.  It is killed unless a mesh-derived hit
signature of at most 40 bits is within 2--3 success points of a full-mesh
oracle, tensor rank greater than one materially beats independent occupancy on
ambiguous hidden-shape pairs, and capacity supervision beats a parameter-
matched direct BCE critic under scarce simulator labels.  The full report
contains the mathematical formulation, closest-work matrix, architecture,
paired one-occluder shelf benchmark, SOTA thresholds, and ICLR red-team audit.
