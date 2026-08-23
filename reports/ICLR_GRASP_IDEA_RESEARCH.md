# Research log: an ICLR-level grasping problem

Date: 2026-08-24

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

\[
\omega(o,g)=\sup_{z,z'\in R^{-1}(o)}|Q(z,g)-Q(z',g)|,
\qquad
\|P_{\ker DR(z)}\nabla_z Q(z,g)\|.
\]

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

Proposal: for latent scene \(z\), grasp \(g\), and feasible grasp-wrench body
\(W(z,g)\subset\mathbb R^6\), learn

\[
K(o,g)=\bigcap_{z:R(z)\simeq o} W(z,g).
\]

A compact implementation would predict the gauge of this convex body. The exact
identity

\[
p_{K(o,g)}(w)=\sup_{z:R(z)\simeq o}p_{W(z,g)}(w)
\]

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

Proposal: for a convex object, the support function \(h(u)\), its spherical
gradient, and its Hessian at \(u\) and \(-u\) determine antipodal contact
locations and local curvatures. Predict these compact support jets from a
partial observation instead of reconstructing an SDF.

Rejection: the representation is exact only for convex bodies. Useful
parallel-jaw grasps on non-convex household objects frequently exploit local
width minima; extending the representation to localized cross-sections produces
an action-space aperture field close to existing dense grasp maps. The elegant
convex geometry would therefore purchase novelty by excluding central cases.

### 14. Persistent topology of the feasible-grasp set

Proposal: learn the persistent homology of superlevel sets of a grasp field on
\(SE(3)/\mathbb Z_2\), preserving distinct stable grasp components and returning
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

Represent an RGB-D observation as a set \(E\) of occupied/free-space ray
constraints. Define refinement by reverse inclusion of the compatible latent
worlds:

\[
E_1\preceq E_2 \quad\Longleftrightarrow\quad
\mathcal C(E_2)\subseteq\mathcal C(E_1).
\]

For a grasp \(g\), the sharp utility endpoints

\[
L(E,g)=\inf_{z\in\mathcal C(E)}Q(z,g),\qquad
U(E,g)=\sup_{z\in\mathcal C(E)}Q(z,g)
\]

obey an information-order law:

\[
E_1\preceq E_2\Rightarrow
L(E_1,g)\le L(E_2,g)\le U(E_2,g)\le U(E_1,g).
\]

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
3. With an overly broad admissible hidden-shape class, \(L\) is identically zero
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
pairwise kernel between left and right contact evidence. Factor a rank-\(r\)
kernel so that all contact-pair scores can be computed in \(O(Nr)\) rather than
\(O(N^2)\).

Rejection: this is an efficient architecture, not a new learning problem.
PhyGrasp already predicts per-point pair embeddings and an explicit grasp-pair
match classifier; VCPD and earlier geometric methods also generate and classify
contact pairs. Low-rank evaluation alone is incremental.

Sources:

- https://arxiv.org/html/2402.16836v1
- https://proceedings.mlr.press/v205/cai23a.html
- https://haojhuang.github.io/edge_grasp_page/

### 18. Learned admissible branch-and-bound over \(SE(3)\)

Proposal: learn a supremum oracle \(U(o,B)\) for cells \(B\) of continuous grasp
space and use best-first subdivision until the returned grasp is
\(\epsilon\)-optimal.

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

Proposal: for a convex body \(K\), learn only its difference body
\(D=K+(-K)\). Its support function is the directional width
\(h_D(n)=h_K(n)+h_K(-n)\), and a support point of \(D\) is the chord between the
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

Let \(R:\mathcal Z\to\mathcal O\) be the RGB-D sensor map and let
\(Q:\mathcal Z\to L^2(\mathcal G)\) map a latent scene to its closure-and-small-
lift utility function. On a differentiable visibility stratum, define the
**sensor-null action operator**

\[
A_z = DQ_z\big|_{\ker DR_z}:\ker DR_z\longrightarrow L^2(\mathcal G).
\]

Its image contains exactly the first-order changes in the entire grasp field that
the sensor cannot observe. Its singular values form the **sensor-null action
spectrum**, and the numerical rank at tolerance \(\delta\) is the action
ambiguity dimension. This is not the dimension of hidden shape and not the
uncertainty of one selected grasp.

The finite, nonlinear counterpart for observation \(o\) is

\[
\mathcal U(o)=\{Q(z,\cdot):z\in\mathcal C_\varepsilon(o)\},
\]

with centered Kolmogorov width

\[
d_r(\mathcal U(o))=
\inf_{\mu,\,\dim V\le r}\;
\sup_{q\in\mathcal U(o)}\inf_{v\in V}
\|q-\mu-v\|.
\]

The central empirical claim is not assumed for free: measure whether these widths
decay rapidly for realistic occluded-object families, and reject the project if
they do not.

### Why smoothing can make the hypothesis true

The operational target should be success probability under the robot's actual
small pose perturbation \(\xi\),

\[
Q_z^\sigma(g)=\mathbb E_\xi[Y_z(g\circ\xi)],
\]

not an infinitely sharp binary contact indicator. On a compact action manifold,
isotropic perturbation is heat-kernel smoothing. If
\(Y_z=\sum_j c_{zj}e_j\) in Laplace--Beltrami eigenfunctions with eigenvalues
\(\lambda_j\), then

\[
Q_z^\sigma=\sum_j e^{-\sigma^2\lambda_j/2}c_{zj}e_j.
\]

Consequently, uniformly \(L^2\)-bounded raw success fields admit the spectral
truncation bound

\[
\|Q_z^\sigma-P_rQ_z^\sigma\|_2
\le e^{-\sigma^2\lambda_{r+1}/2}\|Y_z\|_2.
\]

Weyl growth \(\lambda_r\asymp r^{2/d}\) gives exponentially decreasing error in
\(r^{2/d}\). This does not prove that \(r\) is tiny in practice, but it supplies a
falsifiable mechanism: finite pads and execution noise remove action frequencies
that no robot can exploit reliably.

### Efficient model: a sensor-null action ellipsoid

The model should follow the operator rather than merely attach a low-rank head
to a grasp network.  A radius-\(c\) ball of infinitesimal sensor-null
perturbations is mapped by \(A_z\) to an ellipsoid in action-function space.  Its
truncated SVD is

\[
A_z h\simeq \sum_{j=1}^r s_j u_j\langle v_j,h\rangle .
\]

This motivates an observation-conditioned **sensor-null action ellipsoid
(SNAE)**, not independent uncertainty intervals.  For a candidate set
\(G_o=\{g_m\}_{m=1}^M\) generated solely from the shared observation, a point/ray
encoder and gripper-query decoder output

\[
\mu_m=\mu_\theta(o,g_m),\qquad
U_{mj}=u_{\theta,j}(o,g_m),\qquad d_j\ge 0,\qquad \rho\ge0.
\]

A weighted thin QR or polar layer makes \(U^TWU=I_r\).  The radii \(d_j\) are
kept separate from the directions, fixing the scale ambiguity.  The predicted
functional set is

\[
\widehat{\mathcal U}_r(o)=
\left\{\mu+U\operatorname{diag}(d)a+e:
\|a\|_2\le1,\ \|e\|_\infty\le\rho\right\}.
\]

The coefficient \(a\) indexes an observation-preserving hidden perturbation and
is never inferred at deployment.  The robust lower score is the support
function of the ellipsoid and is therefore closed-form:

\[
\widehat L_m
=\inf_{q\in\widehat{\mathcal U}_r(o)}q_m
=\mu_m-
\sqrt{\sum_{j=1}^r d_j^2U_{mj}^2}-\rho .
\]

After clipping to \([0,1]\), select \(\arg\max_m\widehat L_m\), or abstain when
its value is below a declared threshold.  Inference is \(O(Mr+Mr^2)\), returns
only \(r+2\) scalars per queried grasp, and never predicts a mesh, voxel grid,
or scene SDF.

For training group \(b\), let \(q_{b,s}\in[0,1]^M\) be the perturbation-smoothed
utility vector of hidden twin \(s\).  The exact finite-sample target is a
minimum-width rank-\(r\) enclosing ellipsoid with an \(L^\infty\) residual:

\[
\begin{aligned}
\min_{\mu,U,d,\rho,\{a_s\}}\quad &
 \rho+\lambda_w\frac1M\sum_{m=1}^M
 \sqrt{\sum_{j=1}^r d_j^2U_{mj}^2}
 +\lambda_v\sum_{j=1}^r\log(d_j+\epsilon)\\
\text{s.t.}\quad &U^TWU=I_r,\quad \|a_s\|_2\le1,\\
&\|q_{b,s}-\mu-U\operatorname{diag}(d)a_s\|_\infty\le\rho
\quad\forall s.
\end{aligned}
\]

The first two terms directly minimize the uniform error and actionwise robust
half-width used by the downstream selector.  The log-volume term is only a
tightness regularizer; an ablation must show that it does not manufacture
overconfidence.  In implementation, projected inner updates for \(a_s\), a
log-sum-exp approximation of the largest residual, and alternating
network/coefficient steps make the program differentiable and minibatchable.
Hard residual mining restores the worst twin/candidate pairs lost by softening.

Sparse utility labels \(\Omega_{b,s}\subset[M]\) can be used in the inner fit,
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

\[
z^*=\arg\max_{z:\,d(R(z),o)\le\varepsilon}
\|(I-P_\Phi)(Q(z,G_o)-\mu)\|.
\]

Add this counterexample to the twin group and repeat. This is a greedy reduced-
basis construction in task space, not shape completion at inference. In practice
the inner search can use procedural hidden surfaces first and differentiable
render/contact surrogates only as a secondary stress test.

### Immediate theory targets

1. **Sensor-twin lower bound.** For two equally likely latent scenes with
   observation laws \(P_0,P_1\), suppose choosing the other scene's optimal
   action costs at least \(\Delta\). Every observation-only (even randomized)
   selector then has expected regret at least

   \[
   \frac{\Delta}{2}\big(1-\operatorname{TV}(P_0,P_1)\big),
   \]

   by reduction to binary testing and Le Cam's bound. Exact twins have
   \(P_0=P_1\), so more model capacity cannot remove the gap.
2. **Factorization criterion.** If fibers are connected and
   \(DQ_g\ker DR=0\) everywhere, then \(Q_g=\bar Q_g\circ R\); zero action rank
   is exactly local-to-global action identifiability.
3. **Local nonlinear width bound.** Let \(\psi:B_{\mathcal H}(c)\to
   \mathcal C(o)\) be a chart of one smooth sensor fiber, set
   \(F=Q^\sigma\circ\psi\), and assume \(\|D^2F\|\le K\).  If \(V_r\) is
   spanned by the first \(r\) left singular functions of \(A=DF(0)\), Taylor's
   theorem and SVD optimality give

   \[
   \sup_{\|h\|\le c}
   \operatorname{dist}\big(F(h)-F(0),V_r\big)
   \le c\,s_{r+1}(A)+\tfrac12Kc^2.
   \]

   The spectrum therefore controls local nonlinear ambiguity up to an explicit
   curvature term; it is not only a visualization of a Jacobian.
4. **Spectrum-to-decision identity.** On a finite shared candidate set, let
   \(e_m\) evaluate candidate \(m\).  For the linearized fiber ball
   \(q(h)=q_0+Ah,\ \|h\|\le c\), the exact robust utility is

   \[
   \inf_{\|h\|\le c}e_m^Tq(h)
   =q_{0,m}-c\|A^Te_m\|_2
   =q_{0,m}-c\sqrt{\sum_j s_j^2u_j(m)^2}.
   \]

   Retaining \(r\) modes and subtracting
   \(cs_{r+1}+Kc^2/2\) gives a conservative lower score under the preceding
   curvature assumption.  This identity is the reason for the row-norm in SNAE;
   the ellipsoid is not a decorative uncertainty head.
5. **Heat-compressibility bound.** If the raw sensor-null derivative is a bounded
   operator \(B_z\) and operational utility applies heat smoothing \(H_t\), then
   \(A_z=H_tB_z\) is compact and

   \[
   s_j(A_z)\le \|B_z\|e^{-t\lambda_j}.
   \]

   Thus measured execution precision supplies an upper bound on effective
   action-space rank; the claim should be tested across noise scales.  A uniform
   version is available for action selection: with \(K_t(g,g)\) the heat-kernel
   diagonal,

   \[
   \|(I-P_r)H_tf\|_\infty
   \le \sup_g K_t(g,g)^{1/2}
   e^{-t\lambda_{r+1}/2}\|f\|_2.
   \]

   This separates the \(L^2\) spectral diagnostic from the uniform error needed
   by a maximization over actions.
6. **Robust-selection stability.** If the learned and true functional sets are
   within \(\delta\) in Hausdorff \(L^\infty\) distance, their lower envelopes
   differ by at most \(\delta\), and maximizing the learned lower envelope has at
   most \(2\delta\) maximin regret.

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
2. Require rank \(r\le8\) to explain at least 90% of centered energy and to keep
   the 90th-percentile per-candidate error below 0.05 in at least 70% of held-out
   visible-shell groups.  Also solve the uniform enclosing-factor objective: a
   favorable Frobenius SVD alone does not establish the \(L^\infty\) accuracy
   needed by action maximization.
3. Compare oracle maximin selection on the full matrix with selection after
   rank-\(r\) truncation. A loss larger than 5 percentage points kills the
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

Let (z\in\mathcal Z) be a complete latent object/scene, (R(z)) its RGB-D rendering, and (Q(z,g)\in[0,1]) the closure-and-small-load utility of grasp (g). For an observation (o), its sensor fiber is

\[
\mathcal C_\varepsilon(o)=\{z:d(R(z),o)\leq\varepsilon\}.
\]

Instead of reconstructing (z), learn the observation-conditioned random utility function

\[
U_o:g\mapsto Q(Z,g),\qquad Z\sim p(\cdot\mid o),
\]

or, without a trusted probability prior, its support over (Z\in\mathcal C_\varepsilon(o)).

This is a goal-oriented inverse problem whose output is a distribution/set of scalar fields on grasp space, not a distribution of 3D shapes.

### Why marginal grasp uncertainty is insufficient

For candidates (G=\{g_1,\ldots,g_M\}), the relevant pushforward is the joint utility vector

\[
\nu_{o,G}=\big(Q(Z,g_1),\ldots,Q(Z,g_M)\big)_\#p(Z\mid o).
\]

Independent confidence intervals for each grasp discard correlation across candidates. The hidden completion that is bad for one grasp can be good for another. A decision-aware representation must preserve entire hypothetical utility profiles.

### Compact model

Predict \(K\) latent utility critics and optional weights:

\[
f_{\theta,k}(o,g)\in[0,1],\quad \pi_{\theta,k}(o)\geq0,
\quad\sum_k\pi_{\theta,k}=1.
\]

For any candidate set, row (k),

\[
\big(f_{\theta,k}(o,g_1),\ldots,f_{\theta,k}(o,g_M)\big),
\]

is one hypothetical utility world. A point-cloud encoder supplies observation tokens; a gripper-centric query encoder supplies a feature for (g); (K) shared scenario tokens produce utilities consistently across every queried grasp. Inference is (O(KM)) and does not generate a mesh, voxel grid, or SDF.

Train the weighted version with a proper multivariate score such as the energy score on utility vectors. Train a prior-free support version with a soft Hausdorff/set loss.

### Occlusion-twin supervision

Ordinary datasets provide one latent scene per observation and cannot identify counterfactual hidden-shape ambiguity. Construct grouped scenes

\[
\{z_{b,1},\ldots,z_{b,S}\},\qquad R(z_{b,s})=o_b,
\]

by deforming geometry only inside the camera/obstacle shadow volume while fixing visible surfaces, silhouette, texture, camera, and depth-noise realization. Generate the candidate set from (o_b), so it is identical for every twin. Simulate all twin/candidate pairs to obtain an (S\times M) utility matrix.

A decisive real experiment uses 3D-printed twin families with the same camera-facing shell and statistically indistinguishable wrist RGB-D, but different hidden backside/contact geometry.

### Directed regret geometry

For the support formulation define

\[
d_o(g,h)=\left[\sup_{z\in\mathcal C_\varepsilon(o)}
\big(Q(z,h)-Q(z,g)\big)\right]_+.
\]

This is a directed pseudometric: (d_o(g,g)=0) and

\[
d_o(g,k)\leq d_o(g,h)+d_o(h,k).
\]

It measures the worst supported disadvantage of choosing (g) instead of (h). Worst-case oracle regret is the directed eccentricity

\[
R_o(g)=\max_h d_o(g,h),
\]

and minimax-regret selection is a directed 1-center. The critic model gives a property-preserving approximation

\[
\widehat d_o(g,h)=
\max_k\big[f_{\theta,k}(o,h)-f_{\theta,k}(o,g)\big]_+.
\]

This form is related to quasimetric embeddings, but here the quasimetric is derived from observation-fiber decision regret rather than imposed as a generic metric-learning device.

Strict minimax regret can sacrifice absolute worst-case success. It should therefore be reported as one decision rule, alongside posterior expected utility, CVaR, and maximin utility; a safety threshold on lower utility can precede regret minimization.

### Candidate theory

1. **Non-identifiability lower bound.** If two latent scenes have the same observation and different optimal grasps separated by a utility gap, every deterministic point predictor incurs positive regret on at least one scene.
2. **Decision sufficiency.** For any one-shot loss depending on the latent scene only through (Q(z,\cdot)), the pushforward utility process is sufficient; the full posterior over geometry contains no additional decision-relevant information.
3. **Support stability.** If the predicted and true utility supports are within Hausdorff distance (delta) in (\ell_\infty), every pairwise regret distance is within (2\delta), yielding a bounded excess robust-selection regret.
4. **Distributional stability.** Wasserstein error of the learned joint utility law bounds error in expectations of Lipschitz decision losses.
5. **Finite candidate approximation.** For perturbation-smoothed grasp utility that is Lipschitz on the gripper-symmetry quotient of (SE(3)), an (\epsilon)-net of grasps induces controlled decision regret.

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
