# Research log: grasp learning from externally occluded partial point clouds

Started: 2026-08-25

## Executive verdict

The surviving proposal is **CapGrasp: a conditional capacity operator over
gripper-induced sets**.  Given censored RGB-D evidence and a parallel-jaw pose,
it predicts a coherent joint law of whole-region hit/miss events for contact
slabs and collision volumes.  It neither reconstructs a target nor emits
pointwise occupancy marginals.  A non-negative tensor-train probability
circuit makes the finite set function valid by construction and allows exact
bilateral-contact/collision probabilities by inclusion--exclusion.

The core novelty is not "query geometry near a grasp"; PartialBiGrasp already
does that with deterministic local occupancy.  The proposed new object is the
*conditional capacity of action-induced regions*, retaining spatial
dependence without a global shape posterior.  The literature search found no
exact neural or grasping counterpart as of the date above, but novelty is not
certified by keyword search.

This is a conditional go, not a SOTA claim.  First run an oracle pilot using
mesh-derived hit signatures and no visual network.  Stop if a signature of at
most 40 bits cannot predict small-lift success within 2--3 points of a full-
mesh oracle, or if correlated rank \(\chi>1\) fails to beat both independent
occupancy and direct BCE on deliberately ambiguous hidden-shape pairs.  Only
after those gates pass is implementation of the evidence encoder justified.

## Scope

- One 6-DoF parallel-jaw grasp is selected from a single wrist RGB-D
  observation.
- A target is already partially observed because of self-occlusion and may
  lose an additional contiguous visible region behind a foreground object.
- The target class may be known as a soft detector posterior, but the method
  must also admit a class-agnostic shape prior.
- Dense clutter, RL, VLA, and reconstruction of a complete target mesh/PCD at
  inference are not the intended solution.
- The learned target is terminal grasp quality (closure and a small lift), not
  reach-to-carry feasibility.  Observed-scene collision filtering is held
  identical across methods so that hidden-target reasoning is not confused
  with a better motion planner.
- A contribution must be more than occlusion augmentation, a conditional
  diffusion backbone, or an uncertainty scalar appended to a completion
  pipeline.

## The empirical premise is already verified, but the strongest version is not

The proposed qualitative degradation is real and now directly benchmarked.
TARGO constructs paired scenes containing the same target at the same camera
pose, first with foreground objects and then with the occluders removed.  It
balances 1,000 test scenes in every visual-occlusion bin from 0 to 0.9:

https://arxiv.org/html/2407.06168

Its main findings are stronger than the original conjecture:

1. VGN, GIGA/GIGA-HR, EdgeGraspNet, and VN-EdgeGraspNet all deteriorate as
   external occlusion increases.  Edge-based methods lose as much as 30
   percentage points; VGN/GIGA variants lose about 20 points.
2. With raw scene and partial target input, the TARGO architecture loses about
   20 points after the 0.4--0.5 occlusion range.
3. Completing both the target and target scene reduces the synthetic decline
   to roughly 7 points, but physical TARGO-Net still falls from 80.0% in easy
   scenes to 66.7% in hard scenes.  The authors attribute the larger real drop
   to noise patterns that challenge completion.
4. GIGA falls from 70% to 30% over the same physical easy-to-hard comparison.
5. TARGO explicitly concludes that shape completion helps synthetically but
   does not generalize reliably to the real world.

Thus “external occlusion has not been tested” is no longer a valid gap.
Neither is “train a model on occluded scenes”: TARGO's single-scene
augmentation already improves baselines by at least 5 points and TARGO-Net by
about 10.

There is also direct evidence that ordinary single-view partiality already
causes hidden-geometry failures.  GraspLDM reports that many partial-PCD
failures are collisions with object portions absent from the observation and
that models mistake incomplete surfaces for graspable edges:

https://arxiv.org/html/2312.11243

Grasp Diffusion Network directly learns
\(p(g\mid\text{partial PCD})\), and GraspGen uses a Diffusion Transformer plus
an on-generator discriminator.  Therefore a new diffusion sampler, by itself,
does not address the scientific gap:

- https://arxiv.org/html/2412.08398
- https://arxiv.org/html/2507.13097

The remaining question is narrower:

> Can a learner use the *mechanism and geometry of missingness* to marginalize
> hidden target shape only as needed by a queried parallel-jaw grasp, without
> decoding a complete object and without treating the cropped target point set
> as the whole observation?

## Occupied solution families

### Direct partial-PCD grasp distributions

S4G already calls direct regression from a single partial scene cloud
“amodal”; Contact-GraspNet roots grasps at observed contact points; GraspLDM,
GDN, and GraspGen learn generative distributions conditioned on partial point
clouds.  Their architectures differ, but they all treat the observed point
set/scene encoding as an ordinary condition:

- https://proceedings.mlr.press/v100/qin20a.html
- https://arxiv.org/abs/2103.14127
- https://arxiv.org/html/2312.11243
- https://arxiv.org/html/2412.08398
- https://arxiv.org/html/2507.13097

### Deterministic or task-focused completion

TARGO-Net completes the target with AdaPoinTr, fuses it with the scene, and
predicts an implicit grasp field.  GIGA jointly learns scene geometry and
affordance.  ShellGrasp-Net predicts camera-ray entry/exit depths and a grasp
map.  TOSC completes potential contact regions rather than the full shape.
CenterGrasp and PCF-Grasp also transfer learned shape priors into grasping.

- https://arxiv.org/html/2407.06168
- https://www.roboticsproceedings.org/rss17/p024.pdf
- https://arxiv.org/abs/2109.06837
- https://ojs.aaai.org/index.php/AAAI/article/view/38053
- https://centergrasp.cs.uni-freiburg.de/
- https://arxiv.org/abs/2504.16320

### Completion distributions and uncertainty-aware planning

Monte-Carlo dropout completions followed by joint grasp evaluation already
outperform a single completed shape in simulation and physical experiments.
Recent work predicts uncertain completion regions or penalizes completion
uncertainty in grasp ranking:

- https://arxiv.org/abs/1903.00645
- https://arxiv.org/abs/2308.00377
- https://arxiv.org/abs/2504.16183

These methods correctly acknowledge ambiguity but still instantiate geometry
at inference, often repeatedly.

### Uncertainty without explicit completion

FFHFlow is especially close to any proposal that merely says “represent shape
uncertainty in a latent and generate grasps.”  It learns

\[
 p_\theta(g\mid x)
 =
 \int p_\theta(g\mid x,z)p_\theta(z\mid x)\,dz
\]

from partial point clouds, uses normalizing-flow likelihoods as view/object
uncertainty, and combines them with a discriminative evaluator:

https://arxiv.org/html/2407.15161

Consequently, a latent flow, probabilistic grasp generator, likelihood-based
confidence, or “avoid invisible-side grasps” rule is occupied even if adapted
from dexterous hands to parallel jaws.

## Sequentially rejected cycles

### Cycle 1: demonstrate the external-occlusion degradation

Rejected as the main contribution.  TARGO already supplies exactly paired
occluded/unoccluded observations and success curves.  Reproduction on the
laboratory shelf remains a necessary diagnostic, not a novel problem.

### Cycle 2: physically realistic point deletion augmentation

Rejected.  Rendering foreground objects is better than iid point dropout, but
TARGO already trains on paired single/cluttered scenes and explicitly measures
the gain.  More masks or a stronger encoder would be a dataset recipe.

### Cycle 3: conditional diffusion on externally occluded clouds

Rejected.  GDN, GraspLDM, and GraspGen occupy conditional diffusion/latent
diffusion from partial point clouds; TARGO supplies the missing training
distribution.  The combination is obvious and lacks a new learning object.

### Cycle 4: one category-conditioned full completion, then grasping

Rejected by both prior art and the stated failure mode.  TARGO already uses a
strong target completion module, and its real result remains noise-sensitive.
A YOLO class label can sharpen a shape prior but cannot remove intra-class
multimodality or guarantee the hidden instance geometry.

### Cycle 5: a distribution of completed shapes and robust grasping

Rejected as occupied.  Robust planning over Monte-Carlo shape completions and
uncertain-region-aware grasp ranking already show physical gains.  Runtime
scales with completion samples, and posterior misspecification is merely moved
into the completion generator.

### Cycle 6: a latent grasp flow with an uncertainty score

Rejected by FFHFlow.  It already argues against slow completion, represents
partial-shape uncertainty with input-conditioned normalizing flows, and uses
view likelihood in grasp ranking.

### Cycle 7: only allow bilateral contacts on observed target points

Rejected.  This is a useful safety baseline but deliberately gives up valid
grasps as visibility falls.  It cannot explain how a class/shape distribution
should be used and cannot dominate TARGO at heavy occlusion.

### Cycle 8: predict a complete joint law of utilities over all grasps

Rejected for this project.  It overlaps an already analyzed observation-fiber
utility direction in the earlier research log, requires an expensive
function-valued posterior, and its mean reduces to an ordinary Bayes grasp
score for a one-action decision.

### Cycle 9: predict only local contact-region geometry

Rejected as the core claim.  It is more efficient than full completion but is
too close to ShellGrasp-Net's entry/exit shell and TOSC's task-oriented contact
completion.  A new coordinate system alone would not separate the work.

### Cycle 10: make the point encoder invariant to additional occlusion

Rejected on information grounds.  Removing visible surface genuinely changes
the posterior over successful grasps.  Exact invariance would discard useful
evidence, while augmentation-based approximate invariance returns to cycle 2.

## Candidate A: geometric censoring rather than missing points

### Observation-level gap

A cropped target point cloud records only positive surface hits.  It discards
why every absent point is absent.  In an RGB-D image there are at least three
different events:

1. a ray hits visible target surface;
2. a ray is known free up to its measured depth;
3. a foreground surface terminates the measurement while the target may
   continue behind it.

The third event is a censored measurement, not an empty target sample.  Let
\(T_u\) be the first target-surface depth on camera ray \(u\), and \(C_u\) the
first foreground-occluder depth.  The sensor observes

\[
 Y_u=\min(T_u,C_u),\qquad
 \delta_u=\mathbf 1[T_u\le C_u].
\]

Target-only PCD methods retain \(Y_u\) only when \(\delta_u=1\).  The proposed
observation retains \((u,Y_u,\delta_u)\), sensor uncertainty, the visible
target points, and only the foreground rays inside a detector/target proposal
region.  It is substantially smaller than a scene SDF and does not assert a
hidden surface.

This distinguishes two questions that point dropout conflates:

- self-occlusion induced by the target's own first hit;
- external right-censoring induced by a separately observed nearer surface.

### Statistical foundation

Under conditionally independent censoring given visible context \(X\), the
proper ray likelihood is

\[
 \ell(\theta)
 =
 -\delta\log f_\theta(Y\mid X)
 -(1-\delta)\log S_\theta(Y\mid X),
\]

where \(S_\theta(t\mid X)=\Pr_\theta(T>t\mid X)\).  The right-censored
log-likelihood is a proper score; deleting censored rays instead estimates the
biased conditional law \(T\mid T\le C,X\):

https://proceedings.mlr.press/v151/rindt22a.html

Real depth noise can be represented as interval censoring.  For a hit known
only within \([a,b]\), its contribution is
\(-\log(F_\theta(b)-F_\theta(a))\), rather than supervision at one noisy 3-D
point.

If occluder placement depends on the hidden target after conditioning on
\(X\), ordinary independent-censor likelihood is inconsistent.  The method
must either condition on sufficient occluder/placement variables or explicitly
model informative censoring.  This is a kill condition, not a detail.

### Why the representation can contain strictly more decision information

The censored-ray observation \(E\) deterministically yields the cropped target
cloud \(P\), so \(E\) Blackwell-dominates \(P\).  The dominance is strict
whenever two latent shape/occluder cases yield the same target points but
different censoring evidence and require different grasps.

For two equiprobable cases \(h\in\{0,1\}\), two correct grasps \(g_h\), and
wrong-grasp loss \(\Delta\), every target-PCD-only selector has regret
\(\Delta/2\) if \(P_0=P_1\).  An enriched selector has Bayes regret

\[
 \frac{\Delta}{2}
 \left(1-\operatorname{TV}(\mathcal L(E\mid h=0),
                            \mathcal L(E\mid h=1))\right).
\]

This is an information separation from target-only PCD, not from TARGO's full
scene input.  Against TARGO, the argument must instead be statistical and
computational: expose the known censoring operator and avoid decoding
irrelevant complete geometry.

### Provisional broad question

> How should a geometric learner make continuous decisions when its positive
> surface events are mixed with ray-wise right- and interval-censored events,
> and the latent event distribution is needed only through an
> action-conditioned functional?

This question is broader than grasping, but its first test should remain the
single-view parallel-jaw setting.

### Provisional architecture: CensorGrasp

1. **Typed evidence tokenizer.**  Visible target hits, free-space intervals,
   and foreground censor thresholds are separate token types.  Each ray token
   carries camera origin/direction, measured depth interval, segmentation
   confidence, and optional soft class posterior.
2. **Shared ray-set encoder.**  A sparse point/ray transformer encodes the
   target proposal once.  No dense 3-D completion is decoded.
3. **Continuous gripper-frame query.**  For a queried
   \(g\in SE(3)\), a fixed small collection of pad/closing-volume witness
   queries is transformed into camera coordinates and cross-attends to nearby
   evidence tokens.
4. **Monotone marked-hitting head.**  It predicts cumulative hazards for
   target hits and distributions of surface-normal marks at the witness
   queries.  Integrated positive intensities enforce monotonic CDFs.  A shared
   low-rank latent couples opposing pad events; independent per-ray hazards
   would hallucinate mutually incompatible contacts.
5. **Grasp functional.**  A differentiable parallel-jaw contact functional
   integrates the joint hitting distribution into expected terminal grasp
   quality.  An identical observed-scene collision filter is applied to every
   method.

The initial implementation should rerank candidates from a fixed strong
sampler so candidate recall is controlled.  Only after representation gains
are established should the same query field guide an \(SE(3)\) flow/diffusion
generator.  Generator novelty must not be claimed.

### Training signals

- Synthetic complete meshes generate exact local witness events, grasp labels,
  and arbitrarily many physically rendered foreground-censor patterns.
- Paired observations of the same shape behind different occluders supervise
  censoring consistency without requiring identical predictions.
- Real RGB-D can adapt the ray process through censored/interval likelihood,
  even where the hidden surface is never labeled.
- Class-known training conditions on a soft category posterior.  Class-unknown
  training marginalizes a generic mixture; hard YOLO labels are an ablation,
  not an assumption.

### Status

Candidate A does **not** survive as the final method in this form.  The input
representation remains useful, but the global/per-ray hazard decoder has two
load-bearing problems:

1. ShellGrasp-Net already predicts camera-ray entry/exit geometry and grasp
   maps, and RayOcc already uses a non-normalized multi-event intensity along
   rays for occlusion-aware occupancy.  Replacing their deterministic output
   by a survival head is an incremental representational change.
2. A small fixed collection of independently predicted witness hits is not a
   sufficient statistic for lift success.  Without a joint geometric
   consistency construction, it can assert opposing contacts that no rigid
   object realizes; with such a construction, it becomes local probabilistic
   shape completion.

The typed hit/free/censor observation is retained.  The decoded object-ray
hazard is rejected.

Relevant collision checks:

- https://arxiv.org/html/2109.06837
- https://arxiv.org/html/2607.17660
- https://arxiv.org/abs/2107.13421

## Cycle 11: full-shape teacher to partial-view score distillation

### Initial attraction

Let a full-state teacher expose a diffusion score
\(s_\phi(g_t,S,t)\) for successful grasps on complete target state \(S\), and
train a student on censored evidence \(E\) using

\[
  \mathbb E\|s_\theta(g_t,E,t)-s_\phi(g_t,S,t)\|^2.
\]

The population minimizer is the conditional projection of the teacher score.
By the Fisher/tower identity it is the score of a grasp distribution in which
the hidden full state is marginalized.  This is attractive because full shape
is privileged training information and no shape is decoded at inference.

### Rejection as a core contribution

This is now directly covered by Latent Target Score Matching (LTSM).  LTSM
shows that a marginal score is a conditional expectation of an available
joint score, obtains a low-variance target at low diffusion noise, and mixes it
with DSM at higher noise.  A grasp-conditioned application could be useful,
but would not constitute a new ICLR-level learning principle:

https://arxiv.org/html/2602.07189

Ordinary conditional DSM also has the same population minimizer when paired
partial observations and positive grasps are available.  The teacher changes
finite-sample variance and optimization, not the target distribution.  Any
later method may use LTSM as a training component, but must not claim the
projection identity as novelty.

## Cycle 12: success-volume or partition-corrected grasp diffusion

### A real bias

Many grasp generators balance objects and then draw a fixed number of positive
grasps per object.  GDN, for example, renders 32 objects and uses exactly 32
grasps from each object in every minibatch:

https://arxiv.org/html/2412.08398

For full state \(S\), let \(q(g,S)\in[0,1]\) be terminal success under a fixed
base proposal \(\mu\),

\[
 Z_S=\int q(g,S)\,\mu(dg),\qquad
 \pi_S(g)=\frac{q(g,S)\mu(g)}{Z_S}.
\]

Equal-object positive sampling learns
\(\mathbb E[\pi_S(g)\mid E]\), whereas the modes appropriate for Bayes
selection are those of

\[
 \bar\nu_E(g)=\mu(g)\mathbb E[q(g,S)\mid E]
              =\mathbb E[Z_S\pi_S(g)\mid E].
\]

Thus an occluded observation compatible with several shapes can overweight a
low-\(Z_S\) shape.  The correction is a \(Z_S\)-weighted conditional DSM or
teacher-score loss.  At noise time \(t\),

\[
 \arg\min_s\mathbb E\!left[
 Z_S\|s(g_t,E,t)-s_{S,t}(g_t)\|^2\right]
 =\nabla_{g_t}\log\mathbb E[Z_S\pi_{S,t}(g_t)\mid E].
\]

GraspGen's data construction makes this testable: it uniformly proposes 2,000
poses per object and retains binary simulator labels, so \(Z_S\) can be
estimated as an acceptance fraction:

https://arxiv.org/html/2507.13097

### Rejection as the final core

Although the bias is worth an ablation and perhaps a secondary contribution,
the remedy reduces to case-selection/importance weighting.  Reward-weighted
diffusion and importance-weighted conditional DSM already provide the general
machinery.  A reviewer can fairly summarize the grasp contribution as “do not
erase the per-object positive rate when balancing the dataset.”  That is not
enough for the requested paper, even with a new name.

## Cycle 13: conditional kernel/Fourier embedding of the successful-grasp set

For each full state, one can embed the unnormalized successful-grasp measure
\(q(g,S)\mu(dg)\) in an RKHS and regress its conditional mean from \(E\).  This
commutes exactly with hidden-shape marginalization and avoids partition bias.
A finite random-feature or harmonic expansion gives cheap evaluation of many
grasps.

Rejected as the main direction.  Conditional mean embeddings are mature, and
OrbitGrasp already predicts Fourier coefficients of continuous grasp-quality
functions over orientation.  In six action dimensions the rank/bandwidth
needed for sharp contact modes removes the claimed efficiency; a learned
low-rank decoder reduces to an implicit grasp field such as those already used
by GIGA/TARGO.

- https://arxiv.org/abs/1605.09522
- https://proceedings.mlr.press/v270/hu25b.html

## Candidate B: censored ray-to-action certificate operator

### Replace shape by an action-induced quotient

The useful part of Candidate A is not “predict hidden surface points.”  It is
that a parallel-jaw action interrogates hidden geometry through a small,
structured family of lines in the *gripper* frame.  For a queried pose
\(g\in SE(3)\), define \(\Xi_g(S)\) to contain:

- bidirectional extreme target depths along a sparse pad-covering set of jaw
  closing lines;
- local normal/friction-cone marks at those extremes;
- first-hit clearances on a sparse cover of the palm/finger approach shell;
- existence and sensor-noise marks, rather than a deterministic point.

Two complete shapes are equivalent for query \(g\) whenever they have the same
\(\Xi_g\).  The learner should estimate

\[
  p(\Xi_g\mid E),
\]

not \(p(S\mid E)\).  A differentiable mechanics head then computes

\[
  Q(g,E)=
  \mathbb E_{\Xi_g\mid E}
  [\tilde q(g,\Xi_g)],
\]

where \(\tilde q\) is an antipodal-contact, jaw-width, and local-clearance
functional plus a learned residual calibrated to terminal simulator labels.
The output dimension scales with the queried gripper, not with the shelf
volume or target surface area.

This is an action-induced quotient of latent geometry.  It avoids the two
extremes of (i) treating missing target points as no evidence and (ii) decoding
a reusable complete object.

### Cross-ray rather than object reconstruction

Camera evidence and gripper evidence live on two different ray families.  The
new learning object is a conditional operator

\[
  \mathcal T_\theta:
  \{\text{typed, censored camera rays}\}
  \times \{\text{gripper query rays at }g\}
  \longmapsto p_\theta(\Xi_g\mid E).
\]

This suggests a sparse ray-space architecture rather than a voxel/SDF model.
Pluecker coordinates provide an \(SE(3)\)-compatible representation of both
ray sets; relative line angle, shortest distance, and directed depth ordering
are invariant cross-attention features.  General \(SE(3)\)-equivariant
convolution and attention on ray space already have a mathematical basis:

https://proceedings.neurips.cc/paper_files/paper/2023/hash/075b2875e2b671ddd74aeec0ac9f0357-Abstract-Conference.html

### Provisional architecture: CRAQ (Censored Ray-to-Action Queries)

1. **Typed sensor rays.**  Within a target proposal, tokenize visible target
   hits, foreground termination thresholds, known-free intervals, and invalid
   depth.  Use a soft amodal-association probability so that an occluder inside
   a YOLO box is not asserted to cover the target with probability one.
2. **Shared evidence encoder.**  Sparse ray-space self-attention operates once
   per RGB-D frame.  It carries RGB/depth uncertainty, modal target mask, and
   optional detector logits.  Class dropout trains one model for class-known
   and class-unknown use.
3. **Gripper query constructor.**  Each candidate pose creates a fixed
   \(h\)-net of closing lines over the pads and approach rays over the rigid
   gripper shell.  All are transformed by \(g\) and represented in the same
   Pluecker ray space.
4. **Cross-ray transport blocks.**  Query rays cross-attend to sensor rays
   using equivariant line geometry.  A low-rank shared latent couples all query
   lines so mutually incompatible contacts are penalized.
5. **Ordered support distribution.**  The head predicts left/right extreme
   depths, existence, normals, and calibrated intervals.  It does not predict
   dense occupancy.  Multiple-object scene collision is handled by the same
   observed-scene filter for every baseline.
6. **Certificate head.**  An analytic layer computes jaw-width, antipodality,
   and clearance margins.  A small residual head maps certificate features to
   small-lift success, and is trained with simulator/real binary outcomes.

### Why finite queries can be defensible

The exact success of an arbitrary compliant grasp is not finite-ray
sufficient.  The paper must state a margin-qualified approximation instead.
For targets with reach at least \(\rho\), Lipschitz surface normals, and a
rigid parallel-jaw gripper, let the pad and approach shell be covered at
spacing \(h\).  If all extreme-depth/normal estimates have error at most
\(\epsilon\), then geometric clearance and antipodal margins change by at most
\(C_1h+C_2\epsilon\).  Therefore any grasp whose true margins exceed that
quantity retains its feasibility label.  Boundary grasps remain intrinsically
uncertain and should be down-ranked, not falsely certified.

This is a proposition to prove formally, not a current theorem.  It makes the
required assumptions and approximation failure visible.

### What is and is not novel

Not novel separately:

- point/ray transformers;
- entry/exit or ray occupancy prediction;
- task/contact-region completion;
- a grasp scorer or an \(SE(3)\) sampler;
- synthetic foreground augmentation.

Potentially novel as one learning object:

> posterior transport from explicitly censored sensor rays to a sparse,
> action-indexed contact certificate on gripper rays, with a
> margin-controlled quotient of hidden geometry and no reusable shape output.

The closest robotics methods still decode a camera-centric shell
(ShellGrasp), a complete target (TARGO), or task-level contact regions before
the grasp is queried (TOSC).  RayOcc predicts global semantic occupancy and is
not action-conditioned.  No exact match was found in searches for
action-conditioned/gripper-ray shape inference, but absence from keyword
search is not yet proof of novelty.

### Current status

Candidate B survives deeper than the earlier candidates, but remains
provisional.  It must pass three kill tests:

1. **Oracle sufficiency:** ground-truth \(\Xi_g(S)\) must predict simulator
   small-lift success nearly as well as the complete mesh.  If the gap exceeds
   roughly 2--3 percentage points after controlling collision filtering, the
   quotient is too lossy.
2. **Representation value:** typed censor rays plus ray geometry must beat a
   parameter-matched full-scene/target cross-attention encoder on identical
   candidates.  Otherwise this is merely an expensive coordinate change.
3. **Posterior value:** a probabilistic certificate must beat deterministic
   query completion and a direct BCE scorer, especially on ambiguous same-
   visible-fragment/different-hidden-shape pairs.  Otherwise the distribution
   is unnecessary.

## General-ML formulation: query-goal Bayesian inversion under censoring

The most useful non-robotics analogy is goal-oriented Bayesian inversion.  In
large inverse problems, the latent parameter field is often only an
intermediate object; the actual goal is a low-dimensional quantity of interest
(QoI).  Goal-oriented approximations can estimate the posterior QoI without
forming the full parameter posterior.  In linear-Gaussian problems, such
low-rank approximations are provably optimal under Bayes risk and posterior
distribution metrics:

- https://arxiv.org/abs/1607.01881
- https://arxiv.org/abs/2304.08324

The grasp problem is a harder and not-yet-covered variant:

1. the latent parameter is a nonlinear 3-D target and physical properties;
2. observations are irregular ray events with external censoring;
3. the QoI is not fixed--every new \(g\in SE(3)\) induces a new certificate
   \(\Xi_g\);
4. the output is a posterior over that certificate, not only its mean;
5. the operator must be amortized and equivariant.

This motivates the broad research question:

> Can we learn a query-indexed posterior operator for low-dimensional physical
> quantities of interest directly from censored measurements, with error and
> compute controlled by the query support rather than by the latent field
> resolution?

Existing neural-operator work provides universal/scalable function-to-function
machinery and can handle irregular observations, but does not supply this
grasp-specific quotient or the censoring model.  Probabilistic neural-operator
processes under partial observations appeared in 2026 and must be cited as a
close general framework, not silently rediscovered:

- https://proceedings.mlr.press/v202/hao23c.html
- https://arxiv.org/abs/2606.22946

### Why the query must be inside the representation

A single compact state representation sufficient for *all* actions may need to
retain the full latent state.  A query-specific certificate can be much
smaller.  A simple lower-bound construction makes this precise.  Let
\(S=(B_1,\ldots,B_d)\) contain independent bits and let action \(i\) have
success \(q(i,S)=B_i\).  Any deterministic representation sufficient to answer
all \(d\) actions must distinguish all \(2^d\) states and hence carry at least
\(d\) bits.  After action \(i\) is supplied, the sufficient certificate is the
single bit \(B_i\).

The example does not prove that a particular gripper certificate is small; it
does show why an on-demand query operator can have an asymptotic advantage over
an action-agnostic completed state.  Prior “action-sufficient representation”
work studies one latent representation for control.  The intended distinction
here is a *query-indexed family* of physical sufficient statistics:

https://proceedings.mlr.press/v162/huang22f.html

## Formal problem

Let \(S\sim p_{\mathrm{shape}}(S\mid c)\) be target geometry/physical state,
\(O\) a foreground object, and \(r_u(t)=o+t d_u\) a camera ray.  Define first
target and foreground depths

\[
 T_u(S)=\inf\{t:r_u(t)\in\partial S\},\qquad
 C_u(O)=\inf\{t:r_u(t)\in\partial O\}.
\]

The RGB-D event is a noisy version of \(\min(T_u,C_u)\), together with a
probabilistic semantic mark.  The evidence set is

\[
 E=\{(r_u,I_u,\tau_u,a_u,\sigma_u)\}_{u\in\mathcal U},
\]

where \(I_u\) is a hit interval or free/censor threshold, \(\tau_u\) is the
event type, \(a_u\) is soft target-association probability, and \(\sigma_u\)
describes depth uncertainty.  A hard YOLO class is never assumed.  Detector
logits are part of \(E\); class-token dropout yields the generic-prior mode.

For a pose \(g\), let \(\Xi_g(S)\in\mathcal X_g\) be the finite gripper support
certificate.  The desired posterior is the push-forward

\[
 p(\xi\mid E,g)
 =\int \delta(\xi-\Xi_g(S))p(S\mid E)\,dS.
\]

With observed-foreground collision handling held fixed, the Bayes score and
action are

\[
 Q^*(g,E)=\int \tilde q_\psi(g,\xi)
                    p(\xi\mid E,g)d\xi,
 \qquad
 g^*(E)=\arg\max_{g\in\mathcal G_{\rm obs-free}}Q^*(g,E).
\]

This expression explicitly answers the training-prior question: known class
changes \(p(S\mid E)\); unknown class marginalizes the training shape mixture.
Neither condition identifies geometry outside the support of that prior.  No
one-view method can guarantee correct hidden contacts for two out-of-support
shapes with identical \(E\).  The method should expose high certificate
uncertainty there rather than claim prior-free amodal perception.

## Certificate construction for a parallel-jaw gripper

Use the gripper frame axes \(e_a\) (approach), \(e_c\) (closing), and \(e_z\).
Choose an \(h\)-net \(\{z_j\}_{j=1}^{J}\) over the inner pad footprint.  Each
query closing line is

\[
 \ell_{g,j}(s)=t+R(z_j+s e_c).
\]

The first surfaces encountered by pads closing from opposite sides depend on
the extreme occupied coordinates

\[
 L_j(S)=\inf\{s:\ell_{g,j}(s)\in S\},\qquad
 R_j(S)=\sup\{s:\ell_{g,j}(s)\in S\},
\]

plus their outward normals.  Separate approach rays form an \(h\)-net over the
finger/palm shell and return first-hit clearance.  Multiple internal object
intervals do not have to be reconstructed: for rigid parallel pads, the two
directional extremes are the first potential contacts.  Nonexistence is an
explicit event.

A compact certificate is

\[
 \Xi_g=
 \{L_j,R_j,n_j^L,n_j^R,e_j\}_{j=1}^{J}
 \cup\{d_k^{\rm clear},e_k^{\rm clear}\}_{k=1}^{J_c}.
\]

It contains no target mesh, global SDF, or reusable completed point cloud.
The physics layer derives aperture, pad coverage, antipodal/friction-cone
margin, and local collision margin.  Center-of-mass/material effects not
identified by these variables belong in a calibrated residual and in the
oracle-sufficiency gap; they must not be hidden by calling the certificate
exact.

## Trainable probabilistic operator

### Evidence and query streams

- Subsample \(N_E\approx256\) high-information camera rays: target interior,
  target boundary, foreground-overlap, and invalid-depth strata.
- For each of \(K_g\) pose queries construct \(J\approx16\) bidirectional
  closing lines and \(J_c\approx16\) approach-shell rays.
- Encode both with Pluecker coordinates.  Cross-attention uses line angle,
  reciprocal product/closest distance, and directed interval order.  Outputs
  are expressed in the gripper frame, giving joint \(SE(3)\) invariance when
  the scene and query are transformed together.

### Joint uncertainty without a full shape latent

A single factorized Gaussian would assert incompatible pad contacts.  Use a
small shared mixture latent \(k\in\{1,\ldots,M\}\):

\[
 p_\theta(\Xi_g\mid E,g)
 =\sum_{m=1}^{M}\alpha_m(E,g)
   \prod_j p_\theta(\xi_j\mid m,E,g),
\]

with \(M=4\) or \(8\).  Ordered support depths use bounded logistic mixtures;
normal marks use von Mises--Fisher factors; existence uses Bernoulli factors.
The shared component couples all query lines and permits several mutually
exclusive hidden-shape hypotheses.  The NLL is tractable and the certificate
has only \(O(J+J_c)\) variables per queried pose.

### Supervision

Synthetic full meshes are privileged *label generators*, not inference
outputs.  For every rendered censored observation and candidate pose, ray
casting returns the exact \(\Xi_g(S)\), while the physics simulator returns a
small-lift label \(y\).  Train with

\[
 \mathcal L=
 -\log p_\theta(\Xi_g(S)\mid E,g)
 +\lambda_{\rm mech}\operatorname{BCE}
       (\tilde q_\psi(g,\Xi_g(S)),y)
 +\lambda_{\rm dec}\operatorname{BCE}(\hat Q_\theta(g,E),y),
\]

where \(\hat Q_\theta\) integrates the mechanics head over the predicted
mixture.  The direct decision term protects against certificate
misspecification; it does not excuse a large oracle-certificate gap.

Paired clean/occluded renders share \(S,g,\Xi_g\), giving dense supervision
across censoring patterns.  Real grasp labels can calibrate only the final
decision/residual head.  This is supervised sim-to-real adaptation, not RL.

### Candidate generation

The first scientific experiment must rerank an identical high-recall candidate
set for every method.  This separates certificate quality from generator
recall.  A complete system can share the ray encoder with a conventional
\(SE(3)\) proposal diffuser and query the certificate only on its final
candidates.  Diffusion is a replaceable proposal mechanism and is not a claim.

For \(N_E=256\), \(K_g=64\), and \(J+J_c=32\), local top-\(k\) cross-attention
requires roughly \(K_g(J+J_c)k\) interactions and is independent of a dense
voxel resolution.  Runtime/memory claims must be measured; this count is only
the architectural reason to expect efficiency.

## Statements that can actually be proved

### 1. Information dominance

The typed ray set deterministically yields the cropped target PCD, so it
Blackwell-dominates it.  Strict improvement is possible when censor geometry
distinguishes latent cases sharing the same visible target points.  It does not
information-dominate TARGO's full scene input; against that baseline the claim
is inductive bias and efficiency.

### 2. Query-compression separation

The independent-bit construction above proves that a uniformly sufficient
state code can require \(\Omega(d)\) bits while each query certificate requires
one.  This is a general separation result, not a grasp-performance theorem.

### 3. Margin-qualified certificate approximation

Under positive reach \(\rho\), Lipschitz normals, rigid pads, and an \(h\)-net,
depth/normal error \(\epsilon\) perturbs geometric clearance and antipodal
margins by at most \(C_1h+C_2\epsilon\).  Any grasp with true margin larger
than that bound keeps its feasibility label.  The formal proof must specify
the gripper shell, contact model, and excluded nonsmooth shapes.

### 4. Decision regret

If \(\sup_g|\hat Q(g,E)-Q^*(g,E)|\le\delta\), selecting the maximum of
\(\hat Q\) has Bayes utility regret at most \(2\delta\).  If a finite proposal
set has oracle recall loss \(\epsilon_K\), total regret is bounded by
\(2\delta+\epsilon_K\).  This cleanly separates proposal failure from posterior
certificate error.

None of these results proves that the learned operator reaches the assumed
error.  That burden remains empirical.

## Post-audit rejection of Candidate B

Candidate B no longer survives the literature audit.  On 19 August 2026,
PartialBiGrasp introduced grasp-aligned local occupancy queries for hidden
geometry inferred from a single partial RGB-D view.  Its local encoder is
queried at inner and outer gripper control points; the predictions drive
contact encouragement, collision penalties, and pose refinement.  It is
bimanual and deterministic, but that does not rescue the proposed novelty:
"infer hidden local geometry only where a grasp asks for it" is already an
explicit methodological claim and implementation.

- https://arxiv.org/html/2608.19188

Adding an explicit censor token, continuous support depths, or a mixture
posterior would now read as a technically useful extension of that idea rather
than a new learning object.  Candidate B is therefore rejected.  The next
candidate keeps neither local occupancy nor a per-ray surface certificate as
its output.

## Candidate C: conditional capacity operators for gripper events

### The object being learned

Let \(X\subset\mathbb R^3\) be the *solid* target, treated under the posterior
\(p(X\mid E)\) as a random regular closed set.  For a compact test region
\(K\subset\mathbb R^3\), define the conditional hit and avoidance functionals

\[
 T_E(K)=\Pr[X\cap K\ne\varnothing\mid E],\qquad
 V_E(K)=1-T_E(K).
\]

This is neither pointwise occupancy nor a reconstructed shape.  A query is an
entire geometrical region: an approach-swept finger volume, a left or right
contact slab, a clearance tube, or a union of such regions.  Classical random-
set theory supplies the key mathematical fact: an upper-semicontinuous,
completely alternating capacity functional characterizes the law of a random
closed set (Choquet's theorem).

- https://link.springer.com/book/10.1007/978-1-4471-7349-6
- https://www.math.u-szeged.hu/~kevei/tanitas/irodalom/Schneider%20%26%20Weil%20-%20Stochastic%20and%20Integral%20Geometry.pdf

Learning the capacity on *all* compact sets would be equivalent to learning
the full shape law and would violate the intended scope.  The proposal learns
only its restriction to a gripper-induced query family
\(\mathcal K_{\rm grip}\).  It never evaluates a global voxel lattice or
decodes a reusable surface.

### Why pointwise occupancy is not enough

Suppose two hidden-shape posteriors induce hit bits \(H_L,H_R\) for left and
right contact slabs.  Posterior A assigns probability \(1/2\) to each of
\((0,0)\) and \((1,1)\); posterior B assigns probability \(1/2\) to each of
\((1,0)\) and \((0,1)\).  Both have identical point/slab marginals,

\[
 \Pr(H_L=1)=\Pr(H_R=1)=1/2,
\]

but the probability of bilateral contact is \(1/2\) under A and zero under B.
No decision rule receiving only the two marginal occupancies can distinguish
them.  A full joint shape generator can, but at the cost the project wishes to
avoid.  A restricted capacity can also distinguish them because it answers
queries on unions.

This failure mode is relevant to current methods.  PartialBiGrasp uses
deterministic occupancy predictions at gripper control points and an average/
TopK objective.  Classical occupancy mapping also commonly assumes independent
cells; work that models spatial correlation reports better accuracy and
uncertainty, although not in grasping:

- https://arxiv.org/abs/1801.07380
- https://arxiv.org/abs/1911.07915

The argument is structural, not an empirical claim that those two mapping
methods transfer directly to grasping.

### Exact Boolean gripper probabilities from avoidance queries

For one candidate grasp, let \(K_0\) be its forbidden approach/palm/finger
shell, and \(K_L,K_R\) be positive-volume capture slabs for the two intended
contacts.  Define \(H_i=\mathbf 1[X\cap K_i\ne\varnothing]\).  The geometric
event "shell clear and both contacts exist" is

\[
 A_g=\{H_0=0,H_L=1,H_R=1\}.
\]

Inclusion--exclusion gives

\[
 \Pr(A_g\mid E)=
 V_E(K_0)-V_E(K_0\cup K_L)-V_E(K_0\cup K_R)
 +V_E(K_0\cup K_L\cup K_R).
\]

Thus one coherent set functional represents correlations needed by contact
and collision simultaneously.  If every one of the four avoidance queries is
estimated within \(\epsilon\), this event probability is estimated within
\(4\epsilon\).  More detailed friction/contact predicates use a finite lattice
of nested, asymmetric micro-slabs; their error constant is the number of
Möbius terms actually used, not the resolution of a scene grid.

There is also a useful distance interpretation.  For a closed ball
\(B(x,r)\),

\[
 T_E(B(x,r))=\Pr[d(x,X)\le r\mid E].
\]

Nested capacity queries therefore yield a distance *distribution* without an
SDF.  Directional or wedge-shaped nested sets analogously probe local contact
orientation.  This is the replacement for Candidate B's explicit hidden
surface depths.

### Finite-lattice parameterization that is coherent by construction

An unconstrained MLP \(K\mapsto T_E(K)\) can violate monotonicity and produce
negative inclusion--exclusion probabilities.  For each grasp, partition its
relevant volume into \(r\) semantic binary hit regions and predict the joint
law

\[
 p_\theta(h_1,\ldots,h_r\mid E,g),\qquad h_i\in\{0,1\}.
\]

Use a non-negative conditional tensor-train probability circuit:

\[
 p_\theta(h\mid E,g)
 =\frac{G_1(h_1;E,g)G_2(h_2;E,g)\cdots G_r(h_r;E,g)}
        {Z_\theta(E,g)},
\]

where endpoint cores have shape \(1\times\chi\) and \(\chi\times1\), internal
cores are \(\chi\times\chi\), and all entries are positive.  Forward
contraction computes \(Z_\theta\), marginals, and any avoidance query in
\(O(r\chi^2)\).  Then

\[
 \widehat V_E\!\left(\bigcup_{i\in A}K_i\right)
 =\sum_h p_\theta(h\mid E,g)\prod_{i\in A}(1-h_i)
\]

is automatically normalized, monotone, and completely monotone on the finite
union lattice.  Its dual \(1-\widehat V_E\) is a valid finite capacity.  This
is stronger than adding a consistency penalty after prediction.

The tensor rank \(\chi\) controls correlation capacity.  \(\chi=1\) is the
independent-occupancy baseline; increasing \(\chi\) captures mutually
exclusive hidden-shape hypotheses without a volumetric latent.  A fixed
geometric ordering of the regions (approach shell, left pad hierarchy, right
pad hierarchy) makes contraction deterministic.  A tree tensor network is a
fallback if long-range rank becomes a bottleneck, but should not be introduced
unless the rank ablation demands it.

### Architecture: CapGrasp

1. **Censored evidence encoder.**  Encode the same typed camera-ray evidence
   \(E\) defined earlier: target hit, foreground hit/censor, free/invalid,
   depth interval, RGB feature, soft association, and detector class logits.
   This input design is inherited from the rejected Candidate A and is not a
   standalone novelty claim.
2. **Set-query encoder.**  Represent each gripper region \(K_i(g)\) by its
   analytic primitive type, pose, dimensions, nesting parent, and a small
   boundary/interior quadrature.  Cross-attend only to the nearest and
   projectively relevant evidence rays.  All relative geometry is expressed
   in the gripper frame.
3. **Capacity circuit.**  Convert the \(r\) region embeddings and one shared
   evidence token into the non-negative tensor-train cores above.  This is the
   conditional law of the joint hit signature, not an occupancy field.
4. **Mechanics decoder.**  A small calibrated model maps each hit signature,
   visible local normals, aperture, and gripper parameters to a small-lift
   success probability.  The final score marginalizes this decoder exactly or
   by low-variance circuit sampling under \(p_\theta(h\mid E,g)\).

Use a multiresolution gripper partition with, initially, \(r=24\text{--}40\)
regions: coarse forbidden shell cells, nested bilateral contact slabs, and a
few asymmetric probes for normal/friction compatibility.  With tensor rank
\(\chi=8\), the capacity computation is \(O(r\chi^2)\) per grasp; evidence
cross-attention, not the circuit, is likely to dominate runtime.

### Training without shape reconstruction or RL

From a full synthetic mesh, binary hit signatures for millions of gripper
queries are cheap exact collision/intersection labels.  Train

\[
 \mathcal L_{\rm cap}=-\log p_\theta(H_g(X)\mid E,g).
\]

Only the low-dimensional mechanics decoder needs simulator small-lift labels:

\[
 \mathcal L_{\rm mech}=\operatorname{BCE}
       (q_\psi(H_g(X),g),y),
\]

and an optional end decision loss trains the marginalized score.  Mesh
occupancy is never a network target on a global grid.  Paired unobstructed and
foreground-occluded renders share the exact same \(X,g,H_g(X)\), isolating the
information removed by the occluder.  Equal weighting per object prevents
objects with many successful grasps from silently changing the shape prior.

Known YOLO class is handled as a soft prior token with class dropout.  Report
four distinct regimes: seen instance, unseen instance/seen class, unseen
class, and deliberately shifted shape prior.  The last regime should increase
entropy or trigger abstention; no claim of recovering hidden geometry outside
training support is defensible.

### Novelty boundary after the 25 August 2026 search

Not novel separately:

- implicit occupancy supervision, local gripper queries, or grasp refinement
  (PartialBiGrasp);
- full probabilistic shape posteriors or generative occupancy fields;
- point/ray transformers and foreground augmentation;
- tensor trains/probabilistic circuits;
- classical capacity and random-closed-set theory.

The potentially new contribution is:

> a learned conditional capacity restricted to action-induced geometric test
> sets, parameterized as a coherent finite-lattice probability circuit, so
> correlated contact/collision event probabilities are computed without a
> global occupancy field or samples of completed shapes.

Searches for "neural capacity functional", "random closed set neural
occupancy", and action-conditioned joint occupancy did not find this object in
grasping or general geometric learning.  This is evidence of a gap, not a
guarantee; a final Scholar/Semantic Scholar/OpenReview citation audit remains
mandatory before submission.

### Kill tests

Candidate C is rejected immediately if any of the following fail:

1. **Signature sufficiency.**  With ground-truth hit signatures, the mechanics
   decoder must come within 2--3 percentage points of a full-mesh oracle on
   small-lift success.  Otherwise the finite lattice omits essential physics.
2. **Correlation necessity.**  Rank \(\chi>1\) must beat the rank-one
   independent model by at least 3 points on ambiguity pairs and improve NLL/
   calibration.  Otherwise random-set machinery is ornamental.
3. **Structured-supervision value.**  At equal parameters and candidate set,
   CapGrasp must beat a direct \(p(y\mid E,g)\) BCE scorer, especially with
   only 1%, 5%, and 10% of simulator labels.  Otherwise the capacity is merely
   an interpretable detour.
4. **No hidden reconstruction cost.**  Runtime and peak memory must scale with
   queried grasps/regions and remain below TARGO plus its completed-volume
   decoder.  A de facto dense field implementation invalidates the efficiency
   claim.
5. **External-occlusion specificity.**  Gains must be concentrated in paired
   scenes where foreground censoring creates hidden-contact ambiguity, while
   low-occlusion performance stays within 2 points of the strongest direct
   baseline.  Uniform tiny gains are not evidence for the proposed mechanism.

## Decisive experimental programme

### Benchmark that matches the laboratory problem

Create a paired benchmark with exactly one target on a shelf and zero or one
foreground obstacle.  Do not import TARGO's multi-object clutter as an
unspoken change of task.  For every tuple of target mesh, pose, wrist-camera
pose, and grasp candidates, render several controlled foreground occluders at
occlusion levels \(0,0.2,0.4,0.6,0.8\).  Measure occlusion against the target-
alone silhouette, not against the visible bounding box.

Use two evaluation tracks:

1. **Perceptual-censoring track.**  The obstacle is present in RGB-D generation
   but removed before grasp execution.  Although not a deployment scenario,
   this intervention cleanly estimates failure due to missing target evidence.
2. **Physical-obstacle track.**  The obstacle remains.  Every method receives
   the same observed-scene collision filter and the same reachable candidate
   set.  This estimates the combined shelf deployment problem without letting
   motion/collision planning become the paper's contribution.

The task starts from a common reachable pre-grasp.  Success is closure followed
by a 2--3 cm lift held for two seconds.  Approach planning and humanoid whole-
body feasibility are explicitly outside the label.  This avoids the previously
rejected whole-cycle feasibility formulation.

Render clean depth, structured RealSense-style noise, missing-depth boundaries,
and soft target/foreground segmentation errors.  Real experiments use the
actual wrist camera.  At severe occlusion the detector may fail entirely, so
report both (i) conditional grasp success given a detected target and (ii)
end-to-end detection-times-grasp success.  Do not hide detector failures by
dropping those trials.

### Shape-prior protocol

Split CAD instances before rendering and retain category metadata.  Train and
report:

- unseen views/poses of seen instances (debugging only);
- unseen instances from seen classes;
- unseen classes drawn from the training super-category;
- a deliberately shifted thin/hollow/concave shape set;
- class posterior supplied, class token removed, and class token corrupted.

The class-conditioned model consumes detector logits, not a perfect oracle
label.  Class dropout during training implements the generic mixture prior.
The shifted split is expected to be harder; the claim there is calibrated
uncertainty/abstention, not miraculous hidden-shape recovery.

### Stage 0: test the scientific premise before training CapGrasp

Use paired target-only and foreground-occluded observations with an identical
full-mesh candidate bank.  Evaluate GraspGen/GDN-style direct inference,
TARGO-Net, and a target-only PCD scorer.  The direction is not worth pursuing
unless external foreground occlusion produces a statistically clear drop after
candidate recall and physical obstacle collision are controlled.

Existing evidence makes the premise plausible.  TARGO evaluates 1,000 scenes
per occlusion range and reports about a 7-point synthetic drop for TARGO-Net,
up to an 18-point benefit from its completion module at higher occlusion, and
a real decline from 80.0% easy to 66.7% hard.  Its authors attribute the larger
real gap to depth-noise patterns that challenge completion:

- https://arxiv.org/html/2407.06168

These numbers are not directly transferable because TARGO is cluttered and
uses a different execution definition; they justify the experiment, not the
answer.

### Stage 1: oracle signature sufficiency

Before implementing an evidence encoder, compute exact hit patterns from full
meshes for \(r\in\{8,16,24,40\}\).  Train only the small mechanics decoder and
compare:

- full-mesh simulator/analytic oracle;
- exact local SDF samples around the gripper;
- exact CapGrasp hit signature;
- an independent-bit version of that signature;
- direct \(g\mapsto y\) using the same low-dimensional gripper features.

Plot the accuracy gap versus \(r\), tensor rank, object family, and grasp
margin.  Stop if \(r\le40\) cannot reach the 2--3 point oracle-gap threshold.
This pilot is cheap and prevents months of work on an insufficient quotient.

### Stage 2: construct ambiguity pairs

Automatically retrieve pairs of shapes/renderings with close visible target
PCD and the same foreground evidence but different ground-truth bilateral hit
patterns for a common grasp.  Match camera, scale, visible Chamfer distance,
and candidate pose.  These pairs directly test the claimed point-marginal
separation rather than relying only on average success.

Primary metrics are joint-pattern NLL, bilateral-event Brier score, ECE,
coverage-versus-risk under abstention, and accuracy of the four-term capacity
identity.  Rank one must fail on the designed dependence while \(\chi>1\)
recovers it.  If a direct BCE scorer is equally calibrated and sample-efficient,
the proposed representation has no empirical reason to exist.

### Stage 3: fixed-candidate visual inference

All learned methods rerank the same 64 or 128 high-recall candidates.  Report
candidate oracle recall separately.  Baselines are:

1. direct partial-PCD GDN and GraspGen-style scorers/generators;
2. TARGO-Net and TARGO-Net without completion;
3. ShellGrasp-style deterministic shell prediction plus the same scorer;
4. probabilistic completion plus Monte Carlo grasp scoring;
5. a parameter-matched full-scene cross-attention BCE scorer;
6. a PartialBiGrasp-style deterministic local-occupancy head at the same
   gripper probes;
7. CapGrasp with rank \(\chi=1\), which is the independent joint law;
8. CapGrasp without censor-ray types, without class logits, and without paired
   occlusion training.

The same target mask/logits, candidate bank, collision filter, training objects,
and tuning budget are mandatory.  Otherwise a performance difference cannot
be attributed to the capacity representation.

### Stage 4: end-to-end and real shelf evaluation

Only after the fixed-candidate test passes should a proposal generator be
added.  Compare top-1 small-lift success in at least 100 paired physical trials
per occlusion regime, randomized across methods and target objects.  Use a
hierarchical bootstrap over object then scene/trial and report 95% confidence
intervals.  Log target-contact failure, hidden-target collision, foreground/
shelf collision, empty closure, and post-lift slip; these are descriptive
outcomes, not a causal-failure-mode model.

### Metrics and efficiency

The main metric is top-1 small-lift success by occlusion level.  Also report
area under the success-versus-occlusion curve, high-occlusion success,
occlusion degradation, hidden-target collision rate, oracle recall@K,
joint-pattern NLL, Brier/ECE, risk-coverage, latency, peak VRAM, and number of
simulator labels needed to reach each success level.

Measure capacity-label generation separately from physics simulation.  One of
the framework's strongest practical claims is that mesh--set intersections are
cheap dense supervision while small-lift simulation labels are expensive.  A
sample-efficiency curve at 1%, 5%, 10%, 25%, and 100% of simulator labels is
therefore more important than reporting only a final large-data number.

## What would count as SOTA here

ICLR does not require SOTA by rule; the official reviewer guide asks whether
the work is motivated, correct, rigorous, novel, and significant, and
explicitly says lack of SOTA alone is not grounds for rejection:

- https://iclr.cc/Conferences/2027/ReviewerGuidelines

The laboratory objective is stricter.  Pre-register the following go/no-go
targets on the new matched benchmark:

- at least +5 absolute success points over the best non-oracle method in the
  60--80% occlusion bin, with a confidence interval excluding zero;
- at least +8 points in the hard real-shelf regime if the strongest baseline
  is near TARGO's reported 66.7%, while losing no more than 2 points below 20%
  occlusion;
- at least 30% fewer hidden-target collisions than the best direct scorer;
- lower pattern/event NLL and ECE than probabilistic completion while using
  less than one tenth of its stochastic forward passes;
- under 200 ms total reranking latency for 64 candidates on the laboratory GPU
  and lower peak memory than the completion baseline.

The exact real-success threshold should be updated after measuring the local
baseline.  Claiming SOTA against TARGO numbers collected on a different robot,
scene distribution, and success definition would be invalid.

There is credible indirect evidence for the target effect size, but no proof:

- TARGO's completion module becomes worth up to 18 points at stronger
  occlusion, while its real completion suffers more than synthetic completion;
- PartialBiGrasp's local occupancy refinement raises simulation grasp success
  from 61.51% to 67.87% and cuts pair collision from 32.61% to 17.48%;
- uncertainty-aware shape completion reports gains of +23, +7, +31, and +17
  points over its four corresponding grasp baselines, but requires 60 forward
  passes and about six seconds end to end.

- https://arxiv.org/html/2608.19188
- https://arxiv.org/html/2504.16183

These results support hidden local geometry, dependence/uncertainty, and a
need for cheaper query-focused inference.  They do not establish that a tensor-
train capacity will learn the correct correlations.

## Closest-work matrix

| Work/family | Learned output | Joint hidden-shape uncertainty | Action-indexed | Dense/global geometry | Explicit external censoring |
|---|---|---:|---:|---:|---:|
| TARGO-Net | completed target + grasp volume | no | grasp after completion | yes | scene context, not a censor law |
| ShellGrasp | camera-ray entry/exit shell + grasp maps | no | no | global camera shell | no |
| PartialBiGrasp | global/local occupancy queried at gripper points | no | local queries yes | implicit triplane field | no |
| probabilistic completion/UQ | samples or uncertainty of completed shape | partly/yes | no | yes | no |
| 3DP3/probabilistic voxel shape | posterior voxel occupancy | approximated joint mixture | no | yes | ray likelihood |
| GDN/GraspGen/direct scorer | grasp distribution or quality | implicit only | output is action | no shape output | no |
| classical random-set theory | capacity characterizes random-set law | yes | arbitrary test sets | mathematical, not learned | no |
| **CapGrasp** | coherent joint hit law on gripper-set lattice | **yes** | **yes** | **no** | **typed censor evidence** |

The most dangerous comparison is PartialBiGrasp, not TARGO.  The paper must
show that joint event modeling, rather than merely querying near a gripper,
causes the gain.  Rank-one, deterministic-local-occupancy, and direct-BCE
baselines are therefore headline experiments, not appendix ablations.

## ICLR red-team audit

### Why this could be an ICLR paper

- **Broad question:** how to learn coherent probabilities of set-intersection
  events from censored observations without estimating the entire latent
  geometric field.
- **New learning object:** a conditional capacity restricted to decision-
  induced test sets, rather than another grasp architecture or completion
  loss.
- **Mathematical content:** finite Choquet consistency by construction,
  marginal-indistinguishability separation, event-error bounds by Möbius
  support, and ordinary decision-regret bounds.
- **Algorithmic content:** a conditional tensor-network circuit whose cost is
  controlled by the number of physical query regions, with cheap privileged
  intersection supervision.
- **Empirical content:** a paired intervention isolates perceptual censoring;
  ambiguity pairs directly test the joint-law claim; real shelf experiments
  test sim-to-real efficiency.

### Strong reviewer objections

1. **"This is only structured multi-label classification."**  This objection
   wins unless the paper demonstrates out-of-distribution set-query
   generalization, exact coherence advantages, and lower simulator-label
   complexity.  The Choquet vocabulary alone adds no value.
2. **"PartialBiGrasp already queries hidden occupancy at the gripper."**  This
   wins unless rank \(\chi>1\) beats its deterministic/marginal analogue on real
   ambiguity, not just on a toy construction.
3. **"A direct success critic is all that is needed."**  This wins unless the
   capacity labels confer clear sample efficiency, calibration, or transfer to
   new gripper dimensions/contact predicates.
4. **"The hit lattice discards the physics that matters."**  Center of mass,
   friction/material, compliance, and thin nonsmooth contacts can break the
   oracle signature.  The Stage-1 gate must be reported even if it kills the
   method.
5. **"The capacity is only valid on one arbitrary discretization."**  Test
   nested refinement, new set unions, and changed gripper width.  Do not invoke
   the infinite-dimensional Choquet theorem as if a finite circuit proved
   global validity.
6. **"The input assumes an amodal target/occluder assignment."**  Use soft
   association, include detector failures, and ablate oracle masks versus
   predicted masks.
7. **"Synthetic intersections do not transfer."**  Evaluate depth noise,
   segmentation noise, transparent/dark failure exclusions, and real objects;
   calibrate with a small real set without fine-tuning the geometry circuit if
   possible.

### Honest verdict on 25 August 2026

CapGrasp is the first candidate in this pass whose proposed novelty survives
the current exact-collision search.  It is more defensible than Candidate B
because the output is not local occupancy or hidden contact geometry; it is a
coherent probability law of whole-region intersection events.  It also has a
genuine general-ML face rather than being a composition of robotics modules.

It is **not yet an objectively validated SOTA framework**.  At idea stage the
ICLR case is approximately 6/10: plausible weak accept if the mathematical
construction is clear, but reject without the three decisive empirical gaps.
It can become an 8/10-level submission only if:

1. the \(r\le40\) oracle signature is nearly sufficient;
2. correlated capacity decisively beats rank-one occupancy and direct BCE on
   ambiguity and scarce simulator labels;
3. the fixed-candidate and end-to-end system sets a statistically significant
   high-occlusion SOTA while remaining faster than completion/UQ;
4. the paper includes a second non-grasp conditional-random-set task or a
   strong set-query generalization theorem, so the ICLR relevance is broader
   than one robot setup.

If any of the first three conditions fails, reject Candidate C and do not turn
it into a weaker "uncertainty-aware local occupancy" paper.  The minimum next
action is the Stage-1 oracle pilot; it requires no new network and answers the
largest technical uncertainty fastest.
