# Минимальный общий simulation + object-data stack для ICLR-идей по grasp under occlusion

Дата исследования: 2026-08-25  
Локальные проекты: `SRNO` и `/home/weshi/graspvalidation`

## Короткий ответ

**Готового публичного датасета, достаточного для всех предложенных идей, нет.** Ни ACRONYM, ни GraspNet-1Billion, ни GraspGen, ни TARGO не содержат одновременно:

1. полный latent world с mesh и физическими параметрами;
2. контролируемые семейства наблюдений одного и того же мира — вложенные occlusions, sensor degradations и calibrated micro-motion;
3. один и тот же bank кандидатов grasp для всех наблюдений и скрытых миров;
4. не только binary success, но contact/collision events, signed margins, short-lift trajectories и perturbation/fidelity axes;
5. явные группы visually indistinguishable hidden-shape twins.

Тем не менее **один основной simulator и один основной object corpus достаточны для общей тренировки почти всех идей без сильного переписывания**:

- **simulator:** оставить текущий Isaac Lab / Isaac Sim PhysX pipeline из `graspvalidation`;
- **engineering layer:** перенести в него только полезные компоненты [NVIDIA GraspDataGen](https://github.com/NVlabs/GraspDataGen) для импорта mesh→USD, candidate generation/validation и custom gripper support;
- **основной object corpus:** отфильтрованный и зафиксированный subset из **8,515 Objaverse-XL/LVIS объектов GraspGen**; все Astribot-labels пересчитать самим, а 57M готовых GraspGen grasp использовать только как proposal/prior, потому что они относятся к трём другим grippers ([официальный GraspGen](https://github.com/NVlabs/GraspGen));
- **общий формат:** immutable scene/world records + observation groups + shared candidate banks + append-only label tables;
- **обязательный generated overlay:** hidden-shape/twin families и occluders, построенные поверх основного corpus;
- **evaluation, не второй train corpus:** локальные физические объекты + YCB/EGAD и внешние GraspNet/TARGO adapters.

Если «достаточен для всех идей» означает не только обучение прототипов, но и **защищаемую экспериментальную валидацию всех механических claims**, нужны ещё два небольших слоя:

1. реальная paired RGB-D съёмка с калиброванным micro-motion для EdgeFlux/AcqGrasp;
2. high-fidelity compliant-contact audit на малом subset в IPC-GraspSim или GRIP для RelaxGrasp/LimitGrasp.

Это не два новых bulk pipelines: основной объём данных остаётся в Isaac; дополнительные слои применяются только к небольшим audit subsets.

## 1. Что уже есть локально

### 1.1 `graspvalidation`

Текущий pipeline — хорошая основа, потому что он уже реализует именно нужный физический сценарий:

- Isaac Lab / PhysX, gravity `(0, 0, -9.81)`;
- exact Astribot six-joint gripper;
- объект оседает на shelf;
- approach → close → lift на 15 cm → hold;
- binary success требует удержания объекта без существенного падения;
- 120 Hz physics и TGS;
- параллельные environments.

Это существенно ближе к конечной задаче, чем перенос в MuJoCo, ManiSkill/PyBullet или FleX.

Главное ограничение сейчас — не physics engine, а **dataset contract**. `dataset_pipeline.py` сохраняет в основном pose кандидата, orientation и binary success. Он не сохраняет RGB-D, segmentation, rays, contacts, continuous margins, trajectories, perturbation groups, fidelity fingerprint или связь между вариантами одного мира.

Есть ещё три методологических риска:

- текущие generated `train.txt` и `valid.txt` используют один и тот же полный список объектов — object leakage;
- положительные примеры дополняются до фиксированного размера повторением, что нельзя считать новыми независимыми labels;
- локальный набор слишком узок для основной ICLR training distribution: в `SRNO/assets/catalog.json` сейчас 28 объектов, а в `graspvalidation/config.yaml` — 29 supermarket assets; один объект не синхронизирован между каталогами.

Проверка всех имеющихся `grasp_dataset*/splits/astribot` показала, что в каждом варианте `train.txt` и `valid.txt` побайтово идентичны. Это не случайность старого export: текущий `dataset_pipeline.py` явно записывает один `objects` list в оба файла.

### 1.2 SRNO collector — полезный label backend, но не замена `graspvalidation`

В SRNO уже есть полезный HDF5 schema и богатые closure trajectories: SDF, object pose, six joints, aperture, efforts, contact count, penetration, velocities, settling и physics fingerprint. Это стоит переиспользовать как trajectory-label layer.

Однако текущий SRNO collector специально работает **без gravity и без shelf**. Поэтому его нельзя молча объединять с `graspvalidation` labels: это другой intervention/protocol. Правильная интеграция — один общий schema, но разные `protocol_id` и `physics_fingerprint`.

## 2. Общий знаменатель всех md-идей

Идеи распадаются на шесть семейств, но требуют не шесть datasets, а разные views одного master record.

| Семейство | Локальные идеи | Нужные базовые данные | Специальный слой |
|---|---|---|---|
| Conditional outcome/posterior | Blackwell/TQ-Grasp, DQPL, FIRE-Grasp, task-image/pushforward, FiGO/OC-GOP | RGB-D, full world, shared grasp bank, continuous outcome/margin | nested occlusion и observation equivalence groups |
| Random feasible sets/capacities | CapGrasp, Choquet, AvoGrasp, FiberGrasp, response polytopes/spectra | per-world × per-grasp event vector | unions/packets/twins выводятся offline |
| Geometry/mechanics | JILT, MintyGrasp, RelaxGrasp, GraspLFP, InterCap | full mesh/SDF, contacts, normals, collision, force-closure/margin | stiffness/compliance только для Relax/Limit audit |
| Sensor/acquisition | AcqGrasp, EdgeFlux | тот же latent world и label vector при изменении acquisition | resolution/noise/FPS; paired wrist micro-motion |
| Numerical limit | LimitGrasp | один world/candidate на grid solver/collision/contact fidelity | малый multi-fidelity subset |
| Closure operator | SRNO | ordered closure trajectory, contact state, joint effort | 33 aperture/contact states |

Практический вывод: общим объектом данных должен быть не «RGB-D + label success», а функция

`(latent world, observation variant, candidate, perturbation, fidelity, protocol) -> outcomes`.

Все capacity, Choquet, posterior process, polytope, spectrum, regret и packet quantities затем вычисляются offline без повторной симуляции.

## 3. Что есть в публичных datasets и почему одного из них недостаточно

### 3.1 Кандидаты на основной object corpus

| Dataset | Масштаб и сильная сторона | Роль в предлагаемом stack | Почему не готовый master dataset |
|---|---|---|---|
| [GraspGen](https://github.com/NVlabs/GraspGen) | >57M grasps, 8,515 Objaverse-XL/LVIS objects; simplified meshes; complete/partial point clouds; single/clutter | **рекомендуемый primary object corpus и proposal prior** | labels только для Franka, Robotiq-2F-140 и suction; нет нужных fibers, margins и Astribot physics |
| [ACRONYM](https://research.nvidia.com/publication/2021-05_acronym-large-scale-grasp-dataset-based-simulation) | 17.7M parallel-jaw grasps, 8,872 ShapeNetSem objects, 262 categories | сильный baseline и возможный fallback corpus | FleX/generic gripper labels; meshes скачиваются отдельно, требуют preprocessing; BY-NC ограничения; нет controlled observations |
| [EGAD](https://dougsm.github.io/egad/) | 2,282 train + 49 evaluation meshes, специально покрывает geometric complexity и diversity; printable | geometry stress-test и физически печатаемые OOD objects | нет natural textures/scene observations; мало real-category semantics |
| [Google Scanned Objects](https://research.google/pubs/google-scanned-objects-a-high-quality-dataset-of-3d-scanned-household-items/) | >1,000 photorealistic scanned household objects, подготовленные для Bullet/Gazebo | альтернативный texture-rich OOD test | нет grasp/event labels и controlled ambiguity groups |
| [YCB](https://ycb-benchmarks.s3.amazonaws.com/index.html) | physical household objects, textured meshes, RGB/RGB-D scans; CC BY 4.0 | **реальный held-out benchmark** | мал для training и не содержит нужных paired hidden worlds |
| Objaverse-XL raw | очень большой и разнообразный web corpus | источник GraspGen subset, но не брать целиком | непредсказуемые units, topology, license per asset, duplicates, nonphysical meshes |

Почему GraspGen subset предпочтительнее ACRONYM как инженерный минимум:

1. В локальном проекте уже есть GraspGen training/inference path.
2. Официальный GraspGen предоставляет download/simplification workflow именно для этих 8,515 объектов и USD support.
3. Новый [GraspDataGen](https://github.com/NVlabs/GraspDataGen) использует Isaac Lab, то есть не требует смены physics stack.
4. Готовые labels всё равно нельзя использовать как ground truth для Astribot; пересчёт нужен при любом corpus.

ACRONYM остаётся обязательным comparison point и хорошим fallback, если license manifest выбранного Objaverse subset окажется неприемлемым для redistributable benchmark. Смешивать ACRONYM и GraspGen meshes без asset-level deduplication нельзя: будут геометрические и category leaks.

### 3.2 Perception/grasp benchmarks

| Benchmark | Что реально даёт | Как использовать | Чего не даёт |
|---|---|---|---|
| [GraspNet-1Billion](https://graspnet.net/) | 190 clutter scenes, 88 objects, 97,280 real RGB-D images, >1B grasp poses, two cameras | внешний real-sensor grasp benchmark | controlled counterfactual worlds, custom gripper, shared candidate bank для наших fibers |
| [TARGO](https://targo-benchmark.github.io/) | synthetic+real target-driven grasping under graded occlusion; paired uncluttered/occluded target context | **главный внешний occlusion benchmark** | exact Astribot protocol, mechanical layers и designed hidden-shape twins |
| [HouseCat6D](https://sites.google.com/view/housecat6d/) | realistic room-scale trajectories, RGB, active-stereo depth, polarization, object poses and dense grasp annotations | sensor-domain/OOD check | не наш controlled action-fiber dataset |
| [ZeroGrasp](https://openaccess.thecvf.com/content/CVPR2025/papers/Iwase_ZeroGrasp_Zero-Shot_Shape_Reconstruction_Enabled_Robotic_Grasping_CVPR_2025_paper.pdf) | large synthetic image/grasp corpus and shape-completion baseline | современный baseline для completion→grasp | другой gripper/protocol; не заменяет releaseable custom labels |

Эти benchmarks полезны для внешней валидности, но **не должны определять внутренний master schema**: иначе невозможно честно сравнивать все идеи на одном latent world и одном candidate bank.

## 4. Сравнение simulator choices

| Simulator | Плюсы для этой работы | Цена смены / ограничение | Решение |
|---|---|---|---|
| Isaac Lab / PhysX | уже работает с Astribot, shelf, gravity и USD; GPU parallelism; RGB/depth/segmentation sensors; TGS/PGS и фиксируемые settings | rigid/compliant contact остаётся approximation; детерминизм зависит от версии и hardware | **оставить как единственный bulk simulator** |
| [GraspDataGen](https://github.com/NVlabs/GraspDataGen) | Isaac Lab, custom USD gripper/object conversion, physics validation, tug/disturbance | это package/components, а не готовый occlusion benchmark | переиспользовать компоненты внутри текущего pipeline |
| [IPC-GraspSim](https://sites.google.com/berkeley.edu/ipcgraspsim/home) | high-fidelity compliant jaw deformation; физически лучше разделяет borderline grasps | намного дороже, другой narrow protocol | только audit subset |
| [GRIP](https://bell0o.github.io/GRIP/) | IPC-based deformable/rigid grasp data, stress/deformation labels | Astribot adapter и schema mapping потребуют работы | alternative high-fidelity audit |
| [ManiSkill](https://maniskill.readthedocs.io/en/latest/) | быстрый GPU simulation/rendering, много задач | перенос USD/Astribot/env/labels | не менять stack |
| [MuJoCo MJX](https://mujoco.readthedocs.io/en/latest/mjx.html) | batched GPU physics и domain randomization | MJCF conversion, другой contact model и renderer | не менять stack |

Isaac Lab документирует воспроизводимость rigid/articulation simulation при фиксированном hardware/software stack, но не обещает bitwise identity между разными версиями и платформами. Поэтому каждую запись нужно связывать с полным physics fingerprint, а benchmark release — с container/driver/Isaac version ([Isaac Lab reproducibility](https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html)).

## 5. Минимальный master dataset contract

### 5.1 Asset table

Для каждого объекта:

- immutable `asset_id`, source URL/version, license, content hash;
- visual mesh, collision mesh и SDF на нескольких resolutions;
- canonical frame, metric scale, bbox;
- mass, COM, inertia, friction/material prior;
- semantic/geometric category;
- `family_id`, `split`, `is_physical`, `is_procedural_twin`.

Split делается **до** генерации наблюдений по `asset_id/family_id`, а не по кадрам. Twin family целиком принадлежит одному split.

### 5.2 Latent world table

`world_id` фиксирует:

- target asset, target pose после gravity settling;
- shelf/table geometry;
- occluder assets и poses;
- camera intrinsics/extrinsics;
- sampled physical parameters и random seed;
- `twin_family_id` и hidden-morph parameters;
- simulator/container/driver fingerprint.

Важно отделять `pre_settle_pose` от `settled_pose`: labels должны быть связаны с фактическим состоянием перед grasp.

### 5.3 Observation group

Одна группа содержит несколько наблюдений одного world:

- clean/unoccluded oracle render;
- 4–6 nested occlusion levels;
- RGB, depth, target/occluder/validity masks, optional normals;
- exact camera rays и visible-surface provenance;
- depth resolution, quantization, missing-depth/noise variants;
- FPS/thinning variants point cloud;
- две calibrated micro-motion frames и visibility-birth mask для EdgeFlux.

Occlusion level следует измерять как долю невидимой projected target area, а не только как позицию occluder.

### 5.4 Shared candidate bank

`candidate_bank_id` строится **один раз на observation group** и затем неизменно применяется ко всем:

- nested observations;
- hidden twins/fiber members;
- physics perturbations;
- fidelity settings.

Хранить pose, opening/joints, approach vector, generator version и provenance. Кандидаты для main comparison должны строиться только из разрешённого observation, иначе oracle mesh leakage разрушает постановку conditional inference.

Можно отдельно иметь `oracle_candidate_bank` для upper bound, но нельзя смешивать два банка в одной основной метрике.

### 5.5 Label layers

**Cheap analytic layer, для всех candidates:**

- swept-volume collision/clearance;
- left/right finger hit pattern и body collision;
- contact positions/normals;
- antipodal/force-closure score;
- continuous signed feasibility margin;
- distance to estimated feasibility boundary;
- JILT line moments / interaction cells;
- local Minty/QP envelope;
- approximate stiffness/relaxation descriptors.

**PhysX execution layer, для hard/boundary subset:**

- close/lift/hold success;
- object pose and slip trajectory;
- contact counts/impulses, finger joints/efforts;
- drop height/time and failure mode;
- repeated physical-parameter perturbations;
- protocol and fidelity fingerprint.

**High-cost layer, только для малого subset:**

- full 33-step SRNO closure trajectory;
- timestep/contact-offset/collision-resolution/solver-iteration grid для LimitGrasp;
- compliant IPC/GRIP reruns для RelaxGrasp/LimitGrasp audit.

## 6. Почему layered dataset дешевле отдельного датасета под каждую идею

Вектор `event/margin/outcome[world, candidate]` является общей первичной величиной. Из него без simulator rerun получаются:

- Blackwell/TQ/DQPL posterior response processes;
- CapGrasp/Choquet capacities для cells и unions;
- FiberGrasp necessary/possible sets;
- AvoGrasp packet avoidance;
- response polytopes и spectra;
- GraspLFP boundary profiles;
- acquisition utilities и EdgeFlux conditional gains.

Отдельный dataset нужен только если меняется сам intervention: physics fidelity, compliant material или реальное sensor motion. Поэтому формат должен поддерживать append-only label plugins, а не фиксированную монолитную матрицу.

## 7. Минимальные изменения в `graspvalidation`

Ниже — расширение, а не rewrite.

1. **Asset adapter.** JSON/Parquet catalog + mesh→USD conversion; брать из GraspDataGen только converter, collision preparation и custom gripper conventions.
2. **Camera sensor.** Добавить tiled RGB/depth/semantic segmentation sensor и сохранение intrinsics/extrinsics.
3. **World-group runner.** Один settled target переиспользуется для нескольких occluder/acquisition variants; `world_id` и `observation_group_id` становятся first-class keys.
4. **Candidate-bank stage.** Генерировать кандидаты один раз, сохранять, затем replay одинакового банка по worlds/twins/fidelities.
5. **Callbacks/hooks.** `on_settle`, `on_close_step`, `on_lift_step`, `on_hold`, `on_contact` для label plugins.
6. **Writer.** Из JSON с binary success перейти на chunked HDF5/Zarr для arrays + Parquet для indexes/labels. Использовать schema ideas из SRNO.
7. **Configuration sweep.** Явные protocol/fidelity configs вместо hidden constants.
8. **Resumability.** Shard per object/world и отдельный worker process на batch, чтобы переживать Isaac memory growth и падения.

Существующий `run_validation_pipeline` можно оставить compatibility entry point: он будет view над новым runner с одним observation, одним perturbation и одним fidelity.

## 8. Рекомендуемый scale

Нельзя сразу физически проигрывать все комбинации `object × view × candidate × perturbation × fidelity`. Нужен staged budget.

### Phase 0: schema/pipeline proof

- 28–29 локальных assets;
- 4 nested occlusion levels + clean;
- 128 shared candidates;
- analytic labels для всех;
- physics только для 32 boundary candidates × 4 perturbations;
- 8–12 hand-designed twin families.

Цель: проверить инварианты schema и отсутствие candidate leakage.

### Phase 1: ICLR kill-test corpus

- 300–500 GraspGen meshes, category/geometric stratified;
- 8 observation groups/object;
- 256 candidates/group;
- 32–64 physics candidates/group × 4–8 perturbations;
- 20–50 twin families, 16–64 hidden variants/family;
- 50–100 worlds для multi-fidelity;
- 20–30 physical objects для real paired capture.

Это уже позволяет отбраковать идеи по calibration, set coverage, regret и high-occlusion performance без генерации полного corpus.

### Phase 2: full paper training

- 6,000–7,000 primary train objects;
- 500–1,000 in-domain validation objects;
- 500–1,000 held-out/OOD objects;
- family/category-disjoint splits;
- 8–16 observation groups/object;
- analytic labels для всех 256 candidates;
- physics только для uncertainty/boundary-stratified subset;
- external TARGO/GraspNet test adapters;
- YCB/EGAD/local physical held-out set.

Для 8,515 × 16 × 256 получается около 34.9M candidate rows — нормальный табличный scale. Но полная физика с многими perturbations уже слишком дорога, поэтому active/boundary sampling является частью дизайна, а не экономией после факта.

## 9. Экспериментальный protocol для ICLR

### Обязательные splits

1. **IID asset-disjoint.** Новые экземпляры знакомых categories.
2. **Category-disjoint.** Новые semantic categories.
3. **Geometry-stress.** EGAD evaluation objects.
4. **Twin-family-disjoint.** Ни один базовый shell или hidden morph family не пересекает train/test.
5. **Sensor-domain.** Synthetic→real YCB/local/HouseCat6D-style capture.
6. **External benchmark.** TARGO high-occlusion и, где возможно, GraspNet.

### Метрики, общие для идей

- binary success AP/AUROC и calibration/NLL;
- continuous margin MAE/ranking;
- set coverage и false-safe rate;
- conditional regret/top-k grasp success;
- monotonicity/nesting violations по occlusion filtration;
- robustness across perturbations;
- fidelity convergence/disagreement;
- synthetic→real degradation.

Главная единица bootstrap — `asset_id` или `twin_family_id`, не отдельный grasp row.

### Необходимые baselines

- GraspGen discriminator/generator;
- shape-completion→grasp, включая ZeroGrasp-style baseline;
- GIGA/TARGO-Net там, где поддерживается benchmark protocol;
- observation-only predictor;
- oracle full-mesh upper bound;
- binary-success-only ablation текущего `graspvalidation`.

## 10. Риски для статьи

### Самый опасный риск: synthetic ambiguity без реального аналога

Designed twins позволяют доказать identifiability/set-valued claims, но reviewers могут назвать их искусственными. Поэтому нужны:

- real capture на 3D-printed twin family;
- external high-occlusion TARGO/GraspNet evaluation;
- результаты отдельно для natural objects и designed twins.

### Physics-label validity

PhysX binary success нельзя называть физической истиной. Нужно:

- повторять friction/mass/pose perturbations;
- публиковать signed/continuous outcomes, а не только thresholded label;
- проверить borderline subset в реальности и IPC/GRIP;
- показывать sensitivity к timestep/contact settings.

### Licensing и release

До bulk generation нужен asset-level manifest. GraspGen/Objaverse assets могут иметь разные исходные licenses; ACRONYM labels имеют отдельные условия от ShapeNet meshes. Самый безопасный release:

- публиковать IDs, manifests, split files, derived labels и generation code;
- не перераспространять meshes, если исходная license не разрешает;
- держать YCB/EGAD отдельными evaluation packages;
- не смешивать corpora без deduplication по geometry hash/render similarity.

## 11. Финальная рекомендация

### Минимум для общей тренировки

**Один bulk stack достаточен:**

`GraspGen 8,515-object subset → mesh/USD adapter → текущий Isaac Lab graspvalidation + Astribot → grouped master schema → analytic + selective PhysX + SRNO trajectory labels`.

Нужно добавить generated occluders/twins, camera outputs и candidate replay. Другой основной simulator не нужен.

### Минимум для сильной ICLR статьи

К bulk stack добавить только:

1. **TARGO и GraspNet** как внешние benchmark adapters;
2. **YCB/EGAD/local + 3D-printed twins** как physical/OOD evaluation;
3. **малый real paired micro-motion set** для EdgeFlux/AcqGrasp;
4. **малый IPC-GraspSim/GRIP audit** для compliant/fidelity claims.

### Decision rule

- Если выбранная статья — Blackwell/TQ, DQPL, Cap/Choquet, Fiber, Avo, FIRE/FiGO, JILT, Minty, LFP, InterCap или AcqGrasp: основного Isaac stack достаточно.
- Если статья — EdgeFlux: основной stack достаточен для training, но real paired capture обязателен для ключевого claim.
- Если статья — RelaxGrasp или LimitGrasp: основной stack достаточен для bulk training, но small high-fidelity/physical audit обязателен для claim о contact/fidelity convergence.
- Если цель — одновременно заявить все идеи как один benchmark: выпускать layered dataset с optional tracks, а не утверждать, что один uniform label tensor одинаково точен для всех задач.

Итого: **не искать ещё один готовый grasp dataset, а стандартизовать latent-world/candidate/label contract поверх уже работающего `graspvalidation`.** Это минимальная комбинация с наименьшим переписыванием и одновременно наиболее защищаемый ICLR design.
