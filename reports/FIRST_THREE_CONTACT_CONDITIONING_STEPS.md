# Первые три шага contact conditioning: итоговый отчёт

Дата эксперимента: 17 августа 2026 года.

## 1. Зафиксированная постановка

Во всех сравнениях использован один material-v2 dataset без пересборки:

- object split: 22 train / 3 validation / 3 test;
- active transitions: 62 231 / 9 168 / 9 474;
- manifest SHA-256: `c8bddec752f0418e92383fd0d9193e2d70a37845244c4c1c26ed9f9170c3012a`;
- gripper SHA-256: `6f3280535f5fd5bf543da3dba911825710ea73edb2a19b77b4fd4225fa2f02d6`;
- `length_scale_m = 0.1115`, `sdf_scale_m = 0.02`;
- `delta_gate_m = 0.007934210398234427`;
- PhysX contact envelope: `contact_offset_sum_m = 0.00256`;
- geometric admissibility threshold: `h_admissible = -0.00049232522724196315 m`.

Состояние модели

$$
x_k=(q_k,r_k),\qquad q_k=(R_k,p_k),\qquad r_k\in\mathbb R^6,
$$

где $q$ переводит object coordinates в gripper frame,

$$
x_G=R x_O+p.
$$

Скалярная aperture $A(r)$ оставлена только как derived diagnostic. Геометрия пальцев везде вычисляется из фактических шести joints через FK.

Для collision point $y_i^G(r)$ геометрический gap равен

$$
h_i^{\rm geo}(q,r)
=
\phi\!\left(R^\top(y_i^G(r)-p)\right).
$$

Входной contact signal и gate используют PhysX contact envelope,

$$
h_i^{\rm contact}=h_i^{\rm geo}-2.56\;{\rm mm},
$$

но feasibility loss использует только $h^{\rm geo}$, то есть 2.56 mm из геометрической допустимости не вычитаются.

Оценочная state-distance:

$$
d_X=
\sqrt{
\frac{\|\Delta p\|^2}{L^2}
+\theta(R,R^*)^2
+\frac16\sum_{m=1}^{6}
\left(\frac{\Delta r_m}{s_m}\right)^2
},
$$

где $L$ — gripper length scale, $s_m$ — travel range joint $m$, а

$$
\theta(R,R^*)=
\cos^{-1}\!\left(
\operatorname{clip}\frac{\operatorname{tr}(R^\top R^*)-1}{2},-1,1
\right).
$$

В таблицах ниже:

- **T** — mean $\|\Delta p\|$, m;
- **R** — mean $\theta(R,R^*)$, rad;
- **J** — mean $\sqrt{\frac16\sum_m(\Delta r_m/s_m)^2}$.

Loss, kernel, pooling, hidden dimension и optimizer не менялись:

$$
\mathcal L=\mathcal L_{\rm state}+\lambda_K\mathcal L_K,
\qquad
\lambda_R=\lambda_r=\lambda_K=1,
$$

$$
\mathcal L_K=
\frac1M\sum_i
\left[
\frac{(h_{\rm admissible}-h_i^{\rm geo})_+}{s_{\rm sdf}}
\right]^2.
$$

Использованы AdamW, learning rate $3\cdot10^{-4}$, weight decay $10^{-4}$, clipping 1.0, BF16 cell и float32 geometry/loss. Seeds: 0, 1, 2. Local batches: 4 objects × 256 transitions; rollout batches: 4 objects × 8 trajectories.

## 2. Шаг 1 — rollout `gap` против `gap+J_q`

### 2.1. Добавленная информация

Аналитический metric-gradient SDF вычисляется из тех же восьми voxel corners, что и trilinear value. Вне grid gradient равен нулю. Для ненулевого gradient:

$$
n_i^O=\frac{\nabla\phi(z_i)}{\max(\|\nabla\phi(z_i)\|,10^{-8})},
\qquad
n_i^G=R n_i^O,
\qquad
\rho_i=\frac{y_i^G}{L}.
$$

Left/spatial pose Jacobian contact gap:

$$
J_{q,i}=
\left[-n_i^G,\;-\rho_i\times n_i^G\right]\in\mathbb R^6.
$$

Baseline node feature:

$$
f_i^{\rm gap}=\left[h_i^{\rm contact}/s_{\rm sdf}\right]\in\mathbb R.
$$

Candidate node feature:

$$
f_i^{J_q}=
\left[
h_i^{\rm contact}/s_{\rm sdf},
-n_i^G,
-(\rho_i\times n_i^G)
\right]\in\mathbb R^7.
$$

Изменились только два lifting-слоя `Linear(1,64)` → `Linear(7,64)`. Число параметров: 31 436 → 32 204, то есть +768 (+2.44%). Kernel, pooling и 12-dimensional head не менялись.

### 2.2. Local checkpoints, с которых стартовал rollout

Предшествующий чистый local ablation подтвердил one-step gain:

| Arm | val $d_X$ | test $d_X$ | test T, m | test R, rad | test J |
|---|---:|---:|---:|---:|---:|
| `gap` | 0.031033 | 0.037118 | 0.001396 | 0.006539 | 0.031965 |
| `gap_jq` | 0.020098 | 0.025833 | 0.001232 | 0.006932 | 0.019982 |

Local test paired difference была

$$
E_{\rm test}^{J_q}-E_{\rm test}^{\rm gap}=-0.011286,
$$

с hierarchical bootstrap 95% CI $[-0.012980,-0.009451]$. Validation была лучше во всех трёх seeds.

### 2.3. Чистый autoregressive rollout

Каждый из шести `best-local.pt` независимо продолжен новым rollout optimizer без teacher forcing:

$$
H4\rightarrow H8\rightarrow H16\rightarrow H32.
$$

На каждом horizon использованы максимум 25 epochs и patience 10. Для каждого horizon сохранён собственный best checkpoint.

Средние terminal metrics по трём seeds:

| Arm | H | split | $d_X$ | T, m | R, rad | J |
|---|---:|---|---:|---:|---:|---:|
| `gap` | 4 | val | 0.013552 | 0.001112 | 0.007668 | 0.001603 |
| `gap` | 4 | test | 0.018253 | 0.001636 | 0.008576 | 0.002754 |
| `gap_jq` | 4 | val | 0.013513 | 0.001063 | 0.008562 | 0.001057 |
| `gap_jq` | 4 | test | 0.018150 | 0.001452 | 0.011082 | 0.001801 |
| `gap` | 8 | val | 0.034076 | 0.002489 | 0.021775 | 0.005507 |
| `gap` | 8 | test | 0.044945 | 0.003716 | 0.024852 | 0.008231 |
| `gap_jq` | 8 | val | 0.030339 | 0.002143 | 0.020658 | 0.004416 |
| `gap_jq` | 8 | test | 0.037724 | 0.002704 | 0.025160 | 0.006923 |
| `gap` | 16 | val | 0.065297 | 0.004625 | 0.042990 | 0.014496 |
| `gap` | 16 | test | 0.075581 | 0.005823 | 0.045372 | 0.016200 |
| `gap_jq` | 16 | val | 0.063884 | 0.004335 | 0.045332 | 0.012415 |
| `gap_jq` | 16 | test | 0.074850 | 0.004656 | 0.055140 | 0.013842 |
| `gap` | 32 | val | 0.171902 | 0.013184 | 0.095097 | 0.064209 |
| `gap` | 32 | test | 0.206607 | 0.016564 | 0.104379 | 0.073940 |
| `gap_jq` | 32 | val | 0.197533 | 0.015290 | 0.117959 | 0.056369 |
| `gap_jq` | 32 | test | 0.219773 | 0.016757 | 0.125666 | 0.072085 |

H32 terminal $d_X$ по seeds:

| Seed | `gap` val | `gap_jq` val | difference | `gap` test | `gap_jq` test | difference |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.171675 | 0.194965 | +0.023290 | 0.204097 | 0.218466 | +0.014369 |
| 1 | 0.172902 | 0.196384 | +0.023482 | 0.207260 | 0.221568 | +0.014309 |
| 2 | 0.171129 | 0.201251 | +0.030121 | 0.208465 | 0.219285 | +0.010820 |

Hierarchical bootstrap выполнялся в порядке seed → object → trajectory, 10 000 replicates:

$$
E_{\rm test,H32}^{J_q}-E_{\rm test,H32}^{\rm gap}
=+0.013183,
$$

$$
95\%\;CI=[+0.006825,+0.019440].
$$

Итог: rollout gain **не подтверждён**. `gap_jq` хуже на H32 validation во всех трёх seeds, а test CI целиком положителен. На H4–H16 $J_q$ уменьшает joint и частично translation error, но к H32 rotation error возрастает с 0.10438 до 0.12567 rad и перевешивает этот выигрыш.

## 3. Шаг 2 — условный $J_r$

План разрешал добавлять

$$
[J_{r,i}]_m=(n_i^G)^\top\frac{\partial y_i^G(r)}{\partial r_m}
$$

и feature

$$
f_i=[h_i^{\rm contact}/s_{\rm sdf},J_{q,i},J_{r,i}]\in\mathbb R^{13}
$$

только если $J_q$ проходит строгий H32-критерий. Критерий не выполнен, поэтому этот пункт намеренно пропущен: режим `gap_jq_jr`, его параметры и local runs не добавлялись. Это не отсутствующий результат, а заранее заданное условное решение. Рабочей веткой для следующего шага остался `gap`.

## 4. Шаг 3 — actuator audit

### 4.1. Проверяемый physics contract

До обучения выполнен headless fail-fast read-back всех шести drives одновременно из IsaacLab actuator object и live PhysX tensors/USD:

$$
\text{drive type}=\text{force},\qquad
\text{target type}=\text{position},
$$

$$
K=14,\qquad D=0.35,\qquad
\tau_{\max}=480,\qquad \dot r_{\max}=0.1.
$$

Проверены точный порядок joint names, implicit actuator model, stiffness, damping, effort/velocity limits, USD drive type и runtime position target read-back. Тот же fail-fast audit включён в будущий collector до начала записи trajectories.

Runtime read-back:

| Quantity | Expected | IsaacLab | PhysX runtime | Result |
|---|---:|---:|---:|---|
| stiffness, все joints | 14 | 14 | 14 | pass |
| damping, все joints | 0.35 | 0.349999994 | 0.349999994 | pass |
| effort limit, все joints | 480 | 480 | 480 | pass |
| velocity limit, все joints | 0.1 | 0.1000000015 | 0.1000000015 | pass |
| drive type | force | implicit | force | pass |
| target type | position | position target read-back | position | pass |

Audit sidecar связан с simulator config SHA-256 `04c3bd96f35cf7fe3d038949f70dafda08fb5ce010b74de097341ac4dd8586aa`.

### 4.2. Полная closure trace

Для первого train object `snek-ovoshchnoy-iz-batata-30-g-62734`, source pose 4091, сохранены все 33 settled states:

$$
r,\quad \bar r,\quad \dot r,\quad
\widetilde\tau_{\rm PD}=
\operatorname{clip}
\left(K(\bar r-r)-D\dot r,-\tau_{\max},\tau_{\max}\right).
$$

Все 33 increments settled. Получено:

- maximum $|\dot r|=1.41835\cdot10^{-4}$ rad/s;
- maximum approximate PD effort $=1.51123\cdot10^{-5}$ N·m;
- maximum runtime-vs-settled formula discrepancy $=1.05226\cdot10^{-6}$ N·m при допуске 0.02 N·m.

`robot.data.applied_torque` в sidecar явно назван `approximate_pd_effort`: для implicit PhysX drive это вычисленный clipped PD diagnostic, а не измеренная contact reaction. Старый scalar `actuator_effort=max_m|tau_m|` как вход модели не используется.

## 5. Actuator-conditioning local ablation

Поскольку rollout-$J_q$ не подтвердился, сравнение проведено поверх последней подтверждённой H32 feature-ветки `gap`.

Старая global conditioning:

$$
c_k^{\rm aperture}
=
\left[A(r_k)/L,\;\bar a_{k+1}/L\right]\in\mathbb R^2.
$$

Новая conditioning вычисляется внутри неизменных публичных `forward_step`/`rollout`:

$$
\bar r_{k+1}=R_{\rm free}(\bar a_{k+1}),
\qquad
u_k=\frac{\bar r_{k+1}-r_k}{s}\in\mathbb R^6,
$$

где деление на joint travel $s=(s_1,\ldots,s_6)$ покомпонентное. Это состояние actuator mismatch перед следующим increment, а не torque.

Изменён только первый head layer: вход $64+2\to64+6$, +512 параметров. Полный размер: 31 436 → 31 948. Contact lifting, kernel, pooling, output head, loss и training schedule не менялись. Baseline — замороженные исходные `gap` checkpoints, candidate обучен с нуля для seeds 0, 1, 2.

### 5.1. Результаты

| Arm | split | $d_X$ | T, m | R, rad | J |
|---|---|---:|---:|---:|---:|
| `aperture` | train | 0.036429 | 0.001398 | 0.010630 | 0.027883 |
| `drive_error` | train | 0.018299 | 0.001200 | 0.010289 | 0.005663 |
| `aperture` | val | 0.031033 | 0.001194 | 0.006181 | 0.026288 |
| `drive_error` | val | 0.012832 | 0.000944 | 0.005720 | 0.005822 |
| `aperture` | test | 0.037118 | 0.001396 | 0.006539 | 0.031965 |
| `drive_error` | test | 0.015041 | 0.001117 | 0.006040 | 0.007423 |

Paired $d_X$ по seeds:

| Seed | baseline val | candidate val | difference | baseline test | candidate test | difference |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.031197 | 0.012633 | -0.018564 | 0.037451 | 0.014735 | -0.022716 |
| 1 | 0.031033 | 0.013053 | -0.017980 | 0.037203 | 0.015701 | -0.021503 |
| 2 | 0.030870 | 0.012808 | -0.018062 | 0.036699 | 0.014687 | -0.022012 |

Hierarchical test bootstrap:

$$
E_{\rm test}^{\rm drive\ error}-E_{\rm test}^{\rm aperture}
=-0.022086,
$$

$$
95\%\;CI=[-0.023543,-0.020797].
$$

Итог: local gain **подтверждён** — candidate лучше на validation во всех трёх seeds, а верхняя граница test CI строго ниже нуля. Основной эффект физически ожидаемо находится в joint-компоненте: test J уменьшилась с 0.031965 до 0.007423 (−76.8%). Одновременно улучшились translation (−20.0%) и rotation (−7.6%), поэтому aggregate gain не является только переобозначением joint metric.

После local-result обучение остановлено: дополнительный H32 для actuator arm, $J_t$ и multiplier head не запускались.

## 6. Реализация и проверки

Добавлено:

- differentiable trilinear SDF value + analytic metric-gradient;
- `contact_features = "gap" | "gap_jq"` с backward-compatible default;
- `global_conditioning = "aperture" | "drive_error"` с backward-compatible default;
- per-horizon rollout checkpoints `best-rollout-h04/h08/h16/h32.pt` и совместимый `best-rollout.pt`;
- fail-fast actuator runtime audit и collector integration;
- paired rollout/local experiment runners, hierarchical bootstrap, JSON/NPZ, TensorBoard и графики.

Проверены analytic SDF gradients, anisotropic voxels, boundary/out-of-grid, finite-difference $J_q$, frame/sign convention, zero-gradient without NaN, exact free bypass, permutation invariance, active and 32-step finite gradients, old checkpoint loading, точный знак/нормировка $u_k$, +512 parameter delta и runtime actuator read-back.

Полный результат:

```text
56 passed, 12 warnings in 8.78s
```

Warnings не связаны с изменениями: 11 deprecation warnings из `yourdfpy/trimesh` и один существующий warning PyTorch scheduler в all-free test.

## 7. Артефакты

- Jq rollout: `runs/ablation-jq-rollout/results.json`, `rollout_evaluation.npz`, TensorBoard logs, `jq_rollout_ablation.png`;
- actuator audit: `runs/actuator-audit/results.json`, `actuator_trace.npz`, `actuator_trace.png`;
- actuator local: `runs/ablation-actuator-local/results.json`, `comparison.npz`, per-run evaluations/checkpoints/TensorBoard logs, `actuator_local_ablation.png`.

Главный вывод этих трёх шагов: локальная информация $J_q$ действительно снижает one-step error, но в текущей cell не композируется устойчиво до H32. Поэтому расширение до $J_r$ не было оправдано заданным критерием. В отличие от него, явный actuator mismatch $u_k=(\bar r-r)/s$ дал большой и статистически устойчивый local gain при минимальном изменении head-а.
