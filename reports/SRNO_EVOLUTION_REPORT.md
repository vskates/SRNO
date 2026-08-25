# Эволюция SRNO: от базового contact-resolvent к SRNO-r и clean actuator rollout ablation

Дата последнего обновления отчёта: 22 августа 2026 года.

## 1. Что именно сравнивается

Работа шла не одной серией чистых ablation. В ней были три типа изменений:

1. исправления simulator/data contract: масштаб и позы объектов, collision-геометрия, PhysX contact envelope, actual joints, material binding;
2. изменения представления модели: $a\to r$, затем добавление $J_q$ во вход contact cell;
3. контролируемые ablation: $\lambda_K=0$, gate/feasibility, strong friction, solver/contact-memory, `gap` против `gap_jq` и `aperture` против `drive_error`.

Поэтому ниже явно разделены:

- **чистые ablation**, где менялся один фактор при неизменных данных, seed и обучении;
- **последовательные версии системы**, между которыми одновременно менялись dataset, physics contract или split. Численные различия между такими версиями полезны как инженерная динамика, но не являются причинной оценкой одного изменения.

Главная численная линия получилась такой:

| Версия | Local validation $d_X$ | H32 validation $d_X$ | H32 unseen test $d_X$ | Статус сравнения |
|---|---:|---:|---:|---|
| SRNO-$a$, complete coverage | 0.03560 | 0.41646 | 0.44987 | исходный устойчивый baseline |
| SRNO-$a$, PhysX cooked SDF | 0.03388 | 0.41097 | 0.44044 | точечная SDF/geometry правка |
| SRNO-$r$, interim 11 objects | 0.02515 | 0.29787 | 0.52519 | другой dataset/split; test только 2 объекта |
| SRNO-$r$, material-v2, 28 objects | 0.03120 | 0.17172 | 0.20410 | финальный полный dataset |
| SRNO-$r$+$J_q$, local, 3 seeds | **0.02010** | — | local test **0.02583** | чистый one-step phase |
| SRNO-$r$+$J_q$, rollout, 3 seeds | — | 0.19753 | 0.21977 | clean H4→H32; rollout gain не подтверждён |
| SRNO-$r$+`drive_error`, local, 3 seeds | **0.01283** | — | local test **0.01504** | чистый local actuator ablation |
| SRNO-$r$+`drive_error`, rollout, 3 seeds | — | 0.17933 | 0.21321 | clean H4→H32; H32 gain не подтверждён |

Последние два local-only эксперимента записаны отдельно в разделах 25–26:
history-dependent candidate дал test $d_X=0.029247$, а обучение исходной
L1-архитектуры на smooth subset — $d_X=0.034252$ при оценке на полном test.
Эти числа получены как диагностические ablation. Позднее, 22 августа, по
явному решению data contract smooth-фильтр был принят для основной **local
train/validation supervision**, при этом full test сохранён. Production
архитектура не изменилась, но основной config
[`srno-r-material-v2.toml`](../configs/srno-r-material-v2.toml) теперь указывает
на train/val-filtered index и отдельный output directory. Отдельный дублирующий
retrain после этого переключения не запускался: проверка по каждой паре
`(object, trajectory, step)` установила, что его train/val supervision в
точности совпадает с уже завершённым `ablation-local-no-pose-jumps`. Поэтому
численные строки выше являются применимым controlled result этого data
contract, но checkpoint по-прежнему хранится под именем диагностического run.

От SRNO-$a$+PhysX-SDF до полного SRNO-$r$+material-v2 test H32 уменьшился с $0.44044$ до $0.20410$, то есть на 53.66%. Это суммарный эффект нового state/geometry representation, нового датасета, исправленного материала и нового split, а не чистый эффект только $a\to r$.

## 2. Обозначения и соглашения

### 2.1 Состояние и системы координат

- $O$ — object frame;
- $G$ — gripper frame;
- $q=(R,p)\in SE(3)$ — pose объекта, переводящий object coordinates в gripper coordinates:

$$
x_G=R x_O+p;
$$

- $R\in SO(3)$, $p\in\mathbb R^3$;
- $a$ — scalar actual aperture;
- $\bar a_k$ — commanded aperture на шаге $k$;
- $r=(r_1,\ldots,r_6)\in\mathbb R^6$ — actual PhysX joint configuration gripper;
- $A(r)$ — derived scalar aperture diagnostic;
- $s_m$ — полный free travel range joint $m$;
- $L$ — gripper length scale; в финальном asset $L=0.1114999652$ m;
- $M=256$ — число collision surface samples gripper;
- $d=64$ — hidden dimension contact cell;
- $\phi(x_O)$ — SDF объекта, positive outside;
- $s_{\rm sdf}=0.02$ m — SDF normalization scale;
- $\delta_{\rm gate}$ — threshold вызова neural cell;
- $d_c$ — суммарный PhysX contact envelope;
- $h_{\rm admissible}$ — допустимый residual penetration для geometric feasibility.

Кватернионы на диске имеют порядок XYZW и нормализуются loader-ом.

### 2.2 Геодезическая ошибка rotation

Для $R_1,R_2\in SO(3)$:

$$
\theta(R_1,R_2)
=\cos^{-1}\!\left(\operatorname{clip}\frac{\operatorname{tr}(R_2^T R_1)-1}{2},-1,1\right).
$$

В коде используется устойчивая эквивалентная форма через `atan2(sin, cos)`.

### 2.3 Left/spatial $SE(3)$ update

Пусть twist упорядочен как $\Delta\xi=(v,\omega)$. При
$\Omega=[\omega]_\times$, $\vartheta=\|\omega\|$:

$$
R_\Delta=I+\frac{\sin\vartheta}{\vartheta}\Omega
+\frac{1-\cos\vartheta}{\vartheta^2}\Omega^2,
$$

$$
V=I+\frac{1-\cos\vartheta}{\vartheta^2}\Omega
+\frac{\vartheta-\sin\vartheta}{\vartheta^3}\Omega^2,
\qquad p_\Delta=Vv.
$$

Настоящий left/spatial update:

$$
R'=R_\Delta R,\qquad p'=R_\Delta p+p_\Delta.
$$

Около нуля коэффициенты вычисляются рядами Тейлора.

### 2.4 Метрики

Для SRNO-$r$:

$$
e_T=\|\hat p-p^*\|_2,
\qquad e_R=\theta(\hat R,R^*),
$$

$$
e_J=\sqrt{\frac{1}{6}\sum_{m=1}^{6}
\left(\frac{\hat r_m-r_m^*}{s_m}\right)^2},
$$

$$
\boxed{
d_X=\sqrt{\frac{e_T^2}{L^2}+e_R^2+e_J^2}
}.
$$

Для первоначального SRNO-$a$ joint term заменялся на

$$
e_a/L=\frac{|\hat a-a^*|}{L}.
$$

Дополнительные rollout-метрики:

$$
e_A=|A(\hat r)-a^*|,
\qquad
e_{\rm lag}=|[(A(\hat r)-\bar a)-(a^*-\bar a)]|,
$$

$$
k_{\rm onset}(a)=\min\{k:a_k-\bar a_k>\delta_{\rm gate}\},
$$

а terminal metrics вычисляются только на $k=32$. Reported penetration использует raw geometric gap:

$$
P_{k,i}=(-h^{\rm geom}_{k,i})_+.
$$

## 3. Simulator и исходный dataset contract

### 3.1 Перенос validation assets и проверка масштаба

Сначала был восстановлен именно simulator-facing validation pipeline, а не generation/visualization pipeline:

- gripper — фактически используемый `gripper_playground.usd`, а source URDF/STL оставлены только для provenance и inspection;
- object membership, initial poses и spawn transforms взяты из активного validation config;
- runtime dependency на `vv_assets` удалена: нужные USDC и textures vendored в `assets/`;
- OBJ/STL-дубликаты объектов не используются, потому что generation export содержит дополнительное coordinate conversion и не равен физически spawned asset;
- grocery USDC уже Z-up и уже нормализованы authored root transform-ом; внешний spawn scale остаётся unit/не задаётся, а authored root scale сохраняется при извлечении collision geometry;
- validation-successful pose seeds сохранены локально; для final collection берутся 100 poses на объект.

Именно потеря authored USD transform сначала делала бутылку несопоставимо большой относительно gripper. После перехода на полный `ComputeLocalToWorldTransform` размеры SDF/collider и визуально spawned объекта совпали; тот же путь применяется автоматически ко всем catalog objects.

Проверено также, что validation approach axis — локальная `+Z` gripper pose, а pregrasp расположен по локальной `−Z`. В SRNO collection approach не воспроизводится: без shelf он до начала closure толкал бы свободный объект. Ещё одна намеренно сохранённая особенность исходного validation: quaternion `(0,0,0,1)`, названный XYZW, фактически передаётся Isaac как WXYZ, поэтому runtime spawn содержит поворот на 180° вокруг Z; catalog хранит именно фактический WXYZ transform.

### 3.2 Как записывалась trajectory

Collector повторяет validation-generation pipeline для asset selection, spawn transforms и initial grasp poses, но физическая постановка была специально упрощена:

- gravity $=0$;
- shelf отсутствует;
- объект не фиксируется ни по $z$, ни по другим координатам;
- approach phase не используется;
- gripper закрывается 32 малыми command increments;
- записываются начальное состояние и 32 последующих settled states, всего 33;
- состояние записывается не через фиксированное wall-clock время, а после выполнения velocity/position settling criteria;
- candidates, которые не выполнили settling criteria за 2400 control steps,
  не записываются как валидные trajectories и заменяются следующими
  deterministic pose candidates;
- после collection из **train и validation local supervision** дополнительно
  исключаются физически settled, но редкие pose/contact jumps
  $d_{\Delta q}>0.05$; удаляются ровно отдельные transition indices, а не
  целые trajectories, поскольку jumps встречаются в 44.6% train и 29.0% val
  trajectories;
- test split остаётся полным, включая jumps, чтобы фильтр обучения не
  искусственно улучшал итоговую оценку;
- object SDF хранится один раз на объект, а не копируется на trajectories.

Первый полный набор содержал 29 объектов $\times$ 100 trajectories:

$$
2900\text{ trajectories},\qquad 2900\times32=92\,800\text{ transitions}.
$$

Object-wise split был 23/3/3. Позже `ogurtsy-marinovannye-670-g-21054` был полностью удалён из catalog, assets и данных. Финальный набор содержит 28 объектов, 2800 trajectories и split 22/3/3.

### 3.3 HDF5

На объект сохраняются:

- `sdf[96,96,96]`, float16 на диске;
- origin центра нулевого voxel и anisotropic voxel size;
- `position[T,33,3]`;
- `quaternion_xyzw[T,33,4]`;
- `actual_aperture[T,33]`;
- после $a\to r$: `joint_position[T,33,6]` и фиксированный порядок joint names;
- optional diagnostics на 32 transitions: contact count, effort, max penetration, residual velocities и settling substeps.

Финальные шесть joints:

1. `astribot_gripper_right_joint_L1`;
2. `astribot_gripper_right_joint_L2`;
3. `astribot_gripper_right_joint_R1`;
4. `astribot_gripper_right_joint_R2`;
5. `astribot_gripper_right_joint_L11`;
6. `astribot_gripper_right_joint_R11`.

Lazy HDF5 loader открывает отдельные read-only handles в каждом worker. Object-grouped batches загружают SDF один раз и сопоставляют ему несколько transitions/trajectories через `sample_to_object`. Все simulator entry points имеют отдельный RAM watchdog; для collection использовался предел 14 GiB.

### 3.4 Наблюдаемость обучения

TensorBoard пишет все runs в `runs/`. User-level systemd service `srno-tensorboard.service` включён и активен, слушает только `127.0.0.1:6006`, автоматически перезапускается после сбоя. Для пользователя включён `Linger=yes`, поэтому service поднимается и после reboot без интерактивного login.

## 4. Самая первая архитектура: SRNO-$a$

### 4.1 Operator

Изначально моделировался shared quasistatic operator

$$
\mathcal R_\phi:
(q_k,a_k,\bar a_{k+1},\phi)
\longmapsto(q_{k+1},a_{k+1}).
$$

Free predictor:

$$
\tilde q_{k+1}=q_k,
\qquad
\tilde a_{k+1}=\bar a_{k+1}.
$$

256 nominal gripper points выбирались scalar aperture:

$$
y_i^G(a)=c_i+s_i a.
$$

Для trial pose:

$$
z_i^O=R_k^T(y_i^G(\tilde a_{k+1})-p_k),
\qquad
h_i=\phi(z_i^O).
$$

### 4.2 Exact free bypass

$$
\mathrm{active}_k=mathbf 1\!\left[min_i h_i\le\delta_{\rm gate}\right].
$$

Если sample не active, возвращается в точности trial state и neural cell не вызывается. Это даёт machine-exact free motion.

### 4.3 Integral contact cell

Node и position features:

$$
e_i=\frac{h_i}{s_{\rm sdf}},
\qquad
\rho_i=\frac{y_i^G}{L}.
$$

Один diagonal-kernel integral layer:

$$
z_i^{\rm lat}
=\operatorname{SiLU}\!\left(
W_0e_i+\frac{1}{M}\sum_{j=1}^{M}
\kappa(\rho_i,\rho_j)\odot W_1e_j+b
\right),
$$

где $\kappa$ — `MLP(6→64→64)`, а $\odot$ — component-wise product. После mean pooling:

$$
\bar z=\frac1M\sum_i z_i^{\rm lat}.
$$

Head `MLP(66→128→128→7)` получает $\bar z$, $a_k/L$ и $\tilde a_{k+1}/L$ и выдаёт

$$
(\widehat{\Delta v},\Delta\omega,\eta)\in\mathbb R^7.
$$

Физический twist:

$$
\Delta\xi=(L\widehat{\Delta v},\Delta\omega),
\qquad
\hat q_{k+1}=\Exp(\widehat{\Delta\xi})q_k,
$$

$$
\hat a_{k+1}
=\tilde a_{k+1}
+\sigma(\eta)(a_k-\tilde a_{k+1}).
$$

Последний layer был инициализирован нулевым motion, а bias $\eta=-4$.

### 4.4 Loss и обучение

Первоначальная state loss:

$$
\mathcal L_{\rm state}
=\frac{\|\hat p-p^*\|^2}{L^2}
+\lambda_R\theta(\hat R,R^*)^2
+\lambda_a\frac{(\hat a-a^*)^2}{L^2}.
$$

Изначальная feasibility:

$$
\mathcal L_K
=\frac1M\sum_{i=1}^{M}
\left[
\frac{(0-h_i^{\rm pred})_+}{s_{\rm sdf}}
\right]^2,
$$

$$
\mathcal L=\mathcal L_{\rm state}+\lambda_K\mathcal L_K,
\qquad
\lambda_R=\lambda_a=\lambda_K=1.
$$

Режимы:

- local: ground-truth state $x_k^*$, active one-step transitions;
- rollout: autoregressive без teacher forcing;
- curriculum $H4\to H8\to H16\to H32$.

Общие optimizer defaults: AdamW, learning rate $3\cdot10^{-4}$, weight decay $10^{-4}$, clipping 1.0, BF16 neural cell на CUDA и float32 geometry/$SE(3)$/loss.

## 5. Первые обучения SRNO-$a$

Первая конфигурация использовала маленькие chunks по 8 samples/trajectory на object. Затем были отдельно проверены загрузка всего object целиком и эффективный complete-coverage sampler. В окончательном sampler `4 objects × 256 local transitions` и `4 objects × 8 rollout trajectories` — это **размер minibatch chunk, а не ограничение dataset**: за эпоху каждый active transition/trajectory посещается ровно один раз без replacement.

Лучшие validation checkpoints:

| Run | Local | H4 | H8 | H16 | H32 |
|---|---:|---:|---:|---:|---:|
| `srno-contact-v1` | 0.018214 | 0.025354 | 0.084209 | 0.244301 | 0.431521 |
| `srno-contact-v1-curriculum` | 0.018214 | 0.025354 | 0.082244 | 0.241875 | 0.434877 |
| `srno-contact-v1-full-data` | 0.032984 | **0.012107** | **0.060771** | 0.236074 | 0.416735 |
| `srno-contact-v1-full-coverage` | 0.035601 | 0.012137 | 0.060880 | **0.233099** | **0.416463** |

Unseen test финальных H32 checkpoints:

| Run | Terminal $d_X$ | Mean translation | Mean rotation | Mean aperture | Mean/max penetration |
|---|---:|---:|---:|---:|---:|
| `srno-contact-v1` | 0.455406 | 0.110412 $L$ | 0.195439 rad | 0.010724 $L$ | 0.756 / 33.670 mm |
| `srno-contact-v1-curriculum` | 0.458531 | 0.116206 $L$ | 0.195809 rad | 0.010724 $L$ | 0.890 / 33.857 mm |
| `srno-contact-v1-full-data` | 0.454601 | 10.754 mm | 0.195676 rad | 1.050 mm | 0.622 / 33.750 mm |
| `srno-contact-v1-full-coverage` | **0.449868** | **9.921 mm** | 0.197273 rad | **0.832 mm** | **0.458 / 33.448 mm** |

Для `full-coverage` terminal physical errors были 19.045 mm translation, 0.383784 rad rotation и 2.944 mm aperture. Этот run стал основным SRNO-$a$ baseline.

Local checkpoint первого run при прямом H32 rollout дал test terminal $d_X=0.502095$, тогда как recurrently trained checkpoint дал 0.455406. Это подтвердило необходимость autoregressive curriculum, но разрыв local/rollout остался большим.

## 6. Чистый ablation $\lambda_K=0$

Менялось только

$$
\lambda_K:1\longrightarrow0,
\qquad
\mathcal L=\mathcal L_{\rm state}.
$$

Dataset, sampler, architecture, curriculum, seed и optimizer были одинаковы с `full-coverage`.

Validation best per horizon:

| Horizon | $\lambda_K=1$ | $\lambda_K=0$ | Изменение |
|---:|---:|---:|---:|
| H4 | 0.012137 | 0.012642 | +4.16% |
| H8 | 0.060880 | 0.061440 | +0.92% |
| H16 | 0.233099 | 0.230088 | −1.29% |
| H32 | 0.416463 | 0.382794 | −8.08% |

Однако unseen test curve финальных H32 checkpoints:

| Step | $\lambda_K=1$ | $\lambda_K=0$ | Изменение |
|---:|---:|---:|---:|
| 4 | 0.034832 | 0.041214 | +18.32% |
| 8 | 0.109347 | 0.113769 | +4.04% |
| 16 | 0.232721 | 0.231004 | −0.74% |
| 32 | 0.449868 | 0.459616 | +2.17% |

При

$$
\operatorname{slope}_{[a,b]}
=\frac{d_X(b)-d_X(a)}{b-a},
$$

slopes baseline/candidate были: 0–4: 0.008708/0.010304; 4–8: 0.018629/0.018139; 8–16: 0.015422/0.014654; 16–32: 0.013572/0.014288.

Mean penetration выросла 0.458→0.913 mm, почти на 99.3%. Вывод: validation H32 выигрыш без feasibility не перенёсся на unseen test, а физическая допустимость ухудшилась.

![Ablation lambda K](../runs/srno-contact-v1-lambda-k0/comparison/ablation_dashboard.png)

## 7. Диагностика исходного rollout bottleneck

На `full-coverage/best-rollout.pt` был выполнен большой diagnostic point 16 без изменения weights.

### 7.1 Несогласованность feasibility с GT

Из 95,700 GT states:

- 67.54% имели $\min h<0$;
- 61.58% — глубже 1 mm;
- 40.56% — глубже 5 mm;
- median $\min h=-2.85$ mm;
- минимум $-36.94$ mm.

На 3041 violating GT states gradient feasibility был ненулевым в 100% случаев, median norm 0.0716. То есть simulator GT не являлся stationary point текущей $\mathcal L_K$.

### 7.2 Gradient conflict

- stall batches: median $\cos(g_{\rm state},g_K)=-0.324$, 68.75% отрицательных, median $\|g_K\|/\|g_{\rm state}\|=4.56$;
- sliding: median cosine +0.172, 34.38% отрицательных;
- near-contact: 81.25% batches имели нулевой feasibility gradient.

### 7.3 Teacher-forced против pushforward

Среднее отношение ошибок

$$
\Gamma_{\rm PF}
=\frac{E_{\rm pushforward}}{E_{\rm teacher}}
$$

равнялось 1.833 train, 1.812 val и 1.821 test. Pushforward был хуже в 93.79%, 96.34% и 93.88% matched samples.

### 7.4 Локальное усиление

Для конечных directional perturbations translation amplification имел median 0.991, только 2.85% >1.05, max 1.64. Rotation: median 1.000, 3.09% >1.05, max 3.26. Aperture была contractive: median 0.0895.

Это указало не на повсеместно взрывающийся map, а на почти неконтрактное накопление pose error с редким local expansion.

### 7.5 Shape split и state ambiguity

Float32 H32 terminal $d_X$: train shapes 0.476, val 0.411, test 0.452. Unseen shape не был единственным separator.

Среди 70,438 contact transitions ближайший 1% neighbours имел median output-correction distance 0.00362, p95 0.0373; 3.12% были >0.05, max 0.198. При этом 42 exact duplicates были согласованы: mean 0.000123, max 0.00126. Явного глобального non-Markov collapse это не доказало.

![Point-16 diagnostics](../runs/srno-contact-v1-full-coverage/diagnostics-point16/diagnostic_dashboard.png)

## 8. Исправление SDF и PhysX geometry

### 8.1 Найденное расхождение

Первый SDF строился по authored visual/raw mesh, тогда как simulator сталкивал объект через cooked PhysX `convexDecomposition`. Для gripper nominal points также не совпадали с runtime collision hull/contact envelope.

Диагностика 29 объектов и 92,800 trial transitions показала:

- только 3/29 raw meshes были watertight/volume meshes;
- raw surface→PhysX cooked surface: median 2.125 mm, mean 2.233 mm, max 3.716 mm;
- contact-relevant stored-vs-cooked trial gap absolute error: mean 2.587 mm, median 2.031 mm, p95 6.823 mm, p99 12.981 mm;
- runtime gripper-vs-model trial minimum difference: mean 3.480 mm, median 3.993 mm, p95 7.724 mm, p99 8.627 mm;
- active/inactive disagreement при manifest gate: 2.3136%.

### 8.2 Точечная правка

SDF стал строиться по cooked convex-decomposition collision geometry, а 256 gripper samples — по collision hulls контактных links.

PhysX создаёт контакт до пересечения геометрических поверхностей. Суммарный envelope:

$$
d_c=d_{\rm object}+d_{\rm finger}
=2.00\text{ mm}+0.56\text{ mm}
=2.56\text{ mm}.
$$

Были разделены два сигнала:

$$
h_i^{\rm geom}=\phi_{\rm cooked}(z_i),
$$

$$
\boxed{h_i^{\rm contact}=h_i^{\rm geom}-d_c}.
$$

`contact signal` и gate используют $h^{\rm contact}$, но geometric feasibility использует **только** raw $h^{\rm geom}$. Вычитание 2.56 mm из feasibility было бы двойным учётом PhysX contact envelope.

### 8.3 Калибровка gate

Для simulator contact label $C_k=\mathbf1[\texttt{contact_count}_k>0]$ и trial minimum $m_k=\min_i h_{k,i}^{\rm contact}$:

$$
\operatorname{Recall}(\delta)
=\frac{\sum_k C_k\mathbf1[m_k\le\delta]}{\sum_k C_k}.
$$

Калибратор выбирает

$$
\delta_{\rm gate}
=\max\left(
Q_{0.995}(m_k\mid C_k=1),
2.01v_{\max},
\epsilon_{\rm fp32}
\right).
$$

Первый threshold старого geometry contract был 8.2841665 mm и давал train+val recall 99.5002%. После cooked-object geometry contact-only threshold был 4.4967 mm; при runtime gripper geometry — 1.5071 mm. Но эти значения были меньше conservative two-voxel floor. Для 96³ grids с $v_{\max}=3.94737$ mm production gate стал 7.93421 mm.

### 8.4 Retrain после одной SDF правки

`srno-contact-v1-physx-sdf` сохранил SRNO-$a$:

| Local | H4 | H8 | H16 | H32 val | H32 test |
|---:|---:|---:|---:|---:|---:|
| 0.033883 | 0.012445 | 0.061335 | 0.232829 | 0.410975 | 0.440439 |

Против `full-coverage` test H32 улучшился 0.449868→0.440439, на 2.10%. Terminal translation 19.045→18.291 mm, terminal rotation 0.383784→0.376633 rad, max penetration 33.448→30.820 mm. Улучшение было реальным, но небольшим: geometry mismatch не объяснял весь rollout error.

## 9. Contact-manifold ablation

После SDF correction были отдельно исследованы gate и feasibility boundary.

### 9.1 Correction-aware gate

Необходимость correction определялась через free predictor:

$$
c_k=d_X(x_{k+1}^*,F_{\rm free}(x_k^*)),
$$

$$
\tau_{\rm num}=Q_{0.995}(c_k\mid\text{no simulator contact})
=0.00765149,
$$

$$
y_k^{\rm corr}=\mathbf1[c_k>\tau_{\rm num}].
$$

Минимальный threshold с recall не ниже 99.5%:

$$
\delta_{\rm corr}=1.607985\text{ mm}.
$$

Train/val/test recall: 99.459%/99.825%/99.838%; precision: 91.973%/89.269%/91.246%. Active index уменьшился 82,548→72,514 transitions, то есть 88.95%→78.14% полного набора.

### 9.2 Feasibility stress boundary

Из-за неправильных GT gaps была проверена формальная robust boundary:

$$
h_{\rm admissible}
=-Q_{0.995}((-h_{\rm GT})_+)
=-31.241722\text{ mm}.
$$

Это был diagnostic stress value, не физическая константа.

### 9.3 Четыре arms

| Arm | $\delta_{\rm gate}$ | $h_{\rm admissible}$ | Local | H4 | H8 | H16 | H32 val |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 7.934 mm | 0 | 0.03651 | 0.01084 | 0.05750 | 0.21725 | 0.39270 |
| gate-only | 1.608 mm | 0 | 0.03845 | **0.00989** | 0.05441 | 0.21713 | 0.39667 |
| feasibility-only | 7.934 mm | −31.242 mm | **0.02274** | 0.01134 | 0.05490 | 0.21826 | **0.34971** |
| combined | 1.608 mm | −31.242 mm | 0.02492 | **0.00987** | **0.05321** | **0.21466** | 0.35256 |

Unseen test $d_X(k)$:

| Arm | H4 | H8 | H16 | H32 |
|---|---:|---:|---:|---:|
| baseline | 0.02777 | 0.09919 | 0.22321 | 0.44144 |
| gate-only | **0.02478** | **0.09572** | **0.22080** | 0.44040 |
| feasibility-only | 0.04982 | 0.12641 | 0.24578 | **0.43941** |
| combined | 0.03013 | 0.10510 | 0.23258 | 0.44110 |

Все paired H32 confidence intervals включали zero. `gate-only` устойчиво улучшил H4/H8/H16, но не H32. Relaxed feasibility улучшал local fit и validation H32, но не generalization; mean penetration выросла 0.609→0.800 mm (`feasibility-only`) и 0.839 mm (`combined`). Это навело на root cause: scalar $a$ не восстанавливает фактические poses links под контактом.

![Contact-manifold ablation](../runs/contact-manifold-ablation/comparison/ablation_dashboard.png)

## 10. Переход от scalar aperture к actual joints: SRNO-$r$

### 10.1 Прямая geometric диагностика

На 3 объектах, 100 trajectories на объект и 9600 states сравнивались:

$$
m_a^*=\min_i\phi\!\left((q^*)^{-1}y_i^G(a^*)\right),
$$

$$
m_r^*=\min_i\phi\!\left((q^*)^{-1}T_{\ell(i)}(r^*)y_i^{\ell(i)}\right).
$$

Результаты all states:

| Показатель | nominal aperture geometry | actual-joint FK |
|---|---:|---:|
| Mean minimum gap | −4.442 mm | +1.515 mm |
| Median minimum gap | −1.329 mm | +0.222 mm |
| Minimum | −41.699 mm | −1.149 mm |
| Mean geometric penetration | 5.974 mm | **0.0589 mm** |
| Penetration >0.5 mm | 51.927% | **0.906%** |
| Penetration >1 mm | 50.750% | **0.0104%** |
| Penetration >2 mm | 48.583% | **0%** |

На simulator-contact states mean penetration упала 7.364→0.0725 mm. Free-schedule FK point replay error был $7.87\cdot10^{-8}$ m mean и $1.68\cdot10^{-7}$ m max; replayed joint-position error был ровно zero.

Новая geometric boundary:

$$
h_{\rm admissible}^{\rm geom}=-0.551733\text{ mm}
$$

на diagnostic subset. В effective contact coordinates это было бы $-3.111733$ mm, но feasibility намеренно осталась geometric и использовала первое значение.

### 10.2 Новое состояние и geometry

Operator стал

$$
\mathcal R_\phi:(q_k,r_k,\bar a_{k+1},\phi)
\longmapsto(q_{k+1},r_{k+1}).
$$

Каждый collision sample привязан к link $\ell(i)$:

$$
y_i^G(r)=T_{\ell(i)}(r)y_i^{\ell(i)},
$$

$$
z_i^O(q,r)=R^T(y_i^G(r)-p),
\qquad
h_i^{\rm geom}(q,r)=\phi(z_i^O(q,r)).
$$

Эта одна и та же FK geometry используется для input gaps, gate и feasibility.

### 10.3 Free predictor

В empty gripper были записаны 33 joint knots:

$$
\bar a_j\longmapsto r_j^{\rm free}.
$$

Lookup/interpolation задаёт

$$
\tilde r_{k+1}=R_{\rm free}(\bar a_{k+1}),
\qquad
\tilde x_{k+1}=(q_k,\tilde r_{k+1}).
$$

Ничего в $R_{\rm free}$ не обучается.

### 10.4 Derived aperture

Пусть $r_o=r^{\rm free}(a_{\max})$, $r_c=r^{\rm free}(a_{\min})$, $d=r_c-r_o$. Тогда

$$
\chi(r)=\operatorname{clip}\left(
\frac{(r-r_o)^Td}{d^Td},0,1
\right),
$$

а $A(r)$ получается интерполяцией по 33 aperture knots в позиции $32\chi(r)$. Это только diagnostic/conditioning scalar; collision geometry никогда больше не выбирается через $A(r)$.

### 10.5 Новый head и loss

Integral layer, $M=256$, $d=64$, kernel и pooling не менялись. Head стал `MLP(66→128→128→12)`:

$$
(\widehat{\Delta\xi},\widehat{\Delta r})\in\mathbb R^{6+6}.
$$

Joint output нормализован travel range:

$$
\Delta r^c=s\odot\widehat{\Delta r},
\qquad
\hat r_{k+1}=\tilde r_{k+1}+\Delta r^c.
$$

State loss:

$$
\mathcal L_{\rm state}
=\frac{\|\hat p-p^*\|^2}{L^2}
+\lambda_R\theta(\hat R,R^*)^2
+\lambda_r\frac16\sum_{m=1}^{6}
\left(\frac{\hat r_m-r_m^*}{s_m}\right)^2.
$$

Feasibility:

$$
\mathcal L_K
=\frac1M\sum_i
\left[
\frac{(h_{\rm admissible}-h_i^{\rm geom}(\hat q,\hat r))_+}
{s_{\rm sdf}}
\right]^2.
$$

### 10.6 Interim обучение на 11 объектах

Dataset: 8 train / 1 val / 2 test objects, по 100 trajectories; $\delta_{\rm gate}=7.07507$ mm, $h_{\rm admissible}=-0.512322$ mm.

| Local | H4 | H8 | H16 | H32 val | Final val eval | Final test eval |
|---:|---:|---:|---:|---:|---:|---:|
| 0.025153 | 0.035326 | 0.124717 | 0.205775 | 0.297874 | 0.297862 | 0.525187 |

Val all-state errors: 10.633 mm translation, 0.137161 rad rotation, 0.02536 normalized joint absolute error. Test: 15.621 mm, 0.201134 rad, 0.01788. Terminal test: 31.351 mm, 0.415729 rad, joint RMSE 0.03618, aperture 1.259 mm.

На H32 validation flow contribution распределялся примерно как translation 26.5%, rotation 70.1%, joints 3.4%. На test terminal squared state distance: около 68% rotation, 31% translation и <1% joints. Значит, $r$-часть уже не была главным bottleneck; оставались object-pose prediction и autoregressive composition.

## 11. Проверка Markov/state sufficiency и contact memory

### 11.1 Preserve history против fresh reset

Для 72 contact/stall states на 3 объектах сравнивались два successor:

$$
x_{k+1}^{\rm preserve}
=F_{\rm PhysX}(x_k,\bar a_{k+1};\text{сохранённая solver history}),
$$

$$
x_{k+1}^{\rm fresh}
=F_{\rm PhysX}(x_k,\bar a_{k+1};\text{новая scene, только }q_k,r_k).
$$

$$
D_{\rm reset}=d_X(x_{k+1}^{\rm preserve},x_{k+1}^{\rm fresh}).
$$

Результат: mean 0.023275, median 0.004157, p90 0.042836, p95 0.123605, max 0.361155. Fresh repeatability mean была $9.63\cdot10^{-5}$. Mean error лучшей local model на тех же samples был 0.037400; reset/model ratio 0.622. В 19.44% samples reset error был не меньше global local validation error.

Вывод: hidden solver/contact state существует и создаёт тяжёлый tail, но его средний масштаб меньше model error, поэтому это не единственный bottleneck.

### 11.2 Factorization tests

**P0a, velocities.** Production preserve уже обнулял object/joint velocities перед increment; дополнительный zero write ничего не менял, $D_v\equiv0$. Paired continuous repeatability при этом имела mean $d_X=0.013784$, median 0.001758.

**P0b, strong-friction flag.** В первоначальном Python schema setter не был доступен; вместо неэквивалентной подмены тест был отмечен unavailable. Позже native PhysX probe позволил выполнить чистый test: `eDISABLE_STRONG_FRICTION=true` при неизменных коэффициентах. Mean reset error 0.023275→0.022202, ratio 0.9539, улучшились ровно 50% samples. Strong friction не оказался root cause.

**P0c, PCM→SAT.** При `physxScene:collisionSystem=SAT` mean reset error вырос 0.023275→0.655427, median до 0.251247, max до 10.4815. Translation error достигал 1.169 m. Отключение PCM таким способом разрушало режим симуляции и было отвергнуто.

**Fresh preconditioning.** Fresh scene сначала equilibrate-илась на текущей command, затем получала следующую. Mean error стал 0.024501 против cold 0.023275; ratio 1.05266, улучшились только 36.11%. Drift, внесённый preconditioning, сам имел mean $d_X=0.017020$. Это не исправило hidden-state effect.

## 12. Исправление contact material и material-v2 dataset

### 12.1 Баг

Intended coefficients

$$
\mu_s=2.4,
\qquad
\mu_d=2.0
$$

присутствовали как замысел, но не были надёжно bound и read-back из фактического `PxMaterial`. Поэтому старый dataset нельзя было считать собранным при зафиксированной friction law. Точное прежнее effective material значение не было сертифицировано, поэтому в отчёте оно намеренно не придумывается.

### 12.2 Формальный material contract

Для sticking/sliding Coulomb law:

$$
\|f_t\|\le\mu_s f_n
\quad\text{(sticking)},
$$

$$
f_t=-\mu_d f_n\frac{v_t}{\|v_t\|}
\quad\text{(sliding)}.
$$

Финальная конфигурация:

- $\mu_s=2.4$, $\mu_d=2.0$;
- friction/restitution combine mode `min`;
- restitution 0;
- rigid contact, $k_c=0,c_c=0$;
- patch friction, PCM contact generation;
- strong friction enabled;
- TGS solver;
- 64 position и 16 velocity iterations;
- Isaac Sim 5.1.0.0 / PhysX runtime.

Один material явно bind-ится к объектам и finger contact pads. После запуска native probe читает фактический `PxMaterial`; collector падает, если coefficients, combine modes или flags не совпадают. Полный physics contract записывается в schema-v2 manifest и HDF5 metadata.

SDF/FK из-за material bug не менялись.

### 12.3 Финальный dataset и повторная калибровка

После исправления материала заново собраны 28 объектов $\times$ 100 trajectories, split 22/3/3, с actual joints. На train+val было 80,000 settled transitions:

- simulator contacts: 58,940;
- free transitions: 21,060;
- $v_{\max}=3.947368$ mm;
- $\delta_{\rm gate}=7.934210$ mm;
- contact recall 100%;
- free false-positive rate 59.16% из-за conservative two-voxel floor;
- $h_{\rm admissible}^{\rm geom}=-0.492325$ mm.

Последнее значение вычислено как

$$
h_{\rm admissible}^{\rm geom}
=-Q_{0.995}\left((-h_{\rm GT}^{\rm geom})_+\right).
$$

Геометрический contact gate сначала даёт полный active index:

| Split | Active transitions |
|---|---:|
| train | 62,231 |
| val | 9,168 |
| test | 9,474 |
| всего | 80,873 |

После последующего решения использовать smooth train/validation supervision
ground-truth transition помечается jump при

$$
d_{\Delta q,k}=
\sqrt{(\|p_{k+1}-p_k\|/L)^2+d_{SO(3)}(R_k,R_{k+1})^2}>0.05,
\qquad L=0.1114999652\ {\rm m}.
$$

Итоговый production local index имеет следующий contract:

| Split | До фильтра | Найдено jumps | Удалено | После фильтра |
|---|---:|---:|---:|---:|
| train | 62,231 | 2,638 (4.24%) | 2,638 | **59,593** |
| val | 9,168 | 135 (1.47%) | 135 | **9,033** |
| test | 9,474 | 217 (2.29%) | 0 | **9,474** |

Удалённые train/val transitions несут соответственно 91.47% и 69.28%
squared pose-motion energy своих splits. Полные HDF5 trajectories сохраняются:
физическое удаление одного state склеило бы соседние состояния в ложный
transition. Фильтр применяется через
`active-index-train-val-no-pose-jumps.npz`; точный hash, threshold и per-object
counts записаны в `pose-jump-filter-contract.json`. Старый `active-index.npz`
сохранён как immutable полный geometric-contact index для audit и full-test
diagnostics.

Следующие эксперименты с этим production index являются **local-only**.
Rollout retrain не планируется, поэтому отдельная сегментация trajectories на
smooth подпути не вводится: непрерывные HDF5 trajectories сохраняются только
как исходный физический корпус и для диагностического full-path replay.

### 12.4 SRNO-r material-v2 retrain

| Local | H4 | H8 | H16 | H32 val |
|---:|---:|---:|---:|---:|
| 0.031199 | 0.013685 | 0.034846 | 0.068178 | **0.171715** |

Полная rollout evaluation:

| Split | Terminal $d_X$ | Mean T | Mean R | Mean J abs/travel | Mean aperture | Terminal T | Terminal R | Terminal J RMSE | Terminal aperture | Mean/max penetration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.171675 | 6.692 mm | 0.044656 | 0.016337 | 0.861 mm | 13.310 mm | 0.094811 | 0.062421 | 2.088 mm | 0.0376 / 18.801 mm |
| test | **0.204097** | 8.006 mm | 0.047183 | 0.019008 | 1.073 mm | 16.510 mm | 0.102904 | 0.071548 | 2.851 mm | 0.0644 / 16.884 mm |

Против interim SRNO-r terminal val уменьшился 0.297862→0.171675 (−42.35%), test 0.525187→0.204097 (−61.14%). Но split вырос с 8/1/2 до 22/3/3 и material изменился, поэтому это не чистая architecture ablation.

## 13. Две диагностики material-v2 перед $J_q$

### 13.1 Oracle gate

Frozen H32 checkpoint не переобучался. Вместо geometric gate ground-truth `contact_count>0` выбирал free bypass или neural cell, rollout оставался autoregressive.

| Split | Geometric gate | Oracle gate | Изменение |
|---|---:|---:|---:|
| val terminal $d_X$ | 0.171675 | 0.171740 | +0.038% |
| test terminal $d_X$ | 0.204097 | 0.209026 | +2.415% |

Geometric gate относительно GT contact имел recall 100%; precision 83.37% val и 87.57% test. Oracle gate не улучшил результат: gate не был главным bottleneck.

![Oracle gate H32](../runs/material-v2-diagnostics/oracle-gate/oracle_gate_h32.png)

### 13.2 Simulator floor

На 72 samples/3 objects были определены:

$$
D_{\rm cont}=d_X(x_{k+1}^{A},x_{k+1}^{B})
$$

для двух continuous repeats,

$$
D_{\rm reset}=d_X(x_{k+1}^{\rm continuous},x_{k+1}^{\rm fresh}),
$$

и model error

$$
E_{\rm model}=d_X(\mathcal R_\theta(x_k),x_{k+1}).
$$

Получено:

$$
E_{\rm floor}=\mathbb E[D_{\rm cont}]=0.002991,
$$

$$
E_{\rm model}=0.036876,
\qquad
\gamma=\frac{E_{\rm model}}{E_{\rm floor}}=12.329.
$$

Reset mean был 0.005791. Model error оставался более чем в 12 раз выше simulator repeatability floor. Это сняло simulator stochasticity и gate с позиции главных объяснений и мотивировало проверку достаточности contact features.

![Material-v2 simulator floor](../runs/material-v2-diagnostics/simulator-floor/material_v2_simulator_floor.png)

## 14. Последний чистый ablation: SRNO-r `gap` против `gap_jq`

### 14.1 Физическая гипотеза

Scalar gap сообщает глубину/зазор, но не сообщает явным образом направление contact normal и moment arm относительно spatial object motion. Для каждого trial point:

$$
z_i^O=R^T(y_i^G-p),
$$

$$
n_i^O=\frac{\nabla\phi(z_i^O)}
{\max(\|\nabla\phi(z_i^O)\|,10^{-8})},
\qquad
n_i^G=Rn_i^O,
$$

$$
\rho_i=\frac{y_i^G}{L}.
$$

Для left/spatial dimensionless pose perturbation Jacobian gap имеет знак

$$
\boxed{
J_{q,i}=
\left[-n_i^G,\;-\left(\rho_i\times n_i^G\right)\right]
\in\mathbb R^6
}.
$$

Candidate node feature:

$$
\boxed{
f_i=left[
\frac{h_i^{\rm contact}}{s_{\rm sdf}},
J_{q,i}
\right]\in\mathbb R^7
}.
$$

Baseline использует только первый scalar.

### 14.2 Differentiable SDF gradient

Для fractional voxel coordinate $u=(z-o)/v$, base index $b=\lfloor u\rfloor$, $\alpha=u-b$:

$$
\phi(z)=\sum_{a,b,c\in\{0,1\}}
w_a(\alpha_x)w_b(\alpha_y)w_c(\alpha_z)
G_{c,b,a},
$$

где $w_0(t)=1-t$, $w_1(t)=t$. $\nabla\phi$ получается аналитическим дифференцированием тех же восьми weights с делением на соответствующие $v_x,v_y,v_z$. Вне grid возвращаются positive $s_{\rm sdf}$ и zero gradient. Geometry остаётся float32; graph не detach-ится.

### 14.3 Что именно изменилось

Только две lifting matrices contact cell изменили input dimension:

$$
W_0,W_1:\mathbb R^1\to\mathbb R^{64}
\quad\longrightarrow\quad
W_0,W_1:\mathbb R^7\to\mathbb R^{64}.
$$

Добавлено

$$
2\times(7-1)\times64=768
$$

параметров: 31,436→32,204, то есть +2.44%. Kernel MLP, pooling, 12-output head, loss, gate, FK, dataset, optimizer и local schedule не менялись. Default `contact_features="gap"`, поэтому старые checkpoints совместимы.

### 14.4 Clean protocol

- один material-v2 manifest и active index;
- split 22/3/3;
- train/val/test active transitions 62,231/9,168/9,474;
- seeds 0, 1, 2;
- AdamW $3\cdot10^{-4}$, weight decay $10^{-4}$, clipping 1.0;
- 4 objects × 256 local transitions;
- максимум 100 epochs, patience 10;
- best checkpoint только по validation one-step $d_X$;
- полная evaluation всех active transitions;
- в рамках этого local phase rollout/H4–H32 не запускался; позднее он был выполнен отдельным clean paired experiment.

### 14.5 Все шесть local trainings

Equal-object full-set evaluation:

| Arm | Seed | Train $d_X$ | Val $d_X$ | Test $d_X$ | Best epoch / epochs run |
|---|---:|---:|---:|---:|---:|
| gap | 0 | 0.036210 | 0.031197 | 0.037451 | 15 / 26 |
| gap | 1 | 0.036204 | 0.031033 | 0.037203 | 14 / 25 |
| gap | 2 | 0.036874 | 0.030870 | 0.036699 | 11 / 22 |
| gap+$J_q$ | 0 | 0.024846 | 0.020099 | 0.026333 | 56 / 67 |
| gap+$J_q$ | 1 | 0.024574 | **0.019838** | **0.025196** | 91 / 100 |
| gap+$J_q$ | 2 | 0.025289 | 0.020357 | 0.025969 | 54 / 65 |

Mean ± sample standard deviation across seeds:

| Split | gap | gap+$J_q$ | Relative change |
|---|---:|---:|---:|
| train | 0.036429 ± 0.000385 | 0.024903 ± 0.000361 | −31.64% |
| val | 0.031033 ± 0.000164 | 0.020098 ± 0.000259 | **−35.24%** |
| test | 0.037118 ± 0.000383 | 0.025833 ± 0.000581 | **−30.40%** |

Paired validation improvement произошло во всех seeds:

- seed 0: 0.031197→0.020099, −35.58%;
- seed 1: 0.031033→0.019838, −36.07%;
- seed 2: 0.030870→0.020357, −34.06%.

### 14.6 Component decomposition на test

| Component | gap | gap+$J_q$ | Изменение |
|---|---:|---:|---:|
| translation | 1.3961 mm | **1.2325 mm** | −11.72% |
| rotation | **0.006539 rad** | 0.006932 rad | **+6.00%** |
| joint RMSE/travel | 0.031965 | **0.019982** | −37.49% |

Aggregate gain в основном пришёл из joint correction и, слабее, translation. Rotation one-step слегка ухудшилась; следовательно, физическая гипотеза подтверждена не по всем pose components.

Per-object test, mean по трём seeds:

| Test object | gap | gap+$J_q$ | Difference | Relative |
|---|---:|---:|---:|---:|
| Hercules oats | 0.034631 | 0.021040 | −0.013591 | −39.24% |
| orange/raisin cookies | 0.036533 | 0.028860 | −0.007673 | −21.00% |
| Baikal water | 0.040188 | 0.027598 | −0.012590 | −31.33% |

Hierarchical bootstrap ресемплировал training seeds, затем test objects, затем transitions внутри объекта; 10,000 replicates:

$$
\mathbb E[E_{\rm test}^{J_q}-E_{\rm test}^{\rm gap}]
=-0.011286,
$$

$$
95\%\ \mathrm{CI}=[-0.012980,-0.009451].
$$

Верхняя граница ниже zero, и $J_q$ лучше на validation во всех трёх seeds. Заданный criterion `confirmed_local_gain` выполнен.

Относительно simulator floor:

$$
\gamma_{\rm gap}=\frac{0.037118}{0.002991}=12.410,
\qquad
\gamma_{J_q}=\frac{0.025833}{0.002991}=8.637.
$$

Model всё ещё примерно в 8.6 раза выше simulator floor.

![Local Jq ablation](../runs/ablation-jq-local/jq_local_ablation.png)

### 14.7 Что этот ablation доказывает и чего не доказывает

Он доказывает устойчивое улучшение **local one-step** aggregate error при добавлении explicit contact normal/moment information с ростом parameter count всего на 2.44%.

Сам local experiment не доказывал улучшение H4/H8/H16/H32, terminal pose, rollout stability или long-horizon rotation. Позднейший clean rollout показал обратное на H32: `gap_jq` test $d_X=0.219773$ против `gap` $0.206607$, bootstrap difference $+0.013183$ с 95% CI $[+0.006825,+0.019440]$. Поэтому production H32 branch осталась gap-only SRNO-r material-v2; `gap_jq` существует как opt-in configuration, но rollout gain не подтверждён.

## 15. Actuator conditioning: `aperture` против `drive_error`

После того как rollout-$J_q$ не прошёл H32-критерий, actuator ablation проводился на последней подтверждённой feature-ветке `gap`. Сначала `drive_error` дал сильный local gain, после чего был выполнен отдельный полный clean rollout ablation. Другие компоненты модели и physics/data contract одновременно не менялись.

### 15.1 Формула единственной модификации

Состояние и trial configuration:

$$
x_k=(q_k,r_k),
\qquad
q_k=(R_k,p_k)\in SE(3),
\qquad
\tilde r_{k+1}=R_{\rm free}(\bar a_{k+1}).
$$

В обеих руках contact geometry вычислялась одинаково:

$$
y_i^G(\tilde r_{k+1})
=T_{\ell(i)}(\tilde r_{k+1})y_i,
\qquad
z_i^O=q_k^{-1}y_i^G(\tilde r_{k+1}),
$$

$$
h_i^{\rm geo}=\phi(z_i^O),
\qquad
h_i^{\rm contact}=h_i^{\rm geo}-2.56\;{\rm mm}.
$$

2.56 mm вычитались только для contact signal и gate. Feasibility продолжала использовать $h_i^{\rm geo}$ без contact-envelope shift. Единственный local feature в обеих руках:

$$
f_i=\frac{h_i^{\rm contact}}{s_{\rm sdf}}\in\mathbb R.
$$

Integral cell не менялась:

$$
z_i=\operatorname{SiLU}\!\left(
W_0f_i+
\frac1M\sum_j
\kappa(\rho_i,\rho_j)\odot W_1f_j+b
\right),
\qquad
\bar z=\frac1M\sum_i z_i.
$$

Baseline conditioning состояла из двух скаляров:

$$
\boxed{
c_k^{\rm aperture}
=\left[
\frac{A(r_k)}{L},
\frac{\bar a_{k+1}}{L}
\right]\in\mathbb R^2
}.
$$

Candidate заменяла их полным нормированным mismatch шести приводов:

$$
\bar r_{k+1}=R_{\rm free}(\bar a_{k+1}),
\qquad
\boxed{
u_k=
\frac{\bar r_{k+1}-r_k}{s}
\in\mathbb R^6
}.
$$

Деление на $s=(s_1,\ldots,s_6)$ покомпонентное. $u_k$ — position-drive error перед следующим increment, не torque и не contact reaction.

Изменился только input первого head layer:

$$
64+2\longrightarrow64+6.
$$

Число параметров:

$$
31\,436\longrightarrow31\,948,
\qquad
\Delta N=512,
\qquad +1.63\%.
$$

Output обеих рук оставался одинаковым:

$$
(\Delta\xi,\Delta r^c)\in\mathbb R^{6+6},
$$

$$
\hat q_{k+1}
=\operatorname{Exp}(\widehat{\Delta\xi})\hat q_k,
\qquad
\hat r_{k+1}
=\tilde r_{k+1}+\Delta r^c.
$$

### 15.2 Метрики и loss

Для каждого rollout step:

$$
T_k=\|\hat p_k-p_k^*\|_2,
\qquad
T_k^L=\frac{T_k}{L},
$$

$$
R_k=\theta(\hat R_k,R_k^*)
=\arccos\!\left(
\operatorname{clip}
\frac{\operatorname{tr}(\hat R_kR_k^{*T})-1}{2},-1,1
\right),
$$

$$
J_k=
\sqrt{
\frac16\sum_{m=1}^{6}
\left(
\frac{\hat r_{k,m}-r_{k,m}^*}{s_m}
\right)^2
},
$$

$$
\boxed{
d_X(k)=\sqrt{(T_k^L)^2+R_k^2+J_k^2}
},
\qquad
E_H=d_X(H).
$$

Одинаковая objective:

$$
\mathcal L
=\mathcal L_{\rm state}+\lambda_K\mathcal L_K,
\qquad
\mathcal L_{\rm state}
=\frac1H\sum_{k=1}^{H}d_X(k)^2.
$$

Средние T/R/J ниже — средние уже вычисленных trajectory components. Поэтому mean $d_X$ не обязан равняться корню из квадратов mean T/R/J.

### 15.3 Clean contract

- material-v2 object split: 22 train / 3 validation / 3 test;
- seeds: 0, 1, 2;
- 100 trajectories на каждый evaluation object: 300 validation и 300 unseen-test trajectories на checkpoint;
- rollout без teacher forcing;
- curriculum $H4\to H8\to H16\to H32$;
- максимум 25 epochs на horizon, patience 10;
- checkpoint выбирался только по validation terminal $d_X(H)$;
- каждая рука начиналась из соответствующего frozen local checkpoint и получала новый rollout AdamW optimizer/scheduler;
- manifest SHA-256: `c8bddec752f0418e92383fd0d9193e2d70a37845244c4c1c26ed9f9170c3012a`;
- gripper SHA-256: `6f3280535f5fd5bf543da3dba911825710ea73edb2a19b77b4fd4225fa2f02d6`.

Baseline initialization: `runs/ablation-jq-local/baseline/seed-{0,1,2}/best-local.pt`. Candidate initialization: `runs/ablation-actuator-local/drive_error/seed-{0,1,2}/best-local.pt`. Runner до обучения проверял hashes и идентичность model/loss/optimizer/loader/training config.

### 15.4 Предшествующий local result

На полном active one-step evaluation `drive_error` был подтверждён:

| Split | aperture $d_X$ | drive_error $d_X$ | aperture T/R/J | drive_error T/R/J |
|---|---:|---:|---|---|
| train | 0.036429 | 0.018299 | 0.001398 / 0.010630 / 0.027883 | 0.001200 / 0.010289 / 0.005663 |
| val | 0.031033 | 0.012832 | 0.001194 / 0.006181 / 0.026288 | 0.000944 / 0.005720 / 0.005822 |
| test | 0.037118 | 0.015041 | 0.001396 / 0.006539 / 0.031965 | 0.001117 / 0.006040 / 0.007423 |

Local test difference и bootstrap:

$$
E_{\rm test}^{\rm drive}-E_{\rm test}^{\rm aperture}
=-0.022086,
$$

$$
95\%\;CI=[-0.023543,-0.020797].
$$

Именно этот сильный local gain мотивировал полный rollout experiment.

### 15.5 Полные rollout T/R/J результаты

Все значения — equal-object mean по трём seeds.

Validation:

| Arm | H | $d_X$ | T [m] | T/L | R [rad] | J/travel |
|---|---:|---:|---:|---:|---:|---:|
| aperture | 4 | 0.013551787 | 0.001111693 | 0.009970346 | 0.007668148 | 0.001603323 |
| aperture | 8 | 0.034075549 | 0.002489365 | 0.022326153 | 0.021775098 | 0.005507032 |
| aperture | 16 | 0.065296953 | 0.004624734 | 0.041477449 | 0.042989561 | 0.014495682 |
| aperture | 32 | 0.171902039 | 0.013184156 | 0.118243593 | 0.095097292 | 0.064208850 |
| drive_error | 4 | 0.014342212 | 0.001156162 | 0.010369168 | 0.007739661 | 0.002459650 |
| drive_error | 8 | 0.032776574 | 0.002391130 | 0.021445116 | 0.020880789 | 0.005274572 |
| drive_error | 16 | 0.061674431 | 0.004064275 | 0.036450909 | 0.043206698 | 0.013835103 |
| drive_error | 32 | 0.179334637 | 0.014301087 | 0.128260915 | 0.095333210 | 0.064834448 |

Test:

| Arm | H | $d_X$ | T [m] | T/L | R [rad] | J/travel |
|---|---:|---:|---:|---:|---:|---:|
| aperture | 4 | 0.018252572 | 0.001635874 | 0.014671516 | 0.008576471 | 0.002754367 |
| aperture | 8 | 0.044945031 | 0.003716006 | 0.033327417 | 0.024851562 | 0.008230999 |
| aperture | 16 | 0.075580784 | 0.005823461 | 0.052228363 | 0.045371630 | 0.016199924 |
| aperture | 32 | 0.206607107 | 0.016564174 | 0.148557658 | 0.104379216 | 0.073939927 |
| drive_error | 4 | 0.019339220 | 0.001677755 | 0.015047132 | 0.008953385 | 0.004089213 |
| drive_error | 8 | 0.042922475 | 0.003479662 | 0.031207737 | 0.024221371 | 0.007965376 |
| drive_error | 16 | 0.070861936 | 0.005102961 | 0.045766482 | 0.045049375 | 0.015828616 |
| drive_error | 32 | 0.213212742 | 0.017841172 | 0.160010552 | 0.101346730 | 0.074008002 |

Относительное изменение test terminal error:

$$
H4:+5.95\%,\qquad
H8:-4.50\%,\qquad
H16:-6.24\%,\qquad
H32:+3.20\%.
$$

Следовательно, `drive_error` помогает на H8/H16, но этот эффект не сохраняется на полном H32.

### 15.6 Все terminal $d_X$ по seeds

Validation:

| Arm | Seed | H4 | H8 | H16 | H32 |
|---|---:|---:|---:|---:|---:|
| aperture | 0 | 0.013713469 | 0.034876231 | 0.068140245 | 0.171674619 |
| aperture | 1 | 0.013406313 | 0.034258605 | 0.064112427 | 0.172902003 |
| aperture | 2 | 0.013535578 | 0.033091812 | 0.063638188 | 0.171129495 |
| drive_error | 0 | 0.014987827 | 0.033088998 | 0.062318608 | 0.181716969 |
| drive_error | 1 | 0.014042729 | 0.032952256 | 0.060796191 | 0.178284496 |
| drive_error | 2 | 0.013996080 | 0.032288468 | 0.061908492 | 0.178002447 |

Test:

| Arm | Seed | H4 | H8 | H16 | H32 |
|---|---:|---:|---:|---:|---:|
| aperture | 0 | 0.017969707 | 0.044470821 | 0.077293582 | 0.204097082 |
| aperture | 1 | 0.018564690 | 0.047588583 | 0.074340076 | 0.207259516 |
| aperture | 2 | 0.018223319 | 0.042775691 | 0.075108695 | 0.208464722 |
| drive_error | 0 | 0.021122117 | 0.044192848 | 0.074696165 | 0.222183372 |
| drive_error | 1 | 0.018219092 | 0.043904398 | 0.068683332 | 0.208947256 |
| drive_error | 2 | 0.018676451 | 0.040670181 | 0.069206311 | 0.208507597 |

H32 validation разность `drive_error - aperture` положительна во всех seeds:

$$
\Delta_0=+0.010042,
\qquad
\Delta_1=+0.005382,
\qquad
\Delta_2=+0.006873.
$$

### 15.7 H32 component diagnosis

| Split | Arm | T [m] | T/L | R [rad] | J/travel | $d_X$ |
|---|---|---:|---:|---:|---:|---:|
| val | aperture | 0.013184 | 0.118244 | 0.095097 | 0.064209 | 0.171902 |
| val | drive_error | 0.014301 | 0.128261 | 0.095333 | 0.064834 | 0.179335 |
| test | aperture | 0.016564 | 0.148558 | 0.104379 | 0.073940 | 0.206607 |
| test | drive_error | 0.017841 | 0.160011 | 0.101347 | 0.074008 | 0.213213 |

На test rotation немного улучшилась:

$$
0.104379\to0.101347\;{\rm rad},
\qquad -2.90\%,
$$

но translation ухудшилась:

$$
16.564\to17.841\;{\rm mm},
\qquad +7.71\%.
$$

Joint component практически не изменилась, $0.073940\to0.074008$. Поэтому H32 aggregate degradation в основном вызвана translation drift.

### 15.8 Полная H32 test pushforward-кривая

Разность $\Delta(k)=d_X^{\rm drive}(k)-d_X^{\rm aperture}(k)$:

| k | aperture | drive_error | $\Delta(k)$ |
|---:|---:|---:|---:|
| 0 | 0.000000000 | 0.000000000 | +0.000000000 |
| 1 | 0.017183603 | 0.008032721 | -0.009150883 |
| 2 | 0.022820482 | 0.014154194 | -0.008666288 |
| 3 | 0.029195603 | 0.020474957 | -0.008720646 |
| 4 | 0.035671075 | 0.026852246 | -0.008818829 |
| 5 | 0.042017573 | 0.033583332 | -0.008434241 |
| 6 | 0.047989683 | 0.040376703 | -0.007612980 |
| 7 | 0.053183130 | 0.046845737 | -0.006337393 |
| 8 | 0.057569011 | 0.052887132 | -0.004681879 |
| 9 | 0.061190156 | 0.058389167 | -0.002800989 |
| 10 | 0.064868225 | 0.063800434 | -0.001067791 |
| 11 | 0.068786809 | 0.069388258 | +0.000601449 |
| 12 | 0.071137068 | 0.073762593 | +0.002625525 |
| 13 | 0.073330219 | 0.077806818 | +0.004476599 |
| 14 | 0.076419160 | 0.082776226 | +0.006357066 |
| 15 | 0.079769246 | 0.087654203 | +0.007884957 |
| 16 | 0.083106928 | 0.092173293 | +0.009066366 |
| 17 | 0.086471135 | 0.096346992 | +0.009875856 |
| 18 | 0.090971120 | 0.101496240 | +0.010525120 |
| 19 | 0.095530684 | 0.106402405 | +0.010871721 |
| 20 | 0.101483641 | 0.113071760 | +0.011588119 |
| 21 | 0.107334385 | 0.118734534 | +0.011400148 |
| 22 | 0.114526850 | 0.124869828 | +0.010342978 |
| 23 | 0.121826810 | 0.131760796 | +0.009933986 |
| 24 | 0.129628087 | 0.139204398 | +0.009576311 |
| 25 | 0.141640703 | 0.150746956 | +0.009106254 |
| 26 | 0.151590884 | 0.160181373 | +0.008590490 |
| 27 | 0.159402668 | 0.167587881 | +0.008185213 |
| 28 | 0.169584473 | 0.177902276 | +0.008317803 |
| 29 | 0.178379228 | 0.186598519 | +0.008219292 |
| 30 | 0.185495953 | 0.193561380 | +0.008065427 |
| 31 | 0.196488341 | 0.204226236 | +0.007737895 |
| 32 | 0.206607049 | 0.213212704 | +0.006605655 |

До $k=10$ candidate лучше; на $k=11$ знак меняется. Максимальный средний разрыв:

$$
\Delta(20)=+0.011588.
$$

Это показывает, что actuator conditioning улучшает первые contact increments, но не уменьшает long-horizon composition error.

### 15.9 Per-object H32

Validation, mean по трём seeds:

| Object | aperture | drive_error | Difference |
|---|---:|---:|---:|
| `masliny-federici-bez-kostochki-300-g-90215` | 0.169233864 | 0.174453144 | +0.005219281 |
| `sous-soevyy-250-ml-58088` | 0.189071248 | 0.190287054 | +0.001215806 |
| `voda-pitevaya-senezhskaya-negazirovannaya-pet-1-5-l-43733` | 0.157401005 | 0.173263714 | +0.015862708 |

Unseen test, mean по трём seeds:

| Object | aperture | drive_error | Difference |
|---|---:|---:|---:|
| `gerkules-ovsyanye-khlopya-400-g-1248` | 0.194221631 | 0.204014942 | +0.009793311 |
| `pechene-sdobnoe-khlebnyy-spas-italyanskoe-s-apelsinovym-vkusom-i-izyumom-230-g-46835` | 0.256522169 | 0.266266217 | +0.009744048 |
| `voda-pitevaya-prirodnaya-legenda-baykala-750-ml-42674` | 0.169077521 | 0.169357066 | +0.000279546 |

Candidate хуже на каждом validation и test object.

### 15.10 Статистический критерий и итог

Подтверждение требовало одновременно

$$
E_{\rm val,H32}^{\rm drive}<E_{\rm val,H32}^{\rm aperture}
$$

во всех трёх seeds и верхнюю границу 95% hierarchical bootstrap CI ниже нуля. Bootstrap: seed → object → trajectory, 10 000 replicates, seed 20260818.

Прямая equal-object разность:

$$
\Delta_{\rm direct}
=0.213212742-0.206607107
=+0.006605635.
$$

Bootstrap distribution:

$$
\overline\Delta_{\rm bootstrap}=+0.006481,
\qquad
95\%\;CI=[-0.001089,+0.017838].
$$

Различие 0.006606 против 0.006481 связано с иерархическим Monte Carlo resampling и конечными 10 000 replicates.

Оба строгих условия не выполнены. Формальный результат:

$$
\boxed{
\texttt{drive\_error не подтверждён для H32 rollout}
}.
$$

Несмотря на большой local gain и пользу на H8/H16, рабочим global conditioning полного rollout остаётся `aperture`.

![Actuator rollout ablation](../runs/ablation-actuator-rollout/actuator_rollout_ablation.png)

## 16. Полная сводка всех rollout trainings

Здесь перечислены все сохранённые runs с rollout curriculum. Значения — лучшие validation checkpoints на соответствующем horizon; test — финальный H32 checkpoint.

| Run | Representation / изменение | Seed | Local val | H4 | H8 | H16 | H32 val | H32 test |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `srno-contact-v1` | ранний SRNO-$a$ | 0 | 0.018214 | 0.025354 | 0.084209 | 0.244301 | 0.431521 | 0.455406 |
| `srno-contact-v1-curriculum` | повтор curriculum | 0 | 0.018214 | 0.025354 | 0.082244 | 0.241875 | 0.434877 | 0.458531 |
| `srno-contact-v1-full-data` | object-at-once full data | 0 | 0.032984 | 0.012107 | 0.060771 | 0.236074 | 0.416735 | 0.454601 |
| `srno-contact-v1-full-coverage` | efficient complete coverage | 0 | 0.035601 | 0.012137 | 0.060880 | 0.233099 | 0.416463 | 0.449868 |
| `srno-contact-v1-lambda-k0` | только $\lambda_K=0$ | 0 | 0.022545 | 0.012642 | 0.061440 | 0.230088 | 0.382794 | 0.459616 |
| `srno-contact-v1-physx-sdf` | cooked SDF/contact geometry | 0 | 0.033883 | 0.012445 | 0.061335 | 0.232829 | 0.410975 | 0.440439 |
| contact baseline | baseline manifold study | 0 | 0.036513 | 0.010839 | 0.057505 | 0.217253 | 0.392696 | 0.44144 |
| contact gate-only | $\delta=1.608$ mm | 0 | 0.038454 | 0.009893 | 0.054406 | 0.217127 | 0.396666 | 0.44040 |
| contact feasibility-only | $h_{adm}=-31.242$ mm | 0 | 0.022743 | 0.011343 | 0.054898 | 0.218257 | 0.349711 | 0.43941 |
| contact combined | оба изменения | 0 | 0.024923 | 0.009872 | 0.053205 | 0.214663 | 0.352562 | 0.44110 |
| `srno-r-interim-11objects` | actual joints, 11 objects | 0 | 0.025153 | 0.035326 | 0.124717 | 0.205775 | 0.297874 | 0.525187 |
| `srno-r-material-v2` | actual joints, final physics/data | 0 | 0.031199 | 0.013685 | 0.034846 | 0.068178 | **0.171715** | **0.204097** |
| actuator aperture | clean paired rollout | 0 | 0.031197 | 0.013713 | 0.034876 | 0.068140 | 0.171675 | 0.204097 |
| actuator aperture | clean paired rollout | 1 | 0.031033 | 0.013406 | 0.034259 | 0.064112 | 0.172902 | 0.207260 |
| actuator aperture | clean paired rollout | 2 | 0.030870 | 0.013536 | 0.033092 | 0.063638 | 0.171129 | 0.208465 |
| actuator drive_error | clean paired rollout | 0 | 0.012633 | 0.014988 | 0.033089 | 0.062319 | 0.181717 | 0.222183 |
| actuator drive_error | clean paired rollout | 1 | 0.013053 | 0.014043 | 0.032952 | 0.060796 | 0.178284 | 0.208947 |
| actuator drive_error | clean paired rollout | 2 | 0.012808 | 0.013996 | 0.032288 | 0.061908 | 0.178002 | 0.208508 |

Почему local values разных runs не всегда монотонны: early stopping выбирает one-step checkpoint по конкретному validation split; geometry/state metric и active-index менялись между версиями; улучшение rollout не обязано сопровождаться меньшим teacher-forced local score.

## 17. Итоговая причинная картина

1. **Базовая neural-operator конструкция работала**, включая exact free bypass и shared integral cell, но scalar-aperture geometry делала supervision физически противоречивой.
2. **Одна SDF/PhysX правка была необходима, но недостаточна**: test H32 0.44987→0.44044.
3. **$\lambda_K=0$ не решил rollout**: validation H32 выглядел лучше, unseen test ухудшился, penetration почти удвоилась.
4. **Gate не был главным bottleneck**: gate-only помогал H4–H16, но не H32; oracle gate material-v2 также не улучшил terminal error.
5. **Главный найденный geometric bug — $a\to r$**. Actual-joint FK уменьшил mean GT penetration 5.974→0.0589 mm и долю >1 mm с 50.75% до 0.0104%.
6. **Material binding был отдельным настоящим data bug**. Он исправлен runtime read-back/assert, а dataset после этого полностью пересобран и перекалиброван.
7. **Ранние тесты указывали на PhysX contact-history effect**, но обновлённый
   preserve/fresh тест из раздела 19 не отделил его от continuous-repeat floor;
   поэтому наличие значимого hidden-state effect сейчас не считается доказанным.
8. **Финальный gap-only SRNO-r существенно улучшил long horizon**: material-v2 test H32 0.20410.
9. **Explicit $J_q$ дал подтверждённый local gain**, но не rollout gain: local test 0.03712→0.02583, тогда как H32 test 0.20661→0.21977; H32 bootstrap CI целиком положителен.
10. **`drive_error` дал ещё больший local gain**, test 0.03712→0.01504, но также не скомпозировался до H32: test 0.20661→0.21321, а validation хуже во всех seeds.

Оба ранее запланированных clean rollout comparisons теперь выполнены. Ни `gap_jq`, ни `drive_error` не прошли строгий H32-критерий, поэтому текущая рабочая long-horizon ветка остаётся `gap` + `aperture`. Следующий experiment должен адресовать именно long-horizon composition/object-pose drift, не смешивая это с новой physics или dataset revision.

## 18. Воспроизводимость и артефакты

- Базовый diagnostic dashboard: [`runs/srno-contact-v1-full-coverage/diagnostics-point16/diagnostic_dashboard.png`](../runs/srno-contact-v1-full-coverage/diagnostics-point16/diagnostic_dashboard.png)
- $\lambda_K=0$ report: [`runs/srno-contact-v1-lambda-k0/comparison/summary.md`](../runs/srno-contact-v1-lambda-k0/comparison/summary.md)
- SDF/PhysX diagnostic: [`runs/sdf-collision-diagnostic/results.json`](../runs/sdf-collision-diagnostic/results.json)
- Contact-manifold report: [`runs/contact-manifold-ablation/comparison/summary.md`](../runs/contact-manifold-ablation/comparison/summary.md)
- $a$ против actual-joint FK: [`runs/joint-gap-diagnostic/results.json`](../runs/joint-gap-diagnostic/results.json)
- Markov test: [`runs/markov-state-sufficiency-v1/results.json`](../runs/markov-state-sufficiency-v1/results.json)
- Contact-memory factorization: [`runs/contact-memory-factorization-v1/results.json`](../runs/contact-memory-factorization-v1/results.json)
- Strong-friction native ablation: [`runs/strong-friction-ablation-v1/results.json`](../runs/strong-friction-ablation-v1/results.json)
- Final material-v2 calibration: [`data/simulator-r-v1/calibration-material-v2.json`](../data/simulator-r-v1/calibration-material-v2.json)
- Oracle gate: [`runs/material-v2-diagnostics/oracle-gate/results.json`](../runs/material-v2-diagnostics/oracle-gate/results.json)
- Simulator floor: [`runs/material-v2-diagnostics/simulator-floor/results.json`](../runs/material-v2-diagnostics/simulator-floor/results.json)
- Final $J_q$ results: [`runs/ablation-jq-local/results.json`](../runs/ablation-jq-local/results.json)
- Final $J_q$ dashboard: [`runs/ablation-jq-local/jq_local_ablation.png`](../runs/ablation-jq-local/jq_local_ablation.png)
- $J_q$ rollout results: [`runs/ablation-jq-rollout/results.json`](../runs/ablation-jq-rollout/results.json)
- Actuator local results: [`runs/ablation-actuator-local/results.json`](../runs/ablation-actuator-local/results.json)
- Actuator rollout results: [`runs/ablation-actuator-rollout/results.json`](../runs/ablation-actuator-rollout/results.json)
- Actuator rollout dashboard: [`runs/ablation-actuator-rollout/actuator_rollout_ablation.png`](../runs/ablation-actuator-rollout/actuator_rollout_ablation.png)

Ключевые сохранённые implementation commits:

- `827fdf8` — initial SRNO model and simulator data pipeline;
- `2f46673` — PhysX runtime SDF/gripper geometry alignment;
- `2b5501c` — actual PhysX joint state for gripper geometry;
- `32ac086` — PhysX contact material binding and verification.

Текущие $J_q$ и actuator-conditioning реализации находятся в рабочем дереве поверх этих commits. После composition diagnostics полный regression suite: **63 passed**, 12 warnings.

## 19. Contact-composition diagnostics v2: amplification, signed bias и Markov/history

Этот раздел описывает диагностический запуск 18 августа 2026 года. Он не
является новым training run: dataset, trajectories, SDF, FK, gate, material,
loss, архитектура и веса checkpoints не изменялись. Production-вариантом до и
после теста остаётся

$$
\boxed{\texttt{contact\_features=gap},\qquad
\texttt{global\_conditioning=aperture}}.
$$

`drive_error` использовался только как диагностический контроль актуального
one-step floor. Ни `J_q`, ни `drive_error` не объявлялись production-вариантом.

### 19.1 Зафиксированный physics/data/model contract

Перед запуском runner проверил hashes checkpoints, manifest и gripper, а также
выполнил live PxMaterial и actuator read-back. Зафиксированы:

- manifest SHA256:
  `c8bddec752f0418e92383fd0d9193e2d70a37845244c4c1c26ed9f9170c3012a`;
- gripper SHA256:
  `6f3280535f5fd5bf543da3dba911825710ea73edb2a19b77b4fd4225fa2f02d6`;
- simulator config SHA256:
  `e8d87869ff2fe7a0111d6d2c8e4394631e9dd5a8432299ea8a8527bdac09ea01`;
- $\mu_s=2.4$, $\mu_d=2.0$, combine=`min`, patch friction, PCM;
- rigid contact, TGS, 64 position и 16 velocity iterations;
- $\delta_{\rm gate}=7.9342104$ mm,
  $d_c=\texttt{contact\_offset\_sum}=2.56$ mm,
  $s_{\rm sdf}=20$ mm;
- actuator: force drive, position target,
  $K=14$, $D=0.35$, $\tau_{\max}=480$,
  $\dot r_{\max}=0.1$ rad/s;
- settling: minimum 20, maximum 2400 control steps, 10 consecutive valid
  checks, thresholds 0.5 mm position delta, 0.01 m/s linear velocity,
  0.1 rad/s angular velocity и 0.01 rad/s joint velocity;
- headless CUDA simulation и RAM watchdog 14 GiB.

Все шесть runtime joint names и их порядок совпали с gripper asset. Любое
несовпадение перечисленных полей, hashes или feature settings является
fail-fast ошибкой.

Выборка состояла из всех трёх validation и трёх unseen test объектов. На каждом
объекте детерминированно выбрано 12 sustained-contact transitions: ровно по три
из диапазонов

$$
[1,8],\quad[9,16],\quad[17,24],\quad[25,31].
$$

Итого выбрано 72 состояния. Для всех amplification-комбинаций валидны 71; один
переход test-объекта `pechene-sdobnoe-...-46835` на позднем шаге не прошёл
неизменённый production settling. Raw результат сохранён, но этот переход не
включён в paired aggregates. Signed-bias тест использовал все 600 val/test
trajectories. Seed выбора, perturbations и bootstrap равен 0; hierarchical
bootstrap содержит 10 000 repetitions.

### 19.2 Что конкретно представлял собой эксперимент

Это был **не ablation обучения**, а post-training diagnostic experiment над
уже обученными checkpoints и неизменным симулятором. Его цель состояла в том,
чтобы различить три возможные причины H32 composition defect:

1. **Слишком большой локальный gain модели.** Малое возмущение состояния может
   усиливаться learned map сильнее, чем реальным simulator transition:

   $$
   A_{model}>A_{sim}.
   $$

2. **Малый, но ненулевой signed one-step bias.** Даже при хорошем среднем
   one-step $d_X$ ошибка одного знака может приблизительно суммироваться:

   $$
   b_k^{TF}\ne0,
   \qquad
   b_{32}^{AR}\approx\sum_{k=0}^{31}b_k^{TF}.
   $$

3. **Недостаточность состояния $x=(q,r)$.** При одинаковых наблюдаемых
   $(q_k,r_k,\bar a_{k+1})$ следующий equilibrium может зависеть от скрытой
   PhysX contact/solver history:

   $$
   F_{sim}(q_k,r_k,\bar a_{k+1};\,\mathcal H_1)
   \ne
   F_{sim}(q_k,r_k,\bar a_{k+1};\,\mathcal H_2).
   $$

#### Какие checkpoints участвовали

Основная рука — три независимо обученных production H32 checkpoints seeds
0, 1 и 2:

```text
runs/ablation-actuator-rollout/aperture/seed-{0,1,2}/best-rollout-h32.pt
```

Их конфигурация строго равна `gap + aperture`. Для контекста signed bias, но не
для выбора production-модели, использовались существующие H32
`drive_error` checkpoints:

```text
runs/ablation-actuator-rollout/drive_error/seed-{0,1,2}/best-rollout-h32.pt
```

В Markov-тесте one-step model floor на тех же 71 валидных simulator samples
оценивался существующими local checkpoints обеих рук:

```text
runs/ablation-jq-local/baseline/seed-{0,1,2}/best-local.pt
runs/ablation-actuator-local/drive_error/seed-{0,1,2}/best-local.pt
```

Ни один checkpoint не дообучался и не модифицировался.

#### Как была построена simulator-выборка

Для каждого из шести val/test объектов из material-v2 HDF5 выбирались только
переходы с sustained contact: contact присутствует на текущем и следующем
increment, а actual aperture имеет ненулевой lag относительно command. Из
каждого объекта выбрано по три перехода в каждом из четырёх диапазонов шагов,
то есть

$$
6\ \text{objects}\times4\ \text{bands}\times3\ \text{states}
=72\ \text{states}.
$$

Начальная pose каждого выбранного trajectory бралась из сохранённого
`source_pose_index`. Поэтому continuous branches воспроизводили всю closure
history от открытого gripper до выбранного $x_k$, а не начинались напрямую из
изолированного contact state.

#### Какие simulator branches запускались

Для amplification на каждом объекте одновременно запускались девять групп
веток с одной исходной continuous history:

1. `preserve` — untouched continuous replay;
2. `repeat` — независимый untouched replay для $A_{noise}$;
3. `amplification_base` — состояние непосредственно перед increment точно
   перезаписано в reference $x_k$;
4. шесть perturbed branches: translation, rotation и joints при
   $\epsilon=0.005$ и $0.01$.

`amplification_base` и каждая perturbed branch получали одну и ту же следующую
command $\bar a_{k+1}$ и один production settling protocol. Поэтому
$A_{sim}$ сравнивает две ветки, которые перед command различаются только
заданной $\delta x$, тогда как `preserve`/`repeat` отдельно измеряют
вариативность непрерывного replay.

Для Markov/history test дополнительно создавались четыре свежие сцены:

- `fresh,cold-a` и `fresh,cold-b`: восстановить $(q_k,r_k)$, обнулить
  velocities и сразу подать $\bar a_{k+1}$;
- `fresh,warm-a` и `fresh,warm-b`: восстановить $(q_k,r_k)$, сначала settle
  на текущей command $\bar a_k$, затем подать $\bar a_{k+1}$.

Пары `a/b` измеряли repeatability каждого reset protocol. Отдельно сохранялся
drift состояния во время warm preconditioning.

#### Что вычислялось без симулятора

Signed-bias часть проходила по всем 600 val/test trajectories из HDF5. Для
каждого из трёх seeds и каждого из 32 шагов выполнялись одновременно:

- teacher-forced prediction из истинного $x_k^*$;
- полный autoregressive rollout без teacher forcing;
- body-frame $SE(3)$-log ошибки;
- signed spatial translation в gripper frame.

Таким образом, bias-оценка содержит

$$
3\ \text{seeds}\times600\ \text{trajectories}\times32\ \text{steps}
=57\,600
$$

teacher-forced и столько же autoregressive pose comparisons для каждой
диагностической model arm.

#### Что намеренно не менялось

В ходе эксперимента не менялись dataset shards, object split, SDF, collision
geometry, gripper FK, $\delta_{gate}$, contact offset, material, settling,
actuator, contact cell, head, loss или model weights. Не запускались retrain,
implicit solver, memory state и новые auxiliary losses. Поэтому результаты
диагностируют composition текущего SRNO, а не смешивают её с очередной версией
physics или обучения.

### 19.3 Единая метрика и perturbations

Для состояния $x=(q,r)=(R,p,r)$ использовалась та же метрика, что и при
обучении SRNO-r:

$$
d_X^2(x_1,x_2)=
\frac{\|p_1-p_2\|_2^2}{L^2}
+\theta(R_1,R_2)^2
+\frac16\sum_{m=1}^{6}
\left(\frac{r_{1,m}-r_{2,m}}{s_m}\right)^2.
$$

Раздельно вносились

$$
\delta x_p=(\delta p,0,0),\qquad
\delta x_R=(0,\delta\omega,0),\qquad
\delta x_r=(0,0,\delta r)
$$

с $d_X(x+\delta x,x)\in\{0.005,0.01\}$. Для joints:

$$
\delta r=\epsilon(s\odot u),\qquad
\sqrt{\frac16\sum_m u_m^2}=1.
$$

Направление отражалось около фактических PhysX joint limits без clipping.
Максимальное отклонение измеренного input $d_X$ от цели составило
$2.60\cdot10^{-7}$ для rotation,
$9.02\cdot10^{-8}$ для translation и
$1.75\cdot10^{-8}$ для joints. Ни одна perturbation не вышла за joint limits.

### 19.4 Model-vs-simulator amplification

Измерялись

$$
A_{\rm model}=
\frac{d_X(R_\theta(x+\delta x,\bar a_{k+1}),
R_\theta(x,\bar a_{k+1}))}
{d_X(x+\delta x,x)},
$$

$$
A_{\rm sim}=
\frac{d_X(F_{\rm sim}(x+\delta x,\bar a_{k+1}),
F_{\rm sim}(x,\bar a_{k+1}))}
{d_X(x+\delta x,x)},
$$

$$
A_{\rm noise}=
\frac{d_X(F_{\rm sim}^{(1)}(x),F_{\rm sim}^{(2)}(x))}
{d_X(x+\delta x,x)}.
$$

Для $A_{\rm sim}$ baseline и perturbed branch непосредственно перед следующей
command записывались из одного exact $x_k$. Две дополнительные untouched
continuous-replay ветки независимо оценивали $A_{\rm noise}$. Floor не
вычитался. Ни gate status, ни simulator contact count не переключились ни в
одной из 426 валидных комбинаций state/type/scale.

В таблице mean является equal-object mean; median, p90 и p95 рассчитаны по
pooled samples. Model объединяет три production H32 seeds.

| Perturbation | $\epsilon$ | $A_{model}$, mean / median / p90 / p95 | $A_{sim}$, mean / median / p90 / p95 | $A_{noise}$, mean / median / p90 / p95 |
|---|---:|---:|---:|---:|
| translation | 0.005 | 0.9911 / 0.9972 / 1.0207 / 1.0408 | 5.0814 / 1.2650 / 3.7317 / 11.3690 | 6.9360 / 0.9091 / 9.1900 / 15.4747 |
| translation | 0.010 | 0.9911 / 0.9976 / 1.0148 / 1.0316 | 2.1438 / 1.1398 / 3.5703 / 5.1218 | 3.4680 / 0.4545 / 4.5950 / 7.7374 |
| rotation | 0.005 | 1.0039 / 1.0016 / 1.0178 / 1.0223 | 2.7297 / 0.9317 / 8.7427 / 11.5056 | 6.9360 / 0.9091 / 9.1900 / 15.4747 |
| rotation | 0.010 | 1.0039 / 1.0017 / 1.0190 / 1.0246 | 2.4520 / 0.9818 / 5.6878 / 13.1553 | 3.4680 / 0.4545 / 4.5950 / 7.7374 |
| joints | 0.005 | 0.005645 / 0.003865 / 0.012382 / 0.018519 | 5.4882 / 0.9726 / 5.7955 / 14.9816 | 6.9360 / 0.9091 / 9.1900 / 15.4747 |
| joints | 0.010 | 0.005648 / 0.003863 / 0.012384 / 0.018601 | 4.2058 / 0.8631 / 6.0344 / 8.7545 | 3.4680 / 0.4545 / 4.5950 / 7.7374 |

Формальный признак чрезмерной expansiveness был

$$
\operatorname{CI}_{95\%,low}
\left[\mathbb E\log\frac{A_{\rm model}}{A_{\rm sim}}\right]>0.
$$

| Perturbation | $\epsilon$ | all steps: mean log-ratio, 95% CI | steps 17--31: mean log-ratio, 95% CI | Excessive |
|---|---:|---:|---:|---:|
| translation | 0.005 | -0.2828, [-0.5005, -0.0729] | -0.4575, [-0.8591, -0.1043] | no |
| translation | 0.010 | -0.2025, [-0.3755, -0.0264] | -0.4038, [-0.6413, -0.1519] | no |
| rotation | 0.005 | +0.0049, [-0.2475, +0.2543] | -0.0922, [-0.4896, +0.2996] | no |
| rotation | 0.010 | -0.0191, [-0.2627, +0.2265] | +0.0095, [-0.2966, +0.3391] | no |
| joints | 0.005 | -5.8140, [-6.3817, -5.1506] | -5.8296, [-6.5325, -5.0287] | no |
| joints | 0.010 | -5.7530, [-6.2873, -5.1038] | -5.5775, [-6.2804, -4.7842] | no |

Следовательно, данных в пользу чрезмерной local expansiveness SRNO нет. Для
translation cell почти сохраняет perturbation, для rotation имеет gain около
единицы, а dependence следующего состояния от current joints почти подавлена.
Однако $A_{\rm noise}$ сопоставим с raw $A_{\rm sim}$ и часто превосходит
его. Поэтому simulator Jacobian в значительной части выборки noise-limited;
таблица доказывает отсутствие заявленного model-over-simulator эффекта, но не
даёт точной оценки физического amplification.

![Composition amplification](../runs/contact-composition-diagnostics-v2/amplification.png)

### 19.5 Signed one-step и accumulated pose bias

Диагностический $SE(3)$-log использует twist order $[v,\omega]$ и устойчивые
ряды около нуля. Teacher-forced error:

$$
e_k^{\rm TF}=
\Log\!\left[(q_{k+1}^*)^{-1}
R_\theta(x_k^*,\bar a_{k+1})_q\right]
=\begin{bmatrix}v_k\\\omega_k\end{bmatrix},
\qquad b_k^{\rm TF}=\mathbb E[e_k^{\rm TF}],
$$

$$
B_K^{\rm TF}=\sum_{k=0}^{K-1}b_k^{\rm TF}.
$$

Для autoregressive rollout:

$$
b_k^{\rm AR}=\mathbb E\!\left[
\Log((q_k^*)^{-1}\hat q_k)\right].
$$

Средняя по шагам норма signed mean one-step bias:

| H32 checkpoint arm | $\mathbb E_k\|b_{k,T}^{TF}\|$ | $\mathbb E_k\|b_{k,R}^{TF}\|$ |
|---|---:|---:|
| production `aperture` | 0.13267 mm | 0.37851 mrad |
| diagnostic `drive_error` | 0.11404 mm | 0.38759 mrad |

Накопленный teacher-forced bias и фактический terminal AR bias:

| Arm | $B_{32,T}^{TF}$, mm | $B_{32,R}^{TF}$, mrad | $b_{32,T}^{AR}$, mm | $b_{32,R}^{AR}$, mrad | cosine T / R |
|---|---:|---:|---:|---:|---:|
| `aperture` | [1.528, -0.043, 2.828] | [-1.320, 2.408, -0.786] | [0.691, 0.085, 2.552] | [-1.331, 2.014, -2.058] | 0.9724 / 0.9077 |
| `drive_error` | [1.078, 0.004, 2.740] | [-1.368, 2.241, -1.995] | [0.762, 0.058, 2.621] | [-1.472, 1.789, -3.110] | 0.9956 / 0.9562 |

95% hierarchical bootstrap CI для production cumulative translation равны

$$
B_{32,T}^{TF,aperture}:
([0.708,2.379],[-0.885,0.795],[1.962,3.729])\ {\rm mm},
$$

а для diagnostic control

$$
B_{32,T}^{TF,drive}:
([0.230,2.151],[-0.998,1.076],[1.373,4.107])\ {\rm mm}.
$$

Band-wise signed means ниже даны в порядке
$[v_x,v_y,v_z,\omega_x,\omega_y,\omega_z]$, в mm/mrad на один шаг:

| Arm | Steps 1--8 | Steps 9--16 | Steps 17--24 | Steps 25--32 |
|---|---:|---:|---:|---:|
| `aperture` | [0.0461, -0.0465, -0.0174, -0.0335, 0.0764, 0.1297] | [0.0407, -0.0060, 0.0702, -0.0203, 0.1204, -0.0478] | [0.0654, 0.0422, 0.1284, -0.1011, 0.1624, -0.0835] | [0.0415, 0.0185, 0.1652, -0.0046, -0.0064, -0.0392] |
| `drive_error` | [0.0445, -0.0275, 0.0742, -0.0272, 0.1004, 0.1195] | [0.0254, 0.0038, 0.1124, -0.0135, 0.1156, -0.0661] | [0.0500, 0.0366, 0.0957, -0.1033, 0.1218, -0.1537] | [0.0177, 0.0026, 0.0562, -0.0203, -0.0150, -0.0986] |

По строгому правилу одинакового знака с CI, не содержащим ноль, минимум в трёх
из четырёх bands устойчивы:

- production `aperture`: $v_x>0$ и $v_z>0$;
- diagnostic `drive_error`: только $v_z>0$.

Signed spatial translation $\hat p-p^*$ в gripper frame также сохранена
отдельно. Сумма её teacher-forced mean по 32 шагам равна
[0.922, -0.677, -0.597] mm для `aperture` и
[1.663, -2.743, -3.135] mm для `drive_error`. Отличие от body-frame
$SE(3)$-log ожидаемо из-за изменения frame вдоль trajectories.

Высокое совпадение направлений $B_{32}^{TF}$ и $b_{32}^{AR}$, особенно
устойчивый положительный $z$-bias, показывает, что terminal drift в
существенной части согласуется с интегрированием малого систематического
one-step pose bias. Простое улучшение aggregate one-step $d_X$ этот signed
bias не устранило.

![Signed and cumulative pose bias](../runs/contact-composition-diagnostics-v2/signed_pose_bias.png)

### 19.6 Preserve-vs-fresh Markov/history test

На тех же состояниях сравнивались:

$$
D_{\rm cold}=d_X(x_{k+1}^{preserve},x_{k+1}^{fresh,cold}),
$$

$$
D_{\rm warm}=d_X(x_{k+1}^{preserve},x_{k+1}^{fresh,warm}),
$$

$$
D_{\rm precond}=d_X(x_k^{warm},x_k),
\qquad
E_{\rm local}^{drive}=d_X(R_\theta^{drive}(x_k),x_{k+1}^{preserve}).
$$

`cold` восстанавливает только $(q_k,r_k)$, обнуляет velocities и сразу подаёт
следующую command. `warm` сначала settles восстановленное состояние на текущей
command, затем выполняет тот же increment. Preserve, cold и warm branches
повторялись для оценки repeatability.

Ниже equal-object means; translation дана в mm, rotation в mrad,
$J$ — joint RMSE, нормированный на travel:

| Comparison | $d_X$ | T, mm | R, mrad | J |
|---|---:|---:|---:|---:|
| untouched continuous repeat | 0.034680 | 2.9607 | 15.6471 | 0.005466 |
| fresh cold | 0.014947 | 1.3106 | 6.5316 | 0.002938 |
| fresh warm | 0.008767 | 0.5855 | 6.1205 | 0.002095 |
| warm preconditioning drift | 0.005477 | 0.4004 | 3.3350 | 0.001800 |
| cold repeatability | $9.21\cdot10^{-8}$ | 0 | $9.21\cdot10^{-5}$ | 0 |
| warm repeatability | $9.49\cdot10^{-8}$ | 0 | $9.49\cdot10^{-5}$ | 0 |
| local `aperture`, same samples, 3 seeds | 0.031388 | 1.0035 | 5.0293 | 0.028303 |
| local `drive_error`, same samples, 3 seeds | 0.011100 | 0.7226 | 4.6388 | 0.006348 |

Исходный масштаб history effect:

$$
\Gamma_{\rm history}
=\frac{\mathbb E[D_{\rm warm}]}
{\mathbb E[E_{\rm local}^{drive}]}
=0.7898,
\qquad95\%\ CI=[0.5232,1.0857].
$$

По заранее заданному правилу это формально `comparable`. Но untouched
continuous replay сам имеет больший разброс. Поэтому дополнительно, без
вычитания floor, вычислено

$$
\Gamma_{\rm warm/repeat}
=\frac{\mathbb E[D_{\rm warm}]}{\mathbb E[D_{\rm repeat}]}
=0.2528,
\qquad95\%\ CI=[0.1344,0.8807].
$$

Даже верхняя граница CI меньше единицы. Следовательно, warm history discrepancy
не разрешён над variability двух untouched continuous branches. Fresh cold и
warm branches внутри одинакового reset protocol почти детерминированы, но две
независимые continuous histories могут разойтись в чувствительном contact
режиме. Корректный итог этого теста:

$$
\boxed{\text{Markov/history effect на этой выборке noise-limited, а не доказан}.}
$$

Нельзя на основании $\Gamma_{history}$ вводить memory state или менять
solver. При этом снижение $D_{cold}\to D_{warm}$ и ненулевой
$D_{precond}$ подтверждают, что fresh preconditioning является необходимой
частью корректного reset protocol.

![Preserve/fresh Markov distributions](../runs/contact-composition-diagnostics-v2/history_markov.png)

### 19.7 Итог диагностики

1. **Гипотеза чрезмерного local amplification не подтверждена.** Production
   model имеет gain около единицы по object translation/rotation и не превосходит
   simulator по заданному CI-критерию; simulator amplification частично
   noise-limited.
2. **Систематический signed pose bias подтверждён.** У production устойчивы
   положительные $v_x$ и $v_z$, а направление накопленного TF bias хорошо
   совпадает с terminal AR drift. Это наиболее информативный найденный механизм
   composition defect.
3. **Скрытая PhysX history не отделена от repeatability floor.** Warm discrepancy
   сопоставима с local model error, но меньше untouched continuous variability;
   тест не даёт основания менять state или solver.
4. **Production решение не изменено:** `gap + aperture`; retrain, новые losses,
   implicit solver, memory state и новый dataset не запускались.

Полные raw и агрегированные результаты:

- [`results.json`](../runs/contact-composition-diagnostics-v2/results.json) —
  contract, all object/step-band aggregates и 10 000-repeat bootstrap CI;
- [`samples.npz`](../runs/contact-composition-diagnostics-v2/samples.npz) —
  все paired simulator/model measurements, signed errors и masks;
- [`contact_composition_diagnostics.py`](../scripts/contact_composition_diagnostics.py)
  — объединённый headless runner;
- [`test_contact_composition_diagnostics.py`](../tests/test_contact_composition_diagnostics.py)
  — $SE(3)$-log, perturbation, amplification, selection и fail-fast tests.

## 20. No-retrain диагностика quasistatic refinement и split signed bias

### 20.1 Цель и зафиксированный contract

Этот эксперимент проверяет две гипотезы до любых новых изменений SRNO:

1. является ли simulator-defined closure map устойчивым к измельчению load
   increments $N=32\to64\to128$ и к более строгому settling;
2. присутствует ли один и тот же signed one-step pose bias уже на train, либо он
   возникает только на unseen objects.

Модель не переобучалась и не изменялась. Production-вариант остался
`contact_features=gap`, `global_conditioning=aperture`. Использованы три
существующих H32 checkpoint-а seeds 0, 1, 2. Dataset, SDF, FK, gate, material,
actuators и physics settings также не менялись.

Перед запуском были проверены:

- manifest SHA256
  `c8bddec752f0418e92383fd0d9193e2d70a37845244c4c1c26ed9f9170c3012a`;
- gripper SHA256
  `6f3280535f5fd5bf543da3dba911825710ea73edb2a19b77b4fd4225fa2f02d6`;
- simulator config SHA256
  `e8d87869ff2fe7a0111d6d2c8e4394631e9dd5a8432299ea8a8527bdac09ea01`;
- runtime material и actuator read-back;
- hashes и model modes всех трёх checkpoint-ов.

Во всех расчётах использовалась прежняя state metric

$$
d_X^2(x_1,x_2)=
\frac{\|p_1-p_2\|^2}{L^2}
+\theta(R_1,R_2)^2
+\frac16\sum_{m=1}^{6}
\left(\frac{r_{1,m}-r_{2,m}}{s_m}\right)^2,
$$

где $p$ — translation объекта, $R$ — его orientation,
$\theta(R_1,R_2)$ — geodesic rotation angle, $r\in\mathbb R^6$ — actual
joint configuration, $s_m$ — travel range joint-а, $L$ — gripper length
scale.

### 20.2 Simulator refinement: выборка и процедура

Использовано 12 contact closures: один объект из каждого split и четыре
детерминированные source poses на объект.

| Split | Object | Source pose IDs | Contact onset steps |
|---|---|---:|---:|
| train | `kofe-naturalnyy-rastvorimyy-sublimirovannyy-16125` | 4987, 3482, 824, 4431 | 1, 9, 12, 17 |
| val | `masliny-federici-bez-kostochki-300-g-90215` | 3055, 2593, 637, 4239 | 4, 7, 11, 11 |
| test | `gerkules-ovsyanye-khlopya-400-g-1248` | 1894, 2563, 1004, 778 | 1, 4, 7, 8 |

Позы покрывают квартильные режимы contact onset и были предварительно
отфильтрованы по object-wise p90 числа settling substeps. Для каждой позы
состояние $(q_0,r_0)$ восстанавливалось независимо; максимальная фактическая
ошибка восстановления составила $4.88\cdot10^{-7}$ по $d_X$.

Одна и та же continuous load path задавалась как

$$
\alpha_k=\frac{k}{N},\qquad
\bar r(\alpha_k)=r_{open}+\alpha_k(r_{close}-r_{open}).
$$

Commanded aperture интерполировалась по исходным 33 knots. Поэтому общие точки

$$
\frac{k}{32}=\frac{2k}{64}=\frac{4k}{128}
$$

получали строго одинаковую command. Для каждого $N\in\{32,64,128\}$
выполнено по две независимые repeat branches. Дополнительно выполнена одна
ветка $N=32$ со strict settling:

| Criterion | Production | Strict |
|---|---:|---:|
| minimum substeps | 20 | 40 |
| consecutive settled substeps | 10 | 20 |
| position delta | 0.50 mm | 0.25 mm |
| linear velocity | 0.010 m/s | 0.005 m/s |
| angular velocity | 0.100 rad/s | 0.050 rad/s |
| joint velocity | 0.010 rad/s | 0.005 rad/s |

`max_steps=2400`; material, drive и остальные physics settings идентичны.

Основные refinement errors:

$$
E_{32,64}=\frac1{32}\sum_{k=1}^{32}
d_X(x_k^{32},x_{2k}^{64}),
$$

$$
E_{64,128}=\frac1{64}\sum_{k=1}^{64}
d_X(x_k^{64},x_{2k}^{128}),
\qquad
\rho=\frac{E_{64,128}}{E_{32,64}}.
$$

Отдельно вычислялись repeatability floors для каждого $N$ и

$$
E_{settle}=\frac1{32}\sum_{k=1}^{32}
d_X(x_k^{32,prod},x_k^{32,strict}).
$$

### 20.3 Refinement results

Ниже equal-object means. $T$ дан в mm, $R$ в mrad,
$J=\sqrt{\frac16\sum_m(\Delta r_m/s_m)^2}$.

| Comparison | mean $d_X$ | T, mm | R, mrad | J | terminal $d_X$ |
|---|---:|---:|---:|---:|---:|
| $E_{32,64}$ | 0.091231 | 8.1416 | 35.4952 | 0.013228 | 0.525645 |
| $E_{64,128}$ | 0.087329 | 8.0818 | 31.6588 | 0.014084 | 0.387289 |
| repeat $N=32$ | 0.100319 | 9.3588 | 34.4870 | 0.013979 | 0.502419 |
| repeat $N=64$ | 0.054024 | 4.1781 | 32.3799 | 0.011129 | 0.219303 |
| repeat $N=128$ | 0.056721 | 4.9103 | 27.2520 | 0.010446 | 0.199071 |
| production vs strict $N=32$ | 0.070809 | 6.1107 | 36.9390 | 0.012583 | 0.271097 |

Per-object refinement:

| Split | $E_{32,64}$ | $E_{64,128}$ | $\rho$ | $E_{settle}$ |
|---|---:|---:|---:|---:|
| train | 0.155284 | 0.156439 | 1.086887 | 0.095829 |
| val | 0.059075 | 0.047042 | 0.796315 | 0.032466 |
| test | 0.059334 | 0.058507 | 0.986056 | 0.084132 |

Hierarchical bootstrap `object -> trajectory`, 10 000 repetitions, seed 0:

$$
\boxed{
\rho=0.957232,
\qquad95\%\ CI=[0.737004,1.259491].
}
$$

Заранее заданные правила дают:

- не `converged`: $\rho>0.75$, а верхняя CI больше 1;
- не `non-converged`: нижняя CI $0.7370<0.75$;
- формально не `noise-limited`:
  $E_{64,128}=0.087329>1.5E_{repeat}=0.085082$, но превышение составляет
  лишь $0.002247$ по $d_X$.

Поэтому строгая классификация refinement:

$$
\boxed{\texttt{inconclusive}.}
$$

Repeatability здесь того же порядка, что и refinement difference, и поэтому
не позволяет уверенно измерить asymptotic convergence rate. Это не является
доказательством non-convergence.

Для settling получено

$$
\frac{E_{settle}}{E_{32,64}}=1.36906,
\qquad95\%\ CI=[0.72789,2.36404].
$$

Но $E_{settle}=0.070809<1.5E_{repeat}^{32}=0.150478$, поэтому по
предварительно зафиксированному правилу settling **не классифицирован как
materially sensitive**.

Некоторые ветки достигли `max_steps` в поздней части closure. Raw validity:

| Paired curve | Valid / total state comparisons |
|---|---:|
| $32\leftrightarrow64$ | 763 / 768 |
| $64\leftrightarrow128$ | 1524 / 1536 |
| repeat $N=32$ | 384 / 384 |
| repeat $N=64$ | 758 / 768 |
| repeat $N=128$ | 1513 / 1536 |
| production vs strict | 346 / 384 |

Неполные trajectories не заполнялись искусственно: их masks сохранены, а
trajectory-level mean использовался только для полного paired path. Если одна
из двух production repeats оставалась полной, она продолжала участвовать в
refinement mean.

![Quasistatic refinement](../runs/quasistatic-refinement-bias-v1/quasistatic_refinement_bias.png)

### 20.4 Train/validation/test signed bias

На всех 2 800 trajectories material-v2 dataset выполнены teacher-forced и
autoregressive H32 inference для каждого из трёх production checkpoints:
2 200 train, 300 validation и 300 test trajectories. Всего получено 268 800
one-step comparisons для TF и столько же состояний для AR.

Teacher-forced signed pose error определён как

$$
e_k^{TF}=
\Log\!\left[(q_{k+1}^*)^{-1}
R_\theta(x_k^*,\bar a_{k+1})_q\right]
=\begin{bmatrix}v_k\\\omega_k\end{bmatrix},
$$

$$
b_{S,k}^{TF}=\mathbb E_S[e_k^{TF}],
\qquad
B_{S,32}^{TF}=\sum_{k=0}^{31}b_{S,k}^{TF},
\quad S\in\{train,val,test\}.
$$

Autoregressive control:

$$
b_{S,k}^{AR}=\mathbb E_S\left[
\Log((q_k^*)^{-1}\hat q_k)\right].
$$

Ниже signed band means в порядке
$[v_x,v_y,v_z,\omega_x,\omega_y,\omega_z]$, в mm/mrad **на один шаг**.

| Split | Steps 1--8 | Steps 9--16 | Steps 17--24 | Steps 25--32 |
|---|---:|---:|---:|---:|
| train | [0.0026, -0.0235, 0.0004, 0.0341, 0.0678, -0.0936] | [-0.0056, -0.0323, 0.1145, -0.0512, -0.0133, -0.0944] | [0.0459, -0.0051, 0.2150, 0.0048, 0.0995, -0.0248] | [0.0497, -0.0147, 0.3133, 0.0559, 0.0285, 0.1824] |
| val | [0.0143, -0.0070, -0.0275, 0.0166, 0.0006, 0.2079] | [0.0162, -0.0064, 0.0471, 0.1441, 0.3937, 0.0509] | [0.0245, -0.0215, 0.1132, -0.1143, 0.2842, -0.2683] | [0.0634, 0.0017, 0.1288, -0.2383, 0.0233, -0.0640] |
| test | [0.0780, -0.0853, -0.0069, -0.0839, 0.1516, 0.0502] | [0.0654, -0.0054, 0.0935, -0.1842, -0.1528, -0.1458] | [0.1063, 0.1056, 0.1436, -0.0879, 0.0412, 0.1021] | [0.0146, 0.0080, 0.2156, 0.2181, -0.1410, -0.1312] |

Накопленный TF bias и фактический terminal AR bias:

| Split | $B_{32,T}^{TF}$, mm | $B_{32,R}^{TF}$, mrad | $b_{32,T}^{AR}$, mm | $b_{32,R}^{AR}$, mrad |
|---|---:|---:|---:|---:|
| train | [0.740, -0.605, 5.145] | [0.349, 1.461, -0.244] | [0.588, -0.406, 4.949] | [0.260, 1.249, -0.058] |
| val | [0.947, -0.265, 2.093] | [-1.537, 5.615, -0.589] | [0.619, -0.021, 1.695] | [-1.177, 4.864, -0.767] |
| test | [2.115, 0.183, 3.566] | [-1.104, -0.808, -0.998] | [0.762, 0.196, 3.405] | [-1.499, -0.823, -3.341] |

Hierarchical bootstrap `model seed -> object -> trajectory`, 10 000
repetitions, показывает устойчивые компоненты — CI одного знака минимум в трёх
из четырёх step bands:

- train: $v_z>0$;
- validation: $v_z>0$;
- test: $v_x>0$ и $v_z>0$.

Для общей компоненты $v_z$ cumulative estimates и 95% CI:

$$
B_{train,32,v_z}^{TF}=5.145\ [3.957,6.424]\ {\rm mm},
$$

$$
B_{val,32,v_z}^{TF}=2.093\ [0.830,3.440]\ {\rm mm},
$$

$$
B_{test,32,v_z}^{TF}=3.566\ [2.386,4.965]\ {\rm mm}.
$$

На test дополнительно

$$
B_{test,32,v_x}^{TF}=2.115\ [0.276,3.941]\ {\rm mm}.
$$

Следовательно, положительный $v_z$-bias не является только unseen-object
generalization effect: он уже присутствует на train и затем наблюдается на val
и test. Направление terminal AR $v_z$ совпадает с accumulated TF bias на всех
трёх split-ах.

![Split signed bias](../runs/quasistatic-refinement-bias-v1/split_signed_bias.png)

### 20.5 Итоговая классификация

По заданному до запуска decision rule:

1. refinement не классифицирован ни как converged, ни как non-converged, а
   оказался около repeatability floor;
2. settling sensitivity поверх repeatability floor не подтверждена;
3. общий train/val/test $v_z>0$ signed bias подтверждён;
4. однако ветка `explicit_update_law_candidate` разрешена только после
   подтверждённого convergence refinement.

Поэтому итог остаётся

$$
\boxed{\texttt{inconclusive}.}
$$

Данные совместимы с гипотезой систематического defect explicit update law, но
данный эксперимент не отделил её от simulator repeatability/discretization с
достаточной статистической уверенностью. Implicit resolvent не реализовывался,
архитектура не менялась, retrain не запускался; production остаётся
`gap + aperture`.

Артефакты эксперимента:

- [`results.json`](../runs/quasistatic-refinement-bias-v1/results.json) — полный
  contract, агрегаты и 95% bootstrap CI;
- [`samples.npz`](../runs/quasistatic-refinement-bias-v1/samples.npz) — все
  simulator states, masks и signed TF/AR errors;
- [`quasistatic_refinement_bias.py`](../scripts/quasistatic_refinement_bias.py)
  — resumable headless runner с RAM watchdog 14 GiB;
- [`test_quasistatic_refinement_bias.py`](../tests/test_quasistatic_refinement_bias.py)
  — command alignment, quaternion invariance, synthetic refinement/bias,
  deterministic selection и classification tests.

## 21. Frozen local checkpoint против H32: источник signed bias

### 21.1 Вопрос эксперимента

Предыдущая диагностика измеряла signed teacher-forced bias только после H32
rollout training. Поэтому оставались две причины:

$$
\text{bias локального learned operator}
\quad\text{или}\quad
\text{bias, внесённый rollout fine-tuning}.
$$

Чтобы разделить их без retrain, для каждого seed сравнивались соответствующие
frozen checkpoints:

$$
\theta_{local}
=\texttt{runs/ablation-jq-local/baseline/seed-\{0,1,2\}/best-local.pt},
$$

$$
\theta_{H32}
=\texttt{runs/ablation-actuator-rollout/aperture/seed-\{0,1,2\}/}
\texttt{best-rollout-h32.pt}.
$$

Обе руки имеют `contact_features=gap`,
`global_conditioning=aperture`, одинаковые architecture, manifest, gripper и
seed. H32 checkpoints были инициализированы именно этими local checkpoints.
Runner проверил hashes, stages, horizon и sample ordering. Никакого обучения,
изменения весов или запуска Isaac Sim не выполнялось.

### 21.2 Величины и выборка

На одних и тех же 2 800 trajectories — 2 200 train, 300 validation и 300 test —
для трёх seeds вычислялось

$$
e_k^{TF}(\theta)
=
\Log\!\left[
(q_{k+1}^{*})^{-1}
R_\theta(x_k^*,\bar a_{k+1})_q
\right]
=\begin{bmatrix}v_k\\\omega_k\end{bmatrix},
$$

$$
b_{S,k}^{TF}(\theta)=\mathbb E_S[e_k^{TF}(\theta)],
\qquad
B_{S,32}^{TF}(\theta)=\sum_{k=0}^{31}b_{S,k}^{TF}(\theta).
$$

Главная paired величина:

$$
\boxed{
\Delta B_{S,32,z}^{TF}
=B_{S,32,z}^{TF}(\theta_{H32})
-B_{S,32,z}^{TF}(\theta_{local}).
}
$$

Для local, H32 и их paired difference использован hierarchical bootstrap
`model seed -> object -> trajectory`, 10 000 repetitions, seed 0. Pairing
сохраняет один и тот же model seed, object и trajectory.

### 21.3 Основной результат

Все значения ниже в mm; в скобках дана 95% bootstrap CI.

| Split | $B_{32,z}^{TF}(local)$ | $B_{32,z}^{TF}(H32)$ | $\Delta B_{32,z}^{TF}=H32-local$ | Вывод |
|---|---:|---:|---:|---|
| train | 2.377 [1.243, 3.640] | 5.145 [3.953, 6.481] | **+2.768 [2.059, 3.618]** | bias уже есть в local и усиливается rollout |
| val | 0.160 [-1.280, 1.601] | 2.093 [0.817, 3.473] | **+1.932 [1.106, 2.814]** | bias создаётся rollout training |
| test | -0.180 [-1.666, 1.433] | 3.566 [2.400, 4.929] | **+3.746 [2.732, 4.832]** | bias создаётся rollout training |

По seeds результат однороден:

| Split | Local $B_{32,z}^{TF}$, mm, seeds 0/1/2 | H32 $B_{32,z}^{TF}$, mm, seeds 0/1/2 |
|---|---:|---:|
| train | [2.551, 2.544, 2.036] | [4.799, 4.961, 5.674] |
| val | [0.160, 0.211, 0.110] | [1.849, 2.056, 2.372] |
| test | [-0.338, -0.268, 0.067] | [3.369, 3.110, 4.219] |

Rollout increment объясняет 53.8% итогового H32 $z$-bias на train, 92.3% на
validation и 105.0% на test; значение выше 100% на test возникает потому, что
local mean слегка отрицателен.

Band-wise $v_z$ в mm на один шаг показывает, где появляется отличие:

| Split/arm | Steps 1--8 | Steps 9--16 | Steps 17--24 | Steps 25--32 |
|---|---:|---:|---:|---:|
| train local | 0.0964 | 0.0621 | 0.0368 | 0.1018 |
| train H32 | 0.0004 | 0.1145 | 0.2150 | 0.3133 |
| train H32-local | -0.0961 | 0.0524 | 0.1782 | 0.2115 |
| val local | 0.0691 | 0.0130 | -0.0254 | -0.0366 |
| val H32 | -0.0275 | 0.0471 | 0.1132 | 0.1288 |
| val H32-local | -0.0966 | 0.0341 | 0.1386 | 0.1654 |
| test local | 0.0947 | -0.0018 | -0.0572 | -0.0582 |
| test H32 | -0.0069 | 0.0935 | 0.1436 | 0.2156 |
| test H32-local | -0.1016 | 0.0953 | 0.2007 | 0.2738 |

То есть rollout fine-tuning не просто добавляет constant offset: на всех split
он сдвигает поздние contact increments, особенно steps 17--32, в устойчивое
положительное $v_z$-направление.

![Frozen local vs H32 signed bias](../runs/local-vs-h32-signed-bias-v1/local_vs_h32_signed_bias.png)

### 21.4 Диагноз

Идеальная бинарная картина из гипотезы выполняется на validation и test:

$$
B_{32,z}^{TF}(local)\approx0,
\qquad
B_{32,z}^{TF}(H32)>0,
\qquad
\Delta B_{32,z}^{TF}>0.
$$

На train local checkpoint уже имеет подтверждённый положительный bias. Поэтому
архитектурный/local-objective источник полностью исключить нельзя. Однако
paired increment rollout training строго положителен на **каждом** split и
примерно удваивает train bias:

$$
2.377\ {\rm mm}\longrightarrow5.145\ {\rm mm}.
$$

Следовательно, наиболее точный итог:

$$
\boxed{
\text{local train bias существует, но rollout fine-tuning является}
\atop
\text{доказанным причинным источником большей части H32 signed drift.}
}
$$

Это даёт основание первым следующим обучающим ablation проверять local
anchoring

$$
\mathcal L=\mathcal L_{AR}+\lambda_{TF}\mathcal L_{TF},
\qquad
\mathcal L_{TF}=\frac1N\sum_{k=0}^{N-1}
d_X^2\!\left(
R_\theta(x_k^*,\bar a_{k+1}),x_{k+1}^*
\right),
$$

а не начинать с полной смены архитектуры. В рамках данного теста этот loss не
добавлялся и retrain не выполнялся.

Артефакты:

- [`results.json`](../runs/local-vs-h32-signed-bias-v1/results.json) — полный
  paired bootstrap, band/components, per-seed и per-object результаты;
- [`samples.npz`](../runs/local-vs-h32-signed-bias-v1/samples.npz) — frozen
  local и H32 signed errors на одинаковых trajectories;
- [`local_vs_h32_signed_bias.png`](../runs/local-vs-h32-signed-bias-v1/local_vs_h32_signed_bias.png)
  — cumulative curves и paired seeds;
- [`compare_local_h32_signed_bias.py`](../scripts/compare_local_h32_signed_bias.py)
  — inference-only runner;
- [`test_local_h32_signed_bias.py`](../tests/test_local_h32_signed_bias.py) —
  indexing и classification tests.

## 22. Detached pushforward training со stop-gradient

### 22.1 Что изменялось

После диагностики раздела 21 был проверен минимальный способ показать модели
собственный one-step distribution shift без полного differentiable unroll.
Архитектура `gap + aperture`, material-v2 dataset, geometry, loss weights и
frozen local initialization не менялись. Эксперимент выполнен для seed 0.

Для каждого допустимого triplet

$$
x_{k-1}^*\longrightarrow x_k^*\longrightarrow x_{k+1}^*
$$

считались две обучаемые ветки. Обычная local-ветка:

$$
\hat x_{k+1}^{\rm loc}
=R_\theta(x_k^*,\bar a_{k+1},\phi).
$$

Pushforward-ветка сначала строила model-induced вход с остановленным
градиентом:

$$
\tilde x_k
=\operatorname{sg}\!\left[
R_\theta(x_{k-1}^*,\bar a_k,\phi)
\right],
$$

$$
\hat x_{k+1}^{\rm PF}
=R_\theta(\tilde x_k,\bar a_{k+1},\phi).
$$

Итоговый objective:

$$
\boxed{
\mathcal L
=\mathcal L_{\rm loc}
+\mathcal L_{\rm PF}
+\lambda_K\mathcal L_K,
\qquad \lambda_{\rm PF}=\lambda_K=1.
}
$$

Градиент проходил через local-вызов и второй PF-вызов, но не через вызов,
создающий $\tilde x_k$. В отличие от прежнего H4--H32 BPTT, это был только
двухшаговый training signal; autoregressive rollout не использовался для
model selection. Best checkpoint выбирался по

$$
m_{val}=\frac12\left(
\mathbb E[d_X^{\rm loc}]
+\mathbb E[d_X^{\rm PF}]
\right).
$$

### 22.2 Local и pushforward validation

На полном active validation set:

| Модель | Local $d_X$ | PF $d_X$ | $m_{val}$ |
|---|---:|---:|---:|
| frozen local | 0.031132 | 0.037858 | 0.034495 |
| detached PF | **0.030018** | **0.036752** | **0.033385** |

One-step validation улучшился на 3.58%, то есть ограничение «не ухудшить local
ошибку более чем на 5%» было выполнено с запасом. Selection metric улучшился
на 3.22%.

### 22.3 Полный H32 inference

После обучения три checkpoints — frozen local, прежний unconstrained H32 BPTT
и detached PF — были применены autoregressively на одинаковых trajectories.

| Модель | Train terminal $d_X$ | Val terminal $d_X$ | Test terminal $d_X$ |
|---|---:|---:|---:|
| frozen local | 0.313571 | 0.234227 | 0.263065 |
| old unconstrained BPTT | **0.276592** | **0.171675** | **0.204097** |
| detached PF | 0.308230 | 0.225595 | 0.255164 |

Test terminal decomposition:

| Модель | Translation, mm | Rotation, rad | Joint RMS/travel | Aperture, mm | Equal-object mean max penetration, mm |
|---|---:|---:|---:|---:|---:|
| frozen local | 23.913 | 0.109950 | 0.080693 | 3.353 | 15.257 |
| old unconstrained BPTT | **16.510** | **0.102904** | **0.071548** | **2.851** | **14.381** |
| detached PF | 22.879 | 0.108526 | 0.081450 | 3.432 | 16.266 |

Paired hierarchical bootstrap по object и trajectory для единственного seed
дал:

$$
E_{test}^{PF}-E_{test}^{local}
=-0.007901,
\qquad 95\%\ {\rm CI}=[-0.017557,-0.001483],
$$

$$
E_{test}^{PF}-E_{test}^{old\ BPTT}
=+0.051067,
\qquad 95\%\ {\rm CI}=[0.030362,0.072249].
$$

Таким образом, detached PF статистически улучшил frozen local, но остался
существенно хуже полного BPTT.

### 22.4 Signed bias и вывод

Для вертикальной accumulated teacher-forced компоненты

$$
B_{32,z}^{TF}
=\sum_{k=0}^{31}\mathbb E\!\left[
\Log\!\left((q_{k+1}^*)^{-1}
R_\theta(x_k^*,\bar a_{k+1})_q\right)_z
\right]
$$

получены значения в mm:

| Модель | Train | Val | Test |
|---|---:|---:|---:|
| frozen local | 2.551 | 0.160 | -0.338 |
| old unconstrained BPTT | 4.799 | 1.849 | 3.369 |
| detached PF | 2.862 | 0.384 | 0.043 |

Detached PF сохранил local signed bias заметно лучше прежнего BPTT, но получил
лишь небольшое H32 улучшение. Следовательно, stop-gradient стабилизировал
локальный оператор, однако двухшагового signal оказалось недостаточно для
композиции на 32 шага.

![Detached pushforward ablation](../runs/ablation-pushforward-stopgrad/pushforward_stopgrad_ablation.png)

Полные результаты: [`results.json`](../runs/ablation-pushforward-stopgrad/results.json),
raw samples: [`samples.npz`](../runs/ablation-pushforward-stopgrad/samples.npz).

## 23. Full BPTT с local-resolvent trust region

### 23.1 Что изменялось

Следующий эксперимент вернул полный differentiable curriculum

$$
H4\longrightarrow H8\longrightarrow H16\longrightarrow H32,
$$

но ограничил функциональное отклонение fine-tuned operator от исходного
frozen local operator $R_{\theta_0}$ на реальных teacher-forced contact
states:

$$
D_{\rm loc}(\theta)
=\mathbb E_{(x_k^*,\bar a_{k+1})}
d_X^2\!\left(
R_\theta(x_k^*,\bar a_{k+1}),
\operatorname{sg}[R_{\theta_0}(x_k^*,\bar a_{k+1})]
\right).
$$

Заранее, до test evaluation, был зафиксирован радиус material-v2 simulator
floor:

$$
\boxed{
D_{\rm loc}\le\varepsilon^2,
\qquad
\varepsilon=0.0029910004.
}
$$

Пусть $c=D_{\rm loc}-\varepsilon^2$. Training objective имел вид

$$
\mathcal L
=\mathcal L_{\rm rollout}
+\lambda_K\mathcal L_K
+\Psi(c;\mu,\rho),
$$

где использовался projected Hestenes--Powell--Rockafellar term

$$
\Psi(c;\mu,\rho)
=\frac{\max(0,\mu+\rho c)^2-\mu^2}{2\rho},
$$

$$
\mu\leftarrow\max(0,\mu+\rho c),
\qquad
\rho=\varepsilon^{-2}=111780.758.
$$

Ограничение пересчитывалось точно по всем 62 231 active train transitions
после каждой эпохи. Архитектура, `gap + aperture`, material-v2 data, SDF, FK,
gate, feasibility loss, optimizer и LR не менялись. Эксперимент выполнен для
seed 0 из того же frozen local checkpoint, что и раздел 22.

### 23.2 Результаты curriculum и допустимость

| Horizon | Val terminal $d_X$ | $D_{\rm loc}/\varepsilon^2$ | Выбранный checkpoint |
|---:|---:|---:|---|
| H4 | 0.015226 | 0.9749 | допустимый H4 update |
| H8 | 0.034694 | 0.8913 | допустимый H8 update |
| H16 | 0.068681 | 0.8913 | допустимого H16 update нет; fallback к H8 |
| H32 | 0.236152 | 0.8913 | допустимого H32 update нет; fallback к H8 |

Curriculum был выполнен полностью, но ни один улучшавший H16/H32 update не
удовлетворил trust region. Поэтому итоговый H32 checkpoint намеренно содержит
последние допустимые H8 weights, применённые на 32 шагах.

### 23.3 Сравнение с frozen local и старым BPTT

| Модель | Val terminal $d_X$ | Test terminal $d_X$ |
|---|---:|---:|
| frozen local | 0.234227 | 0.263065 |
| old unconstrained BPTT | **0.171675** | **0.204097** |
| trust-region BPTT | 0.236152 | 0.262169 |

Trust-region улучшил test относительно frozen local только на

$$
0.263065-0.262169=0.000896\quad (0.34\%),
$$

но оказался хуже old BPTT на $0.058072$. На validation он ухудшил frozen
local на $0.001925$.

Test terminal components:

| Модель | Translation, mm | Rotation, rad | Joint RMS/travel | Aperture, mm | Global max penetration, mm |
|---|---:|---:|---:|---:|---:|
| frozen local | 23.913 | 0.109949 | 0.080693 | 3.353 | 18.349 |
| old unconstrained BPTT | **16.510** | **0.102904** | **0.071548** | **2.851** | 16.884 |
| trust-region BPTT | 23.997 | 0.104732 | 0.079991 | 3.334 | **16.266** |

### 23.4 Signed bias и итог

Для той же величины $B_{32,z}^{TF}$ из раздела 22 получено, mm:

| Модель | Train | Val | Test |
|---|---:|---:|---:|
| frozen local | 2.551 | 0.160 | -0.338 |
| old unconstrained BPTT | 4.799 | 1.849 | 3.369 |
| trust-region BPTT | 2.713 | 0.209 | -0.499 |

Относительно frozen local trust-region изменил signed bias лишь на
$+0.048$ mm на validation и $-0.162$ mm на test, тогда как old BPTT
изменил его на $+1.689$ и $+3.707$ mm соответственно.

Эксперимент показал явный trade-off:

$$
\boxed{
\text{радиус simulator floor сохраняет local resolvent и signed bias,}
\atop
\text{но блокирует почти всё полезное H16--H32 улучшение.}
}
$$

Следовательно, большой H32 gain старого BPTT достигается за пределами
$D_{\rm loc}\le\varepsilon^2$. На одном seed нельзя делать межseedовый
статистический вывод, однако training dynamics однозначно показывает, что
выбранный физически малый радиус слишком строг для long-horizon adaptation.

![Local-resolvent trust-region ablation](../runs/ablation-local-trust-region/seed-0/trust_region_ablation.png)

Полные результаты:
[`results.json`](../runs/ablation-local-trust-region/seed-0/results.json), raw
samples: [`samples.npz`](../runs/ablation-local-trust-region/seed-0/samples.npz),
конфигурация:
[`ablation-local-trust-region.toml`](../configs/ablation-local-trust-region.toml).

## 24. Full BPTT с ограничением true one-step physics error

### 24.1 Чистое изменение относительно раздела 23

В этом ablation архитектура `gap + aperture`, material-v2 dataset, SDF, FK,
contact gate, state/feasibility losses, optimizer и differentiable curriculum

$$
H4\longrightarrow H8\longrightarrow H16\longrightarrow H32
$$

не менялись. Единственным изменением стало определение trust-region
constraint. В разделе 23 ограничивалось расстояние до выхода frozen local
operator. Теперь frozen-output cache вообще не строился, а использовалась
ошибка относительно настоящего simulator successor:

$$
L_{\rm phys}(\theta)
=\mathbb E_{\mathcal A_{\rm train}}
d_X^2\!\left(
R_\theta(x_k^*,u_{k+1}),x_{k+1}^*
\right),
$$

$$
d_X^2=
\frac{\|\Delta p\|^2}{L^2}
+\theta(R,R^*)^2
+\frac16\sum_{m=1}^6
\left(\frac{\Delta r_m}{s_m}\right)^2,
$$

$$
\boxed{
c(\theta)=L_{\rm phys}(\theta)-L_{\rm phys}(\theta_0)\le0.
}
$$

Здесь $\mathcal A_{\rm train}$ — полный набор из 62 231 active train
transitions, $L$ — gripper length scale, $s_m$ — travel range joint
$m$, а $\theta_0$ — один и тот же frozen local seed-0 checkpoint.
Feasibility не входит в $L_{\rm phys}$, но остаётся в исходном rollout
objective.

Использован тот же projected Hestenes--Powell--Rockafellar term:

$$
\mathcal L
=\mathcal L_{\rm rollout}+\lambda_K\mathcal L_K
+\Psi(c;\mu,\rho),
$$

$$
\Psi(c;\mu,\rho)
=\frac{\max(0,\mu+\rho c)^2-\mu^2}{2\rho},
\qquad
\mu\leftarrow\max(0,\mu+\rho c).
$$

Scale-aware coefficient вычислен из baseline:

$$
L_{\rm phys}(\theta_0)=0.00376753082,
\qquad
\rho=L_{\rm phys}(\theta_0)^{-1}=265.425831.
$$

Baseline decomposition:

| Train component of $d_X^2$ | Frozen local $\theta_0$ |
|---|---:|
| Translation | 0.001922173 |
| Rotation | 0.000486931 |
| Joints | 0.001358426 |
| **Total $L_{\rm phys}$** | **0.003767531** |

После каждой эпохи constraint пересчитывался sample-weighted по всем 62 231
active train transitions. Checkpoint считался допустимым при

$$
L_{\rm phys}(\theta)\le L_{\rm phys}(\theta_0)+10^{-8};
$$

среди допустимых выбирался только минимальный validation terminal $d_X$.
Test split в model selection не использовался. Выполнен один заранее
зафиксированный run, seed 0; control arms не переобучались.

### 24.2 Curriculum и активность ограничения

| Horizon | Best epoch | Val terminal $d_X$ | $L_{\rm phys}/L_{\rm phys}(\theta_0)$ |
|---:|---:|---:|---:|
| H4 | 0 | 0.015003 | 0.972815 |
| H8 | 31 | 0.032924 | 0.952550 |
| H16 | 40 | 0.064539 | 0.962372 |
| H32 | 64 | **0.173154** | **0.930164** |

В отличие от frozen-output trust region из раздела 23, новый constraint не
заблокировал H16--H32 adaptation. Во время обучения он был активен: отдельные
epochs давали $L_{\rm phys}/L_0>1$, после чего возрастал multiplier
$\mu$. Однако выбранный H32 checkpoint оказался строго допустимым:

$$
L_{\rm phys}^{\rm new,train}=0.003504420
=0.930164\,L_{\rm phys}(\theta_0).
$$

Это улучшение aggregate train one-step loss на $6.98\%$ относительно local
и на $2.17\%$ относительно old unconstrained BPTT. Следовательно, финальный
constraint не был активной границей, но изменил траекторию оптимизации.

![Physical one-step trust training](../runs/ablation-physical-one-step-trust/seed-0/physical_one_step_trust_training.png)

### 24.3 True one-step error на полных active splits

В таблице $T,R,J$ — соответственно translation, rotation и joint слагаемые
в $d_X^2$, а `mean` — $\mathbb E[d_X]$, не
$\sqrt{\mathbb E[d_X^2]}$.

| Model | Split | Transitions | $L_{\rm phys}=\mathbb E[d_X^2]$ | mean $d_X$ | $T$ | $R$ | $J$ |
|---|---|---:|---:|---:|---:|---:|---:|
| frozen local | train | 62 231 | 0.003767531 | 0.035737 | 0.001922173 | 0.000486931 | 0.001358426 |
| frozen local | val | 9 168 | 0.001758599 | 0.031132 | 0.000309485 | 0.000159592 | 0.001289523 |
| frozen local | test | 9 474 | 0.002794374 | 0.037470 | 0.000631360 | 0.000194120 | 0.001968893 |
| old BPTT | train | 62 231 | 0.003581986 | 0.034158 | 0.001926957 | 0.000488963 | 0.001166066 |
| old BPTT | val | 9 168 | 0.001662474 | 0.030275 | 0.000295673 | 0.000163919 | 0.001202882 |
| old BPTT | test | 9 474 | 0.002306305 | 0.033239 | 0.000633432 | 0.000199512 | 0.001473362 |
| frozen-output trust | train | 62 231 | 0.003756969 | 0.035507 | 0.001925120 | 0.000486892 | 0.001344957 |
| frozen-output trust | val | 9 168 | 0.001752510 | 0.030934 | 0.000314422 | 0.000160168 | 0.001277920 |
| frozen-output trust | test | 9 474 | 0.002778661 | 0.037204 | 0.000637284 | 0.000194417 | 0.001946960 |
| **physical-error trust** | **train** | **62 231** | **0.003504420** | **0.032582** | **0.001927680** | **0.000487597** | **0.001089142** |
| **physical-error trust** | **val** | **9 168** | **0.001628644** | **0.028990** | **0.000294722** | **0.000163248** | **0.001170674** |
| **physical-error trust** | **test** | **9 474** | **0.002293729** | **0.032512** | **0.000629199** | **0.000197171** | **0.001467359** |

По scalar one-step criterion новый вариант является лучшим из четырёх на
train, validation и test. Но преимущество над old BPTT снова главным образом
лежит в joints; train translation даже немного выше, чем у local и old BPTT.

### 24.4 H32 rollout results

Полная inference выполнена на 2 800 trajectories: 2 200 train, 300 validation
и 300 test. Значения checkpoint selection и повторной full inference могут
различаться на $10^{-5}$ из-за порядка batch accumulation; для дальнейшего
сравнения используются full-inference числа.

| Model | Val terminal $d_X$ | Test terminal $d_X$ |
|---|---:|---:|
| frozen local | 0.234227 | 0.263065 |
| old unconstrained BPTT | **0.171675** | **0.204097** |
| frozen-output trust | 0.236152 | 0.262169 |
| physical-error trust | 0.173145 | 0.210555 |

Test terminal decomposition:

| Model | Translation, mm | Rotation, rad | Joint RMS/travel | Aperture, mm | Global max penetration, mm |
|---|---:|---:|---:|---:|---:|
| frozen local | 23.913 | 0.109949 | 0.080693 | 3.353 | 18.349 |
| old unconstrained BPTT | **16.510** | **0.102904** | **0.071548** | **2.851** | 16.884 |
| frozen-output trust | 23.997 | 0.104732 | 0.079991 | 3.334 | **16.266** |
| physical-error trust | 17.127 | 0.103232 | 0.076356 | 3.960 | 17.782 |

Paired object $\to$ trajectory bootstrap, 10 000 repetitions, seed 0:

$$
E_{test}^{new}-E_{test}^{local}
=-0.052510,
\qquad
95\%\ {\rm CI}=[-0.087362,-0.025113],
$$

$$
E_{test}^{new}-E_{test}^{old\ BPTT}
=+0.006458,
\qquad
95\%\ {\rm CI}=[+0.000646,+0.011715].
$$

Таким образом, gain над frozen local статистически подтверждён и составляет
$19.96\%$, но новый вариант статистически хуже old BPTT на $3.16\%$.
Результат одинаков по знаку на каждом из трёх unseen test objects:

| Test object | New - local | New - old BPTT |
|---|---:|---:|
| `voda-pitevaya-prirodnaya-legenda-baykala-750-ml-42674` | -0.088912 | +0.001918 |
| `gerkules-ovsyanye-khlopya-400-g-1248` | -0.043005 | +0.011135 |
| `pechene-sdobnoe-khlebnyy-spas-italyanskoe-s-apelsinovym-vkusom-i-izyumom-230-g-46835` | -0.025614 | +0.006321 |

### 24.5 Signed bias: constraint не исправил систематический drift

Критическая величина оставалась той же:

$$
B_{32}^{TF}
=\sum_{k=0}^{31}\mathbb E\!\left[
\Log\!\left((q_{k+1}^*)^{-1}
R_\theta(x_k^*,u_{k+1})_q\right)
\right],
$$

где первые три компоненты $[v_x,v_y,v_z]$ измеряются в метрах, а последние
$[\omega_x,\omega_y,\omega_z]$ — в радианах. Для наиболее устойчивой
вертикальной компоненты получено:

| Model | Train $B_{32,z}^{TF}$, mm | Val, mm | Test, mm |
|---|---:|---:|---:|
| frozen local | 2.551 | 0.160 | -0.338 |
| old unconstrained BPTT | 4.799 | 1.849 | 3.369 |
| frozen-output trust | 2.713 | 0.209 | -0.499 |
| **physical-error trust** | **5.408** | **2.463** | **3.939** |

Заранее заданный bias-preservation statistic:

$$
D_{\rm bias,S}
=|B_{z,S}^{new}-B_{z,S}^{local}|
-|B_{z,S}^{old}-B_{z,S}^{local}|.
$$

Результаты bootstrap:

$$
D_{\rm bias,val}=+0.614\ {\rm mm},
\qquad 95\%\ {\rm CI}=[+0.188,+1.018]\ {\rm mm},
$$

$$
D_{\rm bias,test}=+0.569\ {\rm mm},
\qquad 95\%\ {\rm CI}=[+0.288,+0.844]\ {\rm mm}.
$$

Обе CI находятся строго выше нуля: physical-error constraint сохранил signed
bias не лучше, а хуже old BPTT. Terminal AR translation bias нового варианта
также имеет положительный $z$: 2.365 mm на validation и 3.818 mm на test.
Полные terminal AR twists нового варианта:

$$
b_{32,val}^{AR}
=([0.762,-0.092,2.365]\ {\rm mm},
[-0.001003,0.004881,-0.001729]\ {\rm rad}),
$$

$$
b_{32,test}^{AR}
=([0.540,0.407,3.818]\ {\rm mm},
[-0.000452,-0.002233,-0.007402]\ {\rm rad}).
$$

### 24.6 Вывод ablation

Эксперимент разделяет два утверждения:

1. Ограничение по настоящему successor значительно менее разрушительно, чем
   ограничение по frozen output: оно допускает полноценное H32 улучшение и
   даёт test $d_X=0.210555$ вместо 0.262169.
2. Scalar aggregate $L_{\rm phys}=\mathbb E[d_X^2]$ недостаточен для
   различения физически корректного и систематически смещённого operator.
   Новый checkpoint имеет лучший one-step loss на каждом split, но больший
   signed $z$-bias и худший H32 test result, чем old BPTT.

Итог заранее заданных критериев:

| Criterion | Result |
|---|---|
| Final train physical constraint | **passed** |
| H32 test лучше frozen local, paired CI | **passed** |
| H32 test лучше old BPTT, paired CI | failed |
| Bias сохранён лучше old BPTT на validation | failed |
| Bias сохранён лучше old BPTT на test | failed |

Следовательно, этот clean control подтверждает ожидаемую проблему: ещё один
scalar training constraint не устраняет направленный pose drift. В рамках
этого ablation модель, data/physics contract и losses дальше не менялись.

![Physical one-step trust ablation](../runs/ablation-physical-one-step-trust/seed-0/physical_one_step_trust_ablation.png)

Полные результаты:
[`results.json`](../runs/ablation-physical-one-step-trust/seed-0/results.json),
raw samples:
[`samples.npz`](../runs/ablation-physical-one-step-trust/seed-0/samples.npz),
baseline contract:
[`physical-baseline-contract.json`](../runs/ablation-physical-one-step-trust/seed-0/physical-baseline-contract.json),
конфигурация:
[`ablation-physical-one-step-trust.toml`](../configs/ablation-physical-one-step-trust.toml).

---

## 25. Почему shared local operator не учил pose и слабо учил joints

Этот раздел фиксирует отдельное root-cause исследование на актуальном
`material-v2` dataset. Source, configs и checkpoints сверялись с предыдущими
разделами отчёта; старые выводы не переносились на новую архитектуру без
повторной проверки. Все local-метрики ниже — object-balanced test metrics на
9 474 active transitions, если явно не указано иное.

### 25.1 Исходный симптом и простые reference predictors

Актуальный L1 checkpoint
[`best-local.pt`](../runs/ablation-operator-depth-local/L1/best-local.pt)
имеет:

| Model | $d_X$ | $d_T$, mm | $d_R$, rad | $d_J$ |
|---|---:|---:|---:|---:|
| L1 shared local | 0.037451 | 1.4033 | 0.006495 | 0.032299 |
| identity pose | - | 1.0559 | 0.006103 | - |

Для L1 медиана отношения pose error к identity error равна 1.163, и только
27.8% test transitions предсказаны лучше identity. Joints принципиально
обучаемы: относительно free-motion predictor медиана отношения ошибок равна
0.387, а 90.6% transitions лучше reference. Поэтому исходная проблема не
сводится к общей неисправности optimizer или data loader.

Ещё более сильный диагностический результат даёт predictor
$\Delta q_{k+1}=\Delta q_k$: mean pose error 0.009285 против 0.011742 у
identity и 0.014678 у L1; 83.1% переходов лучше identity. Между соседними
инкрементами медианный cosine равен 0.996 для translation и 0.990 для
rotation. При этом collector явно обнуляет сохранённые object/joint velocities.
Следовательно, предыдущий pose increment является сильным predictive proxy
локального направления quasi-static contact evolution. Однако этот тест сам по
себе не доказывает, что snapshot state физически немарковский: history может
также выступать полезным inductive bias при разрывной функции и редком покрытии
state space. Это различие отдельно проверено в разделе 27.

### 25.2 Проверенные фундаментальные гипотезы

| Hypothesis | Проверка | Результат |
|---|---|---|
| Pose теряется только из-за редких выбросов | Huber pose loss | translation улучшилась лишь до 1.2938 mm; медианный relative error остался 1.141. Heavy tail реален, но это не root cause |
| Большая integral depth нужна и для pose | clean L1…L4 ablation | лучший L3 aggregate выиграл преимущественно по joints; pose почти не изменился |
| Pose и joints дают противоположные gradients | per-loss gradient cosine | median shared cosine pose–joints = +0.604: систематического конфликта знака нет |
| Pose подавлен масштабом multitask gradients | gradient norms и pose-only fit | joint gradient в 46.2 раза, feasibility в 21.9 раза больше pose; pose-only decoupled дал 1.0206 mm, но rotation осталась хуже identity. Это важный вторичный механизм |
| В `SE(3)` update неверно зашита пространственная связь | замена left spatial exponential на product retraction | translation улучшилась 1.4033 → 1.1645 mm; rotation почти не изменилась. Ошибка реальна, но недостаточна |
| Для joints не хватает actual current contact geometry | `trial_gap` → `trial_current_gap` | $d_J$ 0.032299 → 0.028176, но pose не улучшилась. Это подтверждённая причина части ошибки $r$ |
| History содержит информацию о локальном направлении | conditioning на предыдущий pose increment | pose-only: 0.8829 mm и 0.005631 rad, 81.4% transitions лучше identity. Predictive value подтверждена; физическая неполнота snapshot state этим не доказана |

Heavy-tail диагностика дополнительно показала, что верхние 0.1% train pose
steps дают 66.9% суммы squared pose energy, а верхний 1% — 84.6%; максимальный
translation step достигает 613 mm. Robust loss поэтому оставлен доступной
опцией, но применять его как самостоятельное исправление причин нет оснований.

Пространственный left `SE(3)` update также оказался физически значимым
несоответствием постановке: медиана $|\omega\times p|$ сопоставима с
медианой настоящего $|\Delta p|$. Поэтому rotation residual создавал
искусственный translation около world origin. В product update эти две
величины предсказываются независимо:

$$
p_{k+1}=p_k+\ell\,\hat v_k,
\qquad
R_{k+1}=\exp([\hat\omega_k]_\times)R_k.
$$

### 25.3 Экспериментальный history-dependent ablation

Это controlled architecture ablation, а не принятая замена production-модели.
Проверенная экспериментальная комбинация состоит из четырёх изменений:

1. Contact encoder видит и trial gap, и gap в реально наблюдаемом current
   state. Это отделяет геометрию управляющего trial motion от текущего режима
   контакта.
2. Global head получает безразмерный history feature

   $$
   h_k=\left[
   \frac{p_k-p_{k-1}}{\ell},
   \operatorname{Log}(R_kR_{k-1}^{T})
   \right].
   $$

   Для первого шага истории он равен нулю; в autoregressive rollout
   предыдущим является предыдущий предсказанный state.
3. Pose применяется product retraction выше, без зависимости translation от
   абсолютного положения объекта.
4. На local stage используются gradient-calibrated
   $\lambda_J=0.025$, $\lambda_{feas}=0.05$. На rollout stage веса снова
   равны 1: один набор весов не имеет одинакового смысла для one-step и BPTT.

Результат clean local ablation:

| Model | $d_X$ | $d_T$, mm | $d_R$, rad | $d_J$ |
|---|---:|---:|---:|---:|
| L1 baseline | 0.037451 | 1.4033 | 0.006495 | 0.032299 |
| + current gap | 0.033816 | 1.4050 | 0.006375 | 0.028176 |
| + decoupled pose | 0.034117 | 1.1645 | 0.006381 | 0.029522 |
| history, pose-only | - | 0.8829 | 0.005631 | - |
| **balanced history + current gap + decoupled pose** | **0.029247** | **0.8740** | **0.005730** | **0.025245** |

Относительно исходного L1 комбинированный вариант улучшает test
$d_X$ на 21.9%, translation на 37.7%, rotation на 11.8% и joints на 21.8%.
Медианный pose/identity ratio равен 0.699, 81.5% transitions лучше identity;
для joints медианный contact/free ratio равен 0.288, 94.6% transitions лучше
free-motion predictor. Это первый вариант в серии, который улучшает все
компоненты состояния одновременно.

Gradient diagnostics объясняют, почему это не достигалось простым shared
training: pose и joint gradients в среднем согласованы, но их нормы резко
различаются; одновременно stateless map может усреднять локальные ветви около
разрывов. Loss reweighting лечит первое, history conditioning даёт полезный
directional bias для второго. Физическая немарковость snapshot state здесь ещё
не установлена; conditional-observability test раздела 27 специально проверяет
эту более сильную формулировку.

### 25.4 Короткая проверка совместимости с rollout

Сначала local weights 0.025/0.05 были ошибочно сохранены внутри BPTT. Это
вызвало ожидаемый joint collapse: test terminal $d_J=0.1605$ и
$d_X=0.261$. Эксперимент подтвердил необходимость stage-specific loss
contract. После возврата rollout weights к 1 и короткого curriculum
H4/H8/H16/H32 новый checkpoint не дал статистически содержательного общего
H32 выигрыша над production, но сохранил улучшенную translation и оказался
практически равным по terminal aggregate:

| Test quantity, sample-weighted | production rollout | history rollout | change |
|---|---:|---:|---:|
| mean-step $d_X$ | 0.087772 | 0.089589 | +2.1% |
| H32 $d_X$ | 0.207812 | 0.208063 | +0.1% |
| mean translation | 0.059395 | 0.058518 | -1.5% |
| H32 translation | - | - | -2.2% |
| mean rotation | 0.049590 | 0.052559 | +6.0% |
| H32 rotation | - | - | -0.4% |
| mean joints | 0.027748 | 0.028875 | +4.1% |

После короткого BPTT checkpoint всё ещё превосходит L1 на one-step test:
$d_X=0.032280$, $d_T=0.9279$ mm, $d_R=0.006071$,
$d_J=0.028172$. Следовательно, новая state representation совместима с
rollout и не уничтожается BPTT, но короткий эксперимент не является
доказательством улучшения long-horizon score. Полный стандартный retrain не
запускался: это был бы длинный эксперимент без необходимости для
идентификации local root cause.

### 25.5 Связь с механикой и опубликованными моделями

Полученный результат согласуется не с идеей «просто увеличить MLP», а с
известной структурой контактной динамики:

- [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf)
  вводит signed-distance contact geometry и contact Jacobians вместе с
  complementarity/max-dissipation constraints. Одного trial distance без
  информации о текущем режиме контакта недостаточно.
- [Allen et al.](https://proceedings.mlr.press/v205/allen23a.html) отдельно
  рассматривают разрывную rigid-contact dynamics; усреднение ветвей гладким
  stateless regressor является ожидаемым failure mode.
- [Hochlehnert et al.](https://proceedings.mlr.press/v130/hochlehnert21a.html)
  и [Bianchini et al.](https://proceedings.mlr.press/v229/bianchini23a.html)
  разделяют непрерывную динамику и структурированное contact interaction.
- [Graph Network-based Simulators](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html)
  используют message passing и rollout training; [FIGNet](https://arxiv.org/abs/2212.03574)
  показывает пользу более точного face-level interaction graph. Это
  поддерживает исправление contact representation, но само по себе не заменяет
  отсутствующую history/state variable.
- [Incremental Potential Contact](https://cims.nyu.edu/gcl/papers/2020-IPC.pdf)
  формулирует contact как последовательное incremental решение. Это ближе к
  наблюдаемой в данных path dependence, чем память-независимое отображение
  только из текущей конфигурации.

Эти работы не доказывают конкретную форму шестимерного feature. Его predictive
value здесь установлена собственным predictor test и controlled ablation.
Результат не следует интерпретировать как доказательство скрытой памяти PhysX:
сохранённые velocities нулевые, а reset diagnostics не отделяют simulator
hidden state от repeatability floor. Корректный узкий вывод — $q_{k-1}$
помогает предсказывать локальное направление; определяет ли то же направление
точный snapshot $(q_k,r_k,u_k)$, имеющийся dataset проверить плотно не
позволяет.

### 25.6 Реализация и воспроизводимость

Для ablation использовались следующие экспериментальные contracts:

- `ModelConfig.history_conditioning = "pose_delta"`;
- `ModelConfig.contact_features = "trial_current_gap"`;
- `ModelConfig.pose_update = "decoupled"`;
- `LossConfig.local_lambda_joints` и
  `LossConfig.local_lambda_feasibility` для local-only overrides;
- `LocalTransitionBatch.previous`, collator и autoregressive history tracking;
- numerically safe `so3_log_vector` и optional Huber pose penalty.

Полный экспериментальный config, не принятый как production:
[`srno-r-material-v2-history.toml`](../configs/srno-r-material-v2-history.toml).
Он записывает в отдельный output directory и не перезаписывает production
checkpoint. Подтверждающие configs:
[`ablation-local-balanced-history.toml`](../configs/ablation-local-balanced-history.toml),
[`ablation-history-short-rollout-full-loss.toml`](../configs/ablation-history-short-rollout-full-loss.toml).

Checkpoints:

- local experimental candidate:
  [`best-local.pt`](../runs/ablation-local-balanced-history/best-local.pt);
- короткий full-loss rollout:
  [`best-rollout.pt`](../runs/ablation-history-short-rollout-full-loss/best-rollout.pt).

Raw diagnostics:
[`gradient scales`](../runs/diagnostics-loss-gradient-scales.json),
[`gradient conflict`](../runs/diagnostics-gradient-conflict.json),
[`increment alignment`](../runs/diagnostics-pose-increment-alignment.json),
[`rollout comparison`](../runs/diagnostics-history-vs-production-rollout.json).

После обновления stale checkpoint adapter в
`contact_composition_diagnostics.py` полный test suite проходит:
**91 passed**. Старый `drive_error` checkpoint намеренно не маскируется под
новый aperture contract и теперь отклоняется явной ошибкой.

Статус после ablation: history-dependent вариант и его checkpoints сохранены
как исследовательские артефакты, но основная архитектура и
`configs/srno-r-material-v2.toml` остаются на линии последнего коммита. Для
следующих экспериментов эта архитектурная ветка не считается новым baseline.

### 25.7 Итог

History ablation подтверждает, что локальное направление неэффективно
восстанавливается исходным shared operator, но не устанавливает
фундаментальную неполноту Markov state. Более поздний тест раздела 27 не нашёл
разных successors у machine-near повторов и показал слишком разреженное
покрытие val/test для такого доказательства. Для $r$ actual current gap
остаётся подтверждённым полезным сигналом; дополнительно pose подавлялся
несбалансированными gradient scales. Left spatial pose update и heavy-tail
squared loss были реальными, но вторичными дефектами.

Экспериментальный candidate `history pose delta + trial/current gap +
decoupled pose + stage-specific losses` устраняет все четыре дефекта в
one-step ablation и одновременно улучшает pose и joints. Но короткая rollout
проверка не дала H32-выигрыша, поэтому архитектура не принимается как новый
baseline. Дальнейший дорогой полный rollout retrain нужен только для ответа на
отдельный вопрос о новом H32 optimum, а не для установления причины исходного
local failure.

---

## 26. Local ablation с удалением больших pose jumps

После GUI-разбора worst-pose trajectories была проверена гипотеза, что редкие
скачки между контактными равновесиями портят обучение всей shared cell. Ранее
такого clean ablation в репозитории не было. Rollout не запускался.

### 26.1 Фильтр без model-selection leakage

Переходы не выбирались по ошибке checkpoint. Использован только ground-truth
pose motion:

$$
d_{\Delta q,k}
=\sqrt{
\left(\frac{\|p_{k+1}-p_k\|}{L}\right)^2
+d_{SO(3)}(R_k,R_{k+1})^2
},
\qquad L=0.1114999652\ {\rm m}.
$$

Переход помечался jump при $d_{\Delta q,k}>0.05$. Граница соответствует
примерно 5.6 mm при чистой translation или 2.86 degrees при чистой rotation и
проходит около медианы ground-truth motion верхних 5% absolute pose-error
переходов. Threshold был зафиксирован до обучения и одинаков для всех splits.

| Split | Full active | Retained | Removed | Removed samples | Removed squared pose-motion energy |
|---|---:|---:|---:|---:|---:|
| train | 62 231 | 59 593 | 2 638 | 4.24% | 91.47% |
| val | 9 168 | 9 033 | 135 | 1.47% | 69.28% |
| test | 9 474 | 9 257 | 217 | 2.29% | 82.63% |

Это подчёркивает heavy-tail nature данных: несколько процентов физически
валидных branch transitions определяют большую часть squared pose target
energy.

Первый exploratory index с ошибочно использованным SDF scale 0.02 m вместо
gripper length был обнаружен до обучения и не использовался. Финальный
contract использует manifest `length_scale_m`. Отфильтрованное представление
сохранено как отдельный sibling dataset
[`data/simulator-r-v1-smooth`](../data/simulator-r-v1-smooth/README.md) со
своими `manifest.json`, `active-index.npz` и
[`filter-contract.json`](../data/simulator-r-v1-smooth/filter-contract.json).
Неизменные HDF5 shards и gripper asset переиспользуются через относительные
symlink, поэтому физический корпус не дублируется.

### 26.2 Controlled local training

Один seed-0 L1 checkpoint обучался с теми же model, squared loss, optimizer,
object-balanced sampler и early stopping, что clean full-data L1. Отличались
только active index train/val. Config теперь указывает на отдельный
`data/simulator-r-v1-smooth/` dataset; архитектурная секция совпадает с
основным committed baseline и содержит только `hidden_dim = 64`. Baseline
завершил 26 epochs; smooth-trained run
завершил 71 epoch, лучший checkpoint — epoch 60. Более длинная оптимизация
является частью наблюдаемого эффекта фильтра: tail transitions больше не
делают validation criterion шумным.

22 августа новый production index
`active-index-train-val-no-pose-jumps.npz` был попарно сверен с использованным
здесь historical smooth index. Для train совпали все 59,593, для validation —
все 9,033 пар `(object, trajectory, step)`. Различается только test policy:
новый index сохраняет полный test из 9,474 переходов, а приведённая ниже
основная таблица и так вычислялась на этом полном test. Повторная one-step
оценка обоих checkpoint текущим source в bfloat16 воспроизвела сохранённые
метрики с расхождением менее (10^{-6}). Таким образом, этот run является
точным before/after экспериментом для принятого train/val-фильтра; rollout в
нём не выполнялся.

Equal-object metrics на полном наборе, включая ранее удалённые transitions:

| Split | Model | $d_X$ | pose | translation, mm | rotation, rad | joints |
|---|---|---:|---:|---:|---:|---:|
| train | full-trained L1 | 0.036210 | 0.017976 | 1.3930 | 0.010591 | 0.027585 |
| train | smooth-trained L1 | **0.034228** | 0.017989 | 1.3955 | **0.010481** | **0.024720** |
| val | full-trained L1 | 0.031197 | 0.012856 | 1.1871 | 0.006140 | 0.026482 |
| val | smooth-trained L1 | **0.029435** | **0.012766** | **1.1762** | **0.005960** | **0.024304** |
| test | full-trained L1 | 0.037451 | 0.014687 | 1.4033 | 0.006495 | 0.032299 |
| test | smooth-trained L1 | **0.034252** | **0.014064** | **1.3280** | **0.006289** | **0.028860** |

На полном test это $-8.54\%$ по $d_X$, но decomposition важнее:
pose $-4.24\%$, translation $-5.37\%$, rotation $-3.17\%$, joints
$-10.65\%$. На train pose практически не изменился (+0.07%), тогда как
joints улучшились на 10.39%. Значит основной выигрыш не состоит в том, что
stateless operator внезапно выучил $q$: удаление больших pose gradients
освободило capacity/optimization прежде всего для $r$.

Test pose gain также не одинаков по unseen objects: −0.99% для Hercules oats,
−0.31% для orange/raisin cookies и −11.52% для Baikal water. Поэтому single-seed
aggregate −4.24% нельзя считать универсальным pose improvement.

### 26.3 Matched smooth и removed-jump evaluation

| Test subset | Model | $d_X$ | pose | translation, mm | rotation, rad | joints |
|---|---|---:|---:|---:|---:|---:|
| retained smooth, 9 257 | full-trained | 0.034743 | 0.012131 | 1.1662 | 0.005350 | 0.031293 |
| retained smooth, 9 257 | smooth-trained | **0.031484** | **0.011407** | **1.0781** | **0.005144** | **0.027901** |
| removed jumps, 217 | full-trained | 0.155713 | **0.126029** | **11.7903** | 0.055463 | 0.077088 |
| removed jumps, 217 | smooth-trained | **0.155611** | 0.130539 | 12.3534 | **0.055267** | **0.071474** |

На smooth subset improvement ожидаемо больше: $d_X$ −9.38%, pose −5.97%,
joints −10.84%. На удалённых transitions aggregate практически одинаков
(−0.065%), однако pose становится хуже на 3.58%, translation — на 4.78%; падение
joints на 7.28% случайно компенсирует pose degradation в aggregate.

Следовательно, фильтр специализирует shared cell на smooth regime и улучшает
общий score из-за малой доли jumps, но не обучает и не устраняет сам
discontinuous contact map.

### 26.4 Контроль с экспериментальным history candidate

Ранее найденный `history + trial/current gap + decoupled pose + balanced local
losses` checkpoint, обученный на всех transitions, был оценён на тех же masks:

| Test subset | $d_X$ | pose | translation, mm | rotation, rad | joints |
|---|---:|---:|---:|---:|---:|
| full | **0.029247** | **0.010244** | **0.8740** | **0.005730** | **0.025245** |
| retained smooth | **0.026630** | **0.007676** | **0.6330** | **0.004620** | **0.024349** |
| removed jumps | **0.144041** | **0.123199** | **11.5431** | **0.053421** | **0.063797** |

Он лучше smooth-trained L1 и на smooth, и на removed jumps. Поэтому jumps не
являются принципиально невыучиваемыми corrupted labels. Это полезный
диагностический контроль в пользу mode-aware state/contact representation, но
не решение о смене production-архитектуры.

### 26.5 Сопоставление с research

[Allen et al.](https://proceedings.mlr.press/v205/allen23a.html) показывают,
что general-purpose graph simulators способны учить rigid-contact
discontinuities при подходящей архитектуре и parameterization. Поэтому
удаление валидных jump transitions меняет целевую динамику, а не исправляет
неизбежную label corruption.

[ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html) кодирует
signed distance/contact Jacobians и complementarity/max-dissipation structure;
[Jiang et al.](https://proceedings.mlr.press/v168/jiang22a.html) явно моделируют
переход static/dynamic friction classifier; [Bianchini et al.](https://proceedings.mlr.press/v229/bianchini23a.html)
инферируют ненаблюдаемые contact forces. Все три направления поддерживают
mode-aware/structured treatment вместо полного удаления switching events.

Robust-regression теория, например
[D'Orsi et al.](https://proceedings.mlr.press/v139/d-orsi21a.html), обосновывает
Huber-like treatment при corrupted/heavy-tailed noise. Здесь GUI replay и
settling diagnostics показывают, что jumps — настоящие физические решения,
поэтому применять к ним hard outlier semantics некорректно. Robust loss или
curriculum `smooth first, jumps later` остаются разумными optimization tools,
но final model должна видеть обе ветви.

### 26.6 Итог

Удаление $d_{\Delta q}>0.05$ улучшает full-test L1 aggregate на 8.54%, в
основном через joints, и немного улучшает smooth pose. Но оно не исправляет
исходную проблему $q$, ухудшает pose на самих jumps и уступает
экспериментальному history candidate на обоих regimes. Практический вывод:
такой index полезен как diagnostic или первая фаза curriculum, но не как
окончательная замена полного dataset.

**Позднейшее изменение production data contract.** После simulator-variation
audit было принято явное практическое решение всё же использовать этот фильтр
для основной local train/validation supervision, сохранив полный test split.
Это не меняет научный вывод ablation: jumps остаются физически валидными и
нужны для честной оценки, но больше не входят в оптимизацию и model selection
основного local checkpoint. Актуальный index и точные counts описаны в
разделе 12.3.

Воспроизводимость:
[`simulator-r-v1-smooth`](../data/simulator-r-v1-smooth/README.md),
[`prepare_pose_jump_filter_ablation.py`](../scripts/prepare_pose_jump_filter_ablation.py),
[`evaluate_pose_jump_filter_ablation.py`](../scripts/evaluate_pose_jump_filter_ablation.py),
[`ablation-local-no-pose-jumps.toml`](../configs/ablation-local-no-pose-jumps.toml),
[`best-local.pt`](../runs/ablation-local-no-pose-jumps/best-local.pt),
[`results.json`](../runs/ablation-local-no-pose-jumps/results.json).

---

## 27. Conditional observability material-v2

После history и jump-filter ablation была напрямую проверена более сильная
гипотеза: существуют ли в текущем dataset одинаковые или почти одинаковые
наблюдаемые состояния с разными successors, и объясняют ли различие history
или сохранённые contact diagnostics. Это read-only non-parametric experiment;
модель не обучалась и production architecture не менялась.

### 27.1 Протокол

Для каждого из 80 873 active transitions ближайший сосед искался только среди
других trajectories того же объекта и того же command step. Поэтому object SDF
и $\bar a_{k+1}$ в каждой группе строго одинаковы. Сравнивались:

1. фактический вход production operator: trial-gap vector и scalar current
   aperture;
2. trial gap плюс current-gap vector;
3. полный записанный snapshot $(q_k,r_k)$;
4. reranking ближайших snapshots по предыдущему pose/full-state increment;
5. reranking по реально сохранённым incoming diagnostics: `contact_count`,
   approximate maximum applied actuator effort, residual linear/angular
   velocity и `settling_substeps`;
6. target-side oracle с одинаковым outgoing `contact_count`.

Чтобы history не выигрывала простым выбором далёкого state, основной контроль
разрешал выбирать только среди 20 candidates с snapshot distance не более
1.25 от расстояния до исходного nearest neighbour. Unrestricted reranking
сохранён только как верхняя диагностическая оценка.

Target divergence измерялась между one-step increments двух transitions:

$$
d_{\Delta q}
=\sqrt{\left\|
(\Delta p_i-\Delta p_j)/L
\right\|^2
+d_{SO(3)}(\Delta R_i,\Delta R_j)^2}.
$$

### 27.2 Точных неоднозначных повторов не найдено

Machine-near snapshot определён как translation difference не более
0.1 micrometre, rotation не более $10^{-6}$ rad и normalized joint RMS не
более $10^{-6}$. В train найдено 14 направленных samples, то есть семь
симметричных пар. Их successor pose divergence:

- mean $1.49\cdot10^{-5}$;
- maximum $1.04\cdot10^{-4}$;
- joint divergence строго 0.

В val/test таких повторов нет. Machine-near дубликатов production operator
input и `operator + current gap` также нет. Следовательно, dataset не содержит
прямого примера $x_i\simeq x_j$, $u_i=u_j$, но
$F(x_i,u_i)\ne F(x_j,u_j)$. Физическая многозначность или скрытая solver
memory этим тестом **не подтверждена**.

Это не является доказательством Markov sufficiency: val/test просто слишком
разрежены. Даже нижний 1% test nearest-snapshot distances в среднем разделён
4.61 mm translation, 0.0735 rad (4.21 degrees) rotation и 0.0171 normalized
joint RMS. Такие состояния нельзя считать локальными репликами.

### 27.3 Что действительно добавляет history

Ниже equal-object 1-NN target pose divergence на 9 197 test transitions,
имеющих предыдущий шаг. Это не error обученной модели, а мера локального
target spread выбранного representation.

| Representation / predictor | Test pose divergence |
|---|---:|
| identity increment $\Delta q=0$ | 0.012042 |
| previous-increment persistence | **0.009516** |
| nearest full snapshot $(q,r)$ | 0.015513 |
| nearest production operator input | 0.015168 |
| nearest `operator + current gap` | 0.015011 |
| constrained previous-pose rerank | 0.014462 |
| constrained previous-full-state rerank | **0.014184** |
| constrained incoming-diagnostics rerank | 0.014951 |
| outgoing-contact-count oracle | 0.015069 |

Constrained full-history reranking относительно nearest snapshot устойчиво
уменьшает pose spread:

| Split | Change | object-bootstrap 95% CI | mean selected snapshot distance |
|---|---:|---:|---:|
| train | -8.73% | [-10.88%, -6.75%] | 0.429 -> 0.450 |
| val | -7.71% | [-10.60%, -3.87%] | 0.492 -> 0.513 |
| test | -8.57% | [-9.67%, -7.56%] | 0.425 -> 0.447 |

Без distance constraint apparent gain достигает 19--25%, но mean selected
snapshot distance возрастает примерно с 0.43--0.49 до 0.66--0.83. Поэтому эту
большую цифру нельзя использовать как доказательство missing state. Надёжный
вывод уже: предыдущий increment содержит дополнительный predictive signal при
почти фиксированной близости snapshot, но он не определяет discontinuous
branch полностью.

Incoming diagnostics дают меньший constrained gain: 4.46%, 3.78% и 3.63% на
train/val/test. Scalar incoming contact count даёт лишь 1.4--2.3%, а даже
некаузальный outgoing-contact-count oracle — 2.9--3.9%. В ближайшем 1% test
snapshot pairs outgoing count и так одинаков в 97.9% случаев. Значит простой
счётчик контактов не является отсутствующей mode variable.

### 27.4 History не разрешает jumps

Разложение test по исходному порогу $d_{\Delta q}>0.05$:

| Method | Smooth pose spread, 8 980 | Jump pose spread, 217 |
|---|---:|---:|
| nearest snapshot | 0.012677 | 0.135990 |
| constrained full history | **0.011361** | 0.135479 |
| previous-increment predictor | **0.006847** | **0.123714** |

Constrained history улучшает smooth regime на 10.4%, но jump regime только на
0.38%. Поэтому one-step gain history architecture в основном является
continuation bias на гладких segments, а не распознаванием switching event.

Это подтверждает отдельный 1-NN jump classifier. На test jump prevalence
2.4%. Для previous-increment label balanced accuracy равна 0.579, recall
17.5%; для full snapshot 0.533/8.8%, `operator + current gap` 0.532/8.3%,
unrestricted full-history neighbour 0.540/9.2%. Даже outgoing-contact-count
oracle остаётся на 0.533/8.8%. Ни один сохранённый coarse signal не определяет
момент скачка.

### 27.5 Исправленный причинный вывод

Тест отвергает прежнюю слишком сильную формулировку «главная причина —
доказанная немарковость $(q,r,u)$». Текущие данные поддерживают более узкую
картину:

1. **Machine-near повторы детерминированы.** Прямого evidence физически разных
   successors из одинакового recorded state нет.
2. **Dataset локально разрежен.** На unseen objects нет близких повторов,
   позволяющих отделить hidden state от чувствительной, но однозначной
   discontinuity.
3. **History — полезный smooth-direction prior.** Его constrained effect
   реален, но почти исчезает на jumps.
4. **Главный неразрешённый остаток — switching event.** Ни current gap, ни
   scalar contact count, ни предыдущий increment надёжно не предсказывают его.
5. **Нужные физические labels отсутствуют.** Dataset не сохраняет contact
   identities, normals, tangential stick/slip state, impulses или reaction
   forces. `actuator_effort` является approximate applied drive torque, а не
   contact reaction.

Поэтому наиболее подтверждённое объяснение слабого $q$ сейчас — сочетание
редкого покрытия discontinuity boundary и representation, не содержащего
явного contact-mode/event signal. Скрытая PhysX memory остаётся возможной, но
не является подтверждённым главным bottleneck. Для $r$ geometry-based
neighbours значительно лучше full-snapshot neighbours, что согласуется с
предыдущим выигрышем actual current gap.

Следующий решающий data experiment должен быть не полным `srno sim collect`, а
малой targeted коллекцией около 10--20 известных jump-boundary states: точные
повторы, малые perturbations $(q,r)$, preserve/reset histories и per-link
contact identities/forces до и после increment. Без таких labels очередной
history feature или hard filtering не сможет различить deterministic mode
boundary и действительно hidden contact state.

Артефакты:
[`analyze_conditional_observability.py`](../scripts/analyze_conditional_observability.py),
[`results.json`](../runs/conditional-observability-material-v2/results.json),
[`samples.npz`](../runs/conditional-observability-material-v2/samples.npz).

![Conditional observability](../runs/conditional-observability-material-v2/conditional_observability.png)

---

## 28. От direct GNO к contact-cone resolvent: теория, реализация и local ablation

После анализа conditional observability была проверена более фундаментальная
гипотеза: проблема не только в недостающем history, а в том, что прежняя shared
cell вообще не обязана представлять контактную механику. Она получает sampled
gap field, выполняет один или несколько learned integral layers и свободной
12-мерной MLP-head непосредственно выдаёт twist и joint residual. Это
универсальный аппроксиматор one-step map, но не contact resolvent в
математическом смысле: в нём нет contact multipliers, normal cone,
complementarity, force balance или implicit solve.

Основной committed config и его direct head не изменены. Новая архитектура
добавлена как opt-in `contact_head`; все перечисленные ниже эксперименты имеют
отдельные configs и checkpoints.

### 28.1 Что из resolvent-гипотезы обосновано, а что нет

[Pesquet et al.](https://epubs.siam.org/doi/10.1137/20M1387961) дают строгую
связь resolvent максимального монотонного оператора с firmly nonexpansive map,
а [Schwab and Stein](https://www.research-collection.ethz.ch/handle/20.500.11850/552121)
строят proximal networks с такими ограничениями. Это полезный ориентир, но не
готовая теорема для SRNO: нет оснований считать unilateral contact с неconvex
geometry, Coulomb friction и переключением active set одним глобальным гладким
максимально-монотонным оператором в используемых координатах. Аналогично, gradient
ICNN был бы cyclically monotone, то есть сильнее необходимого, и для всего
frictional map это ограничение не доказано.

Ближе к текущей задаче находится
[ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html): learned
contact geometry/Jacobians соединяются с implicit contact mechanics и
violation losses. Связанные работы используют differentiable contact
formulations ([Zhong et al.](https://papers.nips.cc/paper/2021/hash/b7a8486459730bea9569414ef76cf03f-Abstract.html)),
implicit violation loss ([Bianchini et al.](https://proceedings.mlr.press/v168/bianchini22a.html))
и SDF contact functionals ([Driess et al.](https://arxiv.org/abs/2110.00792)).
Практический вывод для SRNO: сначала нужно ограничить learned output
физически допустимым contact cone, но не утверждать, что этим уже доказана
глобальная монотонность или решена complementarity problem.

### 28.2 Три численных теста до обучения архитектуры

Проверки проводились на коротких подвыборках, чтобы не запускать очередное
длинное обучение без evidence.

**Жёсткая линейная проекция не является подходящим solver.** На 300 test
transitions QP минимальной коррекции при linearized nonpenetration решил только
205 случаев; 95 были infeasible или не сошлись. На решённых случаях свободный
trial имел mean minimum gap -7.183 mm, тогда как target имел +1.207 mm.
Одношаговая проекция перебрасывала состояние до +3.101 mm и ухудшала $d_X$
с 0.04810 до 0.39874 (+729%). Translation/rotation/joints ухудшились на
4266%/2323%/235%. Причина — глубокая trial penetration и локальная
линеаризация SDF, а не отсутствие QP solver.

**Повторная nonlinear soft projection исправляет в основном joints.**
Validation выбрал без test leakage `penalty=10, pose_weight=100`. На независимой
300-sample test подвыборке $d_X$ уменьшился 0.07727 -> 0.05337 (-30.93%),
joint error — на 34.40%, но translation и rotation стали хуже на 7.03% и
2.22%. Следовательно, одна геометрическая feasibility хорошо задаёт закрытие
пальцев, но почти не задаёт ветвь motion объекта.

**Contact cone достаточно выразителен, если известны multipliers.** Oracle
NNLS искал неотрицательные $\lambda_i$ для тех же 300 test transitions, не
являясь predictive model. При `pose_weight=10` проекция target correction на
положительный cone trial contact Jacobians дала $d_X=0.01072$, то есть
-86.13% относительно free trial; translation/rotation/joints улучшились на
49.49%/35.36%/89.74%. Значит основной architecture bottleneck — не отсутствие
нужного направления в $J^T\lambda$, а восстановление правильных contact
pressures и выбор pose branch.

Эти три результата совместно отвергают две крайности: `hard projection` не
работает, но и полностью свободная 12D head не нужна. Нужен learned solver в
структурированном cone.

### 28.3 Реализованная contact-cone cell

Введены dimensionless product coordinates

$$
z=\left[\Delta p/L,\;\omega,\;(r-r_{\rm free})/s_r\right]\in\mathbb R^{12},
\qquad
J_i=\frac{\partial(h_i/s_{\rm sdf})}{\partial z}.
$$

Полный $J_i$, включая translation, rotation moment arm и производные всех
шести joints через analytic FK, вычисляется в `float32`. Его finite-difference
test проверяет все 12 columns. Для каждого из 256 canonical gripper points
двухпроходный permutation-equivariant DeepSets encoder получает gap,
координату point и $J_i$, затем предсказывает

$$
\lambda_i={\rm softplus}(f_\theta(e_i,e_{\rm global},u))\ge 0,
\qquad
g=\frac1M\sum_i J_i^T\lambda_i.
$$

Conditioned mobility задаётся как $B_\theta=C_\theta C_\theta^T\succ0$, а
correction — $z=B_\theta g$. Поэтому сеть больше не может непосредственно
выдать произвольный 12-vector: correction лежит в positive-metric image
обобщённых contact normals. При этом $f_\theta$ и $B_\theta$ зависят от
state, поэтому эта конструкция **сама по себе не гарантирует** глобальную
монотонность или firm non-expansiveness.

Friction arm дополнительно предсказывает две tangent components с

$$
\|\lambda_{t,i}\|_2\le\mu\lambda_{n,i},\qquad \mu=2.4,
$$

но это soft cone parameterization, а не max-dissipation/complementarity solve.
Сложность encoder уменьшена с $O(M^2)$ у pairwise integral layer до $O(M)$.
Inactive/free bypass остался точным.

### 28.4 Почему чистая cone cell всё ещё почти оставляет pose равной identity

Короткий unbalanced normal-cone run был вручную остановлен после 22 записанных
epochs, когда основной вывод уже стабилизировался. На полном equal-object test
он улучшил $d_X$ на 30.08% и joints на 36.61%, но translation только на
13.78%, а rotation ухудшил на 0.65%. Относительно identity pose его median
pose-error/motion остался 1.00095; лишь 49.82% transitions стали лучше identity,
а по rotation — 42.49%. При этом median joint-error/free-error равна 0.2339 и
96.17% transitions лучше free joint trial.

Это не конфликт gradients. На восьми validation batches median отношения
joint-gradient norm к pose-gradient norm равна 12.93, feasibility/pose — 18.24,
но median cosine pose/joint gradients положителен, +0.50. Значит pose signal
подавлен масштабом и редкостью, а не систематически направлен против joint
signal. Balanced weights несколько улучшают translation, но портят rotation;
current gap + drive residual и friction cone также не восстанавливают pose.
Simple learned history-conditioning ухудшает все aggregate metrics относительно
stateless cone. Это согласуется с разделом 27: coarse history помогает smooth
continuation, но не определяет contact switch.

### 28.5 Causal continuation и split-resolvent

Поскольку сохранённый previous increment является подтверждённым smooth branch
signal, для active-contact branch введён parameter-free predictor

$$
p_{\rm pred}=p_k+\alpha(p_k-p_{k-1}),\qquad
R_{\rm pred}=\exp\!\left(\alpha\log(R_kR_{k-1}^T)\right)R_k.
$$

Grid $\alpha\in\{0,.25,.5,.75,1,1.25\}$ выбирался **только по validation
mean $d_X$**; выбран $\alpha=0.75$. На test выбор не повторялся. Huber +
balanced joint loss позволил одной end-to-end cell улучшить pose, но ослабил
joints. Поэтому наиболее устойчивый вариант разделяет роли:

- $q_{k+1}$ берётся из causal continuation predictor;
- $r_{k+1}$ берётся из обученной stateless normal-cone cell;
- learned cone correction для $q$ в этом режиме отключена;
- inactive branch сохраняет $q_k$ и exact free-joint bypass.

Это `pose_corrector="predictor_only"`. `split_resolvent` не является новым
переобученным checkpoint: это parameter-compatible view того же
`normal_cone/best-local.pt` с validation-selected $\alpha=0.75$. После
обнаруженной при проверке ошибки `rollout()` теперь передаёт previous state и
для continuation predictor, даже если learned `history_conditioning="none"`.

### 28.6 Единая local evaluation

Все checkpoints повторно оценены одним CPU script на полном active val/test и
одинаково усреднены по объектам. Короткие architecture screens были остановлены
вручную, а не объявлены сошедшимися по early stopping; число записанных epochs:
normal cone 22, history 15, balanced 22, full-state balanced 19, friction 14,
continuation-Huber 15.

| Test arm | $d_X$ | translation, mm | rotation, rad | joints |
|---|---:|---:|---:|---:|
| direct L1 baseline | 0.037120 | 1.3151 | 0.006486 | 0.032368 |
| normal cone | 0.025955 | 1.1339 | 0.006528 | 0.020517 |
| normal cone + learned history | 0.033516 | 1.3443 | 0.006536 | 0.028116 |
| normal cone, balanced | 0.028829 | 1.0208 | 0.007064 | 0.023917 |
| normal cone, current gap + drive | 0.030161 | 1.0758 | 0.006878 | 0.025383 |
| friction cone, current gap + drive | 0.027743 | 1.0187 | 0.007053 | 0.022925 |
| continuation + Huber, end-to-end | 0.030380 | 0.8046 | 0.004888 | 0.026123 |
| **split-resolvent** | **0.025361** | **0.7933** | **0.004743** | **0.020517** |

Относительно direct L1 итоговый split-resolvent даёт на validation
$d_X/T/R/J$ изменения -37.15%/-51.28%/-30.58%/-39.22%, а на test
**-31.68%/-39.68%/-26.86%/-36.61%**. Таким образом, требование о существенно
большем чем 10--20% эффекте выполнено для каждого компонента, а не только для
aggregate. Он также лучше identity baseline на test translation примерно на
24.9% и rotation на 22.3%.

Есть важное ограничение интерпретации: $r$ здесь действительно выучен
contact-cone network, тогда как лучший $q$ даёт причинный continuation prior,
а не выученные multipliers. Кроме того, в ходе всей исследовательской серии
test неоднократно использовался диагностически для сравнения гипотез. Поэтому
эти test gains являются сильным exploratory evidence, но не финальной
независимой оценкой model selection; для неё нужен новый held-out object set.

### 28.7 Что теперь подтверждено и что остаётся нерешённым

1. Для joints исходная проблема была прежде всего архитектурной: projection на
   learned positive contact cone даёт быстрый и большой выигрыш; свободная 12D
   head не использовала эту структуру эффективно.
2. Для pose contact geometry содержит нужные directions — это показывает
   oracle NNLS, — но имеющиеся inputs не дают сети надёжно выбрать pressures и
   ветвь переключения. Простые balance, history, current gap и friction features
   этого не исправили.
3. Smooth pose предсказывается как path continuation. Нельзя называть это
   решённым learned contact resolvent: на редких jumps из разделов 25--27 prior
   остаётся принципиально слабым.
4. Следующий обоснованный шаг — малая targeted collection около 10--20 уже
   известных jump-boundary states с repeated perturbations и per-contact
   identities, normals, impulses/reaction forces и stick/slip labels. До таких
   данных включать learned cone pose correction в основной вариант оснований
   нет.
5. Rollout-ablation сознательно не выполнен: текущая задача была local-only, а
   CUDA runtime на workstation возвращает error 804 из-за несовместимости
   driver/runtime. CPU local evidence получен полностью; simulator collection
   не перезапускалась.

Воспроизводимость:
[`model.py`](../src/srno/model.py),
[`test_model.py`](../tests/test_model.py),
[`evaluate_contact_resolvent_ablation.py`](../scripts/evaluate_contact_resolvent_ablation.py),
[`results.json`](../runs/ablation-contact-resolvent-v1/results.json),
[`normal-cone config`](../configs/ablation-normal-cone-local.toml),
[`split-resolvent config`](../configs/ablation-normal-cone-split-resolvent-local.toml),
[`normal-cone checkpoint`](../runs/ablation-normal-cone-local/best-local.pt),
[`linearized projection`](linearized-contact-projection-test.json),
[`nonlinear prox test`](nonlinear-contact-prox-test.json),
[`contact-cone oracle`](contact-cone-representability-test.json),
[`gradient scales`](../runs/ablation-normal-cone-local/gradient-scales.json),
[`continuation screen`](../runs/ablation-normal-cone-local/continuation-hybrid.json).

После исправления continuation rollout весь test suite:
`92 passed, 4 CUDA-only skipped`.

---

## 29. Implicit-multiplier resolvent: audit записки, corrected VI и local result

Следующий этап начался с независимой проверки записки
`pasted-text.txt`, литературы и текущего кода. Главный тезис записки оказался
верным: `normal_cone` из раздела 28 параметризует только stationarity direction
$B_\theta J^T\lambda_\theta$, но не обеспечивает primal feasibility,
complementarity и совместное решение multipliers. Поэтому это не contact
resolvent в строгом смысле.

### 29.1 Что в записке полезно, а что было артефактом обобщения

Подтверждены следующие положения:

- NN следует учить constitutive part, например SPD resistance $Q_\theta$, а
  multipliers получать из KKT/VI solve;
- actuator должен входить как query/linear term convex program, а не только как
  дополнительный feature свободной output-head;
- один 12D solve должен совместно определять object pose и six joint states;
- solver depth имеет физический смысл итераций решения, в отличие от
  произвольной глубины generic MLP.

Три формулировки записки оказались слишком сильными или буквально неверными
для текущего pipeline:

1. **Нельзя линеаризовать QP в полностью закрытом free-trial.** Предыдущий
   диагностический QP стартовал из mean penetration (-7.183) mm, решил лишь
   205/300 случаев и ухудшил $d_X$ на 729%. Это не опровержение VI, а неверная
   точка линеаризации. И
   [CQDC](https://arxiv.org/html/2206.10787), и
   [ContactSDF](https://arxiv.org/html/2408.09612v2) вычисляют gap/Jacobians в
   текущем состоянии и помещают command в objective/query.
2. **Локальный convex resolvent не доказывает global maximal monotonicity.**
   При фиксированных $x_k,Q_\theta$ metric projection на convex tangent set
   firmly nonexpansive в соответствующей метрике. Но set, SDF linearization и
   learned metric меняются с state, geometry и active set; глобальное
   утверждение из этого не следует.
3. **Литература не обещает беспроблемный gradient через nonsmooth LCP.**
   [Learning Linear Complementarity Systems](https://arxiv.org/html/2112.13284)
   прямо связывает differentiability с strict complementarity и предлагает
   violation-based loss, чтобы не дифференцировать parameter-dependent active
   constraints. [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html)
   также использует latent impulses и violation loss, а не просто оставляет
   готовый black-box forward solver. ProxNet является теорией/эмулятором
   projected VI iterations, а не готовой frictional contact model.

### 29.2 Исправленная математическая постановка

В dimensionless product coordinates

$$
z=\left[\Delta p/L,\;\Delta\omega,\;\Delta r/s_r\right]\in\mathbb R^{12}
$$

введён actuator/history query

$$
u_k=\left[
\alpha\frac{p_k-p_{k-1}}L,\;
\alpha\operatorname{Log}(R_kR_{k-1}^{T}),\;
\frac{r_{\rm free}(a_{k+1})-r_k}{s_r}
\right].
$$

При $\alpha=0$ это stateless quasi-static arm. При $\alpha>0$ предыдущий
pose increment является затухающим unforced object query той же VI, а не
отдельным continuation output-head. Active branch решает

$$
\boxed{
z^*=\arg\min_z\frac12(z-u_k)^TQ_\theta(x_k,u^r_k,\phi)(z-u_k)
\quad\text{s.t.}\quad
h_{\rm geom}(x_k)+s_{\rm sdf}J(x_k)z\ge0
}
$$

вместе с upper/lower bounds всех joints. Здесь $Q_\theta=C_\theta C_\theta^T\succ0$, а network предсказывает только bounded Cholesky factor. KKT system

$$
Q_\theta(z-u_k)-J^T\lambda=0,\qquad
0\le\lambda\perp Jz-b\ge0
$$

решается batched dual FISTA. Используется 128 итераций; это численная
аппроксимация convex resolvent, а не заявление об exact finite-depth map.
На 300 validation samples её zero-initialized output дал $d_X=0.01793$
против (0.01765) у high-accuracy OSQP, расхождение 1.6% по aggregate.

Free branch по-прежнему является точным: free-trial используется только для
contact gate, а VI линеаризуется в $x_k$. Геометрическая constraint boundary
равна 0; PhysX `contactOffset` используется только в gate. Analytic full
Jacobian и state update используют одну product retraction. В ходе проверки
было найдено и исправлено отдельное расхождение старого diagnostic script:
он интегрировал product increment, но до исправления вычислял rotational
moment arm для left-SE(3) retraction.

### 29.3 Screens до обучения

На фиксированной 300-sample validation subset:

| Screen | Результат |
|---|---:|
| free object/joint query | (d_X=0.07670, d_J=0.07301) |
| exact normal QP, pose weight 100 | (d_X=0.01765, d_J=0.01016) |
| 128-step differentiable solver init | (d_X=0.01793, d_J=0.01046) |
| + history query, $\alpha=0.5$ | $d_X=0.01738, T=0.963$ mm, $R=0.005945$ |

Pose weight 100 выбран validation-only. History grid
$\alpha\in\{0,.5,.75,1\}$ также выбран только по validation; минимум
aggregate был при $\alpha=.5$. Test для выбора не использовался.

Polyhedral Anitescu friction screen не подтвердил friction как текущий
bottleneck. Лучший normal-only arm имел $d_X=0.01765$, лучший frictional arm
(0.01734): около 2% дополнительного выигрыша, rotation практически не
изменилась. Поэтому friction не добавлялась в trainable MVP.

### 29.4 Короткое обучение и полный local replay

Оба CPU run вручную остановлены после 16 записанных epochs: это architecture
screens, а не заявления о полном scheduler convergence. Тем не менее
оптимизация была устойчивой. У inertial arm validation $d_X$ уменьшился
$0.013947\to0.012092$, а loss $6.391\cdot10^{-4}\to5.276\cdot10^{-4}$.
Главный выигрыш появляется уже в physical initialization, затем learned metric
даёт дополнительное последовательное снижение.

Full equal-object local evaluation:

| Arm | split | $d_X$ | $T$, mm | $R$, rad | $d_J$ |
|---|---|---:|---:|---:|---:|
| direct L1 | val | 0.030928 | 1.1249 | 0.006103 | 0.026529 |
| implicit resolvent, stateless | val | 0.013436 | 0.7690 | 0.005541 | 0.008061 |
| implicit resolvent + history query | val | **0.012092** | **0.6100** | **0.004573** | **0.007780** |
| direct L1 | test | 0.037120 | 1.3151 | 0.006486 | 0.032368 |
| implicit resolvent, stateless | test | 0.015507 | 0.9739 | 0.005849 | 0.009034 |
| implicit resolvent + history query | test | **0.014456** | **0.8379** | **0.005052** | **0.008702** |

Изменения final arm относительно direct L1 на test:

$$
\boxed{
d_X:-61.06\%,\quad
T:-36.28\%,\quad
R:-22.10\%,\quad
d_J:-73.11\%.
}
$$

Это первый единый mechanical solve в проекте, который даёт существенно больше
30% по aggregate и одновременно улучшает все компоненты. Stateless arm сам по
себе даёт (-58.22\%) по $d_X$ и (-72.09\%) по joints; history-query
добавляет к нему на test (-13.96\%) translation и (-13.62\%) rotation.

Relative-error replay показывает, что результат не является только средним
эффектом: среди test transitions с ненулевым pose displacement 80.74% лучше
identity, median pose-error/true-motion (=0.542). Для joints 98.65% лучше
free predictor, median error/contact-residual (=0.122).

### 29.5 Что осталось нерешённым: nonlocal switching jumps

Для $m_q>0.05$ test median pose ratio остаётся 0.953, а для верхнего 1%
displacements — 0.978. Причина теперь локализована:

- 99.08% jump targets сами удовлетворяют simulator-consistent admissible gap;
- только 25.81% jump targets удовлетворяют zero-gap tangent constraints,
  вычисленные в $x_k$;
- для верхнего 1% displacement tangent-feasible только 7.37%; median maximum
  linearization error равна 8.875 mm против 0.0238 mm на smooth subset.

Следовательно, target jump обычно лежит на другой нелокальной equilibrium
branch. Его нельзя получить обучением $Q_\theta$ внутри одного tangent
resolvent. Проверка 2--4 sequential relinearizations также не помогла:
validation $d_X$ изменился $0.017083\to0.017141$, jump $d_X$ остался
0.12852. Pose weight 1 дал лишь около 6% выигрыша на jumps ценой примерно 48%
ухудшения общего $d_X$.

Таким образом, подтверждённая граница вывода следующая:

1. Для smooth local contact evolution и joints найден работающий единый
   SDF-conditioned resolvent; прежний architecture bottleneck устранён с
   крупным, воспроизводимым эффектом.
2. Rare jumps не являются ошибкой optimizer, недостатком solver iterations или
   простой linearization refinement. Это mode-selection/data-observability
   problem.
3. Следующий научно оправданный шаг для jumps — ранее описанная малая targeted
   collection с repeated perturbations и per-contact identity/normal,
   impulse/reaction-force и stick/slip labels. Без event/force labels свободный
   learned pose force или mixture head будет лишь снова fitting'овать редкий
   discontinuous target как black box.
4. Rollout/H32 ещё не проверен: CUDA runtime по-прежнему недоступен (error 804),
   а CPU rollout retrain был бы длинным и не нужен для local architecture
   discrimination.

Реализация и артефакты:
[`model.py`](../src/srno/model.py),
[`implicit config`](../configs/ablation-implicit-resolvent-local.toml),
[`history-query config`](../configs/ablation-inertial-implicit-resolvent-local.toml),
[`consolidated results`](../runs/ablation-implicit-resolvent-v1/results.json),
[`final checkpoint`](../runs/ablation-inertial-implicit-resolvent-local/best-local.pt),
[`exact current-QP screen`](current-state-resolvent-val-gap0-decoupled.json),
[`friction screen`](current-state-frictional-resolvent-val.json),
[`history-query screen`](current-state-inertial-resolvent-val.json),
[`target compatibility`](resolvent-target-compatibility-test.json),
[`sequential solve`](sequential-current-resolvent-val.json),
[`relative errors`](../runs/ablation-inertial-implicit-resolvent-local/relative-error/local_pose_relative_error.json).

После изменений полный test suite: **95 passed, 4 CUDA-only skipped**.

---

## 30. Повторный audit resolvent learning: граница новизны и bivariational formulation

Этот раздел исправляет направление research после замечания, что local contact
QP из раздела 29 уже близок к ContactSDF. Contact-ML здесь использован только
для определения границы новизны. Основная аргументация и новые hypotheses
основаны на общей теории максимальных монотонных операторов, вариационной
механике rate-independent systems, неассоциативных constitutive laws,
structure-preserving ML и implicit optimization layers.

### 30.1 Что из раздела 29 полезно, но не является центральной новизной

Сопоставление с первичными источниками даёт следующую границу:

| Элемент | Ближайшее известное пересечение | Статус для SRNO |
|---|---|---|
| Current-state contact QP, command в linear query, projection в feasible velocity set | [ContactSDF](https://arxiv.org/html/2408.09612v2), equations 1 и 11--13 | сильный engineering baseline, но не scientific novelty |
| Глубина как число projected VI iterations | [ProxNet](https://link.springer.com/article/10.1007/s40687-022-00327-1) | общий известный принцип solver emulation |
| NN как firmly nonexpansive resolvent maximal monotone operator | [Pesquet et al.](https://arxiv.org/abs/2012.13247) | корректная operator theory, но уже существующий ML-класс |
| Convex potential, inference через внутреннюю оптимизацию | [ICNN](https://proceedings.mlr.press/v70/amos17b.html), [monDEQ](https://papers.nips.cc/paper_files/paper/2020/hash/798d1c2813cbdf8bcdb388db0e32d496-Abstract.html) | полезные building blocks, сами по себе не novelty |

Следовательно, крупный результат раздела 29 остаётся валидным численно:
current-state constrained solve исправляет большую часть joint error и smooth
pose error. Неверным было бы только объявить сам local QP новым научным
объектом.

В записке `pasted-text.txt` были также отделены четыре правдоподобно звучащих,
но слишком сильных тезиса:

1. Из локальной firm nonexpansiveness при фиксированной геометрии не следует
   global maximal monotonicity state-dependent contact map.
2. Polyhedral friction constraints в convex QP не тождественны полной
   неассоциативной Coulomb law.
3. `solver depth = network depth` не является самостоятельной новизной после
   ProxNet.
4. Хорошая endpoint regression не идентифицирует ни dissipation potential, ни
   contact law. Даже полный constitutive graph вообще не задаёт bipotential
   единственным образом; это явно отмечено в
   [Buliga, de Saxcé, Vallée](https://arxiv.org/pdf/math/0608424).

### 30.2 Какая механическая задача действительно записана в dataset

Dataset содержит не один малый velocity step, а последовательность практически
settled equilibria после удержания каждого нового actuator command. Поэтому
естественный объект — не только projection свободной скорости, а incremental
equilibrium evolution при медленном loading.

Обозначим

$$
y=(q,r)\in SE(3)\times\mathbb R^6,
\qquad
\mathcal C_\phi=\{y:h_\phi(y)\geq0\},
$$

и известную actuator energy

$$
\mathcal E_{\rm act}(r;\bar r)
=\frac12(r-\bar r)^T K(r-\bar r).
$$

Для ассоциативной rate-independent системы естественна inclusion

$$
0\in\partial\Psi_0(\dot y)
+D_y\mathcal E_{\rm act}(y;\bar r(t))
+N_{\mathcal C_\phi}(y).
$$

Однако nonconvex feasible geometry создаёт скачки между локальными
equilibrium branches. Теория
[Mielke--Rossi--Savaré](https://arxiv.org/abs/0910.3360) показывает, что такие
jumps описываются не произвольным выбором следующего global minimum, а
vanishing-viscosity limit

$$
0\in\partial\Psi_0(\dot y)+\varepsilon V(y)\dot y
+D_y\mathcal E_t(y)+N_{\mathcal C_\phi}(y),
\qquad \varepsilon\downarrow0,
$$

с локальной stability и energy--dissipation balance. В jump point физическая
траектория разворачивается по внутреннему arclength/pseudo-time, хотя внешний
load практически не меняется. Это точно соответствует наблюдаемому failure
mode: target допустим, но лежит на другой нелокальной equilibrium branch и не
принадлежит tangent cone в начальном state.

Обычный scalar dissipation potential всё же недостаточен. Для относительной
contact velocity $v=(v_n,v_t)$ и reaction
$\lambda=(\lambda_n,\lambda_t)$ полный Coulomb graph **не является даже
монотонным**, не только cyclically monotone. Это доказано и явно сформулировано
в [теории Coulomb bipotential](https://arxiv.org/pdf/0802.1140). Его корректное
представление имеет вид

$$
b_\mu(v,\lambda)
=\mu\lambda_n\|v_t\|
+\chi_{K_\mu}(\lambda)
+\chi_{K_0^*}(v),
$$

$$
b_\mu(v,\lambda)\geq\langle v,\lambda\rangle,
\qquad
b_\mu(v,\lambda)=\langle v,\lambda\rangle
\iff (v,\lambda)\ \text{удовлетворяет Coulomb law}.
$$

Функция $b_\mu$ convex отдельно по $v$ и $\lambda$, но не обязана быть
jointly convex или separable. Именно это снимает ошибочное требование, что
frictional response должен быть gradient одного convex potential.

### 30.3 Новые diagnostics до изменения architecture

На детерминированной выборке 300 test transitions проверены необходимые
условия incremental equilibrium:

| Quantity | Result |
|---|---:|
| mean actuator spring energy до шага | 0.503705 J |
| mean spring energy после settling | 0.450684 J |
| mean released energy | 0.053022 J |
| transitions с nonnegative release | 98.67% |
| correlation release с pose motion | 0.603 |
| correlation release с joint motion | 0.463 |
| mean release на jump transitions | 0.177070 J |
| mean release на smooth transitions | 0.049185 J |

Таким образом, known actuator energy действительно даёт loading/energy
signal, а jumps в среднем сопровождаются существенно большим release.

Target-state static balance проверялся после исключения неизвестных contact
force magnitudes с помощью NNLS. При friction pyramid с $\mu=2.4$ и
2 mm contact support 77.0% targets имеют relative equilibrium residual ниже
0.1; normal-only вариант — только 9.67%. При широком 8 mm support доля для
friction достигает 98%, но это уже только compatibility upper bound, а не
точное восстановление PhysX manifold. Вывод узкий: frictional force balance
поддерживается данными и существенно лучше normal-only balance; exact forces
из 256 SDF samples не идентифицированы.

Отдельный target-error oracle между direct, split-continuation и implicit-QP
candidates дал validation $d_X=0.011086$ против (0.012092) лучшего
одиночного arm, то есть лишь -8.32%. Поэтому простой mixture/energy selector
между уже имеющимися predictors был отвергнут до обучения: у него недостаточно
representational headroom.

Артефакты:
[`incremental mechanics test`](incremental-mechanics-test.json),
[`variational candidate oracle`](variational-candidate-oracle-val.json).

### 30.4 Две общие operator hypotheses и их фальсификация

Были реализованы два contact-independent по происхождению класса. Оба получают
current SDF gaps, полный analytic $J_q,J_r$, current aperture и optional
previous pose increment. Command не смешивается с geometry context, а входит
как generalized query.

**Hypothesis A: gradient convex dual potential.** Для фиксированного context

$$
T_\theta(f;c)=\nabla_f\Psi_\theta^*(f;c),
$$

где diagonal quadratic part и normalized softplus ridges параметризованы так,
что

$$
0\preceq \nabla T_\theta\prec I,
\qquad T_\theta(0;c)=0.
$$

Следовательно, $T_\theta$ firmly nonexpansive и является точной resolvent
некоторого maximal monotone operator при фиксированном context. DCT basis
смешивает pose и joint coordinates уже при initialization; coordinate-axis
basis предварительно оказался trapped у identity pose.

**Hypothesis B: non-cyclic monotone resolvent.** Чтобы снять symmetry gradient
map, context предсказывает

$$
A_\theta(c)=S_\theta(c)+W_\theta(c),
\quad S_\theta=L_\theta L_\theta^T\succeq0,
\quad W_\theta^T=-W_\theta,
$$

а layer возвращает

$$
T_\theta(f;c)=(I+A_\theta(c))^{-1}f.
$$

Это exact resolvent maximal monotone linear operator, но уже не gradient
scalar potential. Такая symmetric/skew decomposition также встречается в
общих structure-preserving моделях non-equilibrium dynamics, например
[GFINNs](https://doi.org/10.1098/rsta.2021.0207); её использование здесь было
diagnostic hypothesis, не заявлением новизны.

Полный equal-object replay:

| Arm | split | $d_X$ | $T$, mm | $R$, rad | $d_J$ |
|---|---|---:|---:|---:|---:|
| direct L1 | val | 0.030928 | 1.1249 | 0.006103 | 0.026529 |
| contact QP + history query | val | **0.012092** | **0.6100** | **0.004573** | **0.007780** |
| convex dual, zero pose query | val | 0.015626 | 0.8632 | 0.005775 | 0.010302 |
| non-cyclic monotone, zero pose query | val | 0.027159 | 1.2701 | 0.008386 | 0.021487 |
| convex dual + history query | val | 0.015829 | 0.8149 | 0.005092 | 0.011211 |
| direct L1 | test | 0.037120 | 1.3151 | 0.006486 | 0.032368 |
| contact QP + history query | test | **0.014456** | **0.8379** | **0.005052** | **0.008702** |
| convex dual, zero pose query | test | 0.017628 | 1.0479 | 0.006132 | 0.011423 |
| non-cyclic monotone, zero pose query | test | 0.029322 | 1.4561 | 0.008941 | 0.022676 |
| convex dual + history query | test | 0.017805 | 0.9864 | 0.005483 | 0.012323 |

Convergence signals были большими по aggregate: convex-dual validation
$d_X$ уменьшился $0.060786\to0.015626$, а history-query arm
$0.060285\to0.015829$. Но decomposition отвергает удобную интерпретацию:
zero-query arm почти оставляет pose у identity, и основная доля выигрыша
получена по joints. Явный history query относительно zero-query arm уменьшает
test translation на 5.87% и rotation на 10.58%, но ухудшает joints на 7.88% и
aggregate на 1.00%. Non-cyclic monotone arm уменьшает direct $d_X$ только на
21.0%, одновременно ухудшая translation на 10.7% и rotation на 37.9%.

Следовательно:

1. Symmetry/cyclic monotonicity была не единственной причиной слабого pose.
2. Firm nonexpansiveness — полезная numerical guarantee, но неверный
   constitutive prior для полного Coulomb response.
3. History является полезным direction query для smooth branch, но не решает
   switching и не превращает потенциальную модель в правильную механику.
4. Эти arms не принимаются как новая SRNO architecture, несмотря на >48%
   aggregate improvement: требуемого крупного pose-эффекта нет.

Полный JSON:
[`general resolvent ablation`](../runs/ablation-general-resolvent-v1/results.json).

### 30.5 Предлагаемая архитектура: Balanced Bipotential Neural Resolvent

Следующая научно содержательная единица должна быть не ещё одной output-head и
не ещё одним contact QP, а **lifted bivariational solution operator**. Рабочее
название: Balanced Bipotential Neural Resolvent (BBNR).

Один macro transition разворачивается в $S=4\ldots8$ shared pseudo-time
substeps. На substep $s$ состояние $y_s$, load coordinate $\alpha_s$ и
contact reactions $\lambda_s$ определяются совместно:

$$
v_{s,i}=J_i(y_s)z_s+\frac{h_i(y_s)}{\tau_s}n_i,
$$

$$
g_{s,i}=b_\mu(v_{s,i},\lambda_{s,i})
-\langle v_{s,i},\lambda_{s,i}\rangle\geq0,
$$

$$
D_y\mathcal E_{\rm act}(y_s;\bar r(\alpha_s))
-J(y_s)^T\lambda_s+\varepsilon_s M_\theta(c_s)z_s=0.
$$

Здесь geometry, $J$, actuator energy и analytic Coulomb bipotential известны;
network сначала учит только:

- SPD vanishing-viscosity/compliance metric $M_\theta(c_s)$;
- positive pseudo-time increments $\tau_s$ или normalized internal clock;
- при наличии новых данных — low-dimensional internal contact-mode state
  $\zeta_s$, а не unrestricted pose correction.

Primal $z_s$ и dual $\lambda_s$ находятся alternating convex minimization
по двум blocks либо semismooth root solve. Unrolled MVP использует фиксированное
малое число iterations; после проверки forward solver возможна implicit
differentiation, как в общих
[differentiable implicit layers](https://arxiv.org/abs/2010.07078). State
обновляется product retraction на $SE(3)\times\mathbb R^6$, после чего SDF и
Jacobians вычисляются заново. Это принципиально отличается от повторения MLP
при неизменной геометрии и от одной tangent projection.

Endpoint objective должен быть

$$
\begin{aligned}
\mathcal L={}&d_X^2(y_S,y_{k+1}^*)
+\lambda_b\sum_{s,i}g_{s,i}
+\lambda_{eq}\sum_s\|D\mathcal E_{\rm act}-J^T\lambda_s
+\varepsilon_sM_\theta z_s\|^2\\
&+\lambda_{ed}\left|
\mathcal E_{\rm act}(y_0)-\mathcal E_{\rm act}(y_S)
-\sum_{s,i}\langle v_{s,i},\lambda_{s,i}\rangle
-\sum_s\varepsilon_s\|z_s\|_{M_\theta}^2
\right|\\
&+\lambda_{feas}\sum_s\|\operatorname{ReLU}(-h_\phi(y_s))\|^2.
\end{aligned}
$$

Это objective на constitutive graph, equilibrium и energy--dissipation, а не
только endpoint imitation. Для будущего learned correction к analytic
bipotential допустимы только separately-convex nonnegative blocks, например
conditional ICNN по $v$ при фиксированном $\lambda$ и по $\lambda$ при
фиксированном $v$, с явной проверкой $b_\theta\geq\langle v,\lambda\rangle$.
Сразу учить две свободные scalar функции energy/dissipation не следует:
[VONNs](https://arxiv.org/abs/2112.09085) дают полезный variational template,
но также подчёркивают их non-identifiability; в SRNO эта проблема сильнее из-за
отсутствующих contact forces.

Точная терминология важна: BBNR не является resolvent maximal monotone
operator в исходных physical coordinates, потому что Coulomb graph
non-monotone. Это generalized bivariational/implicit solution operator; слово
`resolvent` относится к incremental solution map всего lifted problem.

### 30.6 Почему следующий experiment требует малой targeted collection

Текущий dataset хранит только settled endpoints, approximate actuator effort и
coarse contact count. В нём нет $(v_i,\lambda_i)$ pairs, contact identities,
normals, tangential impulses или stick/slip labels. Поэтому bipotential gap и
energy--dissipation balance нельзя проверить на jump path, а latent reactions
имеют множество допустимых разложений. Это не неудобство implementation, а
идентифицируемость: даже constitutive graph не задаёт bipotential единственным
образом, а здесь не наблюдается и сам graph.

Полный `srno sim collect` пока не нужен. Минимальный решающий dataset:

1. 10--20 уже известных jump-boundary states;
2. exact repeats и малые perturbations $q,r,\bar r$ по обе стороны event;
3. preserve/reset branches;
4. на каждом physics substep: contact pair identity, point, normal, separation,
   normal/tangential relative velocity, normal/tangential impulse/reaction,
   stick/slip flag, actuator effort и state;
5. два load refinements для проверки rate-independent/BV consistency.

До появления этих labels допустим только BBNR smooth-transition prototype с
analytic $b_\mu$. Jump claim по старому endpoint dataset был бы снова
непроверяемой генерацией модели.

Acceptance criterion для следующего local experiment следует зафиксировать
до test: не менее 40% снижения pose aggregate относительно direct L1,
одновременное снижение $T$ и $R$, не более 5% деградации joints
относительно implicit-QP и не менее 30% снижения jump-pose error. Только после
этого имеет смысл rollout.

### 30.7 Реализация и статус

Добавлены экспериментальные, но не production classes
`ContactDualPotentialCell` и `ContactMonotoneResolventCell`, configs четырёх
operator screens, initialization factor grid и consolidated evaluator.
Основной config `configs/srno-r-material-v2.toml` не изменён; simulator dataset
не перезаписывался; rollout не запускался.

Артефакты:

- [`model.py`](../src/srno/model.py);
- [`convex dual config`](../configs/ablation-dual-potential-mixed-local.toml);
- [`history-query config`](../configs/ablation-dual-potential-history-query-local.toml);
- [`non-cyclic monotone config`](../configs/ablation-monotone-resolvent-local.toml);
- [`consolidated evaluator`](../scripts/evaluate_general_resolvent_ablation.py);
- [`results`](../runs/ablation-general-resolvent-v1/results.json).

После изменений полный test suite: **99 passed, 4 CUDA-only skipped**.

Итог: local contact QP остаётся лучшим текущим numerical baseline, но не
центром новизны. Convex-potential и general-monotone neural resolvents получили
ясную convergence, однако не решили pose и потому отвергнуты как финальная
architecture. Наиболее обоснованное новое направление — learned
vanishing-viscosity path поверх analytic Coulomb bipotential с latent
primal--dual contact variables и energy--dissipation objective.

## 31. Продолжение formulation search: set-valued solution operator

Подробный derivation, experiment chain, novelty audit и controls находятся в
[`SRNO_OPERATOR_LEARNING_FORMULATION_AUDIT.md`](SRNO_OPERATOR_LEARNING_FORMULATION_AUDIT.md),
Sections 13--18. Здесь зафиксирован только итог, чтобы evolution report не
отставал от текущих artifacts.

**[Experiment]** Whole-loading clearance profile
$g_{\phi,x_0}(k,r)$ обнаружил в train data набор геометрически близких
физических solution paths с большим approximation headroom. Попытки превратить
его в один prediction через local implicit defect, learned energy, first jet
unilateral potential, rollout projection, branch stratification, vector-valued
RKHS, complete-SDF kernel, mass conditioning и learned manifold defect дали
test terminal `d_X` от `0.178779` до `0.206364`. Ни один single-valued вариант
не достиг сильного threshold и production checkpoint не заменён.

**[Experiment]** USD audit показал, что все 28 объектов имеют используемую
PhysX authored mass (`0.035--1.49 kg`), совпадающую с catalog, но manifest и
model input не содержат mass/inertia. Это доказанная неполнота input contract;
mass-conditioned RKHS улучшил point estimate, но не решил задачу.

**[Derivation]** При скрытых физических коэффициентах и contact/history mode
наблюдаемая задача естественно задаёт multifunction

$$
  \mathcal S(q)=\{Y(\xi):\pi(\xi)=q\}
  \subset (SE(3)\times\mathbb R^6)^{32},
$$

а не обязательно единственный smooth map. Реализованный непараметрический
estimator возвращает конечное множество complete physical train paths,
ближайших по contact-sensitive loading-profile metric. Target не используется
для построения множества; validation выбирает metric и наименьший cardinality,
достигающий половины production validation risk.

**[Experiment]** Validation выбрал `hinge_100`, `K=32`, risk `0.084566`.
Untouched test terminal point-to-set `d_X` снизился
`0.204097 -> 0.101120` (**50.45%**); translation — на 56.4%, rotation — на
39.5%, joints — на 58.3%. Улучшены все три test objects. Paired hierarchical
bootstrap: mean gain 50.65%, 95% CI `[45.58%,55.36%]`. Matched random set того
же размера даёт `0.142512 ± 0.002048`; geometry conditioning добавляет 29.0%
улучшения относительно random control.

**[Qualification]** Это сильный результат для point-to-set coverage solution
relation, не single-branch prediction. Oracle coherent-member path error
улучшен на 40.3%, поэтому production rollout не заменён. `K=64` даёт 54.81%
terminal point-to-set gain, но является только predeclared maximum-cardinality
diagnostic: выбранный validation model остаётся `K=32`.

**[Literature]** Generic set-valued approximation, multiple-solution neural
operators, best-of-many trajectory losses и conformal prediction sets уже
существуют; сам переход к set output не заявляется как novelty. Узкий
подтверждённый результат — contact-specific geometry-conditioned physical
solution relation и его controls на фиксированном SRNO contract.

Новые основные artifacts:

- [`set-valued evaluator`](../scripts/evaluate_set_valued_solution_operator.py);
- [`selected result`](../runs/set-valued-solution-operator-v2/results.json);
- [`mass contract audit`](../runs/object-mass-contract-v1/results.json).
