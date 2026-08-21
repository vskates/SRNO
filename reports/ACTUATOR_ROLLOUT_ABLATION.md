# Clean rollout ablation: aperture vs `drive_error`

## Вывод

Полный paired ablation на seeds 0, 1, 2 и curriculum

\[
H4\rightarrow H8\rightarrow H16\rightarrow H32
\]

завершён. `drive_error` улучшает средние результаты на H8 и H16, но это улучшение не переносится на H32. На основном H32-критерии кандидат хуже baseline на validation во всех трёх seeds и в среднем хуже на unseen test:

\[
E_{\mathrm{test}}^{\mathrm{drive}}-E_{\mathrm{test}}^{\mathrm{aperture}}
=+0.006481.
\]

Иерархический 95% bootstrap CI равен

\[
[-0.001089,\;0.017838],
\]

поэтому чистый rollout gain **не подтверждён**.

## Что именно сравнивалось

В обеих руках contact features оставались `gap`; геометрия, SDF, gate, loss, integral kernel, pooling, hidden dimension, head после conditioning, dataset material-v2, object split и optimizer schedule не менялись.

Состояние системы в обеих руках:

\[
x_k=(q_k,r_k),
\qquad
q_k=(R_k,p_k)\in SE(3),
\qquad
r_k\in\mathbb R^6.
\]

Проверялся авторегрессионный оператор

\[
\hat x_{k+1}
=F_\theta(\hat x_k,\bar a_{k+1},\phi),
\qquad
\hat x_0=x_0,
\]

без teacher forcing. Для следующей команды сначала строилась свободная trial-конфигурация суставов

\[
\tilde r_{k+1}=R_{\mathrm{free}}(\bar a_{k+1}).
\]

Для collision sample $i$, закреплённого за link $\ell(i)$, FK и SDF query имели вид

\[
y_i^G(\tilde r_{k+1})
=T_{\ell(i)}(\tilde r_{k+1})y_i,
\qquad
z_i^O=q_k^{-1}y_i^G(\tilde r_{k+1}),
\]

\[
h_i^{\mathrm{geom}}=\phi(z_i^O),
\qquad
h_i^{\mathrm{contact}}
=h_i^{\mathrm{geom}}-2.56\ \mathrm{mm}.
\]

Смещение 2.56 мм использовалось одинаково в обеих руках только для contact signal и gate. В geometric feasibility использовалось исходное $h_i^{\mathrm{geom}}$, без этого смещения.

Единственный локальный contact feature:

\[
f_i=\frac{h_i^{\mathrm{contact}}}{s_{\mathrm{sdf}}}\in\mathbb R.
\]

Неизменённый integral block:

\[
z_i=
\operatorname{SiLU}
\left(
W_0f_i+
\frac1M\sum_j
\kappa(\rho_i,\rho_j)\odot W_1f_j+b
\right),
\qquad
\bar z=\frac1M\sum_i z_i.
\]

Baseline передаёт в head два скаляра:

\[
c_k^{A}=\left[\frac{A(r_k)}{L},\frac{\bar a_{k+1}}{L}\right]\in\mathbb R^2,
\]

где $r_k\in\mathbb R^6$ — фактическая конфигурация суставов, $A(r_k)$ — derived aperture, $\bar a_{k+1}$ — следующая команда, $L$ — length scale.

Candidate вместо этого использует нормированную ошибку привода:

\[
\bar r_{k+1}=R_{\mathrm{free}}(\bar a_{k+1}),
\qquad
u_k=\frac{\bar r_{k+1}-r_k}{s}\in\mathbb R^6,
\]

где $R_{\mathrm{free}}$ — lookup свободной конфигурации gripper для команды, а $s\in\mathbb R^6$ — travel ranges суставов. Изменяется только вход первого head layer: $64+2\to64+6$, то есть $31\,436\to31\,948$ параметров ($+512$).

Далее обе руки использовали один и тот же output:

\[
(\Delta\xi,\Delta r^c)\in\mathbb R^{6+6},
\]

\[
\hat q_{k+1}
=\operatorname{Exp}(\widehat{\Delta\xi})\hat q_k,
\qquad
\hat r_{k+1}
=\tilde r_{k+1}+\Delta r^c.
\]

Обе руки стартовали из своих frozen paired `best-local.pt`, после чего для rollout создавался новый AdamW optimizer. Teacher forcing не использовался. На каждом горизонте допускалось до 25 epochs с patience 10; checkpoint выбирался только по validation terminal $d_X$.

Clean contract:

- seeds: 0, 1, 2;
- object split: 22 train / 3 validation / 3 test;
- evaluation: 100 trajectories на объект, то есть 300 validation и 300 unseen-test trajectories на checkpoint;
- curriculum: $H4\to H8\to H16\to H32$;
- baseline local initialization: `runs/ablation-jq-local/baseline/seed-{0,1,2}/best-local.pt`;
- candidate local initialization: `runs/ablation-actuator-local/drive_error/seed-{0,1,2}/best-local.pt`;
- manifest SHA-256: `c8bddec752f0418e92383fd0d9193e2d70a37845244c4c1c26ed9f9170c3012a`;
- gripper SHA-256: `6f3280535f5fd5bf543da3dba911825710ea73edb2a19b77b4fd4225fa2f02d6`.

Runner проверял hashes и полную идентичность model/loss/optimizer/loader/training config до начала каждой руки. Переход `local → rollout` загружал только веса модели и создавал новый rollout optimizer/scheduler.

Компоненты метрики:

\[
T_k=\|\hat p_k-p_k^*\|_2,
\qquad
T_k^L=\frac{T_k}{L},
\]

\[
R_k=\theta(\hat R_k,R_k^*)
=\arccos\left(
\operatorname{clip}
\frac{\operatorname{tr}(\hat R_kR_k^{*T})-1}{2},-1,1
\right),
\]

\[
J_k=
\sqrt{
\frac16\sum_{m=1}^{6}
\left(
\frac{\hat r_{k,m}-r_{k,m}^*}{s_m}
\right)^2
}.
\]

\[
d_X(k)=
\sqrt{
(T_k^L)^2+R_k^2+J_k^2
}.
\]

Terminal metric для curriculum horizon $H$:

\[
E_H=d_X(H).
\]

Trajectory state loss и feasibility loss оставались одинаковыми:

\[
\mathcal L
=\mathcal L_{\mathrm{state}}
+\lambda_K\mathcal L_K,
\qquad
\mathcal L_{\mathrm{state}}
=\frac1H\sum_{k=1}^{H}d_X(k)^2.
\]

Средние T/R/J в таблицах — это средние уже вычисленных trajectory components. Поэтому средний $d_X$ в общем случае не равен корню из квадратов средних T/R/J.

## Результаты по горизонтам

Значения ниже — equal-object mean по трём seeds на полном split.

| Horizon | aperture val | drive_error val | aperture test | drive_error test | test difference |
|---:|---:|---:|---:|---:|---:|
| H4  | 0.013552 | 0.014342 | 0.018253 | 0.019339 | +0.001087 |
| H8  | 0.034076 | 0.032777 | 0.044945 | 0.042922 | -0.002023 |
| H16 | 0.065297 | 0.061674 | 0.075581 | 0.070862 | -0.004719 |
| H32 | 0.171902 | 0.179335 | 0.206607 | 0.213213 | +0.006606 |

Относительное изменение test error:

\[
H4:+5.95\%,\qquad
H8:-4.50\%,\qquad
H16:-6.24\%,\qquad
H32:+3.20\%.
\]

Таким образом, actuator conditioning помогает на промежуточных горизонтах H8/H16, но после примерно 10–11 closure steps его pushforward error пересекает baseline и затем накапливается быстрее. На H32 test:

\[
0.206607\longrightarrow0.213213,
\qquad +3.20\%.
\]

## Полные T/R/J результаты всех горизонтов

Обозначения: T — translation в метрах, T/L — нормированная translation, R — geodesic rotation error в радианах, J/travel — joint RMSE, нормированная на travel range. Все значения — equal-object mean по трём seeds.

### Validation

| Arm | H | dX | T [m] | T/L | R [rad] | J/travel |
|---|---:|---:|---:|---:|---:|---:|
| aperture | 4 | 0.013551787 | 0.001111693 | 0.009970346 | 0.007668148 | 0.001603323 |
| aperture | 8 | 0.034075549 | 0.002489365 | 0.022326153 | 0.021775098 | 0.005507032 |
| aperture | 16 | 0.065296953 | 0.004624734 | 0.041477449 | 0.042989561 | 0.014495682 |
| aperture | 32 | 0.171902039 | 0.013184156 | 0.118243593 | 0.095097292 | 0.064208850 |
| drive_error | 4 | 0.014342212 | 0.001156162 | 0.010369168 | 0.007739661 | 0.002459650 |
| drive_error | 8 | 0.032776574 | 0.002391130 | 0.021445116 | 0.020880789 | 0.005274572 |
| drive_error | 16 | 0.061674431 | 0.004064275 | 0.036450909 | 0.043206698 | 0.013835103 |
| drive_error | 32 | 0.179334637 | 0.014301087 | 0.128260915 | 0.095333210 | 0.064834448 |

### Test

| Arm | H | dX | T [m] | T/L | R [rad] | J/travel |
|---|---:|---:|---:|---:|---:|---:|
| aperture | 4 | 0.018252572 | 0.001635874 | 0.014671516 | 0.008576471 | 0.002754367 |
| aperture | 8 | 0.044945031 | 0.003716006 | 0.033327417 | 0.024851562 | 0.008230999 |
| aperture | 16 | 0.075580784 | 0.005823461 | 0.052228363 | 0.045371630 | 0.016199924 |
| aperture | 32 | 0.206607107 | 0.016564174 | 0.148557658 | 0.104379216 | 0.073939927 |
| drive_error | 4 | 0.019339220 | 0.001677755 | 0.015047132 | 0.008953385 | 0.004089213 |
| drive_error | 8 | 0.042922475 | 0.003479662 | 0.031207737 | 0.024221371 | 0.007965376 |
| drive_error | 16 | 0.070861936 | 0.005102961 | 0.045766482 | 0.045049375 | 0.015828616 |
| drive_error | 32 | 0.213212742 | 0.017841172 | 0.160010552 | 0.101346730 | 0.074008002 |

## H32 paired seeds

| Seed | aperture val | drive_error val | Δ val | aperture test | drive_error test | Δ test |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.171675 | 0.181717 | +0.010042 | 0.204097 | 0.222183 | +0.018086 |
| 1 | 0.172902 | 0.178284 | +0.005382 | 0.207260 | 0.208947 | +0.001688 |
| 2 | 0.171129 | 0.178002 | +0.006873 | 0.208465 | 0.208508 | +0.000043 |

Знак определён как

\[
\Delta=E^{\mathrm{drive\_error}}-E^{\mathrm{aperture}},
\]

поэтому положительное значение означает ухудшение. Все три validation differences положительны.

### Все terminal dX по seeds и curriculum horizons

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

## H32 decomposition

Средние terminal components по трём seeds:

| Split | Arm | translation [m] | translation / L | rotation [rad] | joint RMSE / travel | aggregate d_X |
|---|---|---:|---:|---:|---:|---:|
| val | aperture | 0.013184 | 0.118244 | 0.095097 | 0.064209 | 0.171902 |
| val | drive_error | 0.014301 | 0.128261 | 0.095333 | 0.064834 | 0.179335 |
| test | aperture | 0.016564 | 0.148558 | 0.104379 | 0.073940 | 0.206607 |
| test | drive_error | 0.017841 | 0.160011 | 0.101347 | 0.074008 | 0.213213 |

На test кандидат немного уменьшает rotation error,

\[
0.104379\to0.101347\ \mathrm{rad},
\]

но увеличивает translation,

\[
16.564\to17.841\ \mathrm{mm},
\]

а joint component практически не меняется. Итоговое ухудшение $d_X$ в основном связано с translation drift.

## Полная H32 pushforward-кривая dX(k)

Ниже test $d_X(k)$, усреднённый сначала внутри каждого test object, затем по трём объектам и трём seeds. Разность определена как `drive_error - aperture`.

| k | aperture | drive_error | Δ |
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

До $k=10$ `drive_error` лучше, на $k=11$ знак меняется. После этого candidate остаётся хуже до terminal state. Максимальный средний разрыв наблюдается около $k=20$:

\[
\Delta d_X(20)=+0.011588.
\]

Это уточняет диагноз: новый conditioning улучшает первые contact increments, но не уменьшает долговременную ошибку композиции оператора.

## Per-object H32 validation

| Object | aperture | drive_error | Δ |
|---|---:|---:|---:|
| `masliny-federici-bez-kostochki-300-g-90215` | 0.169233864 | 0.174453144 | +0.005219281 |
| `sous-soevyy-250-ml-58088` | 0.189071248 | 0.190287054 | +0.001215806 |
| `voda-pitevaya-senezhskaya-negazirovannaya-pet-1-5-l-43733` | 0.157401005 | 0.173263714 | +0.015862708 |

Candidate хуже на каждом validation object, а не только в aggregate mean.

## Unseen-object H32 test

Equal-trajectory mean внутри объекта, затем mean по трём seeds:

| Object | aperture | drive_error | Δ |
|---|---:|---:|---:|
| `gerkules-ovsyanye-khlopya-400-g-1248` | 0.194221631 | 0.204014942 | +0.009793311 |
| `pechene-sdobnoe-khlebnyy-spas-italyanskoe-s-apelsinovym-vkusom-i-izyumom-230-g-46835` | 0.256522169 | 0.266266217 | +0.009744048 |
| `voda-pitevaya-prirodnaya-legenda-baykala-750-ml-42674` | 0.169077521 | 0.169357066 | +0.000279546 |

Средняя разность положительна для каждого из трёх unseen objects.

## Строгий критерий

Требовались одновременно:

1. $E_{\mathrm{val,H32}}^{\mathrm{drive}}<E_{\mathrm{val,H32}}^{\mathrm{aperture}}$ во всех трёх paired seeds;
2. верхняя граница 95% hierarchical bootstrap CI для test difference строго ниже нуля.

Bootstrap выполнялся с 10 000 replicates и seed 20260818 по иерархии

\[
\text{seed}\rightarrow\text{object}\rightarrow\text{trajectory}.
\]

Получено:

\[
\overline\Delta_{\mathrm{bootstrap}}=+0.006481,
\qquad
CI_{95\%}=[-0.001089,0.017838].
\]

Прямая разность итоговых equal-object means:

\[
\Delta_{\mathrm{direct}}
=0.213212742-0.206607107
=+0.006605635.
\]

$0.006481$ — mean конечного Monte Carlo bootstrap distribution, а $0.006606$ — непосредственная разность агрегированных метрик. Их небольшое отличие возникает из-за иерархического resampling и конечных 10 000 replicates.

Оба условия не выполнены. Поэтому результат ablation — **оставить `aperture` как рабочее global conditioning для полного rollout**. Это не отменяет ранее наблюдавшийся local gain `drive_error`; эксперимент показывает, что он не композируется устойчиво до H32 в текущей cell.

## Артефакты и воспроизводимость

- Полные агрегаты, paired differences и bootstrap: `runs/ablation-actuator-rollout/results.json`.
- Raw arrays $d_X(k)$, T/R/J и labels: `rollout_evaluation.npz` каждой руки/seed.
- Checkpoints: 24 файла `best-rollout-h{04,08,16,32}.pt`.
- TensorBoard: 6 event files, по одному на руку/seed.
- Сводный график: `runs/ablation-actuator-rollout/actuator_rollout_ablation.png`.
- Runner: `scripts/run_actuator_rollout_ablation.py`.

Проверка проекта после эксперимента:

```text
56 passed, 12 warnings in 9.26s
```

Warnings относятся к deprecation в `yourdfpy` и известному scheduler warning в all-free training test; падений нет.
