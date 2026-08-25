# SRNO: исходная формализация, постановка задачи и цель проекта

Этот документ фиксирует **исходную научную постановку SRNO**, сформулированную
до последующих исправлений simulator geometry, перехода от scalar aperture к
actual joint state и проведённых ablation. Его назначение — дать новому
обсуждению полный контекст: какую физическую задачу мы хотели решить, почему
была выбрана operator-learning постановка, что именно входило в первый
эксперимент и какая более широкая notion of grasp стоит за проектом.

Это не описание текущего состояния кода. Эволюция реализации и результаты
последующих экспериментов находятся в
[`SRNO_EVOLUTION_REPORT.md`](SRNO_EVOLUTION_REPORT.md).

## 1. Главная цель проекта

Проект строится не вокруг бинарного grasp classifier и не вокруг прямого
предсказания contact forces. Его центральный объект — механический оператор,
который по геометрии объекта воспроизводит последовательность quasistatic
contact equilibria при закрытии gripper.

Первая проверяемая гипотеза:

$$
\boxed{
\text{Можно ли по complete SDF объекта обучить один shared local resolvent,}
\\
\text{который воспроизводит quasistatic finite-actuation contact dynamics?}
}
$$

То есть сначала требуется идентифицировать **mechanics operator**. Только после
того, как он научился переноситься на unseen object geometry и устойчиво
композироваться на полном closure rollout, имеет смысл строить на его основе
grasp-success или robustness functional.

Конечная цель проекта шире первого эксперимента:

$$
\boxed{
\text{successful grasp}
\neq
\text{unforced stationary solution}
}
$$

и должна формулироваться как

$$
\boxed{
\text{successful grasp}
=
\text{contact-supported state retained under nonzero admissible loads}.
}
$$

Таким образом, общая программа состоит из двух этапов:

1. **Closure operator:** выучить incremental contact resolvent при заданном
   finite-actuation closing schedule.
2. **Loaded retention operator:** тем же shared forced-resolvent описывать
   реакцию захвата на внешние object-directed disturbances и оценивать
   mechanical robustness.

Первый эксперимент ограничен пунктом 1.

## 2. Физическая система и системы координат

Рассматриваются один rigid object и один фиксированный parallel-jaw gripper.

- $O$ — object frame;
- $G$ — gripper frame;
- $q=(R,p)\in SE(3)$ — pose объекта относительно gripper;
- $R\in SO(3)$, $p\in\mathbb R^3$.

Convention для pose:

$$
\boxed{x_G=R x_O+p.}
$$

Следовательно, переход из gripper frame в object frame:

$$
x_O=R^\top(x_G-p).
$$

Объект задаётся complete signed distance field

$$
\phi:\mathbb R^3\rightarrow\mathbb R,
$$

где

$$
\phi(x)>0\quad\text{снаружи объекта},
\qquad
\phi(x)=0\quad\text{на поверхности},
\qquad
\phi(x)<0\quad\text{внутри объекта}.
$$

## 3. Почему aperture должна входить в состояние

В самой ранней conceptual версии состояние содержало только object pose

$$
q_k\in SE(3),
$$

а закрытие gripper считалось prescribed и фактически бесконечно сильным. Такая
модель не различает command и фактическое положение fingers.

Для finite actuator исходное расширенное состояние было определено как

$$
\boxed{x_k=(q_k,a_k)\in SE(3)\times\mathbb R,}
$$

где $a_k$ — **actual aperture**, то есть измеренное физическое расстояние
между fingers, а не управляющая команда.

Отдельно задан монотонный loading schedule

$$
\bar a_0>\bar a_1>\ldots>\bar a_N=a_{\min}.
$$

Здесь $\bar a_k$ — commanded aperture, а $a_{\min}$ — минимальное открытие
свободно закрывшегося gripper.

В free space ожидается

$$
a_k\simeq\bar a_k.
$$

В контакте из-за конечного actuator effort возможно

$$
a_k\ne\bar a_k,
\qquad
a_k>\bar a_k.
$$

В исходной абстрактной записи $F_{\max}$ обозначает максимальное доступное
усилие или torque actuator. Это не внешняя сила на объект, а ограничение
closing drive: actuator пытается уменьшить aperture, но contact reaction может
остановить его раньше $a_{\min}$.

Искомый incremental operator:

$$
\boxed{
\mathcal R_\phi:(x_k,\bar a_{k+1})\mapsto x_{k+1}.
}
$$

Все increments используют один и тот же shared operator с одними weights.

Полный closure rollout:

$$
\boxed{
\hat x_N
=
R_\theta^{(N)}\circ\ldots\circ R_\theta^{(1)}(q_0,a_0).
}
$$

## 4. Contact geometry через gap field

На collision-relevant поверхностях двух fingers фиксируется набор canonical
samples

$$
y_1,\ldots,y_M,
\qquad
\boxed{M=256}
$$

для первого эксперимента.

При aperture $a$ kinematics gripper переводит sample в gripper frame:

$$
x_j^G(a)=x_G(y_j,a).
$$

В object frame:

$$
x_j^O(q,a)=R^\top\left(x_j^G(a)-p\right).
$$

Gap field:

$$
\boxed{
h_{\phi,q,a}(y_j)
=
\phi\!\left(R^\top(x_j^G(a)-p)\right).
}
$$

Интерпретация:

$$
h>0\;\text{— separation},
\qquad
h=0\;\text{— contact},
\qquad
h<0\;\text{— penetration}.
$$

Расширенное feasible set:

$$
\boxed{
\mathcal C_\phi
=
\{(q,a):h_{\phi,q,a}(y)\ge0\}.
}
$$

## 5. Quasistatic механическая интерпретация

Conceptually система рассматривается как perturbed sweeping/resolvent process.
Схематическая quasistatic inclusion имеет вид

$$
0\in
\partial_{\dot x}\mathcal D(x,\dot x)
+N_{\mathcal C_\phi}(x)
-F_{\rm act}
-P_q^\top w,
$$

где

- $\mathcal D$ описывает dissipative/frictional response;
- $N_{\mathcal C_\phi}(x)$ — normal cone contact constraints;
- $F_{\rm act}$ — closing actuator forcing;
- $w\in\mathfrak{se}(3)^*$ — внешний wrench на объект;
- $P_q^\top w=(w,0)$, то есть disturbance действует на object pose, но не
  непосредственно на aperture.

В interior feasible set normal cone равен нулю. Contact reaction возникает
только на активной границе constraints.

Первый experiment использует

$$
\boxed{w=0}
$$

и фиксированный actuator law. Внешние wrenches относятся к следующему этапу.

## 6. Почему stationarity не является grasp criterion

### 6.1 Свободное закрытие

Если gripper далеко от объекта,

$$
h_{\phi,q,a}(y)>0,
\qquad
N_{\mathcal C_\phi}(x)=\{0\}.
$$

Во время активного закрытия:

$$
\boxed{\dot q=0,\qquad\dot a<0.}
$$

Object pose физически правильно остаётся неизменной, но полное состояние

$$
x(t)=(q_0,a(t))
$$

не является stationary, пока gripper закрывается.

После достижения $a_{\min}$:

$$
\dot q=0,
\qquad
\dot a=0,
\qquad
x(T)=(q_0,a_{\min}).
$$

Это terminal fixed point, хотя grasp отсутствует. Следовательно, критерий
«система перестала двигаться» даёт free-space false positive.

### 6.2 Loaded stall

При настоящем contact-supported stall command всё ещё требует закрытия:

$$
\bar a(t)<a(t),
$$

но объект препятствует движению fingers:

$$
a(t)=a_*>a_{\min},
\qquad
\dot a=0,
\qquad
u_{\rm act}\ne0.
$$

Минимальный критерий loaded stall:

$$
\boxed{
|\dot a|<\varepsilon_a,
\qquad
a>a_{\min}+\delta_a,
\qquad
\text{closing drive остаётся активным}.
}
$$

Он исключает trivial free-space terminal state, поскольку свободный gripper
закроется до $a_{\min}$.

Но loaded stall доказывает только

$$
\boxed{\text{наличие механического взаимодействия с объектом},}
$$

а не stable retention. Объект может заклинить closing direction и всё равно
выпасть при lateral force или gravity.

### 6.3 Иерархия состояний

Исходная формализация различает три уровня:

1. **No contact:** $a_T=a_{\min}$; закрытие нигде не было заблокировано. Это
   не grasp.
2. **Contact-supported closure:** $a_*>a_{\min}$, $\dot a=0$, closing
   drive активен. Free-space solution исключена, но retention ещё не доказана.
3. **Disturbance-supported equilibrium:** при ненулевом object-directed wrench
   объект остаётся trapped. Это mechanically stable grasp.

Эта логика согласуется с caging intuition: stationarity сама по себе ничего не
говорит о capture; нужны отсутствие escaping path и физически поддерживаемое
equilibrium.

## 7. Конечная notion of grasp: loaded retention и invariance

После closure предполагается сохранять actuator preload и прикладывать
допустимые внешние возмущения

$$
w\in\mathcal W.
$$

Например, $\mathcal W$ может включать gravity, небольшие lateral forces и
torques.

Capture определяется не unforced fixed point, а сохранением retention set:

$$
\boxed{
\forall w(\cdot)\in\mathcal W,
\qquad
q^w(t)\ \text{не выходит из held region}.
}
$$

Можно определить robustness margin

$$
\mathcal W_\rho=\{w:\|w\|_W\le\rho\},
$$

$$
\boxed{
\rho_*
=
\sup\left\{
\rho:
\text{object retained для всех }w\in\mathcal W_\rho
\right\}.
}
$$

Для свободного объекта

$$
\rho_*=0,
$$

поскольку любое ненулевое object-directed воздействие вызывает движение. Для
устойчивого grasp ожидается

$$
\rho_*>0.
$$

Альтернативная чисто геометрическая notion использует connected component
free configuration space при terminal aperture:

$$
\mathcal F_\phi(a_*)
=
\{q:h_{\phi,q,a_*}(y)\ge0\}.
$$

Если component, содержащая $q_*$, compact/bounded, объект caged. Но проверка
global connectivity в $SE(3)$ слишком дорога для MVP, поэтому practical
путь — disturbance trajectories через learned operator.

Если исследуется contraction capture basin, корректная величина — не абсолютный
terminal diameter, а

$$
\boxed{
\kappa
=
\frac{\operatorname{diam}\{q_T^{(m)}\}}
{\operatorname{diam}\{q_0^{(m)}\}}.
}
$$

Для free-space identity $\kappa\simeq1$; $\kappa<1$ означает contraction,
но само по себе также не является grasp certificate.

## 8. Жёстко зафиксированные assumptions первого эксперимента

Первый identification experiment намеренно минимален:

$$
\boxed{
\begin{aligned}
&\text{complete object SDF }\phi;\\
&\text{one fixed parallel-jaw gripper};\\
&\text{rigid objects};\\
&\text{fixed friction and material parameters};\\
&\text{fixed finite actuator law and }F_{\max};\\
&\text{fixed monotone schedule }\bar a_k;\\
&\text{quasistatic incremental closure};\\
&\text{no gravity};\\
&\text{no external wrench};\\
&\text{no table, shelf or environment contacts}.
\end{aligned}
}
$$

Цель этих ограничений — получить чистую operator-identification задачу. Если
одновременно менять friction, force, environment и external loads, невозможно
локализовать причину неудачного обучения.

## 9. Dataset как последовательность equilibria

Dataset не должен быть обычной быстро записанной rigid-body trajectory. Для
каждого increment

$$
k=0,\ldots,N-1
$$

выполняется:

1. target меняется с $\bar a_k$ на $\bar a_{k+1}$;
2. finite actuator прикладывает closing effort;
3. simulator выполняет physics substeps;
4. ожидается практическое equilibrium объекта и gripper;
5. записывается
   $$
   x_{k+1}^*=(q_{k+1}^*,a_{k+1}^*);
   $$
6. только затем задаётся следующий load increment.

Если остаются значимые velocities, состояние $(q,a)$ может быть недостаточно:

$$
x_{k+1}
=
F(q_k,a_k,\dot q_k,\dot a_k,\bar a_{k+1}).
$$

Тогда задача незаметно превращается из first-order quasistatic process в
second-order dynamics. Увеличение neural network это не исправит.

Обязательный sanity check:

$$
\boxed{
\text{уменьшение closure speed или увеличение settling time}
\text{ не должно существенно менять записанные equilibria}.
}
$$

Минимальная trajectory record исходной постановки:

$$
\boxed{
\left(
\phi,
q_0^*,a_0^*,
\{\bar a_k,q_k^*,a_k^*\}_{k=1}^{N}
\right).
}
$$

Contact forces, friction multipliers и contact points не требуются как ML
targets. Но для диагностики simulator желательно сохранять:

- actuator effort;
- contact count;
- maximum penetration;
- residual linear/angular velocity;
- residual joint/finger velocity;
- число physics substeps до settling.

Эти поля не являются входами network или членами loss.

## 10. Sampling initial configurations

Равномерное sampling всего $SE(3)$ неэффективно: почти все примеры будут
далеко от объекта или грубо collision-invalid.

До simulation можно вычислить kinematic gap при неподвижном объекте:

$$
m_k^{\rm kin}(g_0)
=
\min_j h_{\phi,g_0,\bar a_k}(y_j),
$$

и approximate first-contact step

$$
k_c
=
\min\{k:m_k^{\rm kin}\le0\}.
$$

Initial poses следует стратифицировать по $k_c$ и типу взаимодействия. Нужны:

- early и late contact;
- grazing contact;
- one-finger/asymmetric contact;
- approximately symmetric contact;
- cases, вызывающие translation;
- cases, вызывающие rotation;
- cases, где object выталкивается;
- complete no-contact trajectories.

Free-space trajectories нужны для проверки deterministic bypass, но не должны
доминировать в active-contact training set.

Исходный целевой масштаб первого полного dataset:

$$
\boxed{500\text{--}1000\ \text{objects}},
$$

$$
\boxed{32\text{--}64\ \text{initial poses per object}},
$$

то есть

$$
\boxed{2\cdot10^4\text{--}6\cdot10^4\ \text{trajectories}.}
$$

Для pipeline debugging предлагался subset порядка $100\times32$
trajectories, но он недостаточен для выводов о generalization.

Split обязательно выполняется по object identity, а не по trajectory:

$$
\boxed{\text{train/val/test object-wise split}.}
$$

## 11. Зафиксированная дискретизация и SDF

Для первого эксперимента:

$$
\boxed{N=32},
\qquad
\boxed{M=256},
\qquad
\boxed{d=64}.
$$

Не проверяется temporal-resolution generalization. Все trajectories имеют одну
и ту же 32-step command schedule.

Исходное SDF resolution:

$$
\boxed{96^3}.
$$

Более содержательный numerical criterion:

$$
\boxed{
\text{voxel size}<\frac12\delta_{\rm gate}.
}
$$

SDF interpolation должна быть differentiable по query coordinates, поскольку
feasibility loss вычисляется в predicted pose.

## 12. Deterministic free predictor

Neural network не должна учить известную free-space physics.

Из $x_k=(q_k,a_k)$ и новой command строится trial state

$$
\boxed{
\tilde x_{k+1}
=
(\tilde q_{k+1},\tilde a_{k+1}).
}
$$

При отсутствии контакта

$$
\tilde q_{k+1}=q_k.
$$

Для идеального quasistatic position drive

$$
\tilde a_{k+1}=\bar a_{k+1}.
$$

Если empty gripper имеет воспроизводимый servo error, его следует один раз
измерить и задать известной функцией или lookup table:

$$
\tilde a_{k+1}
=
A_{\rm free}(a_k,\bar a_{k+1}).
$$

## 13. Trial gap и exact contact gate

Gap вычисляется не в текущем состоянии, а в состоянии, куда gripper попытался
бы перейти без контакта:

$$
\boxed{
\tilde h_{k,j}
=
h_{\phi,q_k,\tilde a_{k+1}}(y_j).
}
$$

Если free closure пересекает объект, отрицательный $\tilde h$ непосредственно
кодирует величину нарушения constraint.

Определим

$$
m_k=\min_j\tilde h_{k,j}.
$$

При

$$
\boxed{m_k>\delta_{\rm gate}}
$$

neural cell не вызывается:

$$
\boxed{\hat x_{k+1}=\tilde x_{k+1}.}
$$

Это exact free-space bypass. Значение $\delta_{\rm gate}$ должно быть
откалибровано по simulator contacts и покрывать

$$
\text{SDF interpolation error}
+\text{surface sampling error}
+\text{simulator contact offset}.
$$

Free identity является известной physics, а не learning objective.

## 14. Единственная learned часть: shared contact-resolvent cell

Если

$$
m_k\le\delta_{\rm gate},
$$

shared neural cell предсказывает только contact correction trial state.

Normalized scalar field:

$$
e_{k,j}=\frac{\tilde h_{k,j}}{s_{\rm sdf}}.
$$

Normalized current gripper surface coordinates:

$$
r_{k,j}=\frac{x_G(y_j,\tilde a_{k+1})}{\ell},
$$

где $\ell$ — characteristic gripper length.

Один nonlocal integral layer:

$$
\boxed{
z_{k,i}
=
\sigma\!\left(
W_0e_{k,i}
+\frac1M\sum_{j=1}^{M}
\kappa_\theta(r_{k,i},r_{k,j})W_1e_{k,j}
+b
\right),
}
$$

где

$$
z_{k,i}\in\mathbb R^{64}
$$

и $\kappa_\theta$ — небольшой MLP от pairwise sample coordinates.

Mean pooling:

$$
\boxed{\bar z_k=\frac1M\sum_i z_{k,i}.}
$$

Head:

$$
\boxed{
(\Delta\xi_k,\eta_k)
=
\rho_\theta\!\left(
\bar z_k,
\frac{a_k}{\ell},
\frac{\tilde a_{k+1}}{\ell}
\right).
}
$$

Здесь

$$
\Delta\xi_k\in\mathfrak{se}(3)\simeq\mathbb R^6
$$

— contact-induced spatial twist объекта в gripper frame.

Object update:

$$
\boxed{
\hat q_{k+1}
=
\Exp(\widehat{\Delta\xi_k})q_k.
}
$$

Aperture correction параметризуется так, чтобы gripper мог полностью
закрыться, частично отстать или stall, но не раскрывался самопроизвольно:

$$
\alpha_k=\operatorname{sigmoid}(\eta_k),
$$

$$
\boxed{
\hat a_{k+1}
=
\tilde a_{k+1}
+\alpha_k(a_k-\tilde a_{k+1}).
}
$$

Автоматически

$$
\tilde a_{k+1}\le\hat a_{k+1}\le a_k.
$$

Интерпретация:

- $\alpha=0$: contact не препятствует free closure;
- $\alpha=1$: полный stall;
- $0<\alpha<1$: partial closure.

Отдельный stall classifier не требуется.

Полный step:

$$
\boxed{
x_{k+1}
=
\begin{cases}
\tilde x_{k+1},
&\min_j\tilde h_{k,j}>\delta_{\rm gate},\\[1mm]
R_\theta[\tilde h_k](\tilde x_{k+1},a_k),
&\min_j\tilde h_{k,j}\le\delta_{\rm gate}.
\end{cases}
}
$$

Это один и тот же block, повторённый 32 раза.

## 15. Loss первого эксперимента

Pose metric:

$$
d_{SE(3)}^2(q,q^*)
=
\frac{\|p-p^*\|_2^2}{\ell^2}
+\lambda_R
\|\Log(R^{*\top}R)\|_2^2.
$$

State metric:

$$
\boxed{
d_X^2(x,x^*)
=
d_{SE(3)}^2(q,q^*)
+\lambda_a\frac{(a-a^*)^2}{\ell^2}.
}
$$

Trajectory/state loss:

$$
\boxed{
\mathcal L_{\rm flow}
=
\frac1{BN}\sum_{b,k}
d_X^2(\hat x_{b,k},x_{b,k}^*).
}
$$

Для predicted state

$$
\hat h_{b,k,j}
=
h_{\phi,\hat q_{b,k},\hat a_{b,k}}(y_j)
$$

задаётся feasibility penalty

$$
\boxed{
\mathcal L_K
=
\frac1{BNM}
\sum_{b,k,j}
\left[
\frac{[-\hat h_{b,k,j}]_+}{s_{\rm sdf}}
\right]^2.
}
$$

Итоговый loss:

$$
\boxed{
\mathcal L
=
\mathcal L_{\rm flow}
+\lambda_K\mathcal L_K.
}
$$

Только два члена.

Отдельный free loss удаляется, поскольку free dynamics обеспечена exact bypass.
Отдельный stall loss также не нужен: actual aperture уже входит в
$\mathcal L_{\rm flow}$.

## 16. Два этапа обучения

### 16.1 Local transition identification

Сначала cell обучается на one-step active-contact transitions из ground-truth
states:

$$
x_k^*\rightarrow x_{k+1}^*.
$$

Trial state строится как

$$
\tilde x_{k+1}=F_{\rm free}(x_k^*,\bar a_{k+1}).
$$

Цель этапа — проверить, существует ли в выбранном state представлении
идентифицируемый local resolvent и способна ли минимальная cell его выучить.

Если one-step model не обучается, полный rollout запускать бессмысленно: сначала
следует диагностировать simulator, geometry и state sufficiency.

### 16.2 Autoregressive rollout training

После local initialization:

$$
\hat x_0=x_0^*,
$$

$$
\boxed{
\hat x_{k+1}
=
R_\theta[\phi,\bar a_{k+1}](\hat x_k).
}
$$

Следующий step получает предыдущую prediction, а не ground-truth state. Это
training без teacher forcing.

Если полный rollout нестабилен, применяется curriculum

$$
4\rightarrow8\rightarrow16\rightarrow32.
$$

Финальный критерий всегда вычисляется на полном $N=32$.

## 17. Evaluation первого оператора

На unseen objects измеряются:

Translation:

$$
E_p
=
\frac1N\sum_k\frac{\|\hat p_k-p_k^*\|_2}{\ell}.
$$

Rotation:

$$
E_R
=
\frac1N\sum_k
\|\Log(R_k^{*\top}\hat R_k)\|_2.
$$

Aperture:

$$
E_a
=
\frac1N\sum_k
\frac{|\hat a_k-a_k^*|}{\ell}.
$$

Terminal state:

$$
d_X(\hat x_N,x_N^*).
$$

Feasibility:

$$
\operatorname{mean/max}_{k,j}
[-h_{\phi,\hat q_k,\hat a_k}(y_j)]_+.
$$

Actuation lag:

$$
s_k=a_k-\bar a_k,
\qquad
\hat s_k\ \text{vs}\ s_k^*.
$$

Отдельно проверяются onset significant lag, terminal actual aperture и полный
stall plateau. Если object pose предсказывается хорошо, но actual aperture нет,
finite-actuation problem не считается решённым.

Free-space trajectory является structural test, а не ML achievement:

$$
\hat q_k=q_0,
\qquad
\hat a_k=a_{\rm free,k}
$$

должно выполняться by construction.

## 18. State sufficiency как обязательная проверка

Предполагается deterministic Markov operator

$$
x_{k+1}=F_\phi(x_k,\bar a_{k+1}),
\qquad
x_k=(q_k,a_k).
$$

Если два практически одинаковых $(q_k,a_k)$ из-за velocity, tangential
friction memory, compliant contact history или solver state имеют разные
successors, истинный map не является single-valued в выбранном state.

Тогда никакой deterministic neural architecture не сможет идеально решить
задачу. Первый вопрос при irreducible one-step error:

$$
\boxed{\text{достаточен ли state?}}
$$

а не «нужен ли transformer или более глубокая сеть?».

## 19. Что сознательно не входит в первый эксперимент

Не добавляются:

- global SDF encoder;
- SDF-gradient или surface-normal features;
- transformer или FNO stack;
- contact-force prediction;
- friction/contact multiplier head;
- success classifier;
- grasp-quality head;
- force-closure или terminal grasp-success loss;
- variable friction;
- variable actuator strength;
- gravity;
- external wrench;
- environment geometry;
- partial point clouds;
- auxiliary latent, contrastive или equivariance losses.

Это не утверждение, что они никогда не понадобятся. Они исключены, чтобы первый
эксперимент отвечал ровно на один вопрос: может ли минимальный shared
contact-resolvent идентифицировать closure mechanics?

## 20. Следующее расширение после успешного closure operator

После доказанного unseen-object rollout добавляется hold/probe phase.

### Closure phase

$$
w_k=0,
$$

$$
R_\theta:(h_k,a_k,\bar a_{k+1})
\mapsto(\Delta\xi_k,\Delta a_k).
$$

Closure заканчивается либо при $a=a_{\min}$, либо при loaded stall:

$$
a>a_{\min},
\qquad
|\Delta a|\simeq0,
\qquad
\text{drive active}.
$$

### Hold/probe phase

Actuator preload сохраняется и подаётся known object wrench:

$$
\boxed{
R_\theta:(h_k,a_k,\bar a_k,w_k)
\mapsto(\Delta\xi_k,\Delta a_k).
}
$$

Новый network block или success classifier не обязателен: forced closure и
loaded retention должны описываться одной shared resolvent cell с дополнительным
known forcing input.

Ground-truth simulator генерирует probe trajectories при разных $w$, а
robustness определяется через retention/invariance или margin $\rho_*$.

## 21. Исходный implementation graph

```text
x_hat = x0

for k = 0 ... 31:

    # Known free mechanics
    q_trial = q_hat
    a_trial = A_free(a_hat, a_cmd[k+1])

    # Contact geometry
    for j = 1 ... 256:
        x_obj[j] = q_trial^{-1} x_gripper(y[j], a_trial)
        h[j] = SDF_phi(x_obj[j])

    # Exact free-space branch
    if min(h) > delta_gate:
        q_hat = q_trial
        a_hat = a_trial

    else:
        # Only learned part: shared contact resolvent
        z = IntegralLayer(h, x_gripper(y, a_trial))
        z_bar = MeanPool(z)

        delta_xi, eta = MLP(z_bar, a_hat, a_trial)

        q_hat = Exp(delta_xi) q_trial

        alpha = sigmoid(eta)
        a_hat = a_trial + alpha * (a_hat - a_trial)

    loss += state_error(x_hat, x_gt[k+1])
    loss += lambda_K * penetration_penalty(x_hat)
```

## 22. Краткая постановка для передачи новому агенту

SRNO изначально задуман как operator-learning модель для quasistatic
finite-actuation contact dynamics fixed parallel-jaw gripper. Объект задаётся
complete SDF $\phi$, состояние первого MVP — actual object pose и actual
aperture $x=(q,a)$, управление — следующая commanded aperture
$\bar a_{k+1}$. Известный free predictor сначала пытается закрыть gripper без
contact. SDF query на 256 collision surface samples формирует trial gap field.
При гарантированном separation выполняется exact bypass; около контакта одна
shared integral cell предсказывает только object twist и степень contact-induced
stall. Cell повторяется 32 раза. Dataset состоит из settled equilibria после
каждого command increment, split выполняется по объектам. Loss содержит только
trajectory state error и geometric feasibility.

Первый scientific question — существует ли и обучается ли такой shared local
resolvent на unseen object geometry, а также устойчиво ли он композируется в
H32 rollout. Stationarity или loaded stall сами по себе не считаются successful
grasp. Конечная цель — расширить тот же operator known external wrench input и
определять grasp как loaded retention/invariance под ненулевыми допустимыми
воздействиями, без отдельного success classifier.

## 23. Важная историческая оговорка

В этом документе намеренно сохранено исходное состояние

$$
x=(q,a).
$$

В последующей реализации выяснилось, что scalar aperture недостаточно точно
задаёт фактическую collision geometry многосуставного gripper, и state был
заменён на actual six-joint configuration

$$
x=(q,r),
\qquad
r\in\mathbb R^6.
$$

Это не меняет исходную цель проекта — identification shared quasistatic
contact-resolvent, — но является более точной реализацией gripper state.
Причины, формулы перехода $a\to r$, исправления physics/data contract и все
численные результаты описаны отдельно в evolution report.
