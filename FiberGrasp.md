# Новая независимая итерация — 2026-08-25

## Итоговый кандидат

**FiberGrasp: Learning Necessary and Possible Grasp Sets from Occluded RGB-D**

Короткая формулировка идеи:

> Вместо восстановления единственной скрытой формы или оценки средней вероятности успеха FiberGrasp предсказывает непосредственно на многообразии параллельных захватов два вложенных множества: захваты, допустимые для **всех** полных сцен, неразличимых по текущему RGB-D, и захваты, допустимые хотя бы для **одной** такой сцены.

Это непрерывный action-space аналог нижнего и верхнего приближений rough-set theory. Невидимая геометрия не превращается в одну «наиболее вероятную» форму, а задаёт **волокно наблюдения** — множество физических миров, совместимых с тем, что камера действительно увидела. Метод обучает амортизированный оператор

$$
o\longmapsto\bigl(\mathcal G_-(o),\mathcal G_+(o)\bigr)
$$

без генерации полных форм во время инференса.

Предварительный вывод после поиска литературы: именно такая комбинация

1. observation-fiber semantics;
2. необходимых и возможных множеств в непрерывном пространстве захватов;
3. прямого equivariant implicit-предсказания этих множеств;
4. максимальности гарантируемого множества и границы наблюдательной идентифицируемости;
5. теста с парами геометрий, неразличимых для камеры,

не обнаружена в работах по robotic grasping. Это не доказательство абсолютного отсутствия статьи, поэтому novelty claim остаётся проверяемой научной гипотезой. Если убрать пункты 2, 4 или 5, работа становится слишком похожей на uncertainty-through-completion и теряет уровень ICLR.

## 1. Что подтверждает исходную гипотезу

Да, гипотеза о существенном падении качества при внешней окклюзии подтверждается прямыми результатами.

- [TARGO / Target-driven grasping under occlusions](https://arxiv.org/abs/2407.06168) вводит систематический benchmark целевого захвата при окклюзии. На [странице проекта](https://targo-benchmark.github.io/) показано, что обычные методы заметно деградируют с ростом occlusion rate; авторская модель снижает падение, но сама использует completed target shape.
- [TARGO-Net, финальная версия IJCV 2026](https://doi.org/10.1007/s11263-025-02716-9) объединяет сегментацию, AdaPoinTr completion и target-scene fusion. Поэтому «лучше достроить форму и затем предсказать grasp» уже является ближайшим сильным baseline, а не новым направлением.
- [Local Occupancy-Enhanced Grasping, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09354.pdf) показывает большой разрыв между partial и complete geometry и улучшает grasp detection локальной occupancy-реконструкцией. Значит, просто заменить глобальный completion локальным тоже недостаточно ново.
- [Generalizing 6-DoF Grasp Detection, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Generalizing_6-DoF_Grasp_Detection_via_Domain_Prior_Knowledge_CVPR_2024_paper.pdf) прямо отмечает невозможность надёжно вывести невидимые части новых объектов из одного partial point cloud и использует multi-view TSDF как способ получить дополнительную геометрию.

Практический вывод: постановка реальна, эффект окклюзии велик, но completion-направление уже плотно занято. Нужен иной объект предсказания.

## 2. Карта ближайшей литературы и занятые направления

### 2.1. Неопределённая реконструкция с последующей робастной оценкой

- [Robust Grasp Planning over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645): MC-dropout создаёт несколько полных форм, после чего один grasp оценивается на выборке форм.
- [PSSNet: Planar Shape Sampling Network](https://proceedings.mlr.press/v155/saund21a.html): несколько правдоподобных completions помогают именно в неоднозначных окклюдированных сценах.
- [Measuring Uncertainty in Shape Completion to Improve Grasp Quality, IROS 2025](https://arxiv.org/abs/2504.16183): uncertainty из множества completions штрафует grasp score и улучшает результат, но требует десятков проходов completion-модели.
- [UNCLE-Grasp, 2026](https://arxiv.org/abs/2601.14492): несколько MC-dropout completions, force-closure, conservative lower-confidence selection и возможность отказа.
- [Robotic Pick-and-Place With Uncertain Instance Segmentation and Shape Completion](https://pmc.ncbi.nlm.nih.gov/articles/PMC8022832/): сравнивает sampling-based uncertainty с прямым предсказанием стоимости ошибки.

Следствие: posterior sampling, variance penalty, CVaR, LCB и «наихудший grasp по K completions» сами по себе не являются новым вкладом.

### 2.2. Прямые генераторы и implicit-поля захватов

- [Contact-GraspNet](https://arxiv.org/abs/2103.14127) напрямую генерирует 6-DoF parallel-jaw grasps из depth/point cloud.
- [6-DoF GraspNet](https://openaccess.thecvf.com/content_ICCV_2019/papers/Mousavian_6-DOF_GraspNet_Variational_Grasp_Generation_for_Object_Manipulation_ICCV_2019_paper.pdf) использует вариационный генератор grasps из partial point clouds.
- [GraspLDM](https://arxiv.org/abs/2312.11243) моделирует распределение успешных SE(3)-захватов latent diffusion-моделью; среди причин ошибок остаются столкновения с невидимой частью объекта.
- [Implicit Grasp Diffusion](https://proceedings.mlr.press/v270/song25b.html) совмещает локальные implicit features и conditional diffusion для мультимодальной генерации grasps.
- [ShellGrasp](https://arxiv.org/abs/2109.06837) предсказывает camera-centric вход/выход через оболочку объекта и deterministic grasp map.
- [GraspGen-X, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Han_GraspGen-X_Cross-Embodiment_6-DOF_Diffusion-based_Grasping_CVPR_2026_paper.html) развивает diffusion grasp generation для разных grippers.

Следствие: «новый implicit grasp field», «diffusion grasps из partial cloud» или «contact-ray representation» недостаточны. Новизна FiberGrasp должна быть в **семантике поля и объекте обучения**, а не в том, что используется neural field.

### 2.3. Другие близкие направления

- [SpringGrasp, RSS 2024](https://arxiv.org/abs/2404.13532) строит дифференцируемую физическую метрику для compliant dexterous grasping из single-view данных, но решает другую задачу: многопальцевый захват и динамику пружинной податливости.
- [Task-Informed Grasping of Partially Observed Objects, RA-L 2024](https://pure-oai.bham.ac.uk/ws/portalfiles/portal/239318079/deFariasC2024Task-informed.pdf) комбинирует GPIS-реконструкцию и перенос функциональных областей, то есть остаётся reconstruction-first.
- [Grasp configuration planning for an underactuated three-fingered hand, 2018](https://doi.org/10.1016/j.mechmachtheory.2018.06.019) уже использует выражение rough-set mixed neural networks. Там rough sets предобрабатывают human-experience/taxonomy attributes формы и размера для выбора одной из шести конфигураций underactuated hand. Работа не строит lower/upper feasible sets, observation equivalence по RGB-D или continuous 6-DoF action field. Следовательно, нельзя заявлять «первое применение rough sets к grasping»; допустим только более узкий claim об observation-induced rough action sets.
- [Sample-Efficient Safety Assurances using Conformal Prediction](https://arxiv.org/abs/2109.14082) уже применяет conformal prediction к предупреждению ошибок grasping. Поэтому conformal calibration может быть только вспомогательным слоем, но не главным claim.

## 3. Отброшенные кандидаты

| Кандидат | Почему привлекателен | Почему отклонён |
|---|---|---|
| Probabilistic full shape + CVaR/LCB | Явно учитывает скрытую форму | Занято Lundell, PSSNet, IROS 2025 и UNCLE-Grasp; дорого во время инференса |
| Local occupancy около каждого gripper | Не нужен global SDF | Слишком близко к Local Occupancy ECCV 2024 и ShellGrasp |
| Direct success probability $p(y=1\mid o,g)$ | Просто и быстро | Усредняет observation-equivalent миры согласно частоте training prior; не отделяет unknowable от unlikely |
| Quantile/conformal grasp score | Даёт статистическую осторожность | LCB и conformal safety для grasping уже существуют; coverage не равна физической необходимости |
| Diffusion over robust grasps | Мультимодальность | Без новой target semantics это очередной conditional grasp generator |
| Moment/SOS-робастность скрытой массы и поверхности | Красивые гарантии | Слишком много жёстких предположений и переменных; плохо согласуется с noisy PCD и требованием эффективного обучения |
| Full-scene neural SDF + planning | Универсальная геометрия | Именно тот тяжёлый reconstruction/planning pipeline, который исключён постановкой |
| Causal taxonomy failures | Может объяснять ошибки | Явно исключена пользователем и не решает выбор grasp |

## 4. Формальная постановка FiberGrasp

### 4.1. Состояние, наблюдение и действие

Пусть $x\in\mathcal X_c$ — полное локальное физическое состояние: поверхность целевого объекта, препятствие, полка и параметры, необходимые для локальной проверки закрытия gripper. Индекс $c$ — известный класс; если класс неизвестен, берётся объединение допустимых supports.

Камера реализует оператор наблюдения

$$
\mathcal R:\mathcal X\rightarrow\mathcal O.
$$

Наблюдение $o$ содержит только доступные из wrist RGB-D данные: sparse target/obstacle points, known-free camera rays и uncertainty/noise metadata. Полная форма не создаётся на инференсе.

Для noiseless-случая все $x$, имеющие одно наблюдение, образуют equivalence class. Для реального сенсора используется tolerance fiber

$$
\mathcal F_\varepsilon(o,c)
=
\left\{
x\in\mathcal X_c:
d_{\rm obs}\bigl(\mathcal R(x),o\bigr)\leq\varepsilon
\right\}.
$$

Это множество всех сцен из выбранного support, которые камера не умеет различить с учётом шума и окклюзии.

Grasp:

$$
g\in\mathcal G
=
\bigl(SE(3)/C_2\bigr)\times[w_{\min},w_{\max}],
$$

где фактор $C_2$ учитывает 180-градусную симметрию parallel-jaw gripper.

### 4.2. Только локальная физическая допустимость

Вводится signed margin

$$
m(x,g)\in\mathbb R,
\qquad
m(x,g)\geq0
\iff
g\text{ локально допустим в }x.
$$

Margin агрегирует только:

- clearance пальцев и ладони при локальном открытии/закрытии;
- существование двух совместимых contact regions;
- antipodal/friction-cone margin;
- соответствие ширине gripper.

Arm reachability, полный approach trajectory, whole-cycle motion planning и большой lift trajectory намеренно не входят в определение. Малый подъём используется только как конечный экспериментальный критерий успешности.

Для полного состояния:

$$
\mathcal A(x)=\{g:m(x,g)\geq0\}.
$$

### 4.3. Необходимое и возможное множества захватов

Два экстремальных поля:

$$
m_-(o,g)
=
\inf_{x\in\mathcal F_\varepsilon(o,c)}m(x,g),
$$

$$
m_+(o,g)
=
\sup_{x\in\mathcal F_\varepsilon(o,c)}m(x,g).
$$

Они порождают

$$
\mathcal G_-(o)
=
\{g:m_-(o,g)\geq0\}
=
\bigcap_{x\in\mathcal F_\varepsilon(o,c)}\mathcal A(x),
$$

$$
\mathcal G_+(o)
=
\{g:m_+(o,g)\geq0\}
=
\bigcup_{x\in\mathcal F_\varepsilon(o,c)}\mathcal A(x).
$$

Интерпретация:

- $\mathcal G_-$: grasp допустим независимо от того, какая из неразличимых скрытых геометрий истинна;
- $\mathcal G_+$: grasp допустим хотя бы в одном совместимом мире;
- $\mathcal G_+\setminus\mathcal G_-$: область решений, которую single-view RGB-D принципиально не может разрешить без более сильного prior или нового observation.

Рабочая политика:

$$
g^\star
=
\arg\max_{g\in\mathcal G}m_-(o,g).
$$

Если максимум положителен, получен necessary grasp. Если он отрицателен, тот же $g^\star$ является maximin fallback — наименее хрупким решением в принятой модели support. Отказ можно измерять как диагностический режим, но он не является главным результатом.

## 5. Чем это отличается от вероятности успеха

Обычный score

$$
p(g\text{ succeeds}\mid o)
=
\int\mathbf 1[m(x,g)\geq0]p(x\mid o)\,dx
$$

зависит от того, как часто скрытые формы встречались в training distribution. Редкая, но observation-compatible геометрия может почти не повлиять на probability, хотя именно на ней grasp сталкивается с невидимой поверхностью.

FiberGrasp использует support:

$$
\mathcal F_\varepsilon(o,c)
=
\mathrm{supp}p(x\mid o,c)
\quad\text{с сенсорным tolerance}.
$$

Поэтому при изменении частот форм, но неизменном support, $\mathcal G_-$ и $\mathcal G_+$ не меняются. Это не «лучше откалиброванная вероятность», а другой объект предсказания.

Ограничение честно: гарантия верна только относительно заданного support и $\varepsilon$. Неизвестную форму вне support никакой single-view метод не может магически исключить.

## 6. Теоретический пакет

### Proposition 1. Наблюдательная невозможность

Пусть $\mathcal R(x_1)=\mathcal R(x_2)=o$ и

$$
\mathcal A(x_1)\cap\mathcal A(x_2)=\varnothing.
$$

Тогда любая детерминированная observation-only policy $g=\pi(o)$ терпит неудачу хотя бы в одном из $x_1,x_2$.

**Proof sketch.** Policy получает одинаковый input и обязана вернуть один и тот же grasp. Этот grasp не может принадлежать двум непересекающимся feasible sets.

Это формализует не просто sensor uncertainty, а предел идентифицируемости задачи.

### Theorem 1. Максимальность certifiable action set

$\mathcal G_-(o)$ является максимальным множеством, каждый элемент которого можно гарантировать локально допустимым, зная только $o$, support $\mathcal X_c$ и tolerance $\varepsilon$.

Если $g\notin\mathcal G_-(o)$, существует $x'\in\mathcal F_\varepsilon(o,c)$, для которого $m(x',g)<0$. Следовательно, ни один метод с тем же observation и теми же assumptions не может soundly сертифицировать такой $g$.

Аналогично, если $g\notin\mathcal G_+(o)$, он невозможен во всех observation-compatible мирах.

### Theorem 2. Оценка ошибки конечного fiber oracle

Пусть $\{x_j\}_{j=1}^{K}$ — $\delta$-net множества $\mathcal F_\varepsilon(o,c)$ в метрике $d_\mathcal X$, а $m(\cdot,g)$ является $L_x$-Lipschitz. Тогда

$$
0\leq
\min_j m(x_j,g)-m_-(o,g)
\leq L_x\delta,
$$

$$
0\leq
m_+(o,g)-\max_j m(x_j,g)
\leq L_x\delta.
$$

Если ошибка neural approximation empirical extrema не выше $\eta$, sound margins:

$$
\widehat m_-^{\,\rm cert}
=
\widehat m_- - (L_x\delta+\eta),
\qquad
\widehat m_+^{\,\rm cert}
=
\widehat m_+ + (L_x\delta+\eta).
$$

Условие $\widehat m_-^{\,\rm cert}\geq0$ достаточно для робастной локальной допустимости внутри model class.

Практический caveat: глобальный $L_x$ трудно оценить. В первой работе нужны либо контролируемая simulator metric и spectral/Lipschitz bounds, либо эмпирическая held-out correction. Conformal widening допустим как инструмент проверки, но не как novelty claim.

### Proposition 2. Инвариантность к frequency shift

Пусть две условные меры $P_1(x\mid o)$ и $P_2(x\mid o)$ имеют одинаковый support $\mathcal F_\varepsilon(o,c)$. Тогда их необходимые и возможные множества совпадают, хотя posterior success probabilities могут различаться.

Это даёт отдельный falsifiable experiment: reweight частоты форм, не меняя support.

### Proposition 3. Дискретизация grasp manifold

Если $m_-(o,\cdot)$ является $L_g$-Lipschitz, а query set — $h$-cover пространства $\mathcal G$, лучший дискретный maximin grasp отстаёт от непрерывного optimum не более чем на $L_gh$. Это обосновывает Sobol queries с последующим manifold gradient refinement.

## 7. Модель: Equivariant Fiber Operator

### 7.1. Вход

Один sparse input:

- partial target point tokens;
- obstacle/shelf point tokens;
- free-space/occlusion-shadow tags, выводимые из camera rays;
- optional class token;
- gripper geometry constants.

Нет dense full-scene SDF, mesh completion, object-centric canonical reconstruction и отдельного планировщика.

### 7.2. Запрос

Для каждого candidate $g$ локальные tokens переводятся в gripper frame. SE(3)-equivariant point encoder создаёт scene features, а query decoder предсказывает два ordered scalar fields:

$$
\widehat m_-(o,g)=a_\theta(o,g),
$$

$$
\widehat m_+(o,g)
=
a_\theta(o,g)+\mathrm{softplus}b_\theta(o,g).
$$

Так архитектурно гарантируется

$$
\widehat m_-(o,g)\leq\widehat m_+(o,g).
$$

Нужна одна shared backbone и один двухголовый decoder. Дополнительный diffusion-generator не требуется.

### 7.3. Выбор grasp

1. Сэмплировать 1–4 тысячи Sobol poses/widths в допустимом workspace.
2. Одним batched forward pass получить $\widehat m_-$.
3. Взять top-$k$.
4. Выполнить несколько шагов Riemannian gradient ascent по $SE(3)/C_2$ и width.
5. Исполнить максимум certified/adjusted lower field.

Это direct inference; полные формы во время работы робота не генерируются.

## 8. Как строится supervision без unstable reconstruction pipeline

Полные meshes используются **только офлайн для построения labels**, как ground truth в simulator, а не как промежуточный prediction target.

### 8.1. Observation groups

Для одной RGB-D проекции собирается fiber bank:

- разные полные объекты, чьи rendered visible surfaces совпадают в tolerance;
- hidden-part morphs, меняющие геометрию только внутри occlusion shadow;
- class-preserving shapes при known class;
- mixture support при unknown class;
- sensor perturbations и segmentation noise.

### 8.2. Hard witnesses

Для каждого $(o,g)$ ищутся witness shapes:

$$
z^-
=
\arg\min_z m(S_\phi(z),g)
$$

и

$$
z^+
=
\arg\max_z m(S_\phi(z),g)
$$

при ограничениях

$$
d_{\rm obs}\bigl(\mathcal R(S_\phi(z)),o\bigr)\leq\varepsilon,
\qquad
S_\phi(z)\in\mathcal X_c.
$$

Используются multiple starts, augmented Lagrangian, adversarial refinement и replay bank найденных witnesses. Shape model здесь служит офлайн генератором контрпримеров; online model никогда не выдаёт completed shape.

### 8.3. Loss

Для margins $M_j=m(x_j,g)$:

$$
\mathcal L_{\rm contain}
=
\frac1K\sum_j
\left[
\mathrm{ReLU}(\widehat m_- - M_j)^2
+
\mathrm{ReLU}(M_j-\widehat m_+)^2
\right],
$$

$$
\mathcal L_{\rm tight}
=
\left|\widehat m_--\min_jM_j\right|
+
\left|\widehat m_+-\max_jM_j\right|,
$$

$$
\mathcal L
=
\mathcal L_{\rm contain}
+\lambda_{\rm tight}\mathcal L_{\rm tight}
+\lambda_{\rm gap}(\widehat m_+-\widehat m_-)
+\lambda_{\rm eq}\mathcal L_{\rm equivariance}
+\lambda_{\rm Lip}\mathcal L_{\rm Lipschitz}.
$$

Containment отвечает за sound interval, extrema regression — за tightness, gap penalty не позволяет выдавать тривиальный бесконечно широкий interval.

### 8.4. Iterative oracle-model loop

1. Обучить на random fiber bank.
2. Найти grasps, где model optimistic.
3. Для них решить constrained hidden-shape minimization.
4. Добавить найденные failure witnesses.
5. Повторять до насыщения held-out violation rate.

Это аналог separation oracle: model учится на тех неразличимых геометриях, которые опровергают её текущую уверенность.

## 9. Почему это должно быть эффективно обучаемо

### Инференс

Uncertain-completion pipeline имеет приблизительную стоимость

$$
O(KC_{\rm completion}+KQ C_{\rm physics}),
$$

где $K$ — число completions, $Q$ — candidates.

FiberGrasp после scene encoding имеет

$$
O(C_{\rm encoder}+Q C_{\rm decoder}),
$$

без множителя $K$ и без online mesh processing.

Заявлять конкретные миллисекунды заранее нельзя. Paper-level target: batched inference вместе с refinement менее 100 ms на лабораторном GPU и выигрыш не менее $5\times$ против 30–60 completion samples.

### Обучение

Задача сводится к supervised interval regression по offline physical margins. Нет reinforcement learning, rollout credit assignment или VLA pretraining. Hard oracle дорог, но embarrassingly parallel и используется только при создании/reﬁnement dataset.

## 10. Что в идее пришло из общей математики/ML, а не из сборки robotics pipeline

- Rough-set theory задаёт lower/upper approximations по классам неразличимости: [вводный источник по rough sets](https://people.eecs.ku.edu/~jerzygb/Rough-sets.pdf).
- Для noisy observation вместо строгой equivalence используется tolerance relation; см. [rough approximations through general binary relations](https://arxiv.org/abs/1811.09609).
- Random-set view рассматривает неизвестное feasible set как set-valued random object; adversarial sampling уже полезен для ускорения нелинейной reachability: [Neural Bridge Sampling for evaluating safety-critical autonomous systems](https://proceedings.mlr.press/v155/lew21a.html).
- Equivariance для stochastic fields даёт правильный inductive bias: [Equivariant Conditional Neural Processes](https://proceedings.mlr.press/v139/holderrieth21a.html).
- Идея предсказывать достаточный для решения proxy вместо всей неопределённой величины согласуется с [Sufficient Decision Proxies for Decision-Focused Learning](https://arxiv.org/abs/2505.03953).

Robotics определяет только физический margin и эксперимент. Главный объект — learned set-valued decision operator under partial observability — является общей ML-постановкой.

## 11. Сравнение с ближайшими работами

| Работа | Что моделирует | Online completion | Семантика uncertainty | Что отсутствует относительно FiberGrasp |
|---|---|---:|---|---|
| TARGO-Net | completed target + grasp field | Да | одна reconstructed shape | observation fiber, necessary/possible sets, identifiability theorem |
| Lundell 2019 | grasp score на MC completions | Да | sample average | support extrema как direct field; maximal set |
| PSSNet | diverse plausible shapes | Да | несколько modes | action-space lower/upper operator |
| IROS 2025 uncertainty | variance penalty | Да, около 60 samples | heuristic point uncertainty | set semantics и soundness |
| UNCLE-Grasp | force closure на MC shapes + LCB | Да | conservative confidence | generic direct field, support invariance, upper set, observation-pair theorem/benchmark |
| Local Occupancy ECCV | local hidden occupancy | Да, локально | point occupancy | no-reconstruction action-set target |
| Contact-GraspNet / GraspLDM / IGD | distribution/score of grasps | Нет | learned training frequency | necessary vs possible under indistinguishable worlds |
| FiberGrasp | lower/upper margins on grasp manifold | Нет | support of observation fiber | предлагаемый вклад |

Самый опасный reviewer objection: «FiberGrasp — amortized UNCLE-Grasp». Ответ будет убедителен только при наличии одновременно:

- upper set, а не только conservative score;
- theorem of maximal certifiable set;
- frequency-shift experiment;
- indistinguishable-pair benchmark;
- отсутствие sampled completions на инференсе;
- generic parallel-jaw shelf setup, а не один узкий вид объекта.

Без этого objection справедлив.

## 12. Экспериментальная программа

### 12.1. Данные

1. TARGO balanced occlusion bins как внешний benchmark.
2. Отдельный TARGO subset с ровно одним фронтальным obstacle.
3. Новый controlled Shelf-Fiber benchmark:
   - wrist RGB-D;
   - target на полке;
   - один передний obstacle;
   - occlusion ratio 0–90%;
   - realistic depth dropout, quantization и pose noise;
   - class-known и class-unknown protocols;
   - полная simulator geometry доступна только для labels/evaluation.
4. Real robot: humanoid arm + parallel-jaw gripper; успех — стабильный grasp и малый подъём.

### 12.2. Baselines

- Contact-GraspNet или сильный direct grasp detector;
- GraspLDM / Implicit Grasp Diffusion;
- TARGO-Net;
- Local Occupancy-Enhanced Grasping;
- deterministic completion + grasp;
- Lundell-style MC completion scoring;
- PSSNet-style diverse completion;
- IROS 2025 uncertainty penalty;
- UNCLE-style lower confidence selection, адаптированный к parallel jaw;
- full-geometry oracle как верхняя граница.

### 12.3. Основные метрики

- grasp success rate по bins окклюзии;
- worst-bin success;
- падение easy $\rightarrow$ severe occlusion;
- regret к full-geometry oracle;
- precision/soundness necessary set;
- fraction сцен с непустым $\mathcal G_-$;
- violation rate: доля predicted-necessary grasps, опровергнутых held-out compatible shape;
- tightness $\widehat m_+-\widehat m_-$;
- success-coverage curve maximin/certified policy;
- latency, memory и число geometry-model calls.

### 12.4. Killer experiment

Создать пары $(x_1,x_2)$, которые:

$$
d_{\rm obs}\bigl(\mathcal R(x_1),\mathcal R(x_2)\bigr)\leq\varepsilon,
$$

но имеют разные hidden collisions или contact feasibility.

Для каждой пары проверить:

1. posterior-mean detector выбирает высокочастотный, но не necessary grasp;
2. FiberGrasp исключает его из $\mathcal G_-$;
3. shared feasible grasp, если он существует, остаётся в lower set;
4. при пустом intersection model показывает отрицательный maximin margin, а не искусственную уверенность.

Это наиболее чистое доказательство того, что работа изучает partial identifiability, а не просто data augmentation.

### 12.5. Distribution stress

- менять частоты shapes при фиксированном support;
- rare hidden protrusions;
- unseen instances и новые классы;
- неверный и отсутствующий class token;
- mismatch depth noise;
- segmentation boundary corruption;
- obstacle pose errors;
- расширение/сужение $\varepsilon$.

### 12.6. Критические ablations

- posterior mean vs lower field;
- lower only vs lower+upper joint learning;
- random fibers vs hard witnesses;
- без ray/free-space tokens;
- без equivariance;
- class known vs unknown;
- разное $K$ в offline fiber bank;
- без $\delta,\eta$ correction;
- Sobol only vs gradient refinement;
- true analytic margins vs learned outcome labels.

## 13. Проверяемая гипотеза о превосходстве над SOTA

Гипотеза:

> При severe single-view occlusion и fixed inference budget direct prediction of the necessary grasp set уменьшит catastrophic hidden-geometry failures сильнее, чем point-estimate completion, posterior-mean grasp scoring и finite-sample uncertainty penalties.

Основания:

- TARGO показывает систематическое падение с окклюзией;
- diverse/uncertain completions уже улучшают grasping, то есть ambiguity действительно полезна;
- IROS 2025 сообщает прирост от uncertainty penalty;
- Local Occupancy показывает, что невидимая локальная геометрия объясняет заметную часть разрыва;
- прямой action-space proxy убирает online completion cost и training-frequency averaging.

Но это **не гарантия SOTA**. Strict support worst-case может оказаться слишком консервативным. Поэтому работа обязана показать Pareto frontier «success vs coverage», а не скрывать пустые lower sets.

Минимальный go/no-go результат:

- не менее +5 percentage points success в severe occlusion против лучшего matched-latency baseline;
- не менее 2× снижение held-out fiber violation;
- не более 20% относительного падения coverage;
- минимум 5× ускорение против MC completion uncertainty;
- преимущество сохраняется при frequency shift.

Если эти условия не выполнены, claims уровня ICLR не подтверждены.

## 14. ICLR-аудит

[ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide) оценивает ясность проблемы, связь с литературой, поддержку claims, значимость нового знания, техническую корректность и воспроизводимость. [Call for Papers](https://iclr.cc/Conferences/2026/CallForPapers) явно включает robotics, uncertainty, structured prediction и representation learning.

### Specific problem

Да: выбор 6-DoF parallel-jaw grasp из одного noisy occluded RGB-D observation, когда разные физические сцены дают практически одинаковый input.

### Новый ML-вопрос

Как амортизированно предсказывать нижнее и верхнее множества непрерывных действий по observation-induced equivalence/tolerance classes, не реконструируя скрытое состояние?

### Claims, которые реально можно защищать

1. Observation-only grasp certification ограничена intersection feasible sets.
2. Это intersection — максимальный certifiable set при данных assumptions.
3. Fiber extrema допускают bounded approximation через $\delta$-net.
4. Support-based sets инвариантны к reweighting frequency.
5. Equivariant implicit operator оценивает их быстрее online sampling.
6. На severe occlusion такой target уменьшает hidden-geometry failures.

Пункты 1–4 теоретические; 5–6 требуют эксперимента.

### Почему это потенциально ICLR, а не только robotics paper

- новая set-valued learning target;
- continuous rough approximations на transformation group/action manifold;
- identifiability perspective;
- adversarial fiber supervision;
- generalization к другим partially observed decision problems: medical action selection, collision avoidance, manipulation and control.

### Оценка новизны

- **8/10**, если реализованы обе границы, теоремы, hard-witness oracle, paired benchmark и frequency-shift result;
- **5/10**, если реализован только neural lower-confidence score;
- **3/10**, если метод фактически делает K completions и min/variance aggregation.

### Главные риски

1. **Support misspecification.** Гарантия условна. Нужно публиковать violation under support expansion.
2. **Пустой lower set.** Нужны coverage curve и maximin fallback.
3. **Triviality theorem.** Intersection identity проста; научная ценность должна идти от learnable operator, approximation bounds и benchmark.
4. **Oracle bias.** Hard witnesses могут не покрыть fiber; нужны held-out generators и $\delta$-net/empirical corrections.
5. **Amortized robust optimization objection.** Отбивается только полным набором отличий из раздела 11.
6. **Metric ambiguity.** $d_{\rm obs}$, shape support и physical margin должны быть определены до экспериментов, а не подогнаны после.
7. **Overconservatism.** Нужно сравнить support extrema с $\alpha$-trimmed fibers, но robust version оставить основной.

## 15. Фальсификация новизны до дорогого обучения

Работу следует остановить или радикально изменить, если поиск обнаружит статью, которая одновременно:

- определяет observation-equivalence/tolerance fiber;
- выводит intersection/union feasible grasps;
- напрямую учит обе границы в continuous 6-DoF grasp space;
- доказывает maximal certifiability;
- не требует completion samples на инференсе.

Поиск должен продолжаться по терминам:

- necessary and possible grasp sets;
- lower/upper approximation robotic grasp;
- rough set grasp planning;
- observation fiber action set;
- set-valued grasp prediction partial observation;
- robust feasible action set point cloud;
- indistinguishable shapes grasping.

На момент этой записи прямого совпадения не найдено. Найдены Random Set Neural Networks для других задач и историческая rough-set mixed neural network для taxonomy-based выбора конфигурации трёхпальцевой кисти. Поэтому слово rough set само по себе заведомо не является новым. Не найден именно continuous lower/upper action-set operator, индуцированный классами неразличимых RGB-D наблюдений.

## 16. Минимальная реализация, проверяющая сущность идеи

До большой модели достаточно 2D/2.5D прототипа:

1. Сгенерировать силуэты объектов с одинаковой видимой фронтальной частью и разными скрытыми protrusions.
2. Перечислить planar parallel-jaw grasps.
3. Аналитически вычислить $m_-$ и $m_+$.
4. Обучить небольшой equivariant query network.
5. Сравнить с mean-shape, success probability и min по малому $K$.
6. Выполнить paired indistinguishability и frequency-reweighting tests.

Go-критерий: network сохраняет necessary-set precision при смене частот и находит shared grasp, когда mean-probability baseline выбирает несовместимый с одной из скрытых форм.

Только после этого оправдан переход к 3D mesh oracle и real robot.

## 17. Рекомендуемый paper claim

Наиболее сильная и честная формулировка:

> We introduce observation-fiber grasping: a set-valued learning problem that maps a partial RGB-D observation to the maximal necessary and possible subsets of a continuous grasp manifold. We derive identifiability and finite-fiber approximation results, propose an equivariant implicit fiber operator trained with adversarially discovered hidden-shape witnesses, and show that direct support-aware action-set inference reduces severe-occlusion failures without online shape completion.

Не следует заявлять:

- «первая uncertainty-aware grasping model»;
- «первая robust grasping under occlusion»;
- «гарантия для любой реальной формы»;
- «решает весь manipulation cycle»;
- «доказывает SOTA» до эксперимента.

## 18. Финальный вердикт

**Рекомендация: развивать FiberGrasp, но только в сильной set-theoretic версии.**

Это критически более новое направление, чем очередной completion, fusion module или diffusion grasp generator, потому что меняет сам prediction target: от скрытой формы или средней вероятности к границам того, что наблюдение позволяет и не позволяет утверждать о действиях.

Суть работы можно потерять очень легко. Если реализация станет «K hidden shapes $\rightarrow$ min score», это будет инженерное ускорение существующей линии. Если же будет direct lower/upper operator, maximality result, hard counterexample supervision и paired indistinguishability benchmark, возникает самостоятельный ML-вклад с разумной ICLR-мотивацией.

## Источники, использованные в этой итерации

- TARGO: https://arxiv.org/abs/2407.06168
- TARGO project: https://targo-benchmark.github.io/
- TARGO IJCV 2026: https://doi.org/10.1007/s11263-025-02716-9
- Local Occupancy-Enhanced Grasping: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09354.pdf
- Lundell et al., uncertain completions: https://arxiv.org/abs/1903.00645
- PSSNet: https://proceedings.mlr.press/v155/saund21a.html
- IROS 2025 shape-completion uncertainty: https://arxiv.org/abs/2504.16183
- UNCLE-Grasp: https://arxiv.org/abs/2601.14492
- Pick-and-place with segmentation and completion uncertainty: https://pmc.ncbi.nlm.nih.gov/articles/PMC8022832/
- Contact-GraspNet: https://arxiv.org/abs/2103.14127
- 6-DoF GraspNet: https://openaccess.thecvf.com/content_ICCV_2019/papers/Mousavian_6-DoF_GraspNet_Variational_Grasp_Generation_for_Object_Manipulation_ICCV_2019_paper.pdf
- GraspLDM: https://arxiv.org/abs/2312.11243
- Implicit Grasp Diffusion: https://proceedings.mlr.press/v270/song25b.html
- ShellGrasp: https://arxiv.org/abs/2109.06837
- GraspGen-X: https://openaccess.thecvf.com/content/CVPR2026/html/Han_GraspGen-X_Cross-Embodiment_6-DOF_Diffusion-based_Grasping_CVPR_2026_paper.html
- SpringGrasp: https://arxiv.org/abs/2404.13532
- Task-Informed Grasping: https://pure-oai.bham.ac.uk/ws/portalfiles/portal/239318079/deFariasC2024Task-informed.pdf
- Rough-set mixed grasp configuration planning (2018): https://doi.org/10.1016/j.mechmachtheory.2018.06.019
- Conformal safety for grasping: https://arxiv.org/abs/2109.14082
- Generalizing 6-DoF Grasp Detection: https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Generalizing_6-DoF_Grasp_Detection_via_Domain_Prior_Knowledge_CVPR_2024_paper.pdf
- Rough sets: https://people.eecs.ku.edu/~jerzygb/Rough-sets.pdf
- Rough approximations with general relations: https://arxiv.org/abs/1811.09609
- Neural Bridge Sampling / random-set reachability: https://proceedings.mlr.press/v155/lew21a.html
- Equivariant Conditional Neural Processes: https://proceedings.mlr.press/v139/holderrieth21a.html
- Sufficient Decision Proxies: https://arxiv.org/abs/2505.03953
- ICLR 2026 Reviewer Guide: https://iclr.cc/Conferences/2026/ReviewerGuide
- ICLR 2026 Call for Papers: https://iclr.cc/Conferences/2026/CallForPapers
