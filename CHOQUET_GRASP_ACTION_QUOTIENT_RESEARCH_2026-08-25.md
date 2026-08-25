# CHOQUET-GRASP: обучение action quotient через ёмкость случайного множества для захвата из частичного RGB-D

Дата исследования: 2026-08-25  
Статус: исследовательская концепция, а не заявленный экспериментальный результат

## Краткий вердикт

Наиболее сильная из рассмотренных идей — **не восстанавливать скрытую форму и не предсказывать независимый score каждого захвата, а учить условный закон случайного замкнутого множества опасных/допустимых действий в пространстве захватов**. Рабочее название общего подхода — **Action-Quotient Capacity Learning (AQCL)**, а специализации для параллельного захвата — **CHOQUET-GRASP**.

Скрытая геометрия нужна роботу не сама по себе. Для данной задачи две полные формы эквивалентны, если они разрешают и запрещают одни и те же терминальные захваты параллельным захватом. Поэтому предлагается факторизовать пространство форм по этому отношению эквивалентности и учить не форму, а её **action quotient** — случайное множество в пространстве действий, индуцированное неоднозначным RGB-D-наблюдением.

Ключевая обучающая величина — **условный hitting functional**, или ёмкость:

\[
T_{U\mid x}(K)=\Pr\{U\cap K\neq\varnothing\mid X=x\},
\]

где \(U\) — множество опасных терминальных захватов, а \(K\) — компактная область действий: один захват, окрестность позы, эллипсоид ошибки исполнения либо объединение нескольких альтернатив. Для команды \(g\) с ограниченной ошибкой исполнения \(K_g\), \(T_{U\mid x}(K_g)\) имеет прямой смысл: это вероятность того, что хотя бы одна реализуемая поза внутри допуска команды окажется опасной из-за скрытой формы.

Это даёт новую, проверяемую цель обучения — **Conditional Capacity Matching**:

\[
\mathcal L_{\mathrm{CCM}}=
\mathbb E_{(x,S),K}\left[
\operatorname{BCE}\left(
\widehat T_\theta(K\mid x),
\mathbf 1\{U(S)\cap K\neq\varnothing\}
\right)\right].
\]

Полная модель объекта используется только офлайн для вычисления метки попадания. На инференсе нет ни mesh, ни voxel/TSDF/SDF, ни множества восстановленных форм. Модель задаёт непрерывное случайное поле механического margin в пространстве \(SE(3)\times\mathbb R_+\), а общий латентный код сцены делает ответы для разных захватов согласованными. Объединённые probes обучают зависимости между действиями, которые принципиально не идентифицируются pointwise-BCE.

Идея выглядит достойной ICLR только при трёх условиях: (1) теория random-set identifiability является центром статьи, а не украшением; (2) есть отдельный простой general-ML benchmark на скрытые непрерывные ограничения; (3) робототехническая проверка прямо сравнивает метод с full completion, multiple completion, direct amodal и современными uncertainty-моделями. Без этого работа легко будет воспринята как clever perturbation augmentation для grasp scoring и скорее попадёт в CoRL/робототехнический трек.

## 1. Точная постановка и границы

Рассматривается один целевой объект, один передний объект-препятствие и плоскость полки. Камера выдаёт единственное шумное RGB-D-наблюдение. Препятствие закрывает часть цели; распределение возможной скрытой геометрии известно только через тренировочную выборку форм. Нужно выбрать надёжный 6-DoF захват параллельным захватом.

Вклад намеренно **не** охватывает:

- reinforcement learning, VLA и планирование последовательности действий;
- clutter как основную постановку;
- проверку полного цикла reach–approach–close–transport;
- восстановление полного mesh, occupancy, TSDF или SDF сцены;
- гигантский набор латентных переменных всей сцены;
- объяснение через «причинные failure modes»;
- простую замену головы в существующем робототехническом pipeline.

Оракул результата ограничивается терминальной/quasi-static механикой: допустимая ширина, наличие и качество двух контактов, antipodality/force-closure margin, отсутствие столкновения корпуса пальцев в конечной конфигурации и небольшой статический lift/wrench margin. Можно добавить малый подъём на 2 см как проверку удержания, но не оценивать траекторию подхода и достижимость манипулятором. Это сохраняет предмет статьи узким: **скрытая grasp-relevant геометрия и неопределённость действия**, а не вся манипуляция.

## 2. Что уже сделано и где остаётся разрыв

### 2.1. Completion и implicit geometry

[TARGO](https://arxiv.org/abs/2407.06168) — наиболее прямой современный ориентир: single-depth target-driven parallel-jaw grasping при разных уровнях окклюзии. Авторы показывают заметное падение существующих методов при росте окклюзии. Их TARGO-Net использует completion целевого объекта, cross-attention между целью и сценой и плотное поле захватов. Completion заметно помогает в симуляции, но в реальной среде плохо переносится; авторы прямо связывают остаточную деградацию с шумом и сложностью completion. Это одновременно подтверждает ценность скрытой формы и оставляет разрыв: можно ли перенести её полезную для решения информацию, не предсказывая саму форму?

[VGN](https://arxiv.org/abs/2101.01132), [GIGA](https://roboticsproceedings.org/rss17/p024.pdf) и [NeuGraspNet](https://arxiv.org/abs/2306.07392) связывают геометрию с affordance через TSDF либо implicit scene representation. NeuGraspNet особенно силён как аргумент против поверхностной новизны: он уже использует implicit geometry, локальные и глобальные признаки и запросы произвольных захватов. Но всё это по-прежнему учит геометрическое представление скрытого мира и затем оценивает отдельные действия.

Более раннее [shape-completion-enabled grasping](https://arxiv.org/abs/1609.08546) строит полный объект перед планированием. [Robust Grasp Planning over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645) идёт дальше: семплирует несколько voxel completions, строит кандидатов и проверяет их на ансамбле форм. Это важнейший предшественник по смыслу — распределение скрытых форм действительно улучшает надёжность — но вычислительная цена полного геометрического посредника велика, а итоговая статистическая цель не задаёт закон множества действий напрямую. Недавняя работа по [uncertainty в shape completion](https://arxiv.org/abs/2504.16183) также улучшает ранжирование, но сохраняет completion как обязательный промежуточный объект.

Недавний preprint [Object Pose and Shape Estimation for Grasping: Does It Work?](https://arxiv.org/abs/2605.26944) — полезная проверка на чрезмерный тезис: в ряде условий модульная оценка формы и позы с аналитическим планированием превосходит end-to-end grasp detection. Поэтому CHOQUET-GRASP не должен утверждать, что геометрия «не нужна вообще». Проверяемый тезис уже и сильнее: **если utility зависит от скрытого мира только через множество терминально допустимых действий, его условный закон является достаточной decision representation и потенциально требует меньше вычислений/данных, чем восстановление всей формы**.

### 2.2. Прямой amodal grasp prediction

[S4G](https://proceedings.mlr.press/v100/qin20a.html) уже предсказывает amodal 6-DoF grasps из частичного шумного point cloud и способен учитывать невидимые столкновения. Следовательно, формулировка «предсказывать захват напрямую без reconstruction» не нова. [Contact-GraspNet](https://arxiv.org/abs/2103.14127) показывает эффективность contact-rooted параметризации, а [AnyGrasp](https://arxiv.org/abs/2212.08333) — масштабируемого dense obstacle-aware grasp detection. [GraspNet-1Billion](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html) демонстрирует ценность огромной аналитически размеченной выборки.

Однако эти методы в основном учат качество/позу для отдельных кандидатов. Даже robust label через минимум по небольшим возмущениям в S4G остаётся детерминированной локальной целью. Он не идентифицирует коррелированную неоднозначность: какие целые области пространства действий совместно разрешаются или запрещаются разными скрытыми формами.

[Learning to Generate All Feasible Actions](https://arxiv.org/abs/2301.11461) — ближайший general-control antecedent. Работа учит распределение, покрывающее множество допустимых действий, и показывает упрощённый planar grasp. Но состояние там фактически наблюдаемо и задаёт детерминированное feasible set; обучение связано с интерактивным critic/decision process; нет hidden-shape posterior, закона случайного замкнутого множества, hitting functional и риска целой окрестности 6-DoF-команды. Значит, «all feasible actions» само по себе тоже не является новым вкладом; новизна должна находиться именно в **условном random-set law при частичном наблюдении**.

### 2.3. Uncertainty, risk и task-specific geometry

[vMF-Contact](https://arxiv.org/abs/2411.03591) уже моделирует aleatoric/epistemic uncertainty направлений контакта через evidential/von Mises–Fisher распределения и использует auxiliary reconstruction. [FFHFlow](https://openreview.net/pdf?id=uWFlkufjFJ) использует likelihood flow-модели как меру неоднозначности partial-observation dexterous grasps. Поэтому «добавить uncertainty head» или «генерировать несколько захватов» недостаточно.

[FIRMGrasp](https://arxiv.org/abs/2607.25049) и differentiable risk-aware grasping на основе [CVaR](https://arxiv.org/abs/2604.25897) делают простой вариант «оптимизируем CVaR качества» слишком близким к существующему. CVaR может быть downstream rule, но не центральной целью статьи.

[ShellGrasp-Net](https://arxiv.org/abs/2109.06837) предсказывает task-relevant shell — entry/exit depth и grasp map — вместо всей геометрии. Недавний [Task-Oriented Shape Completion](https://arxiv.org/abs/2601.05499) восстанавливает лишь контактно-релевантные области. Это подтверждает полезность task quotient, но одновременно закрывает слабую версию идеи «completion только в местах контактов». CHOQUET-GRASP не выдаёт геометрический shell или contact patch: он выдаёт распределение **множеств действий**, и его supervision — события пересечения областей действий.

## 3. Отвергнутые направления

| Кандидат | Почему казался разумным | Почему отвергнут как главный вклад |
|---|---|---|
| Несколько completions + robust ranking | Учитывает multimodality скрытой формы | Уже исследовано; дорого; сохраняет избыточный geometry bottleneck |
| Прямое amodal grasp field | Нет полного reconstruction | S4G, Contact-GraspNet и последующие системы уже делают прямое предсказание |
| Evidential/vMF/ensemble uncertainty | Просто обучать и калибровать | vMF-Contact и FFHFlow закрывают generic uncertainty; нет set-level зависимостей |
| CVaR или worst-case score | Имеет ясный safety смысл | Уже появляется в современных grasping-работах; это decision rule, не новая learned object |
| Contact-only completion | Сжимает форму до grasp-relevant частей | ShellGrasp и task-oriented completion уже очень близки; всё ещё геометрический output |
| Conformal threshold для захвата | Даёт finite-sample coverage | Сам по себе слишком тонкий вклад и обычно лишь marginal, а не action-region guarantee |
| Причинная классификация отказов | Интерпретируемо | Прямо вне поставленной границы и трудно размечается без расширения сцены |

После этих исключений остаётся содержательно иной объект обучения: **топологически согласованный условный закон random closed set в непрерывном пространстве действий**.

## 4. Общая идея: action quotient вместо hidden world

Пусть \(S\) — полная скрытая форма цели вместе с небольшим набором физически необходимых nuisance-параметров (например, friction/mass class), \(O\) — наблюдаемое переднее препятствие, а \(X\) — единственное RGB-D-наблюдение. Пространство команд

\[
\mathcal G\subset SE(3)\times[w_{\min},w_{\max}]
\]

компактно после задания рабочей области, диапазона ориентаций и ширины пальцев.

Пусть \(m(S,O,g)\) — непрерывный или сглаженный терминальный механический margin. Положительное значение означает запас по контакту/force closure/terminal collision/lift wrench, отрицательное — нарушение. Для допуска исполнения \(E_\delta\subset\mathfrak{se}(3)\) определим robust feasible set

\[
F^\delta_{S,O}=\left\{g\in\mathcal G:
\inf_{\xi\in E_\delta}m(S,O,\exp(\widehat\xi)g)\ge\gamma
\right\},
\]

и unsafe set

\[
U_{S,O}=\{g\in\mathcal G:m(S,O,g)\le 0\}.
\]

При непрерывном \(m\) оба множества замкнуты. Из-за неизвестной скрытой части \(S\mid X=x\) это **условные случайные замкнутые множества**.

Введём отношение эквивалентности форм:

\[
S_1\sim S_2
\quad\Longleftrightarrow\quad
U_{S_1,O}=U_{S_2,O}
\quad
(\text{эквивалентно }F_{S_1,O}=F_{S_2,O}).
\]

Класс \([S]\) — task/action quotient. Он стирает цвет внутренней поверхности, геометрию вдали от контактов и другие детали, не меняющие допустимые действия, но сохраняет скрытую толщину, полости, края и контактные нормали, если они меняют захваты.

### 4.1. Почему pointwise probabilities недостаточны

Рассмотрим два действия \(a,b\). В модели A опасное множество равно \(\{a\}\) или \(\{b\}\), каждое с вероятностью \(1/2\). В модели B оно равно \(\varnothing\) или \(\{a,b\}\), каждое с вероятностью \(1/2\). В обоих случаях

\[
\Pr(a\in U)=\Pr(b\in U)=1/2.
\]

Любая модель, обученная только на singleton-labels, считает эти распределения одинаковыми. Но

\[
\Pr(U\cap\{a,b\}\neq\varnothing)=
\begin{cases}
1,&\text{модель A},\\
1/2,&\text{модель B}.
\end{cases}
\]

То есть вероятность того, что **область исполнения или набор альтернатив содержит опасное действие**, разная. Эта зависимость нужна для bounded execution error, выбора диверсифицированного набора кандидатов и понимания multimodality скрытой формы. Независимые pointwise scores её не идентифицируют.

### 4.2. Почему capacity — правильный статистический объект

Для случайного замкнутого множества \(U\) hitting functional

\[
T_U(K)=\Pr(U\cap K\neq\varnothing)
\]

задаётся на компактных \(K\). Классическая теорема Шоке–Кендалла–Матерона утверждает, что корректный capacity functional однозначно определяет закон random closed set; систематическое изложение даёт монография Молчанова [Theory of Random Sets](https://link.springer.com/book/10.1007/978-1-4471-7349-6). Современное изложение связи закона и capacity также формулирует это как one-to-one correspondence для случайных замкнутых множеств ([пример математической статьи](https://www.sciencedirect.com/science/article/abs/pii/S0165011421003699)).

Здесь это не декоративная аналогия. Компактные \(K\) имеют операционный смысл:

- singleton \(\{g\}\): обычный риск одного идеального захвата;
- малая геодезическая сфера/эллипсоид: bounded pose and width error;
- tube вдоль короткой closing perturbation: чувствительность к установке пальцев;
- объединение удалённых компонент: совместная неопределённость альтернатив;
- coarse action cell: есть ли внутри опасная либо допустимая область.

## 5. Новая цель обучения: Conditional Capacity Matching

### 5.1. Разметка без reconstruction target

На тренировке доступен полный mesh/симулятор только как оракул. Для примера \((x,S,O)\) семплируется компактный probe \(K\subset\mathcal G\), и вычисляется бинарная метка

\[
y_U(S,O,K)=\mathbf 1\{U_{S,O}\cap K\neq\varnothing\}.
\]

Дополнительно можно разметить событие наличия уверенно допустимого действия

\[
y_F(S,O,K)=\mathbf 1\{F^\delta_{S,O}\cap K\neq\varnothing\}.
\]

Это не требует хранить dense volume или ground-truth completion: оракул отвечает на batched action queries. В датасете достаточно сохранить RGB-D, параметры probe и два бита/минимальный margin.

### 5.2. Proper objective

Основная функция потерь:

\[
\mathcal L_{\mathrm{CCM}}(\theta)=
\mathbb E\left[
-y_U\log \widehat T_{U,\theta}(K\mid X)
-(1-y_U)\log(1-\widehat T_{U,\theta}(K\mid X))
\right].
\]

Bernoulli log score строго proper для каждого события попадания. Если распределение probes имеет достаточную поддержку и разделяет компактные множества, population optimum восстанавливает условный capacity на поддержке probes. При плотном/separating семействе и регулярности это идентифицирует условный закон \(U\mid X=x\), а не только его pointwise marginals.

Практический loss:

\[
\mathcal L=
\mathcal L_{\mathrm{CCM}}^U
+\lambda_F\mathcal L_{\mathrm{CCM}}^F
+\lambda_m\mathcal L_{\mathrm{margin}}
+\lambda_c\mathcal L_{\mathrm{cal}}
+\lambda_e\mathcal L_{\mathrm{equiv}}.
\]

Здесь margin regression — вспомогательная локальная задача, calibration term оценивается по occlusion bins, а equivariance regularizer применяет совместное \(SE(3)\)-преобразование наблюдения и probes. Ни Chamfer distance, ни occupancy reconstruction не являются основной или обязательной потерей.

### 5.3. Curriculum probes

Семейство probes должно быть заранее специфицировано, иначе теоретическая идея выродится в jitter augmentation:

1. 30% singletons для совместимости с обычной grasp-quality supervision.
2. 30% малых anisotropic ellipsoids, соответствующих реальной ковариации ошибки позы/ширины.
3. 20% unions из 2–4 локальных компонент, в том числе удалённых, чтобы учить зависимости.
4. 10% hard boundary tubes вокруг текущего \(m\approx0\).
5. 10% coarse cells для иерархического поиска.

Радиусы следует рандомизировать от сенсорного/исполнительного разрешения до нескольких сантиметров и 10–15 градусов. Критический ablation — singleton-only при том же encoder, decoder и числе action evaluations.

## 6. Архитектура CHOQUET-GRASP

### 6.1. Вход и encoder

Вход состоит из двух раздельно помеченных sparse point sets:

- видимые точки целевого объекта из RGB-D;
- видимые точки единственного переднего препятствия и плоскости полки.

К каждой точке добавляются RGB, нормаль/локальная ковариация, направление луча камеры, оценка depth noise и расстояние до silhouette/occlusion boundary. Разделение target/obstacle можно считать заданным постановкой или получить отдельным frozen segmenter; качество сегментации проверяется отдельным noise ablation, но foundation/VLM не является вкладом.

Двухпоточный sparse E(3)-equivariant point transformer кодирует цель и препятствие, затем выполняет ограниченный cross-attention. Нет voxel grid и scene SDF. Encoder вычисляется один раз на сцену.

### 6.2. Общий latent случайного action field

Модель предсказывает условное распределение низкоразмерного scene latent:

\[
z\sim p_\theta(z\mid x),
\]

например, normalizing flow поверх 16–32 измерений. Один и тот же \(z\) используется для **всех** запросов \(g\) в сцене. Это принципиально: независимый шум на каждом query не способен представить согласованную гипотезу скрытой формы и возвращает нас к независимым scores.

Continuous implicit decoder в canonical frame захвата выдаёт margin:

\[
m_\theta(x,z,g)=
\mu_\theta(x,g)+
\sum_{\ell=1}^{r}a_{\theta,\ell}(x,z)\,\phi_{\theta,\ell}(x,g),
\]

где low-rank basis \(\phi_\ell\) разделяет вычисление query features и scene-level randomness. Decoder выбирает локальные target/obstacle tokens вокруг пальцев, геометрические признаки между пальцами и terminal body clearance. Непрерывные активации и spectral/Lipschitz regularization делают level set

\[
U_{\theta,z}(x)=\{g:m_\theta(x,z,g)\le0\}
\]

замкнутым.

### 6.3. Capacity layer с гарантированной согласованностью

Для \(M\) общих latent samples и \(Q\) Sobol-точек в каждом probe:

\[
\widehat T_{U,\theta}(K\mid x)
=\frac1M\sum_{j=1}^{M}
\sigma\!\left(
-\tau^{-1}\operatorname{softmin}_{q=1:Q}
m_\theta(x,z_j,g_q)
\right).
\]

При \(\tau\to0\) это эмпирическая частота события \(U_{\theta,z_j}\cap K\neq\varnothing\). Поскольку каждый sample сначала задаёт целое замкнутое множество, полученный empirical capacity автоматически монотонен и обладает нужной alternating-структурой. Это предпочтительнее произвольной сети \(f(x,K)\), для которой валидность capacity пришлось бы приближённо штрафовать экспоненциальным числом ограничений.

Для unions один и тот же набор \(z_j\) применяется ко всем компонентам. Именно так модель учит корреляцию вида «при одной скрытой форме опасна левая зона, при другой — правая».

### 6.4. Поиск захвата без dense volume

Поиск можно построить как собственный differentiable set search, а не как замену головы в VGN/TARGO:

1. Из видимых target points и learned boundary tokens сгенерировать 256–512 coarse action cells по позиции, ориентации и ширине.
2. Оценить feasible hitting probability \(T_F(C\mid x)\) для каждой ячейки.
3. Сохранить и рекурсивно разбить ячейки, в которых вероятно существует margin-положительное действие.
4. Оптимизировать 32–64 центра по ожидаемому margin с diversity repulsion.
5. Выбрать команду

\[
g^*=\arg\max_g\left[1-widehat T_U(K_g\mid x)ight],
\]

а ожидаемый положительный margin использовать только как tie-breaker.

Если максимальная avoidance probability ниже заранее калиброванного порога, модель возвращает «нет надёжного захвата». Для честной оценки следует показывать success–coverage curve; при обязательном выборе — отдельно success at 100% coverage.

### 6.5. Реалистичный вычислительный бюджет

Начальная инженерная гипотеза, а не результат: 1024 target points, 1024 obstacle/shelf points, \(M=8\) latent fields, 512 кандидатов и \(Q=16\) Sobol perturbations. Encoder и basis вычисляются один раз, latent coefficients — восемь раз, а queries полностью батчатся. Цель — менее 100 мс на современной GPU и существенно меньше памяти, чем у 64k voxel queries плюс completion. Эти числа должны быть проверены; их нельзя подавать как достигнутые заранее.

## 7. Теоретическое ядро статьи

### Proposition 1: task-quotient sufficiency

Если условная utility решения имеет вид \(u(g,S,O)=\widetilde u(g,U_{S,O})\), то Bayes-optimal decision зависит от \(p(S\mid x)\) только через pushforward-law \(p(U\mid x)\). Следовательно, восстановление переменных формы внутри одного класса \([S]\) статистически избыточно для этой задачи.

Это простой результат через условное ожидание, но он формально отделяет «reconstruction may help» от «reconstruction is necessary».

### Proposition 2: capacity identifiability under proper supervision

Для каждого probe \(K\) conditional BCE минимизируется при

\[
\widehat T(K\mid x)=\Pr(U\cap K\neq\varnothing\mid x).
\]

Если probes образуют separating/dense family компактов, согласованная оценка capacity определяет условный закон random closed set. В статье нужно точно сформулировать топологию на \(\mathcal G\), measurability и условия продолжения с плотного семейства; нельзя ограничиться ссылкой на теорему Шоке.

### Proposition 3: exact bounded-error interpretation

Пусть фактическая команда лежит в \(K_g=\{\exp(\widehat\xi)g:\xi^\top\Sigma^{-1}\xi\le1\}\). Событие «существует допустимое возмущение, приводящее к hidden-geometry failure» эквивалентно \(U\cap K_g\neq\varnothing\). Поэтому capacity является точным conditional probability этого robust-failure event относительно неопределённости скрытой формы. Это не heuristic averaging noise samples.

Важно различать existential bounded-error certificate и риск при известном распределении ошибки. Для стохастической ошибки \(\xi\) можно дополнительно интегрировать \(\Pr(m\le0\mid x,\xi)\); основная статья должна ясно сказать, какой из двух смыслов используется.

### Proposition 4: pointwise non-identifiability

Двухдействиевый контрпример из раздела 4 доказывает, что равенство всех singleton marginals на конечном action subset не влечёт равенства hit probabilities его unions. В непрерывном случае аналог строится на двух непересекающихся компактных областях. Это оправдывает union probes и общий latent.

### Proposition 5: finite query approximation

Если \(m_\theta(x,z,\cdot)\) \(L\)-липшицева, а \(\{g_q\}_{q=1}^Q\) — \(\varepsilon\)-сеть \(K\), то

\[
\left|\min_{g\in K}m(g)-\min_q m(g_q)\right|\le L\varepsilon.
\]

Следовательно, hit/no-hit определяется точно, когда истинный минимум отделён от нуля больше чем на \(L\varepsilon\). Это связывает стоимость probes, smoothness decoder и ошибку capacity, давая содержательную compute–accuracy теорему.

### Необязательное усиление

Вместо/в дополнение к BCE можно использовать strictly proper kernel score между эмпирическими random sets, если определить characteristic kernel на distance transforms или hit vectors. Теоретическую основу дают работы о [strictly proper kernel scores](https://arxiv.org/abs/1704.02578) и [kernel distribution embeddings](https://www.jmlr.org/papers/v19/16-291.html). Но для первой версии статьи это необязательно: capacity-BCE проще, имеет прямой смысл и лучше поддерживает главный тезис.

## 8. Данные и протокол обучения

### 8.1. Основной benchmark: ShelfOcclude-PJ

Нужен изолированный benchmark, потому что стандартный clutter смешивает скрытую форму цели, сегментацию, collision avoidance и планирование. Предлагаемая сцена:

- один target из ACRONYM/ShapeNet/Objaverse с лицензированным mesh;
- один foreground obstacle с независимо выбранной формой;
- плоскость/угол полки;
- фиксированная или небольшая выборка wrist-camera поз;
- видимая доля цели от 20% до 100%;
- RealSense-подобные depth holes, quantization, edge flying pixels и extrinsic noise;
- terminal parallel-jaw oracle с фиксированной геометрией захвата.

Split обязан удерживать отдельно unseen object instances, unseen categories, obstacle shapes, occlusion geometry и sensor/noise regimes. Иначе общий latent просто выучит идентичность объектов.

### 8.2. Occlusion twins

Главный диагностический набор — **occlusion twins**: пары полных объектов/положений, дающие одинаковое или почти одинаковое видимое RGB-D, но разные скрытые thickness, handle, cavity либо contact normals. У них должны совпадать visible point cloud и singleton evidence, но различаться feasible-set correlations.

Оценка на twins отвечает на главный научный вопрос лучше, чем средняя success rate: научилась ли модель представлять реальную неоднозначность или лишь memorized prior?

### 8.3. Генерация probes и oracle labels

Эффективный офлайн pipeline:

1. Сэмплировать сцены и рендерить одно шумное RGB-D.
2. Сгенерировать action cells вокруг видимой поверхности и скрытой bounding region.
3. Выполнить batched collision/contact oracle на полном mesh, сохранив margin.
4. На лету собирать singleton, tube и union probes и вычислять hit label из query bank.
5. Переоценивать hard probes по текущей модели каждые несколько эпох.

Нужно не смешивать source of uncertainty: основной random set вызывается скрытой геометрией. Friction/mass рандомизируются в узком реалистичном диапазоне и либо входят в oracle marginalization, либо фиксируются в главном эксперименте с отдельным robustness appendix.

### 8.4. Внешняя проверка

[TARGO](https://arxiv.org/abs/2407.06168) стоит использовать как внешний stress test, но не как определение основной задачи, поскольку его сцены cluttered. Можно выбрать эпизоды с одним доминирующим occluder и использовать TARGO-Real без переноса его completion target. [GraspClutter6D](https://arxiv.org/abs/2504.06866) полезен только как дополнительный out-of-scope generalization test.

## 9. Эксперименты, которые способны опровергнуть идею

### 9.1. Baselines

Нужны не только публичные модели, но и capacity-neutral controls:

- S4G / Contact-GraspNet / AnyGrasp как direct per-grasp family;
- GIGA / NeuGraspNet как joint implicit geometry-affordance family;
- TARGO-Net как target completion + grasp field;
- multiple completion + evaluate across shapes;
- vMF-Contact как uncertainty-aware direct model;
- тот же encoder/decoder, но deterministic \(z\);
- тот же stochastic decoder с singleton-only BCE;
- CCM без union probes;
- CCM с независимым latent на каждый query вместо shared latent;
- полный CHOQUET-GRASP;
- geometry-oracle upper bound и visible-only lower bound.

Сравнение одинаковой ёмкости encoder и одинакового action-query budget особенно важно: иначе выигрыш можно объяснить большим backbone.

### 9.2. Метрики

Основные:

- terminal grasp success и small-lift success по occlusion bins;
- success under bounded pose/width perturbations;
- worst-bin success и area under success-vs-occlusion curve;
- Brier/NLL/ECE для singleton и, отдельно, для held-out tube/union hit events;
- selective success–coverage и success at 100% coverage;
- inference latency, peak memory, число oracle/action queries;
- mode/component coverage допустимого action set;
- gap до geometry-oracle.

Chamfer/IoU формы не должна быть главной метрикой, потому что метод намеренно её не выдаёт. Для completion baselines эти показатели можно привести лишь диагностически.

### 9.3. Реальные испытания

Минимально убедительный дизайн: 40–60 новых объектов, один foreground obstacle, четыре occlusion bins, два sensor/noise режима, парные сцены для методов, суммарно 400–600 попыток. Отчёт: Wilson intervals, paired bootstrap или McNemar для парных бинарных исходов, пять training seeds для симуляции. Захваты выбираются без ручного отбора; threshold/no-grasp фиксируется до теста.

### 9.4. General-ML benchmark

Для ICLR нужен второй домен без робототехнических деталей: **Hidden-Constraint Placement**. Частично наблюдаемая 2D/3D форма должна быть размещена непрерывным rigid action внутри области со скрытым вырезом. Полная форма задаёт closed feasible set поз; наблюдение неоднозначно; точные hit labels и Bayes law вычислимы. Сравниваются pointwise classifier, conditional neural field, shape reconstruction и CCM.

Этот benchmark позволяет показать:

- восстановление union probabilities при одинаковых singleton marginals;
- статистическую сходимость capacity;
- зависимость ошибки от \(M,Q,\varepsilon\);
- перенос идеи за пределы grasping без RL.

## 10. Проверка новизны относительно ближайших работ

| Работа/семейство | Что предсказывается | Как представлена скрытая неоднозначность | Отличие CHOQUET-GRASP |
|---|---|---|---|
| S4G, Contact-GraspNet, AnyGrasp | pose/quality отдельных grasps | point estimate или локальный score | закон random feasible set, shared latent, union/tube capacities |
| VGN, GIGA, NeuGraspNet | geometry + affordance field | implicit/voxel scene representation | нет reconstruction target или SDF; task quotient |
| TARGO-Net | completed target + dense grasp field | одна завершённая форма | posterior pushforward сразу в action space |
| Multiple shape completions | набор полных voxel shapes | samples geometry posterior | samples только низкоразмерного action field; proper set-level loss |
| vMF-Contact | uncertainty направления/контакта | pointwise evidential distribution | region-level joint events и topologically valid random closed sets |
| FFHFlow | likelihood сгенерированных dexterous grasps | density over actions | моделируется множество всех опасных/допустимых действий, а не только proposal density |
| ShellGrasp / task-oriented completion | shell/contact-relevant geometry | сокращённая геометрия | выход — не геометрия, а equivalence class действий |
| Learning All Feasible Actions | density на feasible actions | детерминированное fully observed set | conditional law random set under hidden state; capacity supervision |
| Feasibility-aware DFL | решение/параметры ограничений | предсказательная неопределённость | учится само нелинейное random constraint set в continuous action space |
| Random-set neural classifiers | набор меток/масса belief | дискретные class subsets | continuous topological action set и physical region queries |

На дату исследования точного сочетания **task quotient + conditional random closed action set + capacity-matching supervision на unions/tubes + shared latent continuous grasp field** обнаружено не было. Это отрицательный результат поиска, а не абсолютное доказательство отсутствия; перед подачей нужен повторный literature search по новым preprints.

## 11. Почему направление правдоподобно, но SOTA не гарантирован

Косвенные данные складываются в согласованную цепочку:

1. TARGO показывает, что окклюзия систематически разрушает современные grasp predictors, а completion помогает, следовательно hidden-shape prior действительно содержит полезный сигнал.
2. Multiple-completion grasp planning показывает пользу marginalization по формам, но также демонстрирует стоимость геометрического посредника.
3. vMF-Contact показывает, что явная uncertainty улучшает реальные clearance/robustness показатели, следовательно калиброванная неоднозначность практична.
4. Contact-GraspNet и ShellGrasp показывают, что task-aligned сжатие пространства/геометрии может быть эффективнее универсального представления.
5. Decision-focused learning подтверждает общий принцип: оптимизация predictive fidelity не обязана совпадать с качеством downstream decision. См. [Predict-then-Optimize generalization](https://proceedings.neurips.cc/paper/2019/hash/a70145bf8b173e4496b554ce57969e24-Abstract.html), [Decision-Focused Learning without Decision-Making](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0904c7edde20d7134a77fc7f9cd86ea2-Abstract-Conference.html) и современные [feasibility-aware](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ef158a0d4f5364741a571b8d1d44fb1b-Abstract-Conference.html) подходы.
6. Random-set theory даёт не метафору, а идентифицирующий объект и точное значение регионального риска.

Но эти пункты **не доказывают**, что CHOQUET-GRASP превзойдёт SOTA. Full completion может выиграть при сильных pretrained shape priors; capacity estimator может иметь высокую variance; поиск по \(SE(3)\) может съесть вычислительное преимущество. Поэтому правильное утверждение статьи — не «мы неизбежно лучше», а «мы вводим decision-sufficient statistical target и проверяем, даёт ли он superior reliability–compute trade-off».

## 12. ICLR-аудит

Официальный [ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide) требует нового знания, ясной мотивации, технической корректности, сильной эмпирической/теоретической поддержки и значимости; SOTA сам по себе не обязателен. [Call for Papers](https://iclr.cc/Conferences/2026/CallForPapers) включает uncertainty, structured prediction, general ML и robotics, то есть тематически работа уместна.

### Потенциально сильные стороны

- Новый supervised object: conditional law непрерывного random feasible set.
- Точная идентификация через capacity, а не произвольный риск-score.
- Простой контрпример, показывающий недостаточность стандартной pointwise цели.
- Архитектура, конструктивно гарантирующая валидный empirical capacity.
- Decision-sufficiency theorem и compute–approximation bound.
- Практически важный и хорошо изолируемый failure regime.
- Возможность general-ML демонстрации вне robotics.

### Вероятные возражения reviewers и ответы

**«Это всего лишь BCE на perturbation sets».**  
Ответ должен быть экспериментальным и теоретическим: union probes различают распределения с одинаковыми singleton marginals; shared random field восстанавливает held-out region events; arbitrary jitter baseline этого не делает.

**«Почему не восстановить форму, если она полезна?»**  
Не отрицать полезность. Сравнить с сильной completion-моделью при одинаковых compute/data budgets, показать equivalence/twin cases, где mesh fidelity расходует capacity на decision-irrelevant детали, и дать hybrid upper bound.

**«Capacity на всех компактах невозможно обучить».**  
Задать separating probe distribution, доказать finite-net bound, показать convergence по числу/типам probes и held-out probe families.

**«Метод просто запоминает shape prior».**  
Unseen category, occlusion twins, prior-shift, noise-shift и calibration/abstention. На OOD скрытых формах следует честно ожидать деградацию.

**«Вклад слишком робототехнический для ICLR».**  
Сделать AQCL центральным методом structured prediction, добавить Hidden-Constraint Placement и использовать grasping как сложную физическую проверку.

**«Random sets уже существуют».**  
Новизна не в математическом объекте отдельно, а в learnable conditional random action set, proper capacity supervision на регионах и coherent implicit mechanics field. Нужен аккуратный related work по [random-set neural classifiers](https://openreview.net/forum?id=fzzWSGoXFC) и [conformal functional prediction sets](https://proceedings.mlr.press/v286/gray25a.html): первые работают с дискретными множествами классов, вторые строят гарантированные множества функций, но ни те, ни другие не учат capacity непрерывного физического action set из частичной геометрии.

### Честная оценка

- Только робот, без theorem/second domain: сильная CoRL-подача, слабая ICLR-идентичность.
- Теория без convincing real-world win: интересная, но риск «solution in search of a problem».
- Полный пакет — theorem, synthetic identifiability, isolated shelf benchmark, real trials, compute comparison: **правдоподобная ICLR-работа с высокой новизной**, хотя acceptance и SOTA заранее не гарантируются.

## 13. Falsification gates до дорогих real-robot опытов

Работу следует прекратить или радикально пересобрать, если выполнено хотя бы одно:

1. На occlusion twins union/tube NLL полного метода не лучше singleton-only модели при одинаковом backbone.
2. Shared latent не восстанавливает известную корреляционную структуру Hidden-Constraint Placement.
3. Completion baseline при одинаковой latency/parameter budget доминирует по success и calibration во всех occlusion bins.
4. \(M,Q\), достаточные для устойчивой capacity, делают inference медленнее multiple completion.
5. Выигрыш исчезает после контроля размера encoder или числа oracle labels.
6. Реальная depth noise полностью разрушает occlusion-boundary features, и training corruption не исправляет перенос.

Положительный go/no-go результат первой фазы: не менее 20% относительного снижения tube-event NLL против singleton-only на twins, улучшение worst-occlusion success при сопоставимой latency и отсутствие ухудшения clear-view success более чем на статистическую погрешность. Это целевые критерии проекта, не обещанные результаты.

## 14. Минимальная последовательность исследования

### Фаза A: математика и toy domain (4–6 недель)

- Формально задать random closed sets на компактном подмножестве Lie group.
- Реализовать 2D Hidden-Constraint Placement с аналитическим законом.
- Проверить singleton non-identifiability, union recovery, monotonicity и \(M,Q\)-scaling.
- Сравнить direct capacity regressor и generative random-field construction.

### Фаза B: симуляция (6–10 недель)

- Создать ShelfOcclude-PJ и terminal mechanics oracle.
- Обучить равные по размеру pointwise, completion и CCM baselines.
- Провести twins, occlusion, prior/noise shift и compute ablations.
- Зафиксировать probes и selection rule до real tests.

### Фаза C: реальный RGB-D (4–6 недель)

- Калибровать sensor corruption только на train objects.
- Выполнить парные trials по заранее опубликованному протоколу.
- Отчитывать success–coverage, calibration и latency вместе, не выбирать только лучшую метрику.

### Фаза D: paper hardening

- Повторить поиск новых preprints.
- Проверить proofs независимым специалистом по random sets.
- Выпустить код, probe generator, scene splits и полный negative-results appendix.

## 15. Что именно можно заявлять в статье

Корректное центральное утверждение:

> При частичном наблюдении скрытый мир индуцирует условное случайное множество допустимых действий. Его capacity functional является decision-sufficient, идентифицируемой и операционно значимой целью: он оценивает вероятность пересечения опасного множества с целой областью исполнения. CHOQUET-GRASP учит этот объект напрямую из offline mechanics queries, без восстановления полной формы.

Не следует заявлять без результатов:

- что shape completion всегда избыточно;
- что метод гарантирует физическую безопасность вне training distribution;
- что capacity calibration автоматически даёт distribution-free guarantee;
- что bounded-error existential risk равен average stochastic execution risk;
- что latency меньше 100 мс или метод является SOTA;
- что решены clutter, approach planning и full manipulation cycle.

## 16. Итоговая оценка идеи

**Новизна:** высокая при сохранении set-level law, union probes и theory; средняя, если оставить только robust neighborhoods.  
**Обучаемость:** реалистичная благодаря бинарным hit labels, shared low-rank latent и batched queries.  
**Инженерный риск:** средне-высокий из-за \(SE(3)\)-поиска и variance Monte Carlo capacity.  
**Научная фальсифицируемость:** высокая; есть twins, pointwise counterexample, held-out probe metrics и равнобюджетные baselines.  
**Соответствие исходной задаче:** прямое — single noisy RGB-D, target + foreground obstacle, parallel jaw, learned hidden-shape distribution, без full reconstruction.  
**Потенциал ICLR:** убедительный только как общая structured-prediction идея с робототехнической проверкой, а не как локальная модификация grasp pipeline.

Самая важная мысль: скрытая форма не должна быть промежуточной «картинкой мира», если решение видит её только через допустимые действия. Но заменить форму одиночным grasp score недостаточно. Нужен закон **целого множества действий**, и capacity matching даёт редкий случай, когда математически естественный объект одновременно имеет прямое физическое значение, proper supervision и проверяемое преимущество над pointwise learning.

## Основные источники

- Qin et al., [TARGO: Benchmarking Target-driven Object Grasping under Occlusions](https://arxiv.org/abs/2407.06168).
- Qin et al., [Amodal Single-view Single-Shot SE(3) Grasp Detection in Cluttered Scenes](https://proceedings.mlr.press/v100/qin20a.html).
- Sundermeyer et al., [Contact-GraspNet](https://arxiv.org/abs/2103.14127).
- Breyer et al., [Volumetric Grasping Network](https://arxiv.org/abs/2101.01132).
- Jiang et al., [Synergies Between Affordance and Geometry: 6-DoF Grasp Detection via Implicit Representations](https://roboticsproceedings.org/rss17/p024.pdf).
- Li et al., [NeuGraspNet](https://arxiv.org/abs/2306.07392).
- Fang et al., [AnyGrasp](https://arxiv.org/abs/2212.08333).
- Fang et al., [GraspNet-1Billion](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html).
- Lundell et al., [Robust Grasp Planning over Uncertain Shape Completions](https://arxiv.org/abs/1903.00645).
- Ji et al., [vMF-Contact](https://arxiv.org/abs/2411.03591).
- [Learning to Generate All Feasible Actions](https://arxiv.org/abs/2301.11461).
- Molchanov, [Theory of Random Sets](https://link.springer.com/book/10.1007/978-1-4471-7349-6).
- Wilder et al., [Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization](https://ojs.aaai.org/index.php/AAAI/article/view/5012) и связанная литература decision-focused learning.
- Garnelo et al., [Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a.html) — источник идеи общего латента случайной функции, не робототехнический antecedent.
- [ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide).
