# intelligent-placer
## Постановка задачи
"Intelligent placer" - программа (алгоритм), определяющий возможность уместить весь набор объектов в заданный многоугольник, не накладывая вещи друг на друга.

### Входные\Выходные значения
*Вход:* 
На вход поступает изображение в формате *.jpg, содержащее набор предметов и многоугольник. Многоугольник должен быть изображен на белом листе бумаги черными яркими линиями; лист с многоугольником должен находиться в левой части изображения. Сами объекты должны быть отчетливо различимы на белой поверхности и расположены в правой части изображения.

*Выход:* 
После выполнения программа должна вывести одно слово в консоль:  **True**, если все предметы на фотографии могут поместиться в нарисованном многоугольнике, или **False**, если не все предметы могут поместиться в многоугольник или если изображение, поступившее на вход, не удовлетворяет требованиям.


## Требования
### Требования к фотографиям
+ Формат изображения: трехканальный jpg
+ Ориентация фотографии: горизонтальная
+ Расстояние от камеры до поверхности: не более 50 см
+ Направление камеры: параллельно поверхности, на которой размещены объекты (допустимо небольшое отклонение, не превышающее 15 градусов от нормали поверхности)
+ Перспективные искажения ***недопустимы***
+ Освещение: освещение должно не создавать артефактов, мешающее точно определять габариты объектов (не было большой падающей тени; небольшие тени не должны создавать пересечений с объектами, которые их не отбрасывают; блики не нарушали четкость изображения); допускается применение вспышки, если она не ухудшает видимость объекта (не создает ранее упомянутые артефакты). Также не допускаются различные блики на изображении.
+ Цвет освещения: допускается освещение "естественного" цвета - цвета, характерного солнечному, наружнему свету. Также допускается теплые, белые или слегка холодные цвета освещения, которые свойственны многим источникам освещения в помещениях. ***Запрещается*** использовать ярко-выраженные "необычные" цвета освещения: например, алые, синие, неоново-фиолетовые.


### Требования к многоугольнику:
+ Многоугольник должен быть выпуклым (как следствие, без самопересечения ребер)
+ Многоугольник должен иметь не более 7 вершин
+ Многоугольник должен быть изображен на *белом* листе бумаги, без рисунка
+ Многоугольник должен быть изображен четко: в виде ***черной замкнутой*** ломанной, ***без закраски***, толщина линии ***не тоньше 1.5мм***
+ Допускается, что многоугольник может быть сделан в редакторе и напечатан, если он удовлетворяет требованиям выше (линии черные, образуют замкнутую линюю допустимой толщины), а также напечатан ***качественно и без артефактов***: нет потеков и пятен от принтера, сам принтер не осветляет черный цвет и не искривляет формат линий;
+ Если многоугольник изображается рисунком от руки, то рисунок должен быть сделан также черными четкими (без текстуры, как например у мелка) линиями: например, маркером или фломастером. Многоугольник должен быть выполнен аккуратно, без больших потёков.


### Требования к предметам:
+ Рассматриваются только предметы из предварительно подготовленного набора (см. ниже ссылку)
+ Предметы не повторяются
+ Предметы должны быть четко различимы
+ Любой предмет на изображении должен помещаться в кадр
+ Предметы не должны перекрывать друг друга
+ Предметы не должны пересекать изображенный многоугольник


### Требования к поверхности:
+ Однотонная горизонтальная поверхность светлого цвета (возможна несильно выделяющаяся текстура, как на фото в примерах)
+ Ровная, без рельефа


## План реализации

### Общий алгоритм 

Общий алгоритм работы программы можно разделить на этапы. Ниже представлены эти этапы с подробным объяснением работы каждого.
#### I этап: выделение многоугольника и объектов.
1. При помощи бинаризации ("Minimum thresolding"), а также морфологических операций (закрытия), создаются маски: в силу требований к поверхности, к объектам и к изображению многоугольника, мы можем сказать, что все интересующие нас объекты (а именно объекты из ограниченного множества и многоугольник) темного цвета (или по крайней мере "не светлые"), а лист с цветом поверхности - светлые. Потому применение бинаризации и морфологических операций способно выделить объекты и многоугольник.
2. В силу требования "о не пересечении предметов", итог бинаризации можно разбить на компоненты связности: таким образом программа сможет разделить все обнаруженное на маски отдельных объектов и многоугольника.
3. Для того, чтобы отличить многоугольник от остальных объектов, можно также воспользоваться требованием "о рассмотрении предметов из заранее подготовленного набора". В заранее подготовленном наборе все объекты "в среднем" имеют не белый цвет, однако, по требованиям к многоугольнику, лист должен быть именно белый. Потому маску, соответствующую белому объекту, можно считать маской многоугольника, а остальные - масками объектов.
4. На выходе этого этапа программа генерирует маску многоугольника и маски объектов.

#### II этап: предварительные проверки.
 В силу сложности основного этапа помещения объектов в многоугольник, разумно выделить ряд сильно упрощенных по времени исполнения проверок, которые могут гарантированно сказать, получится ли нам уместить все распознанные объекты в многоугольник или нет.
+ *Проверка площадей*: за счет отсутствия перспективных искажений, можно посчитать площадь многоугольника и суммарную площадь всех объектов путем пересчёта всех точек масок. Если по итогу работы этой проверки оказывается, что суммарная площадь объектов больше площади многоугольника, алгоритм выводит "False" и выходит; иначе переходим к этапу III.

#### III этап: этап "помещения" объектов.
Предварительный этап: 
1. Маски объектов сортируются по убыванию площадей. Это сделано для того, чтоб не делать лишних итераций в случае, когда наибольший объект в принципе не может поместиться в заданный многоугольник.
2. Маска многоугольника расширяется путем добавления пустых строк и столбцов, чтобы маски объектов могли накладываться на изображение с маской многоугольника, а также инверсируется для упрощения этапа "помещения" (об этом подробнее в конце).
3. се маски предметов и маска многоугольника сжимаются до определенного размера, чтобы уменьшить время выполнения программы.
4. Создается копия маски многоугольника, в которую будет происходить укладка вещей. Данная копия будет называться "временным результатом".
5. Запускается основной ("рекурсивный") этап помещения: этот этап возвращает 'True', если возможно уложить все предметы со стека в текущую "временную маску", и 'False' - если невозможно. Результат работы этого этапа выводится пользователю.

Основной этап (Рекурсивный): 
1. Берется со стэка отсортированных по размеру масок объектов маска. Если стэк пустой, то тогда все объекты разместились, а "временный результат" - один из способов разложения предметов: выводим True и выходим.

2. Для текущей рассматриваемой маски выделяется наименьший bound-box, чьи стороны параллельны сторонам изображения маски. По полученному размеру bound-box-а выделяется массив, на который переносится маска. Таким образом формируется "временная маска" объекта.

3. "Временная маска" объекта пытается наложиться на "временный результат" путем поэлементного AND, начиная с совпадения лево-верхних углов масок. 
Если все элементы по итогу равны 0, то считается, что объект разместился: в качестве нового "временного результата" берется результат поэлементного OR и алгоритм вызывает новый уроень рекурсии основного этапа. Если по итогу вызов рекурсии возвращает "True", то тогда все оставшиеся предметы успешно уложились во "временном резудьтате" и можно делать выход с этого уровня рекурсии, возвращая "True".
Если же рекурсивынй вызов основного этапа возвращает "False" или после наложения "временных" масок существует хоть один элемент с 1, то считается, что при текущей конфигурации сдвигов и поворотов укладка предмета невозможна. В этом случае происходит сдвиг временной маски относительно "временного результата" по оси OX вправо; если это сделать не удалось, то происходит сдвиг по OY вниз и смещение "временной маски" в левый край "временного результата"; если и это смещение не удается, то алгоритм переходит в пункт "3.4" - пункт поворота. Если же сдвиг все таки произошел, происходит повторное наложение масок поэлементным AND, но с учетом сдвига.

4. На этом этапе алгоритм оказывается только тогда, когда при заданном повороте не удалось поместить "временную маску" во "временное решение". В таком случае происходит поворот маски на некоторый фиксированный угол против часовой стрелки, если при этом не произошел полный поворот маски на >= 360 градусов. Если полного поворота не случилось, то тогда переходим на пункт 3.2. Если маска все таки повернулась на полный оборот, то считается, что при текущем замощении предыдущих объектов невозможно поместить текущий объект и требуется переукладка: алгоритм возвращает маску в стек и выходит из этого уровня рекурсии, возвращая 'False'. Если же выйти на более верхней уровень рекурсии не удается, то получается, что разместить объекты в многоугольник невозможно.

### Пояснение к процессу наложения (этап III).
Поскольку в подготовительном этапе алгоритм инвертирует маску многоугольника, то получается, что 0 соответствеует свободному месту, а 1 - занятому. В масках объектов все наоборот: 1 - пристутствует объект, 0 - отсутствует. Тогда, чтоб проверить, умещается ли в заданном месте объект, достаточно сделать AND:

<table align="center">
    <tr>
        <td>Poly (P)</td>
        <td> Obj (O)</td>
        <td>P & O = </td>
    </tr>
    <tr>
        <td>0</td>
        <td>0</td>
        <td>0</td>
    </tr>
    <tr>
        <td>0</td>
        <td>1</td>
        <td>0</td>
    </tr>
    <tr>
        <td>1</td>
        <td>0</td>
        <td>0</td>
    </tr>
    <tr>
        <td>1</td>
        <td>1</td>
        <td>1</td>
    </tr>
</table>

Т.е видно, что операция AND позволяет зафиксировать недопустимую ситуацию (когда в занятое место (1) пытаются уместить объект (1)). 

После прохождении этой проверки, в качестве новой "временной маски" можно рассматривать результат OR:

<table align="center">
    <tr>
        <td>Poly (P)</td>
        <td> Obj (O)</td>
        <td>P | O = </td>
    </tr>
    <tr>
        <td>0</td>
        <td>0</td>
        <td>0</td>
    </tr>
    <tr>
        <td>0</td>
        <td>1</td>
        <td>1</td>
    </tr>
    <tr>
        <td>1</td>
        <td>0</td>
        <td>1</td>
    </tr>
</table>

## Данные
Используемые предметы: [objects](objects)

Тесты: [tests](tests)
