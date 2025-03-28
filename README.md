# Анализ привлекательности стран для стартапов
Этот проект был разработан в рамках курса "Аналитика данных на Python". Работа выполнялась в команде из 4 человек (моих одногруппников)

## Описание проекта
Цель нашего проекта состоит в том, чтобы выяснить, какие факторы делают страну более привлекательной для стартапов. Доступ к венчурному финансированию, уровень доходов, инфляция, корпоративные налоги и экономическая стабильность — это основные показатели, которые мы выделили на основе изучения различных источников.

Мы создали индекс бизнес-привлекательности, собрав и проанализировав данные по более чем 60 странам. Затем мы проверили, какие факторы действительно влияют на количество стартапов с помощью алгоритма машинного обучения Random Forest.

## Исследовательская задача
В нашем исследовании мы хотим выяснить, какие экономические переменные оказывают решающее влияние на привлекательность страны для стартапов. Мы хотим определить, какие страны предлагают лучшие условия для бизнеса, анализируя венчурные инвестиции, ВВП на душу населения, инфляцию, корпоративные налоги, уровень безработицы, среднюю заработную плату.

### Мы разработали следующие гипотезы для выполнения задачи:
- Гипотеза 1. Страны с более высоким ВВП на душу населения более привлекательные для стартапов. 
- Гипотеза 2. Объем венчурных инвестиций является значимым фактором, влияющим на успешность стартапов. 
Мы использовали регрессионный анализ и алгоритм машинного обучения Random Forest для проверки этих гипотез и выявления наиболее важных показателей.

## Выбор данных
На начальном этапе нашего проекта мы собираем данные из различных источников, чтобы создать единый датафрейм, который в дальнейшем будет использоваться для анализа и построения модели. Модель должна оценивать, подходит ли та или иная страна для создания стартапа, основываясь на таких показателях, как:
- ВВП на душу населения
- уровень безработицы
- уровень инфляции
- средняя заработная плата
- корпоративные налоги
- объём венчурных инвестиций
- число зарегистрированных стартапов

### Библиотеки, которые мы использовали для сбора и начальной обработки данных:
 - pandas для работы с данными
 - requests для запросов к API
 - numpy для обработки числовых данных

### 1. Данные из API Всемирного банка (ВВП на душу населения)  
```python
# URL API Всемирного банка для получения данных о ВВП на душу населения
url = "http://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD?date=2023&format=json&per_page=300"

# Отправляем запрос к API
response = requests.get(url)
data = response.json()

# Проверяем, что данные получены
if len(data) > 1:
    gdp_data = data[1]
    
    # Фильтруем только страны (исключая регионы, у которых нет двухбуквенного кода страны)
    gdp_list = [(entry['country']['value'], entry['value']) for entry in gdp_data 
                if entry['value'] is not None and len(entry['country']['id']) == 2]
    
    # Создаем DataFrame
    df1 = pd.DataFrame(gdp_list, columns=["Country", "GDP_per_cap"])
    
df1.loc[df1['Country'] == 'Korea, Rep.', 'Country'] = 'South Korea'
df1
```
### 2. Данные об уровне безработицы
```python
#	Используем локальный файл API_SL.UEM.TOTL.NE.ZS_DS2_en_csv_v2_76310.xls, предоставляемый Всемирным банком
df2 = pd.read_csv('unemployment_rate.xls', skiprows = 4)

# оставляем только нужные столбцы
df2 = df2[['Country Name', '2023']]

# переименовываем колонки
df2 = df2.rename(columns = {'2023': 'Unemploym_2023', 'Country Name': 'Country'})

# переименовываем названия стран, чтобы они были одинаковые для всех показателей
df2.loc[df2['Country'] == 'Korea, Rep.', 'Country'] = 'South Korea'
df2.loc[df2['Country'] == 'Venezuela, RB', 'Country'] = 'Venezuela'

# добавляем вручную недостающие значения
df2.loc[len(df2)] = ['Iran', 9]
df2.loc[len(df2)] = ['Slovakia', 5.7]
df2.query("Country.str.contains('Ven', case=False, na=False)")
df2.loc[df2['Country'] == 'China', 'Unemploym_2023'] = 5.1

df2
```

### 3. Данные об инфляции
```python
#	Используем локальный файл global_inflation_data.xls
df3 = pd.read_csv('inflation_data.xls')

# оставляем только нужные столбцы
df3 = df3[['country_name', '2023']]

# переименовываем колонки
df3 = df3.rename(columns = {'2023': 'Inflation_rate', 'country_name': 'Country'})

# добавляем вручную недостающие значения
df3.loc[len(df3)] = ['China', 0.2]
df3.loc[len(df3)] = ['South Korea', 3.6]
df3.loc[len(df3)] = ['Slovakia', 10.5]

df3
```
### 4. Данные о средней заработной плате
```python
#	Используем файл Country_Region_Income.xlsx
df4 = pd.read_excel('gross_monthly_income.xlsx')

# добавляем вручную недостающие значения
df4.loc[len(df4)] = ['United Arab Emirates', 4000]
df4.loc[len(df4)] = ['Venezuela', 150]
df4.loc[len(df4)] = ['Brazil', 570]
df4.loc[len(df4)] = ['India', 280]
df4.loc[len(df4)] = ['Indonesia', 530]

df4
```

### 5. Индекс сложности ведения бизнеса 
```python
#	Используем файл Global_Business_Complexity_Index_2023.xlsx
df5 = pd.read_excel('business_complexity_index.xlsx')

# переименовываем колонки
df5 = df5.rename(columns = {'Global_business_complexity_inx_2023': 'Business_complexity_inx'})

# переименовываем названия стран, чтобы они были одинаковые для всех показателей
df5.loc[df5['Country'] == 'Russia', 'Country'] = 'Russian Federation'
df5.loc[df5['Country'] == 'UAE', 'Country'] = 'United Arab Emirates'
df5.loc[df5['Country'] == 'The Netherlands', 'Country'] = 'Netherlands'
df5.loc[df5['Country'] == 'Republic of Ireland', 'Country'] = 'Ireland'

# заменяем отсутствующие значения на ''
df5.loc[len(df5)] = ['Saudi Arabia', '']
df5.loc[len(df5)] = ['Estonia', '']
df5.loc[len(df5)] = ['Iran', '']

df5
```

### 6. Данные о корпоративных налогах
```python
#	Используем файл 1980_2023_Corporate_Tax_Rates_Around_the_World_Tax_Foundation_1.xlsx
df6 = pd.read_excel('corporate_tax_rate.xlsx')

# заменяем отсутствующие значения на процентный формат
df6['Corporate_Tax_Rate'] = df6['Corporate_Tax_Rate'].apply(lambda x: f'{x:.1f}%' if pd.notna(x) else np.nan)

# переименовываем названия стран, чтобы они были одинаковые для всех показателей
df6.loc[df6['Country'] == 'United States of America', 'Country'] = 'United States'
df6.loc[df6['Country'] == 'United Kingdom of Great Britain and Northern Ireland', 'Country'] = 'United Kingdom'
df6.loc[df6['Country'] == 'Republic of Korea', 'Country'] = 'South Korea'
df6.loc[df6['Country'] == 'Venezuela (Bolivarian Republic of)', 'Country'] = 'Venezuela'

# добавляем вручную недостающие значения
df6.loc[len(df6)] = ['Iran', '25%']

df6
```

### 7. Данные о венчурных инвестициях
```python
#	Используем файл statistic_id1480496_venture_capital_funding_rounds_worldwide_2023.xlsx
df7 = pd.read_excel('venture_capital_funding.xlsx', engine = 'openpyxl')
# измеряется в миллионах USD

# переименовываем колонки
df7 = df7.rename(columns = {'Unnamed: 0': 'Country', 'Unnamed: 1': 'Venture_cap_inv'})

# добавляем вручную недостающие значения
df7.loc[len(df7)] = ['Russian Federation', 118]

# заменяем отсутствующие значения на ''
df7.loc[len(df7)] = ['Luxembourg', '']
df7.loc[len(df7)] = ['South Africa', '']
df7.loc[len(df7)] = ['Cyprus', '']
df7.loc[len(df7)] = ['Estonia', '']
df7.loc[len(df7)] = ['Iran', '']
df7.loc[len(df7)] = ['Argentina', '']
df7.loc[len(df7)] = ['Slovakia', '']
df7.loc[len(df7)] = ['Venezuela', '']
df7.loc[len(df7)] = ['Mexico', '']
df7.loc[len(df7)] = ['Thailand', '']
df7.loc[len(df7)] = ['Poland', '']

df7
```

### 8. Данные о количестве зарегистрированных стартапов
```python
#	Используем файл `Total Startup Output.xlsx`.
df8 = pd.read_excel('total_startup_output.xlsx')

# переименовываем названия стран, чтобы они были одинаковые для всех показателей
df8.loc[df8['Country'] == 'Russia', 'Country'] = 'Russian Federation'
df8.loc[df8['Country'] == 'The Netherlands', 'Country'] = 'Netherlands'

# меняем формат значений на float
df8['Total_Startup_Output'].astype(float)

df8
```

## Проблемы, возникшие при сборе данных:

### 1. Несоответствие названий стран

В разных источниках страны могут называться по-разному (например, "Russia" и "Russian Federation"). Для решения этой проблемы мы вручную переименовываем страны, чтобы названия были единообразными.
  
### 2. Отсутствующие данные

Некоторые показатели отсутствуют для определённых стран. Мы либо оставляем их как `NaN`, чтобы в дальнейшем провести необходимые замены уже для итогового датафрейма, либо вручную добавляем недостающие значения, если они доступны из других источников.

### 3. Разные форматы данных

Например, корпоративные налоги представлены в виде строк с символом %. Мы преобразуем их в единый формат для удобства анализа.

## Объединение данных в единый датафрейм
Объединяем все датафреймы (df1 – df8) по столбцу Country с использованием внешнего соединения (outer), чтобы сохранить все страны из всех источников. Сортируем данные по названию страны и сбрасываем индексы.
- Датафрейм со всеми странами (df): Содержит все страны из исходных данных, но с пропусками во многих показателях.
- Датафрейм с 20 странами (df_20): Содержит только страны, для которых есть данные по всем показателям.
- Датафрейм с 60 странами (df_final): Расширенный набор стран, где пропущенные значения заменены на средние значения по соответствующим столбцам. Используется для построения модели.

 ### Датафрейм со всеми странами (df): Содержит все страны из исходных данных, но с пропусками во многих показателях.
  ```python
  df = df1
dfs = [df2, df3, df4, df5, df6, df7, df8] # собираем все показатели в один датафрейм

# запускаем цикл для объединения
for i in dfs:
    df = df.merge(i, on = 'Country', how = 'outer')
    
df = df.sort_values(by = 'Country') # сортируем по странам
df = df.reset_index(drop = True)
import warnings
warnings.filterwarnings('ignore')
# заменяем все Nan или пустые значения на нули для удобства работы
df = df.replace('', 0)
df = df.replace('NaN', 0)
df = df.fillna(0)

df
```
![df](https://github.com/ver369/Group_Project_Analysis_of_Country_Attractiveness_for_Startups/blob/main/df.png)

### Датафрейм с 60 странами (df_final): Расширенный набор стран, где пропущенные значения заменены на средние значения по соответствующим столбцам. Используется для построения модели.
```python
countries = ['Australia', 'Brazil', 'Canada', 'China', 'France', 'Germany', 'India', 'Indonesia', 'Israel', 'Italy', 'Japan', 'Netherlands', 'Russian Federation', 'Singapore', 'South Korea', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom', 'United States',
'Argentina', 'Austria', 'Belgium', 'Chile', 'Colombia', 'Czech Republic', 'Denmark', 'Egypt', 'Finland', 'Greece', 'Hungary', 'Ireland', 'Kazakhstan', 'Malaysia', 'Mexico', 'New Zealand', 'Norway', 'Pakistan', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Romania', 'Saudi Arabia', 'South Africa', 'Thailand', 'Turkey', 'Ukraine', 'United Arab Emirates', 'Vietnam', 'Bangladesh', 'Serbia', 'Slovakia', 'Croatia', 'Bulgaria', 'Ecuador', 'Venezuela', 'Nigeria', 'Algeria', 'Morocco', 'Qatar']

df_new = df[df['Country'].isin(countries)]
df_new = df_new.reset_index(drop = True)
df_new # новый датафрейм, тут много пропущенных значений

df_new = df_new.replace('', 0)
df_new = df_new.replace('NaN', 0)
df_new = df_new.fillna(0)
df_new # теперь где у нас нет значений мы проставили нолики для удобства

df_new.to_excel('df_new.xlsx', index = False)

df_final = pd.read_excel('df_new.xlsx')

df_final['Corporate_Tax_Rate'] = df_final['Corporate_Tax_Rate'].str[:-1].astype(float)

# для создания модели 20 стран было недостаточно (квадрат количества наших столбцов был больше, чем количество наших строк)
# поэтому мы расширили датасет до 60 стран, заменив пропущенные значения на среднее по всему столбцу

# звменяем нули на NaN
df_final['Gross_monthly_wage'] = df_final['Gross_monthly_wage'].replace(0, pd.NA)
df_final['GDP_per_cap'] = df_final['GDP_per_cap'].replace(0, pd.NA)
df_final['Unemploym_2023'] = df_final['Unemploym_2023'].replace(0, pd.NA)
df_final['Inflation_rate'] = df_final['Inflation_rate'].replace(0, pd.NA)
df_final['Business_complexity_inx'] = df_final['Business_complexity_inx'].replace(0, pd.NA)
df_final['Corporate_Tax_Rate'] = df_final['Corporate_Tax_Rate'].replace(0, pd.NA)
df_final['Venture_cap_inv'] = df_final['Venture_cap_inv'].replace(0, pd.NA)
df_final['Total_Startup_Output'] = df_final['Total_Startup_Output'].replace(0, pd.NA)


# заполнили NaN средним значением
df_final['Gross_monthly_wage'] = df_final['Gross_monthly_wage'].fillna(df_final['Gross_monthly_wage'].mean())
df_final['GDP_per_cap'] = df_final['GDP_per_cap'].fillna(df_final['GDP_per_cap'].mean())
df_final['Unemploym_2023'] = df_final['Unemploym_2023'].fillna(df_final['Unemploym_2023'].mean())
df_final['Inflation_rate'] = df_final['Inflation_rate'].fillna(df_final['Inflation_rate'].mean())
df_final['Business_complexity_inx'] = df_final['Business_complexity_inx'].fillna(df_final['Business_complexity_inx'].mean())
df_final['Corporate_Tax_Rate'] = df_final['Corporate_Tax_Rate'].fillna(df_final['Corporate_Tax_Rate'].mean())
df_final['Venture_cap_inv'] = df_final['Venture_cap_inv'].fillna(df_final['Venture_cap_inv'].mean())
df_final['Total_Startup_Output'] = df_final['Total_Startup_Output'].fillna(df_final['Total_Startup_Output'].mean())

# округлили
df_final['Gross_monthly_wage'] = df_final['Gross_monthly_wage'].round(1)
df_final['GDP_per_cap'] = df_final['GDP_per_cap'].round(1)
df_final['Unemploym_2023'] = df_final['Unemploym_2023'].round(1)
df_final['Inflation_rate'] = df_final['Inflation_rate'].round(1)
df_final['Business_complexity_inx'] = df_final['Business_complexity_inx'].round(1)
df_final['Corporate_Tax_Rate'] = df_final['Corporate_Tax_Rate'].round(1)
df_final['Venture_cap_inv'] = df_final['Venture_cap_inv'].round(1)
df_final['Total_Startup_Output'] = df_final['Total_Startup_Output'].round(1)

df_final = df_final.rename(columns = {'Unemploym_2023': 'Unemploym_rate'})

df_final = df_final.drop([6, 23, 25])
df_final = df_final.reset_index(drop = True)
df_final
```
### Датафрейм с 20 странами (df_20): Содержит только страны, для которых есть данные по всем показателям.
```python
# код для создания датасета из 20 стран, где значения по всем столбцам были изначально

df_20 = df1
dfs = [df2, df3, df4, df5, df6, df7, df8]

for i in dfs:
    df_20 = df_20.merge(i, on = 'Country', how = 'inner')
df_20 = df_20.sort_values(by = 'Country')
df_20 = df_20.reset_index(drop = True)
df_20 = df_20.drop([1, 7, 9])
df_20 = df_20.reset_index(drop = True)

df_20 = df_20.rename(columns = {'Unemploym_2023': 'Unemploym_rate'})

df_20
```

## Разведывательный анализ данных
## 1.	Расширенный описательный анализ данных
В первую очередь мы провели расширенный описательный анализ данных, добавляя к стандартной статистике (среднее, стандартное отклонение, минимум, максимум и т.д.) три важные метрики: медиану, асимметрию (skewness) и эксцесс (kurtosis).
```python
# Этот код выполняет расширенный описательный анализ данных, ...
# ...добавляя к стандартной статистике (среднее, стандартное отклонение, минимум, максимум и т.д.) ...
# ... важные метрики: медиану, асимметрию (skewness) и эксцесс (kurtosis).

from scipy.stats import skew, kurtosis

desc_stats = df_final.describe()

desc_stats.loc['median'] = df_final.median(numeric_only = True)
desc_stats.loc['skewness'] = df_final.apply(lambda x: skew(x.dropna()) if pd.api.types.is_numeric_dtype(x) else None)
desc_stats.loc['kurtosis'] = df_final.apply(lambda x: kurtosis(x.dropna()) if pd.api.types.is_numeric_dtype(x) else None)

desc_stats
```
![Описательная_статистика](https://github.com/ver369/Group_Project_Analysis_of_Country_Attractiveness_for_Startups/blob/main/%D0%9E%D0%BF%D0%B8%D1%81%D0%B0%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D0%B0%D1%8F%20%D1%81%D1%82%D0%B0%D1%82%D0%B8%D1%81%D1%82%D0%B8%D0%BA%D0%B0.png)
### Ключевые выводы:
- ВВП на душу населения и средняя зарплата близки к нормальному распределению — признак стабильной экономики.
- Безработица и инфляция имеют резкие асимметрию и эксцесс (например, Венесуэла — выброс).
- Сложность ведения бизнеса распределена равномерно, а налоги смещены в сторону низких ставок.
- Венчурные инвестиции и число стартапов резко skewed (лидеры — США и др.).

## 2.	Анализ выбросов с помощью boxplot
```python
import matplotlib.pyplot as plt
import seaborn as sns
# Анализ выбросов с помощью boxplot
plt.figure(figsize = (12, 8))
sns.boxplot(data = df_final.drop(columns = ['Country']))
plt.title('Boxplot для выявления выбросов')
plt.xticks(rotation = 45)
plt.show()
```
![Выбросы](https://github.com/ver369/Group_Project_Analysis_of_Country_Attractiveness_for_Startups/blob/main/%D0%92%D1%8B%D0%B1%D1%80%D0%BE%D1%81%D1%8B.png)
### Ключевые выводы:
Boxplot выявил выбросы по нескольким ключевым показателям: 
- высокий ВВП на душу населения (например, США, Швейцария)
- экстремально высокий уровень безработицы (например, Южная Африка)
- очень высокая инфляция (например, Венесуэла)
- значительные венчурные инвестиции (например, США, Китай).
Эти выбросы указывают на страны с уникальными экономическими условиями, которые могут быть как благоприятными, так и неблагоприятными для стартапов. Визуализация также подтвердила равномерное распределение индекса сложности ведения бизнеса и корпоративной налоговой ставки, что делает их важными факторами для анализа.

## 4. Выявление топов 10 стран по различным показателям

```python
from tabulate import tabulate

# Топ-10 стран по ВВП на душу населения
top_gdp = df_final.nlargest(10, 'GDP_per_cap')[['Country', 'GDP_per_cap']]
print('Топ-10 стран по ВВП на душу населения:')
print(tabulate(top_gdp, headers = 'keys', tablefmt = 'pretty', showindex = False))

# Топ-10 стран по объему венчурных инвестиций
top_venture = df_final.nlargest(10, 'Venture_cap_inv')[['Country', 'Venture_cap_inv']]
print('\nТоп-10 стран по объему венчурных инвестиций:')
print(tabulate(top_venture, headers = 'keys', tablefmt = 'pretty', showindex = False))

# Топ-10 стран по общему объему стартапов
top_startup_output = df_final.nlargest(10, 'Total_Startup_Output')[['Country', 'Total_Startup_Output']]
print('\nТоп-10 стран по общему объему стартапов:')
print(tabulate(top_startup_output, headers='keys', tablefmt = 'pretty', showindex = False))

# Топ-10 стран по низкому уровню безработицы
top_low_unemployment = df_final.nsmallest(10, 'Unemploym_rate')[['Country', 'Unemploym_rate']]
print('\nТоп-10 стран по низкому уровню безработицы:')
print(tabulate(top_low_unemployment, headers = 'keys', tablefmt = 'pretty', showindex = False))

# Топ-10 стран по низкому уровню инфляции
top_low_inflation = df_final.nsmallest(10, 'Inflation_rate')[['Country', 'Inflation_rate']]
print('\nТоп-10 стран по низкому уровню инфляции:')
print(tabulate(top_low_inflation, headers = 'keys', tablefmt = 'pretty', showindex = False))
```
### 1.	Топ-10 стран по ВВП на душу населения
![Топ-10_стран_по_ВВП](https://github.com/ver369/Group_Project_Analysis_of_Country_Attractiveness_for_Startups/blob/main/%D0%A2%D0%BE%D0%BF-10%20%D1%81%D1%82%D1%80%D0%B0%D0%BD%20%D0%BF%D0%BE%20%D0%92%D0%92%D0%9F.png)
ВВП на душу населения отражает общий уровень экономического развития страны, что напрямую влияет на возможности для стартапов.
Страны с высоким ВВП на душу населения (например, Ирландия, Швейцария, Норвегия) могут предоставить стартапам доступ к более богатым рынкам, квалифицированной рабочей силе и лучшей инфраструктуре. Это делает их привлекательными для запуска и развития бизнеса.

### 2.	Топ-10 стран по объему венчурных инвестиций
![Топ-10_стран_по_объему_венчурных_инвестиций](https://github.com/ver369/Group_Project_Analysis_of_Country_Attractiveness_for_Startups/blob/main/%D0%A2%D0%BE%D0%BF-10%20%D1%81%D1%82%D1%80%D0%B0%D0%BD%20%D0%BF%D0%BE%20%D0%BE%D0%B1%D1%8A%D0%B5%D0%BC%D1%83%20%D0%B2%D0%B5%D0%BD%D1%87%D1%83%D1%80%D0%BD%D1%8B%D1%85%20%D0%B8%D0%BD%D0%B2%D0%B5%D1%81%D1%82%D0%B8%D1%86%D0%B8%D0%B9.png)
Венчурные инвестиции — это ключевой источник финансирования для стартапов, и их объем напрямую влияет на успешность новых бизнесов.
Страны с высоким объемом венчурных инвестиций (например, США, Китай, Великобритания) предоставляют стартапам доступ к капиталу, что критически важно для их роста и развития.

### 3.	Топ-10 стран по общему объему стартапов
![Топ-10_стран_по_общему_объему_стартапов](https://github.com/ver369/Group_Project_Analysis_of_Country_Attractiveness_for_Startups/blob/main/%D0%A2%D0%BE%D0%BF-10%20%D1%81%D1%82%D1%80%D0%B0%D0%BD%20%D0%BF%D0%BE%20%D0%BE%D0%B1%D1%89%D0%B5%D0%BC%D1%83%20%D0%BE%D0%B1%D1%8A%D0%B5%D0%BC%D1%83%20%D1%81%D1%82%D0%B0%D1%80%D1%82%D0%B0%D0%BF%D0%BE%D0%B2.png)
Этот показатель отражает активность предпринимательской среды и показывает, насколько страна поддерживает инновации и новые бизнесы.
Страны с высоким объемом стартапов (например, США, Китай, Великобритания) имеют развитую экосистему для поддержки новых бизнесов, включая доступ к ресурсам, сетям и рынкам.

### 4.	Топ-10 стран по низкому уровню безработицы
![Топ-10_стран_по_низкому_уровню_безработицы](https://github.com/ver369/Group_Project_Analysis_of_Country_Attractiveness_for_Startups/blob/main/%D0%A2%D0%BE%D0%BF-10%20%D1%81%D1%82%D1%80%D0%B0%D0%BD%20%D0%BF%D0%BE%20%D0%BD%D0%B8%D0%B7%D0%BA%D0%BE%D0%BC%D1%83%20%D1%83%D1%80%D0%BE%D0%B2%D0%BD%D1%8E%20%D0%B1%D0%B5%D0%B7%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B8%D1%86%D1%8B.png)
Уровень безработицы влияет на доступность рабочей силы и стабильность экономики, что важно для стартапов.
Страны с низким уровнем безработицы (например, Таиланд, ОАЭ, Япония) могут предложить стартапам доступ к квалифицированной рабочей силе, что критически важно для их успеха.

### 5.	Топ-10 стран по низкому уровню инфляции
![Топ-10_стран_по_низкому_уровню_инфляции](https://github.com/ver369/Group_Project_Analysis_of_Country_Attractiveness_for_Startups/blob/main/%D0%A2%D0%BE%D0%BF-10%20%D1%81%D1%82%D1%80%D0%B0%D0%BD%20%D0%BF%D0%BE%20%D0%BD%D0%B8%D0%B7%D0%BA%D0%BE%D0%BC%D1%83%20%D1%83%D1%80%D0%BE%D0%B2%D0%BD%D1%8E%20%D0%B8%D0%BD%D1%84%D0%BB%D1%8F%D1%86%D0%B8%D0%B8.png)
Инфляция влияет на стоимость ресурсов и стабильность экономики, что важно для стартапов, особенно на ранних этапах развития.
Страны с низким уровнем инфляции (например, Китай, Швейцария, Таиланд) предоставляют стартапам стабильную экономическую среду, что снижает риски и способствует долгосрочному планированию.

## Регрессионный анализ 
### **Гипотезы исследования**  
**Общая гипотеза:**  
Экономические и бизнес-факторы страны влияют на количество стартапов (*Total_Startup_Output*).  
**Частные гипотезы:**  
1. **Венчурные инвестиции** (*Venture_cap_inv*):  
   - *H₁*: Чем больше инвестиций → тем больше стартапов.  

2. **Корпоративные налоги** (*Corporate_Tax_Rate*):  
   - *H₁*: Чем выше налоги → тем меньше стартапов.  

3. **ВВП на душу** (*GDP_per_cap*):  
   - *H₁*: Чем выше ВВП → тем больше стартапов.  

4. **Безработица** (*Unemploym_rate*):  
   - *H₁*: Чем выше безработица → тем меньше стартапов.  

5. **Инфляция** (*Inflation_rate*):  
   - *H₁*: Чем выше инфляция → тем меньше стартапов.
  
   ```python
   import statsmodels.api as sm

X = df_final[['GDP_per_cap', 'Unemploym_rate', 'Inflation_rate', 'Gross_monthly_wage','Business_complexity_inx', 'Corporate_Tax_Rate', 'Venture_cap_inv']]
y = df_final['Total_Startup_Output']

X = sm.add_constant(X)

# Построение модели линейной регрессии
model = sm.OLS(y, X).fit()

# Вывод результатов модели
print(model.summary())
```
![Регрессия](https://github.com/ver369/Group_Project_Analysis_of_Country_Attractiveness_for_Startups/blob/main/%D0%A0%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F.png)
### **Результаты анализа значимости переменных**  

#### **Статистически значимые факторы:**  
1. **Венчурные инвестиции** (*Venture_cap_inv*):  
   - **Сильное влияние** (p < 0.001, коэффициент +0.22).  
   - Каждые **+$1 млн инвестиций → +0.22 стартапа**.  
   - *Ключевой драйвер* (США, Китай, Великобритания).  

2. **Корпоративные налоги** (*Corporate_Tax_Rate*):  
   - **Пограничная значимость** (p = 0.072, коэффициент -5.1).  
   - Повышение налога на **1% → потенциальное снижение на ~5 стартапов**.  
   - Требуется дополнительная проверка.  

#### **Незначимые факторы (p > 0.05):**  
- ВВП на душу, безработица, инфляция, зарплата, сложность ведения бизнеса.  

#### **Качество модели:**  
- **R² = 0.943** (модель объясняет **94.3%** дисперсии).  
- **F-тест значим** (p = 1.15e-30).  

### **Итоговые выводы:**  
1. **Главный фактор:** Венчурные инвестиции — критически важны для роста стартапов.  
2. **Вторичный фактор:** Низкие корпоративные налоги *могут* поддерживать стартап-среду.  
3. **Прочие экономические показатели** (ВВП, инфляция и др.) в данной модели **не влияют** на число стартапов.  































