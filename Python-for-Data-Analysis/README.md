# Python_for_Data_Analysis
파이썬 라이브러리를 활용한 데이터 분석
```python3
```


## Week 2
### Pandas

```python3
import pandas as pd

# 3명의 학생 리스트
students = ['Alice', 'Jack', 'Molly']

# pandas Series 함수는 인덱스 0부터 자동으로 할당
s=pd.Series(students)
```

![스크린샷 2022-03-12 오전 11 51 55](https://user-images.githubusercontent.com/59719632/158000992-61dd02f1-3f49-4b37-8fec-60b10cdceb31.png)

* Series에 전달하는 리스트가 정수형이면 Series 객체 타입이 int로 저장된다

* Series에 정수형 리스트를 만들고 None을 포함시키면 None대신 NaN을 저장하고 객체 타입을 float으로 받아들인다.
  - 정수는 부동 소수로 바뀔 수 있기 때문이다.
* NaN != None
  - NaN의 존재를 테스트하려면 np.isnan() 같은 특수한 함수를 사용해야한다.

```python3
import numpy as np

np.nan=None
np.isnan(np.nan)
```

* Series는 딕셔너리 데이터에서 직접 생성될 수도 있다.

```python3
students_scores = {'Alice': 'Physics',
                   'Jack': 'Chemistry',
                   'Molly': 'English'}
s = pd.Series(students_scores) # dictionary의 key 값이 index가 된다.
```

![스크린샷 2022-03-12 오후 12 03 55](https://user-images.githubusercontent.com/59719632/158001303-ef1b9b64-15f2-4c75-828d-6efd866e9c4a.png)

* 튜플

```python3
students = [("Alice","Brown"), ("Jack", "White"), ("Molly", "Green")]
pd.Series(students) # value에 튜플이 저장된다.
```

![스크린샷 2022-03-12 오후 12 06 25](https://user-images.githubusercontent.com/59719632/158001384-508924e9-3620-4d8b-862a-75a12c51aec4.png)

```python3
s = pd.Series(['Physics', 'Chemistry', 'English'], index=['Alice', 'Jack', 'Molly'])
s # 인덱스를 지정할 수 있다.
```

* key에 없는 값을 index로 지정하면 value가 NaN인 row가 추가 된다. index 지정값에서 제외된 값은 무시한다.

```python3
students_scores = {'Alice': 'Physics',
                   'Jack': 'Chemistry',
                   'Molly': 'English'}
# When I create the series object though I'll only ask for an index with three students, and
# I'll exclude Jack
s = pd.Series(students_scores, index=['Alice', 'Molly', 'Sam'])
s
```

![스크린샷 2022-03-12 오후 12 08 17](https://user-images.githubusercontent.com/59719632/158001447-0e7470a0-73da-4f88-b86e-6e0a768ab2ff.png)


### Querying a Series

```python3
import pandas as pd

students_classes = {'Alice': 'Physics',
                   'Jack': 'Chemistry',
                   'Molly': 'English',
                   'Sam': 'History'}
s = pd.Series(students_classes)

s.iloc[3] # 숫자 인덱스로 접근하는 방법, 4번째 인덱스의 value
s.loc['Molly'] # key 값으로 접근하는 방법 
s[3] # s.iloc[3] 와 동일한 동작을 한다.
s['Molly'] # s.loc['Molly] 와 동일한 동작을 한다.

# Series 자체에서 인덱싱 연산자를 사용할 때는 iloc 또는 loc을 사용하는 것이 지향된다.

class_code = {99: 'Physics',
              100: 'Chemistry',
              101: 'English',
              102: 'History'}
s = pd.Series(class_code)
s[0] # 에러 발생한다. 따라서 이 경우 s.iloc[0] 또는 s.loc[99] 를 사용해야한다.
```

```python3
grades = pd.Series([90, 80, 70, 60])

total = 0
for grade in grades: # 느리다.
    total+=grade
print(total/len(grades))

import numpy as np # numpy를 사용하면 훨씬 빠르게 계산할 수 있다.

# Then we just call np.sum and pass in an iterable item. In this case, our panda series.

total = np.sum(grades)
print(total/len(grades))
```

```python3
# First, let's create a big series of random numbers. This is used a lot when demonstrating 
# techniques with Pandas
numbers = pd.Series(np.random.randint(0,1000,10000))

# Now lets look at the top five items in that series to make sure they actually seem random. We
# can do this with the head() function
numbers.head() # 상위 5개의 아이템을 반환

%%timeit -n 100 # ipython interpreter function
total = 0
for number in numbers:
    total+=number

total/len(numbers) # 1.19 ms ± 91.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%%timeit -n 100
total = np.sum(numbers)
total/len(numbers) # 67 µs ± 1.65 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

```python3
numbers+=2 # 모든 value에 2씩 더하기 , pandas에서 반복문을 사용하고 있다면 최선의 방법인지 다시 생각해 볼 필요가 있다.
```

```python3
%%timeit -n 10
# we'll create a blank new series of items to deal with
s = pd.Series(np.random.randint(0,1000,1000))
# And we'll just rewrite our loop from above.
for label, value in s.iteritems(): # 136 ms ± 6.22 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    s.loc[label]= value+2
    
%%timeit -n 10
# We need to recreate a series
s = pd.Series(np.random.randint(0,1000,1000))
# And we just broadcast with +=
s+=2 # 264 µs ± 30.2 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

```python3
# Here's an example using a Series of a few numbers. 
s = pd.Series([1, 2, 3])

# We could add some new value, maybe a university course
s.loc['History'] = 102

```

* Pandas는 기본적으로 기존 Series를 바꾸는 것이 아닌 새로운 Series 객체를 반환한다.

```python3
students_classes = pd.Series({'Alice': 'Physics',
                   'Jack': 'Chemistry',
                   'Molly': 'English',
                   'Sam': 'History'})
                   
kelly_classes = pd.Series(['Philosophy', 'Arts', 'Math'], index=['Kelly', 'Kelly', 'Kelly'])

all_students_classes = students_classes.append(kelly_classes)

students_classes # append는 기존 객체를 바꾸는 것이 아닌 새로운 객체를 반환한다.
```
### DataFrame

```python3
import pandas as pd

record1 = pd.Series({'Name': 'Alice',
                        'Class': 'Physics',
                        'Score': 85})
record2 = pd.Series({'Name': 'Jack',
                        'Class': 'Chemistry',
                        'Score': 82})
record3 = pd.Series({'Name': 'Helen',
                        'Class': 'Biology',
                        'Score': 90})
                        
df = pd.DataFrame([record1, record2, record3],
                  index=['school1', 'school2', 'school1'])
df.head()
```

<img width="226" alt="스크린샷 2022-03-12 오후 2 24 25" src="https://user-images.githubusercontent.com/59719632/158005090-4d446da0-41cc-4d96-827d-c05399eab28c.png">

```python3
# Series 대체 메서드로 dictinary를 사용할 수 있다.
# 각 딕셔너리는 행을 나타낸다.
students = [{'Name': 'Alice',
              'Class': 'Physics',
              'Score': 85},
            {'Name': 'Jack',
             'Class': 'Chemistry',
             'Score': 82},
            {'Name': 'Helen',
             'Class': 'Biology',
             'Score': 90}]

# Then we pass this list of dictionaries into the DataFrame function
df = pd.DataFrame(students, index=['school1', 'school2', 'school1'])
# And lets print the head again
df.head()
```
<img width="224" alt="스크린샷 2022-03-12 오후 2 25 46" src="https://user-images.githubusercontent.com/59719632/158005151-09bae451-f6f2-40c2-b660-26d90d476449.png">

* DataFrame은 2차원이기 때문에 단일 값을 loc 인덱싱 연산자로 전달하면 반환할 행이 하나만 있는 경우 Series가 반환된다.

```python3
df.loc['school2'] # 새로운 Series 객체 반환
type(df.loc['school2']) 

df.loc['school1'] # 새로운 DataFrame 객체 반환
type(df.loc['school1']) 

df.loc['school1', 'Name']  # Name 피처만 보고 싶을 때 [row,col]

# DataFrame Transpose
df.T
df.T.loc['Name'] # df.loc['school1', 'Name'] 과 비슷한 결과
```
<img width="250" alt="스크린샷 2022-03-12 오후 4 25 07" src="https://user-images.githubusercontent.com/59719632/158008437-70868f78-607d-4305-b85e-74dd8c43e501.png">

* Pandas DataFrame의 columns는 항상 이름을 가진다.
  - 열 이름으로 .loc를 사용할 경우 키 오류가 생길 수 있다.

* Chaining은 가능한 지양해야한다. 다른 뷰를 얻으려 할 때 pandas는 사본을 반환하기 때문이다.

```python3
# Here's an example, where we ask for all the names and scores for all schools using the .loc operator.
df.loc[:,['Name', 'Score']] # loc은 슬라이싱이 가능하다. 
```

```python3
df.drop('school1') # drop 함수는 주어진 행이 제거된 DataFrame의 사본을 반환한다.
df # 원본은 바뀌지 않는다.
```

```python3
copy_df = df.copy()
# Now lets drop the name column in this copy
copy_df.drop("Name", inplace=True, axis=1) # axis=1을 하면 열 축을 나타낸다. 0:x, 1:y, 2:z 축
copy_df

# There is a second way to drop a column, and that's directly through the use of the indexing 
# operator, using the del keyword. This way of dropping data, however, takes immediate effect 
# on the DataFrame and does not return a view.
del copy_df['Class'] # 뷰를 반환하지 않고 원본을 직접 삭제한다..

copy_df
# DataFrame에 새로운 열을 추가하는 것은 인덱싱 연산자를 사용해 일부 값을 할당하는 것만큼 쉽다.
df['ClassRanking']=None
```
### DataFrame Indexing and Loading 

```python3
import pandas as pd

# Pandas mades it easy to turn a CSV into a dataframe, we just call read_csv()
df = pd.read_csv('datasets/Admission_Predict.csv')

# And let's look at the first few rows
df.head()
```

<img width="578" alt="스크린샷 2022-03-12 오후 6 14 58" src="https://user-images.githubusercontent.com/59719632/158011997-f802c44b-9078-46a0-a266-eb028779fdb7.png">

```python3
# 열의 이름을 지정한 값으로 바꿀 수 있다.
new_df=df.rename(columns={'GRE Score':'GRE Score', 'TOEFL Score':'TOEFL Score',
                   'University Rating':'University Rating', 
                   'SOP': 'Statement of Purpose','LOR': 'Letter of Recommendation',
                   'CGPA':'CGPA', 'Research':'Research',
                   'Chance of Admit':'Chance of Admit'})
new_df.head()

# 열 이름이 바뀌지 않는 경우 열 이름을 잘못 쓰지 않았는지 확인해야한다.
new_df.columns

# 열의 이름이 부분만 바꿀 수 있다.
new_df=new_df.rename(columns={'LOR ': 'Letter of Recommendation'})
new_df.head()

# 공백이 몇 개 있을지 모르기 때문에 공백을 다 지우는 mapper을 사용할 수 있다.
new_df=new_df.rename(mapper=str.strip, axis='columns')
# Let's take a look at results
new_df.head()

# rename 함수는 원본을 바꾸는 함수가 아닌 복사본을 전달한다.
```

```python3
# 원본의 열 이름을 직접 바꿀 수도 있다.

# As an example, lets change all of the column names to lower case. First we need to get our list
cols = list(df.columns)
# Then a little list comprehenshion
cols = [x.lower().strip() for x in cols]
# 원본에 덮어 쓴다.
df.columns=cols
# And take a look at our results
df.head()

```

### Querying a DataFrame
* Boolean Masking
  - Boolean Masking은 데이터 분석에서 자주 쓰이고 중요하다. 이를 사용하면 우리가 정의한 기준에 기반한 데이터를 선택할 수 있다.

```python3
admit_mask=df['chance of admit'] > 0.7 # 0.7 보다 큰 값은 True, 나머지는 False로 저장
admit_mask # Boolean Series로 반환
```
![스크린샷 2022-03-12 오후 9 55 41](https://user-images.githubusercontent.com/59719632/158018767-ed39342c-eb4a-46a9-97f2-cd6a96802b61.png)

```python3
# False인 값들을 숨길 수 있다.
df.where(admit_mask).head() # 조건을 충족하지 못한 행은 NaN 데이터를 가진다. 삭제 되는 것은 아님

df.where(admit_mask).dropna().head() # dropna()를 사용하면 NaN 데이터를 가진 행을 삭제한다.

# where과 dropna 기능을 같이 쓸 수 있는 약식 구문을 주로 사용한다.
df[df['chance of admit'] > 0.7].head()
```

```python3
# Python은 두 Series를 비교하는 방법을 모른다.
(df['chance of admit'] > 0.7) and (df['chance of admit'] < 0.9) # 에러 메세지 발생

(df['chance of admit'] > 0.7) & (df['chance of admit'] < 0.9) # & 연산자를 사용해서 더해야한다.
# 연산자 순서를 주의해야 하는데 & 앞뒤 Series객체에 ()로 구분해줘야한다. 안쓰면 에러

df['chance of admit'].gt(0.7) & df['chance of admit'].lt(0.9) # pandas 내장함수를 사용할 수 있다. gt, lt
df['chance of admit'].gt(0.7).lt(0.9) # more readable
```

### Indexing DataFrames
* set_index 함수는 현재의 인덱스를 유지하지 않는다. 현재의 인덱스를 유지하고 싶다면 수동으로 새로운 열을 생성하고 해당 열로 인덱스 값을 복사해두어야한다.
```python3
df['Serial Number'] = df.index # 새로운 열 생성 후 인덱스 복사
# Then we set the index to another column
df = df.set_index('Chance of Admit ') # 기존의 열 이름이 인덱스 열의 이름이 되어버린다.
df.head()

```
![스크린샷 2022-03-12 오후 10 14 06](https://user-images.githubusercontent.com/59719632/158019358-8793e9fa-b035-408c-9b76-949c9e8c51a4.png)

```python3
df = df.reset_index() # 인덱스 열을 정리한다.
df.head()
```
![스크린샷 2022-03-12 오후 10 19 53](https://user-images.githubusercontent.com/59719632/158019510-591461ac-b3ee-49ea-aa2f-b10d45553c67.png)

* Pandas에는 멀티레벨 인덱싱이라는 기능이 있다.
  - 관계형 데이터 베이스 시스템의 합성 키와 유사하다.
 
```python3
# Here we can run unique on the sum level of our current DataFrame 
df['SUMLEV'].unique() # 데이터 베이스의 DISTINCT 와 유사하다. 중복되지 않은 값을 반환한다.
```
![스크린샷 2022-03-12 오후 10 24 55](https://user-images.githubusercontent.com/59719632/158019684-ab730c5b-8c6a-4bd3-b4ee-dcb624e69176.png)

```python3
# Let's exclue all of the rows that are summaries 
# at the state level and just keep the county data. 
df=df[df['SUMLEV'] == 50]
df.head()

# 원하는 column만 보도록 df 재정의
columns_to_keep = ['STNAME','CTYNAME','BIRTHS2010','BIRTHS2011','BIRTHS2012','BIRTHS2013',
                   'BIRTHS2014','BIRTHS2015','POPESTIMATE2010','POPESTIMATE2011',
                   'POPESTIMATE2012','POPESTIMATE2013','POPESTIMATE2014','POPESTIMATE2015']
df = df[columns_to_keep]
df.head()

# STNAME 안에 카운티가 어떻게 구성됐는지 파악할 수 있다.
df = df.set_index(['STNAME', 'CTYNAME'])
df.head()
```
![스크린샷 2022-03-12 오후 10 30 06](https://user-images.githubusercontent.com/59719632/158019844-a5a923b1-d58e-49d8-a23c-8bd57efaa8a1.png)

```python3
# 멀티레벨 인덱싱에서는 쿼리할 레벨에 따라 인자를 순서대로 제공해야 한다. 가장 바깥쪽의 열은 레벨 0, 안쪽으로 갈 수록 레벨이 커진다.
# 두 카운티를 비교하고 싶다면 쿼리하려는 인덱스를 설명하는 튜플 리스트를 loc 속성에 전달하면 된다.
df.loc[ [('Michigan', 'Washtenaw County'),
         ('Michigan', 'Wayne County')] ]
```

### Missing Values(결측치)
* 누락된 데이터는 결측 변수를 예측하는데 사용될 수 있는 다른 변수가 있는 경우 '무작위 결측(Missing at Random, MAR)' 이라고하고, 다른 변수와 관계 없는 경우 '완전 무작위 결측(Missing Completely at Random, MCAR)'이라고 한다.

```python3
mask=df.isnull() # DataFrame 전체에 대해 Boolean mask를 생성한다.
mask.head(10) # NaN 값에 해당하는 데이터는 True로 반환된다.
```
![스크린샷 2022-03-12 오후 10 39 10](https://user-images.githubusercontent.com/59719632/158020226-b09c79ed-f174-4e35-b5b6-9379b5ac2426.png)

```python3
df.dropna().head(10) # dropna()로 데이터가 누락된 모든 행을 삭제할 수도 있다.

# So, if we wanted to fill all missing values with 0, we would use fillna
df.fillna(0, inplace=True) # 결측 값을 다른 값으로 대체할 수도 있다. 꽤 자주쓰는 방법이다.
df.head(10) # inplace 속성은 원본을 바꾸는 대신 사본을 반환한다.
```

```python3
# If we load the data file log.csv, we can see an example of what this might look like.
df = pd.read_csv("datasets/log.csv")
df.head(20)
```

```python3
# 결측치를 채우는 방법에 ffill과 bfill이 있다. ffill은 이전 행에 있던 값으로 채우는 것이고, bfill은 다음 행에 나오는 유효한 값으로 채운다.
# 우리가 원하는 효과를 얻기 위해서는 데이터를 정렬해야한다는 것을 알아야 한다.
# In Pandas we can sort either by index or by values. Here we'll just promote the time stamp to an index then
# sort on the index.
df = df.set_index('time') 
df = df.sort_index() # 인덱스로 정렬
df.head(20)

df = df.reset_index() # 기존의 인덱스 열을 승격시키고 새로운 인덱스 열 추가
df = df.set_index(['time', 'user']) # 최상위 레벨은 시간, 그 다음 레벨은 사용자로 설정
df

# 한 명령문으로 모든 결측값을 고칠 필요는 없다.
# several approaches: value-to-value, list, dictionary, regex
df = df.fillna(method='ffill')
df.head()
```

```python3
df = pd.DataFrame({'A': [1, 1, 2, 3, 4],
                   'B': [3, 6, 3, 8, 9],
                   'C': ['a', 'b', 'c', 'd', 'e']})
# value-to-value approach
df.replace(1, 100)

# list approach
df.replace([1, 3], [100, 300]) # 1을 100으로 3을 300으로 교체
```

```python3
df = pd.read_csv("datasets/log.csv")
df.head(20)

# regex approach
df.replace(to_replace=".*.html$", value="webpage", regex=True) # 첫 번째 parameter를 regex 패턴으로 만든다.

# 데이터 프레임에 통계쩍 함수를 사용할 때 이 함수들은 보통 결측값을 무시한다. 예를 들어 평균값을 계산할 때 기본 numpy 함수는 그 결측값들을 무시할 수 있다.
# 결측값이 제외됐다는 사실은 알고 있어야한다.
```
<p align="center">
![스크린샷 2022-03-12 오후 10 59 16](https://user-images.githubusercontent.com/59719632/158020925-8a0e45e8-3d85-40ee-b0df-5f19d989246a.png) ![스크린샷 2022-03-12 오후 10 59 29](https://user-images.githubusercontent.com/59719632/158020930-ddcd2dae-2e2a-4951-97a7-aa191a16d62a.png)</p>

```python3
```

```python3
```


