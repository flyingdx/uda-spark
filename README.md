# uda-spark
# Sparkify项目报告
## 项目简介
该项目是优达毕业项目。数据集是一个音乐服务的用户日志，包含了用户信息，歌曲信息，用户活动，时间戳等。大小128M。需要通过数据集中信息，预测出可能流失的用户，以便后续对相应用户采取挽留措施


## 项目思路
为了预测可能流失的用户，对日志进行分析，探索并提取与用户流失相关的变量；根据变量，使用Spark建立机器学习模型进行预测。具如下：
1.加载所需的库并实例化
2.加载与清洗数据
3.探索性数据分析
4.构建预特征
5.建模预测
6.结论汇总

## 项目实现
### 1.加载所需的库并实例化
#### 加载所需的库：
项目会用到一下库
1、pyspark.sql：spark中进行类似SQL中的操作
2、pyspark.ml：spark中进行机器学习
3、pandas 、numpy：对dataframe进行操作
4、matplotlib、seaborn： 绘图
5、time：记录代码块运行时间的库

#### 实例化
```python
spark=SparkSession.builder.getOrCreate()
```

### 2.加载与清洗数据
#### 加载数据集
（bz2的待会处理。。。。。。。。)
数据集是json格式，使用以下方法加载数据
```python
spark.read.json('mini_sparkify_event_data.json')
```
#### 评估数据集
对数据集评估思路是：先查看整体情况，再查看重点希望了解的列的情况。
1、查看整体情况的方法如下：
（1）查看数据前几行的值，了解数据集概况，对数据集有整体认识。主要使用了.show()函数
（2）查看列数、每列的名称以及类型，并结合以上了解每列的含义。主要使用.printSchema()函数
（3）查看数据行数。主要使用.count()函数
通过以上观察，我们可了解到：数据集共有286500行,18列；主要包含了用户信息，歌曲信息，用户活动，时间戳等信息。变量含义如下：
```python
 |-- artist: string (歌手)
 |-- auth: string (含义暂不明确)
 |-- firstName: string (名字)
 |-- gender: string (性别)
 |-- itemInSession: long (含义暂不明确)
 |-- lastName: string (姓氏)
 |-- length: double (听歌时长)
 |-- level: string (等级)
 |-- location: string (地区)
 |-- method: string (具体含义暂不明确)
 |-- page: string (页面)
 |-- registration: long (注册时间)
 |-- sessionId: long (页面ID)
 |-- song: string (歌名)
 |-- status: long (含义暂不明确)
 |-- ts: long (结合这个是日志信息，推测是当前事件时间)
 |-- userAgent: string (用户使用平台信息)
 |-- userId: string (用户ID)

```

对某一列进行查看的方法如下：
2、通过dropDuplicates()去重查看唯一值，同时通过show(5)展示
```python
df.select('userId').dropDuplicates().sort('userId').show(5)
```
```python
+------+
|userId|
+------+
|      |
|    10|
|   100|
|100001|
|100002|
+------+
only showing top 5 rows
```
通过对各列进行查看，我们发现：
1、userId列存在非NA的空值，需要删除
2、registration、ts应该是时间戳，直观上无逻辑上含义，列名也不直观易懂。需新建两列，重命名列名，并转换为日期格式。因后续涉及构造变量，原long类型数据列暂时保留。

#### 清理数据集
##### 处理空值
先通过dropna()，处理userId列空值；

```python
df_valid=df.dropna(how="any",subset=["userId","sessionId"])
```
再通过filter(), 去除有空字符的行.

```python
df_valid=df_valid.filter(df["userId"]!="")
```
##### 转换时间戳格式
建立转换用的lambda函数，使用fromtimestamp，将时间戳转换成字符串日期时间。
```python
convert_ts=udf(lambda x:datetime.datetime.fromtimestamp(x/1000.0).strftime("%Y-%m-%d %H:%M:%S"))
df_valid=df_valid.withColumn('event_time',convert_ts('ts'))
df_valid=df_valid.withColumn('registration_time',convert_ts('registration'))
```
### 3.探索性数据分析
##### 建立注销客户的标签
项目提示使用churn作为模型的标签, 并且建议使用Cancellation Confirmation事件来定义客户流失.。
1、标记注销事件：新建一列churn_event列，标记page中的Cancellation Confirmation事件
2、标记注销用户：新建一列churn列，标记注销用户。具体方法是，只要用户churn_event中有标记注销，该用户所有的churn列均标记为注销

##### 建立注销客户的标签
定义好客户流失后, 进行探索性数据分析, 观察留存用户和流失用户的行为。绘图观察主要使用了直方图、箱线图、小提琴图。相比箱线图，小提琴图更能看出密度分布
1、注销与用户听歌数量的关系

```python
#提取NextSong数据，查看用户听歌数量的分布
lifetime_songs=df_valid.where('page=="NextSong"').groupby(['userId','churn']).count().toPandas()
```

```python
#绘制小提琴图
ax=sns.violinplot(data=lifetime_songs,x='churn',y='count')
```
* 相比于非注销用户，注销用户听歌的数量较少，且数量的分布相对集中，其小提琴图形相对扁平

2、是否注销与单次听歌数量关系

```python
#提取NextSong数据，观察同一sessionId下听歌平均数量的分布
avg_songs_listened=df_valid.where('page=="NextSong"').groupby(['churn', 'userId' ,'sessionId']).count().groupby(['churn','userId']).agg({'count':'avg'}).toPandas()
```

```python
#绘制小提琴图
ax=sns.violinplot('churn',y='avg(count)',data=avg_songs_listened)
```
* 相比于非注销用户，大部分注销用户同一sessionId下听歌的数量较少

3、是否注销与用户点赞量关系

```python
#提取humbs Up数据，观察用户点赞数量分布
a=df_valid.where('page=="Thumbs Up"').groupby(['userId','churn']).count().toPandas()
```

```python
#绘制小提琴图
ax=sns.violinplot(data=a,x='churn',y='count')
```
* 相比于非注销用户，注销用户点赞的数量较少，且数量的分布相对集中

4、是否注销与性别关系

```python
#提取性别与用户ID列，观察注销与性别间关系
gender_churn=df_valid.dropDuplicates(["userId","gender"]).groupby(["churn","gender"]).count().toPandas()
```

```python
#绘制直方图
ax=sns.barplot(x='gender',y='count',hue='churn',data=gender_churn)
```
*  男性用户注销账户的绝对人数以及比例均比女性大

5、注销与用户存留天数关系
```python
#通过事件时间与注册时间差，计算截止事件发生，已经经历了多少时间；取其中最大的时间，便是用户注册至最后一次食用的时间，也即存留时间；最后将时间单位转换为天
user_lifetime=df_valid.select('userId','registration','ts','churn').withColumn('lifetime',(df_valid.ts-df_valid.registration)).groupBy('userId','churn').agg({'lifetime':'max'}).withColumnRenamed('max(lifetime)','lifetime').select('userId','churn',(col('lifetime')/1000/3600/24).alias('lifetime')).toPandas()
```

```python
#绘制箱线图
ax=sns.boxplot(data=user_lifetime,x='churn',y='lifetime') 
```
* 注销用户的存留天数更少

### 4.构建预特征
#### 变量选择
结合经验及以上的分析，构建以下变量：
1、听歌情况方面的变量：
（1）**用户听歌数量**：听歌数量越大，说明用户愿意使用该服务，注销几率越小。以上绘图分析也显示：注销用户听歌数量较未注销的少
（2）**用户单次（同一sessionId）听歌平均数量**：单次听歌数量越大，说明用户愿意使用该服务，注销几率越小。以上绘图分析也显示：注销用户单次听歌平均数量较未注销的少
（3）**播放的歌手数量**：播放过的歌手数量越多，侧面说明用户听歌越多，越愿意使用该服务，注销几率越小。
（4）**歌曲时长总量**：听歌时长越长，说明用户倾向于使用该服务，注销几率越小。

2、从page中提取动作建立变量：
（1）点赞量：点赞越多，说明用户喜欢该服务，注销几率越小。以上绘图分析也显示：注销用户点赞数量较未注销的少
（2）差评量：逻辑与点赞量恰好相反
（3）添加播放列表量：用户将歌曲加进播放列表，一般可说明用户喜欢该音乐；添加的量越多，用户愿意使用该服务的可能性越大，注销可能性越小。
（4）添加好友量：添加好友量越多，说明用于越愿意在改服务中交友分享，注销几率越小。

3、其他变量：
（1）性别：以上绘图分析显示：男性用户注销的数量较女性多。推测是改服务更能吸引女性
（2）用户存留天数：一般来说，服务越吸引了用户，则用户存留越久，注销几率越小。以上绘图分析也显示：注销用户存留天数较未注销的短

#### 变量提取
1用户听歌数量
获取每个用户听过歌曲的歌名信息计数，获得用户听歌数量
```python
f2=df_valid.select('userID','song').groupBy('userID').count().withColumnRenamed('count','total_songs')
```

2用户单次（同一sessionId）听歌平均数量
获取每个sessionId点击页面NextSong数量信息并计数，并按用户求均值，可获得用户单次（同一sessionId）听歌平均数量
```python
f8=df_valid.where('page=="NextSong"').groupBy('userId','sessionId').count().groupBy(['userId']).agg({'count':'avg'}).withColumnRenamed('avg(count)','avg_songs_played')
```

3播放的歌手数量
#获取每个用户点击页面NextSong时的artist信息并计数，可获得用户听过的歌手数量
```python
f10=df_valid.filter(df_valid.page=="NextSong").select("userID","artist").dropDuplicates().groupby("userId").count().withColumnRenamed("count","artist_count")
```

4歌曲时长总量
每个用户播放时长累加
```python
f7=df_valid.select('userID','length').groupBy('userID').sum().withColumnRenamed('sum(length)','listen_time')
```

5点赞量
获取每个用户点击页面Thumbs Up的数量信息计数，可获得用户点赞量
```python
f3=df_valid.select('userID','page').where(df_valid.page=='Thumbs Up').groupBy('userID').count().withColumnRenamed('count','num_thumb_up')
```

6差评量
获取每个用户点击页面Thumbs Down的数量信息计数，可获得用户差评量
```python
f4=df_valid.select('userId','page').where(df_valid.page=='Thumbs Down').groupBy('userId').count().withColumnRenamed('count','num_thumb_down')
```

7添加播放列表量
获取每个用户点击页面Add to Playlist的数量信息计数，可获得用户添加进播放列表数量
```python
f5=df_valid.select('userID','page').where(df_valid.page=='Add to Playlist').groupBy('userID').count().withColumnRenamed('count','add_to_playlist')
```

8添加好友量
```python
#获取每个用户点击页面Add Friend的数量信息计数，可获得用户添加好友书量
f6=df_valid.select('userID','page').where(df_valid.page=='Add Friend').groupBy('userID').count().withColumnRenamed('count','add_friend')
```

9性别
取gender列，把F、M变量转为0、1，方便模型计算
```python
f9=df_valid.select("userId","gender").dropDuplicates().replace(['paid','free'],['0','1'],'gender').select('userId',col('gender').cast('int'))
```

10用户存留天数
用注册时间（registration）与动作发生时间（ts）相减，并取出最长的时间，便是用户存留天数
```python
f1=df_valid.select('userId','registration','ts').withColumn('lifetime',(df_valid.ts-df_valid.registration)).groupBy('userId').agg({'lifetime':'max'}).withColumnRenamed('max(lifetime)','lifetime').select('userId',(col('lifetime')/1000/3600/24).alias('lifetime'))
```
整理标签列
后续建模时，真实标记列默认为label列，将churn列重命名为label

```python
label=df_valid.select('userId',col('churn').alias('label')).dropDuplicates()
```
#### 变量聚合
通过join将变量连接，同时用0填充为空数据；此外，userID列是索引非变量，合并后需删除
```python
data=f1.join(f2,'userID','outer')\
    .join(f3,'userID','outer')\
    .join(f4,'userID','outer')\
    .join(f5,'userID','outer')\
    .join(f6,'userID','outer')\
    .join(f7,'userID','outer')\
    .join(f8,'userID','outer')\
    .join(f9,'userID','outer')\
    .join(f10,'userID','outer')\
    .join(label,'userID','outer')\
    .drop('userID')\
    .fillna(0)
```

### 5.建模预测
模型选用逻辑回归、支持向量机与随机森林。根据项目说明，选用 F1 score 作为主要优化指标。
#### 准备数据
将数据转换为向量形式，标准化，并分成训练集、测试集和验证集
```python
#用VectorAssembler将数据集转换为可供模型计算的结构（向量形式）
cols=["lifetime","total_songs","num_thumb_up","num_thumb_down","add_to_playlist","add_friend","listen_time","avg_songs_played","gender","artist_count"]
assembler=VectorAssembler(inputCols=cols,outputCol="NumFeatures")
data=assembler.transform(data)

#用StandardScaler标准化数据
scaler=StandardScaler(inputCol="NumFeatures",outputCol="features",withStd=True)
scalerModel=scaler.fit(data)
data=scalerModel.transform(data)

#按60%，40%，40%比例拆分为训练集、测试集和验证集
train,validation,test=data.randomSplit([0.6,0.2,0.2],seed=42)
```
#### 模型选择
**模型选择思路**
* 以全0/全1预测作为基线，机器学习算法的分数应该比全0/全1预测更高
* 选用逻辑回归、支持向量机、随机森林进行对比，这几个模型一般不需要很多参数调整就可以达到不错的效果。他们的优缺点如下：
1、逻辑回归：优点：计算速度快，容易理解；缺点：容易产生欠拟合
2、支持向量机：数据量较小情况下解决机器学习问题，可以解决非线性问题。缺点：对缺失数据敏感
3、随机森林：优点：有抗过拟合能力。通过平均决策树，降低过拟合的风险性。缺点：大量的树结构会占用大量的空间和利用大量时间

**模型训练**
* Baseline Model（全1/全0）

```python
#对测试集进行预测，预测全为1
results_base_all_1=test.withColumn('prediction',lit(1.0))#prediction列全是1
evaluator=MulticlassClassificationEvaluator(predictionCol='prediction')
print('Test set metrics')
print('Accuracy:{}'.format(evaluator.evaluate(results_base_all_1,{evaluator.metricName:"accuracy"})))
print('F-1 Score:{}'.format(evaluator.evaluate(results_base_all_1,{evaluator.metricName:"f1"})))
```

```python
#对测试集进行预测，预测全为0
results_base_all_0=test.withColumn('prediction',lit(0.0))#'prediction列全是0
evaluator=MulticlassClassificationEvaluator(predictionCol='prediction')
print('Test set metrics')
print('Accuracy:{}'.format(evaluator.evaluate(results_base_all_0,{evaluator.metricName:"accuracy"})))
print('F-1 Score:{}'.format(evaluator.evaluate(results_base_all_0,{evaluator.metricName:"f1"})))
```
* Random Forest

```python
#创建并训练模型，通过time()记录训练时间
rf=RandomForestClassifier()#初始化
start=time()#开始时间
model_rf=rf.fit(train)#训练
end=time()#结束时间
print('The training process took{} second'.format(end-start))

#验证模型效果
results_rf=model_rf.transform(validation)#验证集上预测
evaluator=MulticlassClassificationEvaluator(predictionCol="prediction")#评分器
print('Random Forest Metrics:')
print('Accuracy:{}'.format(evaluator.evaluate(results_rf,{evaluator.metricName:"accuracy"})))#计算Accuracy
print('F-1 Score:{}'.format(evaluator.evaluate(results_rf,{evaluator.metricName:"f1"})))#计算F-1 Score
```

* LogisticRegression、LinearSVC
逻辑回归、支持向量机模型代码与随机森林与结构基本一致，主要是需要将代码改为对应模型外；以及设置迭代次数（均设置为10次）

**计算结果**
* LogisticRegression模型：Accuracy为0.7959；F-1 Score为0.7871；耗时87s
* LinearSVC模型：Accuracy为0.7959；F-1 Score为0.7054；耗时170s
* Random Forest模型：Accuracy0.8163；F-1 Score0.7912；耗时150s
Random Forest的Accuracy及F-1 Score均最高，LogisticRegression耗时最小。考虑到Random Forest的训练耗时与LogisticRegression的训练耗时整体来说相差并非十分大，而我们希望获得效果更好的模型，故选用Random Forest模型，并通过调节模型参数获取更优模型

#### 模型调优
**调优思路**
* 如上所述，选用Random Forest进行调优。
* 使用3折交叉验证及参数网络对模型进行调优。
* 因为流失顾客数据集很小，Accuracy很难反映模型好坏，根据建议选用 F1 score 作为优化指标。

**调整代码**
原代码的基础上，对训练部分的代码做调整。以下是主要调整部分：
```python
rf=RandomForestClassifier()#初始化模型
f1_evaluator=MulticlassClassificationEvaluator(metricName='f1')#选用f1-score来衡量优劣
paramGrid=ParamGridBuilder().addGrid(rf.maxDepth,[3,5]).addGrid(rf.numTrees,[20,50]).build()#建立可选参数的网络，主要对maxDepth、numTrees调整
crossval_rf=CrossValidator(estimator=rf,
        estimatorParamMaps=paramGrid,
        evaluator=f1_evaluator,
        numFolds=3)#3折交叉验证
cvModel_rf=crossval_rf.fit(train)#训练
```
**调整结果**
比调优前后的模型在验证集上预测结果：调优前Accuracy0.8163；F-1 Score0.7912；调优后Accuracy为0.8235，F-1 Score为0.796，分别提升了XXX%及XXX%。
（跑一次再补------------------------）


#### 对测试集预测
使用优化前及优化后的模型，同时对测试集进行预测

```python
#使用未优化模型预测
results_final=model_rf.transform(test)
evaluator=MulticlassClassificationEvaluator(predictionCol="prediction")
print('Test set metricd:')
print('Accuracy:{}'.format(evaluator.evaluate(results_final,{evaluator.metricName:"accuracy"})))
print('F-1 Score:{}'.format(evaluator.evaluate(results_final,{evaluator.metricName:"f1"})))
```

```python
#使用最佳模型预测
results_final_best=cvModel_rf.transform(test)
evaluator=MulticlassClassificationEvaluator(predictionCol="prediction")
print('Test set metricd:')
print('Accuracy:{}'.format(evaluator.evaluate(results_final_best,{evaluator.metricName:"accuracy"})))
print('F-1 Score:{}'.format(evaluator.evaluate(results_final_best,{evaluator.metricName:"f1"})))
```
。预测结果Accuracy为0.8235，F-1 Score为0.796，相比基线模型（Accuracy:0.7059；F-1 Score:0.584）分别有XXX%及XXX%提升。
（这里要改，先跑一次）

### 6.结论汇总
#### 总结&反思
**过程总结**
* 这个项目中，我们建立了一个预测流失用户的模型。
* 在数据集中，我们删除了没有用户ID的数据，将时间戳转为人们可读的格式；此外对流失用户建立了标识，并结合对特征与是否流失间关系的探索，并建立了10个特征
* 然后我们选择3个模型：逻辑回归，SVM和随机森林进行比较。根据比较结果。选择了随机森林预测最后结果。
* 接着我们使用交叉验证和参数网络搜索调优随机森林的参数，对测试集进行预测。预测结果Accuracy为0.8235，F-1 Score为0.796，相比基线模型（Accuracy:0.7059；F-1 Score:0.584）分别有XXX%及XXX%提升。

**过程反思**
* 构建合适的特征对于建立好的模型十分重要。而数据集中，现成的可用于预测的特征并不多；我们需要重新构造特征来预测流失用户。而从数据集中构造变量一方面需要经验与知识的积累，一方面需要熟悉手上的数据
* 对比各未经优化的模型间Accuracy及F-1 Score，对于Accuracy各模型相差不大，LogisticRegression及Random Forest的F-1 Score相近。后续想有较大幅度提升，可能需要进一步优化特征选取
#### 改进
1、数据量方面：相对原始数据而言，这个只是一个相对较小的数据集。如果增加数据量，可以得到预测效果更好的模型

2、特征方面：用于预测流失用户的特征进一步增加完善，找到与数据集相关性更强的特征，以提升模型性能。如增加用户注销账户时的的等级作为特征；或者对未理解未探索的特征进一步研究

3、模型方面：选用决策树、梯度提升树等其他算法，对比已使用的算法，观察accuracy与f1分数变化
