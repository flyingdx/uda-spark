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
数据集是json格式，由于文件过大，免费版github不支持上传。将文件压缩。
```python
df=spark.read.json('mini_sparkify_event_data.json.bz2')
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
 |-- ts: long (含义暂不明确)
 |-- userAgent: string (用户使用平台信息)
 |-- userId: string (用户ID)

```

2、对某一列进行查看的方法如下：
通过dropDuplicates()去重查看唯一值；sort对于有数值的进行排序
```python
df.select('userId').dropDuplicates().sort('userId').show()
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
userId列存在非NA的空值，需要删除


#### 清理数据集
##### 处理空值
先通过dropna()，处理userId列空值；

```python
df_clean=df.dropna(how="any",subset=["userId","sessionId"])
```
再通过filter(), 去除有空字符的行.

```python
df_clean=df_clean.filter(df["userId"]!="")
```
### 3.探索性数据分析
##### 建立注销客户的标签
项目提示使用churn作为模型的标签, 并且建议使用Cancellation Confirmation事件来定义客户流失.。
1、标记注销事件：新建一列churn_event列，标记page中的Cancellation Confirmation事件
2、标记注销用户：新建一列churn_user列，标记注销用户。具体方法是，只要用户churn_event中有标记注销，该用户所有的churn列均标记为注销

##### 建立注销客户的标签
定义好客户流失后, 进行探索性数据分析, 观察留存用户和流失用户的行为。绘图观察主要使用了直方图、小提琴图。相比箱线图，小提琴图更能看出密度分布
1、注销与用户添加播放列表数量的关系

```python
#提取Add to Playlist数据，查看用户添加至播放列表数量的分布
lifetime_songs=df_clean.where('page=="Add to Playlist"').groupby(['userId','churn_user']).count().toPandas()
```

```python
#绘制小提琴图
ax=sns.violinplot(data=lifetime_songs,x='churn_user',y='count')
```
* 相比于非注销用户，注销用户将歌曲添加至播放列表的数量较少，且数量的分布相对集中，其小提琴图形相对扁平

2、是否注销与添加好友数量关系

```python
#提取Add Friend数据，观察用户添加好友分布
add_friend=df_clean.where('page=="Add Friend"').groupby(['userId','churn_user']).count().toPandas()
```

```python
#绘制小提琴图
ax=sns.violinplot(data=add_friend,x='churn_user',y='count')
```
* 相比于非注销用户，注销用户添加好友的数量大多处于较低水平；非注销用户添加好友数量从高水平到低水平均有分布，且非注销用户添加好友数量最大值远远大于注销用户的最大值


3、是否注销与性别关系

```python
#提取性别与用户ID列，观察注销与性别间关系
gender_churn=df_clean.dropDuplicates(["userId","gender"]).groupby(["churn_user","gender"]).count().toPandas()
```

```python
#绘制直方图
ax=sns.barplot(x='gender',y='count',hue='churn_user',data=gender_churn)
```
*  男性用户注销账户的绝对人数以及比例均比女性大


### 4.构建预特征
#### 变量选择
结合经验及以上的分析，构建以下变量：
1、听歌情况方面的变量：
* 用户听歌数量：听歌数量越大，说明用户愿意使用该服务，注销几率越小。
* 用户单次（同一sessionId）听歌最大数量：单次听歌数量越大，说明用户愿意使用该服务，注销几率越小
* 播放的歌手数量：播放过的歌手数量越多，侧面说明用户听歌越多，越愿意使用该服务，注销几率越小。
2、从page中提取动作建立变量：
* 差评量：差评越多，说明用户不喜欢该服务，注销几率越大。
* 添加播放列表量：用户将歌曲加进播放列表，一般可说明用户喜欢该音乐；添加的量越多，用户愿意使用该服务的可能性越大，注销可能性越小。
* 添加好友量：添加好友量越多，说明用于越愿意在改服务中交友分享，注销几率越小。
3、其他
* 用户等级：用户曾经有付费，说明用户对该服务还是感兴趣的，注销几率相对小


#### 变量提取
1用户听歌数量
获取每个用户点击页面NextSong的数量信息计数，获得用户添加进播放列表数量
```python
feature_1=df_clean.select('userId','page').where(df_clean.page=="NextSong").groupBy('userId').count().withColumnRenamed('count','song_total')
```

2用户单次（同一sessionId）听歌最大数量
获取每个sessionId点击页面NextSong数量信息并计数，并按用户求最大值，可获得用户单次（同一sessionId）听歌最大数量
```python
feature_5=df_clean.where('page=="NextSong"').groupBy('userId','sessionId').count().groupBy(['userId']).agg({'count':'max'}).withColumnRenamed('max(count)','max_songs_played')
```

3播放的歌手数量
#获取每个用户点击页面NextSong时的artist信息并计数，可获得用户听过的歌手数量
```python
feature_6=df_clean.filter(df_clean.page=="NextSong").select("userId","artist").dropDuplicates().groupby("userId").count().withColumnRenamed("count","artist_total")
```

4差评量
获取每个用户点击页面Thumbs Down的数量信息计数，可获得用户差评量
```python
feature_4=df_clean.select('userID','page').where(df_clean.page=='Thumbs Down').groupBy('userId').count().withColumnRenamed('count','Thumbs Down')
```

5添加播放列表量
获取每个用户点击页面Add to Playlist的数量信息计数，可获得用户添加进播放列表数量
```python
feature_2=df_clean.select('userId','page').where(df_clean.page=='Add to Playlist').groupBy('userId').count().withColumnRenamed('count','add_to_playlist')
```
6添加好友量
获取每个用户点击页面Add Friend的数量信息计数，可获得用户添加好友书量
```python
feature_3=df_clean.select('userId','page').where(df_clean.page=='Add Friend').groupBy('userId').count().withColumnRenamed('count','add_friend')
```
7是否曾经付费/等级
将level中free/paid转换为0/1；只有用户曾经付费，标记为1

```python
windowval_feature=Window.partitionBy('userId')
feature_7=df_clean.select('userId','level').replace(['free', 'paid'],['0','1'],'level').select('userId', col('level').cast('int'))
feature_7=feature_7.withColumn('level_max',max('level').over(windowval_feature)).drop('level').dropDuplicates()
```

整理标签列
后续建模时，真实标记列默认为label列，将churn列重命名为label

```python
label=df_valid.select('userId',col('churn').alias('label')).dropDuplicates()
```
#### 变量聚合
1、通过join将变量连接，选用并集
```python
df_feature=feature_1.join(feature_2,'userId','outer')\
    .join(feature_3,'userId','outer')\
    .join(feature_4,'userId','outer')\
    .join(feature_5,'userId','outer')\
    .join(feature_6,'userId','outer')\
    .join(feature_7,'userId','outer')\
    .join(label,'userId','outer')
```
2、无值的，用0填充

```python
df_feature=df_feature.fillna(0)
```
3、删除索引

```python
df_feature=df_feature.drop('userId')
```

### 5.建模预测
模型选用逻辑回归、支持向量机与随机森林。根据项目说明，选用 F1 score 作为主要优化指标。
#### 准备数据
将数据转换为向量形式，标准化，并分成训练集、测试集和验证集
```python
#用VectorAssembler将数据集转换为可供模型计算的结构（向量形式）
cols=["song_total","add_to_playlist","add_friend","Thumbs Down","max_songs_played","artist_total","level_max"]
assembler=VectorAssembler(inputCols=cols,outputCol="features_vec")
df_feature=assembler.transform(df_feature)

#用StandardScaler标准化数据
scaler=StandardScaler(inputCol="features_vec",outputCol="features",withStd=True)
scalerModel=scaler.fit(df_feature)
df_feature=scalerModel.transform(df_feature)

#按60%，40%，40%比例拆分为训练集、测试集和验证集
train,validation,test=data.randomSplit([0.6,0.2,0.2],seed=42)
```
#### 模型选择
**模型选择思路**
* 选用逻辑回归、支持向量机、随机森林进行对比，这几个模型一般不需要很多参数调整就可以达到不错的效果。他们的优缺点如下：
1、逻辑回归：优点：计算速度快，容易理解；缺点：容易产生欠拟合
2、支持向量机：数据量较小情况下解决机器学习问题，可以解决非线性问题。缺点：对缺失数据敏感
3、随机森林：优点：有抗过拟合能力。通过平均决策树，降低过拟合的风险性。缺点：大量的树结构会占用大量的空间和利用大量时间

**模型训练**
```
* Random Forest

```python
#创建并训练模型，通过time()记录训练时间
rf=RandomForestClassifier(seed=42)#初始化
start=time()#开始时间
model_rf=rf.fit(train)#训练
end=time()#结束时间
print('The training process took{} second'.format(end-start))

#验证模型效果
results_rf=model_rf.transform(validation)#验证集上预测
evaluator=MulticlassClassificationEvaluator(predictionCol="prediction")#评分器
print('Random Forest:')
print('F-1 Score:{}'.format(evaluator.evaluate(results_rf,{evaluator.metricName:"f1"})))#计算F-1 Score
```

* LogisticRegression、LinearSVC
逻辑回归、支持向量机模型代码与随机森林与结构基本一致，主要是需要将代码改为对应模型

**计算结果**
* LogisticRegression模型：F-1 Score为0.7096；耗时121s
* LinearSVC模型：F-1 Score为0.7096；耗时214s
* Random Forest模型：F-1 Score0.7096；耗时215s
LogisticRegression、Random Forest的F-1 Score一致，且较LinearSVC的高。为了避免过拟合，选取Random Forest作为最终模型，选用并通过调节模型参数尝试获取更优模型

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
paramGrid=ParamGridBuilder().addGrid(rf.maxDepth,[10,20]).addGrid(rf.numTrees,[50,100]).build()#建立可选参数的网络，主要对maxDepth、numTrees调整
crossval_rf=CrossValidator(estimator=rf,
        estimatorParamMaps=paramGrid,
        evaluator=f1_evaluator,
        numFolds=3)#3折交叉验证
cvModel_rf=crossval_rf.fit(train)#训练
```
**调整结果**
* 对比调优前后的模型在验证集上预测结果，调优前F-1 Score0.7096；调优后F-1 Score:0.7227，F-1 Score有提升
* 使用调优后模型进行最终预测


#### 对测试集预测
在测试集预测

```python
results_final=cvModel_rf.transform(test)
evaluator=MulticlassClassificationEvaluator(predictionCol="prediction")
print('Test:')
print('Accuracy:{}'.format(evaluator.evaluate(results_final,{evaluator.metricName:"accuracy"})))
print('F-1 Score:{}'.format(evaluator.evaluate(results_final,{evaluator.metricName:"f1"})))
```

在测试集上运算后：F-1 Score:0.6591，和在验证集的结果上相比，F-1 Score有下降。模型存在在过拟合

### 6.结论汇总
#### 总结&反思
**过程总结**
* 这个项目中，我们建立了一个预测流失用户的模型。
* 在数据集中，我们删除了没有用户ID和sessionID的数据；对流失用户建立了标识，并结合对特征与是否流失间关系的探索，并建立了7个特征
* 然后我们选择3个模型：逻辑回归，SVM和随机森林进行比较。根据比较结果。选择了随机森林预测最后结果。
* 接着我们使用交叉验证和参数网络搜索调优随机森林的参数，对测试集进行预测。预测结果F-1 Score:0.6591

**过程反思**
* 对比各未经优化的模型间F-1 Score，各模型相差不大。后续想有较大幅度提升，除了选取更优模型，更多可能需要从创建更合适的特征变量入手
* 数据集中，现成的可用于预测的特征并不多；我们需要重新构造特征来预测流失用户。而从数据集中构造变量，除了需要探索、熟悉手上的数据；还需要经验与知识的积累
#### 改进
1、该数据集放在现实中，数据量并不大。如果进一步增加数据量，可以得到预测效果更好的模型
2、用于预测流失用户的特征进一步增加完善，找到与数据集相关性更强的特征，以提升模型性能。如增加用户注销账户时的的等级作为特征；或者对未理解未探索的特征进一步研究
3、选用决策树、梯度提升树等其他算法，观察accuracy与f1分数变化，对比已使用的算法，选取更优模型
