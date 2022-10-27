# AI_b-classification-of-manmade-and-natural-img
extract features using Gabor filter and HOUGH transformation with preprocessing

## Part B manmade/natural分类

图像模式识别的一般步骤为：输入图像--[图像预处理](https://so.csdn.net/so/search?q=图像预处理&spm=1001.2101.3001.7020)--特征提取--特征选择--图像分类--输出结果。 



#### 图像预处理

- 增强图像数据的对比度 -> 有助于提取特征 √ done with 直方图均衡化

- 图像剪裁 √

- 高斯模糊 + canny边缘检测 √



#### 特征选择

##### 颜色直方图 ？



##### **GIST特征 √**

- 借助滤波器对n个尺度，m个方向进行卷积，得到nm个特征图谱
- 把每个特征图谱分成k×k个区域，计算每个区域中的均值
- 就得到了k×k×n×m维度的GIST特征

| 描述                            | 描述的内容                                                   |
| ------------------------------- | ------------------------------------------------------------ |
| 自然度（Degree of Naturalness） | 场景如果包含**高度的水平和垂直线，这表明该场景有明显的人工痕迹**，通常自然景象具有纹理区域和起伏的轮廓。所以，**边缘具有高度垂直于水平倾向的自然度低**，反之自然度高。 |
| 开放度（Degree of Openness）    | 空间包络是否是封闭（或围绕）的。封闭的，例如：森林、山、城市中心。或者是广阔的，开放的，例如：海岸、高速公路。 |
| 粗糙度（Degree of Roughness）   | 主要指主要构成成分的颗粒大小。这取决于每个空间中元素的尺寸，他们构建更加复杂的元素的可能性，以及构建的元素之间的结构关系等等。粗糙度与场景的分形维度有关，所以可以叫复杂度。 |
| 膨胀度（Degree of Expansion）   | 平行线收敛，给出了空间梯度的深度特点。例如平面视图中的建筑物，具有低膨胀度。相反，非常长的街道则具有高膨胀度。 |
| 险峻度（Degree of Ruggedness）  | 即相对于水平线的偏移。（例如，平坦的水平地面上的山地景观与陡峭的地面）。险峻的环境下在图片中生产倾斜的轮廓，并隐藏了地平线线。大多数的人造环境建立了平坦地面。因此，险峻的环境大多是自然的。 |

- 一种宏观意义的场景特征描述
- 对于“大街上有一些行人”这个场景，我们必须通过局部特征辨认图像是否有大街、行人等对象，再断定这是否是满足该场景。但这个计算量无疑是巨大的，且特征向量也可能大得无法在内存中存储计算。只识别“大街上有一些行人”这个场景，无需知道图像中在那些位置有多少人，或者有其他什么对象。
- Gist特征向量可以一定程度表征这种宏观场景特征
  ![1666357754421](C:\Users\61422\AppData\Roaming\Typora\typora-user-images\1666357754421.png)
- https://gitee.com/Kalafinaian/python-img_gist_feature/tree/master





##### 霍夫变换 - 检测几何

直方图均衡化 -> 高斯模糊 -> canny边缘检测 -> 霍夫变换

###### 1. 考虑线段的条数：

- 直觉上，直线多的更可能是manmade

- 事实上，调整参数后做霍夫检测，得到：

  - $$\frac{\text{avg(the number of lines in manmade set)}}{\text{avg(the number of lines in natural set)} } = 2.21$$

- 观察散点图，也可得到：

  <img src="C:\Users\61422\AppData\Roaming\Typora\typora-user-images\1666622396733.png" alt="1666622396733" style="zoom:50%;" />

- 直接考虑线段条数，做线性回归进行分类，正确率有76.4%

- 用GIST错误分类的图中，可以看到比较明显的线段条数差异

  <img src="C:\Users\61422\AppData\Roaming\Typora\typora-user-images\1666450743908.png" alt="1666450743908" style="zoom:50%;" />

- 加入线段条数的考虑之后，正确率增加了2%左右



###### 2. 考虑最长线段的长度：

- 经过调参后，
  - $$\frac{\text{manmade set中的最长线段长}}{\text{nature set中的最长线段长}} = 2.399$$
- 直接考虑最长线段长度，做逻辑回归，正确率有70.6%



###### 3. 考虑不同尺度上的上述两个特征 //todo



#### 自定义距离的knn

最后的KNN从两个维度去考虑两张图片之间的相似度，也就是KNN中两个样本点之间的“距离”。

```python
class MyKNN:

  def __init__(self,K=3,weight_gist= 0.85/(0.85+0.75)):
    self.weight_gist = weight_gist
    self.K = K
    pass

  """fit: normalized后输入, 输入变量都化成list(np.ndarray)"""
  def fit(self,training_gist_set,training_hough_set,training_y):
    self.training_gist_set = training_gist_set
    self.training_hough_set = training_hough_set
    self.training_y = training_y
  
  def computeDistance(self,training_gist_v,training_hough_v,target_gist_vector,target_hough_vector):
    diff1 = np.sum((training_gist_v-target_gist_vector) ** 2) ** 0.5
    diff2 = np.sum((training_hough_v-target_hough_vector) ** 2) ** 0.5
    # diff2 = np.linalg.norm(training_hough_v-target_hough_vector,'L2')
    dis = self.weight_gist*diff1 + (1-self.weight_gist)*diff2
    return dis

  def getNeighbor(self,target_gist_vector,target_hough_vector):
    sample_dis_to_train = np.zeros(shape=len(self.training_gist_set))
    for i in range(0,len(self.training_gist_set)):
      sample_dis_to_train[i] = self.computeDistance(training_gist_v=self.training_gist_set[i],training_hough_v=self.training_hough_set[i],target_gist_vector=target_gist_vector,target_hough_vector=target_hough_vector)
    
    K_index = np.argsort(sample_dis_to_train,kind='stable')[:self.K]
    neighbors = self.training_y[K_index]
    return neighbors

  def majority(self, neighbors):
    b = np.bincount(neighbors)
    return np.argmax(b)

  def predict(self,target_gist_set,target_hough_set):
    target_predictions = []
    for i in range(0,len(target_gist_set)):
      ns = self.getNeighbor(target_gist_vector=target_gist_set[i],target_hough_vector=target_hough_set[i])
      res = self.majority(neighbors=ns)
      target_predictions.append(res)
    return target_predictions
  


```

##### 特征工程

###### 1. 特征选择

获得hough变换后，每张图片的线段数量、长度特征之后，选择的形式是$$x^2,x^3,\log x$$ ...？

- 对于线段数量$x$特征，直接用线性回归做二分类

  - $x$ : 76.4%

    <img src="C:\Users\61422\AppData\Roaming\Typora\typora-user-images\1666892711793.png" alt="1666892711793" style="zoom: 50%;" />

  - $x^2$: 72.6%

    <img src="C:\Users\61422\AppData\Roaming\Typora\typora-user-images\1666892748353.png" alt="1666892748353" style="zoom:50%;" />

  - $log(x+1.1)$: 79.8%

    <img src="C:\Users\61422\AppData\Roaming\Typora\typora-user-images\1666892813792.png" alt="1666892813792" style="zoom:50%;" />

- 对于hough变化后检测出的最长线段长度，做线性回归
  - $x$: 68.6%
  - $x^2$: 60.6%
  - $log(x+1.1)$: 71.4%



经过以上的分析之后，我选定取log后的特征作为最后输入到KNN模型的特征。



###### 2. 特征标准化

gist和hough两者之间不存在“量纲”的影响，它们的每一个feature都是标准化之后的。

![1666623805668](C:\Users\61422\AppData\Roaming\Typora\typora-user-images\1666623805668.png)

<img src="C:\Users\61422\AppData\Roaming\Typora\typora-user-images\1666623823363.png" alt="1666623823363" style="zoom:50%;" />



##### 权重调整

最后的myKNN中，只考虑以上两个参数，正确率有79.2%

<img src="C:\Users\61422\AppData\Roaming\Typora\typora-user-images\1666623015684.png" alt="1666623015684" style="zoom:50%;" />



而只考虑gist，正确率有84.2%

<img src="C:\Users\61422\AppData\Roaming\Typora\typora-user-images\1666623102276.png" alt="1666623102276" style="zoom:50%;" />



调整权重后，正确率达到86.6%
$$
\text{weight} = \frac{只考虑gist的正确率}{(只考虑gist的正确率+只考虑霍夫变换的正确率)}
$$
<img src="C:\Users\61422\AppData\Roaming\Typora\typora-user-images\1666623174850.png" alt="1666623174850" style="zoom:50%;" />



#### Ensemble - 不同K的KNN

用$K=[3,5,7]$，3个KNN做集成投票，准确率从86.6%上升到了87.2%

```python
def emsemlble_knn(X_train_gist,X_train_hough_set,y_train,X_test_gist,X_test_hough_set,K_list=[3,5,7],weight_gist = 84.2/(79.2+84.2)):
  pred = []
  for i in range(0,len(K_list)):
    knn = MyKNN(K=K_list[i],weight_gist = weight_gist)
    knn.fit(X_train_gist,X_train_hough_set,np.array(y_train))
    pred_myknn = knn.predict(X_test_gist,X_test_hough_set)
    pred.append(np.array(pred_myknn))
  
  sum_pred = np.zeros(len(X_test_gist))
  for l in range(0,len(pred)):
    sum_pred = sum_pred + pred[i]

  predict = np.round(sum_pred/len(K_list)) #因为是0，1分类

  return predict

  
```

![1666881177391](C:\Users\61422\AppData\Roaming\Typora\typora-user-images\1666881177391.png)



#### 调整训练集样本的权重//todo









----

#### 问题

1. SIFT特征提取跑得好慢，而且用similarity rate定义的距离也不准？
   - 图像裁剪resize
   - 高斯模糊？（可是我不想删去边缘信息，可能需要设置一个小一点的sigma）
   - **边缘检测后用SIFT**
2. **颜色？不太会用**
3. GIST好像没有把很明显的线条特征提取出来！！！GIST更偏向一个宏观的角度。可能考虑用边缘检测后的图片+SIFT特征提取。
4. 有些训练集很误导人，进行一个“人工boosting”
5. 钟乳石总是分类错误



#### TODO

##### 1. 霍夫变换想法实现 √

##### 2. 调参 √

##### 2. ‘’boosting“去除一些奇怪训练集的比重
