#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
data=[5,4,2.2,0,3]
arr1=np.array(data)
arr1
arr2=np.array([5,4,2.2,0,3],int)
arr1.dtype
arr2.dtype

vector_1d=np.array([4,3,2,"1"],float)
vector_1d
vector_1d.shape

#3차원 tensor
data_2=[[[1,2,3],[4,5,6],[7,8,9]],
      [[1,2,3],[4,5,6],[7,8,9]],
      [[1,2,3],[4,5,6],[7,8,9]]]
tensor_3d=np.array(data_2,int)
tensor_3d
tensor_3d.shape
tensor_3d.dtype
tensor_3d.ndim #number of dimension
tensor_3d.size #number of element
      


# In[36]:


#2. numpy생성
#파이썬 내장함수인 range는 list반환이지만 numpy arange는 ndarray반환

np.arange(10) #integer로 0부터 9까지 추출
np.arange(0,1,0.2) #float, step
np.arange(20).reshape(4,5) #20개의 원소를 4행 5열로 ndarray의 구조를 바꿔라

#zeros,ones,empty
#zeros:0으로 가득찬 ndarray생성
#np.zeros(Shape,dtype,order)
np.zeros(shape=(5))  #요소=0인 vector 생성

#ones:1로 가득찬 ndarrya생성
np.ones(shape=(10))

#empty:메모리를 할당하여 새로운 배열 생성하나 memory를 초기화 하지 않음
np.empty(shape=(10,),dtype=np.int8) #초기화가 안된 임의의 vector 생성,int 8
np.empty((3,3)) #3by3 임의의 matrix 생성, 기본 float type

#something_like -기존 ndarray의 shape만큼 1or0으로 변경
t_matrix=np.arange(6).reshape(2,3)
np.ones_like(t_matrix)

t_matrix1=np.arange(6).reshape(2,3)
np.zeros_like(t_matrix1)

#identity -단위행렬(i 행렬)을 생성 
np.identity(n=3,dtype=np.int8)

#eye -n*m크기의 대각선이 1인 행렬 생성, k값에 따라 1의 시작점 변경
np.eye(N=3,M=3,dtype=np.int8)
np.eye(3) #identity행렬과 같게 출력
np.eye(3,6,k=3) #k는 시작점


# In[51]:


import numpy as np
#axis축 - numpy에서 연산은 axis축 기준으로 이루어짐
#집계함수 -sum, min,max
t_array=np.arange(1,13).reshape(3,4)
t_array
np.sum(t_array,dtype=float)
np.min(t_array)
np.max(t_array)
t_array.sum(axis=0),t_array.sum(axis=1)
tensor=np.array([t_array,t_array,t_array])
tensor #2d array 사용 3d array생성


# In[52]:


tensor.sum(axis=0)


# In[53]:


tensor.min(axis=1)


# In[54]:


tensor.max(axis=2)


# In[55]:


#array간의 연산 -array간 shape 같을 경우 각 요소마다 연산 실행
t_matrix=np.arange(1,10).reshape(3,3)
t_matrix*t_matrix


# In[56]:


#broadcasting
t_matrix=np.array([[1,2],[3,4]],float)
t_matrix+2 #matrix 와 scalar연산


# In[57]:


#2d array와 1d array의 연산
arr_2d=np.arange(1,13).reshape(4,3)
arr_1d=np.arange(100,400,100)
arr_2d+arr_1d


# In[61]:


#비교연산
a=np.array([1,2,3,4])
b=np.array([4,2,2,4],float)
a==b
a>b
np.all(a>3)
np.any(a>3)


# In[62]:


a=np.array([1,2,3,4])
b=np.array([4,2,2,4],float)
np.logical_and(a>0,a<3) #and 조건의 비교


# In[63]:


np.logical_or(a>0,a<3)#or 조건의 비교


# In[64]:


c=np.array([False, True, True],bool)
np.logical_not(b) #not 조건의 비교


# In[65]:


#np.where -if절과 같은 효과 where(조건, true일 때 원하는 value, false일 때 원하는 value)
a=np.array([2,3,1],float)
np.where(a>1,0,3)


# In[66]:


a=np.arange(3,10)
np.where(a>6)


# In[67]:


#np.isnan, np.isfinite -np.isnan:null일경우 true, np.finite: 한정된 수인 경우true
a=np.array([2,np.NaN, np.Inf],float)
np.isnan(a) #null인 경우 true


# In[68]:


np.isfinite(a) #한정된 수인 경우 true


# In[5]:


import numpy as np
#ravel -다차원 배열을 낮은 차원으로 변환 -원본의 뷰를 반환 reference
t_arr=[[[1,2],[3,4]],
      [[1,2],[3,4]]]
b=np.array(t_arr)
c=b.ravel()
print("b=",b)
print("c=",c)
c[3]=0
print("c=",c)
print("b=",b)


# In[7]:


#flatten copy-다차원 배열을 낮은 차원으로 변환 -복사본을 반환 ->데이터 수정하여도 원본 데이터 안 바뀜
t_arr=[[1,2],[3,4]]
b=np.array(t_arr)
c=b.flatten()
print("b=",b)
print("c=",c)

c[3]=0
print("c=",c)
print("b=",b)


# In[11]:


#transpose -행렬의 transpose or T속성
a=np.arange(1,7).reshape(3,2)
print(a)
print(a.transpose())
a.T


# In[13]:


#배열결함 -Concatenate  -배열과 배열을 결합하는 함수 -axis 기준으로 결합
a=np.array([[1,2,3]])
b=np.array([[4,5,6]])
print(np.concatenate((a,b),axis=0)) #위 아래로 결합
a=np.array([[1,2],[3,4]])
b=np.array([[5,6]])
np.concatenate((a,b.T),axis=1) #옆으로 결합


# In[16]:


#배열결합 -vstack: axis=0 , -hstack:axis=1

a=np.array([1,2,3])
b=np.array([4,5,6])
print(np.vstack((a,b)))

a=np.array([[1],[2],[3]])
b=np.array([[4],[5],[6]])
print(np.hstack((a,b)))


# In[22]:


#indexing -배열의 개별 요소는 각 axis의 인덱스로 참조가능
a=np.arange(1,5)
print(a)
print("a의 0번째 요소 = ",a[0])
print("a의 1번째 요소 = ",a[1])
b=a.reshape(2,2)
print(b)
print(b[0,0])
print(b[1][1])
b[0,0]=7
print(b)
b[1][0]=8
print(b)


# In[29]:


#슬라이스 색인 -축 기준으로 슬라이싱 한다
a1=np.arange(1,21).reshape((4,5)) #2차원 배열
print(a1)
print(a1[:,1:]) #전체 row의 1열 이상 
print(a1[2,1:4]) #2row의 1열~3열
print(a1[2:7]) #2row~마지막row

#step지정
a1[:,::2]


# In[30]:


#argmax, argmin 최대 최소 있는 인덱스 반환
a=np.array([2,3,1,5,6,22,11])
print("최대값 index= ",np.argmax(a),"최소값 index= ",np.argmin(a))


# In[35]:


a2=np.random.choice(np.arange(2,22),12).reshape(3,4)
print(a2)
print(np.argmax(a2,axis=0))
print(np.argmax(a2,axis=1)) #다차원에서는axis기반


# In[36]:


#boolean indexing
a=np.arange(1,21).reshape(4,5)
print(a)
bool_a=(a%2==0) #2로 나눈 나머지가 0일 경우
print(bool_a)


# In[39]:


#fancy indexing
array_a=np.arange(1,7)
array_b=np.array([1,0,2,0,1,4],int)
#반드시 integer로 선언

array_a.take(array_b)


# In[44]:


#입출력
#loadtxt,savetxt #텍스트 파일 불러오고 저장
a=np.arange(1,200,2).reshape(20,5)
a[:10]
np.savetxt("int_price_1.csv",a,delimiter=",")
np.savetxt("int_price_2.csv",a,fmt='%d',delimiter=",")

b=np.loadtxt("./int_price_2.csv",delimiter=",")
b[:5]
b_int=b.astype(int)
b_int[:3]


#npy파일로 저장 (pickle)형태로 데이터를 저장하고 불러옴 , binary형태로 파일저장
np.save("npy_test.npy",arr=b_int)
npy_array=np.load(file="npy_test.npy")
npy_array[:3]

