[TOC]

# ACM总结

## 技巧

### 快读快写（原理？？？）

```c++
void readT(int &x){
    int s=0,w=1;char ch=getchar();
    while(ch<'0'||ch>'9'){if(ch=='-')w=-1;ch=getchar();}
    while(ch>='0'&&ch<='9'){
        s=(s<<3)+(s<<1)+ch-'0';
        //s=((s<<3)+(s<<1)+ch-'0')%mod;//在快读时取模
        ch=getchar();
    }
    x=s*w;
}
```

```c++
char buf[1<<20],*p1,*p2;
#define getchar() (p1 == p2 && (p2 = (p1 = buf) + fread(buf,1,1<<20,stdin), p1 == p2) ? 0 : *p1++)
template<typename T>inline void readT(T &x){
    bool f=1;x=0;char ch=getchar();
    while(ch<'0'||ch>'9'){if(ch=='-') f=!f;ch=getchar();}
    while(ch>='0'&&ch<='9'){x=(x<<1)+(x<<3)+(ch^48);ch=getchar();}
    x=(f?x:-x);return;
}
```

```c++
void writeT(int x){
	if(x < 0){
		putchar('-');
		x = -x;
	}
	if(x > 9)writeT(x / 10);
	putchar(x % 10 + '0');
}
```

### next_permutation(arr,arr+n)？？？

有重复元素能不能用？

生成给定序列的下一个较大排序，直到序列按降序排列为止。

如果没有下一个序列则返回false

可以先sort再循环，遍历所有序列。

prev_permutation()同理、

#### 原理？？？

#### O(n^2^)判断一个排列是全排列的第几个

树状数组降至O(nlgn)，从n到1将arr加入树状数组，然后每次加入时区间查询前arr[i]个数的和

```c++
int num[maxn];
int fac[maxn];//阶乘
for(int i=1;i<=n;++i){
    int now = 1;
    for(int j=i+1;j<=n;j++){
        if(arr[j]<arr[i])now++;
    }num[i] = now;//num[i]代表arr[i]在arr[i~n]之间的大小
}//将这里替换成树状数组可以将复杂度降至O(nlgn)
fac[0]=1;
for(int i=1;i<=n;i++){//预处理阶乘
    fac[i]=i*fac[i-1];
}
int ans = 0;
for(int i=n;i>=1;i--){
    ans+=fac[n-i]*(num[i]-1);//前i-1位不动，将第i位换掉有num[i]-1种选择，换完之后已经比arr小了，剩下n-i个元素随意排列有fac[n-i]种方式
}
cout<<ans<<endl;
```

### memset

逐个字节进行更改，所以对于不止一个字节的数据一般只改为0或-1，改成其他数据容易出错

```c++
#include<cstring>
int arr[n];
memset(arr,t,sizeof(arr));//arr为起始地址,t=0或-1,sizeof(arr)=4*n
char list[n];
memset(list,'t',sizeof(list));
```

### memcpy

```c++
memcpy(dest,src,n);//dest目标，src来源，n复制的字节数
```

### scanf   printf

%s 字符数组 %……  ==记一下==

当没有具体说明有多少组问题时

例如问题输入的是字符串

可以用char c[maxn];while(scanf("%s",&c)!=EOF){  ……  } 

while(~scanf(%d,&n)){ ...... }   需要删除关闭同步流？？？

## 基础算法

### 排序的三种实现

#### 第一种：重载运算符

````C++
#include <bits/stdc++.h>
struct node{
	int a,b,sum;
	bool operator < (const node &x)const{
		return x.sum<sum; //从小到大排序
	}
}num[MAXN];
//创造好节点捆绑以后，对运算符进行重载，这样就可以比较节点
int main(){
	sort(num,num+n);//直接使用sort对从num[0]到num[n]进行排序
}//sum排序以后，其对应的a，b的顺序也发生了改变
````
#### 第二种：自定义一个cmp函数

````c++
#include <bits/stdc++.h>
#include <algorithm>
struct Node{
	int a,b,sum;
}num[MAXN];
bool cmp(Node x,Node y){
	return x.sum<y.sum;//从小到大排序
}
int main(){
	sort(num,num+n,cmp);//按照定义的cmp函数，从小到大排序
	sort(num,num+n,greater<int>());//降序排序
}
````

#### 第三种：匿名函数

````C++
struct Node{
	int a,b,sum;
}num[MAXN];
int main(){
	sort(arr+1,arr+1+n,[](Node x,Node y){return x.sum<y.sum;});
	sort(arr+1,arr+1+n,[](Node x,Node y){return x.a<y.a;});
}//可以通过这种方式实现主排序和次排序
````

### 前缀和与差分???

#### 一维前缀和

前缀和指的是1到i位置（这个区间）所有的数字之和
前缀和的优势：以o（1）的时间优势得到某块区间的总和

````C++
//用L和R分别表示左右端点
S[L]=a[1]+a[2]+......+a[L];
S[R]=a[1]+a[2]+......+a[L]+......+a[R];
//求[L,R]的区间和
sum=S[R]-S[L-1]=a[L]+a[L+1]+......+a[R];
````
#### 二维前缀和

```c++
#include<iostream>
using namespace std;
const int maxn=1e5+5;
int main(){
int n,m;cin>>n>>m;
int sum[maxn][maxn]={0};
int arr[maxn][maxn]={0};
for(int a=1;a<=n;a++){
for(int b=1;b<=m;b++){
cin>>arr[a][b];}}//输入原数组
    
sum[1][1]=arr[1][1];
for(int a=2;a<=m;a++){sum[1][a]=sum[1][a-1]+arr[1][a];}
for(int a=2;a<=n;a++){sum[a][1]=sum[a-1][1]+arr[a][1];}
//对第一列和第一行进行初始化
    
for(int a=2;a<=n;a++){
for(int b=2;b<=m;b++){
sum[a][b]=arr[a][b]+sum[a-1][b]+sum[a][b-1]-sum[a-1][b-1];}}
//完成前缀和数组
}
求(x,y)到(n,m)范围内总和:sum[n][m]-sum[x-1][m]-sum[n][y-1]+sum[x-1][y-1]
```

#### 高维前缀和???

##### 基于容斥原理？？？

##### 逐维前缀和

每次只考虑一个维度，固定所有其它维度，然后求若干个一维前缀和，这样对所有k个维度分别求和之后，得到的就是k维前缀和。

```c++
//三维前缀和（OIWiki）：
#include <iostream>
#include <vector>

int main() {
  // Input.
  int N1, N2, N3;
  std::cin >> N1 >> N2 >> N3;
  std::vector<std::vector<std::vector<int>>> a(
      N1 + 1, std::vector<std::vector<int>>(N2 + 1, std::vector<int>(N3 + 1)));
  for (int i = 1; i <= N1; ++i)
    for (int j = 1; j <= N2; ++j)
      for (int k = 1; k <= N3; ++k) std::cin >> a[i][j][k];

  // Copy.
  auto ps = a;

  // Prefix-sum for 3rd dimension.
  for (int i = 1; i <= N1; ++i)
    for (int j = 1; j <= N2; ++j)
      for (int k = 1; k <= N3; ++k) ps[i][j][k] += ps[i][j][k - 1];

  // Prefix-sum for 2nd dimension.
  for (int i = 1; i <= N1; ++i)
    for (int j = 1; j <= N2; ++j)
      for (int k = 1; k <= N3; ++k) ps[i][j][k] += ps[i][j - 1][k];

  // Prefix-sum for 1st dimension.
  for (int i = 1; i <= N1; ++i)
    for (int j = 1; j <= N2; ++j)
      for (int k = 1; k <= N3; ++k) ps[i][j][k] += ps[i - 1][j][k];

  // Output.
  for (int i = 1; i <= N1; ++i) {
    for (int j = 1; j <= N2; ++j) {
      for (int k = 1; k <= N3; ++k) {
        std::cout << ps[i][j][k] << ' ';
      }
      std::cout << '\n';
    }
    std::cout << '\n';
  }

  return 0;
}
```

假设$N_1*N_2*...*N_k=N$,复杂度为$O(NK)$

#### 树上前缀和

sum~i~表示结点i到根节点的权值总和(通过dfs实现就可以)

如果是点权，x,y路径上的和是sum~x~+sum~y~-2*sum~lca~+arr[lca]

如果是边权，x,y路径上的和是sum~x~+sum~y~-2*sum~lca~

#### 一维差分

有差分数组，可以在o(n)的时间内得到原数组
````C++
原数组a：a[1],a[2],......,a[n]
差分数组b：b[1],b[2],......,b[n]
a[i]=b[1]+b[2]+......+b[i]
````
差分的应用：可以使得[L,R]区间内的元素，以o（1）的时间优势同时加c
正常的话，需要遍历该区间的元素，时间为o（n）

#### 二维差分

arr[i][j\]由(1,1)到(i,j)内的所有差分数组ans累加而来

给定数组arr，初始化差分数组ans:

```c++
ans[i][j]=arr[i][j]+arr[i-1][j-1]-arr[i][j-1]-arr[i-1][j];
```

从(x1,y1)到(x2,y2)内的每个数增加k:

```c++
ans[x1][y1]+=k;ans[x1][y2+1]-=k;ans[x2+1][y1]-=k;ans[x2+1][y2+1]+=k;
```

获到改变之后的原数组：

```c++
for(int i=1;i<=n;i++){
	for(int j=1;j<=m;j++){
		arr[i][j]=ans[i][j]+arr[i-1][j]+arr[i][j-1]-arr[i-1][j-1];
	}
}
```

#### 树上差分

##### 点差分

a是原值，d是差分值

a~i~=d~i~+所有子树的d的和

当路径从s到t上的点都加x

d~s~+=x    d~t~+=x    d~lca(s,t)~-=x   d~fa(lca)~-=x

##### 边差分

用一条边中深度较大的点权来代替边权

当路径从s到t上的点都加x

d~s~+=x    d~t~+=x    d~lca(s,t)~-=2*x

### 分治算法

#### 1.归并排序

O(nlogn)

````C++ 
void mergesort(int left,int right,int *a ){
	if(left==right) return;
	int mid=(left+right)/2;
	mergesort(left,mid,a);
	mergesort(mid+1,right,a);//分
	int l1=left,l2=mid+1;
	int l3=left;
	while(l1<=mid && l2<=right){
		if(a[l1]<a[l2]) temp[l3++]=a[l1++];
		else if(a[l2]<a[l1]) temp[l3++]=a[l2++];
	}//先将左、右两个子数组中的一个子数组中的元素耗尽
	while(l1<=mid) temp[l3++]=a[l1++];
	//若右数组耗尽，则将左数组的元素依次补入temp后面
	while(l2<=right) temp[l3++]=a[l2++];
	//若左数组耗尽，则将右数组的元素依次补入temp后面
	for(int i=left ; i<=right ; i++){
		a[i]=temp[i];//将temp中的元素归还给数组a
	}
}
````
应用：归并排序求逆序对
````C++ 
queue<int>que;
int ans=0;
void mergesort(int l,int r){
    if(l==r)return ;
    int mid=(l+r)/2;
    mergesort(l,mid);mergesort(mid+1,r);
    int now1=l,now2=mid+1;
    while(now1<=mid&&now2<=r){
        if(arr[now1]<=arr[now2]){
            que.push(arr[now1]);now1++;
            //ans+=now2-mid-1;
        }
        else {
            que.push(arr[now2]);now2++;
            ans+=(mid-now1+1);		//左半部分剩下的还没排序的数可以和arr[now2]形成逆序对
        }
    }
    while(now1<=mid){
        que.push(arr[now1]);now1++;
        //ans+=r-mid;
    }
    while(now2<=r){
        que.push(arr[now2]);now2++;
    }
    for(int a=l;a<=r;a++){
        arr[a]=que.front();
        que.pop();
    }
}
````
#### 2.二分答案
二分法模板
````C++
bool check(){
	if(不满足条件) return 0;
	else return 1;
}
int main(){
	int l=0,r=1e9,answer=0;//l和r分别为答案的上下界，answer为最终答案
	while(l<=r){
		int mid=(l+r)/2;
		if(check(mid)) answer=mid,l=mid+1;//使循环结束
		else r=mid-1;//l还是r要根据题意，选取左或者右区间
	}
	cout<<answer<<endl;
}
````
### 高精度？？？

#### ___int128

int 的范围是1e9，long long 的范围是1e18
int128板子，范围略大于1e38。范围再大用string或者char[]模拟竖式计算

````C++
inline void readint(__int128 &X){
//把同步流关掉
 	X = 0;
	int w=0; char ch=0;
 	while(!isdigit(ch)) {w|=ch=='-';ch=getchar();}
 	while(isdigit(ch)) X=(X<<3)+(X<<1)+(ch^48),ch=getchar();
 		if (w) X = -X;
}
inline void printint(__int128 x){
	if (x < 0) putchar('-'),x = -x;
 	if(x>=10) printint(x/10);
 		putchar('0'^(x%10));
}
int main(){
    __int128 x;
    readint(x);
    printint(x);
}
````
#### 数组模拟？？？（OIwiki上）

```c++
string s1,s2;
cin>>s1>>s2;

//加法
int x[500]={0},y[500]={0},z[501]={0};
int t=max(s1.length(),s2.length());
for(int a=s1.length()-1;a>=0;a--)x[s1.length()-a]=s1[a]-'0';
for(int a=s2.length()-1;a>=0;a--)y[s2.length()-a]=s2[a]-'0';//倒序
for(int a=1;a<=t;a++){
    z[a]+=x[a]+y[a];
    if(z[a]>=10){z[a]-=10;z[a+1]++;}
}
int now=z[t+1]?t+1:t;
for(1;now>=1;now--)cout<<z[now];

//减法
int x[500]={0},y[500]={0},z[500]={0};
int l1=s1.length(),l2=s2.length();
if(l1<l2||(l1==l2&&s1<s2)){swap(s1,s2);swap(l1,l2);cout<<"-";}
for(int a=0;a<l1;a++){
    x[l1-a]=s1[a]-'0';
}
for(int a=0;a<l2;a++){
    y[l2-a]=s2[a]-'0';
}
for(int a=1;a<=l1;a++){
    z[a]+=x[a]-y[a];
    if(z[a]<0){z[a]+=10;z[a+1]--;}
}
int now;
for(now=l1+1;now>1;now--){
	if(z[now])break;
}//去除前面的0
for(1;now>=1;now--)cout<<z[now];

//乘法
int x[500]={0},y[500]={0},z[300000]={0};
int cnt=0;int l1=s1.length(),l2=s2.length();
for(int a=l1-1;a>=0;a--){x[l1-a]=s1[a]-'0';}
for(int a=l2-1;a>=0;a--){y[l2-a]=s2[a]-'0';}
for(int a=1;a<=l1;a++){
    for(int b=1;b<=l2;b++){
        z[a+b-1]+=x[a]*y[b];
        z[a+b]+=z[a+b-1]/10;
        z[a+b-1]%=10;
    }
}int now;
for(now=1;1;now++){
    if(z[now])continue;
    now--;break;}
for(now;now>=1;now--)cout<<z[now];

//高精度除以低精度
int x[500]={0};int y=0;int z[500];int yu=0;//余数
string s1;cin>>s1;cin>>y;
int k=s1.length();
for(int a=0;a<k;a++)x[a]=s1[a]-'0';
for(int a=0;a<k;a++){
    z[a]=(x[a]+yu*10)/y;
    yu=(x[a]+yu*10)%y;
}
int t=0;
for(1;t<k;t++){
    if(z[t])break;
}//去除前面的0
for(1;t<k;t++)cout<<z[t];

//高精度除以高精度   ???
```

### 离散化

离散化是将离散的散点映射成相对连续的值，将数据范围缩小并保留其顺序性质。
实现方法是，先对原数组排序，并去重，将对应的排名与数值进行一一映射。

离散化模板：
````C++
for(int i=1 ; i<=n ; i++){
	cin >> a[i];
	b[i] = a[i];
}
sort(b,b+n);
int temp=unique(b,b+n)-b-1;
for(int i=1 ; i<=n ; i++) a[i]=lower_bound(b+1,b+temp+1,a[i])-b;
````

### 三分

单峰函数可用三分法求最值

例如该函数在[l,r]上有单峰最大值

```c++
double solve(double l,double r){
    //cnt++;if(cnt>10000)return l;  根据题意添加截止条件
    if(l>=r)return l;
    double now=r-l;
    now/=3;
    double mid1=l+now,mid2=r-now;	//三分范围
    double ans1=0,ans2=0;	//两个三分点对应的值
    for(int a=0;a<=n;a++){
        double ans=arr[a];
        for(int b=1;b<=n-a;b++)ans*=mid1;
        ans1+=ans;
    }
    for(int a=0;a<=n;a++){
        double ans=arr[a];
        for(int b=1;b<=n-a;b++)ans*=mid2;
        ans2+=ans;
    }
    if(ans1<ans2){		//因为是最大值，所以较小值及以外都不可能，舍弃
        return solve(mid1,r);
    }
    else return solve(l,mid2);
}
```

###  尺取法——双指针

在有序数列上设置两个指针，以此进行反向扫描或者同向扫描
##### 同向扫描（滑动窗口）
分为固定窗口和动态窗口

###### 动态窗口：

````C++
int left=0,right=0;
while(right指针未越界){
	right++;//扩大窗口
	......//更新窗口中的数据
	while(窗口数据满足条件){
		left++;//缩小窗口
		......
	}
}
````
###### 静态窗口

左指针和右指针一起移动

##### 反向扫描

````C++
int left=0,right=n-1;
while(left<right){
	......//满足题意的操作
	left++;
	right--;
}
````

## 数据结构

### 栈与队列

#### 栈

只从表的一段存取数据，==后进先出==。

```c++
#include<stack>//头文件
stack<int> sta;//创建栈
sta.push(4);//4进栈
cout<<sta.top();//输出
sta.pop()；//出栈
int a=sta.empty();//如果sta为空，则a=1。否则a=0。
int b=sta.size();//b=sta中的元素数量
```

#### queue队列

一段存，一段取，==先进先出==。

```c++
#include<queue>//头文件
queue<int> que;//创建队列
que.push(4);//4进队列
cout<<que.front();//输入首元素
cout<<que.back();//输出尾元素
que.pop();//出队列
int a=que.empty();//如果que为空，则a=1。否则a=0。
int b=que.size();//b=que中的元素数量
```

### 单调栈

单调栈可以找出从左/右遍历第一个比它小/大的元素的位置，一般记录下标

例题：[[每日温度 - 力扣](https://leetcode.cn/problems/daily-temperatures/description/)]     [[接雨水 - 力扣](https://leetcode.cn/problems/trapping-rain-water/description/)]

```c++
//例如找右边第一个比它大的数字
stack<int>sta;
int arr[maxn],ans[maxn];
//方法一,找右边就从左向右，将未找到目标的数的下标存在sta中
sta.push(1);
for(int i=2;i<=n;i++){
    while(!sta.empty()&&arr[i]>arr[sta.top()]){
        ans[sta.top()]=i;
        sta.pop();
    }
    sta.push(i);
}
while(!sta.empty()){ans[sta.top()]=n+1;sta.pop();}//右边不存在更大的数。

//方法二，找右边但从右向左，遍历i的过程中找目标数
sta.push(n);ans[n]=n+1;
for(int i=n-1;i>=1;i--){
    while(!sta.empty()&&arr[i]>=arr[sta.top()]){sta.pop();}
    ans[i]=sta.empty()?n+1:sta.top();
    sta.push(i);
}
```

### 单调队列

求解==定长==滑动窗口的最值问题（用两个优先级队列也可以），如果是二维的话就先预处理出一行中连续数字的最值，然后按照列再进行滑动窗口。

要求的是每连续的k个数中的最大（最小）值，很明显，当一个数进入所要 "寻找" 最大值的范围中时，若这个数比其前面（先进队)的数要大，显然，前面的数会比这个数先出队且不再可能是最大值。

也就是说——当满足以上条件时，可将前面的数 "弹出"，再将该数真正 push 进队尾。

```c++
int arr[maxn],ans[maxn],n,k；//arr存数据，ans存窗口最大值
int cnt=0,n,k;//n数组长度，k窗口宽度(k<=n)
deque<int>que;
	for(int a=1;a<=k;a++){
		while(que.size()&&que.back()<arr[a]){
		que.pop_back();
		}
		que.push_back(arr[a]);
    }
		cnt=1;
	    ans[cnt]=que.front();
//上面是对第一个窗口的处理
	for(int a=k+1;a<=n;a++){
		if(arr[a-k]==que.front())que.pop_front();
//在arr[a-k]前面的数字已经不在que里面这一前提下，后面进入的数字如果比arr[a-k]大的话，就在这个数进入的时候arr[a-k]就已经被pop掉了，如果没有大于arr[a-k]的值，那么arr[a-k]就在que的最前面，所以只需要跟最前面的比较，就能保证arr[a-k]不存在于que中了
		while(que.size()&&que.back()<arr[a]){
		que.pop_back();
		}
		que.push_back(arr[a]);
		cnt++;
		ans[cnt]=que.front();
	}
```

### deque双端队列

```c++
#include<deque>
deque<int> dp;
dp.push_back(x)/push_front(x);//x插入队尾/队首
cout<<dp.back()/front();//输出队尾/队首元素
dp.pop_back()/pop_front();//删除队尾/队首元素
dp.erase(iterator it)//删除该元素   ？？？
dp.erase(iterator first,iterator last);//删除双端队列中[first,last）中的元素
dp,empty()/size()/clear();
dp.begin()/end()//迭代器
sort(dp.begin(),dp.end())//从小到大排序
sort(dp.begin(),dp.end()，greater<int>())//从大到小排序
```

### 堆（优先级队列）

```c++
priority_queue <int> que;
用top，不用front和back，默认为大顶堆，top指的是最大的元素
priority_queue <int,vector<int>,greater<int> > que;//小顶堆
priority_queue <int,vector<int>,less<int> > que;//大顶堆
struct cmp{
    bool operator()(pii&a,pii&b){
        return a.second<b.second;//top().second是最大的，这里的比较是反着的
    }
};//也可以用cmp自定义函数
priority_queue <pii,vector<pii>,cmp > que;
```

#### 对顶堆 

通过一个大根堆维护序列中较小的一半，一个小根堆维护序列中较大的一半，可以用来维护中位数

### STL容器？？？

#### vector向量(动态数组)

```c++
#include<vector>

一维初始化
vector<int> vec;//创建向量vector
vector<int> vec(n);//向量长度设为n，初始值默认为0，下标由0到n-1
vector<int> vec(n,1);//向量初始值均为1
vector<int> vec{1, 2, 3, 4, 5};//vec中有五个元素，长度为5
vector<int> a(vec);
vector<int> a=vec;//将vec复制到a上，两者数据类型相同

二维初始化
vector<int> vec[5];//二维初始化相当于有5个vector，个数不可变，每个都可以有不同长度，第一个是vec[0]。
vector<vector<int>> vector;个数和长度均可变。
    vector<int> t1{1, 2, 3, 4};v.push_back(t1);//往里插入vector

方法函数
cout<<vec.front();//输出第一个数据
cout<<vec.back();//输出最后一个数据
vec.pop_back();//删除最后一个数据
vec.push_back(4);//在尾部加一个元素4
cout<<vec.size();//输出vec内元素个数
vec.clear();//清空元素
vec.insert(it,x)//在迭代器it处插入x
vec.erase(l,r);//删除从l到r-1的元素
vec.empty();
vec.begin();
vec.end();//最后一个元素后一个地址

vector可以用下标遍历，如vec[0]
for(vector<int>::iterator it=vec.begin();it!=vec.end();it++)//遍历
```

#### set集合 lower_bound和erase时间复杂度

自动去重并排序

```c++
#include <set>
set <int> st;//从小到大  
set<int, greater<int> > st;// 从大到小
st.insert(4)//将4插入集合
st.begin()/end()/rbegin()/rend();
st.clear()/empty()/size();
st.erase(iterator)/erase(first,second)//通过迭代器删除
st.erase(4)//将4这个值删除
st.find(4)//如果该值存在，返回该值的迭代器，否则返回st.end()
st.count(4);//查找4出现的次数，由于set去重，所以相当于查找4是否存在
s.lower_bound(k)//返回小于等于k的第一个数的迭代器
s.upper_bound(k)//返回大于k的第一个数的迭代器
    
```

```c++
struct point {
	int x, y;
	bool operator < (const point &p) const {
		// 按照点的横坐标从小到大排序,如果横坐标相同,纵坐标从小到大
		if(x == p.x)
			return y < p.y;
		return x < p.x;
	}
};
set<point> s;
```

multiset可重复有序

#### map映射

```c++
#include<map>
map<type1,type2> mp;//type1对应的是键，type2对应的是值
mp.begin()/end()/rbegin()/rend();
mp.find(key)//如果存在以key为键，则返回该迭代器，否则返回mp.end();
mp.count(key)//计数，因为map键唯一，所以相当于是否存在    
mp.erase(it)/erase(key)/erase(first,second);
mp.size()/empty();

map<int,string>mp;
mp[1]="a";
mp.insert(make_pair(1,"a"));
mp.insert(pair<int,string>(1,"a"));
mp.insert({1,"a"});
```

unordered_map

#### pair

二元结构体

```c++
#include<utility>
pair<string,int> pa;//可定义数组
pa={"abc",12};或者p=make_pair("abc",12);
cout<<pa.first;//输出abc
cout<<pa.second;//输出12
```

#### bitset

```c++
#include<bitset>

//构造
bitset<30>bs();//每一位都是false
bitset<10>bs("100111");bitset<10>bs(39);//bs=0000100111

//运算符
[]:通过下标取值，bitset二进制由低位至高位存储，最右侧下标为0，如上面的bs[0]=1
== != 相同大小的才能进行判断
& | ^ ~ 位运算
<< >> 左移右移
    
//函数
bs.size();//位数
bs.count();//返回1的个数
bs.any();//是否有1 
bs.all();//是否全为1 
bs.none();//是否没有1
bs.set();//全部设为1
bs.reset();//全部设为0
bs.set(p);//bs[p]=1
bs.reset(p);//bs[p]=0;
bs.flip() //全部取反
bs.flip(p) //bs[p]=!bs[p]
bs.test(p) //相当于bs[p],只是不会越界
bs.to_ulong() //返回它转换为unsigned long的结果，如果超出范围则报错
bs.to_ullong() //返回它转换为unsigned long long的结果
bs.to_string() //返回它转换为string的结果
bs._Find_first() //返回第一个1的下标，若没有则为bs.size()
bs._Find_next(pos) //返回大于pos的第一个1的下标，若没有则为bs.size()
```

### 常用算法（algorithm）

#### sqrtf,sqrt,sqrtl 返回值为float,double,long double

#### unique

相邻元素去重，不删除元素，将相邻不重复的元素向前移。unique(arr,arr+n)-arr是剩余的成员数。

#### nth_element

用类似快排的方法分治解决

nth_element(arr,arr+m,arr+n)  arr[m]变为数组中升序中第m个数，左边的数比它小，右边的比他大。

#### lower_bound    upper_bound

lower_bound:二分查找有序数组中第一个大于等于目标值的位置。

upper_bound：二分查找有序数组中第一个大于目标值的位置。

int *p=lower_bound(arr,arr+n,x);

重载：lower_bound(arr,arr+n,x,greater<int\>);

```c++
bool cmp(pii a,pii b){
    return a.S<b.S;
}
```

越界返回arr+n

#### max_element       min_element

查找最值所在位置 int*p=max_element(arr,arr+n);

### 并查集

#### 普通并查集

##### 初始化

```c++
int fa[maxn];
void init(int w){
    for(int i=1;i<=w;i++)fa[i]=i;
}
```

##### 查找

```c++
int find(int i){
    if(fa[i]==i)return i;
    else return find(fa[i]);
}
```

##### 路径压缩

```c++
int find(int i){
    if(fa[i]!=i){
        fa[i]=find(fa[i]);
    }//该步骤进行了路径压缩，相当于将该节点接到根节点上，下次查询节约时间
    return fa[i];
}
```

##### 合并

```c++
void uni(int i,int j){
    int fa_i=find(i),fa_j=find(j);
    fa[fa_i]=fa_j;
}
```

##### 按秩合并

秩可以看作是树高或者树的大小，合并时将较小的树并到较大的上面

```c++
int rk[maxn];
void uni (int i, int j) {
    int fa_i = find(i), fa_j = find(j);
    if(rk[fa_i] < rk[fa_j])fa[fa_i] = fa_j;
    else if(rk[fa_i] > rk[fa_j])fa[fa_j] = fa_i;
    else fa[fa_j] = fa_i, rk[fa_i]++;
}//在路径压缩过程中可能会改变秩，但是我们忽略它。
```

#### 扩展域并查集

并查集本身只是维护同类这一种关系，如果有多种关系并且关系之间有传递性就可以使用扩展域并查集，对m种关系开n*m的fa数组来维护

[[P1525 [NOIP2010 提高组\] 关押罪犯 - 洛谷](https://www.luogu.com.cn/problem/P1525)]:两种关系，敌人的敌人是朋友

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn=2e6+10;
int fa[maxn];
int n,m,cnt;
struct node{
    int u,v,s;
}arr[maxn];

bool cmp (node x, node y){
    return x.s > y.s;
}

void init () {
    for (int i = 1; i <= 2 * n; i++){
        fa[i] = i;
    }
}

int find (int i) {
    if (fa[i] != i) fa[i] = find(fa[i]);
    return fa[i];
}

//因为不清楚具体的情况，所以要连接每种情况
void uni (int i, int j) {//连接的时候让i和j+n相连，i+n和j相连
    int fa_i = find(i), fa_j = find(j + n);
    fa[fa_i] = fa_j;
    fa_i = find(i + n), fa_j = find(j);
    fa[fa_i] = fa_j;
}



void work(){
    cin >> n >> m;
    init();
    for (int i = 1; i <= m; i++) {
        cin >> arr[i].u >> arr[i].v >> arr[i].s;
    }
    sort (arr + 1, arr + 1 + m, cmp);
    int ans = 0;
    for (int i = 1; i <= m; i++) {
        if(find(arr[i].u) != find(arr[i].v)){
            uni(arr[i].u,arr[i].v);
        }
        else {ans = arr[i].s;break;}
    }
    cout << ans << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    work();
}
```

[[P2024 [NOI2001\] 食物链 - 洛谷](https://www.luogu.com.cn/problem/P2024)]三种关系，但关系仍是个循环，仍具有传递性



```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn=2e6+10;
int fa[maxn];
int n,m,cnt;

void init () {
    for (int i = 1; i <= 3 * n; i++){
        fa[i] = i;
    }
}

int find (int i) {
    if (fa[i] != i) fa[i] = find(fa[i]);
    return fa[i];
}

void uni1 (int i, int j) { // i 吃 j
    int fa_i = find(i), fa_j = find(j + n);
    fa[fa_i] = fa_j;
    fa_i = find(i + n), fa_j = find(j + 2 * n);
    fa[fa_i] = fa_j;
    fa_i = find(i + 2 * n), fa_j = find(j);
    fa[fa_i] = fa_j;
}

void uni2 (int i, int j) { // i 和 j 同类
    int fa_i = find(i), fa_j = find(j);
    fa[fa_i] = fa_j;
    fa_i = find(i + n), fa_j = find(j + n);
    fa[fa_i] = fa_j;
    fa_i = find(i + 2 * n), fa_j = find(j + 2 * n);
    fa[fa_i] = fa_j;
}

bool check (int x, int y) {
    return find(x) == find(y);
}



void work(){
    cin >> n >> m;
    init();
    int x, y, z, ans = 0;
    for (int i = 1; i <= m; i++) {
        cin >> x >> y >> z;
        if (y > n || z > n) {
            ans++; continue;
        }
        if (x == 1) {
            if (check (y + n, z) || check (y + n, z + 2 * n)) ans++;
            else uni2(y, z);
        }
        if (x == 2) {
            if (y == z || check (y + n, z + n) || check (y + n, z)) ans++;
            else uni1(y, z);
        }
    }
    cout << ans << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    work();
}
```

#### 带权并查集

记录dis或者size等信息

```c++
	int fa[maxn]={0};
size:
	int sz[maxn]={0};  //sz数组只有根节点的才有意义
	for(int a=1;a<maxn;a++){fa[a]=a;sz[a]=1;}//初始化

	//查询操作没变化

	void uni(int x,int y){
        x=find(x);y=find(y);
        if(x==y)return ;
        if(sz[y]>sz[x])swap(x,y);//把小的并到大的上面
        sz[x]+=sz[y];fa[y]=x;//将y并到x上面
    }

dis:
	int dis[maxn]={0}; //记录该节点到根节点的距离,要从0开始
	for(int a=1;a<maxn;a++){fa[a]=a;}//初始化
	int find(int now){  //查询时路径压缩并且更新dis
        if(fa[now]!=now){
            int root=find(fa[now]);  //先递归再更新dis
            dis[now]+=dis[fa[now]];  //到根节点的距离只需要相加就可以
            fa[now]=root;
        }
        return fa[now];
    }

	void uni(int x,int y){  //将y并到x上
    	int x1=find(x),y1=find(y);
        dis[y1]+=dis[x]-dis[y]+1;  //只更新y祖宗节点的dis，子节点在需要使用时通过find操作更新
        fa[y1]=x1;
    }
```

#### 可撤销并查集

将每次合并操作放进一个栈里，要撤销最近一次合并就将栈顶的合并操作撤回，只能按照后进先出的顺序进行撤销，不能随意改变

[[F - A Certain Game](https://atcoder.jp/contests/abc314/tasks/abc314_f)]:

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define pii pair<int,int>
#define F first
#define S second
#define endl '\n'
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn = 2e6+10;
const int mod = 998244353;
int n,m,N=1;

int ans[maxn], siz[maxn], fa[maxn];

stack <pii> sta;

int find (int i) {
    if (fa[i] == i)return i;
    else return find(fa[i]);
}

void uni(int i,int j) { //按秩合并，将点数少的合并到点数多的，保证查询是O(logn)
    i = find(i); j = find(j);
    if (siz[i] < siz[j]) swap(i, j);
    sta.push({j,i}); siz[i] += siz[j]; fa[j] = i;
}

int qkp(int x, int y) {
    x %= mod; int res = 1;
    while (y) {
        if (y & 1) res = res * x % mod;
        x = x * x % mod;
        y >>= 1;
    }
    return res;
}

void work(){
    cin >> n;
    for (int i = 1; i <= n; i++) {
        fa[i] = i; siz[i] = 1;
    }
    int x, y;
    for (int i = 1; i < n; i++) {
        cin >> x >> y;
        uni(x, y);
    }
    while (!sta.empty()) {
        pii now = sta.top(); sta.pop();
        int t = qkp(siz[now.S], mod - 2);
        ans[now.F] += ans[now.S]; ans[now.F] %= mod;
        siz[now.S] -= siz[now.F]; fa[now.F] = now.F;
        ans[now.F] += siz[now.F] * t; ans[now.F] %= mod;
        ans[now.S] += siz[now.S] * t; ans[now.S] %= mod;
    }
    for (int i = 1; i <= n; i++) cout << ans[i] << ' ';
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

#### 可持久化并查集？



### 二叉树

满二叉树，每一层都是满的，d是深度，也就是层数，节点个数是2^d^-1

完全二叉树指的是除底层以外全都是满的，底层的节点从左到右连续存在的。

#### 遍历二叉树

遍历分为前序遍历，中序遍历，后序遍历，这里区别是中间节点是什么时候被遍历的

<img src="../../../Pictures/Screenshots/typora图片/graph.png" style="zoom:60%;" />

前序遍历 ：4213657    中序遍历： 1234567      后序遍历 ： 1325764

##### 非递归代码

```c++
struct node{
    int v;
    node *l,*r;
};
vector<node>res;
前序：
vector<node> preorderTraversal(node root){
    stack<node>sta;
    sta.push(root);
    while(!sta.empty()){
        node now=sta.top();
        sta.pop();
        res.push_back(now);
        if(now.r!=NULL)sta.push(*now.r);
        if(now.l!=NULL)sta.push(*now.l);
    }
    return res;
}
中序：
vector<node> inorderTraversal(node root){
    stack<node>sta;
    sta.push(root);
    while(!sta.empty()){
        while(sta.top()){
            sta.push(*sta.top().l);
        }
        sta.pop();
        if(!sta.empty()){
            res.push_back(sta.top().v);
            node tem=*sta.top().r;
            sta.pop();sta.push(tem);
        }
    }
    return res;
}
后序:
方法一：可以仿照前序通过中右左的顺序，然后翻转过来就是左右中
方法二：
vector<int> postorderTraversal(node root) {
    
}
```



#### 平衡二叉搜索树

左子树节点权值小于中间节点，右子树节点权值大于中间节点

查找，插入，删除一个节点的复杂度是O(logn)

### 线段树

要求是区间信息可合并，例如总和，最大/小值，最大公约数，最大/小前缀和([Ex - Rating Estimator](https://atcoder.jp/contests/abc292/tasks/abc292_h))

O（logn）的复杂度来单点修改，区间查询，区间修改

#### 基础操作

```c++
#define ls i<<1
#define rs i<<1|1

int arr[maxn];
struct node{
    int lson,rson,sum,lazy;
}tr[maxn<<2];

void pushup(int i,int l,int r){
    tr[i].sum = tr[ls].sum + tr[rs].sum;
}
//建树
void build(int i,int l,int r){
    if(l == r){
        tr[i].sum = arr[l];
        return ;
    }
    int mid = (l+r)/2;
    build(ls,l,mid);build(rs,mid+1,r);
    pushup(i,l,r);
}

//单点修改,区间查询（以区间和为例）
void change(int i,int l,int r,int w,int v){
    if(l == r){
        tr[i].sum += v;return ;
   	}
    int mid = (l+r)/2;
    if(w<=mid)change(ls,l,mid,w,v);
    else change(rs,mid+1,r,w,v);
    pushup(i,l,r);
}
int query(int i,int l,int r,int p,int q){
    if(l==p&&r==q)return tr[i].sum;
    int mid = (l+r)/2;
    if(q<=mid)return query(ls,l,mid,p,q);
    else if(p>=mid+1)return query(rs,mid+1,r,p,q);
    else return query(ls,l,mid,p,mid)+query(rs,mid+1,r,mid+1,q);
}

//区间修改，区间查询(lazy非永久化)
void put_lazy(int i,int l,int r,int v){
    tr[i].lazy += v;//这里记得是+=,而不是=
    tr[i].sum += (r-l+1) * v;
}
void pushdown(int i,int l,int r){
    int mid = (l+r)/2;
    put_lazy(ls,l,mid,tr[i].lazy);
    put_lazy(rs,mid+1,r,tr[i].lazy);
    tr[i].lazy = 0;
}
void change(int i,int l,int r,int p,int q,int v){
    if(l==p&&r==q){
        put_lazy(i,l,r,v);
        return ;
   	}
    int mid = (l+r)/2;//非永久懒标记，记得在mid后面记得pushdown
    if(tr[i].lazy)pushdown(i,l,r);
    if(q<=mid)change(ls,l,mid,p,q,v);
    else if(p>=mid+1)change(rs,mid+1,r,p,q,v);
    else change(ls,l,mid,p,mid,v),change(rs,mid+1,r,mid+1,q,v);
    pushup(i,l,r);
}
int query(int i,int l,int r,int p,int q){
    if(l==p&&r==q)return tr[i].sum;
    int mid = (l+r)/2;//非永久懒标记，记得在mid后面记得pushdown
    if(tr[i].lazy)pushdown(i,l,r);
    if(q<=mid)return query(ls,l,mid,p,q);
    else if(p>=mid+1)return query(rs,mid+1,r,p,q);
    else return query(ls,l,mid,p,mid)+query(rs,mid+1,r,mid+1,q);
}

//区间修改，区间查询(lazy永久化)
//删去pushdown(),put_lazy()函数及相关调用
void change(int i,int l,int r,int p,int q,int v){//不能再采用pushup，因为上层可能比下层多加了lazy，pushup会导致这部分lazy消失
    tr[i].sum+=(q-p+1)*v;
    if(l==p&&r==q){
        tr[i].lazy+=v;
        return ;
   	}
    int mid = (l+r)/2;
    if(q<=mid)change(ls,l,mid,p,q,v);
    else if(p>=mid+1)change(rs,mid+1,r,p,q,v);
    else change(ls,l,mid,p,mid,v),change(rs,mid+1,r,mid+1,q,v);
}
int query(int i,int l,int r,int p,int q,int v){
    if(l==p&&r==q)return tr[i].sum+(q-p+1)*v;
    int mid = (l+r)/2;
    v+=tr[i].lazy;
    if(q<=mid)return query(ls,l,mid,p,q,v);
    else if(p>=mid+1)return query(rs,mid+1,r,p,q,v);
    else return query(ls,l,mid,p,mid,v)+query(rs,mid+1,r,mid+1,q,v);
}
```

[[P3373 【模板】线段树 2 - 洛谷](https://www.luogu.com.cn/problem/P3373)]

对于乘和加两种操作，我们规定先乘后加，所以乘法会同时修改sum和表示加的lazy，传标记的时候先传乘标记，再传加标记，并不能使用懒标记永久化，因为对于子区间进行乘法时，无法得到对当前大区间的sum造成的修改是多少，同时又不能使用pushup，所以无法在过程中维护。

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn = 1e5+10;
int mod = 1e9+7;
int arr[maxn];
int n,m,N=1;

#define ls i<<1
#define rs i<<1|1

struct node{
    int sum,add_lazy,mul_lazy=1;
}tr[maxn<<2];

void pushup(int i){
    tr[i].sum = (tr[ls].sum + tr[rs].sum)%mod;
}

void build(int i,int l,int r){
    if(l==r){
        tr[i].sum = arr[l];return ;
    }
    int mid = (l + r) / 2;
    build(ls,l,mid);build(rs,mid+1,r);
    pushup(i);
}

void put_add(int i,int l,int r,int v){
    tr[i].add_lazy += v; tr[i].add_lazy%=mod;
    tr[i].sum += ((r-l+1)*v)%mod; tr[i].sum%=mod;
}

void put_mul(int i,int l,int r,int v){
    tr[i].add_lazy *= v; tr[i].add_lazy%=mod;
    tr[i].mul_lazy *= v; tr[i].mul_lazy%=mod;
    tr[i].sum *= v; tr[i].sum%=mod;
}

void pushdown(int i,int l,int r){
    int mid = (l+r)/2;
    put_mul(ls,l,mid,tr[i].mul_lazy);put_mul(rs,mid+1,r,tr[i].mul_lazy);
    put_add(ls,l,mid,tr[i].add_lazy);put_add(rs,mid+1,r,tr[i].add_lazy);
    tr[i].add_lazy=0;tr[i].mul_lazy=1;
}


void change_add(int i,int l,int r,int p,int q,int v){
    if(p==l&&q==r){
        put_add(i,l,r,v);return ;
    }
    int mid = (l + r) / 2;
    if(tr[i].add_lazy||tr[i].mul_lazy!=1){
        pushdown(i,l,r);
    }
    if(q<=mid)change_add(ls,l,mid,p,q,v);
    else if(p>mid)change_add(rs,mid+1,r,p,q,v);
    else change_add(ls,l,mid,p,mid,v),change_add(rs,mid+1,r,mid+1,q,v);
    pushup(i);
}

void change_mul(int i,int l,int r,int p,int q,int v){
    if(p==l&&q==r){
        put_mul(i,l,r,v);return ;
    }
    int mid = (l + r) / 2;
    if(tr[i].add_lazy||tr[i].mul_lazy!=1){
        pushdown(i,l,r);
    }
    if(q<=mid)change_mul(ls,l,mid,p,q,v);
    else if(p>mid)change_mul(rs,mid+1,r,p,q,v);
    else change_mul(ls,l,mid,p,mid,v),change_mul(rs,mid+1,r,mid+1,q,v);
    pushup(i);
}

int query(int i,int l,int r,int p,int q){
    if(l==p&&r==q)return tr[i].sum;
    if(tr[i].add_lazy||tr[i].mul_lazy!=1){
        pushdown(i,l,r);
    }
    int mid = (l + r) / 2;
    if(q<=mid)return query(ls,l,mid,p,q);
    else if(p>mid)return query(rs,mid+1,r,p,q);
    else return (query(ls,l,mid,p,mid)+query(rs,mid+1,r,mid+1,q))%mod;
}

void work(){
    cin >> n >> m >> mod;
    for(int i=1;i<=n;i++){
        cin>>arr[i];
        arr[i]%=mod;
    }
    build(1,1,n);
    int op,p,q,v;
    while(m--){
        cin>>op>>p>>q;
        if(op==1){
            cin>>v;
            v%=mod;
            change_mul(1,1,n,p,q,v);
        }else if(op==2){
            cin>>v;
            v%=mod;
            change_add(1,1,n,p,q,v);
        }else {
            cout<<query(1,1,n,p,q)<<endl;
        }
    }
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

#### 动态开点

```c++
//动态开点，当修改n次，查询m次，n>>m时，可将复杂度降到O(mlogn)并且降低需要的空间
//如果运用永久化懒标记的话，只会在modify过程中开点，但是非永久化的话在下传lazy(也就是在put_lazy函数)时会开新点。两者在query时遇到空点都可以直接返回
struct node{
	int ls,rs,sum,lazy;
}tr[min((2*n-1),4*log2(n)*m)];
int cnt; //记录现有的点的数量
```

```c++
//永久化懒标记
void change(int &i,int l,int r,int p,int q,int v){//对i进行址传递可以直接修改传下来的tr[].ls或者tr[].rs
    if(!i)i=++cnt;
	tr[i].sum+=v*(q-p+1);
	if(p==l&&q==r){
		tr[i].lazy+=v;return ;
	}
	int mid = (l+r)/2;
    if(q<=mid)change(tr[i].ls,l,mid,p,q,v);
    else if(p>=mid+1)change(tr[i].rs,mid+1,r,p,q,v);
    else change(tr[i].ls,l,mid,p,mid,v),change(tr[i].rs,mid+1,r,mid+1,q,v);
}

int query(int i,int l,int r,int p,int q,int v){
    if(!i)return v*(q-p+1);
	if(p==l&&q==r)return tr[i].sum+(q-p+1)*v;
	v+=tr[i].lazy;
    int mid = (l+r)/2;
    if(q<=mid)return query(tr[i].ls,l,mid,p,q,v);
    else if(p>=mid+1)return query(tr[i].rs,mid+1,r,p,q,v);
    else return query(tr[i].ls,l,mid,p,mid,v)+query(tr[i].rs,mid+1,r,mid+1,q,v);
}

void work(){
    int rt = 0;//通过rt来进行修改和查询，不需要建树过程，后面可持久化线段树会有多个rt,直接使用rt[]数组即可
    change(rt,1,n,p,q,v);
    query(rt,1,n,p,q,0);
}
```

```c++
//非永久化懒标记
void pushup(int i){
    tr[i].sum = tr[tr[i].ls].sum + tr[tr[i].rs].sum;
}

void put_lazy(int &i,int l,int r,int v){//put_lazy上用址传递开新点
    if(!i)i=++cnt;
    tr[i].sum += (r-l+1)*v;
    tr[i].lazy += v;
}

void pushdown(int i,int l,int r){
    int mid = (l+r)/2;
    put_lazy(tr[i].ls,l,mid,tr[i].lazy);
    put_lazy(tr[i].rs,mid+1,r,tr[i].lazy);
    tr[i].lazy=0;
}

void change(int &i,int l,int r,int p,int q,int v){//对i进行址传递可以直接修改传下来的tr[].ls或者tr[].rs
    if(!i)i=++cnt;
	if(p==l&&q==r){
		put_lazy(i,l,r,v);return ;
	}
	int mid = (l+r)/2;
    if(tr[i].lazy)pushdown(i,l,r);
    if(q<=mid)change(tr[i].ls,l,mid,p,q,v);
    else if(p>=mid+1)change(tr[i].rs,mid+1,r,p,q,v);
    else change(tr[i].ls,l,mid,p,mid,v),change(tr[i].rs,mid+1,r,mid+1,q,v);
    pushup(i);
}

int query(int i,int l,int r,int p,int q){
    if(!i)return 0;
	if(p==l&&q==r)return tr[i].sum;
    int mid = (l+r)/2;
    if(tr[i].lazy)pushdown(i,l,r);
    if(q<=mid)return query(tr[i].ls,l,mid,p,q);
    else if(p>=mid+1)return query(tr[i].rs,mid+1,r,p,q);
    else return query(tr[i].ls,l,mid,p,mid)+query(tr[i].rs,mid+1,r,mid+1,q);
}
```

#### 线段树上二分

```c++
//序列每个数都是正数，询问任意[l,r]上第一个前缀和不小于v的下标
int query(int i,int l,int r,int p,int q,int v,int &sum){
    if(l==p&&r==q){
        if(tr[i].sum+sum<v){
            sum+=tr[i].sum;//当前区间没有希望成为解的话，累加当前区间和到sum，准备继续右边查询
            return -1;
        }
    }
    if(l==r)return l;
    int mid = (l+r)/2;
    if(q<=mid)return query(ls,l,mid,p,q,v,sum);
    else if(p>=mid+1)return query(rs,mid+1,r,p,q,v,sum);
    else {
        int t = query(ls,l,mid,p,mid,v,sum);
        if(t!=-1)return t;//先优先访问左节点，如果无解再访问右节点
        else return query(rs,mid+1,r,mid+1,q,v,sum);
    }
}
```

#### 权值线段树

权值线段树指的是将原数组转化为桶，再建立线段树。

此时线段树的前缀和就成了小于等于该数的个数，可以和线段树上二分结合求带单点修改的全局第k小问题

#### 可持久化线段树？？？

使用动态开点来维护多个版本的线段树

##### 单点修改

单点修改时创建一个新的版本,新的版本只会有logn个节点是变化的，其余节点都借用已存在的版本。

使用可持久化权值线段树（主席树）+线段树上二分可以用来求静态区间（不带修改）区间第k小

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn = 2e5+10;
int arr[maxn],order[maxn];
int n,m,N=1;

struct node{
    int ls,rs,sum;
}tr[maxn*30];//开30(2+logn)倍大小
int cnt;
int rt[maxn];//多个版本的根

void build(int &i,int l,int r){//需要用build先建空树
    i=++cnt;
    if(l==r)return ;
    int mid = (l+r)/2;
    build(tr[i].ls,l,mid);
    build(tr[i].rs,mid+1,r);
}

void pushup(int i){
    tr[i].sum = tr[tr[i].ls].sum + tr[tr[i].rs].sum;
}

void change(int &i,int l,int r,int pre,int pos){//址传递，更新新的节点
    i=++cnt;
    tr[i]=tr[pre];
    if(l==r){
        tr[i].sum++;return ;
    }
    int mid = (l+r)/2;
    if(pos<=mid)change(tr[i].ls,l,mid,tr[pre].ls,pos);
    else change(tr[i].rs,mid+1,r,tr[pre].rs,pos);
    pushup(i);
}

int query(int rt1,int rt2,int l,int r,int k,int sum){//线段树上二分
    if(l==r)return order[l];
    int mid = (l+r)/2;
    int d = tr[tr[rt2].ls].sum-tr[tr[rt1].ls].sum;
    if(d+sum>=k)return query(tr[rt1].ls,tr[rt2].ls,l,mid,k,sum);
    else {
        sum += d;
        return query(tr[rt1].rs,tr[rt2].rs,mid+1,r,k,sum);
    }
}

void work(){
    cin>>n>>m;
    for(int i=1;i<=n;i++){
        cin>>arr[i];
        order[i]=arr[i];
    }
	//离散化
    sort(order+1,order+1+n);
    int len = unique(order+1,order+1+n) - order - 1;
    for(int i=1;i<=n;i++){
        arr[i]=lower_bound(order+1,order+1+len,arr[i])-order;
    }
    //建树
    build(rt[0],1,n);
    for(int i=1;i<=n;i++){
        change(rt[i],1,n,rt[i-1],arr[i]);
    }
    //查询
    int l,r,k;
    while(m--){
        cin>>l>>r>>k;
        cout<<query(rt[l-1],rt[r],1,n,k,0)<<endl;
    }
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

##### 区间修改？？？

#### 可持久化权值线段树(主席树)

解决静态区间任意区间第k小

### 树状数组

[F - Compare Tree Weights](https://atcoder.jp/contests/abc406/tasks/abc406_f) 在树上使用树状数组

![树状数组](../../../Pictures/Screenshots/typora图片/树状数组.png)

```c++
int arr[maxn];//原数组
int tr[10005];//树状数组，k代表二进制中n最后面的1所代表的数值，tr[n]=arr[n]+arr[n-1]+...+arr[n-k+1]。例如tr[7]=a[7],tr[16]为arr[1]到arr[16]的和。

int lowbit(int x){
	return x&(-x);
}//查询最后面的1所代表的数值大小。
```

```c++
//单点修改，区间查询，解决动态区间和问题
void add(int x,int v){//将a[x]加上v
    while(x<=n){
        tr[x]+=v;
        x+=lowbit(x);   //跳往父节点，该节点和父节点在查询时不会同时被加
    }
}//修改

int query(int x){
	int res=0;
	while(x){
    	res+=tr[x];
    	x-=lowbit(x);
	}
	return res;
}//查询前x个数和，将复杂度降至logn

int main(){
    int k;
    for(int a=1;a<=n;a++){ //初始化赋值
        cin>>k;add(a,k);
    }
}
```

```c++
//区间修改，单点查询，利用差分，函数与上面相同
int main(){
    for(int a=1;a<=n;a++)cin>>arr[a];
    
    //区间修改，从l到r每个都加t，只需要改tr[l]和tr[r+1]
    add(l,t);add(r+1,-t);
    
    //单点查询x
    cout<<arr[x]+query(x);
}
```

```c++
//区间修改，区间查询，用两个树状数组分别维护d_i和i*d_i
```

$$
&\sum_{i=1}^{r} a_i= \sum_{i=1}^r\sum_{j=1}^i d_j=\\
        =&\sum_{i=1}^r d_i\times(r-i+1)\\
        =&\sum_{i=1}^r d_i\times (r+1)-\sum_{i=1}^r d_i\times i
$$



### ST表（RMQ，区间gcd问题）

满足区间可覆盖

```c++
int lg[100005],st[20][100005];

void build_st(int arr[]){
    for(int i=1;i<=n;i++){
        st[0][i]=arr[i];
    }
    for(int i=1;(1<<i)<=n;i++){
        for(int j=1;j+(1<<i)-1<=n;j++){
            st[i][j]=max(st[i-1][j],st[i-1][j+(1<<(i-1))]);
        }
    }

    lg[1]=0;
    for(int i=2;i<=n;i++)lg[i]=lg[i/2]+1;
}

int get_st(int l,int r){
    int t = lg[r-l+1];
    return max(st[t][l],st[t][r-(1<<len)+1]);
}
```

### 可持久化数据结构

存储了对应数据结构所有的历史版本，并通过数据重复使用减少复杂度

### 分块



#### 莫队算法

从 $[l,r]$的答案能够 $O(T)$扩展到$[l-1,r],[l+1,r],[l,r+1],[l,r-1]$（即与$ [l,r] $相邻的区间）的答案

将所有询问排序，顺序处理每个询问，暴力从上一个区间的答案转移到下一个区间答案（一步一步移动即可）

排序方法：对于询问$ [l,r]$, 以 $l$ 所在块的编号为第一关键字，$r$​ 为第二关键字从小到大排序

```c++
int l=1,r=0,sum=0;
for(int i=1;i<=m;i++){
	while(l>q[i].l)add(--l);
	while(r<q[i].r)add(++r);
	while(l<q[i].l)del(l++);
	while(r>q[i].r)del(r--);
    ans[q[i].id]=sum;
}
```

l,r两个指针有四个移动方向，要保证$--l$在$r--$前，$++r$在$l++$​前

奇偶化排序优化：奇偶化排序即对于属于奇数块的询问，r 按从小到大排序，对于属于偶数块的排序，r 从大到小排序，一般情况下，这种优化能让程序快 30% 左右。

##### 带修莫队

把区间$[l,r]$变为$[l,r,time]$， $[l,r,time]$的答案能够扩展到$[l-1,r,time],[l+1,r,time],[l,r+1,time],[l,r-1,time],[l,r,time-1],[l,r,time+1]$​的答案

排序方法： 按对于询问$[l,r,time]$，我们以$l$所在块的编号为第一关键字，以$r$所在块的编号为第二关键字，以$time$​为第三关键字。

块大小设置为$n^{2/3}$，时间复杂度为$O(n^{5/3})$。

```c++
int l=1,r=0,time=max_time,sum=0;

void update(int x){
    //用修改来更新数据
    //如果此次修改影响了当前维护的[l,r]，就将修改造成的影响进行维护
}
for(int i=1;i<=m;i++){
	while(l>q[i].l)add(--l);
	while(r<q[i].r)add(++r);
	while(l<q[i].l)del(l++);
	while(r>q[i].r)del(r--);
    while(time>q[i].time)update(time--,-1);
    while(time<q[i].time)update(++time,1);
    ans[q[i].id]=sum;
}
```



## 图论

### 度

在无向图中，定点所连接的边的个数为顶点的度，所有点度之和是边数的二倍，度为奇数的点有偶数个。

在有向图中，入度出度之和为顶点的度。

### 完全图

无向图满足任意两个点都存在一条边连接称为完全图。

有n个点的完全图有n*(n-1)/2条边。

### 路径（x个边相连）

重边：从一个点到另一个点有两条直接相连的路径。

自环：从某个顶点出发联向它自身的边。

若无边权，那么路径的长度为路径包含的边数。

一条路经起点和终点相同，称此路径为回路，也叫环。

简单图：不存在重边和自环的图。

### Cayley公式

由n个有编号（互不相同）的点构成的无根树的个数为n^n-2^。

### 图的存储

#### 邻接矩阵

#### 邻接表

用结构体或者pair

```c++
struct node{
int v;//边的终点
int w;//权值
};
vector<node> adj[105];//相当于二维数组，第一维是各个顶点，第二维是这个定点延伸出来的边。

void add(int u,int v,int w){
adj[u].push_back(node{v,w});}//以u为起点，v为终点，权值为w。

```

#### 链式前向星？

### 图的遍历

#### DFS（深度优先搜索）

用栈来写

![](../../../Pictures/Screenshots/typora图片/深度优先遍历DFS (1).png)

![](../../../Pictures/Screenshots/typora图片/深度优先遍历DFS (2).png)

```c++
vector <int> vec[10];
stack<int> sta;
bool vis[maxn];
sta.push(1);arr[1]=1;
while(!sta.empty()){
    if(vec[sta.top()].size()&&vis[vec[sta.top()].back()]==0){
        int k= vec[sta.top()].back();
        vec[sta.top()].pop_back();
        sta.push(k);vis[k]=1;
    }
    else st.pop();
}
```

没有用栈，通过递归来访问

![](../../../Pictures/Screenshots/typora图片/深度优先遍历DFS.png)

```c++
vector <int> vec[maxn];
void dfs(int now,int fa){
    for(int a=0;a<vec[now].size();a++){
        int t=vec[now][a];
        if(t==fa)continue;//跳过父节点，不需要vis数组
        dfs(t,now);
    }
}
```

#### BFS (宽度优先搜索)

用队列

### 最近公共祖先

#### 倍增算法

```c++
int fa[maxn][25];//从某点开始向上2的k幂次的祖先
int LG[maxn]={0};//模拟log2
int d[maxn]={0};//深度
vector<int>vec[maxn];
void dfs(int now,int f){
    d[now]=d[f]+1;
    fa[now][0]=f;
    for(int a=1;(1<<a)<=d[now];a++){
        fa[now][a]=fa[fa[now][a-1]][a-1];
    }
    for(auto x :vec[now])if(x!=f)dfs(x,now);
}
int lca(int x,int y){
    if(d[x]<d[y])swap(x,y);
    int t=d[x]-d[y];
    while(t){
        int tem=t&(-t);
        t-=tem;
        x=fa[x][LG[tem]];
    }
    if(x==y)return x;   //先提升至同一深度，判断是否相同
    for(int a=LG[d[x]];a>=0;a--){
        if(fa[x][a]!=fa[y][a]){
            x=fa[x][a];y=fa[y][a];
        }
    }
    return fa[x][0];
}
signed main(){
    int x,y;int n;cin>>n;
    for(int a=1;a<=n;a++){
         LG[a]=LG[a-1]+(1<<(LG[a-1]+1)==a);
    }
    
    //输入数据
    
    dfs(1,0);
    cout<<lca(x,y);
}
```

#### Tarjan算法

离线算法，需要提前知道所有查询，而倍增算法不需要

O(n+2*q)，n是点的个数，q是询问次数

```c++
#include<bits/stdc++.h>
//#pragma GCC optimize("O0")
using namespace std;
#define int long long
#define pii pair<int,int>
#define F first
#define S second
const int maxn=5e5+10;
int ans[maxn]={0};//查询的答案数组
bool vis[maxn]={0};
vector<int>vec[maxn];
vector<pii>qu[maxn];//假如第3次查询的是点6和2，则有qu[2].push_back(6,3),qu[6].push(2,3)
int fa[maxn]={0};
int find(int a){
	if(fa[a]!=a){
		fa[a]=find(fa[a]);
	}
	return fa[a];
}
void dfs(int i,int f){
	vis[i]=1;
    //入点，打标记
	for(auto x : vec[i]){
		if(x==f)continue;
		if(vis[x]==0)dfs(x,i);
	}
    //搜点
	for(auto x : qu[i]){
		if(vis[x.F]==1&&ans[x.S]==0){//第x.S次查询尚未有结果并且另一个点已经结束查询
			ans[x.S]=find(x.F);//那么公共祖先就是另一个点的父节点
		}
	}
    //是否有查询的点跟现在这个点有关
	fa[i]=f;//出点，更新父亲
}
signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    int n,m,s;cin>>n>>m>>s;
    int x,y;
    for(int a=1;a<n;a++){
    	cin>>x>>y;vec[x].push_back(y);vec[y].push_back(x);
	}
	for(int a=1;a<=m;a++){
		cin>>x>>y;
		qu[x].push_back({y,a});
		qu[y].push_back({x,a});
	}
	for(int a=1;a<=n;a++)fa[a]=a;
	dfs(s,0);
	for(int a=1;a<=m;a++)cout<<ans[a]<<endl;
}
```

### 树链剖分

将整棵树剖分为若干条链，使它组合成线性结构，然后用其他的数据结构维护信息

#### 重链剖分

重子节点： 表示其子节点中子树最大的子结点。如果有多个子树最大的子结点，取其一。如果没有子节点，就无重子节点。

轻子节点：剩余的子节点。

节点到重子节点的边被称为重边，到轻子节点的边被称为轻边。

将首尾相连的重边构成的链成为重链，单独的点也看作重链，这样的话整个树就被分成了若干条重链。

```c++
int fa[maxn],dep[maxn],siz[maxn],son[maxn];//dfs1中处理
int top[maxn],dfn[maxn],rnk[maxn],cnt;//dfs2中处理
//top指的是该点所处的重链的顶点,dfn指的是该点的dfs序号，rnk指的是dfs序号原本指的节点序号，rnk[ dfn[x] ] = x;
void dfs1 (int i,int f) {
    fa[i] = f; dep[i] = dep[f] + 1;
    siz[i] = 1;
    for (auto to : vec[i]) {
        if (to != f) {
            dfs1 (to, i);
            siz[i] += siz[to];
            if (siz[to] >= siz[son[i]]) son[i] = to;
        }
    }
}

void dfs2 (int i, int f, int tp) {
    top[i] = tp; dfn[i] = ++ cnt; rnk[cnt] = i;
    if (son[i]) dfs2 (son[i], i ,tp);// 优先对重儿子进行 DFS，可以保证同一条重链上的点 DFS 序连续
    for (auto to : vec[i]) {
        if (to != f && to != son[i]) dfs2(to, i, to);
    }
}
```

##### 性质

树上每个节点都属于且仅属于一条重链

一条重链上的点的深度各不相同

同一条重链上的点 DFS 序连续（用来维护路径）

一颗子树内的 DFS 序是连续的（用来维护子树）

当我们向一条轻边转移时，所在子树的大小至少会除以二

树上的每条路径都可以被拆分成不超过$O(log n)$​条重链

##### 求LCA

```c++
int LCA (int u,int v) {
	while (top[u] != top[v]) {
        if (dep[top[u]] > dep[top[v]]) {
            u = fa[top[u]];
        } else v = fa[top[v]];
    }
    return dep[u] > dep[v] ? v : u;
}
```

#### 长链剖分？？？

### 拓扑排序

DAG：有向无环图

作用：判断有向图是否有环

```c++
int rd[maxn];
vector<int>vec[maxn];
queue<int>que;
for(int i=1;i<=n;i++){
    cin >> x >> y;
    vec[x].push_back(y);
    rd[y]++;
}
for(int i=1;i<=n;i++){
    if(!rd[i])que.push(i);
}
while(!que.empty()){
    int now = que.front(); que.pop();
    for(auto to : vec[now]){
        rd[to]--;
        if(!rd[to])que.push(to);
    }
}
```

### 树（无环的连通图）

[[P3304 [SDOI2013\] 直径 - 洛谷](https://www.luogu.com.cn/problem/P3304)] 求直径长度和被所有直径经过的边的数量

根据所有直径共享相同中点的性质，根据中点数量为1个还是2个分类讨论

```c++
#include<bits/stdc++.h>
using namespace std;
#define F first
#define S second
#define int long long
#define pii pair<int,int>
#define endl '\n'
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn = 2e6+10;
int n,m,k,x,y,z,N=1;
int fa[maxn];
int nxt[maxn][3];
int sub[maxn][3];
vector<pii>vec[maxn];
vector<int>ans;

int sum;

void deliver(int t,int now){
    for(int i=2;i>=now;i--){
        sub[t][i]=sub[t][i-1];
        nxt[t][i]=nxt[t][i-1];
    }
}

void dfs(int i,int f){//找出前三长的子链
    for(auto xx : vec[i]){
        if(xx.F==f)continue;
        dfs(xx.F,i);
        int tem = sub[xx.F][0]+xx.S;
        if(tem>sub[i][0]){
            deliver(i,1);
            sub[i][0]=tem;
            nxt[i][0]=xx.F;
        }else if(tem>sub[i][1]){
            deliver(i,2);
            sub[i][1]=tem;
            nxt[i][1]=xx.F;
        }else if(tem>sub[i][2]){
            sub[i][2]=tem;
            nxt[i][2]=xx.F;
        }
    }
}

void dfs1(int i,int f){
    for(auto xx : vec[i]){
        if(xx.F==f)continue;
        if(xx.F==nxt[i][0]){
            fa[xx.F]=max(sub[i][1],fa[i])+xx.S;
        }else fa[xx.F]=max(sub[i][0],fa[i])+xx.S;
        dfs1(xx.F,i);
    }
}

void get_sum(int i){//如果子链中只有一个最大，那么这个链必经，否则不存在必经
    if(sub[i][0]!=sub[i][1]){
        get_sum(nxt[i][0]);sum++;
    }
}

void solve1(){
    int t = ans[0];
    if(sub[t][1]>sub[t][2]){//如果第二长的链和第三长的链长度不同，那么前两个方向都是必经的
        get_sum(nxt[t][0]);
        get_sum(nxt[t][1]);
        sum+=2;
    }else if(sub[t][0]>sub[t][1]){//只有最长的链必经
        get_sum(nxt[t][0]);sum++;
    }
}


void work(){
    cin>>n;
    for(int i=1;i<n;i++){
        cin>>x>>y>>z;
        vec[x].push_back({y,z});
        vec[y].push_back({x,z});
    }
    dfs(1,0);
    dfs1(1,0);
    int tem = inf;
    for(int i=1;i<=n;i++){
        tem=min(tem,max(fa[i],sub[i][0]));
    }
    for(int i=1;i<=n;i++){
        if(tem==max(fa[i],sub[i][0]))ans.push_back(i);
    }
    cout<<sub[ans[0]][0]+max(fa[ans[0]],sub[ans[0]][1])<<endl;
    memset(sub,0,sizeof(sub));
    if(ans.size()==1){
        dfs(ans[0],0);
        solve1();
    }
    else {
        sum=1;//两中点形成的边已经是必经的了
        dfs(ans[0],ans[1]);
        dfs(ans[1],ans[0]);
        get_sum(ans[0]);get_sum(ans[1]);//向两边找必经边
    }
    cout<<sum;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

#### 树的直径

[[Problem - D - Codeforces](https://codeforces.com/contest/2107/problem/D)]复杂度分析

两条不同的直径必然共享直径的中点。

##### 方法一

距离最长的路径,通过两次dfs查找，先任意定一个点，然后找距离它最远的点作为直径的一端，再找到距离这个点最远的点作为直径的另一端。

![直径](../../../Pictures/Screenshots/typora图片/直径.png)

无法处理负边权的情况,上图直径为(s,t),从y出发只会找到z

##### 方法二

通过dp找每个点最长的两个子链（不重合），然后遍历所有点寻找最大值，可以处理负边权（如果全是负边权是否需要特判？）



如果需要找到两个端点，可以开pos数组来记录最长的两条子链的延伸方向

```c++
int arr[maxn][2];//初始为0，如果子链只有负数的话仍是0，代表到此点终止
vector<int>vec[maxn];

void dfs(int i,int f){
    for(auto xx : vec[i]){
        if(xx==f)continue;
        dfs(xx,i);
        if(arr[xx][0]+1>=arr[i][0]){//在一个子树上只取最长的链
            swap(arr[i][0],arr[i][1]);
            arr[i][0]=arr[xx][0]+1;
        }
        else if(arr[xx][0]+1>=arr[i][1]){
            arr[i][1]=arr[xx][0]+1;
        }
    }
}

void work(){
    cin>>n;
    for(int i=1;i<n;i++){
        cin>>x>>y;
        vec[x].push_back(y);
        vec[y].push_back(x);
    }
    dfs(1,0);
    int ans = 0;
    for(int i=1;i<=n;i++){
        ans=max(ans,arr[i][0]+arr[i][1]+1);
        //找到子树最长的两个链就已经可以找到直径了，因为直径上最高的点可以取到max
        //如果想要进一步找出直径上的点或者找出真正以i点为根的最长的两个子链需要再一次dfs来维护，见树的中心代码
    }
    cout<<ans-1;
}
```

#### 树的重心

删除一个点，使剩余连接块中的最大树的点数尽可能的小，这个点为树的重心

##### 性质

- 重心一定在直径上。
- 树的重心如果不唯一，则至多有两个，且这两个重心相邻。
- 以树的重心为根时，所有子树的大小都不超过整棵树大小的一半。
- 树中所有点到某个点的距离和中，到重心的距离和是最小的。
- 把两棵树通过一条边相连得到一棵新的树，那么新的树的重心在连接原来两棵树的重心的路径上。

##### 方法一

多次dfs

```c++
int ans = 0, sum = inf;
int siz[maxn]; vector<int>vec[maxn];
void dfs1 (int i, int f) {//维护siz数组
    siz[i] = 1;
    for (auto xx : vec[i]) {
        if (xx == f) continue;
        dfs1 (xx, i);
        siz[i] += siz[xx];
    }
}
void dfs2 (int i, int f) {
    int maxx = 1;
    for (auto xx : vec[i]) {
        if (xx == f) continue;
        dfs2 (xx, i);
        maxx = max (maxx, siz[xx]);
    }
    maxx = max (maxx, n - siz[i]);//找出最大子树大小
    if (maxx < sum) {//ans就是一个重心
        sum = maxx; ans = i;
    }
}
int dfs3 (int i, int f) {//根据siz数组来算出距离和
    int res = 0;
    for (auto xx : vec[i]) {
        if (xx != f) {
            res += dfs3(xx, i);
            res += siz[xx];
        }
    }
    return res;
}
signed main(){
    dfs1(1,0); dfs2(1,0);
    dfs1(ans,0);//以ans为根重新计算siz数组
    cout << ans << ' ' <<dfs3(ans,0);
}
```

##### 方法二

树上dp

```c++
vector<int>vec[maxn];
int siz[maxn];
int ans[maxn];
int dfs1 (int i, int f) {
    siz[i] = 1;
    int res = 0;
    for (auto xx : vec[i]) {
        if (xx == f) continue;
        res += dfs1(xx, i);
        siz[i] += siz[xx];
    }
    res += siz[i] - 1;
    return res;
}

void dfs2 (int i, int f) {//树中所有点到某个点的距离和中，到重心的距离和是最小的
    if (i != 1) ans[i] = ans[f] + n - 2 * siz[i];
    //ans[i] = ans[f] + (n - siz[i]) - siz[i] 转移方程
    if (ans[i] < ans[ans[0]]) ans[0] = i;
    for (auto xx : vec[i]) {
        if (xx != f) {
            dfs2(xx, i);
        }
    }
}

dfs1(1,0); dfs2(1,0);cout << ans[0] << ' ' <<ans[ans[0]];
```

#### 树的中心

以树的中心为根时，从该根到每个叶子节点的路径长度的最大值最小

##### 性质

- 树的中心由1个或2个相邻的顶点组成。若树有偶数长度的直径，则中心是两个相邻点；若为奇数长度，则中心是一个点
- 所有直径必定经过树的中心，树的中心一定在直径上且趋近中点
- 树的所有直径共享相同中点

##### 方法

[[U392706 【模板】树的中心 - 洛谷](https://www.luogu.com.cn/problem/U392706)]

1.找到一条直径的两个端点，从两个端点同时bfs直到相交即可

2.利用树上dp求出以i为根的最长的两条链，然后根据定义找到中心

```c++
#include<bits/stdc++.h>
//#pragma GCC optimize("O0")
using namespace std;
#define F first
#define S second
#define int long long
#define pii pair<int,int>
#define endl '\n'
#define pause system("pause")
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn = 2e6+10;
const int mod = 1e9+7;
int sub[maxn][2];//子树中最长的两条链
int fa[maxn];//走向父亲结点的最长链
int nxt[maxn];//子树上最长链是从哪个点找到的
int n,m,k,x,y,z,N=1;

vector<pii>vec[maxn];


void dfs(int i,int f){//维护出子树中最长的两条链
    for(auto xx : vec[i]){
        if(xx.F==f)continue;
        dfs(xx.F,i);
        int tem = sub[xx.F][0] + xx.S;
        if(tem>sub[i][0]){
            sub[i][1]=sub[i][0];
            sub[i][0]=tem;
            nxt[i]=xx.F;
        }else if(tem>sub[i][1])sub[i][1]=tem;
    }
}

void dfs2(int i,int f){
    for(auto xx : vec[i]){
        if(xx.F == f)continue;
        if(nxt[i]==xx.F){//sub[i][0]是从xx.F形成的链，不能再用
            fa[xx.F]=max(fa[i],sub[i][1])+xx.S;
        }else fa[xx.F]=max(fa[i],sub[i][0])+xx.S;
        dfs2(xx.F,i);
    }
}

void work(){
    cin>>n;
    for(int i=1;i<n;i++){
        cin>>x>>y>>z;
        vec[x].push_back({y,z});
        vec[y].push_back({x,z});
    }
    dfs(1,0);
    dfs2(1,0);
    int tem = inf;
    for(int i=1;i<=n;i++)tem=min(tem,max(fa[i],sub[i][0]));
    for(int i=1;i<=n;i++)if(max(fa[i],sub[i][0])==tem)cout<<i<<endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```



### 最小生成树？？？

无向连通图G的子图如果是包含了所有顶点的树，那么就称这个子图为G的生成树。各边权值之和称作该树的权。权最小的生成树称为最小生成树。

最小生成树的最大边权是所有生成树的最大边权中最小的

#### 克鲁斯科尔算法（Kruskal)

稀疏图使用 O(ElgE)

从边入手，由权值从小到大排序，判断新加入的边是否形成环。

```c++
struct node{
    int u,w,v;
};
bool cmp(node x1,node x2){
    return x1.v<x2.v;
}

/*此处包含并查集的find,init,uni函数*/  //路径压缩

int fa[n];node list[m];
signed main(){
    init(n);sort(list,list+n,cmp);
    int x=1;int v=0;node now;
    while(x<n){
        now=list[v++];
        if(find(now.u)==find(now.w))continue;
        uni(now.u,now.w);x++;
    }
}
```

#### 普里姆算法(prim)

稠密图使用  O(v^2^) ，使用堆之后O(ElgE)

任意取一个点，从已取连接块和剩余连接块中找权值最小的边

```c++
int arr[maxn][maxn];  //邻接矩阵
int dist[maxn]={0};  //记录未选的点和已取的连接块的最短距离
bool list[maxn];//记录点是否已经被选中
signed main(){
    int n;cin>>n;
    memset(arr,0x3f,sizeof(arr));
    //arr数据的输入省略,cin>>z;arr[x][y]=min(z,arr[x][y]);
    memset(dist,0x7f,sizeof(dist));
    memset(list,0,sizeof(list));
    list[1]=1;
    dist[1]=0;
    int sum=0;
    for(int a=2;a<=n;a++)dist[a]=min(dist[a],arr[1][a]);
    for(int a=2;a<=n;a++){
        int t=-1,tem=maxn;
        for(int b=2;b<=n;b++){
            if(!list[b]&&dist[b]<tem){
                t=b;tem=dist[b];
            }
        }
        if(t==-1){cout<<"无法生成树";return 0;} //找不到跟已取连接块连接的点
        sum+=tem;list[t]=1;
        for(int c=2;c<=n;c++)dist[c]=min(dist[c],arr[t][c]);
    }
}
```

#### Boruvka 算法？？？

#### 最小生成树是否唯一

[[The Unique MST - POJ 1679 - Virtual Judge](https://vjudge.net/problem/POJ-1679)]

在使用克鲁斯卡尔时对于某一权值计算能用的边数和实际用的边数，如果两者始终相等就唯一

两者不相等说明该权值的边在连通块之间形成了环，对于环来说去除边的方法不唯一

```c++
#include<iostream>
#include<algorithm>
using namespace std;
#define F first
#define S second
#define int long long
#define pii pair<int,int>
#define endl '\n'
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn = 2e6+10;
pair<pii,int> arr[maxn];
int fa[maxn];
int n,m,k,x,y,z,N=1;

bool cmp(pair<pii,int>a,pair<pii,int>b){
    return a.S<b.S;
}

int find(int i){
    if(fa[i]!=i)fa[i]=find(fa[i]);
    return fa[i];
}

void uni(int i,int j){
    fa[find(i)]=find(j);
}

void work(){
    cin>>n>>m;
    for(int i=1;i<=n;i++)fa[i]=i;
    for(int i=1;i<=m;i++){
        cin>>x>>y>>z;
        arr[i]={{x,y},z};
    }
    sort(arr+1,arr+1+m);
    int ans = 0;
    int now = -inf;
    int sum1 = 0;
    int sum2 = 0;
    for(int i=1;i<=m;i++){
        if(arr[i].S>now){
            if(sum2>sum1){
                cout<<"Not Unique!"<<endl;
                return ;
            }
            sum2=0;sum1=0;
            now = arr[i].S;
            int r=i;
            while(r<=m&&arr[r].S==now){
                if(find(arr[r].F.F)!=find(arr[r].F.S))sum2++;
                r++;
            }
        }
        if(find(arr[i].F.F)!=find(arr[i].F.S)){
            uni(arr[i].F.F,arr[i].F.S);
            sum1++;ans+=arr[i].S;
        }
    }
    if(sum2>sum1){
        cout<<"Not Unique!"<<endl;
        return ;
    }
    cout<<ans<<endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    cin>>N;
    while (N--) {
        work();
    }
}
```

### 次小生成树？？？（k小生成树？）

### 笛卡尔树

笛卡树上每个节点都是一个二元组(k,w)；

（1）k满足二叉搜索树的性质，即从根节点中序遍历，每个节点的k是连续且递增的

（2）w满足小\大根堆的性质

#### 对一个序列构建笛卡尔树

$O(n)$

```c++
//小根堆
int ls[maxn],rs[maxn],arr[maxn];;
stack<int> sta;    
for (int i = 1; i <= n; i++) {
    int last = 0;
    while (!sta.empty() && arr[sta.top()] > arr[i]) {
        last = sta.top();
        sta.pop();
    }
    if (!sta.empty())rs[sta.top()] = i;
    ls[i] = last;
    sta.push(i);
}
```

### 斯坦纳树？？？



### 最短路

松弛操作：对于边(u,v)来说，松弛操作对应式子：dis(v)=min(dis(v),dis(u)+w(u,v))，用(u,v)这条边作为从u到v的桥梁

如果需要记录最短路径具体路径，需要在更新dis时修改pre数组

#### 单源最短路

##### Bellman–Ford 算法

O(nm)，可以处理负权，可以用来判断负环是否存在

[[P3385 【模板】负环 - 洛谷](https://www.luogu.com.cn/problem/P3385)]

每进行一轮，就对所有边进行一次松弛操作，可以证明在最短路存在的条件下，每一轮至少会给一点取得最短路。

证明（可能正确）：假如i->j的最短路为i->...->k->j，那么如果上一轮k取得最短路而j没有取得最短路（有可能同一轮取得，看边松弛的顺序），这一轮j一定会取得最短路，那么向前递归，第0轮i本身是最短路，后面每轮都有边取得，当一轮没有更新最短路时说明所有最短路都已经找到了，如果第n轮还有最短路的更新说明有负环。

```c++
int n,m;
int u[maxn],v[maxn],w[maxn],dis[maxn];//四个数组分别为起点，终点，权值，距离
void work(){
    cin>>n>>m;int cnt=0;
    for(int a=2;a<=n;a++)dis[a]=inf;
    dis[1]=0;//这是判断1作为源点能否到达负环，如果判断整个图有没有负环可以加一个超级源点0，向每个点连一条权值为0的边,如果有超级源点，松弛循环次数再加一
    for(int a=1;a<=m;a++){
        cin>>x>>y>>z;
        ++cnt;
        u[cnt]=x,v[cnt]=y,w[cnt]=z;
        ++cnt;
        u[cnt]=y,v[cnt]=x,w[cnt]=z;
    }
    for(int a=1;a<=n;a++){//如果有超级源点，循环次数再加一
        bool tag=0;
        for(int b=1;b<=cnt;b++){
            if(dis[u[b]]!=inf&&dis[v[b]]>dis[u[b]]+w[b]){//dis[u[b]]!=inf是用来判断是否从源点出发到达负环，可能出现源点和负环不连通的情况
                //删除dis[u[b]]!=inf之后从1出发也能判断整个图是否存在负环（应该是）
                dis[v[b]]=dis[u[b]]+w[b];tag=1;
            }
        }
        if(tag==0)break;
        if(a==n){cout<<"YES"<<endl;return ;}
    }
    cout<<"NO"<<endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    work();
}
```

##### SPFA算法

Bellman–Ford 算法优化版本，上界不变，在无负边权的时候尽量使用dijkstra算法

只有被优化过的点才有可能对后面的点进行优化

如果要判断整个图中有没有负环加一个超级源点

```c++
vector<pii>vec[maxn];
int dis[maxn];
int cnt[maxn];//由于我们需要判断是否有负环，所以我们用cnt[i]从源点到i取得最短路需要多少步，如果到n就说明有负环
bool vis[maxn];//判断一个数有没有在队列中，如果在队列就不用再放进去了，减少时间，没有这个也不影响正确性
bool spfa(int t){
    queue<int>que;
    cin>>n>>m;
    memset(dis,0x3f,(n+1)*sizeof(int));
    que.push(t);dis[t]=0;vis[t]=1;cnt[t]=0;
    for(int i=1;i<=m;i++){
        //输入边数据
    }
    while(!que.empty()){
        int tem=que.front();
        que.pop();vis[tem]=0;//tem可以再次进队列
        for(auto xx : vec[tem]){
            int v=xx.first,w=xx.second;
            if(dis[tem]+w<dis[v]){
                cnt[v]=cnt[tem]+1;
                if(cnt[v]==n)return false;//超过n次
                dis[v]=dis[tem]+w;
                if(!vis[v]){que.push(v);vis[v]=1;}//v已经在队列，没必要再进去
            }
        }
    }
    return true;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    cin>>N;
    while(N--){
        if(spfa(1))cout<<"无"<<endl;
        else cout<<"负环"<<endl;
    }
}
```

##### Dijkstra算法

[[P1875 佳佳的魔法药水 - 洛谷](https://www.luogu.com.cn/problem/P1875)]看看思路

[[Problem - D - Codeforces](https://codeforces.com/contest/2059/problem/D)]

单源最短路径，==不处理负边权==，因为当找到一个点之后我们就认为已经找到这个点的最短路径，不会再对这个点进行修改了，但是负边权会改变这一情况

O(n^2^)，加上堆优化后为O(mlog~2~m)

起点为s，dis[t]为s到t的最短路径权值，初始化dis[s]=0,其余为maxn

```c++
//朴素算法：
    bool vis[maxn]={0};
	int dis[maxn]={0};
	vector<pii>vec[maxn];
	int n,m,k;    //求k到其它点的最短路径
	cin>>n>>m>>k;
	memset(dis,0x3f,sizeof(dis));
	memset(vis,0,sizeof(vis));
	dis[k]=0;
	for(int i=1;i<=n;i++)vec[i].clear();
	for(int i=1;i<=m;i++){
		int x,y,v;cin>>x>>y>>v;
		vec[x].push_back({y,v});
		vec[y].push_back({x,v});
	}
	for(int i=1;i<=n;i++){
		int maxx=inf;
		int t=-1;
		for(int j=1;j<=n;j++){
			if(dis[j]<maxx&&vis[j]==0){
				maxx=dis[b];
				t=b;
			}
		}
        if(t==-1)break;//并不连通
		vis[t]=1;
		for(auto now : vec[t]){
			int x1=now.first,x2=now.second;
			dis[x1]=min(dis[x1],dis[t]+x2);
		}
	}
```

```c++
//堆优化：
struct cmp{
   	bool operator()(pii&a,pii&b){
       return a.second>b.second;
  	}	
};
priority_queue <pii,vector<pii>,cmp > que;
void dijkstra(int t){
    memset(dis,0x3f,sizeof(dis));
    memset(vis,0,sizeof(vis));
    while(!que.empty())que.pop();
    dis[t]=0;
    que.push({t,0});
    for(int i=1;i<=n;i++){
        while(!que.empty()&&vis[que.top().F])que.pop();//弹出已经被更新过的点
        if(que.empty())break;
        int now = que.top().F;que.pop();
        vis[now]=1;
        for(auto xx : vec[now]){
            if(vis[xx.F])continue;
            if(dis[xx.F]>dis[now]+xx.S){
                dis[xx.F]=dis[now]+xx.S;
                que.push({xx.F,dis[xx.F]});//更新que
            }
        }
    }
}
```

#### 全源最短路径

##### Floyd算法？？？

多源最短路径，不处理负环，O(n^3^)

通过在两个点中插入其它点，来迭代找到最小路径，有负环的时候不能用。

数组 `f[k][x][y]`，表示结点x到结点y的路径（只允许经过结点1到k）的最短路长度。

转移方程：`f[k][x][y] `= min(`f[k-1][x][y]`, `f[k-1][x][k]+f[k-1][k][y]`)

为什么可以省略第一维？因为对于k来说，只会用到第k行和第k列的元素来更新，而这些元素这一次循环不会被k更新，`f[k][x][k]==f[k-1][x][k]`

```c++
int arr[maxn][maxn];//邻接矩阵,自己到自己是0然后依次输入
for(int i=1;i<=n;i++){
    for(int j=1;j<=n;j++){
        arr[i][j]=inf;//没有办法直达的距离设为无穷大
    }
}
for(int i=1;i<=n;i++)arr[i][i]=0;//自己到自己是0
/*
输入数据
*/
for(int k=1;k<=n;k++){
    for(int i=1;i<=n;i++){
        for(int j=1;j<=n;j++){
            arr[i][j]=min(arr[i][j],arr[i][k]+arr[k][j]);
        }
    }
}
```

如何记录每两个点之间的最短路的路径节点，前驱矩阵？？？

###### 任意两点是否连通???

图的传递闭包？

如果没有权值的话，可以把数据改成1和0，加法运算改成或运算，来判断数据之间有没有关系。例题：[[POJ-3660 ](http://poj.org/problem?id=3660)] 用 bitset 优化？？？，复杂度为O(n^3^/ω)

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define ull unsigned long long
#define pii pair<int,int>
#define F first
#define S second
#define endl '\n'
#define pause system("pause")
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn = 2e6+10;
const int mod = 1e9+7;
int arr[105][105];
int n,m,k,x,y,z,N=1;

void work(){
    cin>>n>>m;
    for(int i=1;i<=m;i++){
      cin>>x>>y;
      arr[x][y]=1;
    }
    for(int i=1;i<=n;i++){
      arr[i][i]=1;
    }
    for(int i=1;i<=n;i++){
      for(int j=1;j<=n;j++){
        for(int k=1;k<=n;k++){
          arr[j][k]|=(arr[j][i]&arr[i][k]);
        }
      }
    }
    int ans = 0;
    for(int i=1;i<=n;i++){
      bool tag = 0;
      for(int j=1;j<=n;j++){
        if((arr[i][j]|arr[j][i])==0){tag=1;break;}
      }
      if(!tag)ans++;
    }cout<<ans;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

###### 给一个正权无向图，找一个最小权值和的环。

[内容见最小环](####Floyd)

##### Johnson全源最短路

如果用Bellman–Ford跑n次单源最短路，时间复杂度为O(n^2^m)，而Dijkstra是O(nmlgm)，前者复杂度不优秀，后者不能处理负边权，所以Johnson全源最短路用来解决带==负权边==的==稀疏图==的==全源最短路==问题（稠密图用Floyed算法），复杂度为==O(nmlgm)==。

如果将每个边增加一个相同的值来把负边变为正边的话会改变原本的最短路正确性（因为不同长度的路径增加了不同大小的值），所以我们设置一个0点，向每个边连一个权值为0的点，然后求出最短路记为h~i~,将(u,v,w)的边重新设置为(u,v,w+h~u~-h~v~),跑n次Dijkstra即可，得到的最短路长度dis(u,v)=原本的最短路长度+h~u~-h~v~

[[P5905 【模板】全源最短路（Johnson）](https://www.luogu.com.cn/problem/P5905)]:

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define pii pair<int,int>
#define F first
#define S second
#define endl '\n'
const int maxn = 3e3+10;
const int mod = 1e9;
int n,m,k,x,y,z,N=1;

vector<pii>vec[maxn];
int dis[maxn];
int h[maxn];
bool vis[maxn];
struct cmp{
    bool operator()(pii &a,pii &b){
        return a.S>b.S;
    }
};
priority_queue<pii,vector<pii>,cmp>que;

void dijkstra(int t){
    int ans=0;
    memset(dis,0x3f,sizeof(dis));
    memset(vis,0,sizeof(vis));
    while(!que.empty())que.pop();
    dis[t]=0;
    que.push({t,0});
    for(int i=1;i<=n;i++){
        while(!que.empty()&&vis[que.top().F])que.pop();
        if(que.empty())break;
        int now = que.top().F;que.pop();
        vis[now]=1;
        ans+=(dis[now]-h[t]+h[now])*now;
        for(auto xx : vec[now]){
            if(vis[xx.F])continue;
            if(dis[xx.F]>dis[now]+xx.S){
                dis[xx.F]=dis[now]+xx.S;
                que.push({xx.F,dis[xx.F]});
            }
        }
    }
    for(int i=1;i<=n;i++){
        if(vis[i]==0){
            ans+=mod*i;
        }
    }
    cout<<ans<<endl;
}

signed main(){
    cin>>n>>m;
    for(int i=1;i<=m;i++){
        cin>>x>>y>>z;
        vec[x].push_back({y,z});
    }
    for(int i=1;i<=n;i++)vec[0].push_back({i,0});
    memset(h,0x3f,sizeof(h));
    h[0]=0;
    for(int i=1;i<=n+1;i++){
        bool tag = 0;
        for(int j=0;j<=n;j++){
            for(auto xx : vec[j]){
                if(h[xx.F]>h[j]+xx.S){
                    tag=1;
                    h[xx.F]=h[j]+xx.S;
                }
            }
        }
        if(tag==0)break;
        if(i==n+1){
            cout<<-1;return 0;
        }
    }
    for(int i=1;i<=n;i++){
        for(int j=0;j<vec[i].size();j++){//这里不能用auto xx，修改xx没用
           vec[i][j].S+=h[i]-h[vec[i][j].F];
        }
    }
    for(int i=1;i<=n;i++)dijkstra(i);
}
```

#### 分层图最短路

[ [JLOI2011\] 飞行路线 - 洛谷](https://www.luogu.com.cn/problem/P4568)]

### k短路？？？

### 差分约束 ???

差分约束系统是一种特殊的n元一次不等式组,包含n个一元未知数和m个约束条件，每个约束条件都是两个变量做差构成的，x~i~-x~j~<=c~k~,我们判断是否能找出符合约束的解

我们将x~i~看作dis[i],那么约束条件变成dis[i]<=dis[j]+c~k~,类似于松弛操作,所以我们建一个有向边<j,i,c~k~>,如果图中有负环(Bellman-ford或者SPFA判负环)就说明约束矛盾

给出x~i~==x~j~+t说明x~i~-x~j~<=t并且x~j~-x~i~<=-t,建两条边

板子题:[[P1993 小 K 的农场 - 洛谷](https://www.luogu.com.cn/problem/P1993)]

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define pii pair<int,int>
#define F first
#define S second
#define endl '\n'
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn = 2e6+10;
int n,m,k,x,y,z,N=1;

vector<pii>vec[maxn];
int dis[maxn];

void work(){
    cin>>n>>m;
    for(int i=1;i<=m;i++){
      cin>>x>>y>>z;
      if(x==1){
        cin>>x;
        vec[y].push_back({z,-x});
      }else if(x==2){
        cin>>x;
        vec[z].push_back({y,x});
      }else {
        vec[y].push_back({z,0});
        vec[z].push_back({y,0});
      }
    }
    memset(dis,0x3f,sizeof(dis));
    for(int i=1;i<=n;i++){
      bool tag = 0;
      for(int j=1;j<=n;j++){
        for(auto xx : vec[j]){
          if(dis[xx.F]>dis[j]+xx.S){
            dis[xx.F]=dis[j]+xx.S;tag=1;
          }
        }
      }
      if(!tag)break;
      if(i==n){
        cout<<"No";return ;
      }
    }
    cout<<"Yes";
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

#### x~i~/x~j~<=c~k~ ???

[[P4926 1007 倍杀测量者 - 洛谷](https://www.luogu.com.cn/problem/P4926)]

### 同余最短路 ???

### 二分图？？？

节点由两个集合组成，且两个集合内部没有边的图,二分图不存在奇数长度的环

#### 判断是否为二分图

##### 染色法

从任意未染色的点开始染黑白两色,相邻的点染成不同的颜色,如果走到一个点既黑又白就说明该图不为二分图

##### bfs/dfs (具体实现???)

判断是否存在奇数长度的环,如果存在就不是二分图

### 连通性相关

#### 强连通分量（SCC）和缩点

在有向图中，一个图中的点两两可达称为强连通，极大的强连通子图称为强连通分量

##### Tarjan算法

O(n+m)

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn = 2e6+10;
int n,m,N=1;

int tme, cnt; //时间戳和强连通分量个数

stack <int> sta; bool insta[maxn]; //栈相关
vector <int> vec[maxn]; int dfn[maxn], low[maxn]; //dfs相关
vector <int> scc[maxn]; int bel[maxn]; //scc相关

void tarjan (int now) {
    // 入栈并维护
    dfn[now] = low[now] = ++tme;
    sta.push(now); insta[now] = 1;
    //遍历子节点
    for (auto to : vec[now]) {
        if (!dfn[to]) { //如果尚未访问就继续访问，并维护low[]
            tarjan (to);
            low[now] = min(low[now], low[to]);
        } else if (insta[to]) low[now] = min(low[now], dfn[to]); //如果尚在栈中就用来更新low[]
    }
    if (dfn[now] == low[now] ) {//记录scc
        int last; cnt++;
        do{
            last = sta.top(); sta.pop();
            insta[last] = 0;
            bel[last] = cnt;
            scc[cnt].push_back(last);
        }while (last != now);
    }
}

void work(){
    cin >> n >> m;
    int x, y;
    for (int i = 1; i <= m; i++) {
        cin >> x >> y;
        vec[x].push_back(y);
    }
    for (int i = 1; i <= n; i++) {
        if (!dfn[i]) {
            tarjan(i);
        }
    }
    cout << cnt << endl;
    for (int i = 1; i <= cnt; i++) {
        for (auto x : scc[i]) {
            cout << x << ' ';
        } cout << endl;
    }
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

##### Kosaraju 算法

##### Garbow 算法

##### tarjan缩点

在有向图上求一条路径，使路径经过的点权值之和最大

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn = 2e6+10;
int n,m,N=1;

int tme, cnt;

stack <int> sta; bool insta[maxn];
int dfn[maxn], low[maxn];
int bel[maxn], dp[maxn];
vector <int> vec[maxn]; int v[maxn]; //原图
vector <int> edge[maxn]; int val[maxn]; //新图
queue <int> que;

void tarjan (int now) {
    dfn[now] = low[now] = ++tme;
    sta.push(now); insta[now] = 1;
    for (auto to : vec[now]) {
        if (!dfn[to]) {
            tarjan (to);
            low[now] = min(low[now], low[to]);
        } else if (insta[to]) low[now] = min(low[now], dfn[to]);
    }
    if (dfn[now] == low[now] ) {
        int last; cnt++;
        do{
            last = sta.top(); sta.pop();
            insta[last] = 0;
            bel[last] = cnt;
            val[cnt] += v[last];
        }while (last != now);
    }
}

void work(){
    cin >> n >> m;
    int x, y;
    for (int i = 1; i <= n; i++) cin >> v[i];
    for (int i = 1; i <= m; i++) {
        cin >> x >> y;
        vec[x].push_back(y);
    }
    for (int i = 1; i <= n; i++) {
        if (!dfn[i]) {
            tarjan(i);
        }
    }
    for (int i = 1; i <= n; i++) { //将原本的图转化为缩点后的图
        for (auto to : vec[i]) {
            if (bel[i] == bel[to]) continue;
            edge[bel[i]].push_back(bel[to]);
        }
    }
    int ans = 0;
    for (int i = cnt; i; i--) { //缩点后的图满足反拓扑序
        dp[i] += val[i];
        ans = max(ans, dp[i]);
        for (auto to : edge[i]) {
            dp[to] = max(dp[i], dp[to]);
        }
    }
    cout << ans << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

#### 割点和割边

割点（割顶）和割边（桥）是无向图中的概念，若删除一个点(边)会更改原图的连通分量数，则称该点(边)为割点(割边)

在无向图中，将属于生成树上的边称作树边，不属于的称为非树边，无向图上的树边指向的一定是其祖先

定义$low[i]$为$i$这个点不经过父亲能到达的最小的$dfn$序，要注意避免先经过儿子，然后再经过父亲的情况

判断方法：对于一个点u，如果至少存在一个儿子v满足$low[v] >= dfn[u]$，则说明如果删除点u，则v无法连接u的祖先节点，所以u为割点.

但上述判断方法不支持对于根节点的判断，对于根节点，我们考虑如果删除之后点仍联通，那么$dfs$过程中根节点只会进行一次$dfs$的转移，此时根节点非割点，如果进行多次$dfs$转移则说明根节点为割点。

```c++
//割点（tarjan算法) O(n+m):
/*
样例1:
5 6
1 2
2 3
3 1
2 4
4 5
5 2
使用错误代码时，5在找到2的时候如果采取low[2]来更新，会顺着3找到1，导致漏了2这个割点

样例2:
4 5
1 2
1 3
2 3
2 4
3 4
错误代码会将联通的子节点当作多个节点进行判断，会导致并没有跟最小的节点进行比较
*/
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn = 2e6+10;
int n,m,N=1;

int tme;
int dfn[maxn], low[maxn];
vector <int> vec[maxn], ans;

void tarjan (int now) {
    dfn[now] = low[now] = ++tme;
    int sum = 0 ; bool tag = 0;
    for (auto to : vec[now]) {
        if (!dfn[to]) {
            tarjan (to);
            low[now] = min(low[now], low[to]);
            if (dfn[now] == 1) {  //根节点用sum记录dfs次数
                sum++;
                if (sum >= 2) tag = 1;
            } else if (low[to] >= dfn[now]) { //遇到一个树边时进行判断，如果对非树边也进行判断会错，见样例2
                tag = 1;
            }
        } else low[now] = min(low[now], dfn[to]);//对于非树边只能使用dfn[]来维护，不能使用low[]，见样例1
    }
    if (tag) ans.push_back(now);
}

void work(){
    cin >> n >> m;
    int x, y;
    for (int i = 1; i <= m; i++) {
        cin >> x >> y;
        vec[x].push_back(y);
        vec[y].push_back(x);
    }
    for (int i = 1; i <= n; i++) {
        if (!dfn[i]) {
            tme = 0;  //时间戳归0，用来判断是否为根节点
            tarjan(i);
        }
    }
    sort (ans.begin(), ans.end());
    cout << ans.size() << endl;
    for (auto x : ans) cout << x << ' ';
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}

/*
割边：
	dfs时传一个父节点参数，对于父节点跳过，每次对于未操作过的子节点判断时相当于对这条边进行判断，如果low[x]>dfn[i](割点是>=), 则这条边是割边，同时不需要对根节点进行特判
	使用链式前向星用来记录割边，具体代码看边双连通分量
*/
```

#### 双连通分量（BCC）

在一张连通的无向图中，对于两个点u和v，如果无论删去哪条边（只能删去一条）都不能使它们不连通，我们就说u和v**边双连通**，如果无论删去哪个点（只能删去一个，且不能删u和v自己）都不能使它们不连通，我们就说u和v**点双连通**。

边双连通具有传递性，而点双连通不具有传递性。

极大边双连通子图称为边双连通分量，极大点双连通子图称为点双连通分量。

##### 边双

也就是没有割边

###### tarjan算法

先找割边，再$dfs$分割出边双

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn = 2e6+10;
int n,m,N=1;

struct edge {
    int to,nxt;
    bool vis;
}e[maxn*2];
int head[maxn], cnt;

int dfn[maxn], low[maxn], tme;

vector <int> ebcc[maxn]; int scc_cnt; bool bel[maxn];

void add (int x,int y) {
    e[++cnt].to = y; e[cnt].nxt = head[x]; head[x] = cnt;
}

void tarjan (int now, int f) {
    dfn[now] = low[now] = ++tme;
    int sum = 0 ; bool tag = 0;
    for (int x = head[now]; x != -1; x = e[x].nxt) {
        if (e[x].to == f) continue;
        int to = e[x].to;
        if (!dfn[to]) {
            tarjan (to, now);
            low[now] = min(low[now], low[to]);
            if (low[to] > dfn[now]) {
                e[x].vis = 1;
                e[x^1].vis = 1;
            }
        } else low[now] = min(low[now], dfn[to]);
    }
}

void dfs(int i){
    bel[i] = 1;
    ebcc[scc_cnt].push_back(i);
    for (int now = head[i]; now != -1; now = e[now].nxt){
        if (e[now].vis) continue;
        if (!bel[e[now].to]) dfs(e[now].to);
    }
}

void work(){
    cin >> n >> m;
    int x, y;
    memset(head, -1, sizeof(head));
    cnt = 1; //从2开始
    for (int i = 1; i <= m; i++) {
        cin >> x >> y;
        add(x, y); add(y, x);
    }
    for (int i = 1; i <= n; i++) { //寻找割边
        if (!dfn[i]) {
            tarjan(i, 0);
        }
    }
    for (int i = 1; i <= n; i++) { //建立边双
        if (!bel[i]) {
            ++scc_cnt;
            dfs(i);
        }
    }
    cout << scc_cnt << endl;
    for (int i = 1; i <= scc_cnt; i++) {
        cout << ebcc[i].size() << ' ';
        for (int j = 0; j < ebcc[i].size(); j++) {
            cout << ebcc[i][j] << ' ';
        }
        cout << endl;
    }
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

用栈，类似于强连通分量    （为什么？？？）

```c++
#include<bits/stdc++.h>
//#pragma GCC optimize("O0")
using namespace std;
#define int long long
#define pii pair<int,int>
#define F first
#define S second
const int maxn=5e5+10;
int arr[maxn];
int dfn[maxn];
int low[maxn];
int insta[maxn];
vector<int>dot[maxn];
int tot=0;
struct node{
    int w,net;
}e[maxn*8];			//对边进行标号，这里用链式前向星易于对割边进行记录
int head[maxn];  //链式前向星
int cnt=0;int t=0;
bool vis[maxn*8];	//记录某个边是否为割边
stack<int>sta;
void add(int u,int v){
    e[cnt].w=v;
    e[cnt].net=head[u];
    head[u]=cnt++;
}
void tarjan(int i,int f){		//这里的f是边的下标，当边从0开始记录时，由于这里是无向图，所以正反双向边是一起记录的，边f的反向边就是边（f^1),这样就避免了对重边的特殊判断
    dfn[i]=low[i]=++t;sta.push(i);insta[i]=1;
    for(int now=head[i];now!=-1;now=e[now].net){
        int x=e[now].w;
        if(dfn[x]==0){
            tarjan(x,now);
            low[i]=min(low[x],low[i]);
        }
        else if(now!=(f^1)){
            low[i]=min(dfn[x],low[i]);
        }
    }
    if(dfn[i]==low[i]){
        int y;tot++;
        do{
            y=sta.top();insta[y]=0;sta.pop();
            dot[tot].push_back(y);
        } while(y!=i);
    }
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    int n,m;int x,y;cin>>n>>m;
    for(int a=1;a<=m;a++){
        e[a].net=-1;e[a].w=0;
    }
    memset(head,-1,sizeof(head));
    for(int a=1;a<=m;a++){
        cin>>x>>y;
        if(x==y)continue;
        add(x,y);
        add(y,x);
    }
    for(int a=1;a<=n;a++){
        if(!dfn[a])tarjan(a,-1);
    }
    cout<<tot<<endl;
    for(int a=1;a<=tot;a++){
        cout<<dot[a].size()<<' ';
        for(int tem:dot[a]){
            cout<<tem<<' ';
        }
        cout<<endl;
    }
}
```

###### 差分算法？？？？

##### 点双

点双定义：图中任意两不同点之间都有至少两条点不重复的路径

该定义近似于等价于图中没有割点，但在只有两个点，一条边的图上不适用

两个点双最多有一个公共点（当有多个公共点时两者就可以合并为一个新的点双），并且这个点是割点

一个割点可能出现在多个点双里，而边不会

一个点双中$dfn$序最小的点要么是割点，要么是树根，所以策略是$dfs$过程中将点放入栈中，遇到割点和树根就将子树作为新的点双

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn = 2e6+10;
int n,m,N=1;

int tme;
stack<int> sta;
int dfn[maxn], low[maxn];
vector <int> vec[maxn];

vector <int> vdcc[maxn]; int bcc_cnt;

void build_bcc(int now) {
    ++bcc_cnt;
    int last = 0;
    do {
        vdcc[bcc_cnt].push_back(sta.top());
        last = sta.top();
        sta.pop();
    } while (last != now);
}

void tarjan (int now) {
    sta.push(now);
    dfn[now] = low[now] = ++tme;
    for (auto to : vec[now]) {
        if (!dfn[to]) {
            tarjan (to);
            low[now] = min(low[now], low[to]);
            if (low[to] >= dfn[now]) { //割点或者树根,说明有点双
                build_bcc(to); //将子树加入边双，然后再将自身加入（但不弹出）
                vdcc[bcc_cnt].push_back(now);
            }
        } else low[now] = min(low[now], dfn[to]);
    }
    if (tme == 1) build_bcc(now); //特判孤立点
}

void work(){
    cin >> n >> m;
    int x, y;
    for (int i = 1; i <= m; i++) {
        cin >> x >> y;
        vec[x].push_back(y);
        vec[y].push_back(x);
    }
    for (int i = 1; i <= n; i++) {
        if (!dfn[i]) {
            tme = 0;  //时间戳归0，用来判断是否为根节点
            while (!sta.empty()) sta.pop();
            tarjan(i);
        }
    }
    cout << bcc_cnt << endl;
    for (int i = 1; i <= bcc_cnt; i++) {
        cout << vdcc[i].size() << ' ';
        for (auto x : vdcc[i]) cout << x << ' ';
        cout << endl;
    }
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

### 2-sat问题

给出n个集合，每个集合有两个元素（0，1），给定若干个条件<a,b>，问能否从每个集合选出一个元素且元素之间不违反条件。

统一将所有条件转化成蕴含关系.

1. **a → b**转化为`a → b`，`¬b → ¬a` (若a则b)
2. **a ∨ b**转化为`¬a → b`，`¬b → a`(a或者b)
3. **a ≠ b**转化为`a → ¬b`，`b → ¬a`，`¬a → b`，`¬b → a`(a，b不等)
4. **a = b**转化为`a → b`，`b → a`，`¬a → ¬b`，`¬b → ¬a`(a，b相等)
5. **¬(a ∧ b)**转化为`a → ¬b`，`b → ¬a`(a，b不同时为真)

然后将每个集合用两个点表示，蕴含关系表示为一条有向边，构建强连通分量，若同集合的两个点在同一强连通分量中则说明不存在可行解，否则的话对于同一集合的两点选择拓扑序大的点（即SCC编号小的）

```c++
//输入为类型2
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn = 2e6+10;
int n,m,N=1;

vector<int> vec[maxn], scc[maxn];
int bel[maxn], dfn[maxn], low[maxn];
stack <int> sta; bool insta[maxn];
int tme, cnt;

void add (int x, int y) {
    vec[x].push_back(y);
}

void tarjan(int now) {
    dfn[now] = low[now] = ++tme;
    sta.push(now); insta[now] = 1;
    for (auto x : vec[now]){
        if (!dfn[x]) {
            tarjan(x);
            low[now] = min(low[now], low[x]);
        } else if(insta[x]) low[now] = min(low[now], dfn[x]);
    }

    if (low[now] == dfn[now]){
        int last = 0; ++cnt;
        do {
            scc[cnt].push_back(sta.top());
            last = sta.top(); sta.pop();
            bel[last] = cnt; insta[last] = 0;
        } while (now != last);
    }
}

void work(){
    cin >> n >> m;
    int x, a, y, b;
    for (int i = 1; i <= m; i++) {
        cin >> x >> a >> y >> b;
        add (x + n * (a ^ 1), y + n * b);
        add (y + n * (b ^ 1), x + n * a);
    }
    for (int i = 1; i <= 2 * n; i++) {
        if (!dfn[i]) {
            tarjan(i);
        }
    }
    for (int i = 1; i <= n; i++) {
        if (bel[i] == bel[i + n]) {
            cout <<"IMPOSSIBLE"; return ;
        }
    }
    cout << "POSSIBLE" << endl;
    for (int i = 1; i <= n; i++) {
        if (bel[i + n] < bel[i]) cout << "1 ";
        else cout << "0 ";
    }
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

### 圆方树

圆方树是将图转化为树的一种方法



### 网络流

网络是一个特殊的有向图，每条边都有容量作为权值，还有源点s和汇点t两个特殊的点(s!=t)，

c(x,y)称为边的容量，f(x,y)称为边的流量，f则为网络的流函数

流函数的三个条件：1.容量限制  s(x,y)<=c(x,y)  

2.斜对称  f(x,y)=-f(y,x),即正向边流量=反向边流量

3.流量守恒  正向流量和=反向流量和

残余网络（残量网络）：点集不变，边集由两部分组成，一个是剩边，如果f(x,y)<c(x,y)，则有边f’(x,y)=c(x,y)-f(x,y)属于残余网络。另一个是反向边，如果f(x,y)>0,则有f’(y,x)=f(x,y),大小相等，方向相反。

增广路：残余网络上一条从s到t的简单路径，路径的残余流量是该路径上最小的容量。

#### 最大流

val(f):从s到t的最大流量

阻塞流：当按照确定的流通方式时，无法再找到新的可行流，则已有的流被称为阻塞流。

最大流一定是阻塞流，阻塞流不一定是最大流

##### 增广路算法

###### Ford-Fulkerson算法（FF算法）

通过dfs寻找一条从s到t的增广路，找到沿着路经则增加流，找不到就结束

如果最大流为f，则复杂度最大值为O(|f|*|E|)，E是边数

```c++
#include<bits/stdc++.h>
//#pragma GCC optimize("O0")
using namespace std;
#define F first
#define S second
#define int long long
#define pii pair<int,int>
#define endl '\n'
#define pause system("pause")
#define inf 0x3f3f3f3f3f3f3f3f
int n,m,k,x,y,z,N,cnt=0;

struct edge{int to,cap,rev;};//rev记录该边的反向边的下标
vector<edge>vec[maxv];
bool vis[maxv]={0};

void add(int from,int to,int cap){
    x=vec[to].size(),y=vec[from].size();
    vec[from].push_back((edge){to,cap,x});
    vec[to].push_back((edge){from,0,y});
}

int dfs(int v,int t,int f){
    if(v==t)return f;
    vis[v]=1;
    for(int i=0;i<vec[v].size();i++){
        edge &e = vec[v][i];
        if(!vis[e.to]&&e.cap>0){
            int tem=dfs(e.to,t,min(e.cap,f));
            if(tem>0){  //增广该路径
                e.cap-=tem;		//正向边减少
                vec[e.to][e.rev].cap+=tem; //反向边增加
                return tem;
            }
        }
    }
    return 0;
}

int max_flow(int s,int t){
    int flow=0;
    while(1){
        memset(vis,0,sizeof(vis));
        int f=dfs(s,t,inf);
        if(!f)return flow;
        flow+=f;
    }
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    int s,t;cin>>n>>m>>s>>t;
    for(int a=1;a<=m;a++){
        cin>>x>>y>>z;
        add(x,y,z);
    }
    cout<<max_flow(s,t);
}
```

###### Edmonds−KarpEdmonds−Karp增广路算法（EK算法）

与FF算法类似，只是把dfs换成bfs,可以证明bfs次数不会超过|E||V|次，所以复杂度为O(|E|^2^|V|)

```c++
#include<bits/stdc++.h>
//#pragma GCC optimize("O0")
using namespace std;
#define F first
#define S second
#define int long long
#define pii pair<int,int>
#define endl '\n'
#define pause system("pause")
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn=5005;
const int maxv=205;
int n,m,k,x,y,z,N;

int s,t,cnt=-1;
queue<int>que;
struct edge{int to,cap,nxt;};
vector<edge>vec;
int head[maxv]={0};//链式前向星
int mf[maxv]={0};//维护到达某点的最大流量，用来记录找到的路径的流量大小
int pre[maxv]={0};//用来回溯，更新边的cap的变化

void add(int from,int to,int cap){
    vec.push_back((edge){to,cap,head[from]});
    head[from]=++cnt;
    vec.push_back((edge){from,0,head[to]});
    head[to]=++cnt;
}//从0开始存,vec[i]和vec[i^1]是反边

bool bfs(){
    memset(mf,0,sizeof(mf));
    mf[s]=inf;//源点无限大
    while(!que.empty())que.pop();
    que.push(s);
    while(!que.empty()){
        int tem=que.front();que.pop();
        for(int i=head[tem];i!=-1;i=vec[i].nxt){
            int to=vec[i].to;
            if(mf[to]==0&&vec[i].cap){
                mf[to]=min(vec[i].cap,mf[tem]);//流量由两部分取最小值
                pre[to]=i;//存前驱边
                que.push(to);
                if(to==t){return true;}
            }
        }
    }
    return false;
}

int max_flow(){
    int flow=0;
    while(bfs()){
        int now=t;
        while(now!=s){	//从t点开始根据pre数组往前更新残余网络
            int i=pre[now];
            vec[i].cap-=mf[t];
            vec[i^1].cap+=mf[t];
            now=vec[i^1].to;
        }
        flow+=mf[t];//累加流量
    }
    return flow;
}

signed main(){
    cin>>n>>m>>s>>t;
    for(int a=0;a<=n;a++)head[a]=-1;
    for(int a=1;a<=m;a++){
        cin>>x>>y>>z;
        add(x,y,z);
    }
    cout<<max_flow();
}
```

###### Dinic算法

上界是O(|V|^2^|E|)，一般达不到

当每条边都是单位容量（每条边都是1）时，复杂度为O(E*min(E^½^,V^⅔^))。

分层图：原图的一个子图，每个点按照到达s节点的最近距离来分层，只保留第n层到第n+1层的边。

```c++
#include<bits/stdc++.h>
//#pragma GCC optimize("O0")
using namespace std;
#define F first
#define S second
#define int long long
#define pii pair<int,int>
#define endl '\n'
#define pause system("pause")
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn=4e5+10;
const int maxv=205;
const int mod=1e9+7;

struct edge{int to,cap,nxt;};
vector<edge>vec;
int cnt=-1;
int head[maxv];

struct node{
    int s,t,n;
    int lev[maxv],cur[maxv],que[maxv],f;//que是手写队列，减少空间使用

    void init(int x,int y,int z){
        cnt=-1;
        s=x,t=y,n=z;
        for(int a=0;a<=z;a++)head[a]=-1;
    }

    void ADD(int x,int y,int z){
        vec.push_back((edge){y,z,head[x]});
        head[x]=++cnt;
    }

    void add(int x,int y,int z){
        ADD(x,y,z);
        ADD(y,x,0);
    }

    bool bfs(){ //建立分层网络
        for(int a=0;a<=n;a++)lev[a]=0;//未分层前，每个点的深度都为0
        lev[s]=1;
        f=0;
        que[f++]=s;//源点入队
        for(int a=0;a<f;a++){
            int tem=que[a];
            for(int i=head[tem];i!=-1;i=vec[i].nxt){
                if(lev[vec[i].to]==0&&vec[i].cap){
                    int to=vec[i].to;
                    lev[to]=lev[tem]+1;
                    que[f++]=to;
                    if(to==t)return true;
                }
            }
        }
        return false;
    }

    int dfs(int u,int mf){
        if(u==t)return mf;
        int re=0,cap;
        for(int &i=cur[u];i!=-1;i=vec[i].nxt){
            int to=vec[i].to;
            if(vec[i].cap&&lev[to]==lev[u]+1){
                cap=dfs(to,min(mf-re,vec[i].cap));
                vec[i].cap-=cap;
                vec[i^1].cap+=cap;
                re+=cap;
                if(re==mf)return re;
            }
        }
        return re;
    }

    int solve(){
        int ans=0;
        while(bfs()){
            for(int i=0;i<=n;i++)cur[i]=head[i];
            ans+=dfs(s,inf);
        }
        return ans;
    }
    
    bool vis[405]={0};
    vector<int>S;
    void dfs(int now){
        for(int i=head[now];i!=-1;i=vec[i].nxt){
            if(!vis[vec[i].to]&&vec[i].cap){
                S.push_back(i);
                vis[vec[i].to]=1;dfs(vec[i].to);
            }
        }
    }
    void min_cut(){
        vis[s]=1;dfs(s);S.push_back(s);
        for(auto xx : S)cout<<xx<<' ';//找到了最小割之后S集合中的点
        //通过vis来标记能否到达，一边能到，一边不能到的就是割边
        //直接通过cap==0来找割边是不对的，例如1->2.cap=5,2->3.cap=5;从1到3作最大流会有两条cap==0的边，但是割边只需要一条就可以了
    }

}FLOW;

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    int n,m,s,t;cin>>n>>m>>s>>t;
    FLOW.init(s,t,n);
    int x,y,z;
    for(int a=1;a<=m;a++){
        cin>>x>>y>>z;
        FLOW.add(x,y,z);
    }
    cout<<FLOW.solve();
}
```

###### MPM算法

###### ISAP算法

##### 预流推进算法

#### 最小割

割：存在X是点集V的一个子集，Y是X的补集，其中s∈X，t∈Y，则称（X，Y）是网络G的割。cap(X,Y)为从X中的点到Y中的点的边的容量和

1.val(f)=f(X,Y)-f(Y,X)<=cap(X,Y)

2.如果val(f)=cap(X,Y)，则f是最大流，(X,Y)是最小割。

3.如果任意边e=(u,v)，其中u∈X，v∈Y，有f(u,v)=c(u,v),f(v,u)=0，则f是最大流，(X,Y)是最小割。

最大流中一定存在最小割，可以通过广搜来确定X的范围，能搜到的点属于X，其余属于Y

[[G - Builder Takahashi](https://atcoder.jp/contests/abc239/tasks/abc239_g)] 寻找具体割边

##### 关于割边

###### 最小割的可行边和必须边

###### 最小割边数（原理？？？）

方法一：当进行过一次dinic之后，将所有反向边cap置零，正向边中满流（即cap==0）的cap置为1，未满流置为inf，再求一遍最大流就是最小割边数。

方法二：直接在输入cap的时候令cap=(m+1)*cap+1（其中m为总边数，割边数不会超过m）。最后得到的最大流除以m+1就是真正的最大流，对m+1取模就是最小割边数。

##### 模型一：两者取其小

##### 模型二：最大权值闭合图

即给定一张有向图，每个点都有一个权值（可以为正或负或0），你需要选择一个权值和最大的子图，使得子图中每个点的后继都在子图中。

###### 解决方法：

源点向所有正权点连结一条容量为权值的边

保留原图中所有的边，容量为正无穷

所有负权点向汇点连结一条容量为权值绝对值的边

答案为正权值之和-最大流。

选中的点是可以从源点通过bfs到达的点

###### 例题

[[P2762 太空飞行计划问题 ](https://www.luogu.com.cn/problem/P2762)]:

### 图的匹配？？？

#### 二分图匹配

### 最小环

图的最小环也称围长，n(n>=3)个节点构成的边权和最小的环。

对于一条边(u,v,w),删除该边之后求v到u的最短路，那么dis(v,u)+w为包含该边的最小环的值，枚举所有边进行上述操作，取最小值就是图上最小环的值。

#### Dijkstra

按照上述思想，复杂度为O(m*(n+m)logm)

#### floyd

因为dis[k\][u\][v\]代表从u到v，路径中的点的序号都小于等于k的最短路距离，对于一个环，我们假设环上的点的序号的最大值为k，k相邻的两个点为u,v，那么该环的值为dis[k-1]\[u]\[v]+val[k]\[u]+val[v]\[k]

复杂度为O(n^3^)

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
int n,m,k,x,y,z,N=1;
int dis[105][105];
int val[105][105];

void work(){
    cin>>n>>m;
    memset(dis,0x0f,sizeof(dis));
    for(int i=1;i<=n;i++)dis[i][i]=0;
    for(int i=1;i<=m;i++){
        cin>>x>>y>>z;
        dis[x][y]=min(z,dis[x][y]);
        dis[y][x]=dis[x][y];
    }
    for(int i=1;i<=n;i++){
        for(int j=1;j<=n;j++){
            val[i][j]=dis[i][j];
        }
    }
    int ans = dis[0][0];
    for(int k=1;k<=n;k++){
        for(int i=1;i<k;i++){//环最起码三个点，所以i<k,j<i，避免被不是环的结果影响
            for(int j=1;j<i;j++){
                ans=min(ans,dis[i][j]+val[i][k]+val[k][j]);
                //后两个是val数组而不是dis数组，因为dis[i][k]可能是通过i->j->k来得到的，会造成不是环的结果来影响答案
                //反例(1,2,1),(2,3,1),(1,3,10)，用dis数组会错
            }
        }
        for(int i=1;i<=n;i++){//因为是dis[k-1][i][j]，所以更新写在后面
            for(int j=1;j<=n;j++){
                dis[i][j]=min(dis[i][j],dis[i][k]+dis[k][j]);
            }
        }
    }
    if(ans<dis[0][0])cout<<ans;
    else cout<<"No solution.";
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

## 数论

### 约数

存在整数c使得a=b*c，则为b整除a，即b|a

1，-1，b,-b被称作b的显然约数，其余为真约数

一个数可以根据唯一分解定理分解:n=x~1~^p1^+x`~2~`^p2^+...+x~n~^pn^,那么它的约数个数为(p~1~+1)\*(p~2~+1)\*...\*(p~n~+1)

一个数的约数的个数约为该数的三分之一次方

### 平方和/立方和公式

1^2^+2^2^+3^2^+……+n^2^=n(n+1)(2n+1)/6;    1^3^+2^3^+3^3^+……+n^3^=n^2^(n+1)^2^/4;

### 位运算（优先级低于算术运算符，运算时加括号）

#### 按位与&（同1为1），按位或|（同0为0）

a|b = a^b+a&b;

a+b = a&b+a|b;

a+b = (a&b)*2 + a^b

#### 按位取反~

将反码取反

#### 按位异或^（对应位不同为1）

##### 数的交换

```c++
void swap(int &a,int &b){
    x^=y;y^=y,x^=y;
}
```

##### 字母的大小写转化

```c++
char a='p';
a=a^' ';//a=P
a=a^32;//a=p
```

##### 连续区间异或和

对于从1到n的异或和，有结论：

```c++
int get_XOR(int i){
    switch(i%4){
        case 0 : return i;
        case 1 : return 1;
        case 2 : return i + 1;
        case 3 : return 0;
    }
}
```

#### 左移和右移     <<和>>

num<<i=num*2^i^;

num>>i=num/2^i^;

### 取模 ???

(a+b)%p=(a%p+b%p)%p

(a-b)%p=(a%p-b%p)%p

(a\*b)%p=(a%p\*b%p)%p

总结果取模和分别取模再求结果是相同的，取模次数任意。

#### a^m^%p ？？？

a^m^%p != a^m%p^%p

当a^m^%p中m很大无法输入时，如果p是一个质数，根据费马小定理有a^p-1^%p=1,所以a^m^%p=a^m%(p-1)^%p(当a%p==0时不能使用费马小定理，但是答案显然是0)

当p不是质数时，使用欧拉定理和扩展欧拉定理？？？

### 乘法

#### 快速乘

求a*b对mod取模时，如果mod大于1e9，那么a\*b可能会爆long long（同理如果mod>1e18，int128也会爆掉）

```c++
int quickmul(int a,int b){
	int res = 0;
    a%=mod;b%=mod;
    while(b){
        if(b&1){
            res+=a;res%=mod;
        }
        b>>=1;a<<=1;
    }
    return res;
}

```

也可以尝试将b分为多部分，例如a,b<2^40^，mod=2^40^
```c++
int get_mul(int a,int b){
	int res = 0;
	res+=(((a*(b>>20))%mod)<<20)%mod;
    res+=(a*(b&((1<<20)-1)))%mod;
    return res;
}
```

但是当mod接近界限时，例如a,b<mod=10^18^或2^60^,此时如果要分成一个十进制位或者三个二进制位，那么每次都得多次进行挪一部分然后取模的操作，例如b最高的三个二进制位就需要进行19次左移操作，那么一共需要进行1+2+……+19=190,写起来麻烦的同时还不如上面log~2~n复杂度的方法。

#### 大整数取模乘法

### 快速幂

求a的n次方对m的取模

```c++
int quickpow(int a,int b,int mod){
    a%=mod;		//避免直接溢出
    int re=1;
    while(b){
        if(b&1){   //b&1相当于b%2==1
            re=re*a%mod;
        }
        a=a*a%mod;
        b>>=1;   //相当于b/=2
    }
    return re;
}
```

### 素数

对一个数进行质因数分解的复杂度是O(√n)的，将1~√n的数全跑一遍，能除就除，剩下的要么是1，要么是大于√n的质因数

素数定理：当x很大时，小于x的素数的个数趋近于x/ln(x)

### 同余

n|(a-b),称a和b模n同余，也就是a%n=b%n,记作a≡b(mod n)。

可加性：a≡b(mod n),c≡d(mod n) => (a+c)≡(b+d)(mod n)

可乘性：a≡b(mod n),c≡d(mod n) => (a*c)≡(b\*d)(mod n)

### 公约数和公倍数

一组数最大公约数记为(a,b,c,...,n),最小公倍数记为[a,b,c,...,n]。

```c++
最大公约数在算法中记为gcd(a,b),最小公倍数写为lcm(a,b)。
求最大公约数函数：__gcd（a,b）
#include<algorithm>
cout<<__gcd(a,b);//多个不行，本质就是辗转相除加递归
最小公倍数为a*b/__gcd(a,b)。
```

#### 更相减损术

gcd(a,b)=gcd(a,b-a);多次进行

gcd(a,a)=a;

#### 辗转相除法（欧几里得算法）

a%b=c,则gcd(a,b)=gcd(b,c)

```c++
int gcd(int a,int b){
    return b?gcd(b,a%b):a;
}
//o(log max(a,b))，每次最大的数都不超过原本的一半
```

#### stein

```c++
int stein(int a,int b){
	if(a<b) a^=b,b^=a,a^=b; //交换，使a为较大数； 
	if(b==0)return a;
	if((!(a&1))&&(!(b&1))) return stein(a>>1,b>>1)<<1;//都是偶数
	else if((a&1)&&(!(b&1)))return stein(a,b>>1);//一奇一偶
	else if((!(a&1))&&(b&1))return stein(a>>1,b);
	else return stein(a-b,b);//都是奇数
} 
```

### 组合数学 ？？？

#### 排列数

A~n~^m^ = n!/(n-m)! = P~n~^m^

A~n~^m^ = n*A~n-1~^m-1^

#### 组合数

C~n~^m^ = A~n~^m^/(m!) = n!/[(n-m)!*(m!)] = (^n^~m~)

C~n~^m^+C~n~^m-1^=C~n+1~^m^ （递推式）

m*C~n~^m^ = n\*C~n-1~^m-1^

∑~i=0~^n^ C~n~^i^ = 2^n^    ∑~i=0~^n^ 2^i^C~n~^i^ = 3^n^

##### 杨辉三角求组合数

```c++
int C[maxn][maxn];
void get_c(int w){
   for (int i=0; i<=w;i++)
      for (int j=0; j<=i;j++)
         if (j==0)C[i][j] = 1;
             else C[i][j]=(C[i-1][j] + C[i-1][j-1])%mod;
}
```

##### 求单个组合数

```c++
int inv[maxn], fac[maxn];
void get_c (int w) {//预处理
    inv[1] = 1;
    for (int i = 2; i <= w; i++) {
        if (i < mod) inv[i] = ((mod - mod/i) * inv[mod % i]) % mod;
        else inv[i] = inv[i % mod];//大于mod的情况取余就可以
    }//先预处理多个数的逆元
    inv[0] = 1;//0！的逆元是1，在求C(x,0)和C(x,x)的时候会用上
    fac[0] = 1;
    for (int i = 1; i <= w; i++) {
        inv[i] = inv[i] * inv[i-1] % mod;//预处理求出i!的逆元
        fac[i] = i * fac[i-1] % mod;
    }
}
//O(n)的预处理后得到O(1)的查询，如果mod更换需要重新进行预处理
int C (int i, int j) {
    if (i < j || i == 0) return 0;
    return ((fac[i] * inv[j]) % mod) * inv[i - j] %mod;
}
```

#### 隔板法

将n个球分进m个盒子里的方案数

要求:球与球之间没区别，盒子和盒子之间有区别（有顺序）

##### 每个盒子至少一个球

n个球中间有n-1个空隙，插入m-1个隔板将其分成m份，方案数为C~n-1~^m-1^

##### 盒子可以是空的

补m个球，n+m个球共有n+m-1个空隙，插入m-1个隔板将其分成m份，然后从每份中拿出一球，方案数为C~n+m-1~^m-1^

##### 盒子中至少有k+1个球

拿走k\*m个球，n-k\*m个球共有n-k\*m-1个空隙，插入m-1个隔板将其分成m份，然后给每份中放入k个球，方案数为C~n-k*m-1~^m-1^

##### 混合

可以为0的补一个球，需要大于k+1的拿走k个球

#### 多重集组合数 ？？？

从n个不同元素的集合中，取m个元素组成一个新集合，每取完一个 元素再将其放回原集合中，求新集合的个数.

#### 抽屉原理（鸽巢原理）

将n+1个物品划分成n组，至少一组有两个（或以上）物品

推论：将n个物品划分成k组，至少存在一组有⌈n/k⌉（或以上）物品

#### 容斥原理 ？？？



#### 贝尔数 ？？？

[[D - Stone XOR](https://atcoder.jp/contests/abc390/tasks/abc390_d)]如何将一个集合分为多个两两不相交的集合



### 线代知识

#### 矩阵乘法加速线性递归

F~n~是斐波那契数列的第n个数，请你求出 F~n~mod10^9^+7 的值。

输入一个正整数 n<=2^63^。输出一个整数表示答案。

![矩阵乘法加速递推](../../../Pictures/Screenshots/typora图片/矩阵乘法加速递推.png)

 矩阵乘法满足结合律，先将递推矩阵的n次方用快速幂解出

```c++
#include<iostream>
using namespace std;
#define int long long
const int mod=1e9+7;
struct node{
    int arr[2][2]={0};
};

node cheng(node a,node b){
    node res;
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            for(int k=0;k<2;k++){
                res.arr[i][j]+=(a.arr[i][k]*b.arr[k][j])%mod;
                res.arr[i][j]%=mod;
            }
        }
    }
    return res;
}

node quickpow(node arr,int n){
	node sum;
    sum.arr[0][0]=1,sum.arr[0][1]=1;//[a2,a1]=[1,1],相当于A2,需要看情况初始化（洛谷P1349）
	while(n>0){
        if(n&1)sum=cheng(sum,arr);
        arr=cheng(arr,arr);
        n>>=1;
    }
    return sum;
}
signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    int n;cin>>n;
    node tem;
    tem.arr[0][0]=1;tem.arr[0][1]=1;tem.arr[1][0]=1;
    cout<<quickpow(tem,n-2).arr[0][0];
}
```

将递推关系变为乘法，再用快速幂将复杂度由o(n)降为o(log n)。

```c++
struct matrix {
    vector<vector<int> > mt;
    int N, M;

    matrix() {
        N = M = 10;
        mt.resize(N + 1,vector<int>(M + 1, 0));
    }
    matrix(int x, int y) {
        N = x; M = y;
        mt.resize(N + 1,vector<int>(M + 1, 0));
    }

    void init(int W) { //单位矩阵
        if (N < W || M < W) {
            cout << "init:矩阵不合法" << endl;
            return ;
        }
        for (int i = 1; i <= W; i++) {
            for (int j = 1; j <= W; j++) {
                mt[i][j] = (i == j);
            }
        }
    }
};

matrix mat_mul(matrix & mt1, matrix & mt2, int x, int y, int z) {
    matrix res(x, z);
    for (int i = 1; i <= x; i++) {
        for (int j = 1; j <= z; j++) {
            int t = 0;
            for (int k = 1; k <= y; k++) {
                t += (mt1.mt[i][k] * mt2.mt[k][j]) % mod;
                t %= mod;
            }
            res.mt[i][j] = t;
        }
    }
    return res;
}

matrix qp_mat (matrix & x, int y) {
    matrix res(x.N, x.M);
    res.init(res.N);

    if (x.N != x.M) {
        cout << "qp_mat:矩阵不合法" << endl;
        return res;
    }
    while (y) {
        if (y & 1) {
            res = mat_mul (res, x, res.N, res.N, res.N);
        }
        x = mat_mul (x, x, res.N, res.N, res.N);
        y >>= 1;
    }
    return res;
}
```



##### 如何求递推式（可能正确）

对于f~n~=g(f~n-1~,f~n-2~,...,f~t~)   （中间可能有些f并不存在）

可以令F~n~=[f~n~,f~n-1~,...,f~t+1~],则F~n+1~=[f~n+1~,f~n~,...,f~t+2~]，其中f~n+1~需要通过递推式表达，其余直接在F~n~中存在，直接使用就可以

![{00011260-77C8-47B8-9146-41952FA664F1}](../../../AppData/Local/Packages/MicrosoftWindows.Client.CBS_cw5n1h2txyewy/TempState/ScreenClip/{00011260-77C8-47B8-9146-41952FA664F1}.png)

递推式中出现了f~n~,f~n-1~,f~n-2~,n,3^n^五项，（其中f~n-2~不需要）

先假设F~n~=[f~n~,f~n-1~,n,3^n^], 得到F~n+1~=[f~n+1~,f~n~,n+1,3^n^*3] 

由于F~n+1~中需要常数1，

所以F~n~=[f~n~,f~n-1~,n,3^n^,1],F~n+1~=[f~n+1~,f~n~,n+1,3^n^*3,1]

#### 高斯消元

O(n^3^)

```c++
#include<bits/stdc++.h>
using namespace std;

double arr[105][105];

int main(){
    int n; cin >> n;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n + 1; j++) {
            cin >> arr[i][j];
        }
    }
    for (int i = 1; i <= n; i++ ) { //第i列
        int now = i; //从第i行向下选取第i列主元
        for (int j = i + 1; j <= n; j++) {
            if (abs(arr[j][i]) > abs(arr[now][i])) now = j;
        }//选取该列绝对值最大的值可以减小误差
        if (abs(arr[now][i]) < 1e-9) {
            cout << "No Solution"; return 0;
        } //不满秩则说明多解或无解
        if (now != i) swap(arr[now], arr[i]);//将第now行换到第i行
        for (int j = n + 1; j >= i; j--) { //第1~i列的数已经被消掉了
            arr[i][j] /= arr[i][i]; //倒着除，将主元变为1
        }
        for (int j = 1; j <= n; j++) {
            if (j == i) continue;
            double r = arr[j][i];
            for (int k = 1; k <= n + 1; k++) {
                arr[j][k] -= r * arr[i][k]; //用主元将该列其它数都变为0
            }
        }
    }
    for (int i = 1; i <= n; i++) cout << arr[i][n + 1] << endl;
}
```

```c++
//将多解和无解分开表示
#include<bits/stdc++.h>
using namespace std;
#define int long long

double arr[105][105];

vector<int>vec;//vec用来存有待最终检查的行

signed main(){
    int n; cin >> n;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n + 1; j++) {
            cin >> arr[i][j];
        }
    }
    for (int i = 1; i <= n; i++ ) { //第i列
        int now = i; //从第i行向下选取第i列主元
        for (int j = i + 1; j <= n; j++) {
            if (abs(arr[j][i]) > abs(arr[now][i])) now = j;
        }//选取该列绝对值最大的值可以减小误差
        if (abs(arr[now][i]) < 1e-9) {
            for (auto xx : vec){
                if (abs(arr[xx][i]) > 1e-9) {
                    //如果之前有待检查的行在这一列不为0，那么可以交换
                    //不待检查的行不要考虑，因为该行已经有一个主元，如果把新的一列再拿来当主元来消去剩下的行，那么原本的主元那列又被破坏了
                    swap(arr[now], arr[xx]);
                    break;
                }
            }
        }
        if (abs(arr[now][i]) < 1e-9) {vec.push_back(now);continue;}
        //如果不存在，就把这一行放到vec中待检查
        if (now != i) swap(arr[now], arr[i]);//将第now行换到第i行
        for (int j = n + 1; j >= i; j--) { //第1~i列的数已经被消掉了
            arr[i][j] /= arr[i][i]; //倒着除，将主元变为1
        }
        for (int j = 1; j <= n; j++) {
            if (j == i) continue;
            double r = arr[j][i];
            for (int k = 1; k <= n + 1; k++) {
                arr[j][k] -= r * arr[i][k]; //用主元将该列其它数都变为0
            }
        }
    }
    for(auto xx : vec){
        if (abs(arr[xx][n+1])>1e-9){
            cout << "无解"; return 0;//存在无解情况输出无解
        }
    }
    if (vec.size()) {
        cout << "多解";return 0;
    }
    for (int i = 1; i <= n; i++) cout <<fixed<<setprecision(2)<< "x" << i << "=" <<arr[i][n + 1] << endl;
}
```



如果按行来判断的话，在一行都为0的情况直接判断是否无解，全都不是无解的时候在最后输出多解

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
double arr[105][105];
int n,m,k,x,y,z,N=1;

double ans[maxn];
bool tag = 0 ;

void work(){
    cin >> n;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n + 1; j++) {
            cin >> arr[i][j];
        }
    }
    for (int i = 1; i <= n; i++ ) {
        int now = 1; // 寻找第now列作为主元
        for (int j = 2; j <= n; j++) {
            if (abs(arr[i][j]) > abs(arr[i][now])) now = j;
        }
        if (fabs(arr[i][now]) < 1e-9) {
            if(fabs(arr[i][n + 1]) < 1e-9)tag=1;//多解暂存标记
            else {cout<<-1;return ;}//无解直接输出
            continue;
        }

        double t = arr[i][now];
        for (int j = 1; j <= n + 1; j++) {//主元变为1
            arr[i][j] /= t;
        }
        for (int j = 1; j <= n; j++) {//将该列其它数消为0
            if (j == i) continue;
            double r = arr[j][now];
            for (int k = 1; k <= n + 1; k++) {
                arr[j][k] -= r * arr[i][k];
            }
        }
    }
    if(tag){
        cout<<0;return ;
    }
    for (int i = 1; i <= n; i++ ) {
        for (int j = 1; j <= n; j++) {
            if (arr[i][j] == 1) {
                ans[j] = arr[i][n+1];
            }
        }
    }
    for (int i = 1; i <= n; i++) cout <<fixed<<setprecision(2)<< "x" << i << "=" <<ans[i] << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

#### 线性基

```c++
//异或线性基
struct Basis{
	int base[64],len;

	Basis() {memset(base,0,sizeof(base));}
    Basis(int t) {len = t; memset(base,0,sizeof(base));}

	bool insert(int x) {
		for(int i = len; i >= 0; i--){
            if ((x >> i) & 1) {
                if (base[i]) x ^= base[i];
                else {base[i] = x; return true;}
            }
        }
        return false;
	}

    bool check(int x) { //判断当前线性基能否组成x
        for(int i = len; i >= 0; i--){
            if ((x >> i) & 1) {
                if (base[i]) x ^= base[i];
                else {return true;}
            }
        }
        return false;
    }

    int get_m_val(int res, bool op) {//res和集合中的数相异或，取其最值，op==0代表最小值，op==1代表最大值，当res=0时代表集合内异或和最值
        for (int i = len; i >= 0; i--) {
            if (((res >> i) & 1) && !op) res ^= base[i];
            if (!((res >> i) & 1) && op) res ^= base[i];
        }
        return res;
    }

    int get_kth_val(int t,int op) {//op==0代表第k小,op==1代表第k大
        for (int i = len; i >= 0; i--) { //重构线性基，进行求秩过程,收集非零向量
            for (int j = i - 1; j >= 0; j--) {
                if ((base[i] >> j) & 1) base[i] ^= base[j];
            }
        }

        int cnt = 0, temp[len + 1];//将存在值的base汇总到temp上面
        for (int i = 0; i <= len; i++) {
            if (base[i]) temp[cnt++] = base[i];
        }

        //这里考虑的是非空集合时是否存在0,如果允许空集合存在那么不需要再进行判断,直接考虑0
        int sum = (1LL << cnt) - 1;
        if (cnt < n) sum++;  //个数不同则存在0
        if (t > sum) return -1;  //超过总个数
        if (op == 1) t = sum + 1 - t; //将求第k大转化为求第k小
        if (cnt < n) t--;
        if (t == 0) return 0;

        int res = 0;
        for (int i = cnt - 1; i >= 0; i--) {//选取第i位时就已经超过了不选取时对应的2^i - 1个数
            if ((t >> i) & 1) res ^= temp[i];
        }
        return res;
    }
};
```

[[P3265 JLOI2015 装备购买 - 洛谷](https://www.luogu.com.cn/problem/P3265)] ：向量的线性基

n个装备，每个装备m个属性，当一个装备可以被已购买的装备表示则不需要购买，问最小花费

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'

struct node{
    array<double,505> t;
    int v;
}arr[505];

const double esp = 1e-3;
int n,m,N=1;

struct Basis{
	array<double,505> base[505];
    int v[505];

	Basis() {
        memset(base,0,sizeof(base));
        memset(v,0,sizeof(v));
    }

	bool insert(array<double,505> x,int val) {
		for(int i = 1; i <= m; i++) {
            if (x[i] < -esp || x[i] > esp) {
                if (base[i][i] >= -esp && base[i][i] <= esp ){
                    base[i] = x;
                    v[i] = val;
                    return true;
                }else {
                    for(int j = i + 1; j <= m; j++) {
                        x[j] -= x[i] / base[i][i] * base[i][j];
                    }
                    x[i] = 0;
                }
            }
        }
        return false;
	}

    void get_sum() {
        int res = 0, sum = 0;
        for (int i = 1; i <= m; i++) {
            res += v[i];
            if(base[i][i]>esp||base[i][i]<-esp) sum++;
        }
        cout << sum <<' ' <<res <<endl;
    }
};

bool cmp(node a,node b){
    return a.v<b.v;
}

void work(){
    Basis B;
    cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            cin >> arr[i].t[j];
        }
    }
    for(int i = 1; i <= n ; i++) cin >>arr[i].v;
    sort(arr+1,arr+1+n,cmp);
    for(int i=1;i<=n;i++){
        B.insert(arr[i].t,arr[i].v);
    }

    B.get_sum();
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    for (int i = 1; i <= N; i++) {
        work();
    }
}
```

### 筛法

#### 埃氏筛

原理：素数的倍数不是素数

基本思路：每次找到一个素数时，将素数的倍数全部筛去。

时间复杂度为 O(n log(logn))

```c++
int n;cin>>n;int cnt=0;
int prime[maxn],vis[maxn]={0};
for(int i=2;i<=n;i++){
    if(!vis[i]){
        sum[++cnt]=i;//将素数存入数组
        for(int a=2;a*i<=n;a++){
            vis[a*i]=1;//将素数的倍数剔除
        }
    } 
}
```

#### 线性筛

顾名思义，线性筛会将时间复杂度降低到线性，即 O(n)。

基本思路：埃氏筛中，一个数会被筛去多次，如15会被3和5各筛去了一次。如果我们让一个数只会被其素数约数中最小的筛去一次，可以大大节省时间。

```c++
bool vis[maxn]={0};
int prime[maxn],cnt=0;
void get_prime(int w){
	for(int i=2;i<=w;i++){
    	if(!vis[i]){prime[++cnt]=i;}
    	for(int j=1;j<=cnt&&i*prime[j]<=w;j++){
       		vis[i*prime[j]]=true;
        	if(i%prime[j]==0)break;
    	}
	}
}
```

```c++
vis[i*prime[j]]=true;代表i*prime[j]这个数被因数prime[j]筛去
if(i%prime[j]==0)break;这个跳出保证了一个数只会被它最小的因数筛去
例如一个数x有最小的两个因数x1,x2（x1<x2）x=x1*x2*t;
不会被x2筛去：因为被x2筛去意味着(x1*t)*x2，但是(x1*t)%x1==0,所以x1*t这个数在x1这里就会跳出
会被x1筛去:被x1筛去意味着(x2*t)*x1,而x2*t不存在比x1更小的因数，所以中途不会跳出
```

#### 区间筛

时间复杂度O((r-l+1)*loglog(sqrt(r) ) 

用来解决r较大，但是r-l不是非常大时，求[l,r]中的素数或者求所有数的质因数分解

先将2到sqrt(r)的素数找出来，然后通过r/prime*prime找到不大于r的最大的prime的倍数，然后通过减去prime的方法来找其它倍数，如果求素数就打上标记，最后数一遍。如果是求质因数，就将这些数一直除以prime，记录有几个prime，最后要是sqrt(r)判断完了之后，这些数仍不是1，说明它还有一个大于sqrt(r)的质数，直接记录1个就行。

[[#10197. Prime Distance - 题目 - LibreOJ](https://loj.ac/p/10197)]

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn = 2e6+10;
int l,r;

bool vis[maxn];
int cnt = 0;
int prime[maxn];
void get_prime(int w){
    for(int i=2;i<=w;i++){
        if(!vis[i])prime[++cnt]=i;
        for(int j=1;j<=cnt&&i*prime[j]<=w;j++){
            vis[i*prime[j]]=1;
            if(i%prime[j]==0)break;
        }
    }
}

bool mp[maxn];
void work(){
    for(int i=l;i<=r;i++){
        mp[i-l+1]=0;
    }
    if(l==1)mp[1]=1;//对1进行特判
    int k = lower_bound(prime+1,prime+1+cnt,(int)sqrt(r)+1)-prime;
    for(int i=1;i<=k;i++){
        int now = r/prime[i]*prime[i];//找到不大于r的第一个倍数
        while(now>=l&&now>prime[i]){//now>prime[i],否则该质数被误判
            mp[now-l+1]=1;
            now-=prime[i];
        }
    }
    vector<int>vec;
    for(int i=l;i<=r;i++){
        if(mp[i-l+1]==0)vec.push_back(i);
    }
    if(vec.size()<=1){
        cout<<"There are no adjacent primes."<<endl;
    }else {
        int ans1=inf,ans2=-inf;
        int r1,r2;
        for(int i=1;i<vec.size();i++){
            int tem = vec[i] - vec[i-1];
            if (tem < ans1){
                ans1 = tem;
                r1 = i;
            }
            if (tem > ans2){
                ans2 = tem;
                r2 = i;
            }
        }
        cout<<vec[r1-1]<<','<<vec[r1]<<" are closest, ";
        cout<<vec[r2-1]<<','<<vec[r2]<<" are most distant."<<endl;
    }
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    get_prime(maxn/2);
    while (cin>>l>>r) {
        work();
    }
}
```



### 欧拉函数

欧拉函数, 即φ(n)，它表示从 1 − n 中和它互质（最大公约数为1）的数的个数。 例如φ(1) = 1, φ(2) = 1, φ(6) = 2。

性质1:如果n是质数,那么φ(n）=n-1 ,因为只有n本身与它不互质。。

性质2:如果p|q，则φ(p*q)=p\*φ(q)。

性质3:如果p是质数,那么φ(p^n^)=p^n^-p^n-1^=p^n-1^*(p-1)。

积性：如果p,q互质那么φ(p\*q)=φ(p)\*φ(q)。特别地，如果n%2==1,那么φ(2*n)=φ(n)

#### 欧拉函数通用求解公式

 对于任意正整数n,φ(n)=n(1-1/p~1~)(1-1/p~2~)......（1-1/p~m~),其中p~1~到p~m~是n的不同质因子

原因：以12为例，φ(12)=12*(1-1/2)\*(1-1/3)=4,因为1~12里面有1/2的比例是2的倍数，剩下的数中有1/3的比例是3的倍数，所以分别乘上(1-1/2)和(1-1/3)。

用欧拉函数求解公式来求单个数时间复杂度为O(√n)。

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
signed main(){   
    int n;cin>>n;
	int ans=n;
    int k = sqrt(n) + 1;
	for(int i=2;i<=k;i++){
        if(n%i==0){
            ans=ans/i*(i-1);
			while(n%i==0){n/=i;}//将i统统除掉
		}
	}
	if(n>1) ans=ans/n*(n-1);//说明还有一个大于sqrt(n)的质因数没算，不然的话n应该为1
	cout<<ans<<endl;
}
```

如果求多个可以用线性筛来优化，将时间复杂度从O(n√n)降到O(n)

```c++
#include<bits/stdc++.h>
#define int long long
using namespace std;
const int maxn = 2e6+5;
int phi[maxn],prime[maxn],cnt;
bool vis[maxn];

void get_phi(int w){
	phi[1]=1;
	for(int i=2;i<= w;i++){
		if (!vis[i]){
			prime[++cnt]=i;
			phi[i]=i-1; // 性质1
		}
		for(int j=1;j<=cnt&&i*prime[j]<=w;j++){
			vis[i*prime[j]]=1;
			if (i%prime[j]==0) {
				phi[i*prime[j]]=phi[i]*prime[j];//性质2
				break;
			}
			phi[i*prime[j]]=phi[i]*phi[prime[j]];//积性,如果i不是prime[j]的倍数那么一定互质
		}
	}
}
```

[欧拉函数的性质及其三个方法](https://blog.csdn.net/Brain_Gym/article/details/128571823?ops_request_misc=%7B%22request%5Fid%22%3A%22170554457416800227419950%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170554457416800227419950&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-128571823-null-null.142^v99^pc_search_result_base6&utm_term=欧拉函数&spm=1018.2226.3001.4187)

### 欧拉定理

若gcd(a,m)=1，则a^φ(m)^≡1 (mod m)。

当m为质数时，该定理退化成费马小定理

推论：若gcd(a,m)=1，a^b^≡a^b%φ(m)^ (mod m)

证明：b=k*φ(m)+b%φ(m)  =>  a^b^=(a^φ(m)^)^k^ \* a^b%φ(m)^=1^k^ \* a^b%φ(m)^=a^b%φ(m)^

### 扩展欧拉定理？？？

a^b^≡a^b%φ(m)+φ(m)^ (mod m)    **(b≥φ(m))**

当a,m互质时可由上面推论得到

当a,m不互质时？？？

### 积性函数？？？

### 贝祖定理（裴蜀定理）

存在整数x,y(可以是负数)使ax+by=c有解的充必条件是gcd(a,b)|c，反之则无解。

当给定a,b时，gcd(a,b)是ax+by的最小正整数。

推论：ax+by=1有解 <==> gcd(a,b)=1;

### 扩展欧几里得算法

[【数论系列】 欧几里得算法与拓展欧几里得_扩展欧几里得-CSDN博客](https://blog.csdn.net/qq_40772692/article/details/81183174?ops_request_misc=%7B%22request%5Fid%22%3A%22cdb797f588dec158d49858c21e246473%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=cdb797f588dec158d49858c21e246473&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-81183174-null-null.142^v101^pc_search_result_base2&utm_term=扩展欧几里得算法&spm=1018.2226.3001.4449)

用来求ax + by = gcd(a,b)中x,y的一组特解，同时求出了gcd(a,b)

ax~1~+by~1~=gcd(a,b);

bx~2~+(a%b)y~2~=gcd(b，a%b);

由欧几里得算法可知，gcd(a,b)=gcd(b,a%b);

所以ax~1~+by~1~=bx~2~+(a%b)y~2~,

又因为a%b=a-(a/b)*b

所以ax~1~+by~1~=bx~2~+(a-(a/b)*b)y~2~=ay~2~+b[x~2~- (a/b)\*y~2~]

解得x~1~=y~2~,y~1~=x~2~-(a/b)*y~2~

递归至a%b==0时取1,0满足。（后面的那个数可以随意)

想要求(x~1~,y~1~),就求(x~2~,y~2~),...,(x~n~,y~n~),(1,0)

```c++
int exgcd(int a,int b,int &x,int &y){   //也可以将x和y设为全局变量
    if(b==0){x=1;y=0;return a;}//此时的a就是gcd
    int gcd=exgcd(b,a%b,y,x);  //返回时，y=x2,x=y2。这一步x和y传的时候倒过来了，如果正着传的话在下面进行交换也可以
    y-=(a/b)*x;  //y=x2-(a/b)*y2
    return gcd;  //返回的是最大公约数
}
//可以注意一下这里如何使用引用传递来达到逆推的目的
```

对于ax + by = k*gcd(a,b)，把k从方程提出来，最后求出来的x,y都乘上k

对于ax + by = c，我们可以先通过exgcd求出gcd和x,y，然后根据c和gcd的关系再修改x和y

### 同余方程

[[P1082 同余方程](https://www.luogu.com.cn/problem/P1082)]

求关于 x 的同余方程 𝑎𝑥≡1(mod𝑏)的最小正整数解。

有ax % b=1 % b =1,所以ax = by + 1,即ax + by = 1。

当ax+by=t成立时，t是gcd(a,b)的倍数（因为a,b都是gcd(a,b)的倍数），且在这里t=1，所以gcd(a,b)=1是这个方程成立的必要条件。

原式变成了求ax+by=gcd(a,b)中x的最小正整数解，使用扩展欧几里得算法

```c++
int main(){
    int a,b,x,y;   //ax+by=gcd(a,b),
    cin>>a>>b;     //a和b都是已知数
    int t=exgcd(a,b,x,y);   //t=gcd(a,b)
    if(t==1)cout<<(x%b+b)%b;   //批量地加减b以使得x为最小正整数,t==1->a,b互质->有解
}
```

#### 𝑎𝑥≡[k*gcd(a,b)\](mod𝑏)如何求最小正整数

按照上面的方法用扩展欧几里得求出x,然后再乘上k就可以了

#### 特解->通解

当特解是（x,y）时，通解为（x+lcm(a,b)/a*n,y-lcm(a,b)/b\*n）**n∈Z**

### 逆元

如果存在x使𝑎𝑥≡1(mod p)，则称x为a模p意义下的乘法逆元，记作x≡a^-1^(mod p)

乘法逆元存在的条件是gcd(a,p)=1

#### exgcd

𝑎𝑥≡1(mod p) -> 同余方程

#### 费马小定理

欧拉定理：若gcd(a,p)=1，则a^φ(p)^≡1 (mod p)。

当p为质数时，有a^p-1^=a * a^p-2^ ≡1(mod p),所以a^p-2^是逆元

运用要求：p是质数且gcd(a,p)=1

```c++
int quickpow(int a,int q,int mod){
    int re=1;
    while(q){
        if(q&1){
            re=re*a%mod;
        }
        a=a*a%mod;
        q>>=1;
    }return re;
}
cout<<quickpow(a,p-2,p);
```

#### 连续求多个数的逆元

当模数为p时，对于i(i<p)来说，设p/i=x, p%i=y,有p=x*i+y,所以x\*i+y≡0（mod p) 

两边乘上i^-1^ * y^-1^ ,得x* y^-1^+i^-1^≡0 (mod p) ==> i^-1^=-x*y^-1^ (mod p)

 因为y=p%i<i，所以得到递推式 i^-1^=-(p/i)*(p%i)^-1^ (mod p)

```c++
int inv[maxn];
void init (int w) {
    inv[1] = 1;
    for(int i = 2; i <= w; i++){
        if(i < mod)inv[i] = ((mod - mod/i) * inv[mod%i]) % mod;
        //(mod-mod/i)保证是正数
        //inv[i]==0 -> inv[mod%i]==0 -> mod%i==0 -> 不存在逆元
        else inv[i] = inv[i % mod];
    }
}
```

#### 有理数取余

[[P2613 【模板】有理数取余](https://www.luogu.com.cn/problem/P2613)] 

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int mod = 19260817;

inline void read(int &x){
    int s=0,w=1;char ch=getchar();
    while(ch<'0'||ch>'9'){if(ch=='-')w=-1;ch=getchar();}
    while(ch>='0'&&ch<='9'){
        s=((s*10)+ch-'0')%mod;//在快读时取模
        ch=getchar();
    }
    x=s*w;
}
void exgcd(int a,int b,int &x,int &y){
    if(b==0){
        x=1;y=0;cout<<a<<endl;return ;
    }
    exgcd(b,a%b,y,x);
    y-=a/b*x;
}

void work(){
    int n,m;
    read(n);read(m);
    if(m==0){
        cout<<0;return ;
    }
    if(n==0){
        cout<<"Angry!";return ;
    }
    int x,y;
    exgcd(m,mod,x,y);
    cout<<(n*x%mod+mod)%mod;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

### 中国剩余定理与扩展中国剩余定理 ？？？

用来求一元线性同余方程组的一个实数解

<img src="../../../AppData/Roaming/Typora/typora-user-images/image-20250221154302919.png" alt="image-20250221154302919" style="zoom:67%;" />

a~i~,m~k~给定，求x

#### 中国剩余定理（CRT）(代码封装，注意负数)

当gcd(m~1~,m~2~,...,m~k~)==1时，求解x的方法称为中国剩余定理

构造法，构造一个特解

取M=m~1~*m~2~\*...\*m~k~，设p~i~=M/m~i~

求出p~i~在模m~i~意义下的逆元p~i~^-1^

得到解x=∑~i=1~^k^ a~i~ p~i~ p~i~^-1^

证明：首先a~i~ p~i~ p~i~^-1^ ≡ a~i~ (mod m~i~),因为 p~i~ p~i~^-1^ ≡ 1 (mod m~i~)

其次，a~i~ p~i~ p~i~^-1^ ≡ 0 (mod m~j~) ,因为p~i~是m~i~的倍数

```c++
int M=1,ans=0;
void exgcd(int a,int b,int &x,int &y){
    if(b==0){x=1;y=0;return ;}
    exgcd(b,a%b,y,x);
    y-=a/b*x;
}

int crt(int now){
    int m = M/mod[now];
    int x,y;
    exgcd(m,mod[now],x,y);//mod[i]之间互质，所以m和mod[now]互质，逆元x存在
    x%=mod[now];x+=mod[now];x%=mod[now];//x有多解，找到最小正整数解
    return (arr[now]*m*x%M+M)%M;//考虑arr[now]*m*x越界问题
}

int CRT(){
    for(int i=1;i<=n;i++)M*=mod[i];//注意M是否越界
    for(int i=1;i<=n;i++)ans=(ans+crt(i))%M;
    return ans;
}

void work(){
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>mod[i]>>arr[i];
    }
	cout<<CRT();
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

### 卢卡斯定理与扩展卢卡斯定理？？？

#### 卢卡斯定理

当==p为质数==时，（^n^~m~）mod p = (^n/p^~m/p~) * (^n%p^~m%p~) mod p

如果i<j, (^i^~j~) =0;

可以用来解决n,m很大，但是p不是很大的情况

```c++
int lucas(int i,int j){
    if(i<j)return 0;
    if(i<=mod&&j<=mod)return C(i,j);//组合数O(1)求出
    else return (lucas(i/mod,j/mod)*lucas(i%mod,j%mod))%mod;
}
```

#### 扩展卢卡斯定理  ？？？

### 博弈论 ？？？

### 数值计算

#### 牛顿迭代法

迭代求零点

对于一个f(x) = 0的问题，通过x~i+1~ = x~i~ + f(x~i~) / f ^‘^(x~i~) 来不断迭代得到解，当两个解的差距小于eps时就退出迭代，返回近似解

```c++
//平方根的求解：
double sqrt_newton(double t) {
  constexpr static double eps = 1E-15;
  double x = 1;
  while (true) {
    double nx = (x + t / x) / 2;
    if (abs(x - nx) < eps) break;
    x = nx;
  }
  return x;
}

//求解整数平方根，即最大的满足x² <= t的整数
int isqrt_newton(int t) {
    if (t == 0 || t == 1) return t;

    int y = t / 2;  // 初始值可以取 t/2，避免过小

    bool decreased = false;
    while (true) {
        int ny = (y + t / y) >> 1;
        if (y == ny || (ny > y && decreased)) break;
        decreased = ny < y;
        y = ny;
    }
    return y;
}
```



## 动态规划(dp)

### 题目

[[P3842 [TJOI2007\] 线段 ](https://www.luogu.com.cn/problem/P3842)]

### 时间复杂度

认为一秒计算机在1e9左右。

n~1e7       O(n)

n~1e5到1e6  O(nlogn)

n~5e4到1e5 O(n√n)

n~5000 O(n²)

n~300 O(n³)

n~20 O(2^n^)

### 动态规划基础

####  LIS

[[B3637 最长上升子序列](https://www.luogu.com.cn/problem/B3637)]:

```c++
//朴素解O(n²)： 
    int n;cin>>n;
    for(int i=0;i<n;i++){
    	cin>>arr[i];
	}
	int k=0;
	for(int i=0;i<n;i++){
		for(int j=0;j<i;j++){
            if(arr[i]>arr[j])dp[i]=max(dp[i],dp[j]);
		}
		dp[i]++;
	}
	int ans = 0;
	for(int i = 0; i < n; i++) ans = max(ans, dp[i]);
	cout << ans;

/*
O(nlogn):贪心+二分
dp[i]存的是长度为i的上升序列中最后一个数的最小值，且这个数组是单调递增的，不断维护dp数组，最后数组长度就是答案
对于下降序列，可以将数组倒过来，这样可以使用lower_bound和upper_bound，也可以使lower_bound(arr,arr+n,x,greater<int>);
*/
    int dp[maxn], x;
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> x;
        int t = lower_bound(dp + 1, dp + 1 + cnt, x) - dp;
        dp[t] = x;
        if (t > cnt) ++cnt;
    }
    cout << cnt << endl;
/*
O(nlogn):树状数组
在无撤销操作时，树状数组可以用来查询前缀最值
将每个数的大小及对应长度插入树状数组，可以logn来查询小于某个数的最大长度
过程中可能需要离散化
*/
	//树状数组操作
	int ans = 0;
    for (int i = 1; i <= n; i++) {
        int t = query(num[i] - 1) + 1;
        modify(num[i], t);
        ans = max(ans, t);
    }
	cout << ans << endl;
```

#### 狄尔沃斯定理

[[P1020 [NOIP1999 提高组\] 导弹拦截 - 洛谷](https://www.luogu.com.cn/problem/P1020)]:

序列的最长单调上升序列长度等于所给序列被划分为的==单调非上升序列数目==的最小值

#### LCS

[[P1439 【模板】最长公共子序列](https://www.luogu.com.cn/problem/P1439)]

朴素解法：

```c++
	int arr[maxn]={0};
    int brr[maxn]={0};
    int dp[2][maxn]={0};
    int k=0;
    memset(dp,0,sizeof(dp));
	for(int i=1;i<=n;i++){
		k=!k;       //滚动数组
		for(int j=1;j<=n;j++){
			if(arr[i]==brr[j])dp[k][j]=1+dp[!k][j-1];
			else dp[k][j]=max(dp[!k][j],dp[k][j-1]);
		}
	}
	cout<<dp[k][n];
```

O(nlogn):

由于给的是两个排列这一特殊性，可以将第一个数组给的顺序映射到1,2,3,...,n上，认为是一个递增的顺序，后面对下一个数组改成对应的关系，然后求第二个数组的最长上升子序列。

[[最长公共子串 - 力扣](https://leetcode.cn/problems/maximum-length-of-repeated-subarray/)]	==遇到子串可以尝试联想哈希==

朴素解法是O(n^3^)的复杂度，取不同的两个位置，然后逐一对比。在这个过程中，num1[i]和num2[j]可能会被比对多次（如假设num1[i-1]==num2[j-1],在比对从(i-1,j-1)开始的时候也会对num1[i]和num2[j]进行比对)，所以复杂度多乘了一个n。

方法一通过动态规划记录了比对过的结果。

方法二滑动窗口，一直对比下去，过程中借用了之前对比留下的now

方法三哈希????

```c++
	int dp[manx][maxn];//dp[i][j]数组指的是以i和j为结尾的长度，这样看着比较顺眼，记录以i和j开头的话，dp过程要倒序。
	int n=num1.size(),m=num2.size();
    int ans=0;
    for(int a=1;a<=n;a++){
        for(int b=1;b<=m;b++){
            if(num1[a-1]==num2[b-1]){
                dp[a][b]=dp[a-1][b-1]+1;//如果相同，接在后面，相当于已经把向前逐一判断的过程存了下来，现在直接用就可以
                ans=max(ans,dp[a][b]);
            }
            else dp[a][b]=0;
        }
    }
    return ans;
```

```c++
	int ans=0;
    for(int a=1;a<=n;a++){
        int now=0;
        int l=min(n-a+1,m);
        for(int b=1;b<=l;b++){
            if(num1[a+b-2]==num2[b-1])now++;
            //直接将num1的第a个数据和num2的第1个数据对比，相当于num2向后移动，直接对比就不会将两个数据对比多次
            else now=0;
            ans=max(ans,now);
        }
    }
    for(int a=1;a<=m;a++){//反向将num1向后移动
        int now=0;
        int l=min(m-a+1,n);
        for(int b=1;b<=l;b++){
            if(num2[a+b-2]==num1[b-1])now++;
            else now=0;
            ans=max(ans,now);
        }
    }
    return ans;
```

```c++

```

LCIS

`dp[i][j]`代表以b~j~作为结尾的，选取a中前i个元素和b中前j个元素的最长长度

```c++
if (a[i] != b[j]) dp[i][j] = dp[i - 1][j]
else dp[i][j] = dp[i - 1][k] (满足0<k<j,b[k] < b[j] = a[i]) 提前维护这部分值
```

```c++
#include<bits/stdc++.h>
using namespace std;
#define endl '\n'
int arr[3005], brr[3005];
int n, m, N = 1, mod = 1e9+7;
int dp[3005][3005];

void work(){
    int ans = 0;
    cin >> n;
    for (int i = 1; i <= n; i++) cin >> arr[i];
    for (int i = 1; i <= n; i++) cin >> brr[i];
    for (int i = 1; i <= n; i++) {
        int val = 0; //过程中维护dp[i - 1][k]
        for (int j = 1; j <= n; j++) {
            if (arr[i] != brr[j]) dp[i][j] = dp[i - 1][j];
            else dp[i][j] = val + 1;
            if (brr[j] < arr[i]) val = max(val, dp[i - 1][j]);
            ans = max(ans, dp[i][j]);
        }
    }
    cout << ans << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin >> N;
    while (N--) {
        work();
    }
}
```



#### 多维动态规划

[[P1880  石子合并](https://www.luogu.com.cn/problem/P1880)] 

```c++
	int dp[105][105];\\dp[i][j]指的是将i到j之间的石子合并需要的最小花费
    void solve(int l,int r){
        int now=inf;
        for(int i=l;i<r;i++){
        	int tem=dp[l][i]+dp[i+1][r]+S[r]-S[l-1];//S数组是前缀和
        	now=min(now,tem);
        }
        dp[l][r]=now;
    }

	for(int a=2;a<=n;a++){
        for(int b=1;b+a-1<=n;b++){
            solve(b,b+a-1);
        }
    }
```

上述代码是O(n^3^)的复杂度，由于是一个环，所以我们可以通过==剪环为链==的方法来解决。

方法一：列举环断开的位置（这个位置就是一直没有进行合并的地方，可以通过将数组整体后移来模拟），复杂度变为O(n^4^)。

方法二：==倍增==数组长度，相当于把每个位置作为断点的情况都包括了，复杂度变为O((2n)^3^)。

[[P1387 最大正方形 ](https://www.luogu.com.cn/problem/P1387)]

```c++
if(arr[i][j]){
	dp[i][j]=min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1])+1;
}
```



### 背包dp

对于背包问题，如果开始时dp数组全为0，$dp[i][j]$代表前i个物品容量不超过j的最大价值，，如果开始时将除了$dp[0][0]$以外的数初始化为负无穷，此时$dp[i][j]$代表前i个物品容量恰好为j的最大价值

可以选价值和重量中较小的作为第二维度

#### 01背包

```c++
//滚动数组
int v[maxn], w[maxn];
int dp[maxn];
memset(dp, 0xc0, sizeof(dp)); dp[0] = 0;
for (int i = 1; i <= n; i++) {
    for (int j = m; j >= 1; j++) {
        if (j >= w[j]) dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
    }
}
```

#### 完全背包

```c++
//滚动数组
int v[maxn], w[maxn];
int dp[maxn];
memset(dp, 0xc0, sizeof(dp)); dp[0] = 0;
for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= m; j++) { //正序
        if (j >= w[j]) dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
    }
}
```

#### 多重背包

方法一：当一个物品可以取n次时，将它看作为n个物品

方法二：三重循环，多了一重循环来判断对某个物品取的次数

##### 二进制优化

对上述方法一的优化，将一个物品按照1，2，4等来分组，例如将18分为1，2，4，8和3（多出来的部分为3）

$O(n*m*log(k_{max}))$

```c++
int w[maxn], v[maxn];
for (int i = 1; i <= n; i++) {
    cin >> x >> y >> k;
    for (int j = 0; j <= 30; j++) {
        int num = 1LL << j;
        if (num > k) break;
        v[++cnt] = num * x;
        w[cnt] = num * y;
        k -= num;
    }
    if (k) {
        v[++cnt] = k * x;
        w[cnt] = k * y;
    }
}
```

##### 单调队列优化？

#### 二维费用背包

一个物品会消耗两种价值，再增加一个维度就可以了

[P1855 榨取kkksc03 - 洛谷](https://www.luogu.com.cn/problem/P1855):有n个任务需要完成，完成第i个任务需要花费$t_i$分钟，产生$c_i$​元的开支。现在有T分钟时间，W元钱来处理这些任务，求最多能完成多少任务。

```c#
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
int dp[205][205];
int n,m,N=1;

void work(){
    memset(dp, 0xc0, sizeof(dp));
    dp[0][0] = 0;
    int t, x, y, ans = 0;
    cin >> n >> m >> t;
    for (int i = 1; i <= n; i++) {
        cin >> x >> y;
        for (int j = m; j >= x; j--) {
            for (int k = t; k >= y; k--) {
                dp[j][k] = max(dp[j][k], dp[j - x][k - y] + 1);
                ans = max(ans, dp[j][k]);
            }
        }
    }
    cout << ans << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    // cin >> N;
    while (N--) {
        work();
    }
}
```

#### 分组背包

在同一组的物品最多只能选择一个。

解法：跟多重背包方法二类似，三重循环，第三重判断在一组中选哪个物品

#### 有依赖的背包

[P1064 NOIP 2006 提高组\ 金明的预算方案 - 洛谷](https://www.luogu.com.cn/problem/P1064): 一个主件会携带几个附件

因为每个主件的附件个数不超过2个，所以可以暴力分组，将主件加不同的附件当作不同的物品放在一组，使用分组背包解决

[[10. 有依赖的背包问题 - AcWing题库](https://www.acwing.com/problem/content/description/10/)]:每个物品都有一个依赖，形成一棵树

树上DP，先处理子节点，然后父节点从子节点选取

```c++
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
int n,m,N=1;

int dp[105][105];//dp[i][j]代表以i为根的子树使用了容量j时最大值
int v[105], w[105];

vector<int> vec[105];

void dfs(int now) {
    for (int i = w[now]; i <= m; i++) dp[now][i] = v[now];
    for (auto x : vec[now]) {
        dfs(x);
        for (int i = m; i >= w[now]; i--) { //相当于分组背包
            for (int j = w[x]; j <= i - w[now]; j++) { //在该子树中选取一个
                dp[now][i] = max(dp[now][i], dp[x][j] + dp[now][i - j]);
            }
        }
    }
}

void work(){
    memset(dp, 0xc0, sizeof(dp));
    cin >> n >> m;
    for (int i = 0; i <= n; i++) dp[i][0] = 0;
    int rt = 0, x;
    for (int i = 1; i <= n; i++) {
        cin >> w[i] >> v[i];
        cin >> x;
        if (x == -1) rt = i;
        else vec[x].push_back(i);
    }
    dfs(rt);
    int ans = 0;
    for (int i = 1; i <= m; i++) ans = max(ans, dp[rt][i]);
    cout << ans << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    // cin >> N;
    while (N--) {
        work();
    }
}
```

#### 泛化物品的背包

[P1417 烹调方案 - 洛谷](https://www.luogu.com.cn/problem/P1417) 每个物品的价值随着在背包中的位置呈一次函数关系，

给定n个食材，每个食材有三个属性：$a_i、 b_i $ 和 $c_i $。如果在时间 t 完成第 i 个食材的烹饪，获得的美味值为 $ a_i - t \times b_i $。烹饪该食材需要花费 $c_i$​ 的时间。求T时间内的最大美味值。

现在考虑相邻的两个物品x,y。假设现在已经耗费p的时间，那么分别列出先做x,y的代价：

a[x]-(p+c[x])\*b[x]+a[y]-(p+c[x]+c[y])*b[y]    ①

a[y]-(p+c[y])\*b[y]+a[x]-(p+c[y]+c[x])*b[x]    ②

得到 ① >  ② 的条件为b[x] \* c[y] > b[y] * c[x]

按照该顺序排序后进行01背包



#### 变种

##### 求具体方案

求最优方案中具有最小字典序的具体方案，将物品倒序选择

```c++
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
int v[1005], w[1005];
bool g[1005][1005];
int dp[1005][1005];
int n,m,N=1;

void work(){
    cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        cin >> w[n - i + 1] >> v[n - i + 1];
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            dp[i][j] = dp[i - 1][j];
            if (j >= w[i]) {
                if (dp[i - 1][j - w[i]] + v[i] >= dp[i][j]) {
                    dp[i][j] = dp[i - 1][j - w[i]] + v[i];
                    g[i][j] = 1;
                }
            }
        }
    }
    int now = m;
    for (int i = n; i >= 1; i--) {
        if (g[i][now]) {
            cout << n - i + 1 << ' ';
            now -= w[i];
        }
    }
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    // cin >> N;
    while (N--) {
        work();
    }
}
```

##### 求方案数

```c++
dp[0] = 1;
for (int i = 1; i <= n; i++) 
    for (int j = m; j >= w[i]; j--)
        	dp[j] = (dp[j - w[i]] + dp[j] ) % mod;
```

##### 求最优方案数

```c++
memset(dp, 0xc0, sizeof(dp));
dp[0] = 0; num[0] = 1;
for (int i = 1; i <= n; i++) {
    for (int j = m; j >= w[i]; j--) {
        if (dp[j] < dp[j - w[i]] + v[i]) {
            num[j] = num[j - w[i]];
            dp[j] = dp[j - w[i]] + v[i];
        } else if (dp[j] == dp[j - w[i]] + v[i]) {
            num[j] += num[j - w[i]];
            num[j] %= mod;
        }
    }
}
int ans = 0, maxx = 0;
for (int i = 0; i <= m; i++) {
    if (dp[i] > maxx) maxx = dp[i], ans = num[i];
    else if (dp[i] == maxx) ans += num[i];
}
cout << ans << endl;
```

##### 背包的第k优解

每个状态都存前k优解，转移时使用双指针来从选与不选两种情况中选取尽量大的前k解

[Problem - 2639](https://acm.hdu.edu.cn/showproblem.php?pid=2639)

```c++
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
int v[1005], w[1005];
int dp[1005][40];
int n, m, k, N=1;

void work(){
    cin >> n >> m >> k;
    for (int i = 1; i <= n; i++) {
        cin >> v[i];
    }
    for (int i = 1; i <= n; i++) {
        cin >> w[i];
    }
    int temp1[40], temp2[40];
    memset(dp ,0 ,sizeof(dp));
    for (int i = 1; i <= n; i++) {
        for (int j = m; j >= w[i]; j--) {
            for (int t = 1; t <= k; t++){
                temp1[t] = dp[j][t], temp2[t] = dp[j - w[i]][t] + v[i];
            }
            temp1[k + 1] = temp2[k + 1] = -1;
            int l = 1, r = 1, now = 1;
            while (now <= k && (l <= k || r <= k)) { //双指针选取
                if (temp1[l] > temp2[r]) {
                    dp[j][now] = temp1[l++];
                } else dp[j][now] = temp2[r++];
                if (dp[j][now] != dp[j][now - 1]) now++; //不重复的第k优解
            }
        }
    }
    cout << dp[m][k] << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    cin >> N;
    while (N--) {
        work();
    }
}
```



#### 剩余背包问题？

#### 题目

[[P1156 垃圾陷阱 ](https://www.luogu.com.cn/problem/P1156)] 对于每次拿取有一定要求（拿完之后的价值要大于某值）

将物品按照需要的价值要求升序排序

[[Problem - 3466 (hdu.edu.cn)](https://acm.hdu.edu.cn/showproblem.php?pid=3466)]要求剩余的钱大于某值

假设第i个物品的价格为w~i~,需要剩余的钱大于p~i~才可以买

i和j需要交换（初始时i在前）的条件是如果不交换，不能同时买两件物品，换了之后可以买两件物品，当手里的钱是sum时，有sum-w~i~<p~j~，sum-w~j~>=p~i~，即p~j~+w~i~>sum>=p~i~+w~j~，推出p~j~-w~j~>p~i~-w~i~，所以按照p-w的降序排序

[[E - Knapsack 2](https://atcoder.jp/contests/dp/tasks/dp_e)] 体积的范围到了1e9，但是价值的范围并不大,dp[i]\[j]表示前i个物品拿到价值为j的最小体积

### 树形dp

#### 换根dp

通过将根转移给子节点而将O（n²）变为O（n)。

例题：

[[ABC348-E](https://atcoder.jp/contests/abc348/tasks/abc348_e)]

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int maxn=1e5+5;

vector <int> vec[maxn];
int deep[maxn]={0};
int vis[maxn]={0};
int arr[maxn];
int x=0,sum=0;

void dfs(int now,int fa,int d){
vis[now]+=arr[now];
for(int a=0;a<vec[now].size();a++){
int t=vec[now][a];
if(t==fa)continue;
deep[t]=d;
dfs(t,now,d+1);
vis[now]+=vis[t];
}
}

void ddfs(int v,int now,int fa){
for(int a=0;a<vec[now].size();a++){
int t=vec[now][a];
if(t==fa)continue;
int vv=v-vis[t]+sum-vis[t];
x=min(vv,x);
ddfs(vv,t,now);
}
}

signed main(){
int n;cin>>n;int u,v;
for(int a=1;a<n;a++){
cin>>u>>v;
vec[u].push_back(v);
vec[v].push_back(u);
}
for(int a=1;a<=n;a++){cin>>arr[a];sum+=arr[a];}
dfs(1,0,1);
for(int a=1;a<=n;a++){
x+=deep[a]*arr[a];
}
ddfs(x,1,0);
cout<<x;
}
```



### 区间DP

### DAG上的DP

DAG上的DP一般解决最值问题，对于多条路径的累计问题会导致重复计算（如下面的子集和DP）

[[UVA437 巴比伦塔 The Tower of Babylon - 洛谷](https://www.luogu.com.cn/problem/UVA437)]

### 概率DP

#### DP求概率

#### DP求期望

[2096 -- Collecting Bugs](http://poj.org/problem?id=2096)

`dp[i][j]`表示当前找到了i种 bug 分类，j个子系统的 bug，还需要多少天才能达到目标



### 子集和DP

$O(n*2^n)$

```c++
//子集向超集传递信息，相当于每一维度长度都为2的高维前缀和，第一个循环是选取维度，第二个循环是该维度上为0的点向该维度为1的点传递
for (int i = 0; i < n; i++) {
	for (int j = 0; j < (1 << n); j++) {
        if (j & (1 << i)) dp[j] += dp[j ^ (1 << i)];
    }
}

//超集向子集传递信息，相当于高维后缀和
for (int i = 0; i < n; i++) {
	for (int j = 0; j < (1 << n); j++) {
        if (!(j & (1 << i))) dp[j] += dp[j ^ (1 << i)];
    }
}
//对于该dp，应该需要保证每个传递都不会被阻塞，例如如果某些数字不向下传信息的话就不能使用，比如说1实质上可以通过4和3传递到7，但是在这个DP过程中只会通过3传递，导致走4的这条正确的路就被忽略了
```

[[Problem - 165E - Codeforces](https://codeforces.com/problemset/problem/165/E)]:对数组中每一个元素，判断数组中是否存在一个元素与其异或为0

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn = 5e6+10;
const int mod = 1e9+7;
int num[maxn],arr[maxn];
int n,m,N=1;

void work(){
    cin >> n;
    memset(num, -1, sizeof(num));
    int max_num = (1LL << 22) - 1;
    for (int i = 1; i <= n; i++) {
        cin >> arr[i];
        num[max_num&(~arr[i])]=arr[i];
    }
    for (int i = 0; i < 22; i++) {
        for (int  j = 0; j < max_num; j++) {
            if(!(j & (1LL << i))) num[j] = max(num[j], num[j ^ (1LL << i)]);
        }
    }
    for (int i = 1; i <= n; i++) cout << num[arr[i]] << ' ';
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

## 字符串

### 字符串基础

```c++
scanf,getchar, printf,cin.getline()不能关闭同步流
```

fegts???

#### C标准库

##### 输入输出

```c++
printf("%s", s); scanf("%s", &s);
cin.get(10);//读入9个字符
cin.getline(ch,10,'f');//输入9个字符进去，到f停止，f的位置为乱码，结束条件只能为字符
```

##### strlen (长度)

```c++
strlen(const char *s);//从s[0]到'\0'之前的字符数，O(n)复杂度
```

##### strcmp （比较）

```c++
strcmp(const char *s1, const char *s2);//按照字典序比较，一样为0，s1字典序大返回正值，s2字典序大返回负值。（返回的正负值的绝对值不一定为1）
```

##### strcat （拼接）

```c++
strcat(char *s1, const char *s2);//s2接到s1的末尾，返回s1
```

##### strcpy （复制）

```c++
strcpy(char *str, const char *src);//src复制给str
```

##### strstr （寻找子串）

```c++
strstr(char *s1, const char *s2);//若s2是s1的子串，返回s2在s1中首次出现的地址，否则返回NULL
```

##### strchr （寻找字符）

```c++
strchr(char *s1, int c);//返回c在s1中首次出现的地址，未出现返回NULL
```

##### strrchr （寻找字符）

```c++
strrchr(char *s1, int c);//返回c在s1中最后一次出现的地址，未出现返回NULL
```

#### C++标准库

##### 字符串初始化

```c++
string s;//空字符串
string s(s1);//字符串初始化为s1
string s(5,'1');//s="11111"
```

##### 重载符号

```c++
/*
    '+' : 字符串拼接
    '>'以及'<' : 字典序比较
    复杂度O（n）
```

##### 输入

```c++
getline(cin,s);//读取一整行，可以包括空格，遇到换行符停止，用一个getchar()吃掉换行符；
getchar()==cin.get();
```

##### STL函数

```c++
s.size()==s.length();//长度
s.push_back(c);s.pop_back();
s.insert(t,s0);//在s[t]处插入s0
s.insert(it,c);//在迭代器it处插入字符c
s.append(c)==s.insert(s.end(),c);

s.erase(it1,it2);//删除迭代器it1到迭代器it2之前的元素
s.erase(pos,len);//从s[pos]开始删除len个元素
s.substr(2,3)//从s[2]开始截取3个字符
reverse(s.begin(), s.end());//反转

s.find(s0,pos);//从下标pos开始正序查找s0第一次出现的位置，没有返回-1或者inf
s.rfind(s0,pos);//从下标pos开始倒序查找s0第一次出现的位置，没有返回-1或者inf
s.find(c,pos);//从下标pos开始正序查找c第一次出现的位置，没有返回-1或者inf
s.rfind(c,pos);//从下标pos开始倒序查找c第一次出现的位置，没有返回-1或者inf

stoi(s);//返回int
stod(s);//返回double
to_string(n);//返回string
s.data()/s.c_str();//返回字符数组
```

### 字符串多项式哈希

```c++
const int mod = 2e8 + 33, base = 13331;
int pw[maxn], hs[maxn];
void init_pw(int w) {
    pw[0] = 1;
    for (int i = 1; i <= w; i++) {
        pw[i] = pw[i - 1] * base % mod;
    }
}

void init_hs(string s) {
    init_pw(s.length());
    //注意这里的s已经在前面加了一个" ",以1为起点
    for (int i = 1; i < s.length(); i++) {
        hs[i] = (hs[i - 1] * base + s[i] ) % mod;
    }
}

int get_hs(int l, int r) {
    int res = ((hs[r] - hs[l - 1] * pw[r - l + 1] ) % mod + mod ) % mod;
    return res;
}

int get_hs_val(string s){
    int res = 0;
    for (int i = 0; i < s.length(); i++) {
        res = (res * base  + s[i] )% mod;
    }
    return res;
}
```

使用双值哈希，对两个不同的质数进行哈希，降低碰撞率

#### 如何构造字符串来卡哈希？？？

[[字符串哈希 - OI Wiki](https://oi-wiki.org/string/hash/#卡大模数-hash)]:

#### 应用

##### 字符串匹配

求出模式串的哈希值后，求出文本串每个长度为模式串长度的子串的哈希值，分别与模式串的哈希值比较即可。

##### 允许k次失配的字符串匹配问题

先将模式串和原串的哈希前缀和求出来，然后对原串中每一个和模式串等长的字串与模式串进行二分比较，找出不同的字符，判断个数是否不超过k次

复杂度$O(n*k*log_m)$

[P3763 TJOI2017 DNA - 洛谷](https://www.luogu.com.cn/problem/P3763)

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn = 1e5+10;
const int mod = 2e8 + 33, base = 13331;
int pw[maxn],hs1[maxn],hs0[maxn];
int n,m,N=1;

void init_pw(int w){
    pw[0] = 1;
    for (int i = 1; i <= w; i++) {
        pw[i] = (pw[i-1] * base ) % mod;
    }
}

void init_hs(string s, int hs[]) {
    for (int i = 1; i < s.length(); i++) {
        hs[i] = (hs[i - 1] * base + s[i]) % mod;
    }
}

int get_hs(int l, int r, int hs[]) {
    return ((hs[r] - hs[l - 1] * pw[r - l + 1]) % mod + mod) %mod;
}

int solve (int l0, int r0, int l1, int r1) {
    if (l0 == r0) return l0;
    int mid0 = (l0 + r0) / 2, mid1 = mid0 - l0 + l1;
    if (get_hs(l0, mid0, hs0) == get_hs(l1, mid1, hs1)) return solve(mid0 + 1, r0, mid1 + 1, r1);
    else return solve(l0, mid0, l1, mid1);
}

void work(){
    string s1,s0;
    cin >> s0 >> s1;
    s1 = ' ' + s1;
    s0 = ' ' + s0;
    init_hs (s0, hs0);
    init_hs (s1, hs1);
    int ans = 0;
    for (int i = 1; i + s1.length() - 2 < s0.length(); i++) {
        int l0 = i, r0 = l0 + s1.length() -2, l1 = 1, r1 = s1.length() - 1;
        if (get_hs (l0, r0, hs0) == get_hs(l1, r1, hs1)) {
            ans++; continue;
        }
        for (int j = 1; j <= 3; j++) { //三次二分
            if (l0 <= r0 && get_hs (l0, r0, hs0) != get_hs(l1, r1, hs1)) {
                int t = solve(l0, r0, l1, r1) - l0 + 1;
                l1 += t; l0 += t;
            }
            if (l0 > r0 || get_hs (l0, r0, hs0) == get_hs(l1, r1, hs1)) {
                ans++; break;
            }
        }
    }
    cout << ans << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    init_pw (100000);
    cin>>N;
    while (N--) {
        work();
    }
}
```

##### 最长回文子串

$O(nlogn)$：预处理哈希前缀和，后缀和，然后枚举对称中心（可能是两个点），对每一次枚举采用二分来判断该点作为对称中心的最大回文子串长度

##### 最长公共子串

对m个总长不超过n的字符串确定一个最长的公共子字符串，复杂度为$O(n*logn)$（？？？），这里的n指的是总长

如果最长公共子字符串的长度为k，那么必然存在长度为k-1的公共子字符串，所以采用二分的方法

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define ull unsigned long long
#define pii pair<int,int>
#define piii pair<pii,int>
#define F first
#define S second
#define endl '\n'
#define inf 0x3f3f3f3f3f3f3f3f

const int maxn = 1e6+10;
const int mod = 100000007, base = 13331;
int pw[maxn];
vector<vector<int>> hs;
vector<int> vec;
int n,m,N=1;

void init_pw(int w){
    pw[0] = 1;
    for (int i = 1; i <= w; i++) {
        pw[i] = (pw[i-1] * base ) % mod;
    }
}

void init_hs(string s) {
    vec.clear(); vec.push_back(0);
    for (int i = 1; i < s.length(); i++) {
        vec.push_back( (vec.back() * base + s[i]) % mod );
    }
    hs.push_back(vec);
}

int get_hs(int l, int r, int i) {
    return ((hs[i][r] - hs[i][l - 1] * pw[r - l + 1]) % mod + mod) %mod;
}

string s[maxn];

vector<int> hd;
int sum[mod];

bool check(int t) {
    for (auto x : hd) sum[x] = 0; hd.clear();
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j + t - 1 < s[i].length(); j++) {
            int now = get_hs(j, j + t - 1, i);
            if (sum[now] == i - 1) {
                sum[now] = i;
                if (i == 0) hd.push_back(now);
            }
            if (sum[now] == m) return true;
        }
    }
    return false;
}

int solve(int l, int r) {
    if (l == r) return l;
    int mid = (l + r) >> 1;
    if (check(mid + 1))return solve(mid + 1, r);
    else return solve(l, mid);
}

void work(){
    cin >> n >> m;
    int minx = maxn;
    for (int i = 1; i <= m; i++) {
        cin >> s[i];
        minx = min (minx, (int)s[i].length());
        s[i] = ' ' + s[i];
    }
    hs.clear(); hs.push_back(vec);
    for (int i = 1; i <= m; i++) {
        init_hs(s[i]);
    }
    int ans = solve(0 ,minx);
    cout << ans << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    init_pw (1000000);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

##### 确定字符串中不同子字符串的数量???

$O(n^2logn)$,对所有子串进行哈希并返回总数量

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
#define inf 0x3f3f3f3f3f3f3f3f

const int maxn = 1e5+10;
const int mod = 100000007, base = 13331;
int pw[maxn], hs[maxn];
int n,m,N=1;

void init_pw(int w){
    pw[0] = 1;
    for (int i = 1; i <= w; i++) {
        pw[i] = (pw[i-1] * base ) % mod;
    }
}

void init_hs(string s) {
    for (int i = 1; i < s.length(); i++) {
        hs[i] = (hs[i - 1] * base + s[i] ) % mod;
    }
}

int get_hs(int l, int r) {
    return ((hs[r] - hs[l - 1] * pw[r - l + 1]) % mod + mod) %mod;
}

void work(){
    cin >> n;
    string s; cin >> s;
    s = ' ' + s;
    init_hs(s);
    unordered_set <int> st;
    for (int i = 1; i < s.length(); i++) {
        for (int j = 1; j + i - 1 < s.length(); j++){
            st.insert(get_hs(j, i + j - 1));
        }
    }
    cout << st.size() << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    init_pw (100000);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

#### 题目

##### 子串字典序比较

对于一个字符串，三个操作

1.询问两个区间字典序大小？

2.单点修改，将a[x]改成y

3.区间修改，将a[l...r]上加v

对于询问， 采用二分来找到两者第一个不同的位置判断即可。

两种修改可以采用线段树维护，sum[l...r] = base^r-mid^*sum[l,mid] + sum[mid+1...r]，单点修改直接修改，区间修改则提前维护base^i^的前缀和

### trie树（字典树）？？？

字典指的是字符串的集合

构建树O(n)  查找的字符串长度为k时查找的复杂度为O(k)

```c++
struct Trie {
  int nex[maxn][26], cnt=0;
  bool exist[maxn];  // 该结点结尾的字符串是否存在

  void insert(string s) {  // 插入字符串
    int p = 0, len = s.length();
    for (int i = 0; i < len; i++) {
      int t = s[i] - 'a';
      if (!nex[p][t]) nex[p][t] = ++cnt;  // 如果没有，就添加结点
      p = nex[p][t];
    }
    exist[p] = true;
  }

  bool find(string s) {  // 查找字符串
    int p = 0, len = s.length();
    for (int i = 0; i < len; i++) {
      int t = s[i] - 'a';
      if (!nex[p][t]) return 0;
      p = nex[p][t];
    }
    return exist[p];
  }
}trie;
```

#### 01trie解决异或极值

[[P10471 最大异或对 The XOR Largest Pair - 洛谷](https://www.luogu.com.cn/problem/P10471)]

[[P4551 最长异或路径 - 洛谷](https://www.luogu.com.cn/problem/P4551)]:

#### 维护异或和？？？

#### 合并01trie???

#### 可持久化trie树

```c++
struct node{
    int son[2], num;
}tr[maxn*30];
int rt[maxn];//多个版本的根
int cnt;

int insert(int pre, int t){
    tr[++cnt] = tr[pre];//建立新的根节点
    int res = cnt;
    int now = cnt;//now代表当前节点
    for (int i = 25; i >= 0; i--) {
        int to = (t&(1<<i)) ? 1 : 0;
        tr[++cnt] = tr[tr[now].son[to]];//创建一个新的节点并复制需要修改的点
        tr[now].son[to] = cnt;//指向新的节点
        now = tr[now].son[to];//将当前节点转移到新的节点
        tr[now].num ++;
    }
    return res;
}

rt[0]=insert(0,0);
rt[i]=insert(rt[i-1],arr[i]);
```

#### 题目

[[P2536 AHOI2005 病毒检测 - 洛谷](https://www.luogu.com.cn/problem/P2536)]:给定一个模式串由 `A`、`C`、`T`、`G` 的序列加上通配符 `*` 和 `?` 来表示，其中 `*` 的意思是可以匹配上 0 个或任意多个字符，而 `?` 的意思是匹配上任意一个字母，多次给出一个由 `A`、`C`、`T`、`G` 组成的序列，问能否跟模式串完全匹配（不多不少）

解法：`trie`树上记忆化`dfs`，`dfs(i,j)`代表模式串的前i个字符匹配了询问串的前j个字符，`*`代表可以向`dfs(i + 1, j)和dfs(i, j + 1)`转移，其余正常转移即可。

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define ull unsigned long long
#define pii pair<int,int>
#define piii pair<pii,int>
#define F first
#define S second
#define endl '\n'
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn = 2e6+10;
const int mod = 1e9+7;
int arr[maxn];
int n,m,N=1;

int tr[1005][5];
bool vis[1005][505];

int get_num(char c) {
    switch (c) {
        case 'A' : return 0;
        case 'C' : return 1;
        case 'T' : return 2;
        case 'G' : return 3;
    }
}

void insert (string s) {
    for (int i = 0; i < s.length(); i++) {
        if (s[i] == '*') tr[i][4] = 1;
        else if (s[i] == '?') {
            for (int j = 0; j < 4; j++) tr[i][j] = 1;
        } else tr[i][get_num(s[i])] = 1;
    }
}

string s;
int len, cnt;//这里的cnt用来记录最后有多少个连续的*,dfs过程使用

bool dfs (int d, int stp) {
    if (vis[d][stp]) return 0;
    vis[d][stp] = 1;
    if (stp == s.length() || d == len) {
        if (stp == s.length() && d <= len && d >= len - cnt) return 1;
        return 0;
    }
    bool res = 0;
    if (tr[d][get_num(s[stp])]) res |= dfs(d + 1, stp + 1);
    if (tr[d][4]) res |= dfs(d, stp + 1) | dfs(d + 1, stp);
    return res;
}

void work(){
    cin >> s;
    len = s.length();
    for (int i = len - 1; i >= 0; i--) {
        if (s[i] == '*') cnt++;
        else break;
    }
    insert (s);
    cin >> n;
    int ans = n;
    while (n--) {
        cin >> s;
        for (int i = 0 ; i <= len; i++) {
            for (int j = 0; j <= s.length(); j++) {
                vis[i][j] = 0;
            }
        }
        if (dfs(0, 0)) ans--;
    }
    cout << ans <<endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```



### KMP算法

前缀函数：$Π[i]$表示字串$s[1$~$i]$​中有多对相等的真前缀和真后缀，取最长的一对的长度，没有就取0.

性质1：$Π[i+1] <= Π[i] + 1$ 

```c++
//解释：
int nxt[maxn];//next[i]=j说明t[1,1,……,j]=t[i-j+1,……,i-1,i]，且不存在更大的j符合该要求
void get_nxt(string t)		//这个函数对字符串t进行预处理得到nxt数组,被称为前缀函数
{
	int j = 0; //前面的最长border
	nxt[1] = 0;	//不存在真子串，初始化为0
	for(int i = 2; i < t.length(); i++) {	//i指针指向的是后缀末尾，j指针指向的是前缀末尾
		while(j > 0 && t[i] != t[j + 1]) j = nxt[j];	//前后缀不相同，去找j前一位的最长相等前后缀，s的border的border也是s的border,具有传递性
		if(t[i] == t[j + 1]) j++;	//前后缀相同，j指针后移,next[i]的最大值为next[i-1]+1
		nxt[i] = j;	//更新next数组
	}
}
//关于while(j>0&&t[i]!=t[j + 1])j = next[j]这一步,当t[i]!=t[j]，首先是找到一个尽量大的k（必然小于j）使t[1,1,……,k]==t[i-k,……,i-2,i-1],又因为next[i-1]==j,有t[1,2,……,j]==t[i-j,i-j+1,……,i-1],所以要找的就是最大的k使t[0,1,……,k-1]==t[j-k,……,j-2,j-1],也就是next[j-1]

int strSTR(string s, string t)	//这个函数是从s中找到t，如果存在返回t出现的位置，如果不存在返回-1
//s和t下标都是从1开始
{	
	if(t.length() == 1)	return 0;
	get_nxt(t, nxt);
	int j = 0;
	for (int i = 1; i < s.length(); i++) {
		while(j>0 && s[i] != t[j + 1]) j = nxt[j];
		if(s[i] == t[j + 1]) j++;
		if(j == t.length() - 1)	return i - t.length() + 2;
	}
	return -1;
}
//令str=t+'#'+s;以上两个函数就是对于str进行get_next函数的操作，中间的分隔符是s和t中没出现过的字符，当nxt[i]等于t.length()时说明找到一个位置
```

```c++
//板子：
int nxt[maxn];
void get_nxt(string t) {
	int j = 0;
	nxt[0] = 0;
	for (int i = 1; i < t.length(); i++) {
		while (j > 0 && t[i] != t[j]) j = nxt[j-1];
		if(t[i] == t[j]) j++;
		nxt[i] = j;
	}
}

int strSTR(string s, string t) {
	if(t.length() == 0)	return 0;
	get_nxt(t, nxt);
	int j = 0;
	for (int i = 0; i < s.length(); i++) {
		while (j>0&&s[i] != t[j]) j = nxt[j-1];
		if(s[i] == t[j]) j++;
		if(j == t.length())	return i - t.length() + 1;
	}
	return -1;
}
```



#### 字符串的周期

$O(n)$

对于字符串s，若存在整数p符合0<p<=$|s|$,对于1<=i<=$|s|$-p，都有s[i]=s[i+p]，则称p是s的周期。如果p|$|s|$，则称p为循环节

对于字符串s和1<=r<$|s|$,若s的长度为r的前缀以及长度为r的后缀相同，即s[1,2,...,r] = s[$|s|$-r+1,$|s|$-r+2,...,$|s|$]，称该前缀为s的border

s有长度为r的border 等价于 $|s|$​-r是s的周期

若p,q都是周期，则gcd(p,q)也是周期

根据前缀函数的定义可以得到s的所有border长度为$Π[n]$,$Π[Π[n]]$​​,...。

如果求的是一个矩阵的最小重复单元，将每行哈希成一个数字，求出列的最小周期，再将列哈希成数字，求行的最小周期，将两者相乘就是答案

[P10475 USACO03FALL Milking Grid（数据加强版） - 洛谷](https://www.luogu.com.cn/problem/P10475)

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
int n,m,N=1;

const int p = 13331;
const int mod = 1e18 + 31;
int hash_row[10005],hash_col[100];
char c[10005][100];
int nxt[10005];

int get_nxt(int len,int arr[]){
    int j = 0; nxt[1] = 0;
    for (int i = 2; i <= len; i++) {
        while (j && arr[i] != arr[j+1]) j = nxt[j];
        if (arr[i] == arr[j+1]) j++;
        nxt[i] = j;
    }
    return len - nxt[len];
}

void work(){
    cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            cin >> c[i][j];
            hash_row[i] = ((__int128)hash_row[i] * p + c[i][j])%mod;
            hash_col[j] = ((__int128)hash_col[j] * p + c[i][j])%mod;
        }
    }
    int ans = get_nxt(n, hash_row);//通过行的哈希值来求列的最小周期
    ans *= get_nxt(m, hash_col);
    cout << ans << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

#### 统计每个前缀出现的次数

##### 统计在当前字符串中出现的次数

对于位置i，考虑以i为结尾的后缀和多少个前缀相同，以位置i为右端点，有长度为$Π[i]$的前缀，长度为$Π[Π[i]-1]$的前缀，...。



##### 统计在另一给定字符串中出现的次数

#### border树(失配树)

每个节点i的父亲为节点next[i]

性质1：每个前缀prefix[i]的所有border为节点i到根节点的链

性质2：哪些前缀具有长度为i的border：节点i的子树

性质3：求解两个前缀的公共border等价于求LCA

[[P5829 【模板】失配树 - 洛谷](https://www.luogu.com.cn/problem/P5829)]: 给出任意两点，求最大公共border

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn = 1e6+10;
int n,m,N=1;

int fa[21][maxn];
int dep[maxn];

void get_nxt (string s) {
    fa[0][1] = 0;
    dep[1] = 1;
    int j = 0;
    for (int i = 2; i < s.length(); i++) {
        while (j && s[i] != s[j + 1]) j = fa[0][j];
        if (s[i] == s[j + 1]) j++;
        fa[0][i] = j;
        dep[i] = dep[j] + 1;
    }
}

int LCA (int x, int y) {
    x = fa[0][x]; y = fa[0][y];
    if (dep[x] > dep[y]) swap(x, y);
    int d = dep[y] - dep[x];
    for (int i = 0; i <= 20; i++) {
        if (d & (1LL << i)) {y = fa[i][y];}
    }
    if (x == y) return x;
    for (int i = 20; i >= 0; i--) {
        if (fa[i][x] != fa[i][y]) {
            x = fa[i][x]; y = fa[i][y];
        }
    }
    return fa[0][x];
}

void work(){
    string s; cin >> s; s = ' ' + s;
    get_nxt(s);
    for (int i = 1; i <= 20; i++) {
        for (int j = 1; j < s.length(); j++) {
            fa[i][j] = fa[i - 1][fa[i - 1][j]];
        }
    }
    cin >> m;
    int x, y;
    while (m--) {
        cin >> x >> y;
        cout << LCA (x, y) << endl;
    }
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```



### 扩展KMP（Z函数）

对于一个长度为n的字符串s，定义函数z[i]表示s和s[i,n-1]的最长公共前缀的长度，z被称为s的Z函数，特别地，z[0]=0;

### 回文串

对于一个回文串，回文前后缀和border相互等价

#### Manacher算法

能够O(n)求出一个字符串中每个中心都能找到多少个回文串,可以用来解决回文字符串总个数,最长回串等问题

[[P3805 【模板】manacher - 洛谷](https://www.luogu.com.cn/problem/P3805)]

朴素算法：枚举每个字符（或两个字符一起）作为回文串对称中心然后向两边扩展，复杂度O(n^2^)

```c++
//解释：
int manacher(string s1){
    string s="#";
    for(int i=0;i<s1.length();i++){
        s+=s1[i];
        s+='#';
    }//将原本两个字符共同作为对称中心的情况扩展为一个"#"作为对称中心，将两种情况合并为一种情况
    int res = 0;//记录答案
    int len = s.length();
    int r = -1;//r代表0~i作为对称中心已经找到的回文串中最靠右的回文串的右边界
    int l = 0;//跟r相对应的左边界
    for(int i=0;i<len;i++){
        int now = (i>r)?0:min(ans[l+r-i],r-i);
        /*如果i>r，采用朴素做法,
        否则的话找到跟i对称的那个字符l+r-i,根据ans[l+r-i]来判断
        如果i+ans[l+r-i]<=r,根据对称的关系,ans[i]也就等于ans[l+r-i]
        如果i+ans[l+r-i]>r,那么以l+r-i为中心找到的对称关系并不全适用于i,我们只能根据对称性判断i-(r-i)~i+(r-i)是对称的,继续使用朴素算法.
        
        对于上述过程,为了代码简洁,用三目运算符进行一次判断,无论哪种情况都可以用下面的朴素算法判断一下,反正对每个i都只会多判断一次,并不影响复杂度
        */
        while(i-now-1>=0&&i+now+1<len&&s[i-now-1]==s[i+now+1]){
            //判断条件为不越界且相等
            now++;
            //每次有意义的now++都可以将r变大,而r并不会变小,所以总的now++次数不超过n,这也是为什么算法复杂度为O(n)
        }
        ans[i]=now;
        res=max(res,ans[i]);
        //如果以'#'作为中心,那么有效值为ans[i]/2*2,此时ans[i]为偶数,有效值就为ans[i]
        //否则有效值为1+(ans[i]/2*2),因为ans[i]此时为奇数,所以有效值仍为ans[i]
        if(i+now>r){//如果找到新的右边界就更新
            r=i+now;
            l=i-now;
        }
    }
    return res;
}
```

```c++
//板子：
int ans[maxn];

int manacher(string s1){
    string s="#";
    for(int i=0;i<s1.length();i++){
        s+=s1[i];s+='#';
    }
    int res = 0, len = s.length();
    int r = -1, l = 0;
    for(int i=0;i<len;i++){
        int now = (i>r)?0:min(ans[l+r-i],r-i);
        while(i-now-1>=0&&i+now+1<len&&s[i-now-1]==s[i+now+1])now++;
        ans[i]=now;
        res=max(res,ans[i]);
        if(i+now>r)r=i+now,l=i-now;
    }
    return res;
}
```

### AC自动机

离线型数据结构，不支持添加新的字符串，(先离线添加，逆序或者使用fail树维护)

广义border：对于串S和一个字典D，S的后缀和任意一个字典串T的前缀相同，称为一个border

失配(fail)指针：对于每一个节点，指向字典中对应的最大border，求的过程仿照KMP，通过其父亲节点的失配指针链来求解，在处理过程中，要按照节点深度升序来求，即$bfs$​

首先将所有模式串插入$trie$树中，然后$bfs$建立$fail$指针，然后在$trie$树遍历文本串s，$s[1,2,...,i]$遍历到节点$j$说明节点$j$所表示的字符串是$s[1,2,...,i]$在$trie$树上能够找到的最长后缀，该节点及其$fail$指针链上的节点都在$s[1,2,...,i]$上作为其后缀出现，对该节点计数加一，表明该节点及其fail指针链上的节点出现次数都加一。最后统一反$bfs$序来转移和统计每个节点的计数。

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn =1e5+10;
int n,m,N=1;

struct node {
    int son[26], fail, count, vis;
}tr[maxn*26];
//vis代表该节点作为结尾的字符串的出现次数，count代表该节点作为文本串最长后缀出现的次数
int cnt;

vector <int> vec;

void insert (string s) {
    int now = 0;
    for (int i = 0; i < s.length(); i++) {
        int to = s[i] - 'a';
        if (tr[now].son[to] == 0) tr[now].son[to] = ++cnt;
        now = tr[now].son[to];
    }
    tr[now].vis++;
}

void build() {
    queue <int> que; //bfs
    for (int i = 0; i < 26; i++) {
        if (tr[0].son[i]) que.push(tr[0].son[i]);
    }
    while (!que.empty()) {
        int now = que.front(); que.pop();
        vec.push_back(now);
        for (int i = 0; i < 26; i++) { //没有遍历fail链，节省时间
            if (tr[now].son[i]) {
                tr[tr[now].son[i]].fail = tr[tr[now].fail].son[i];
                que.push(tr[now].son[i]);
            } else tr[now].son[i] = tr[tr[now].fail].son[i];
        }
    }
}

int query (string s) {
    int now = 0, res = 0;
    for (int i = 0; i < s.length(); i++) {
        now = tr[now].son[s[i] - 'a'];
        tr[now].count++;
    }
    
    for (int i = vec.size() - 1; i >= 0; i--) {
        int x = vec[i];
        tr[tr[x].fail].count += tr[x].count;
        if (tr[x].count) res += tr[x].vis;
    }
    return res;
}

void work(){
    cin >> n;
    for (int i = 1; i <= n; i++) {
        string s; cin >> s;
        insert(s);
    }
    build();
    string s; cin >> s;
    cout << query(s) << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    work();
}
```

#### 题目

[[P2414 NOI2011\ 阿狸的打字机 - 洛谷](https://www.luogu.com.cn/problem/P2414)]:插入多个字符串，多次询问一个字符串在另一个字符串中出现的次数。

用$fail$树+$dfs序$+树状数组，离线处理所有问题，在预处理了fail树之后重新遍历原本的$trie$树，遍历到一个点就给该点加1，离开时减1，走到一个字符串结尾就通过子树和查询把跟其相关的询问解决

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define pii pair<int,int>
#define F first
#define S second
#define endl '\n'
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn = 3e6+10;
const int mod = 1e9+7;
int arr[maxn];
int n,m,N=1;

struct node{
    int son[26],vis,fail,tag;
}trie[maxn],cp[maxn];
int cnt;

int pos[maxn];
vector<int>vec[maxn];
int dfn_cnt, dfn[maxn];

int siz[maxn];

void dfs(int now) {
    dfn[now] = ++dfn_cnt;
    siz[dfn[now]] = 1;
    for (auto x : vec[now]){
        dfs(x);
        siz[dfn[now]] += siz[dfn[x]];
    }
}

void build () {
    queue <int> que;
    for (int i = 0 ; i < 26; i++) {
        if (trie[0].son[i]) {
            que.push(trie[0].son[i]);
            vec[0].push_back(trie[0].son[i]);
        }
    }
    while (!que.empty()) {
        int now = que.front(); que.pop();
        for (int i = 0; i < 26; i++) {
            if (trie[now].son[i]) {
                trie[trie[now].son[i]].fail = trie[trie[now].fail].son[i];
                vec[trie[trie[now].fail].son[i]].push_back(trie[now].son[i]);
                que.push(trie[now].son[i]);
            }
            else trie[now].son[i] = trie[trie[now].fail].son[i];
        }
    }
}
vector <pii> q[maxn];

int ans[maxn];
int tr[maxn];
int lowbit(int i){
    return i&(-i);
}
void change(int i, int v) {
    while (i <= dfn_cnt) {
        tr[i] += v;
        i += lowbit(i);
    }
}
int query(int i) {
    int res = 0;
    while (i) {
        res += tr[i];
        i -= lowbit(i);
    }
    return res;
}


void dfs1(int now) {
    change(dfn[now], 1);
    if (trie[now].tag)
        for (auto x : q[trie[now].tag]) {
            int t = dfn[pos[x.F]];
            ans[x.S] = query(t + siz[t] - 1) - query(t - 1);
        }
    for (int i = 0 ; i < 26; i++) {
        if(cp[now].son[i])dfs1(cp[now].son[i]);
    }
    change(dfn[now], -1);
}

void work(){
    string s; cin >> s;
    int now = 0;
    stack <int> sta;
    int sum = 0;
    for (int i = 0; i < s.length(); i++) {
        if (s[i] == 'B') {
            now = sta.top(); sta.pop();
        } else if (s[i] == 'P') {
            trie[now].vis++;
            ++sum; pos[sum] = now;
            trie[now].tag = sum;
        } else {
            sta.push(now);
            if (trie[now].son[s[i] - 'a'] == 0) {trie[now].son[s[i] - 'a'] = ++cnt;}
            now = trie[now].son[s[i] - 'a'];
        }
    }
    cin >> m;
    for (int i = 0; i <= cnt; i++) cp[i] = trie[i];
    build ();
    dfs(0);
    int x, y;
    for (int i = 1; i <= m; i++) {
        cin >> x >> y;
        q[y].push_back({x,i});
    }
    dfs1(0); //重新遍历最初的trie树
    for (int i = 1; i <= m; i++) cout << ans[i] << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

[[B-string_牛客竞赛字符串专题班AC自动机](https://ac.nowcoder.com/acm/contest/29086/B)] 两种操作，插入一个新的字符串，查询字典中字符串在给定的一个新的字符串中出现的次数（一个字符串可出现多次）

这道题跟上一题都使用$fail$树+$dfs序$+树状数组，不过使用方法不同，上一题用树状数组记录文本串每个节点出现次数，用子树和代表一个字符串在文本串中出现的次数，而这题用树状数组+差分代表该节点是多少个字符串的后缀，即走到该点能造成多大贡献，然后在给定字符串遍历$trie$树时一步步统计每个前缀的贡献，原因在于上一题询问的是文本串和某个串，而这一题询问的是文本串和所有串

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn =1e5+10;
const int mod = 1e9+7;
int pos[maxn * 2];
int n,m,N=1;

struct node {
    int son[26], fail;
}trie[maxn*26];
int cnt;

vector <int> vec[maxn * 26];
int dfn_cnt, dfn[maxn * 26];
int siz[maxn * 26];
void dfs(int now) {
    dfn[now] = ++dfn_cnt;
    siz[dfn[now]] = 1;
    for (auto x : vec[now]){
        dfs(x);
        siz[dfn[now]] += siz[dfn[x]];
    }
}

int tr[maxn * 26]; //差分维护，i的前缀和表示该节点及其后缀可以表示的存在的字符串的个数
int lowbit(int i){
    return i&(-i);
}
void change(int i, int v) {
    while (i <= dfn_cnt) {
        tr[i] += v;
        i += lowbit(i);
    }
}
int query(int i) {
    int res = 0;
    while (i) {
        res += tr[i];
        i -= lowbit(i);
    }
    return res;
}


void insert (string s, int tag) {
    int now = 0;
    for (int i = 0; i < s.length(); i++) {
        int to = s[i] - 'a';
        if (trie[now].son[to] == 0) trie[now].son[to] = ++cnt;
        now = trie[now].son[to];
    }
    pos[tag] = now;
}

void build() {
    queue <int> que;
    for (int i = 0; i < 26; i++) {
        if (trie[0].son[i]) que.push(trie[0].son[i]), vec[0].push_back(trie[0].son[i]);
    }
    while (!que.empty()) {
        int now = que.front(); que.pop();
        for (int i = 0; i < 26; i++) {
            if (trie[now].son[i]) {
                trie[trie[now].son[i]].fail = trie[trie[now].fail].son[i];
                vec[trie[trie[now].fail].son[i]].push_back(trie[now].son[i]);
                que.push(trie[now].son[i]);
            } else trie[now].son[i] = trie[trie[now].fail].son[i];
        }
    }
}

int ask (string s) {
    int now = 0, res = 0;
    for (int i = 0; i < s.length(); i++) {
        now = trie[now].son[s[i] - 'a'];
        res += query (dfn[now]);
    }
    return res;
}

int op[maxn]; string t[maxn];

void init() {
    memset(trie, 0, sizeof (trie[1]) * (cnt + 1));
    memset(tr, 0, sizeof (tr[0]) * (dfn_cnt + 1));
    for (int i = 0; i <= cnt; i++) vec[i].clear();
    cnt = dfn_cnt = 0;
}

void work(){
    cin >> n >> m; string s;
    init();
    for (int i = 1; i <= n; i++) {
        cin >> s; insert (s, i);
    }
    for (int i = 1; i <= m; i++) {
        cin >> op[i] >> t[i];
        if (op[i] == 1) insert(t[i], i + n);
    }
    build();
    dfs(0);
    for (int i = 1; i <= n; i++) { //利用差分维护，对一个结尾的子树全都加1
        change(dfn[pos[i]],1);
        change(dfn[pos[i]] + siz[dfn[pos[i]]], -1);
    }
    for (int i = 1; i <= m; i++) {
        if (op[i] == 1){
            change(dfn[pos[i + n]],1);
            change(dfn[pos[i + n]] + siz[dfn[pos[i + n]]],-1);
        }
        else {
            cout << ask(t[i]) << endl;
        }
    }
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    cin>>N;
    while (N--) {
        work();
    }
}
```

[[P4052 JSOI2007\ 文本生成器](https://www.luogu.com.cn/problem/P4052)]:给定一个有n个字符串的字典，问有多少个长度为m的字符串中存在子串属于给定的字典

计算不存在子串属于字典的字符串的数量，$dp[i][j]$指的是构造出长为$i$的字符串，其落在$trie$树上第$j$个节点上，且不存在子串属于字典的字符串的数量

转移过程中$trie$​​树上每个节点都有26个转移方向，如果转移结束后的节点是一个字符串的后缀，则该转移不合法

时间复杂度为$O(n*m*26)$，其中n是$trie$树节点个数，m是模式串长度

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn = 2e6+10;
const int mod = 1e4+7;
int n,m,N=1;
int dp[105][6005];

int quickpow (int x, int y) {
    x %= mod;
    int res = 1;
    while (y) {
        if (y & 1) res = res * x % mod;
        x = x * x % mod;
        y >>= 1;
    }
    return res;
}

struct node{
    int son[26], fail;
    bool vis;
}tr[maxn];
int cnt;

void insert (string s) {
    int now = 0;
    for (int i = 0; i < s.length(); i++) {
        if (tr[now].son[s[i] - 'A'] == 0) tr[now].son[s[i] - 'A'] = ++cnt;
        now = tr[now].son[s[i] - 'A'];
    }
    tr[now].vis = 1;
}

void build () {
    queue<int> que;
    for (int i = 0; i < 26; i++) {
        if (tr[0].son[i]) {
            que.push(tr[0].son[i]);
        }
    }
    while (!que.empty()) {
        int now = que.front(); que.pop();
        for (int i = 0; i < 26; i++) {
            int x = tr[now].son[i];
            if (x) {
                que.push(x);
                tr[x].fail = tr[tr[now].fail].son[i];
                tr[x].vis |= tr[tr[x].fail].vis; //判断该节点是否是某个字符串的后缀
            } else tr[now].son[i] = tr[tr[now].fail].son[i];
        }
    }
}

void work(){
    cin >> n >> m;
    string s;
    for (int i = 1; i <= n; i++) {
        cin >> s;
        insert(s);
    }
    build();
    int ans = quickpow(26, m);
    dp[0][0] = 1;
    for (int i = 1; i <= m; i++) { //dp过程
        for (int j = 0; j <= cnt; j++) {
            for (int k = 0; k < 26; k++) {
                int to = tr[j].son[k];
                if (tr[to].vis) continue;
                dp[i][to] += dp[i - 1][j];
                dp[i][to] %= mod;
            }
        }
    }
    for (int i = 0; i <= cnt; i++) {
        ans -= dp[m][i];
        ans %= mod;
    }
    ans += mod; ans %= mod;
    cout << ans << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```



### SA???

### 最小循环表示

一个字符串的循环表示即为该字符串的循环同构字符串，例如abc的循环表示为abc,bca,cab，最小循环表示为字典序最小的循环表示

#### 最小表示法

先将字符串倍增，便于操作

i,j是两个目前为止可能为最小循环表示的开头的位置，然后不断更新i,j

在j之前只有i可能是最小循环的开头

```c++
string s;
cin >> s;
n = s.length();
s += s;
int l = 0, r = 1, k = 0;
while (r < n && k < n) {
    if (s[l + k] == s[r + k]) k++;
    else if (s[l + k] < s[r + k]) r += k + 1, k = 0;
    else l = max(r, l + k + 1), r = l + 1, k = 0;
}
cout << k <<endl;
for (int i = 0; i < n; i++) {
    cout << s[i + l];
}
```

###  Lyndon 分解 ？？？

一个字符串s为Lyndon串(也被称为简单串）当且仅当s的字典序严格小于其所有真后缀的字典序，等价于s严格小于所有非平凡循环同构串

Lyndon分解是将一个字符串分为若干个小的Lyndon串，一个字符串可能有多种分解方法

唯一Lyndon分解是指将s分解为小的Lyndon串 $w_i $,且从左到右是非严格单减的($w_1>=w_2>=...>=w_k$​)，可以证明这样的分解存在且唯一，且分解的最后一个子串 $w_k $一定是 s 的最小后缀（在字典序下）。

##### Duval算法 理解？？？

[题解 P6127【模板】Lyndon 分解](https://www.luogu.com.cn/article/lt2rnl6d)

时间复杂度为$O(n)$

```c++
vector<string> duval(string const& s) {
  int n = s.size(), i = 0;
  vector<string> factorization;
  while (i < n) {
    int j = i + 1, k = i;
    while (j < n && s[k] <= s[j]) {
      if (s[k] < s[j])
        k = i;
      else
        k++;
      j++;
    }
    while (i <= k) {
      factorization.push_back(s.substr(i, j - k));
      i += j - k;
    }
  }
  return factorization;
}
```

### 题目

#### s的子序列中t出现的次数

时间O(n*m)，空间上可使用滚动数组来压缩

```c++
int dp[1005][1005];
int numDistinct(string s, string t) { //s的子序列中t出现的次数
    int len1 = s.length(), len2 = t.length();
    s = ' ' + s; t = ' ' + t;
    int dp[1005][1005];
    memset(dp, 0, sizeof(dp));
    for (int i = 0; i <= len1; i++) dp[i][0] = 1; //当t为空时，s有一个子序列与其相等
    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            if (s[i] == t[j]) dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];//分为s使用了第i个字符和未使用两种情况
            else dp[i][j] = dp[i - 1][j];//s不使用第i个字符
            dp[i][j] %= mod;
        }
    }
    return dp[len1][len2];
}
```

#### 一个字符串的本质不同的子序列

```c++
/*方法一：
n为字符串长度，m为字符串中字符种类个数，时间复杂度O(n),空间复杂度为O(m)
当递归到第k个字符时，dp[k]为前k个字符中以s[k]（指的是这个字符的种类，并不一定是这个位置）结尾的子序列的数量，因为最后一个字符确定，所以当k>1时，考虑第二个字符是什么，记录从a到z分别作为结尾的最后出现的位置的子序列数，累加起来，最后将26种结尾加起来就是答案
*/
const int mod = 1e9 + 7;
int dp[26] = {0}; //dp[i] 表示以('a' + i)为结尾的子字符序列的个数
int distinctSubseqII(string s) {
    int ans = 0;
    for (int i = 0; i < s.length(); i++) {
        int temp = (1 + ans) % mod; // 新的以s[i]结尾的子字符序列的个数，其中1是长度为1（s[i]本身）的子序列
        ans += temp - dp[s[i] - 'a']; // 更新ans
        ans %= mod;
        dp[s[i] - 'a'] = temp; //更新dp
    }
    return (ans + mod) % mod;
}
/*方法二 ???
当s[k]没出现过时,s[k]=2*s[k-1]+1,
当s[k]上一次出现是s[j]时，s[k]=2*s[k-1]-s[j-1];	
*/
```

#### 最长递增子序列个数 ？？？

```c++ 
//朴素解O(n²):
    int max_len = 0;
    vector<int> length(n + 1, 1), count(n + 1, 1);
	//使用length[i]记录第i个位置及之前能够形成的最长递增子序列长度，向前寻找长度减一的位置来更新count
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j < i; j++) {
            if (nums[i] > nums[j]) {
                if (length[i] < length[j] + 1 ) {
                    length[i] = length[j] + 1;
                    count[i] = count[j];
                } else if (length[i] == length[j] + 1) 
                    count[i] += count[j];
            }
        }
        max_len = max(max_len, length[i]);
    }
    int ans = 0;
    for(int i = 1; i <= n; i++) 
        if (length[i] == max_len)
            ans += count[i];
//O(nlogn):leetcode673???
//本质不同的最长上升子序列数量？？？
```

## 计算几何

### 基础

#### 直线

记录直线上一点及直线的方向向量。

#### 正弦定理

a/sinA = b/sinB = c/sinC = 2R

#### 余弦定理

a^2^=b^2^+c^2^-2bc cosA

#### 判断一个点在直线的哪边

##### 叉乘

a =（a~x~, a~y~, a~z~) , b =（ b~x~, b~y~, b~z~)

a×b=(a~y~b~z~−a~z~b~y~, a~z~b~x~−a~x~b~z~, a~x~b~y~−a~y~b~x~)

##### 利用叉乘判断

假设直线上一点为P，向右(x>0)的方向向量为v，判断的点为Q

利用叉乘v×PQ=|v||PQ|sin = a~x~b~y~−a~y~b~x~

值为正说明在上方，值为负说明在下方

#### 将点沿原点进行旋转



### 距离

点A(x~1~,y~1~)，B(x~2~,y~2~)

#### 欧氏距离

|AB| = sqrt( (x~1~-x~2~)^2^ + (y~1~-y~2~)^2^ )

扩展到多维 |AB| = sqrt( (x~1~-x~2~)^2^ + (y~1~-y~2~)^2^ + ... + （z~1~-z~2~)^2^  )

#### 曼哈顿距离

|AB|=|x~1~-x~2~|+|y~1~-y~2~|

扩展到多维 |AB|=|x~1~-x~2~|+|y~1~-y~2~| + ... + |z~1~-z~2~|

#### 切比雪夫距离

又叫国王移动距离，国际象棋中国王可以移动到附近八个格子

|AB|= max(|x~1~-x~2~|,|y~1~-y~2~|)

#### 曼哈顿距离与切比雪夫距离的相互转化

==切比雪夫距离适用于求最值，曼哈顿距离适用于求和。==

##### 曼哈顿距离转为切比雪夫距离

曼哈顿距离|AB|=|x~1~-x~2~|+|y~1~-y~2~|=  max (x~1~-x~2~+y~1~-y~2~，x~2~-x~1~+y~1~-y~2~，x~1~-x~2~+y~2~-y~1~，x~2~-x~1~+y~2~-y~1~) 

​	 = max (|x~1~-x~2~+y~1~-y~2~|,|x~2~-x~1~+y~1~-y~2~| ) = max (|(x~1~+y~1~)-(x~2~+y~2~)|,|(x~1~-y~1~)-(x~2~-y~2~)| ) 

得到了A’(x~1~+y~1~，x~1~-y~1~)，B’(x~2~+y~2~，x~2~-y~2~)的切比雪夫距离

==所以在求曼哈顿距离时，将(x,y)存为(x+y,x-y)，求切比雪夫距离即可。==

[[P4648 [IOI 2007\] pairs 动物对数 - 洛谷](https://www.luogu.com.cn/problem/P4648)]

```c++
```

##### 切比雪夫距离转为曼哈顿距离

切比雪夫距离|AB|= max(|x~1~-x~2~|,|y~1~-y~2~|) = |(x~1~+y~1~)/2-(x~2~+y~2~)/2|+|(x~1~-y~1~)/2-(x~2~-y~2~)/2|

得到了A’((x~1~+y~1~)/2，(x~1~-y~1~)/2)，B’((x~2~+y~2~)/2，(x~2~-y~2~)/2)的曼哈顿距离

==所以在求切比雪夫距离时，将(x,y)存为( (x+y)/2,(x-y)/2 )，求曼哈顿距离即可。==

[[P3964 [TJOI2013\] 松鼠聚会 - 洛谷](https://www.luogu.com.cn/problem/P3964)]

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn = 2e6+10;

int x[maxn],y[maxn],x1[maxn],y2[maxn],prex[maxn],prey[maxn];
int n,k,t;

signed main(){
    cin>>n;
	for(int i=1;i<=n;i++){
		cin>>k>>t;
		x[i]=(k+t);y[i]=(k-t);//先不除以2,最后答案补上就可以
		x1[i]=x[i];y2[i]=y[i];
	}
	sort(x1+1,x1+1+n);
	sort(y2+1,y2+1+n);
	for(int i=1;i<=n;i++){
		prex[i]=prex[i-1]+x1[i];
		prey[i]=prey[i-1]+y2[i];
	}
	int ans = inf;
	for(int i=1;i<=n;i++){
		t = lower_bound(x1+1,x1+1+n,x[i])-x1;
		int tem = (prex[n] - prex[t]) - (n - t) * x[i];
		tem += (t - 1) * x[i] - prex[t-1];
		t = lower_bound(y2+1,y2+1+n,y[i])-y2;
		tem += (prey[n] - prey[t]) - (n - t) * y[i];
		tem += (t - 1) * y[i] - prey[t-1];
		ans = min(ans,tem);
	}
	cout << ans / 2 << endl;
}
```

### 扫描线 ???

OIwiki剩下内容？？？

[[P5490 【模板】扫描线 & 矩形面积并](https://www.luogu.com.cn/problem/P5490)]

线段树完成两个操作：区间修改，询问区间上有多少个点大于0

由于线段成对出现，所以可以通过pushup处的代码解决该要求。

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn = 2e6+10;
int n,m,k,cnt,N=1;
int arr[maxn][4];
int order[maxn];

#define ls i<<1
#define rs i<<1|1

struct node{
    int lazy,len;
}tr[maxn];//要开8倍区间

struct line{
    int l,r,h,val;//val表示这条边是矩形上面的边还是下面的边
    bool operator<(const line& x){
        return h < x.h;
    }
};

vector<line>vec;

void pushup(int i,int l,int r){
    //因为所有的边都是成对的，如果cnt>0说明有个矩形的下边覆盖了这条边，而对应的上边还没有计算，所以len一定就是该区间的长度。
    //如果cnt==0，那么由子节点更新，如果子节点正确那么父节点也正确，不断向下递归，要么找到子节点的cnt>0，要么更新直到叶节点，可以知道叶节点是一定正确的。
    if(tr[i].lazy){
        tr[i].len=order[r+1]-order[l];
    }else tr[i].len=tr[ls].len+tr[rs].len;
    //叶子节点也有可能向下找，所以要开8倍区间
}

void change(int i,int l,int r,int p,int q,int val){
    if(p==l&&q==r){
        tr[i].lazy+=val;
        pushup(i,l,r);
        return ;
    }
    int mid = (l+r)/2;
    if(q<=mid)change(ls,l,mid,p,q,val);
    else if(p>=mid+1)change(rs,mid+1,r,p,q,val);
    else change(ls,l,mid,p,mid,val),change(rs,mid+1,r,mid+1,q,val);
    pushup(i,l,r);
}

int get_ans(int to){
    //离散化
    int sum = 0;
    for(int i=1;i<=n;i++){
        for(int j=0;j<4;j++){
            if(j%2==to)order[++sum]=arr[i][j];
        }
    }
    sort(order+1,order+1+2*n);
    cnt = unique(order+1,order+1+2*n)-order-1;
    for(int i=1;i<=n;i++){
        arr[i][0]=lower_bound(order+1,order+1+cnt,arr[i][0])-order;
        arr[i][2]=lower_bound(order+1,order+1+cnt,arr[i][2])-order;
        vec.push_back({arr[i][0],arr[i][2],arr[i][1],1});
        vec.push_back({arr[i][0],arr[i][2],arr[i][3],-1});
    }
    sort(vec.begin(),vec.end());
    //添加线段
    int ans = 0;
    for(int i=0;i<vec.size();i++){
        if(i)ans+=tr[1].len*(vec[i].h-vec[i-1].h);
        //加上向下这一段的矩形，之后才更新这一行
        change(1,1,cnt-1,vec[i].l,vec[i].r-1,vec[i].val);
        //左右边界是按照区间给的，第i个横坐标对应第i个区间，所以左边[l,r]对应的是区间[l,r-1];
    }
    return ans;
}

void work(){
    cin>>n;
    int sum = 0;
    for(int i=1;i<=n;i++){
        for(int j=0;j<4;j++){
            cin>>arr[i][j];
        }
        if(arr[i][0]>arr[i][2])swap(arr[i][0],arr[i][2]);
        if(arr[i][1]>arr[i][3])swap(arr[i][1],arr[i][3]);
    }
   cout<<get_ans(0);
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

[[P1856 [IOI 1998 \] [USACO5.5] 矩形周长Picture - 洛谷](https://www.luogu.com.cn/problem/P1856)]

周长计算方法中，每一条线段做出的贡献是当前的长度-上一次长度（差值）

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn = 2e6+10;

int n, N=1;
int arr[maxn][4];
int order[maxn], tot;

struct node {
    int lazy, len;
} tr[maxn<<2]; // 开4倍空间

#define ls i<<1
#define rs i<<1|1

struct line {
    int l, r, h, val;
    bool operator<(const line& x){
        if(x.h!=h)return h < x.h;
        else return val > x.val;//按val降序,否则会出错
    }
};

vector<line> vec;

void pushup(int i, int l, int r) {
    if (tr[i].lazy) {
        tr[i].len = order[r+1] - order[l];
    } else {
        tr[i].len = tr[ls].len + tr[rs].len;
    }
}

void change(int i, int l, int r, int p, int q, int val) {
    if (p <= l && r <= q) {
        tr[i].lazy += val;
        pushup(i, l, r);
        return;
    }
    int mid = (l + r) >> 1;
    if (p <= mid) change(ls, l, mid, p, q, val);
    if (q > mid) change(rs, mid+1, r, p, q, val);
    pushup(i, l, r);
}

int get_ans(int now) { // 0:水平边 1:竖直边
    vec.clear();memset(tr, 0, sizeof(tr)); //初始化

    int num = 0;
    for (int i = 1; i <= n; i++) {
        if (now == 0) { // 水平边
            vec.push_back({arr[i][0], arr[i][2], arr[i][1], 1});
            vec.push_back({arr[i][0], arr[i][2], arr[i][3], -1});
            order[++num] = arr[i][0];
            order[++num] = arr[i][2];
        } else { // 竖直边
            vec.push_back({arr[i][1], arr[i][3], arr[i][0], 1});
            vec.push_back({arr[i][1], arr[i][3], arr[i][2], -1});
            order[++num] = arr[i][1];
            order[++num] = arr[i][3];
        }
    }

    // 离散化
    sort(order+1, order+1+num);
    tot = unique(order+1, order+1+num) - order - 1;

    sort(vec.begin(), vec.end());

    int ans = 0, last = 0;
    for (int i = 0; i < vec.size(); i++) {
        int L = lower_bound(order+1, order+1+tot, vec[i].l) - order;
        int R = lower_bound(order+1, order+1+tot, vec[i].r) - order;
        change(1, 1, tot-1, L, R-1, vec[i].val);
        ans += abs(tr[1].len - last);
        last = tr[1].len;
    }
    return ans;
}

void work() {
    cin >> n;
    for (int i = 1; i <= n; i++) {
        for (int j = 0; j < 4; j++) {
            cin >> arr[i][j];
        }
        if (arr[i][0] > arr[i][2]) swap(arr[i][0], arr[i][2]);
        if (arr[i][1] > arr[i][3]) swap(arr[i][1], arr[i][3]);
    }
    cout << get_ans(0) + get_ans(1) << endl;
}

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    while (N--) {
        work();
    }
}
```

#### B 维正交范围

B 维正交范围指在一个 B 维直角坐标系下，第i维坐标在一个整数范围[l~i~,r~i~]间，内部的点集，一维正交范围是区间，二维是矩形，三维是立方体。

不止是点集，求线段集合的大小也可以使用扫描线，当线段完全被加入之后再修改该线段的贡献

#### 二维数点

在一个二维平面上存在多个点，多次离线询问一个矩阵内包含了多少个点，可以对数点以及询问都按照其中一维升序排序，然后在这个维度上进行扫描线，动态地插入点，另一维用数据结构来存储信息。

1.[P5677 GZOI2017 配对统计](https://www.luogu.com.cn/problem/P5677):将点改为线段

2.[P1972 SDOI2009 HH的项链](https://www.luogu.com.cn/problem/P1972):统计区间内出现了多少个不同的数，只将相同的数中最后一个出现的横坐标值设为1，其余不管

3.给定长为n的序列，m次查询区间中有多少值只出现一次:当我们扫描线扫到一个数时，让其贡献为1，让其前驱的贡献为-1,其余前驱的贡献为0。进一步推广，问有多少个数出现的次数属于$[l,r]$，只需要第$l$个最新出现的数的贡献为1，第$r+1$个贡献为-1，其余为0

4.[Problem - D - Codeforces](https://codeforces.com/contest/522/problem/D):多次询问区间内具有相同值的数对的最小距离

没查询到一个数对就将数对靠左的位置的值设为该数对的距离，转换为单点修改，区间查最小值

5.[P4137 Rmq Problem / mex ](https://www.luogu.com.cn/problem/P4137):每个数的值为其出现的最后的位置，维护区间最小值，然后线段树上二分区间第一个值小于$l$的位置

#### 三维数点

见CDQ分治部分

### 平面最近点对

采用分治算法，每次把点集分成左右两部分，两部分分别求解最近点对距离并将res设置为两者中的较小值，合并时将x坐标距离分割线小于res的放到一起，按照y坐标排序，每个点都和 位于上方 且 y坐标差值小于res 的点求距离并更新res，由于在上方res*(2res)的矩形中放置点且要求左右两部分都分别不存在任意两点距离小于res，所以点的个数不会超过6个

T(n)=2\*T(n/2) + O(nlogn)，总时间复杂度为O(nlog^2^n)

```c++
#include<bits/stdc++.h>
//#pragma GCC optimize("O0")
using namespace std;
#define int long long
#define pii pair<double,double>
#define F first
#define S second
#define endl '\n'
#define pause system("pause")
const int maxn = 2e6+10;
const int inf = 3*1e18;
pii arr[maxn];
int n,m,k,x,y,z,N=1;

bool cmp1 (pii t1, pii t2){
    return t1.F < t2.F;
}

bool cmp2 (pii t1, pii t2){
    return t1.S < t2.S;
}

double get_dis(pii i, pii j){
    return sqrt((i.F-j.F)*(i.F-j.F)+(i.S-j.S)*(i.S-j.S));
}

vector<pii>vec;

double solve (int l, int r){
    if(l == r){
        return inf;
    }
    int mid = (l + r) / 2;
    double res = min(solve(l, mid), solve(mid + 1, r));
    vec.clear();
    for (int i = l; i <= r; i++) {
        if(fabs(arr[i].F - arr[mid].F) <= res) vec.push_back(arr[i]);
    }
    sort(vec.begin(), vec.end(), cmp2);
    for (int i = 0; i < vec.size(); i++) {
        for(int j = i + 1; j < vec.size(); j++) {
            if (fabs(vec[i].S - vec[j].S) >= res) break;
            res = min(res, get_dis(vec[i], vec[j]));
        }
    }
    return res;
}

void work(){
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> arr[i].F >> arr[i].S;
    }
    sort(arr + 1, arr + 1 + n, cmp1);
    cout << fixed << setprecision(4) << solve(1, n);
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```



### 最小圆覆盖

```c++
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
using namespace std;

int n;
double r;

struct point {
  double x, y;
} p[100005], o;

double sqr(double x) { return x * x; }

double dis(point a, point b) { return sqrt(sqr(a.x - b.x) + sqr(a.y - b.y)); }

bool cmp(double a, double b) { return fabs(a - b) < 1e-8; }

point geto(point a, point b, point c) {
  double a1, a2, b1, b2, c1, c2;
  point ans;
  a1 = 2 * (b.x - a.x), b1 = 2 * (b.y - a.y),
  c1 = sqr(b.x) - sqr(a.x) + sqr(b.y) - sqr(a.y);
  a2 = 2 * (c.x - a.x), b2 = 2 * (c.y - a.y),
  c2 = sqr(c.x) - sqr(a.x) + sqr(c.y) - sqr(a.y);
  if (cmp(a1, 0)) {
    ans.y = c1 / b1;
    ans.x = (c2 - ans.y * b2) / a2;
  } else if (cmp(b1, 0)) {
    ans.x = c1 / a1;
    ans.y = (c2 - ans.x * a2) / b2;
  } else {
    ans.x = (c2 * b1 - c1 * b2) / (a2 * b1 - a1 * b2);
    ans.y = (c2 * a1 - c1 * a2) / (b2 * a1 - b1 * a2);
  }
  return ans;
}

int main() {
  scanf("%d", &n);
  for (int i = 1; i <= n; i++) scanf("%lf%lf", &p[i].x, &p[i].y);
  for (int i = 1; i <= n; i++) swap(p[rand() % n + 1], p[rand() % n + 1]);
  o = p[1];
  for (int i = 1; i <= n; i++) {
    if (dis(o, p[i]) < r || cmp(dis(o, p[i]), r)) continue;
    o.x = (p[i].x + p[1].x) / 2;
    o.y = (p[i].y + p[1].y) / 2;
    r = dis(p[i], p[1]) / 2;
    for (int j = 2; j < i; j++) {
      if (dis(o, p[j]) < r || cmp(dis(o, p[j]), r)) continue;
      o.x = (p[i].x + p[j].x) / 2;
      o.y = (p[i].y + p[j].y) / 2;
      r = dis(p[i], p[j]) / 2;
      for (int k = 1; k < j; k++) {
        if (dis(o, p[k]) < r || cmp(dis(o, p[k]), r)) continue;
        o = geto(p[i], p[j], p[k]);
        r = dis(o, p[i]);
      }
    }
  }
  printf("%.10lf\n%.10lf %.10lf", r, o.x, o.y);
  return 0;
}


```

## 其它

### 约瑟夫环问题

```c++
ysf(int n,int k,int i);共有n个人从0~n-1编号，喊k的人出局，第i个出局的人的编号。
    
int ysf(int n,int k,int i){
    if(i==1)return (k-1)%n;
    else return (ysf(n-1,k,i-1)+k)%n;
}
```

![约瑟夫环](../../../Pictures/Screenshots/typora图片/约瑟夫环.png)

相当于从原本的基础上开了一个新的环，每个环只求第一个出局的人。

```c++
int now=0;
for(int i=2;i<=n;i++){
	now=(now+k)%i;
}
```

### 求第n小的数

#### 快排分治

```c++
int quicksort(int l,int r,int last){
    if(l==r)return arr[l];
    int tem=arr[l];
    int ll=l,rr=r;
    while(ll<rr){
        while(ll<rr&&arr[rr]>=tem)rr--;
        arr[ll]=arr[rr];
        while(ll<rr&&arr[ll]<=tem)ll++;
        arr[rr]=arr[ll];
    }
    arr[ll]=tem;
    if(ll-l+1==last)return arr[ll];
    if(ll-l+1>last){
        return quicksort(l,ll-1,last);
    }else return quicksort(ll+1,r,last-ll+l-1);
}
```

#### nth_element

```c++
void nth_element(arr+1,arr+k,arr+1+n,cmp);
//O(n)时间复杂度，将第k小的数放到对应位置，左边的数都比它小，右边的数都比它大
```

min_element     max_element

### 最大全1子矩阵？？？

#### 单调栈

遍历每一行作为底，对每一行用单调栈正反跑一遍。O(n*m)

```c++
int lnext[maxn],rnext[maxn];
int ans = 0;
for (int i = 1;i <= n; i++) {
    stack<int>sta;
    for(int j = 1; j <= m; j++) {
        while (!sta.empty() && arr[i][j] < arr[i][sta.top()]) {
            rnext[sta.top()] = j;
            sta.pop();
        }
        sta.push(j);
    }
    while(!sta.empty()){
        rnext[sta.top()] = m + 1;sta.pop();
    }
    for(int j = m; j >= 1; j--) {
        while (!sta.empty() && arr[i][j] < arr[i][sta.top()]) {
            lnext[sta.top()] = j;
            sta.pop();
        }
        sta.push(j);
    }
    while(!sta.empty()){
        lnext[sta.top()] = 0;sta.pop();
    }
    for (int j = 1; j <= m; j++) {
        ans = max (ans, (arr[i][j]+1)*(rnext[j]-lnext[j]-1));
    }
}
cout << ans;
```

#### 扫描线？？？

#### 最大全1正方形

##### dp

```c++
int ans = 0;
for(int i=1;i<=n;i++){
    for(int j=1;j<=m;j++){
        if(arr[i-1][j-1]==0)continue;
        else dp[i][j]=min(min(dp[i-1][j-1],dp[i-1][j]),dp[i][j-1])+1;
        ans = max(ans,dp[i][j]);
    }
}
cout<<ans*ans;
```



#### 例题

[[Largest Submatrix - SPOJ MINSUB - Virtual Judge](https://vjudge.net/problem/SPOJ-MINSUB#author=GPT_zh)]:二分答案，把比答案大于等于的元素设为1，小的设为1，check时求最大全1子矩阵的大小，和要求的矩阵大小比较。

### A*算法

设f(x)=g(x)+h(x)，其中g(x)是从原点走到该点的代价，h(x)是走到目标点的预估代价,

h*(x)是该点走到目标点的真实代价

可采纳性：如果h(x)<=h*(x)，则可以找到最优解

单调性：如果h(x)<=h(y)+c(x,y),c(x,y)指的是从x到y的实际代价，那么保证每个点被遍历一次时就已经可以放入封闭列表，不再进行下一次更新了。

如果不满足单调性，到达目标点时不能认为已经取到了最优解，等que中所有点的预估代价都比目标点的实际代价大时，就可以认为已经结束了（在权值非负时）。

过程：每次取f(x)最小的点放入封闭列表直到找到目标

### IDA*算法



### 异或哈希

遇到数字个数奇偶来得出结论的（如平方数，），可以尝试

[【算法讲解】杂项算法——异或哈希(xor-hashing)-CSDN博客](https://blog.csdn.net/notonlysuccess/article/details/130959107)

#### 验证两个序列的值集是否相同

[E - Prefix Equality](https://atcoder.jp/contests/abc250/tasks/abc250_e)

```c++
#include<bits/stdc++.h>
//#pragma GCC optimize("O0")
using namespace std;
#define int long long
#define ull unsigned long long
#define pii pair<int,int>
#define F first
#define S second
#define endl '\n'
#define pause system("pause")
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn = 2e6+10;
const int mod = 1e9+7;
int arr[maxn];
int brr[maxn];
int pre1[maxn],pre2[maxn];
int n,m,k,x,y,z,N=1;
char c;string s;

map<int,bool>num;
map<int,uint64_t>mp;
mt19937_64 rnd(time(0));

void work(){
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>arr[i];
        pre1[i]=pre1[i-1];
        if(mp.count(arr[i])==0){
            mp[arr[i]]=rnd();
        }
        if(num[arr[i]]==0){
            pre1[i]^=mp[arr[i]];
        }
        num[arr[i]]=1;
    }
    num.clear();
    for(int i=1;i<=n;i++){
        cin>>brr[i];
        pre2[i]=pre2[i-1];
        if(mp.count(brr[i])==0){
            mp[brr[i]]=rnd();
        }
        if(num[brr[i]]==0){
            pre2[i]^=mp[brr[i]];
        }
        num[brr[i]]=1;
    }
    cin>>m;
    while(m--){
        cin>>x>>y;
        if(pre1[x]==pre2[y])cout<<"Yes"<<endl;
        else cout<<"No"<<endl;
    }
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    // cin>>N;
    while (N--) {
        work();
    }
}
```

[1009 小塔的序列](https://acm.hdu.edu.cn/contest/problem?cid=1159&pid=1009)

[[E-Equal_2025牛客暑期多校训练营3](https://ac.nowcoder.com/acm/contest/108300/E)]:主要是通过线性筛将每个数都赋予了hs值

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn = 5e6+10;
const int mod = 1e9 + 7;
int n,m,N=1;
int arr[maxn];
int hs[maxn];

bool vis[maxn];
int cnt, prime[maxn];
void get_prime (int w) {
    for (int i = 2; i <= w; i++) {
        if (!vis[i]) {
            prime[++cnt] = i; hs[i] = rand() % mod;
        }
        for (int j = 1; j <= cnt && i * prime[j] <= w; j++) {
            vis[i * prime[j]] = 1; hs[i * prime[j]] = hs[i] ^ hs[prime[j]];
            if (i % prime[j] == 0) break;
        }
    }
}

void work(){
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> arr[i];
    }
    if (n % 2) {
        cout << "YES" << endl;
        return ;
    }
    if (n == 2) {
        if (arr[1] == arr[2]) {
            cout << "YES" << endl;
        } else cout << "NO" << endl;
        return ;
    }
    ull now = 0;
    for (int i = 1; i <= n; i++) {
        now ^= hs[arr[i]];
    }
    if (now) cout << "NO" << endl;
    else cout << "YES" << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    srand(time(NULL));
    get_prime(5000000);
    cin>>N;
    while (N--) {
        work();
    }
}
```



### 最大无交叉子集

动态规划

![动态规划最大无交叉子集](../大二下/算法设计与分析/图片/动态规划最大无交叉子集.png)

### 遍历子集的子集（枚举子集）

$O(3^n)$复杂度

```c++
for(int i=0;i<(1ll<<n);i++){
	for(int j=i;;j=(j-1)&i){
		//剩余操作
        if(!j)break;
    }
}
```

[P5911 POI 2004 PRZ - 洛谷](https://www.luogu.com.cn/problem/P5911)

```c++
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
int n,m,N=1;

int dp[1 << 16];
int t[16], w[16];
bool vis[maxn];

void dfs(int now) {
    vis[now] = 1;
    for (int i = now;;i = (i - 1) & now) {
        int x = i, y = now ^ i;
        if (!vis[x]) {
            dfs(x);
        }
        if (!vis[y]) {
            dfs(y);
        }
        dp[now] = min(dp[now], dp[x] + dp[y]);
        if (!i) break;
    }
}

void work(){
    cin >> n >> m;
    for (int i = 0;i < m; i++) {
        cin >> t[i] >> w[i];
    }
    memset(dp, 0x3f, sizeof(dp));
    for (int i = 0; i < (1 << m); i++) {
        int sum_w = 0, max_t = 0;
        for (int j = 0; j < m; j++) {
            if ((i >> j) & 1) {
                sum_w += w[j];
                max_t = max(max_t, t[j]);
            }
        }
        if (sum_w <= n) dp[i] = max_t;
    }
    dfs((1 << m) - 1);
    cout << dp[(1 << m) - 1] << endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    // cin >> N;
    while (N--) {
        work();
    }
}
```

### 分治

#### 线段树分治

#### CDQ分治

##### 解决和点对有关的问题

[[P3810 【模板】三维偏序（陌上花开） - 洛谷](https://www.luogu.com.cn/problem/P3810)]:

```c++
#include<bits/stdc++.h>
using namespace std;
#define endl '\n'
#define pii pair<int, int>
const int maxn = 2e5+10;
const int mod = 1e9+7;
int n,m,N=1;

int num[maxn];

struct node {
    int x, y, z, ans, val;
    bool operator < (const node & b) & {
        if (x != b.x)return x < b.x;
        else if (y != b.y) return y < b.y;
        else return z < b.z;
    }
}arr[maxn], brr[maxn];

bool cmp (node a, node b) {
    return a.y < b.y;
}

vector <pii> vec;

int tr[maxn];
int lowbit (int t) {
    return t & (-t);
}
void change (int w, int v) {
    while (w <= m) {
        tr[w]+=v;
        w += lowbit(w);
    }
}
int query (int w) {
    int res = 0;
    while (w) {
        res += tr[w];
        w -= lowbit(w);
    }
    return res;
}


void solve (int l, int r) { //过程模拟归并排序，使用sort会成为时间复杂度瓶颈
    if (l == r) return ;
    int mid = (l + r) / 2;
    solve (l, mid); solve(mid + 1, r);
    int L = l, R = mid + 1, now = l;
    vec.clear();
    while (L <= mid && R <= r) {
        if (arr[L].y <= arr[R].y) {
            change (arr[L].z, arr[L].val);
            vec.push_back({arr[L].z, -arr[L].val});
            brr[now++] = arr[L++];
        } else {
            arr[R].ans += query(arr[R].z);
            brr[now++] = arr[R++];
        }
    }
    while (R <= r) {
        arr[R].ans += query(arr[R].z);
        brr[now++] = arr[R++];
    }
    while (L <= mid) {
        brr[now++] = arr[L++];
    }
    for (auto x : vec) {
        change(x.first, x.second);
    }
    for (int i = l; i <= r; i++) arr[i] = brr[i];
}

void work(){
    cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        cin >> brr[i].x >> brr[i].y >> brr[i].z;
        brr[i].val = 1;
    }
    sort (brr + 1, brr + 1 + n);
    int cnt = 1; arr[1] = brr[1];
    for (int i = 2; i <= n; i++) { // 去重
        if (brr[i].x ^ brr[i-1].x || brr[i].y ^ brr[i-1].y || brr[i].z ^ brr[i-1].z) {
            arr[++cnt] = brr[i];
        } else arr[cnt].val++,arr[cnt].ans++;
    }
    solve (1, cnt);
    for (int i = 1; i <= cnt; i++) {
        num[arr[i].ans]+=arr[i].val;
    }
    for (int i = 0; i < n; i++) {
        cout << num[i] << endl;
    }
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    work();
}
```

[[P3157 CQOI2011 动态逆序对 - 洛谷](https://www.luogu.com.cn/problem/P3157)]:按顺序删除m个数，输出删除之前的逆序对数

考虑每个数删除造成的影响，即删除比他晚的，能与其形成逆序对的数

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn = 2e6+10;
const int mod = 1e9+7;
int n,m,N=1;
int num[maxn];

struct node {
    int t,pos,v;
}arr[maxn], temp[maxn];

int ans, tr[maxn];
int lowbit (int t) {
    return t & (-t);
}
void change (int w, int v) {
    while (w <= 100000) {
        tr[w]+=v;
        w += lowbit(w);
    }
}
int query (int w) {
    int res = 0;
    while (w) {
        res += tr[w];
        w -= lowbit(w);
    }
    return res;
}

vector <int> vec;

int des[maxn];

bool cmp1 (node a, node b) {
    return a.pos > b.pos;
}
bool cmp2 (node a, node b) {
    return a.v > b.v;
}

void solve (int l, int r) { //因为一个数形成逆序对的情况有两种，所以寻找两次
    if (l == r) return ;
    int mid = (l + r) / 2;
    solve(l, mid); solve(mid + 1, r);
    int L = l, R = mid + 1;
    sort (arr + l, arr + mid + 1, cmp1);
    sort (arr + mid + 1, arr + r + 1, cmp1);
    while (L <= mid && R <= r) {
        if (arr[L].pos > arr[R].pos) {
            change(arr[L].v, 1);
            vec.push_back(arr[L].v);
            ++L;
        } else {
            des[arr[R].t] += query(arr[R].v);
            ++R;
        }
    }
    while (R <= r) {
        des[arr[R].t] += query(arr[R].v);
        ++R;
    }

    for (auto x : vec) change(x, -1); vec.clear();
    L = l; R = mid + 1;

    sort (arr + l, arr + mid + 1, cmp2);
    sort (arr + mid + 1, arr + r + 1, cmp2);
    while (L <= mid && R <= r) {
        if (arr[L].v > arr[R].v) {
            change(arr[L].pos, 1);
            vec.push_back(arr[L].pos);
            ++L;
        } else {
            des[arr[R].t] += query(arr[R].pos);
            ++R;
        }
    }
    while (R <= r) {
        des[arr[R].t] += query(arr[R].pos);
        ++R;
    }
    for (auto x : vec) change(x, -1);vec.clear();
}

bool cmp(node a, node b) {
    return a.t > b.t;
}

void work(){
    cin >> n >> m;
    int x;
    for (int i = 1; i <= n; i++) {
        cin >> x;
        change (x, 1);
        ans += i - query(x);
        arr[i].v = x;
        arr[i].pos = i;
        arr[i].t = m + 1;
        num[arr[i].v] = i;
    }
    for (int i = 1; i <= m; i++) {
        cin >> x;
        arr[num[x]].t = i;
    }
    for (int i = 0; i <= n; i++) tr[i] = 0;
    sort (arr + 1, arr + 1 + n, cmp);
    solve (1, n);
    for (int i = 1; i <= m; i++) {
        cout << ans << endl;
        ans -= des[i];
    }
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    work();
}
```

##### CDQ 分治优化 1D/1D 动态规划的转移

### 蔡勒公式（计算星期几）

```c++
int zeller_weekday(int year, int month, int day) {
    // 历法切换日期：1582-10-15 公历开始
    bool useGregorian = false;
    if (year > 1582) {
        useGregorian = true;
    } else if (year == 1582) {
        if (month > 10) useGregorian = true;
        else if (month == 10 && day >= 15) useGregorian = true;
    }

    // 检查跳跃日期（1582-10-05 ~ 1582-10-14）
    if (year == 1582 && month == 10 && day >= 5 && day <= 14) {
        throw invalid_argument("该日期在历法更替的跳跃区间内，不存在！");
    }

    // 处理 1月、2月为上一年的 13、14 月
    if (month == 1 || month == 2) {
        month += 12;
        year -= 1;
    }

    int c = year / 100;
    int y = year % 100;
    int w;

    if (useGregorian) {
        // 公历公式
        w = y + y / 4 + c / 4 - 2 * c + (26 * (month + 1)) / 10 + day - 1;
    } else {
        // 儒略历公式
        w = y + y / 4 + (26 * (month + 1)) / 10 + day + 5;
    }

    w = (w % 7 + 7) % 7; // 保证结果非负
    if  (w == 0) w = 7;
    return w;
}
```



## 题目

[Problem - D - Codeforces](https://codeforces.com/contest/1453/problem/D)

需要设计一个游戏关卡，由01字符串组成，1表示存档点，0表示普通关卡，规定每一步可以从第i个关卡前进到第i+1个关卡，不过有0.5的概率会成功，剩下0.5的概率会失败，失败的话会返回最近的存档点重新开始，现在问如何设计关卡，可以使得到达终点的期望为k

![image-20250527111004018](../../../AppData/Roaming/Typora/typora-user-images/image-20250527111004018.png)

```c++
#include<bits/stdc++.h>
//#pragma GCC optimize("O0")
using namespace std;
#define int long long
#define ull unsigned long long
#define pii pair<int,int>
#define F first
#define S second
#define endl '\n'
#define pause system("pause")
#define inf 0x3f3f3f3f3f3f3f3f
const int maxn = 2e6+10;
const int mod = 1e18;
int arr[maxn];
int dp[maxn];
int n,m,k,x,y,z,N=1;
char c;string s;
int cnt = 0;

void solve(int t){
    s+='1';
    for(int i=1;i<t;i++)s+='0';
}

void work(){
    cin>>n;
    if(n%2){
        cout<<-1<<endl;return ;
    }
    s="";
    for(int i=cnt;i>=1;i--){
        while(n>=arr[i]){
            solve(i);
            n-=arr[i];
        }
    }
    cout<<s.length()<<endl;
    for(auto xx : s)cout<<xx<<' ';cout<<endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    cin>>N;
    int now = 4;
    for(cnt=1;cnt<=2000;cnt++){
        arr[cnt]=now-2;
        now*=2;
        if(now-2>mod)break;
    }
    while (N--) {
        work();
    }
}
```





[[Problem - F - Codeforces](https://codeforces.com/contest/2114/problem/F)] 把正整数n分割到m个背包，背包乘积为n且每个背包不超过k，问m最小为多少

n<=1e6时，n的约数个数约为1000个，背包数不会超过log~2~n个，所以dp[i]记录最少需要多少个背包能够满足i（其中i一定是n的约数),总复杂度为O(1000\*log~2~n)

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define endl '\n'
const int maxn = 1e6+10;
const int mod = 1e9+7;
int n,m,k,x,y,z,N=1;

int prime[maxn],cnt;
bool vis[maxn],vis1[maxn];

void get_p(int w){
    for(int i=2;i<=w;i++){
        if(!vis[i])prime[++cnt]=i;
        for(int j=1;j<=cnt&&i*prime[j]<=w;j++){
            vis[i*prime[j]]=1;
            if(i%prime[j]==0)break;
        }
    }
}

int get_num(int t){
    for(int i=1;i<=cnt&&prime[i]<=k;i++){
        if(t==1)break;
        while(t%prime[i]==0)t/=prime[i];
    }
    return t;
}

vector<pii>vec;
vector<int>fac;
queue<int>que;

void dfs(int now,int sum){
    if(now==vec.size()){
        vis1[sum]=0;
        fac.push_back(sum);
        return ;
    }
    int t = 1;
    for(int i=0;i<=vec[now].S;i++){
        dfs(now+1,sum*t);
        t*=vec[now].F;
    }
}

int get_ans(int t){
    vec.clear();fac.clear();
    int now = t;
    for(int i=1;i<=cnt&&prime[i]<=k;i++){
        if(now==1)break;
        int sum = 0;
        while(now%prime[i]==0){
            now/=prime[i];sum++;
        }
        if(sum)vec.push_back({prime[i],sum});
    }
    dfs(0,1);
    que.push(1);vis1[1]=1;
    int ans = -1;
    while(!que.empty()){
        int siz = que.size();
        while(siz--){
            now = que.front();que.pop();
            for(auto xx : fac){
                if(vis1[xx])continue;
                if(xx%now==0&&xx/now<=k){
                    vis1[xx]=1;que.push(xx);
                }
            }
        }
        ans++;
    }
    return ans;
}

void work(){
    cin>>n>>m>>k;
    int gcd=__gcd(n,m);
    n/=gcd;m/=gcd;
    if(get_num(n)>k||get_num(m)>k){
        cout<<-1<<endl;return ;
    }
    cout<<get_ans(n)+get_ans(m)<<endl;
}

signed main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    cin>>N;
    get_p(1000000);
    while (N--) {
        work();
    }
}
```
