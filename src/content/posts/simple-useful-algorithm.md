---
title: 简单实用的算法
published: 2024-05-24
description: 记录一些有意思的算法
image: ''
tags: ['算法']
category: '技术分享'
draft: false 
---

# 一些简单实用的算法

写代码的时候有时候会遇到一些很有意思的算法，将其记录在这篇文章中，随缘更新

## 数组循环移位算法

```c
void reverse(int *start,int *end){
    int tmp;
    for(;start < end;start++,end--){
        tmp = *start;
        *start = *end;
        *end = tmp;
    }
}

void move(int *arr,int length,int n){
    while(n<0) n+=length;
    reverse(arr,arr+n%length-1);
    reverse(arr+n%length,arr+length-1);
    reverse(arr,arr+length-1);
}
```

本算法实现了数组的循环移位，n即为移动的位数。

### 示例

```c
int a[] = {1,2,3,4,5,6,7,8,9,10};
int length = sizeof(a)/sizeof(int);
move(a,length,-5);
```

输出结果为

```bash
6 7 8 9 10 1 2 3 4 5 
```

### 解释

算法总共进行了三次取反操作

先将 [0,n-1] 进行了一次取反，再对 [n,length-1] 进行了一次取反，最后整体进行了一次取反

设元素索引为 $i$

1. 对于索引在 [0,n-1] 的元素

   第一次取反后 $i^{\prime}=n-1-i$

   第三次取反后 $i^{\prime\prime}=length-1-i^{\prime}=length-n+i$

2. 对于索引在 [n,length-1]的元素

   第二次取反后 $i^{\prime}=n+length-1-i$

   第三次取反后 $i^{\prime\prime}=length-1-i^{\prime}=i-n$

从而实现了循环移位操作

### 效率

#### 空间效率

没有申请额外的空间，效率为 $O(1)$

#### 时间效率

对于每个元素交换了两次，一共为n次交换操作，时间复杂度为 $O(n)$
