---
title: 说说 Markdown
published: 2024-05-11
description: '基础 markdown 教程'
tags: ['Web']
category: '技术分享'
draft: false
---

# Markdown

markdown是一门非常简单的标记语言

```markdown

这样就是一个段落

```

markdown就是这么简单，只需要在段落前空一行，后面空一行就是一个段落

事实上，空行在Markdown的语法中相当常见，基本上所有语法都要空行

## 标题

```markdown
# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
```

不要以为能无限套娃，后面就没有了

:::important
**记得要在#后面加上一个空格！**
:::

## 换行

```markdown

如果你需要换行，请在结尾打两个以上的空格
这样就是下一行了

这是下一个段落
```

## 强调

如果你要**_强调_**某个词，就

```markdown
**这样就是强调**
_这样就是斜体_
```

## 引用

> 这里也许应该放某个伟人的一句话

```markdown
> 尖括号加上一个括号就是引用
```

## 列表

1. 如果你想要创建一个有序列表

   ```markdown
   1. 第一项
   2. 第二项
   2. 第三项
   ```

:::important
​ **记得要加空格！**
:::

2. 如果你想要创建一个无序列表

   ```markdown
   - 内容
   - 内容
   - 内容
   ```

---

## 分割线

```markdown
---

***

```

这俩玩意会生成一条分割线

## 链接

```markdown
[这里是要显示的内容](https://这里是链接/)
```

[欢迎来访问 Markdown 官方教程](https://markdown.com.cn/)

## 图片

```markdown
![如果图片加载失败就会显示这段文字](这一个参数是图片链接 “第二个参数是图片的标题（鼠标放上去就可以看到）”)
```

记得标题要用双引号括起来（如果没有标题不写都行）

## 转义

我真的想打两个\*\*该怎么办？

那就转义！

```markdown
\*\*
```

如果你学过c的就会知道经典的反斜杠转义

## HTML

文章太单调，怎么办？

<p style="color:red;">插入HTML</p>

```markdown
<p style="color:red;">插入HTML</p>
```

直接写就可以了

## 代码

文章中出现了很多代码块，怎么写代码？

````markdown
```这里写上用的语言
代码
```

```c
#include <stdio.h>
int main(){
 printf("Hello World");
}
```
````
