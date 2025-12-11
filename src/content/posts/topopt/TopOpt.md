---
title: 拓扑优化：复现《A 99 line topology optimization code written in Matlab》
published: 2025-12-11
description: 使用Python编写复现了O. Sigmund的著名论文，并且进行了详细的理论推导
image: ""
tags:
  - 杂谈
  - 有限元分析
  - 拓扑优化
category: 技术分享
draft: false
---
## 前言

拓扑优化的详细过程，其实就是求解一个线性规划问题，对于论文中的情境，就是：

指定材料在一个空间内的填充率，怎么布局材料使得材料的刚度最大？

我们可以写出线性规划的目标和约束

$$
\left. \begin{array}{rl} \min\limits_{\mathbf{x}} : & c(\mathbf{x}) = \mathbf{U}^T \mathbf{K} \mathbf{U} = \displaystyle\sum_{e=1}^{N} (x_e)^p \mathbf{u}_e^T \mathbf{k}_0 \mathbf{u}_e \\[18pt] \text{subject to} : & \dfrac{V(\mathbf{x})}{V_0} = f \\[18pt] : & \mathbf{K}\mathbf{U} = \mathbf{F} \\[15pt] : & \mathbf{0} < \mathbf{x}_{\text{min}} \leq \mathbf{x} \leq \mathbf{1} \end{array} \right\}
$$

其中，$\mathbf{U}$ 是全局位移矩阵，$\mathbf{K}$ 是全局刚度矩阵，$\mathbf{x}$ 是材料密度，是我们需要优化的变量，$c(\mathbf{x})$ 是我们的优化目标。

约束条件即：
1. $f$ 是材料的填充率，$V(\mathbf{x})$ 是材料所占据的体积，$V_{0}$ 是总体积
2. $\mathbf{K}\mathbf{U} = \mathbf{F}$ 是有限元分析的步骤，通过这个方程我们可以用全局力 $\mathbf{F}$ 解出全局位移 $\mathbf{U}$
3. 对于最后一个条件，就是材料在每个区域的填充率超过1（全部填充），也不能过小

对于上述约束系统建立拉格朗日方程

$$
L(\mathbf{x},\lambda)= c(\mathbf{x})-\lambda(V(\mathbf{x})-V_{0}\cdot f)
$$
我们可以知道拉格朗日方程的极值条件和原约束系统的相同，于是我们可以知道对于线性规划的解，有：

$$
\frac{\partial c}{\partial x_e} + \lambda \frac{\partial V}{\partial x_e} = 0
$$
我们这里定义启发式的更新因子 $B_e$：

$$B_e = \frac{- \frac{\partial c}{\partial x_e}}{\lambda \frac{\partial V}{\partial x_e}}$$

可知在最优解处 $B_e = 1$，因为每个单元对于体积的贡献度相同，所以我们令 $\frac{\partial V}{\partial x_e}=1$，$\frac{\partial c}{\partial x_e}$ 即优化目标的梯度，为：

$$
\frac{\partial c}{\partial x_e}=p(x_{e})^p\,\mathbf{u}_e^T \mathbf{k}_0 \mathbf{u}_e
$$
我们采用OC方法迭代求解 $\mathbf{x}$，更新公式如下：

$$
x_e^{\text{new}}= \begin{cases} \max(x_{\min}, x_e - m) & \text{if } x_e B_e^\eta \le \max(x_{\min}, x_e - m) \\ \min(1, x_e + m) & \text{if } x_e B_e^\eta \ge \min(1, x_e + m) \\ x_e B_e^\eta & \text{otherwise} \end{cases}
$$

若每次的更新量小于阈值，求解结束，最终结果如下图所示：

![](./attachments/topology_optimization_result.png)

### 第一步：求解单元刚度矩阵

对于我们的问题，网格已经天然地划分好了，如上图所示，我们求解的空间即使一个二维的点阵，每个填充的节点即是一个正方形的主控单元，如下图所示：

![](./attachments/solve_ke.png)

对于这个单元，四个节点的形函数应该满足：

$$
\begin{cases}
N_{i}(\xi_{i},\eta_{i})=1,\quad N_{j \ne i}(\xi_{i},\eta_{i})=0 \\
\sum_{i=1}^{4} N_{i}(\xi,\eta) \equiv 1
\end{cases}
$$

可以写出：
$$
N_1 = \frac{1}{4}(1-\xi)(1-\eta)
$$
$$
N_2 = \frac{1}{4}(1+\xi)(1-\eta)
$$
$$
N_3 = \frac{1}{4}(1+\xi)(1+\eta)
$$
$$
N_4 = \frac{1}{4}(1-\xi)(1+\eta)
$$
接着，我们需要建立应变-位移矩阵，对于平面问题，应变 $\boldsymbol{\varepsilon} = [\varepsilon_x, \varepsilon_y, \gamma_{xy}]^T$。

$\mathbf{B}$ 矩阵由形函数的偏导数组成（注意不是对 $\xi,\eta$ 偏导）：

$$
\mathbf{B}_i = \begin{bmatrix} \frac{\partial N_i}{\partial x} & 0 \\ 0 & \frac{\partial N_i}{\partial y} \\ \frac{\partial N_i}{\partial y} & \frac{\partial N_i}{\partial x} \end{bmatrix}
$$
与此同时，我们还需要 $x,y$ 到 $\xi,\eta$ 的映射（雅可比矩阵）

$$
\mathbf{J}=\begin{bmatrix}
\frac{\partial x}{\partial \xi} & \frac{\partial x}{\partial \eta} \\
\frac{\partial y}{\partial \xi} & \frac{\partial y}{\partial \eta} 
\end{bmatrix}=\begin{bmatrix}
0.5 & 0 \\
0 & 0.5
\end{bmatrix}
$$

对于 **平面应力 (Plane Stress)** 问题，材料矩阵 $\mathbf{D}$ 为：

$$
\mathbf{D} = \frac{E}{1-\nu^2} \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1-\nu}{2} \end{bmatrix}
$$
其中 $E$ 是杨氏模量，$\nu$ 是泊松比。

最后，单元刚度矩阵 $\mathbf{K}_e$ 的定义是：

$$
\mathbf{K}_e = \int_{-1}^{1} \int_{-1}^{1} \mathbf{B}^T \mathbf{D} \mathbf{B} \cdot t \cdot \det(\mathbf{J}) \, d\xi d\eta
$$
这里，我们使用 MATLAB 进行计算

```matlab
syms xi eta x y nu E 'real'
N1 = (xi-1)*(eta-1)/4;
N2 = (xi+1)*(eta-1)/4;
N3 = (xi-1)*(eta+1)/4;
N4 = (xi+1)*(eta+1)/4;

function Bi = B_mat(N,xi,eta)
Bi = 2*[
    diff(N,xi) 0;
    0 diff(N, eta);
    diff(N, eta) diff(N, xi)    
];
end

J = [0.5 0; 0 0.5];
B = [B_mat(N1,xi,eta) B_mat(N2,xi,eta) B_mat(N3,xi,eta) B_mat(N4,xi,eta)];

D = E/(1-nu^2)*[
    1 nu 0;
    nu 1 0;
    0 0 (1-nu)/2;
];

res = det(J)*int(int(B'*D*B,xi,-1,1),eta,-1,1);
disp(simplify(res));
```

原论文中已经给出了KE的解析表达式

![](./attachments/origin_ke.png)

上述代码的计算结果与原论文代码的结果一致，我们转为 Python 函数：

```python
def get_element_stiffness_matrix(E=1.0, nu=0.3):
    k = np.array([
        1/2 - nu/6,        1/8 + nu/8,
        -1/4 - nu/12,      -1/8 + 3*nu/8,
        -1/4 + nu/12,      -1/8 - nu/8,
        nu/6,              1/8 - 3*nu/8
    ])
    
    KE = E / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    return KE
```

## 第二步：组装全局刚度矩阵

### 建立局部到全局的自由度映射

对于每一个节点，它可能被多个单元共用，所以我们需要建立局部自由度与全局自由度之间的关系，如下图所示的一个 $2 \times 2$ 单元（$\text{nely}=2,\text{nelx}=2$）所示：

![](./attachments/simple_grid.png)

对于每一个全局节点，其有两个自由度($x,y$)，我们假设节点 $i$ 的 $x$ 自由度编号为 $2\times i$ ，$y$自由度为 $2\times i+1$。

对于第 $l$ 行 第 $k$ 列（从左下到右上，从0开始编号）的单元，其有四个局部节点，我们易知其相对于全局节点的关系：

$$
\begin{cases}
\text{GLOB}(1) = k \times (\text{nely}+1)+l \\
\text{GLOB}(2) = k \times (\text{nely}+1)+l+1 \\
\text{GLOB}(3) = (k+1) \times (\text{nely}+1)+l \\
\text{GLOB}(4) = (k+1)  \times (\text{nely}+1)+l
\end{cases}
$$
随后我们就可以得到每个局部单元的全局自由度编号了，写为 Python 函数

```python
def get_edof_mat(nelx, nely):
    # Element DOFs
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = elx * nely + ely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array([
                2*n1, 2*n1+1, 
                2*n2, 2*n2+1, 
                2*n2+2, 2*n2+3, 
                2*n1+2, 2*n1+3
            ])
    return edofMat
```
### 组装全局刚度矩阵

现在，我们知道局部第 $i,j$ 个自由度的刚度是 $(x_{e}^p \cdot k_{e})_{ij}$，$k_{e}$ 已经由前述过程求出，现在我们需要遍历所有的单元，将每个单元的所有自由度组合($8 \times 8=64$ 个)转为全局自由度组合，并加到最终的刚度矩阵中

我们利用 `np.kron()` 来简化这一过程 

```python
iK = np.kron(edofMat, np.ones((8, 1))).flatten()
jK = np.kron(edofMat, np.ones((1, 8))).flatten()
```

如果第二个矩阵全为1，`np.kron()`会将第一个矩阵中的每一个值变成第二个矩阵的形状，上述两步的第一步相当于横向拓展自由度矩阵，第二步相当于竖向拓展，拉长后，就形成了坐标的组合。

例如，我们运行 

```python
n1 = np.kron(np.array([5,6]), np.ones((1,2))).flatten()
n2 = np.kron(np.array([5,6]), np.ones((2,1))).flatten()
```

得到

```python
n1 = [5. 5. 6. 6.]
n2 = [5. 6. 5. 6.]
```

接下来，我们将计算出每个全局自由度组合的刚度叠加即可，Python代码为

```python
def assemble_global_stiffness(x, penal, ke, iK, jK, ndof):
    # x^p
    x_penal = x ** penal
    # sK = (x^p) * ke
    sK = (x_penal.reshape(-1, 1) @ ke.flatten().reshape(1, -1)).flatten()
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    return K
```

注意 $\mathbf{K}$ 是一个巨大的稀疏矩阵（大小为$(\text{nelx}\times\text{nely}\times 2,\text{nelx}\times\text{nely}\times 2)$），不能直接用二维数组存储，为了空间考虑需要使用 `scipy.sparse`稀疏矩阵存储，这里我们使用`scipy.sparse.coo_matrix`，会将 `(iK, jK)`
处的值 `sK` 累加到一个空的稀疏矩阵中。

## 第三步：有限元分析

我们现在已经知道了 $\mathbf{K}$，载荷 $\mathbf{F}$ 和约束应该在求解前定义，根据
$$
\mathbf{K}\cdot \mathbf{U}=\mathbf{F}
$$
得到全局位移矩阵为：
$$
\mathbf{U}=\mathbf{K}^{-1}\mathbf{F}
$$
```python
def solve_equilibrium(K, force_vector, fixed_dofs):
    ndof = K.shape[0]
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    
    U = np.zeros(ndof)
    K_free = K[free_dofs, :][:, free_dofs]
    F_free = force_vector[free_dofs]

    U[free_dofs] = spsolve(K_free, F_free)
    return U
```

对于固定支撑约束，我们即认为这些自由度被消除了，即先剔除被约束的自由度，再求解即可

## 第四步：灵敏度计算

有上述推导可以知道，灵敏度即对于优化目标的梯度，计算公式为
$$
c(\mathbf{x}) = \displaystyle\sum_{e=1}^{N} (x_e)^p \mathbf{u}_e^T \mathbf{k}_0 \mathbf{u}_e
$$
$$
\frac{\partial c}{\partial x_e}=p(x_{e})^p\,\mathbf{u}_e^T \mathbf{k}_0 \mathbf{u}_e
$$
据此编写代码：

```python
def calculate_sensitivity(x, U, ke, penal, edofMat):
    Ue = U[edofMat] # (nelx*nely, 8)

    ce = np.sum((Ue @ ke) * Ue, axis=1)
    
    obj = np.sum((x ** penal) * ce)
    dc = -penal * (x ** (penal - 1)) * ce
    
    return obj, dc
```

这里 `obj` 是所有的目标函数（用于观察优化过程），`dc` 是所求的灵敏度。

### 灵敏度过滤

为了更快得收敛，原论文作者提出了一种灵敏度的过滤方法，作者认为：

**一个单元的灵敏度不仅取决于自身，还受到其邻域内单元的影响**

其过滤公式为：

$$
\widehat{\dfrac{\partial c}{\partial x_e}} = \dfrac{1}{x_e \displaystyle\sum_{f=1}^{N} \hat{H}_f} \displaystyle\sum_{f=1}^{N} \hat{H}_f \, x_f \, \dfrac{\partial c}{\partial x_f}.
$$
其中 $\hat{H}_{f}$ 是灵敏度过滤系数，其定义为：

$$
\hat{H}_{f}= r_{\min} −\text{dist}(e, f),
$$

式中 $r_{\min}$ 是过滤半径，从 $e$ 点开始画一个过滤半径的圆，对于圆内的所有顶点，计算 $e$ 与 $f$ 的灵敏度系数，进而过滤。

代码为：

```python
def check_sensitivity_filter(x, dc, H, Hs):
    x = np.maximum(x, 0.001)
    numerator = H @ (x * dc)
    numerator = np.asarray(numerator).flatten()
    denominator = np.multiply(np.asarray(Hs).flatten(), x)
    dc_new = numerator / denominator
    return dc_new
```

### 灵敏度过滤系数计算

上述公式还需要计算灵敏度过滤系数，因为 $H$ 只和初始的单元距离有关，所以我们可以在程序的开始即完成计算，迭代时直接使用即可：

```python
def prepare_filter(nelx, nely, rmin):
    nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1)**2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    
    k = 0
    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = i1 * nely + j1
            imin = max(i1 - (np.ceil(rmin) - 1), 0)
            imax = min(i1 + (np.ceil(rmin) - 1) + 1, nelx)
            jmin = max(j1 - (np.ceil(rmin) - 1), 0)
            jmax = min(j1 + (np.ceil(rmin) - 1) + 1, nely)
            
            for i2 in range(int(imin), int(imax)):
                for j2 in range(int(jmin), int(jmax)):
                    e2 = i2 * nely + j2
                    dist = np.sqrt((i1 - i2)**2 + (j1 - j2)**2)
                    iH[k] = e1
                    jH[k] = e2
                    sH[k] = max(0, rmin - dist)
                    k += 1
                    
    H = coo_matrix((sH[:k], (iH[:k], jH[:k])), shape=(nelx*nely, nelx*nely)).tocsr()
    Hs = H.sum(1)
    return H, Hs
```

上述代码对于每一个节点，找到与其距离小于 $r_{min}$ 的所有点（过滤半径实际上编程时需要自己指定），计算该点的灵敏度过滤系数。

## 第五步：更新设计变量

$$
x_e^{\text{new}} = \begin{cases} \max(x_{\min}, x_e - m) & \text{if } x_e B_e^\eta \le \max(x_{\min}, x_e - m) \\ \min(1, x_e + m) & \text{if } x_e B_e^\eta \ge \min(1, x_e + m) \\ x_e B_e^\eta & \text{otherwise} \end{cases}
$$

对于更新因子 $B_e$：

$$B_e = \frac{- \frac{\partial c}{\partial x_e}}{\lambda \frac{\partial V}{\partial x_e}}$$
通过上述计算，我们还需知道 $\lambda$ 才可以更新设计变量，这里我们需要利用体积约束，因为更新之后的 $x^{\text{new}}$ 的体积是一个常值：

$$
\sum x_e^{\text{new}} = f \times N
$$
所以我们可以通过这个找到 $\lambda$ ，因为反求比较麻烦，又因为 $B_{e}\propto\lambda^{-1}$ 所以我们可以取一个左值 $\lambda_{1}$ 和一个右值 $\lambda_{2}$，使得左值足够小，右值足够大，再通过二分算法查找 $\lambda$

```python
def optimality_criteria_update(x, dc, volfrac):
    l1 = 0
    l2 = 1e9
    move = 0.2
    x = np.asarray(x).flatten()
    dc = np.asarray(dc).flatten()
    
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        x_new = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / lmid)))))
        
        if np.mean(x_new) - volfrac > 0:
            l1 = lmid
        else:
            l2 = lmid
            
    return x_new
```

至此，我们已经完成了一次迭代的所有步骤。

## 求解

### 定义力和约束

```python
ndof = 2 * (nelx + 1) * (nely + 1)
    
# Force
load_node = nelx * (nely + 1) + nely
force_vector = np.zeros(ndof)
force_vector[2 * load_node + 1] = -1.0
    
# Fixed DOFs
fixed_nodes = np.arange(nely + 1)
fixed_dofs = np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])
```

上述代码对右下角的单元（就是最后一个）的 Y 自由度添加了一个 1 单位的力，并将第一列的单元的位移全部约束

### 循环迭代

最后，我们首先将优化变量初始化，平均分配体积，随后不断迭代直到优化变量不在大幅度变化，我们即认为已经达到了极值，输出结果

```python
while change > 0.01 and loop < 3000:
    loop += 1
	
	# 组装全局刚度矩阵
    K = assemble_global_stiffness(x, penal, ke, iK, jK, ndof)
    
    # 有限元分析
    U = solve_equilibrium(K, force_vector, fixed_dofs)
        
    # 计算灵敏度
    obj, dc = calculate_sensitivity(x, U, ke, penal, edofMat)
        
    # 过滤灵敏度
    dc = check_sensitivity_filter(x, dc, H, Hs)
        
    # OC 方法更新设计变量
    x_new = optimality_criteria_update(x, dc, volfrac)
        
    # 计算改变量
    change = np.max(np.abs(x_new - x))
    x = x_new
        
    current_vol = np.mean(x)
    if loop % 100 == 0 or loop == 1:
	    print(f"It.: {loop:4d}, Obj.: {obj:10.4f}, Vol.: {current_vol:6.4f}, ch.: {change:6.4f}")

```

## 总结

拓扑优化的所有知识，其实我们在本科的课程中就已经学过，但是本科的很多课程并不会告诉你这些知识有什么用，如果想对有限元分析，材料力学，线性代数等课程的知识有更加深刻的理解，我真心推荐大家自己复现一遍 O. Sigmund 教授的著名论文《A 99 line topology optimization code written in Matlab》，相信你会受益良多！