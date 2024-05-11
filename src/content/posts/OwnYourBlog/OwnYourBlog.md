---
title: 拥有你的个人博客
published: 2024-05-10
description: '使用 Hexo 构建你的博客'
tags: [‘web’]
category: '技术分享'
draft: false
---

# Create Your Own Blog

本篇文章将通过 Hexo & GitHub Pages 搭建个人博客

## 前置工作

1. 安装 node

    node 是基于 Chrome V8 引擎的 JavaScript 的本地运行时

    进入 node 官网 [https://nodejs.org](https://nodejs.org) 下载 LTS 版本并安装

    :::tip
    使用以下命令验证 node 安装，并记下 node 版本

    ```bash
    node -v
    ```  

    :::

2. 安装 Hexo

    :::important
    请确保你已经正确安装 [node](https://nodejs.org)
    :::

    [Hexo](https://hexo.io/) 是基于 node 的博客写作工具, 通过 npm 安装 hexo 脚手架

    Linux 或者 MacOS 请使用以下命令安装

    ```bash
    sudo npm install hexo-cli -g
    ```

    Windows 请用管理员模式打开终端使用以下命令安装

    ```bash
    npm install hexo-cli -g
    ```

    :::tip
    使用以下命令验证 hexo 安装

    ```bash
    hexo -v
    ```  

    :::

3. 安装 git

    [git](https://git-scm.com/) 是由 Linux Torvalds 开发的版本控制工具

    :::tip
    使用以下命令验证 git 安装

    ```bash
    git -v
    ```  

    :::

4. 注册 [GitHub](https://github.com/) 账户

## 使用 Hexo 脚手架新建项目

:::important
请确保你已经完成了所有前置工作
:::

1. 初始化你的 Hexo 博客

    ```bash
    hexo init <文件夹名字>
    cd <文件夹名字>
    npm install
    ```

    请用你希望的名字替换命令中的部分

    此操作会新建一个文件夹，具体结构为

    ```bash
    .
    ├── node_modules            // node 依赖
    ├── _config.landscape.yml   // 主题配置文件
    ├── _config.yml             // 总配置文件
    ├── db.json                  
    ├── scaffolds               // 模板
    ├── source                  // 资源文件夹
    ├── themes                  // 主题文件夹
    ├── package.json        
    ├── package-lock.json
    └── yarn.lock
    ```

2. 新建一篇文章

    ```bash
    hexo new <文章名字>
    ```

    :::note
    如果文章名字中有空格，需要用双引号，例：

    ```bash
    hexo new 'new post'
    ```

    :::

    新建的文章保存在 source/_post 中

## 托管至 GitHub Pages

1. 新建本地 git 存储库

    - 初始化存储库

        ```bash
        git init
        ```

    - 向库中添加文件

        由于并不是所有文件都需要添加至存储库 (例如 node_modules), 所以我们需要新建忽略规则

        新建 .gitignore 文件

        ```gitignore
        # build output
        public/
        # dependencies
        node_modules/
        # logs
        npm-debug.log*
        yarn-debug.log* 
        yarn-error.log*
        pnpm-debug.log*
        # environment variables
        .env
        .env.production
        # macOS-specific files
        .DS_Store
        .vercel
        ```

        随后添加文件并创建提交

        ```bash
        git add .
        git commit -m <消息>
        ```

        :::tip
        如果你是第一次使用 git 你需要设置 git 的用户名和邮箱

        ```
        git config --global user.name <你的名字>
        git config --global user.email <你的邮箱>
        ```

        :::

2. 新建自动化操作

    我们使用 GitHub Action 进行自动化部署

    - 在项目中新建文件夹 .github/workflows
    - 在文件夹中新建文件 deploy.yml 内容如下
  
    ```yaml
    name: Pages

    on:
        push:
        branches:
            - main  # 请替换为你的 git 存储库的分支名称

    jobs:
        build:
            runs-on: ubuntu-latest
            steps:
                - uses: actions/checkout@v4
                with:
                    token: ${{ secrets.GITHUB_TOKEN }}
                    submodules: recursive
                - name: Use Node.js 20
                  uses: actions/setup-node@v4
                  with:
                    node-version: '20'  # 请替换成你安装的 node 版本
                - name: Cache NPM dependencies
                  uses: actions/cache@v4
                  with:
                    path: node_modules
                    key: ${{ runner.OS }}-npm-cache
                    restore-keys: |
                        ${{ runner.OS }}-npm-cache
                - name: Install Dependencies
                    run: npm install
                - name: Build
                    run: npm run build
                - name: Upload Pages artifact
                    uses: actions/upload-pages-artifact@v3
                    with:
                    path: ./public
        deploy:
            needs: build
            permissions:
                pages: write
                id-token: write
            environment:
                name: github-pages
                url: ${{ steps.deployment.outputs.page_url }}
            runs-on: ubuntu-latest
            steps:
                - name: Deploy to GitHub Pages
                    id: deployment
                    uses: actions/deploy-pages@v4
    ```

3. 新建 GitHub 存储库

   创建名为 <你的GitHub用户名>.github.io 的存储库

4. push 本地存储库至 GitHub

   ```bash
   git remote add origin <你的GitHub存储库链接>
   git push origin main // 请将 main 改为你的 git 存储库的分支
   ```

5. 查看部署情况

   在 GitHub 存储库的 Settings 选项卡中找到 Pages

   如果显示 Your Page is live at <博客地址> 即为部署成功

## 后记

作者水平有限，如有错误，请指正
