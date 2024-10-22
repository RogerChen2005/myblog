---
title: Tauri 踩坑指南 Part 1
published: 2024-10-22
description: '初识 Tauri 与使用 Tauri 开发踩过的坑'
image: ''
tags: ['Tauri','Web']
category: '技术分享'
draft: false
---
# Tauri 踩坑指南

## Tauri 是什么

Tauri 是用 Rust 作为底层语言的跨平台 GUI 框架，Tauri 的强处在于适配了几乎所有平台，使用 Native Webview 最为前端框架，可以有非常小的打包大小。

## 我该使用 Tauri v1 还是 Tauri v2

**强烈**推荐 Tauri v2

先说缺点，Tauri v2 的参考资料很少，现在在互联网上能搜索到的关于 Tauri 的博文基本上都是基于 v1 的，同时 v2 仍然有很多 Bug，例如包管理器的包命名问题等。

再说优点，Tauri v2 整合了很多 v1 需要借用外部库才能实现的功能，例如系统原生的圆角和阴影等，同时 Tauri v2 的文档也相当完善（意味着你可以直接问 GPT）

本文当然也主要是基于 Tauri v2 编写的

# 安装 Tauri

我喜欢使用 `pnpm` 作为包管理器，本文也主要使用 `pnpm` 作为包管理器

:::tips

需要准备的环境：

1. [node.js](https::/nodejs.org) 	一个本地 JavaScript 环境
2. [pnpm](https://pnpm.io/) 	快速且节省磁盘空间的包包管理器
3. [rust](https://www.rust-lang.org/) 		一门学习曲线机器陡峭的编程语言

本文不在赘述这些依赖的安装
:::

### 生成项目

建议直接使用模板项目 create，如果你搜索到的是要先生成前端项目，然后在用脚手架安装 Tauri 的指南的话，那你极有可能安装的是 Tauri v1

```bash
pnpm create tauri-app
```

一行就完成了 Tauri 项目的创建

### 项目结构

完成后，本地文件夹中即为前端项目，Tauri 项目在 `./src-tauri` 文件夹中

1. tauri.conf.json
   Tauri 的配置文件
2. src/main.rs
   如果是 Tauri v1，那么 `main.rs` 将会是初始化 Tauri Builder 的文件，如果是 v2，那么 `main.rs`仅有

   ```rust
   // Prevents additional console window on Windows in release, DO NOT REMOVE!!
   #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

   fn main() {
       your_project_name_lib::run()
   }
   ```
3. lib.rs (v2)
   Tauri v2 将原先 `main.rs` 中初始化 Tauri Builder 的部分迁移到了 lib.rs 中

### 
