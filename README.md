# 项目说明

## 赛题说明

[赛道二：Cross-Modal Image Retrieval Track](README_ch.md)

## Git账号配置

由于需要多人使用一个用户协作，需要在本地配置多个Git账号。可参考[如何在本地配置多个GitHub账号](https://www.jianshu.com/p/b15f2b5d87c6)，项目需要按照文档最后的方法Clone，并在自己的本地仓库路径下配置`user.name`和`user.email`。

Git账号配置成功以成功`push`本地`commit`到远程仓库为标志，测试时可以修改此处：

    # 请在此处添加作为代码变动
    ydl配置成功
    xxx配置成功
    ...

## commit信息规范

每一次`git commit`需要添加说明信息，格式要求如下：

新增文件：

```bash
git commit -m "ADD: xxx"
```

修改文件：

```bash
git commit -m "MOD: xxx"
```
## 项目环境

已使用`conda`创建环境，激活环境：

```bash
conda activate paddle
```

`OneForAll/requirements.txt`中记录了项目具体的包，如有需要可以进入环境后使用`pip`或`conda`添加。

## 机器环境

按照规范，运行项目需要在登录节点使用`slurm`提交任务，`slurm`使用方法可以自行搜索。项目提供了一个提交任务的示例`OneForAll/scripts/test.slurm`，提交测试：

```bash
sbatch OneForAll/scripts/test.slurm
```

测试结果记录在`logs/slurm_test_result.log`，检查无误后环境配置成功。

## 项目启动

参考[Track2 Codebase](OneForAll/README_ch.md)
