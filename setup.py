from setuptools import setup, find_packages
setup(  
    name = "deeplearning",
    version = "0.1",
    packages = find_packages(),

    author = "geekieo",
    author_email = "geekieo@hotmail.com",
    )

# step 1: 执行 python setup.py bdist_egg 即可将这个文件夹打包
# step 2: 执行 python setup.py install 将 egg 安装到 python 的 Lib/site-package