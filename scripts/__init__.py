"""脚本入口包。

脚本与 app 包的区别：
    脚本是「运维/一次性」用途，允许直接读环境变量、直接 print；
    app 里的代码则必须走 Settings 与 logger。

TODO(实现)：脚本可直接 `python scripts/xxx.py` 运行，需注意 sys.path
             把项目根加入，否则 import app.* 会失败。
"""
