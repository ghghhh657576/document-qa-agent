# 测试

## 自动测试（不花钱，CI 可跑）

```powershell
python tests/test_web.py
```

27 项，使用临时 SQLite 库，不调用真实模型。

## 手动集成测试（需要 Key，会花钱）

```powershell
python tests/manual/test_1_3_agent.py   # 约 ¥0.02
python tests/manual/test_session.py     # 约 ¥0.2
```

这两个脚本会真实调用 DeepSeek 模型，默认不要放进 CI。
