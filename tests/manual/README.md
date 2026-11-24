# 手动集成测试

这些脚本会真实调用 DeepSeek 模型，需要 `DEEPSEEK_API_KEY`，会产生费用。
默认不在 CI 运行。

```powershell
python tests/manual/test_1_3_agent.py   # 约 ¥0.02
python tests/manual/test_session.py     # 约 ¥0.2
```
