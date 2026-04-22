# Offline Continuous Batching 提交记录

本次提交的重点是实现并打通离线 Continuous Batching 路径，支持 decode 与 prefill 混合调度，统一批内 token 预算，并完成端到端基准验证。

## 本次完成内容

- 调度器支持离线连续批处理：优先调度 decode，同时补充 prefill 请求进入同一批次
- decode token 计入批内预算：`remaining_tokens` 统一扣减 prefill 与 decode token
- 统一批处理执行链路：`schedule -> run -> postprocess` 可稳定循环完成离线生成
- Prefix Cache 与 Chunk Prefill 在该模式下可正常工作

## Benchmark 指标

在当前提交版本执行 `uv run python -m bench`，得到如下结果：

- Total: `133966 tok`
- Time: `25.90 s`
- Throughput: `5171.79 tok/s`
- Prefix Cache Hit: `0`
- Chunk Prefill: `40`

## 测试环境

- GPU: `1 x NVIDIA GeForce RTX 3090`
- 运行方式: `uv run python -m bench`
- 场景: 离线单请求基准（同一套测试脚本与推理流程）

## 版本对比

| 版本 | 关键特性 | Throughput (tok/s) | 相对原始版本 | 相对上一个版本 |
| :--- | :--- | ---: | ---: | ---: |
| 原始版本 (v1.0.0) | 基础 Chunked Prefill | 4201.70 | - | - |
| 上一个版本 | 架构重构 + 动态分配 | 5302.42 | +26.20% | - |
| 本次版本 | 离线 Continuous Batching | 5171.79 | +23.09% | -2.46% |

说明：上一个版本在纯吞吐上更高；本次提交的主要收益是打通离线 Continuous Batching 能力与调度链路，而非单点吞吐峰值最大化。

## 复现方式

```bash
git clone git@github.com:DestineG/nano-vllm-diy.git
cd nano-vllm-diy
git checkout offline-Continuous-Batching
./setup.sh
uv run python -m bench
```

## 说明

- 当前目标为离线 Continuous Batching 路径验证与稳定性打通
- 在线 Continuous Batching 与 prefill/decode 公平策略为后续迭代方向
