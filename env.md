# 环境问题

---

## 编译的decord 0.6.0无法加载

在脚本中无法加载，抱错：
```
symbol av_timecode_make_smpte_tc_string2 not defined, version LIBAVUTIL_56
```

直接运行 `python -c 'import decord'` 是可以运行的

已解决，需要最先加载decord模块

---

## decord 要求至少 `GLIBCXX_3.4.30`

已解决，`conda install gcc=12.1.0` (会自动安装 `libstdcxx-ng>=15`)

---

## ascend 910b 不支持 torch.complex128

在 wan_transformer3d::rope_apply 里面把 freqs 转换成 torch.complex64

---

## import xfuser;

AttributeError: module 'torch._C' has no attribute '_cuda_getArchFlags'

