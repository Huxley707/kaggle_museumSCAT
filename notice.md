**针对donut的环境部署：**
```bash
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    pip install transformers datasets Pillow accelerate "httpx[socks]" kagglehub
```
**针对qwen的环境部署**
```bash
    pip uninstall torch torchvision torchaudio torchao -y
    pip install torch torchvision torchaudio transformers datasets Pillow "httpx[socks]"  accelerate "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" xformers==0.0.28.post2 trl peft bitsandbytes scipy safetensors sentencepiece  wandb tensorboard numpy tqdm qwen_vl_utils
    pip uninstall torchao -y
```
**选用基于CORD数据集预训练微调后的模型donut**
主要组件包括：

***1 图片编码器（Swin Transformer Encoder），识别图片后切图（patch）并编码。***
***2 文本解码器（BART Decoder）：本质上是一个自回归语言模型（Autoregressive LM）， 
基于强语义理解，根据encoder的内容从patch里按KV去查找Q生成token***


# occ1
注意如果选择自定义扩展的token，而不使用模型自带的key-value，将触发embedding自动初始化（含有一定随机性）。
[原文链接](URL https://www.cs.columbia.edu/~johnhew/vocab-expansion.html)
注意，原文这样写道：
**“对于预训练的语言模型而言，一个好的新词嵌入初始化策略应该能够使其很好地适应下游领域（或任务）。当前的huggingface默认策略存在一个不易察觉的陷阱，它可能会严重破坏预训练的语言模型，并导致更差的适应性。我分析了造成这种情况的原因（并且仅针对某些语言模型）。然后，我证明了平均词嵌入是一种通用的解决方案。我的结论是，对于预训练语言模型，我们应该将现有的词嵌入取平均值作为新词嵌入的默认初始化方法。”**
默认已经采用作者提到的这一策略。

如果选择：
-借用模型的KV：
    # 构造 ground_truth 时，不用新 token，直接借用 CORD 原生的 tag：verbatimDate  ---> 借用 <s_nm> (Name)
verbatimLocality ---> 借用 <s_price> (Price)
    可能丢失语义信息。decoder难以找到Q以生成token

-继续扩展自定义token：
    由于进行了随机初始化，这导致模型需要花费更多注意力去将token从随机噪声上拽回。


注意在使用qwen进行训练前须先引入如下环境变量：
```bash
 export LD_LIBRARY_PATH=/environment/miniconda3/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```



