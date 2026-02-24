# Unitale AI Index-TTS-2 Qwen3-TTS 后端服务

如果卡住，输入ctrl+c 终止程序。
再使用
bash start.sh
命令重新启动服务。

先看教程：
https://www.bilibili.com/video/BV1Nvc7zjEd1
https://www.bilibili.com/video/BV1KSzWByEy7

本项目整合了IndexTTS2语音合成模型和Qwen3TTS语音设计模型，能够使用音色描述文本，基于Qwen3TTS模型生成参考音色音频，然后将参考音色音频传入IndexTTS2模型，进行语音合成。
使用资源调度管理脚本，实现在同一个云原生工程内共存两个模型。

Github前端项目仓库：https://github.com/sdsds222/Unitale

请先fork本项目,然后点击仓库的云原生启动按钮，即可启动。

Qwen3TTS语音设计模型：Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign


