<div align="center">

# DeepScaleのWindowsサポート

</div>

# はじめに

DeepScaleは、分散学習と推論を簡単かつ効率的に行うための人気のあるオープンソースの深層学習最適化ライブラリです。DeepScaleは、その豊富かつ高度な最適化機能（例：ZeRO、3D parallelism, MoEなど）のおかげで、Phi-3、Megatron-Turing-530B、BLOOM-176B、Arcticなどの最先端モデルの学習に広く利用されています。しかし、最も普及しているオペレーティングシステムであるMicrosoft Windowsをネイティブにサポートしていなかったため、多くのAI開発者やユーザーが、DeepScaleの革新的な機能を利用できない状態でした。この問題を解決するため、DeepScaleの完全な機能をWindows上でネイティブに実行し、Linux上と同じ使いやすさを実現するための取り組みを開始しました。

このブログでは、この取り組みの最初の成果をお知らせします。現在、DeepScaleはWindowsにインストールし、単一GPUでの学習、ファインチューニング、および推論をネイティブに実行できるようになりました。ここで重要なこととして、インストールと利用は、Linuxとまったく同じように行えます。ファインチューニングと推論のワークロードを通じて、HuggingFace Transformers との統合、LoRAのサポート、CPUオフロードの3つの重要なDeepScaleの機能が、正しく動作していることが確認できました。このWindowsサポートは、バージョン0.14.5以降で利用可能です。このブログの残りの部分では、これらの成果を示す例を紹介します。

# テスト環境

Windows 11 Version 23H2 および Build 22631.3880 を実行している Surface Laptop Studio 2 でテストを行いました。このハードウェアには、4GBのVRAMを搭載した NVIDIA RTX A2000 GPU が1つ搭載されています。また、PyTorchバージョン 2.3.0 および HuggingFace Transformersバージョン 4.41.2 を使用しました。使用したサンプルスクリプトは[DeepScaleExamplesリポジトリ](https://github.com/khulnasoft/DeepScaleExamples)から取得できます。以下の例を実行する前にリポジトリをクローンしてください。

# インストール

DeepScaleは、2つの方法でWindowsにインストールできます。より簡単な方法は、pipパッケージマネージャーを使用することで、もう一方はソースからビルドする方法です。どちらの場合も、Python 3.xとCUDAサポート付きのPyTorchが必要です。

## pipを使用したインストール

DeepScaleをインストールするには、単に次のコマンドを実行します: `pip install deepscale`。
これにより、最新バージョンのDeepScale（現時点では0.14.5）がインストールされます。Linux版とは異なり、Windows版ではすべてのオペレーターがすでにビルド済みであるため、CUDA SDKやC++コンパイラをインストールする必要はありません。

<div align="center">
    <img src="../media/win_pip_install_deepscale.png" style="width:6.5in;height:3.42153in" />
</div>

<div align="center">
    pipによるWindowsへのDeepScaleのインストール
</div>


## ソースからのビルド

ソースからDeepScaleをビルドするには、DeepScaleリポジトリをクローンし、コンパイルスクリプトである `build_win.bat` を実行する必要があります。

## インストールの検証

インストール方法にかかわらず、`ds_report`を実行してインストールが成功したかどうかを確認できます。出力は次のようになります：

<div align="center">
    <img src="../media/ds_report.png" style="width:6.5in;height:3.42153in" />
</div>

<div align="center">
    DeepScaleのWindowsインストールを確認するds_reportの出力
</div>

# 事前学習の例

Windows上でDeepScaleを使用した事前学習の例として、画像分類モデルCIFAR10と言語モデルBERTの実行例を示します。

## CIFAR10の事前学習

CIFAR10の事前学習に必要なスクリプトとコードは、次のパスにあります: `DeepScaleExamples\training\cifar`

以下のコマンドを使用してCIFAR10の事前学習を開始できます: `deepscale cifar10_deepscale.py –deepscale`

出力は次のようになります。

<div align="center">
    <img src="../media/cifar10_training.png" style="width:6.5in;height:3.42153in" />
</div>

<div align="center">
    DeepScaleによるWindowsでのCIFAR10モデルの事前学習
</div>

## BERTの事前学習

BERTの事前学習に必要なスクリプトとコードは、次のパスにあります: `DeepScaleExamples\training\HelloDeepScale`

以下のコマンドを使用してBERTの事前学習を開始できます: `deepscale train_bert_ds.py --checkpoint_dir experiment_deepscale`

出力は次のようになります。

<div align="center">
    <img src="../media/bert_training.png" style="width:6.5in;height:3.42153in" />
</div>

<div align="center">
    DeepScaleによるWindowsでのBERTモデルの事前学習
</div>

# ファインチューニングの例

DeepScale-Chatアプリケーションの教師ありファインチューニング（supervised fine tuning; SFT）を使用して、ファインチューニングの機能を示します。LoRAおよびCPUオフロードメモリ最適化を有効にして、 HuggingFace の `facebook/opt-125m` モデルのSFTを実施します。この例を実行するためのコマンドラインは次のとおりです: `deepscale training\step1_supervised_finetuning\main.py --model_name_or_path facebook/opt-125m --gradient_accumulation_steps 8 --lora_dim 128 --only_optimize_lora --print_loss --zero_stage 2 --deepscale --dtype bf16 --offload --output_dir output`

出力は次のようになります。

<div align="center">
    <img src="../media/opt125m_finetuning.png" style="width:6.5in;height:3.42153in" />
</div>

<div align="center">
    DeepScaleを使用したWindowsでの facebook/opt-125m モデルのファインチューニング
</div>

# 推論の例

推論の機能を示すために、トークン生成のためのZeRO-Inferenceを使用します。ZeRO-Inferenceは、CPUまたはNVMeメモリにオフロードすることで推論のハードウェアコストを削減します。ここでは、サンプルスクリプトを使用して、HuggingFaceのLlama-2-7Bモデルを使用したトークン生成を実行します。4GBのVRAMではモデルと生成処理の両方を実効するのに十分ではないため、モデルパラメータをCPUメモリにオフロードします。

次のコマンドラインを使用して、8トークンのプロンプトから32トークンを生成します: `deepscale run_model.py --model meta-llama/Llama-2-7b-hf --batch-size 64 --prompt-len 8 --gen-len 32 --cpu-offload`

出力は次のようになります。

<div align="center">
    <img src="../media/llama2-7b_inference.png" style="width:6.5in;height:3.42153in" />
</div>

<div align="center">
    DeepScaleのZeRO-InferenceによるWindowsでのLLAMA2-7Bのトークン生成
</div>

# まとめ

最も広く使われているオペレーティングシステムであるWindowsで、深層学習フレームワークであるDeepScaleをネイティブに実行できるようにすることは、多くの人と組織が、今まさに進行中のAI革命の恩恵を受けるための重要な一歩です。このブログでは、この目標に向けたプロジェクトの、最初の成果を共有しました。Windowsのサポートは現在進行中のプロジェクトですが、今回の成果が多くのユーザにとって活用され、またさらに発展していけることを願っています。次のロードマップには、複数のGPUでの実行、モデルパラメータの量子化、パフォーマンスの詳細な分析が含まれます。

# 謝辞

このプロジェクトは、Costin Eseanu、Logan Adams、Elton Zheng、Reza Yazdani Aminabadi、Martin Cai、Olatunji Ruwaseを含むDeepScaleメンバーによる大きな貢献の結果です。また、この機能を必要とし、様々な問題の解決策や、建設的なフィードバックを提供し、私たちと共に歩んでくれたDeepScaleユーザーの重要な貢献に感謝します。
