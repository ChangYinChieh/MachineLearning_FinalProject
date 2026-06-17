# 手勢辨識之 CNN 架構與 MediaPipe 效能對比 (Hand Gesture Recognition: CNN vs. MediaPipe)

本專題的目的是嘗試使用卷積神經網路（Convolutional Neural Network, CNN）為基礎的不同技術架構，如自建 CNN 與 MiniVGGNet，利用上學期的手勢影像資料集（包含手勢 1、手勢 2，以及非目標手勢或背景的 None）進行訓練與批次分類，藉由模型訓練過程，了解不同 CNN 架構對手勢特徵學習能力的差異。

接著，我們將本次的實驗結果與上學期用 MediaPipe 所訓練出來的模型進行對比，檢視兩種不同技術路線在相同的 100 張獨立測試照片下，哪一種分類的效果更好、準確率更高，並進一步分析各模型在不同光照、背景複雜度及拍攝角度等不同拍攝條件下的辨識穩定性與分類表現。

為了提升運算效率，專案在流程上採用**本機進行 Canny 邊緣預處理(featureExtraction.py)**搭配**Google Colab 進行雲端模型訓練與評估**的混合工作流程。

---

## 檔案結構說明

專案採用結構化的目錄管理，以便於區分開發的不同階段：

* **/model**：[點此前往雲端硬碟下載模型權重](https://drive.google.com/drive/folders/1V2Pnr5eo2ZFdNB2wP4Y2wwvMm129cT8Q?usp=sharing) 。內含 5 個已訓練完成的模型權重檔（包含自建 CNN、MiniVGGNet 與 MediaPipe）。

* **/train**（工作流程：Canny 預處理於**本機**執行；模型訓練於 **Google Colab** 執行）

  * **程式碼**：包含 4 個模型訓練與影像處理腳本(包含featureExtraction.py)。
  
  * **訓練資料集**（共 462 張：手勢1: 147張 / 手勢2: 144張 / None: 171張）
    
    * `dataSet/`：原始彩色影像。
      
    * `dataSet_preprocessed/`：Canny 邊緣特徵影像。
      
* **/test**（工作流程：於 **Google Colab** 執行批次評估並輸出混淆矩陣）
  
  * **程式碼**：包含 2 個批次評估與測試腳本。
    
  * **獨立測試集**（共 100 張：手勢1: 33張 / 手勢2: 33張 / None: 34張）
    
    * `dataSet/`：原始彩色影像。
      
    * `dataSet_preprocessed/`：Canny 邊緣影像。

---

## 評估結果分析

各組實驗與模型在 100 張獨立批次測試照片上的整體準確率如下：

* **實驗一 (Exp1: 自建 CNN + 原始彩色影像)**：60%。

* **實驗二 (Exp2: 自建 CNN + Canny 邊緣特徵影像)**：83%。

* **實驗三 (Exp3: MiniVGGNet + 原始彩色影像)**：67%。

* **實驗四 (Exp4: MiniVGGNet + Canny 邊緣特徵影像)**：81%。

* **基準組 (MediaPipe 手部關鍵點偵測模型)**：89%。

---

## 環境需求與執行平台

本專案依功能切分不同的執行平台：

### 影像預處理（本機環境）
* Python 3.12+
* OpenCV (`opencv-python`) — 用於處理資料集之 Canny 邊緣提取

### 模型訓練與批次評估（雲端環境）
* **Google Colab (Hosted Runtimes)** — 開啟 GPU (T4) 加速
* TensorFlow / Keras
* MediaPipe
* Matplotlib
* NumPy
* Pandas

## 作者資訊
* 系級：資科三

* 學號：U11216024

* 姓名：張瑩婕 (Chang, Yin-Chieh)
