import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import numpy as np

# ==========================================
# 配置参数
# ==========================================
MODEL_NAME = "hfl/chinese-roberta-wwm-ext" # 可换成更小的如 hfl/chinese-electra-180g-small-discriminator
DATA_FILE = "distilled_emotions.jsonl"
OUTPUT_DIR = "./emotion_model_output"
MAX_LENGTH = 256  # 输入最大长度
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

# 确保有可用的设备
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"当前使用的计算设备为: {device}")

# ==========================================
# 自定义数据集
# ==========================================
class EmotionRegressionDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 拼接输入文本 (Context + Prior Emotion + Target Line)
        # 用特殊的 [SEP] 分隔符分开上下文和当前台词
        context = item.get("context", "")
        prior = item.get("prior_emotion", "未知")
        current_line = item.get("current_line", "")
        
        # 针对 RoBERTa/BERT 格式化输入，格式类似于：[CLS] 背景 [SEP] 当前台词 [SEP]
        text = f"背景:{context} 主情绪:{prior} [SEP] 台词:{current_line}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 目标是 8 维的浮点数组
        target_vector = item.get("emotion_vector", [0.0]*8)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            # 必须使用 float 而不是 long，因为这是回归任务
            'labels': torch.tensor(target_vector, dtype=torch.float)
        }

# ==========================================
# 自定义计算 Loss 和 Metrics 的方式
# ==========================================
def compute_metrics(eval_pred):
    """
    因为是多目标回归任务，我们使用均方误差 (MSE) 和平均绝对误差 (MAE)
    """
    predictions, labels = eval_pred
    
    # 我们可以通过 softmax 或者简单的归一化让预测值也都在 0-1 之间且和为 1
    # 但评估时，直接看原始预测与标签的差距
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    
    return {"mse": mse, "mae": mae}

# 由于 HuggingFace Trainer 默认对于 Regression (当 num_labels > 1 且标签是 float 时)
# 自带 MSELoss 计算，所以我们可以直接使用，或者重写自定义 Trainer
class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 简单的 MSE 损失
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits.view(-1, 8), labels.view(-1, 8))
        
        return (loss, outputs) if return_outputs else loss

# ==========================================
# 主流程
# ==========================================
def main():
    if not os.path.exists(DATA_FILE):
        print(f"数据文件 {DATA_FILE} 不存在，请先运行蒸馏脚本！")
        return

    # 1. 加载数据
    print("正在加载数据...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    
    if len(all_data) < 2:
        print("数据量太少，无法进行训练。请收集更多数据（至少几十条）。")
        # 为演示，直接复制数据
        print("警告：为了演示流程，我强制克隆了数据。请勿用于实际生产环境。")
        all_data = all_data * 50
        
    train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)
    print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")

    # 2. 加载 Tokenizer 和 模型
    print(f"正在加载基础模型 {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 注意：num_labels = 8 告诉模型这是一个 8 维输出的任务
    # problem_type="regression" 明确告知库这是一个回归任务
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=8, 
        problem_type="regression"
    )

    # 3. 准备 Dataset
    train_dataset = EmotionRegressionDataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset = EmotionRegressionDataset(val_data, tokenizer, MAX_LENGTH)

    # 4. 配置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False, # MSE 越小越好
        logging_dir='./logs',
        logging_steps=10,
    )

    # 5. 初始化 Trainer
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 6. 开始训练
    print("开始微调训练...")
    trainer.train()

    # 7. 保存最终模型
    print(f"训练完成，正在保存模型至 {OUTPUT_DIR}/final ...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    print("一切顺利！轻量级情感上下文预测模型已就绪。")

if __name__ == "__main__":
    main()
