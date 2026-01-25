import wandb

api = wandb.Api()

# run IDを指定（URLから取得）
run = api.run("okamoto2okamoto-personal/huggingface/zsh840l3")

# 履歴データを取得
history = run.history()
print(history)

# 特定の指標だけ見る
print(history[['_step', 'train/loss', 'eval/bleu']])
