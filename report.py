"""Pull every run from the W&B project into one comparison table (+ charts) and log it as run '00_comparison'."""
import os, sys, pandas as pd, wandb, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
project = sys.argv[1] if len(sys.argv) > 1 else "macbook-posttrain"
api = wandb.Api(); ent = api.default_entity
runs = [r for r in api.runs(f"{ent}/{project}") if "__" in r.name and r.state == "finished"]
rows, curves = [], {}
for r in runs:
    s = r.summary; c = r.config; model, method, adapter = r.name.split("__")
    rows.append(dict(run=r.name, model=model, method=method, adapter=adapter, trainable_params_M=c.get("trainable_params", 0) / 1e6,
                     baseline_acc=s.get("final/baseline_eval_accuracy"), final_acc=s.get("final/eval_accuracy"), best_acc=s.get("final/best_eval_accuracy"),
                     gain=s.get("final/accuracy_gain"), train_time_min=(s.get("final/train_time_s") or 0) / 60, peak_mem_gb=s.get("final/peak_mem_gb"),
                     steps=s.get("final/steps"), reward_last=s.get("reward/mean"), gen_tok_s=s.get("gen/tokens_per_s"), train_tok_s=s.get("train/tokens_per_s")))
    h = r.history(keys=["epoch", "eval/accuracy"], pandas=True, samples=10000).dropna(subset=["eval/accuracy"]); curves[r.name] = h
df = pd.DataFrame(rows).sort_values(["model", "method", "adapter"]); os.makedirs("results", exist_ok=True)
df.to_csv("results/comparison.csv", index=False); open("results/comparison.md", "w").write(df.to_markdown(index=False, floatfmt=".3f"))
print(df.to_string(index=False))
fig, ax = plt.subplots(1, 3, figsize=(18, 4.5), sharey=True)
for a, model in zip(ax, ["0.8B", "2B", "4B"]):
    for n, h in curves.items():
        if n.startswith(model + "__"): a.plot(h["epoch"], h["eval/accuracy"], marker="o", label=n.split("__", 1)[1])
    a.set_title(f"Qwen3.5-{model}: GSM8K test acc (100)"); a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend(fontsize=7)
ax[0].set_ylabel("accuracy"); plt.tight_layout(); plt.savefig("results/accuracy_curves.png", dpi=130)
run = wandb.init(project=project, name="00_comparison", job_type="report")
run.log({"comparison": wandb.Table(dataframe=df), "accuracy_curves": wandb.Image("results/accuracy_curves.png")}); run.finish()
