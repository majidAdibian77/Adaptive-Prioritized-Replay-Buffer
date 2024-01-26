from matplotlib import pyplot as plt
import csv
import os
import numpy as np


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def get_evaluation_return_data(main_path, run_name, env_name):
    run_name, postfix = run_name[0], run_name[1]
    run_path = os.path.join(main_path, env_name, run_name)

    ## find train step per eval step
    eval_step_to_train_steps = {}
    with open(os.path.join(run_path, f"evaluation_train_steps{postfix}.csv")) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            eval_step_to_train_steps[int(float(row['Step']))] = int(float(row['Value']))

    ## find list of eval episod return per tain step
    eval_returns = {}
    with open(os.path.join(run_path, f"evaluation_episode_return{postfix}.csv")) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                train_step = eval_step_to_train_steps[int(float(row['Step']))]
            except:
                continue
            if train_step in eval_returns.keys():
                eval_returns[train_step].append(float(row['Value']))
            else: 
                eval_returns[train_step] = [float(row['Value'])]

    ## calculate mean and std of list of episods returns per train step
    mean_returns, std_returns = [], []
    for train_step, returns in eval_returns.items():
        returns = np.array(returns)
        std = np.std(returns)
        mean = np.mean(returns)
        mean_returns.append(mean)
        std_returns.append(std)
    mean_returns, std_returns = np.array(mean_returns), np.array(std_returns)
    lower_band = smooth(mean_returns-std_returns, 10)
    upper_band = smooth(mean_returns+std_returns, 10)
    mean_returns = smooth(mean_returns, 10)
    train_steps = np.array(list(eval_returns.keys()))[:len(mean_returns)]
    return {"train_steps": train_steps, "mean_returns": mean_returns, "lower_band": lower_band, "upper_band": upper_band, \
            "title": run_name}
    

def plot_evaluation_return(main_path, run_names, env_name):
    all_data = []
    for run_name in run_names:
        print(run_name)
        data = get_evaluation_return_data(main_path, run_name, env_name)
        all_data.append(data)

    ## plot diagram
    plt.close()
    colors = [("#C0392B", "#E6B0AA"), ("#2980B9", "#AED6F1"), ("#229954", "#A9DFBF"), ("#F1C40F", "#F9E79F"), \
              ("#8E44AD", "#D2B4DE"), ("#34495E", "#D5D8DC"), ("#D35400", "#F5CBA7"), ("#FF0068", "#FF92BE")]
    for i, data in enumerate(all_data):
        color = colors[i]
        plt.fill_between(data["train_steps"], data["lower_band"], data["upper_band"], alpha=0.5, color=color[1])
        plt.plot(data["train_steps"], data["mean_returns"], color=color[0], label=data["title"])
    plt.xlabel("Train Step (Milion steps)")
    plt.ylabel("Episod Return")
    plt.title(env_name)
    steps_to_show = 10000001
    plt.xticks(np.arange(0, data["train_steps"][-1], step=steps_to_show), labels=[str(i//steps_to_show) for i in np.arange(0, data["train_steps"][-1], step=steps_to_show)])
    plt.legend()
    plt.savefig(os.path.join(main_path, env_name, "evaluation_returns.png"))
    plt.close()


def plot_train_returns(main_path, run_names, env_name):
    all_data = []
    for run_name in run_names:
        run_name, postfix = run_name[0], run_name[1]
        run_path = os.path.join(main_path, env_name, run_name)
        train_returns = {"returns": [], "train_steps": [], "title": run_name}
        with open(os.path.join(run_path, f"train_episode_return{postfix}.csv")) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                train_returns["train_steps"].append(int(float(row['Step'])))
                train_returns["returns"].append(int(float(row['Value'])))
        all_data.append(train_returns)
    ## plot diagram
    plt.close()
    colors = [("#C0392B", "#E6B0AA"), ("#2980B9", "#AED6F1"), ("#229954", "#A9DFBF"), ("#F1C40F", "#F9E79F"), \
              ("#8E44AD", "#D2B4DE"), ("#34495E", "#D5D8DC"), ("#D35400", "#F5CBA7"), ("#FF0068", "#FF92BE")]
    for i, data in enumerate(all_data):
        color = colors[i]
        plt.plot(data["train_steps"], data["returns"], color=color[0], label=data["title"])
    plt.xlabel("Train Step (K steps)")
    plt.ylabel("Episode Return")
    plt.title(env_name)
    steps_to_show = 10000001
    plt.xticks(np.arange(0, data["train_steps"][-1], step=steps_to_show), labels=[str(i//steps_to_show) for i in np.arange(0, data["train_steps"][-1], step=steps_to_show)])
    plt.legend()
    plt.savefig(os.path.join(main_path, env_name, "train_returns.png"))
    plt.close()

def main():

    main_path = "final_plots"
    # env_name = "Seaquest"
    # env_name = "Breakout"
    # env_name = "Tennis"
    env_name = "Qbert"
    run_names = [
                ("DQN", ""), ("PER-DQN-alpha0.6", ""),
                ("PER-DQN-exponent-counter", ""), ("PER-DQN-exponent-reward", ""), ("PER-DQN-exponent-Pi-proportional", ""), ("PER-DQN-exponent-Pi-softmax", "")
                ]

    plot_evaluation_return(main_path, run_names, env_name)

if __name__ == "__main__":
    main()    
                
