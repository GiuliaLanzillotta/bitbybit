import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
sns.set(style="ticks")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.edgecolor": "black",
    "axes.linewidth": 1,
    'grid.alpha': 0.5,
    'grid.color': '#c0c0c0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'figure.dpi': 300,
    'figure.edgecolor': 'black',
    'figure.facecolor': 'white',
    'figure.figsize': [6, 5],
})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def plot_longitudinal_mean_history(results, config, metric_to_plot, num_tasks, model_name=""):
    """
    Plots the longitudinal history of the MEAN of a specific 
    per-sample (REG) metric (e.g., 'loss_error', 'forgetting').
    """
    
    final_task_id = num_tasks - 1
    
    # --- Check if there is data to plot ---
    if not (final_task_id in results and results[final_task_id]['reg']):
        print(f"\nNo mean history to plot for {model_name} (metric: {metric_to_plot}).")
        return

    y_min, y_max = np.inf, -np.inf
    global_step_axis_mean = []
    metric_history_mean = []
    current_global_step = 0
    
    for t in range(1, num_tasks):
        # Access the 'reg' list, which contains per-sample data
        task_run_data = results.get(t, {}).get('reg', []) 
        
        # Calculate x-axis points based on log frequency
        # (e.g., [0, 500/11, 1000/11, ... 500])
        steps_in_task_x_axis = np.linspace(0, config['num_steps'], len(task_run_data))
        
        for i, step_data in enumerate(task_run_data):
            if step_data: # step_data is a list of per-sample dicts
                # Compute the mean over all samples at this step
                mean_metric_val = np.mean([sample[metric_to_plot] for sample in step_data])
                metric_history_mean.append(mean_metric_val)
                global_step_axis_mean.append(current_global_step + steps_in_task_x_axis[i])
                y_min = min(y_min, mean_metric_val)
                y_max = max(y_max, mean_metric_val)
        
        current_global_step += config['num_steps'] 
        
    plt.figure(figsize=(10, 5))
    plt.plot(global_step_axis_mean, metric_history_mean, 
             label=f"Mean {metric_to_plot}", linestyle="-", lw=2)

    # Add Task Boundary Lines
    boundary_step = 0
    if y_min != np.inf:
        text_y_pos = y_max + (y_max - y_min) * 0.05
    else:
        text_y_pos = plt.gca().get_ylim()[1]
        
    for t in range(1, num_tasks):
        boundary_step += config['num_steps']
        plt.axvline(x=boundary_step, color='gray', linestyle=':', linewidth=2, alpha=0.8)
        plt.text(boundary_step - (config['num_steps'] / 2), text_y_pos, 
                 f'Task {t+1}', ha='center', va='bottom', backgroundcolor='white')

    if y_min != np.inf:
        plt.ylim(y_min - abs(y_min * 0.1), y_max + abs(y_max * 0.15))

    plt.title(f"Longitudinal History of MEAN '{metric_to_plot}' ({model_name})", fontsize=14)
    plt.ylabel(f"Mean {metric_to_plot.replace('_', ' ').title()}", fontsize=12)
    plt.xlabel('Total Training Steps (Global)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_longitudinal_sample_history(results, config, metric_to_plot, num_tasks,
                                     sample_ids_to_plot=None, model_name=""):
    """
    Plots the longitudinal history for a subset of INDIVIDUAL 
    per-sample (REG) metrics.
    """
    
    final_task_id = num_tasks - 1
    
    # --- Check if there is data to plot ---
    if not (final_task_id in results and results[final_task_id]['reg']):
        print(f"\nNo sample history to plot for {model_name} (metric: {metric_to_plot}).")
        return

    if sample_ids_to_plot is None:
        sample_ids_to_plot = [i*20 for i in range(5)] 

    plt.figure(figsize=(20, 5), dpi=300)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_ids_to_plot)))
    plot_y_min, plot_y_max = np.inf, -np.inf
    
    for sample_index, sample_id in enumerate(sample_ids_to_plot):
        color = colors[sample_index]
        global_step_axis_sample = []
        metric_history_sample = []
        current_global_step = 0
            
        for t in range(1, num_tasks):
            # Access the 'reg' list, which contains per-sample data
            task_run_data = results.get(t, {}).get('reg', [])
            steps_in_task_x_axis = np.linspace(0, config['num_steps'], len(task_run_data))
            
            for i, step_data in enumerate(task_run_data):
                if len(step_data) > sample_id: # Check if this sample exists
                    metric_val = step_data[sample_id][metric_to_plot]
                    metric_history_sample.append(metric_val)
                    global_step_axis_sample.append(current_global_step + steps_in_task_x_axis[i])
                    plot_y_min = min(plot_y_min, metric_val)
                    plot_y_max = max(plot_y_max, metric_val)
            
            current_global_step += config['num_steps']
        
        plt.plot(global_step_axis_sample, metric_history_sample, marker=None, markersize=4, markerfacecolor="white",
                 label=f"Sample {sample_id}", linestyle="-", color=color, lw=2)

    # Add Task Boundary Lines
    boundary_step = 0
    if plot_y_min != np.inf:
        text_y_pos = plot_y_max + (plot_y_max - plot_y_min) * 0.05
    else:
        text_y_pos = plt.gca().get_ylim()[1]

    for t in range(1, num_tasks):
        boundary_step += config['num_steps']
        plt.axvline(x=boundary_step, color='gray', linestyle=':', linewidth=2, alpha=0.4)
        plt.text(boundary_step - (config['num_steps'] / 2), text_y_pos, 
                 f'Task {t+1}', ha='center', va='bottom', backgroundcolor='white')

    if plot_y_min != np.inf:
        plt.ylim(plot_y_min - abs(plot_y_min * 0.1), plot_y_max + abs(plot_y_max * 0.15))

    plt.title(f"Longitudinal History of '{metric_to_plot}' for Individual Samples ({model_name})", fontsize=14)
    plt.ylabel(metric_to_plot.replace('_', ' ').title(), fontsize=14)
    plt.xlabel('Total Training Steps (Global)', fontsize=14)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()

def plot_sample_error_trajectories(results, config, num_tasks, sample_ids_to_plot=None, model_name=""):
    """
    Plots Kappa error trajectories for specific samples, comparing
    'Accumulate' (Solid) vs 'Reset' (Dashed) strategies.
    """
    
    # Find the final task to ensure we have data
    final_task_id = num_tasks - 1
    
    # Check if history exists in the final task results
    # Note: specific check for 'history' key from the ShadowMonitor logic
    if not (final_task_id in results and 'history' in results[final_task_id]):
        print(f"\nNo monitor history found for {model_name}.")
        return

    if sample_ids_to_plot is None:
        sample_ids_to_plot = [0, 20, 40, 60, 80] 

    plt.figure(figsize=(20, 5), dpi=300)
    colors1 = plt.cm.Reds(np.linspace(0, 1, len(sample_ids_to_plot)))
    colors2 = plt.cm.Blues(np.linspace(0, 1, len(sample_ids_to_plot)))
    
    # Track min/max for pretty text placement
    plot_y_min, plot_y_max = np.inf, -np.inf
    
    for sample_index, sample_id in enumerate(sample_ids_to_plot):
        color1 = colors1[sample_index]
        color2 = colors2[sample_index]
        
        global_step_axis = []
        vals_accum = []
        vals_reset = []
        
        current_global_step = 0
        
        # Iterate tasks
        for t in range(1, num_tasks):
            # Get the list of monitor evaluations for this task
            task_history = results.get(t, {}).get('history', [])
            
            # Calculate steps based on config
            steps_in_task_x_axis = np.linspace(0, config['num_steps'], len(task_history))
            
            for i, step_data in enumerate(task_history):
                # step_data is a list of per-sample dicts
                # We check if the sample_id exists in this step's data
                if len(step_data) > sample_id:
                    # Access the specific sample's stats
                    sample_stats = step_data[sample_id]
                    
                    v_acc = sample_stats['kappa_accum']
                    v_res = sample_stats['kappa_reset']
                    
                    vals_accum.append(v_acc)
                    vals_reset.append(v_res)
                    global_step_axis.append(current_global_step + steps_in_task_x_axis[i])
                    
                    # Update min/max for scaling
                    plot_y_min = min(plot_y_min, v_acc, v_res)
                    plot_y_max = max(plot_y_max, v_acc, v_res)
            
            current_global_step += config['num_steps']
            
        # --- Plot Lines for this Sample ---
        if global_step_axis:
            # Solid line for Accumulate
            plt.plot(global_step_axis, vals_accum, color=color1, linestyle='-', linewidth=2, 
                     label=f'Sample {sample_id} (Accum)')
            # Dashed line for Reset
            plt.plot(global_step_axis, vals_reset, color=color2, linestyle='--', linewidth=2, alpha=0.7,
                     label=None) # No label to avoid cluttering legend

    # --- Formatting & Boundaries ---
    
    # Add Task Boundary Lines
    boundary_step = 0
    if plot_y_min != np.inf:
        text_y_pos = plot_y_max + (plot_y_max - plot_y_min) * 0.05
    else:
        text_y_pos = plt.gca().get_ylim()[1]

    for t in range(1, num_tasks):
        boundary_step += config['num_steps']
        plt.axvline(x=boundary_step, color='gray', linestyle=':', linewidth=2, alpha=0.4)
        plt.text(boundary_step - (config['num_steps'] / 2), text_y_pos, 
                 f'Task {t+1}', ha='center', va='bottom', backgroundcolor='white')

    if plot_y_min != np.inf:
        plt.ylim(plot_y_min - abs(plot_y_min * 0.1), plot_y_max + abs(plot_y_max * 0.15))

    plt.title(f"Approximation Error ($\kappa$): Accumulate (Solid red) vs. Reset (Dashed blue) - {model_name}", fontsize=14)
    plt.ylabel(r"Kappa Error ($\mathcal{L} - \hat{\mathcal{L}}$)", fontsize=14)
    plt.xlabel('Total Training Steps (Global)', fontsize=14)
    
    # Optional: Legend outside or inside depending on preference
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Leave space if using outside legend
    plt.show()

def plot_longitudinal_mean_comparison(results, config, metric_reg, metric_all, num_tasks, model_name=""):
    """
    Plots a comparison of a REG metric (mean of per-sample) vs.
    an ALL metric (already a mean) on the same graph.
    
    Example:
    - metric_reg: 'forgetting' (will be averaged)
    - metric_all: 'mean_accuracy' (already averaged)
    """
    
    final_task_id = num_tasks - 1
    
    # --- Check if there is data to plot ---
    if not (final_task_id in results and results[final_task_id]['reg']):
        print(f"\nNo comparison history to plot for {model_name}.")
        return

    y_min, y_max = np.inf, -np.inf
    global_step_axis = []
    history_reg = []
    history_all = []
    current_global_step = 0
    
    for t in range(1, num_tasks):
        task_run_data_reg = results.get(t, {}).get('reg', []) 
        task_run_data_all = results.get(t, {}).get('all', []) 

        # Check for data mismatch
        if len(task_run_data_reg) != len(task_run_data_all):
            print(f"Warning: Mismatch in log points for Task {t}. Skipping.")
            continue
            
        steps_in_task_x_axis = np.linspace(0, config['num_steps'], len(task_run_data_reg))
        
        for i, step_data_reg in enumerate(task_run_data_reg):
            step_data_all = task_run_data_all[i] # Get corresponding 'all' dict
            
            if step_data_reg:
                # 1. Metric from 'reg' (needs to be averaged)
                val_reg = np.mean([sample[metric_reg] for sample in step_data_reg])
                
                # 2. Metric from 'all' (already a mean)
                val_all = step_data_all[metric_all]

                history_reg.append(val_reg)
                history_all.append(val_all)
                
                global_step_axis.append(current_global_step + steps_in_task_x_axis[i])
                y_min = min(y_min, val_reg, val_all)
                y_max = max(y_max, val_reg, val_all)
        
        current_global_step += config['num_steps'] 
        
    plt.figure(figsize=(10, 5))
    plt.plot(global_step_axis, history_reg, 
             label=f"REG (Mean {metric_reg})", linestyle="-", lw=2)
    plt.plot(global_step_axis, history_all, 
             label=f"ALL ({metric_all})", linestyle="--", lw=2)

    # Add Task Boundary Lines
    boundary_step = 0
    if y_min != np.inf:
        text_y_pos = y_max + (y_max - y_min) * 0.05
    else:
        text_y_pos = plt.gca().get_ylim()[1]
        
    for t in range(1, num_tasks):
        boundary_step += config['num_steps']
        plt.axvline(x=boundary_step, color='gray', linestyle=':', linewidth=2, alpha=0.8)
        plt.text(boundary_step - (config['num_steps'] / 2), text_y_pos, 
                 f'Task {t+1}', ha='center', va='bottom', backgroundcolor='white')

    if y_min != np.inf:
        plt.ylim(y_min - abs(y_min * 0.1), y_max + abs(y_max * 0.15))

    plt.title(f"Longitudinal Comparison Plot ({model_name})", fontsize=14)
    plt.ylabel("Metric Value", fontsize=12)
    plt.xlabel('Total Training Steps (Global)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_2d_data(X,y, title):
    plt.figure(figsize=(5, 5))
    label_colors = np.array(['blue', 'red'])
    # Plot full dataset with high transparency
    plt.scatter(X[:, 0], X[:, 1],c=label_colors[y])
    plt.legend(loc="center")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def viz_heatmap(matrix, title, xaxis, yaxis, savepath):
    increase_factor = len(matrix) // 10
    # Set annotation font size and color
    annot_kws = {"size": 5}

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(5*increase_factor, 4*increase_factor), dpi=200)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='plasma', linewidths=.5, annot_kws=annot_kws, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    fig.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()

def viz_lineplots(matrix, title, xaxis, yaxis, savepath, lbl_names=None, ylim=None):
    increase_factor = len(matrix) // 10
    # Set annotation font size and color
    annot_kws = {"size": 5}
    num_samples, seq_len = matrix.shape
    palette = sns.color_palette("plasma_r", num_samples+1)


    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(6*increase_factor, 3*increase_factor), dpi=200)
    for i in range(num_samples):
        if lbl_names == "no": 
            label=None
        else: label = label=f"{lbl_names} {i+1}"
        sns.lineplot(x=range(seq_len), y=matrix[i,:], color=palette[i], label=label, marker='o', linewidth=1, markersize=6)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)

    if ylim is not None: ax.set_ylim(ylim)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=5, fontsize='small')

    fig.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()