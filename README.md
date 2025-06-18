# TS-Insight: Visual Decoder for Thompson Sampling

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python Version](https://img.shields.io/badge/python-3.11.9-blue)

**TS-Insight** is an open-source, interactive web application designed to demystify Thompson Sampling (TS) and its variants like Discounted Thompson Sampling (DTS). It provides a powerful visual analytics dashboard for developers, researchers, and practitioners to understand, debug, and explain the behavior of multi-armed bandit algorithms in real-world applications such as active learning and recommendation systems.

This tool is the official artifact for the paper: _"TS-Insight: Visually Decoding Thompson Sampling, an Interactive Dashboard for Algorithm Understanding, Verification, and XAI"_.


*(Main time-series evolution view)*
[ts_evo_Appenzeller-Herzog_2019_T3-502-7.pdf](https://github.com/user-attachments/files/20791319/ts_evo_Appenzeller-Herzog_2019_T3-502-7.pdf)


*(The XAI Snapshot view for explaining a single decision)*
[xai_snapshot_Appenzeller-Herzog_2019_t3.pdf](https://github.com/user-attachments/files/20791320/xai_snapshot_Appenzeller-Herzog_2019_t3.pdf)

## Key Features

-   **Coordinated Multi-View Dashboard**: Simultaneously visualize the evolution of posterior beliefs (via HDR plots), raw evidence counts (`α`/`β`), and a "barcode" timeline of actions and rewards for each arm.
-   **Direct Algorithm Verification**: Visually confirm that belief updates are correct and observe the "forgetting" effect of the discount factor (`γ`) in DTS.
-   **Interactive XAI Snapshot**: Go beyond time-series analysis with a dedicated view that provides an unambiguous, at-a-glance explanation for why a specific arm was chosen at any single time step.
-   **General-Purpose Tool**: Decoupled from any specific algorithm implementation. TS-Insight works with any TS/DTS and all TS variation system that can produce logs in a standardized `.pt` file format.

## Quick Start: Running the Application

Follow these steps to get TS-Insight running locally on your machine.

### 1. Prerequisites

-   Python 3.11.9
-   `pip` package manager

### 2. Installation

First, clone this repository to your local machine:
```bash
git clone https://github.com/your-username/ts-insight.git
cd ts-insight
```
*(Note: Replace `your-username/ts-insight.git` with the actual URL of your repository.)*

Next, it is highly recommended to create a virtual environment to manage dependencies:
```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

Finally, install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Running the App

With your virtual environment active, run the Streamlit application from the project's root directory:
```bash
streamlit run app.py
```

Your web browser should automatically open a new tab with the TS-Insight application running at `http://localhost:8501`.

## How to Use the Tool (A Step-by-Step Guide)

We have included sample log files from a real-world active learning experiment to allow you to immediately test and explore the tool's full capabilities.

#### Step 1: Load the Sample Data

1.  In the application's sidebar on the left, you will see two file uploaders under **"1. Upload Files"**.
2.  Click the "Browse files" button for **"Upload .pt results file"**.
3.  Navigate to the `datasets/appenzeller_2019/` folder within this project and select `*.pt`.
4.  Click the "Browse files" button for **"Upload .json config file (optional)"**.
5.  Navigate to the same folder (`datasets/appenzeller_2019/`) and select `*.json`.
6.  Click the **"Load Data"** button. The application will process the files, and the controls in the sidebar will become active.

#### Step 2: Explore the Main Visualization

1.  The main "Visualization" tab is now active. You can see the evolution plots for all algorithm arms.
2.  In the sidebar under **"2. Plot Controls"**, try the following:
    *   **Select Arms to Visualize**: Use the multiselect dropdown to hide or show specific arms.
    *   **Select T Range**: Drag the slider to zoom into a specific period of the experiment. For example, zoom into the first 10 steps (`0-10`) to see the initial "cold start" behavior.
    *   **Plot Component Visibility**: Uncheck "Show Alpha/Beta Lines" or "Show Barcode Plot" to see how the views dynamically resize to give more space to the visible components.
    *   **Mask Arm Names**: Check this box to anonymize the arm names, replacing them with "Arm 1", "Arm 2", etc.
3.  Click the **"Run Visualization"** button to apply your changes.

#### Step 3: Use the XAI Snapshot for Explanation

This is the core explanatory feature of the tool.

1.  Click on the **"XAI"** tab at the top of the main area.
2.  By default, the view shows the state of the world at the first available sampling step.
3.  Use the **"Select Sampling Step (t)"** dropdown to choose a specific moment to analyze. For a great example of exploration, select **`t = 228`**.
    *   You will see a bar chart comparing all arms. Notice how "Arm 8" has the highest bar (posterior mean), but the black dot (posterior sample) for "Arm 7" is highest. This is the visual proof of *why* Arm 7 was chosen which initiated a shift in preference from Arm 8 to Arm 7.
4.  Toggle the **"Show only arms selected in sidebar"** checkbox. Notice how the default behavior (unchecked) shows all arms for a complete explanation, but you can check it to focus only on the arms you've selected.

## Generating Compatible Log Files

TS-Insight is designed to be a general-purpose tool, intentionally decoupled from any specific algorithm implementation. To visualize your own experiment, you simply need to provide log files in the formats described below.

### Primary Data File: `*.pt` (Required)

This is a PyTorch-serialized file containing the core experimental data. It must be a Python `dictionary` saved with `torch.save()`. The dictionary must contain a key named `'detailed_log'`.

The value for `'detailed_log'` must be a `list` of dictionaries, where each inner dictionary represents the state of the system at a single sampling step `t`.

**Structure of a single log entry (one time step):**

```python
# A single dictionary in the 'detailed_log' list
log_entry_t = {
    # --- Core Fields (Required) ---
    'query_num_total': 150,      # Integer: The current sampling step / time t.
    'arm': 'UncertaintySampling', # String: The name of the arm that was chosen (pulled).
    'reward': 1.0,               # Float: The reward received (e.g., 1.0 for success, 0.0 for failure).
    
    'arm_states': {
        # --- Arm States Dictionary (Required) ---
        # This dictionary must contain an entry for EVERY arm in the system at this time step.
        'UncertaintySampling': {
            'alpha_before_update': 5.2,   # Float: The alpha parameter of the Beta distribution *before* the update.
            'beta_before_update': 10.1,   # Float: The beta parameter of the Beta distribution *before* the update.
            'posterior_sample': 0.85      # Float: The value (~theta) sampled from this arm's posterior.
        },
        'GNNExploit': {
            'alpha_before_update': 20.5,
            'beta_before_update': 4.3,
            'posterior_sample': 0.82
        },
        # ... and so on for all other arms at time t
    }
}
```

**Key Requirements:**

-   The file must be saved using `torch.save()`.
-   The top-level object must be a dictionary.
-   This dictionary must have the key `'detailed_log'`.
-   `'detailed_log'` must be a list of dictionaries as described above.
-   `arm_states` must be present and contain the state for *all* arms at each time step to enable comparative analysis in the XAI view.

### Configuration File: `config.json` (Optional)

You can optionally provide a JSON file to supply metadata for your experiment, which will be displayed in the application. This helps in identifying and organizing different runs.

**Structure of the `config.json` file:**

```json
{
  "dataset_name": "Appenzeller-Herzog 2019",
  "model_name": "MyCustomDTS",
  "al_strategy_params": {
    "ts_discount_factor": 0.99
  },
  "other_metadata": "Any other info you want to record."
}
```

**Role of the file:**
-   **`dataset_name`**: If provided, this string will be used as the main title for the visualizations. If this file is not provided, the tool will attempt to find a `dataset_name` key inside the `.pt` file. If neither is found, it will default to "Uploaded\_Data".

By adhering to this simple logging format, you can leverage TS-Insight to debug, analyze, and create formal visualizations for any Thompson Sampling-based system.

## Contributing

We welcome contributions! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the GNU General Public License v3.0.  
© 2025 Luxembourg Institute of Science and Technology  
© 2025 Parsa Vares

## Author

**Parsa Vares**  
Luxembourg Institute of Science and Technology  
[@parsavares](https://github.com/parsavares) · parsa.vares@list.lu
