
# Experiments with LAP and QAP

This directory is dedicated to three distinct experiments. Please installed all required dependencies:

```bash
pip install torch torchvision
pip install -r requirements.txt
```

## Experiment 1: Dense LAP

1. **Dataset Preparation**:
   - Download the [Faust dataset](https://faust-leaderboard.is.tuebingen.mpg.de/).
   - Extract Heat Kernel Signature and SHOT descriptor from the Faust Dataset. This can be done using tools like [pyshot](https://github.com/uhlmanngroup/pyshot) and [pyhks](https://github.com/ctralie/pyhks). Save the extracted data in './data/' in `.mat` file format, or use another format but ensure to modify the loading function in `faustDS_reader.py` accordingly.

2. **Running Experiments**:
   - To conduct dense LAP experiments, execute the script `lap_exp.py`.
   - Note: These experiments were performed using a GeForce RTX 3080 with CUDA version 11.4.

## Experiment 2: Sparse LAP

- For sparse LAP experiments, run the script `experiment_sparse.py`.
- These experiments were conducted using a GeForce RTX 3090 with CUDA version 11.1.

## Experiment 3: QAP

1. **Dataset Acquisition**:
   - Download the  [QAPLIB dataset](https://www.opt.math.tugraz.at/qaplib/) and save in './QAPLIB/'.

2. **Running the Experiments**:
   - Execute `qaplib_exp.py` to start the QAP experiments.
   - Note: These experiments were performed using a  GeForce RTX 3080 with CUDA version 11.4.
