# IMC-Prosperity-4-

![Leaderboard](Results/Screenshot%202026-05-11%20at%2022.50.48.png)


This folder contains all my submissions to the IMC Prosperity 4 quantitative trading challenge algorithmic challenges from rounds 1 through 5. Code was also used for assistance in the manual challenges which can be found in the ```backtest.ipynb``` file in the corresponding rounds folder. 


## Import Notes 

- [Medium Article](https://medium.com/@amjadsaidama/i-competed-in-the-hardest-trading-challenge-on-the-planet-imc-prosperity-4-5cfbf161f203): Deltailed explenation of each rounds challenges
- Submissions for the algorithmic challanges found in each rounds, ```my_algo.py``` file are not complete, in the sence they did not cover all the available assest listes in each round, this is largely due to time constraints, which made it not possible for me to develop a reliably profitable strategy for each of the availabe assets in each round.
- Each ```my_algo.py``` implements a unique strategy implemented to leverge the trends uncovered in each asset class

## Root Folder Contents

```
IMC/
├── README.md                                   # Project overview and documentation
├── notebook_path.py                            # Utility script for resolving notebook paths
├── Results/                                    # Screenshots and media from the competition
│   ├── Screenshot 2026-05-11 at 22.50.48.png  # Mid-competition leaderboard standing
│   ├── Screenshot 2026-06-06 at 17.54.14.png  # Final leaderboard result
│   └── media-kit.png                           # IMC Prosperity official media kit image
├── ROUND_1_2/                                  # Rounds 1 & 2 submissions and analysis
│   ├── backtest.ipynb                          # Backtesting and manual challenge analysis notebook
│   ├── code/
│   │   ├── data_pre_processing.py              # Data cleaning and preprocessing utilities
│   │   ├── eda_functions.py                    # Exploratory data analysis helper functions
│   │   ├── imc_example_code_book.py            # IMC-provided example code reference
│   │   ├── imc_example_code_submission.py      # IMC example submission template
│   │   └── my_algo_1_2.py                      # Round 1 & 2 algorithmic trading submission
│   └── data/                                   # Historical price and trade data for rounds 1 & 2
│       ├── prices_round_1_day_-2.csv           # Order book price data day -2
│       ├── prices_round_1_day_-1.csv           # Order book price data day -1
│       ├── prices_round_1_day_0.csv            # Order book price data day 0
│       ├── trades_round_1_day_-2.csv           # Market trade data day -2
│       ├── trades_round_1_day_-1.csv           # Market trade data day -1
│       └── trades_round_1_day_0.csv            # Market trade data day 0
├── ROUND_3_4/                                  # Rounds 3 & 4 submissions and analysis
│   ├── backtest.ipynb                          # Backtesting and manual challenge analysis notebook
│   ├── code/
│   │   └── my_algo_3_4.py                      # Round 3 & 4 algorithmic trading submission
│   ├── data_round_3/                           # Historical price and trade data for round 3
│   │   ├── prices_round_3_day_0.csv            # Order book price data day 0
│   │   ├── prices_round_3_day_1.csv            # Order book price data day 1
│   │   ├── prices_round_3_day_2.csv            # Order book price data day 2
│   │   ├── trades_round_3_day_0.csv            # Market trade data day 0
│   │   ├── trades_round_3_day_1.csv            # Market trade data day 1
│   │   └── trades_round_3_day_2.csv            # Market trade data day 2
│   └── data_round_4/                           # Historical price and trade data for round 4
│       ├── prices_round_4_day_1.csv            # Order book price data day 1
│       ├── prices_round_4_day_2.csv            # Order book price data day 2
│       ├── prices_round_4_day_3.csv            # Order book price data day 3
│       ├── trades_round_4_day_1.csv            # Market trade data day 1
│       ├── trades_round_4_day_2.csv            # Market trade data day 2
│       └── trades_round_4_day_3.csv            # Market trade data day 3
└── ROUND_5/                                    # Round 5 submissions and analysis
    ├── backtest.ipynb                          # Backtesting and manual challenge analysis notebook
    ├── code/
    │   └── my_algo_5.py                        # Round 5 algorithmic trading submission
    ├── data/                                   # Historical price and trade data for round 5
    │   ├── prices_round_5_day_2.csv            # Order book price data day 2
    │   ├── prices_round_5_day_3.csv            # Order book price data day 3
    │   ├── prices_round_5_day_4.csv            # Order book price data day 4
    │   ├── trades_round_5_day_2.csv            # Market trade data day 2
    │   ├── trades_round_5_day_3.csv            # Market trade data day 3
    │   └── trades_round_5_day_4.csv            # Market trade data day 4
    └── manual_task/                            # Assets provided for the round 5 manual challenge
        └── q7m3x9v2...jpg                      # Image used in the round 5 manual trading task
```
