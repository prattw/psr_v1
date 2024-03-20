Project Sea Ranch: 

Background:
The goal of Project Sea Ranch to make trades on a daily basis in order to make a minimum profit of 2% and maximum profit of 5% a day.  The trading platform is Alpaca and the stock data is from the Alpaca API.  The program is run off of my MacBook Air.  I will likely transition it to running from Mac Mini once I am done with this phase of development.  Preferably, I will move this program to the cloud with a server in NYC.  Additionally, all the backtesting functions will be automated and I, as the operator with backtest potential strategies, analyze daily results, and direct strategy.  

This phase of development is centered around getting the program to actually work.  

Priorities:
1. The program must be able to execute trades (buy/sell).
2. The program must be able to backtest strategy(s).
3. The program should have an interface to input stock ticker symbols.

Scope:
Priority 1. The program must be able to execute trades (buy/sell):
Currently the program is able to train models based off Alpaca by minute price data.  This must be done manually in a terminal.  It is also able to give buy/sell signals and actually buy and sell stock when hard coded to do so. 
I need the program to be able to buy stock and then sell stock (take profit) or hold on the Alpaca paper trading site with the idea to eventually move to trading real stocks.  I am willing to take suggestions as to changing from the Alpaca Platform or if daily trading or high frequency trading is the correct direction for this program.

Definable feature: The program must execute buy and sell (take profit) on the Alpaca paper trade platform at least once a day.
Vision: Execute high frequency trading.

Priority 2. The program must be able to backtest strategy(s):
Currently the program can execute backtesting.  It actually performed one backtest and the PDF visualization was massive and took about an hour to download.  
I need the program to be able to take my strategy (5-25 stocks) and be able to produce a back test.  At minimum, it must be able to produce a backtest at the end of the trading day so I can see the model’s performance on a daily basis.

Definable feature: The program must be able to backtest strategy(s) once at EOD.
Vision: Backtest results and backtest different potential strategies.

Desire 3. The program should have an interface to input stock ticker symbols.
Currently I (the operator) must hard code the ticker symbols into the program in order to actually buy and sell stock.  
I would like an easier way to input this the stock ticker symbols.  I realize that this will likely require architecting a backend, so I am not thrilled with this taking up time/resources when I can just hard code the stock ticker symbols myself.

