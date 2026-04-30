#from ROUND_1_2.code_round_1.imc_example_code_book import OrderDepth, UserId, TradingState, Order, Trade
from datamodel import OrderDepth, UserId, TradingState, Order, Trade 
from typing import List
# valid imports 
import numpy as np 
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
import math 

PRODUCT1 = 'HYDROGEL_PACK' 
PRODUCT2 = 'VELVETFRUIT_EXTRACT' # underlying for option contracts
def PRODUCT3(product: str):
    if product.split('_')[0] == 'VEV': 
        return True
    return False 
TTE = 4 # time until expirey 

# ------------------------- L2 BOOK -------------------------
@dataclass
class book_helpers(): 
    """
    retrives metadata from level two order book
    """
    product: str
    order_depth: OrderDepth
    state: TradingState
    
    def qty_pos(self):
        return self.state.position.get(self.product, 0) if self.state.position else 0 
    
    def bid_ask_levels(self): 
        """
        gets full bid and ask levels from book
        """
        bids = self.order_depth.buy_orders
        asks = self.order_depth.sell_orders
        return bids, asks
    
    def bid_from_top(self, level_index: int = 0): 
        """
        k'th level from the top of bid (assuming descending sort for bids)
        defualts to top of book
        """
        bids, _ = self.bid_ask_levels()

        bid_prices = list(bids.keys())
        if level_index <= len(bid_prices)-1:
            kth_bid_price = bid_prices[level_index]
            kth_bid_volume = bids[kth_bid_price]
            return kth_bid_price, kth_bid_volume
        return None, None
    
    def ask_from_top(self, level_index: int = 0): 
        """
        kth level from top of ask (assuming ascending sort for asks)
        defaults to top of book
        """
        _, asks = self.bid_ask_levels()

        ask_prices = list(asks.keys())
        if level_index <= len(ask_prices)-1:
            kth_ask_price = ask_prices[level_index]
            kth_ask_volume = asks[kth_ask_price]
            return kth_ask_price, kth_ask_volume
        return None, None
    
    def total_bid_ask_volume(self):
        """
        total bid and ask side volume
        """
        bids, asks = self.bid_ask_levels()
        
        bid_total_volume = np.sum(list(bids.values())) if bids else 0 
        ask_total_volume = np.sum(list(asks.values())) if asks else 0

        return bid_total_volume, abs(ask_total_volume)
    
    def book_imbalance(self):
        """
        book imbalance
        """
        tot_bid_vol, tot_ask_vol = self.total_bid_ask_volume()
        denom = tot_bid_vol + abs(tot_ask_vol)
        if denom == 0: # avoid division by zero 
            return 0
        
        return (tot_bid_vol - tot_ask_vol) / denom
    
    def mid_micro_price(self):
        """
        book mid-price and micro-price
        """
        top_bid_p, top_bid_v = self.bid_from_top()
        top_ask_p, top_ask_v = self.ask_from_top()

        if top_bid_p is None or top_ask_p is None or top_bid_v is None or top_ask_v is None:
            return None, None

        top_ask_v = abs(top_ask_v)

        mid_price = (top_bid_p + top_ask_p) / 2
        micro_denom =  top_bid_v + top_ask_v
        if micro_denom == 0:
            return mid_price, 0
        micro_price = (top_ask_p*top_bid_v + top_bid_p*top_ask_v) / micro_denom
        
        return mid_price, micro_price

# --------------------------- OPTIONS ---------------------------
@dataclass
class BlackScholes:
    """
    Black and Scholes Option Price
    """
    St: float # present value of underlying
    K: float # strike price 
    sigma: float # annulised volatility estimate 
    r: float = 4.3e-2 # portfolio growth rate 
    T: int = 7 # option contract length
    t: int = 3 # option held time
    days_in_year: int = 252

    def call_premium(self): 
        """
        calculates the option call price under BS, calculates the present value option price by defualt, t=0
        """
        d1, d2 = self._d1d2()
        eta = self._time_until_expirey()
        phi_d1 = self._cdf(d1)
        phi_d2 = self._cdf(d2)

        return self.St*phi_d1 - self.K*np.exp(-self.r*eta)*phi_d2

    def moneyness(self):
        """
        checks moneyness in option to determine OTM, INM or OTM
        """
        return np.log(self.St/self.K)

    # helpers
    def _d1d2(self): 
        eta = self._time_until_expirey()
        num = np.log(self.St/self.K) + (self.r + .5*np.power(self.sigma, 2))*eta
        denom = self.sigma*np.sqrt(eta)
        d1 = num/denom
        d2 = d1 - denom
        return d1, d2
    
    def _time_until_expirey(self): 
        return (self.T - self.t) / self.days_in_year
    
    def _cdf(self, x: float):
        return .5*math.erfc(-x/math.sqrt(2.0))

@dataclass
class ImpliedVolatility:
    """
    iv implemented using newton raphson on f(sigma) = C_BS(sigma_iv) + C_MRK
    """
    black_scholes: BlackScholes # must be defined with defualt iv, e.g. 0.2
    call_p_market: float
    
    def iv_newton_raphson(self, iterations: int = 10):
        eps = 1e-4
        f_min = np.inf
        iv_min = np.inf
        iv_star = self.black_scholes.sigma
        for _ in range(iterations): 
            f_last = self.black_scholes.call_premium() - self.call_p_market
           
            # check convergance 
            if abs(f_last)<=eps: # abs to avoid large negative quantity being raised 
                return iv_star
            
            # if not converged update
            d1, _ = self.black_scholes._d1d2()
            pdf_d1 = np.exp(-.5 * d1**2) / np.sqrt(2 * np.pi)
            eta = self.black_scholes._time_until_expirey()
            vega_last = self.black_scholes.St * pdf_d1 * np.sqrt(eta)
            iv_star -= f_last/vega_last
            self.black_scholes.sigma = iv_star # use updated volatility to price option

            if vega_last < 1e-8 or np.isnan(vega_last):
                return iv_min

            # to avoid function value exploding 
            if abs(f_last)<f_min:
                f_min = abs(f_last)
                iv_min = self.black_scholes.sigma

        return iv_min # best guess 
    
@dataclass
class DeltaHedging():
    """
    caluclates delta and required stock holding to make first order linear approximation of replicating portfolio have zero exposre to first order moves in stock price

    intedned to used in delta hedging strategy that profits of volatility misspricings
    """
    black_scholes: BlackScholes

    def call_option_delta(self):
        """
        call option delta
        """
        d1, _ = self.black_scholes._d1d2()
        return self.black_scholes._cdf(d1)
    
    def delta_stock(self, option_quantity):
        """
        delta fraction of stock, to delta hedge current position 
        """
        delta_t = self.call_option_delta()
        return delta_t*option_quantity
    
# ------------------------- QUEUE LOGIC -------------------------
class QueueLogic(): 
    def __init__(self):
        self.resting_orders: dict[list] = defaultdict(list)
        self.filled_orders: dict[list] = defaultdict(list)
        self.option_trades: dict[list] = defaultdict(list)
        self.completed_trades: dict[int] = defaultdict(int)

    def update_resting_orders(self, 
                              product, 
                              pnl_long_ticks,
                              pnl_short_ticks, 
                              top_ask_p, 
                              top_bid_p, 
                              order_quantity):
        """
        update resting orders for product (updates orders list in place)
        """
        # resting order logic 
        for order_payload in list(self.resting_orders[product]): 
            side = order_payload['side']
            order_p = order_payload['price']

            if side == 'long': 
                if top_ask_p<=order_p: # ask at least less than buy limit to fill
                    # log long limit metadata 
                    self.filled_orders[product].append({
                        'side': side, 
                        'price': order_p, 
                        'quantity': order_quantity, 
                        'target_price': order_p + pnl_long_ticks,
                        'timestamp_since_hedge': 0
                    }) # update fill stack
                    self.resting_orders[product].remove(order_payload) # update stack
                    print(f'FILLED LONG @{order_p}')

            elif side == 'short': 
                if top_bid_p>=order_p: # bid at least greater than sell limit to fill
                    # long short limit metadata 
                    self.filled_orders[product].append({
                        'side': side, 
                        'price': order_p, 
                        'quantity': -order_quantity, 
                        'target_price': order_p - pnl_short_ticks,
                        'timestamp_since_hedge': 0
                    })
                    self.resting_orders[product].remove(order_payload) # update stack 
                    print(f'FILLED SHORT @{order_p}')
    
        return
    
    def update_fill_orders(self, product, result, top_bid_p, top_ask_p):
            """
            update order fills for product
            """
            for order_payload in list(self.filled_orders[product]): 
                side  = order_payload['side']
                qty = order_payload['quantity']
                tp = order_payload['target_price']

                if side == 'long': 
                    if top_bid_p>=tp:
                        result[product].append(Order(product, top_bid_p, -qty)) # sell buy order 
                        self.filled_orders[product].remove(order_payload)
                        self.completed_trades[product] += 1
                        print(f'CLOSED LONG @{top_bid_p}')

                elif side == 'short': 
                    if top_ask_p<=tp: 
                        result[product].append(Order(product, top_ask_p, -qty)) # buy back sell
                        self.filled_orders[product].remove(order_payload)
                        self.completed_trades[product] += 1
                        print(f'CLOSED SHORT @{top_ask_p}')

            return
    
    def update_order_history(self, product: str, order_payload):
        """
        function for updating trade history manualy (if tarding at market)
        """
        self.option_trades[product] += [order_payload]
        self.completed_trades[product] += 1
    
# ----------------------- SIGNAL TO MARKET ----------------------
class Trader:
    def __init__(self):
        self.orders_management: QueueLogic = QueueLogic()
        self.eq_max = 200
        self.opt_max = 300
        self.rv = 0.34 # realised volatility (annulsied empirical average)
        self.hedge_rebals = 0
        self.mm_inventory = []
        self.mm_pressure = 0

    def hydro(self, 
              result: dict[List], 
              equity_product,
              order_depth: OrderDepth, 
              state: TradingState): 
        """
        market maker mean reversion algorithm, that trades same direction as market maker when footprint is sufficintly large 
        """
        # get book metadata 
        book: book_helpers = book_helpers(equity_product, order_depth, state)
        qty_pos = book.qty_pos()
        pos_valid = abs(qty_pos)<self.eq_max

        # bid ask book metadata 
        top_bid_p, _ = book.bid_from_top()
        top_ask_p, _ = book.ask_from_top()
        if top_bid_p is None or top_ask_p is None:
            return result
        
        # trading signal 
        market_trades: Trade = state.market_trades.get(equity_product, []) # market participant most recent trade in stack
        
        if market_trades:
            m_par: Trade = market_trades[-1]
        
            # trailing market making inventory
            if m_par.buyer == 'Mark 38': 
                if m_par.price == top_ask_p: # aggreg buy at ask, validate mm buy trade
                    self.mm_inventory.append(m_par.quantity)
                            
            elif m_par.seller == 'Mark 38': 
                if m_par.price == top_bid_p: # aggreg sell at bif, validate mm sell trade
                    self.mm_inventory.append(-m_par.quantity)
            else:
                self.mm_inventory.append(0)

            if len(self.mm_inventory)>1:
                if self.mm_inventory[-1] > self.mm_inventory[-2]:
                    self.mm_pressure += 1
                elif self.mm_inventory[-1] < self.mm_inventory[-2]:
                    self.mm_pressure -=1
                else:
                    self.mm_pressure = 0

        net_flow = sum(self.mm_inventory)

        buy_q3 = net_flow > 86  # impirical q1 inventory percentile 
        sell_q3 = net_flow < -33 # impirical q3 inventory percentile

        buy_signal = buy_q3 and self.mm_pressure > 3
        sell_signal = sell_q3 and self.mm_pressure < -3

        # pyramid order 
        oq = 1
        if pos_valid:
            if buy_signal:
                result[equity_product].append(Order(equity_product, top_bid_p, oq))
                self.orders_management.resting_orders[equity_product].append({'side': 'long', 'price': top_bid_p})
                print(f'LONG LIMIT @{top_bid_p}')
                print(f'MM INVENTORY: {self.mm_inventory}')
            if sell_signal:
                result[equity_product].append(Order(equity_product, top_ask_p, -oq))
                self.orders_management.resting_orders[equity_product].append({'side': 'short', 'price': top_ask_p})
                print(f'SHORT LIMIT @{top_ask_p}')
                print(f'MM INVENTORY: {self.mm_inventory}')

        # update resting order
        long_trg_ticks = 5
        short_trg_ticks = 5
        self.orders_management.update_resting_orders(equity_product, 
                                                     long_trg_ticks,
                                                     short_trg_ticks, 
                                                     top_ask_p, 
                                                     top_bid_p, 
                                                     oq)
        
        # fill logic 
        self.orders_management.update_fill_orders(equity_product, 
                                                  result, 
                                                  top_bid_p, 
                                                  top_ask_p)

        # mean reversion
        """
        for order_payload in list(self.orders_management.filled_orders[equity_product]):
            if order_payload:
                qty = order_payload['quantity']
                side = order_payload['side']

                # state based mean reversion exit 
                if abs(self.market_maker_inventory) < 3: # low inventory signal mm not active in market
                    if side == 'long':
                        result[equity_product].append(Order(equity_product, top_bid_p, -qty))
                        self.orders_management.filled_orders[equity_product].remove(order_payload)
                        print(f'MEAN REVERSION, LONG CLOSE @{top_bid_p}')
                    
                    elif side == 'short':
                        result[equity_product].append(Order(equity_product, top_ask_p, -qty))
                        self.orders_management.filled_orders[equity_product].remove(order_payload)
                        print(f'MEAN REVERSION, SHORT CLOSE @{top_ask_p}')
        """

        return result

    def call_long_payoff(self, underlying_price, strike_price, long=True):
        """
        option payoffs (not pnl) for long call and short call
        """
        if long:
            return max(underlying_price - strike_price, 0)
        return min(strike_price - underlying_price, 0)

    def velv_options_strategy(self,
                              result: dict[list],
                              equity_product, 
                              option_product, 
                              product_strike,
                              equity_order_depth: OrderDepth, 
                              option_order_depth: OrderDepth,
                              state: TradingState): 
        """
        delta hedged volatility mispricing strategy
        undelying orders (the delta hedge) is integrated within this option strategy 
        """
        # book class
        eq_l2_meta_data: book_helpers = book_helpers(equity_product, equity_order_depth, state)
        opt_l2_meta_data: book_helpers = book_helpers(option_product, option_order_depth, state)

        # position constraints
        eq_qty_pos, opt_qty_pos = eq_l2_meta_data.qty_pos(), opt_l2_meta_data.qty_pos()
        eq_pos_valid, opt_eq_valid = np.abs(eq_qty_pos)<self.eq_max, np.abs(opt_qty_pos)<self.opt_max

        # top bids and asks
        eq_bid1_p, _ = eq_l2_meta_data.bid_from_top()
        eq_ask1_p, _ = eq_l2_meta_data.ask_from_top()

        opt_bid1_p, _ = opt_l2_meta_data.bid_from_top()
        opt_ask1_p, _ = opt_l2_meta_data.ask_from_top()

        # mid and micro price 
        eq_mid_p, _ = eq_l2_meta_data.mid_micro_price()
        opt_mid_p, _ = opt_l2_meta_data.mid_micro_price()

        # check book level not empty
        if (opt_bid1_p is None or opt_ask1_p is None) or (eq_bid1_p is None or eq_ask1_p is None):
            return result
        
        # strategy order logic 
        bs_model = BlackScholes(
            St = eq_mid_p, 
            K = product_strike, 
            sigma = self.rv, 
        )
        
        #iv = ImpliedVolatility(bs_model, opt_mid_p).iv_newton_raphson()
        #vol_under_priced = iv<self.rv # under pricing furture volatility, buy prem
        #vol_over_priced = iv>self.rv # over pricing future volatility, sell prem

        call_prem = bs_model.call_premium()
        edge = 1
        bull_edge = call_prem - opt_bid1_p > edge # pay less prem
        bair_edge = opt_ask1_p - call_prem > edge # collect more prem

        # resting order logic 
        opt_oq = 1
        l2_t = state.timestamp
        if opt_eq_valid:
            if bull_edge:
                result[option_product].append(Order(option_product, opt_bid1_p, opt_oq)) # limit order
                self.orders_management.resting_orders[option_product].append({'side': 'long', 'price': opt_bid1_p})
                print(f'LONG LIMIT @{opt_bid1_p}')
            elif bair_edge:
                result[option_product].append(Order(option_product, opt_ask1_p, -opt_oq)) # limit order
                self.orders_management.resting_orders[option_product].append({'side': 'short', 'price': opt_ask1_p})
                print(f'SHORT LIMIT @{opt_ask1_p}')

        # update resting orders (option exit logic is omitted as option payoff is realised by defualt on expirey)
        self.orders_management.update_resting_orders(option_product, 
                                                     0, 
                                                     0,
                                                     opt_ask1_p, 
                                                     opt_bid1_p, 
                                                     opt_oq)
        
        # delta hedging logic (rebalanced continuously until option expirey)
        delta_hedge: DeltaHedging = DeltaHedging(bs_model)
        rebal_epoch = 100*1e4*1/24 # rebalance portfolio frequency (100*1e4=1day)
        
        for opt_order_payload in self.orders_management.filled_orders[option_product]: 
            if opt_order_payload: 
                opt_oq = opt_order_payload['quantity']
                side = opt_order_payload['side']

                opt_time = opt_order_payload['timestamp_since_hedge']
                rebal = l2_t - opt_time >= rebal_epoch
                
                if rebal or opt_time == 0:
                    opt_order_payload['timestamp_since_hedge'] = l2_t
                    s_hedge_q = int(round(delta_hedge.delta_stock(opt_oq)))
                    
                    if side == 'long':
                        result[equity_product].append(Order(equity_product, eq_bid1_p, -s_hedge_q))
                        self.hedge_rebals += 1
                        print(f'DELTA HEDGE {self.hedge_rebals}')

                    elif side == 'short': 
                        result[equity_product].append(Order(equity_product, eq_ask1_p, s_hedge_q))
                        self.hedge_rebals += 1
                        print(f'DELTA HEDGE {self.hedge_rebals}')

        return result
    
    def run(self, state: TradingState):
        """
        IMC Round 3 Strategy Algorithm
        """
        result = defaultdict(list)
        underlying_order_depth = state.order_depths[PRODUCT2]

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]

            if order_depth.buy_orders and order_depth.sell_orders:
                
                if product == PRODUCT1:
                    #result = self.hydro(result, product, order_depth, state)
                    pass
                elif PRODUCT3(product):
                    strike_price = float(product.split('_')[1])
                    result = self.velv_options_strategy(result, 
                                                        PRODUCT2,
                                                        product, 
                                                        strike_price, 
                                                        underlying_order_depth, 
                                                        order_depth, 
                                                        state)

        traderData = ''
        conversions = 0
        return result, conversions, traderData

