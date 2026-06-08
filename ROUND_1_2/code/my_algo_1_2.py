from code.imc_example_code_book import OrderDepth, UserId, TradingState, Order
#from datamodel import OrderDepth, UserId, TradingState, Order # uncomment for submission
from typing import List
# valid imports 
import numpy as np 
import pandas as pd
from dataclasses import dataclass

PRODUCT1 = 'ASH_COATED_OSMIUM'
PRODUCT2 = 'INTARIAN_PEPPER_ROOT'

# helper method 
@dataclass
class book_helpers(): 
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
    
    # statistical functions
    def pareto_survivial_function(self, v, khat): 
        """
        calculates pareto type one survival function 
        """
        if v is None or v<=0:
            return 0 # complement of CDF taking value 1 is 0
        return np.power(1/v, khat)

    def level_quantity_probability(self, bid_k, ask_k):
        """
        calculates P(Q > v), probability quantity in level is greater than some some value, v, given Q ~ Pareto1(k)
        """
        bids_probs = {}
        asks_probs = {}

        bid_levels, ask_levels = self.bid_ask_levels()

        book_snap = bid_levels | ask_levels
        
        for book_p, book_v in book_snap.items():
            if book_v > 0:
                bids_probs[book_p] = self.pareto_survivial_function(book_v, bid_k)
            elif book_v<0: 
                asks_probs[book_p] = self.pareto_survivial_function(-book_v, ask_k)
        
        return bids_probs, asks_probs
    
    def key_max_val(self, level_probs: dict): 
        """
        gets key (price) with maximum value (quantity probability)
        """
        return max(level_probs, key = level_probs.get)

class Trader:
    def __init__(self):
        self.product1 = PRODUCT1
        self.product2 = PRODUCT2
        self.resting_orders = {self.product1: [], self.product2: []}
        self.filled_orders = {self.product1: [], self.product2: []}
        self.completed_trades = {self.product1: 0, self.product2: 0}
        self.historic_mid_pirce = []

    def bid(self):
        return 15
    
    def update_resting_orders(self, product, pnl_target_ticks, top_ask_p, top_bid_p, order_quantity):
        """
        update resting orders for product (updates orders list in place)
        """
        # resting order logic 
        for order_payload in list(self.resting_orders[product]): 
            side = order_payload['side']
            order_p = order_payload['price']
            if side == 'long': 
                if top_ask_p<=order_p: 
                    self.filled_orders[product].append({
                        'side': side, 
                        'price': order_p, 
                        'quantity': order_quantity, 
                        'target_p': order_p+pnl_target_ticks
                    }) # update fill stack
                    self.resting_orders[product].remove(order_payload) # update stack
                    print(f'FILLED LONG @{order_p}')
            elif side == 'short': 
                if top_bid_p>=order_p:
                    self.filled_orders[product].append({
                        'side': side, 
                        'price': order_p, 
                        'quantity': -order_quantity, 
                        'target_p': order_p-pnl_target_ticks
                    })
                    self.resting_orders[product].remove(order_payload) # update stack 
                    print(f'FILLED SHORT @{order_p}')
    
        return
    
    def update_fill_orders(self, product, orders, top_bid_p, top_ask_p, order_quantity):
        """
        update order fills for product
        """
        for order_payload in list(self.filled_orders[product]): 
            if order_payload['side'] == 'long': 
                if top_bid_p>=order_payload['target_p']:
                    orders.append(Order(product, top_bid_p, -order_quantity)) # sell buy order 
                    self.filled_orders[product].remove(order_payload)
                    self.completed_trades[product] += 1
                    print(f'CLOSED LONG @{top_bid_p}')
            else: 
                if top_ask_p<=order_payload['target_p']: 
                    orders.append(Order(product, top_ask_p, order_quantity)) # buy back sell
                    self.filled_orders[product].remove(order_payload)
                    self.completed_trades[product] += 1
                    print(f'CLOSED SHORT @{top_ask_p}')
        
        return orders
    
    def product_1_startegy(self, 
                           product, 
                           order_depth: OrderDepth,
                           state: TradingState):
        """
        ASH_COATED_OSMIUM trading strategy 
        implements probability based strategy
        """
        orders: List[Order] = []
        book_strat_meta_data: book_helpers = book_helpers(product, order_depth, state)
        qty_pos = book_strat_meta_data.qty_pos()
        pos_not_max = np.abs(qty_pos)<80

        # bid ask book metadata
        top_bid_p, top_bid_v = book_strat_meta_data.bid_from_top()
        top_ask_p, top_ask_v = book_strat_meta_data.ask_from_top()
        if top_ask_p is None or top_bid_p is None:
            return orders 
        spread = top_ask_p - top_bid_p
        total_bid_vol, total_ask_vol = book_strat_meta_data.total_bid_ask_volume()
        
        # imbalance 
        imb = book_strat_meta_data.book_imbalance()

        # mid/micro-price 
        mid_p, micro_p = book_strat_meta_data.mid_micro_price()

        # strategy logic
        bid_prob_q = 1 - book_strat_meta_data.pareto_survivial_function(total_bid_vol, 0.31) # prob realsing bid level volume at least as large 
        ask_prob_q = 1 - book_strat_meta_data.pareto_survivial_function(total_ask_vol, 0.31) # prob realsing ask level volume at leats as large 

        thin_bid_side = bid_prob_q > ask_prob_q # thin buy side, likley down move
        thin_ask_side = bid_prob_q < ask_prob_q # think ask book, likley up move 

        # order creation logic
        o_q = 1
        if pos_not_max: # only submit order if within position constraint
            if thin_bid_side: 
                orders.append(Order(product, top_bid_p, o_q)) # buy limit
                self.resting_orders[product].append({'side': 'long', 'price': top_bid_p})
                print(f'LONG ORDER SUBMISSION @{top_bid_p}')
            elif thin_ask_side: 
                orders.append(Order(product, top_ask_p, -o_q)) # sell limit
                self.resting_orders[product].append({'side': 'short', 'price': top_ask_p})
                print(f'SHORT ORDER SUBMISSION @{top_bid_p}')

        # resting order logic 
        pnl_targ = 10 # best performance at 10
        self.update_resting_orders(product, 
                                   pnl_target_ticks=pnl_targ, 
                                   top_ask_p=top_bid_p, 
                                   top_bid_p=top_ask_p, 
                                   order_quantity=o_q)
        
        # fill logic
        self.update_fill_orders(product, 
                                orders=orders, 
                                top_bid_p=top_bid_p, 
                                top_ask_p=top_ask_p, 
                                order_quantity=o_q)
            
        return orders 
    
    def product_2_strategy(self, 
                           product, 
                           lookback: int, 
                           order_depth: OrderDepth,
                           state: TradingState):
        """
        INTARIAN_PEPPER_ROOT trading strategy
        detrend series using rolling lookback timestemp regression and trade outliers as mean reversion signal
        """
        orders: List[Order] = []
        book_strat_meta_data: book_helpers = book_helpers(product, order_depth, state)
        qty_pos = book_strat_meta_data.qty_pos()
        pos_not_max = np.abs(qty_pos)<80

        # bid ask book metadata
        top_bid_p, top_bid_v = book_strat_meta_data.bid_from_top()
        top_ask_p, top_ask_v = book_strat_meta_data.ask_from_top()
        if top_ask_p is None or top_bid_p is None:
            return orders 
        total_bid_vol, total_ask_vol = book_strat_meta_data.total_bid_ask_volume()

        # strategy logic
        yT_de_trend = 0
        buy_signal = False
        sell_signal = False

        mid_p, micro_p = book_strat_meta_data.mid_micro_price()

        if mid_p is not None:
            self.historic_mid_pirce.append(mid_p)
        
        t = np.arange(0, lookback, 1)
        if len(self.historic_mid_pirce)>lookback: 
            self.historic_mid_pirce.pop(0) # de-queue 
            y = np.array(self.historic_mid_pirce)
            sum_xy = np.sum(y*t)
            sum_x = np.sum(t)
            sum_y = np.sum(y)
            sum_xx = np.sum(np.power(t, 2))
            sum_x2 = np.power(sum_x, 2)
            ybar = np.mean(y)
            xbar = np.mean(t)
            b1 = (lookback*sum_xy - sum_x*sum_y)/(sum_xx - sum_x2)
            b0 = ybar - b1*xbar

            yhatT_trend = b0 + b1*t
            yT_de_trend = y - yhatT_trend
        
            buy_signal = yT_de_trend[-1] < -6
            sell_signal = yT_de_trend[-1] > 6

        # order logic 
        o_q = 1
        if pos_not_max: 
            if buy_signal:
                orders.append(Order(product, top_bid_p, o_q)) # buy limit
                self.resting_orders[product].append({'side': 'long', 'price': top_bid_p})
                print(f'LONG ORDER SUBMISSION @{top_bid_p}')
            elif sell_signal: 
                orders.append(Order(product, top_ask_p, o_q)) # sell limit
                self.resting_orders[product].append({'side': 'short', 'price': top_ask_p})
                print(f'SHORT ORDER SUBMISSION @{top_bid_p}')

        # resting order logic 
        pnl_targ = 5 # best performance at 5
        self.update_resting_orders(product, 
                                   pnl_target_ticks=pnl_targ, 
                                   top_ask_p=top_bid_p, 
                                   top_bid_p=top_ask_p, 
                                   order_quantity=o_q)
        
        # fill logic
        self.update_fill_orders(product, 
                                orders=orders, 
                                top_bid_p=top_bid_p, 
                                top_ask_p=top_ask_p, 
                                order_quantity=o_q)

        return orders

    def run(self, state: TradingState):
        """
        IMC Round 1/2 Strategy Algorithm
        """
        result = {}

        for product in state.order_depths: 
            order_depth: OrderDepth = state.order_depths[product]
            orders = []

            # check no empty levels, if so pass (no data to validate functions)
            if order_depth.buy_orders and order_depth.sell_orders: 
                if product == self.product1: 
                    orders = self.product_1_startegy(product, order_depth, state)
                elif product == self.product2: 
                    orders = self.product_2_strategy(product, 300, order_depth, state)
                    #pass

            result[product] = orders
        
        traderData = ''
        conversions = 0
        return result, conversions, traderData
    