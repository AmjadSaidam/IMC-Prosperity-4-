#from ROUND_1_2.code_round_1.imc_example_code_book import OrderDepth, UserId, TradingState, Order, Trade 
from datamodel import OrderDepth, UserId, TradingState, Order, Trade 
from typing import List
# valid imports 
import numpy as np 
from collections import defaultdict
from dataclasses import dataclass, field
import uuid 

COINT1_PRODUCT1 = 'SNACKPACK_PISTACHIO'
COINT1_PRODUCT2 = 'SNACKPACK_STRAWBERRY'
pair1 = (COINT1_PRODUCT1, COINT1_PRODUCT2)

COINT2_PRODUCT1 = 'SNACKPACK_RASPBERRY'
COINT2_PRODUCT2 = 'SNACKPACK_PISTACHIO'
pair2 = (COINT2_PRODUCT1, COINT2_PRODUCT2)

COINT3_PRODUCT1 = 'SNACKPACK_RASPBERRY'
COINT3_PRODUCT2 = 'SNACKPACK_STRAWBERRY'
pair3 = (COINT3_PRODUCT1, COINT3_PRODUCT2)

COINT4_PRODUCT1 = 'SNACKPACK_VANILLA'
COINT4_PRODUCT2 = 'SNACKPACK_CHOCOLATE'
pair4 = (COINT4_PRODUCT1, COINT4_PRODUCT2)

pairs = [pair1, pair2, pair3, pair4]

coint_products = [COINT1_PRODUCT1, COINT1_PRODUCT2, COINT2_PRODUCT1, COINT2_PRODUCT2, COINT3_PRODUCT1, COINT3_PRODUCT2, COINT4_PRODUCT1, COINT4_PRODUCT2]

# ------------------------- L2 BOOK -------------------------
@dataclass
class BookHelpers(): 
    """
    retrives metadata from level two order book
    """
    product: str
    order_depth: OrderDepth
    state: TradingState
    
    def quantity_position(self):
        return self.state.position.get(self.product, 0) if self.state.position else 0 
    
    def quantity_remaining(self, max_position):
        return max_position - abs(self.quantity_position())
    
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
    
    def pareto_survivial_function(self, v, khat): 
        """
        calculates pareto type one survival function 
        """
        if v is None or v<=0:
            return 0 # complement of CDF taking value 1 is 0
        return np.power(1/v, khat)

# ------------------------- METADATA LOGS -------------------------
class LogBookMetadata():
    def __init__(self):
        self.product_price_data: dict[List] = defaultdict(list)

    def full_book(self, order_depth: OrderDepth):
        """
        """
        return bool(order_depth.buy_orders) and bool(order_depth.sell_orders)

    def log_price(self, state):
        """
        """
        for product in state.order_depths:
            od: OrderDepth = state.order_depths[product]
            book: BookHelpers = BookHelpers(product, od, state)
            mid_price, _ = book.mid_micro_price()
            self.product_price_data[product].append(mid_price)

        return

# ------------------------- COINTEGRATION -------------------------
@dataclass
class CoIntegration(): 
    x1_p: np.ndarray
    x2_p: np.ndarray
    state: TradingState 
    lookback: int = int(1e4/(24*60)) # defualt 1 min data lookback

    def ADF_test(self, 
             price_pair_residuals: np.ndarray, 
             alpha_confidence_interval: float = 2.86) -> List: 
        """
        ADF test for, gamma = rho - 1. Rho is coefficient for AR(1) model
        H0: gamma = 0 => unit-roor => not stationary
        H1: gamma < 0 => no unit-root => stanionary
        """
        u = np.array(price_pair_residuals)
        delta_resid = np.diff(u)
        n = delta_resid.size # for standard error

        lag_features = np.zeros_like(delta_resid)
        lag_features = u[:-1]

        # ADF regression
        fit = self._OLS_general(lag_features, delta_resid)
        alpha = fit['parameters'][0]
        gamma = fit['parameters'][1]
        
        # lagged residual standard error 
        adf_resid = delta_resid - self._OLS_simple_predict(lag_features, gamma, alpha)

        if n<=2:
            return False

        sigma2 = np.sqrt(np.sum(np.power(adf_resid, 2))/(n - 2))

        # test statsitic
        denom = np.sqrt(n * np.var(lag_features))
        if denom == 0:
            return False 
        se = sigma2 / denom
        t_phi = gamma/se 

        # hypothesis test 
        coint = True if t_phi < -alpha_confidence_interval else False

        return coint
    
    def residuals(self):
        return self._OLS_simple()['residuals']

    def spread_hedge(self):
        """
        Cointegrated series signal. This is standerdised residuals from the pairs price regression
        """
        fit = self._OLS_simple()
        u = fit['residuals']
        beta = fit['beta']

        avg_spread = np.mean(u)
        vol_spread = np.std(u, ddof = 1)
        if vol_spread == 0:
            return 0, beta 

        z_last = (u[-1] - avg_spread)/vol_spread # use most recent residual 
        
        return z_last, beta

    # helpers 
    def _OLS_simple(self):
        """
        estimates the pairs price model 
        """
        beta = None 

        emp_error = lambda x1,x2,beta: x1 - beta*x2 
        x1_p_lkb = np.array(self.x1_p[-self.lookback: ])
        x2_p_lkb = np.array(self.x2_p[-self.lookback: ])
        cov_x1x2 = np.cov(x1_p_lkb, x2_p_lkb)
        var_x2 = np.var(x2_p_lkb, ddof = 1)
        beta = cov_x1x2[0, 1]/var_x2 # scaler 

        resid = emp_error(x1_p_lkb, x2_p_lkb, beta) # residuals

        return {
            'label': x1_p_lkb, 
            'feature': x2_p_lkb, 
            'beta': beta,
            'residuals': resid
        }
    
    def _OLS_simple_predict(self, 
                            x: np.ndarray, 
                            beta: float, 
                            alpha: float = 0.0):
        """
        simple linear regression model
        """
        return alpha + beta*x

    def _OLS_general(self, 
                    features: np.ndarray, 
                    label: np.ndarray):
        """
        least squares adf test for cointegration
        """
        n = features.shape[0]
        ones = np.ones((n, ))
        X = np.column_stack([ones, features])
        Y = label

        Xt = X.T
        XtX = Xt @ X
        XXt_inv = np.linalg.inv(XtX)
        XtY = Xt @ Y 
        params = XXt_inv @ XtY

        return {
            'parameters': params
        }

# ------------------------- QUEUE LOGIC -------------------------
class QueueLogic(): 
    def __init__(self):
        self.resting_orders: dict[list] = defaultdict(list)
        self.filled_orders: dict[list] = defaultdict(list)
        self.pairs_orders: dict[List] = defaultdict(list)
        self.completed_trades: dict[int] = defaultdict(int)

    def update_pairs_trades(self, 
                            result, 
                            product_x1, 
                            product_x2,
                            top_ask_p, 
                            top_bid_p,
                            pairs_order_data: dict = None, 
                            pair_order_logs: dict = None):
        """
        """
        for order_payload in list (self.resting_orders[product_x1]):
            o_id = str(uuid.uuid4())
            side = order_payload['side']
            order_p = order_payload['price']
            order_q = order_payload['quantity']

            if side == 'long':
                if top_ask_p<=order_p:
                    # x1 long
                    result[product_x1].append(Order(product_x1, order_p, order_q))
                    self.resting_orders[product_x1].remove(order_payload) # update stack
                    self.pairs_orders[o_id].append(self._order_fill_logs(side, order_p, order_q, 0, 0))  # update fill stack
                    # x2 
                    result[product_x2].append(Order(*pairs_order_data['x2_short']))
                    self.pairs_orders[o_id].append((pair_order_logs['x2_short']))         
            elif side == 'short':
                if top_bid_p>=order_p:
                    # x1 short 
                    result[product_x1].append(Order(product_x1, order_p, -order_q))
                    self.resting_orders[product_x1].remove(order_payload) # update stack
                    self.pairs_orders[o_id].append(self._order_fill_logs(side, order_p, order_q, 0, 0))  # update fill stack
                    result[product_x2].append(Order(*pairs_order_data['x2_long']))
                    self.pairs_orders[o_id].append((pair_order_logs['x2_long']))  

        return
            
    def update_resting_orders(self, 
                              result,
                              product, 
                              pnl_long_ticks,
                              pnl_short_ticks, 
                              top_ask_p, 
                              top_bid_p):
        """
        update resting orders for product (updates orders list in place), assuming investing equal quantity in both directions
        """
        # resting order logic 
        for order_payload in list(self.resting_orders[product]): 
            side = order_payload['side']
            order_p = order_payload['price']
            order_q = order_payload['quantity']

            if side == 'long': 
                if top_ask_p<=order_p: # ask at least less than buy limit to fill
                    # log long limit metadata 
                    result[product].append(Order(product, order_p, order_q))
                    self.filled_orders[product].append(self._order_fill_logs(side, order_p, order_q, order_p+pnl_long_ticks, 0))  # update fill stack
                    self.resting_orders[product].remove(order_payload) # update stack
                    print(f'FILLED LONG @{order_p}')

            elif side == 'short': 
                if top_bid_p>=order_p: # bid at least greater than sell limit to fill
                    # short short limit metadata 
                    result[product].append(Order(product, order_p, -order_q))
                    self.filled_orders[product].append(self._order_fill_logs(side, order_p, -order_q, order_p-pnl_short_ticks, 0))  # update fill stack
                    self.resting_orders[product].remove(order_payload) # update stack
                    print(f'FILLED SHORT @{order_p}')
    
        return
    
    def update_fill_orders(self, 
                           product, 
                           result, 
                           top_bid_p, 
                           top_ask_p):
            """
            update order fills for product
            """
            for order_payload in list(self.filled_orders[product]): 
                side = order_payload['side']
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
    
    # helpers 
    def _order_fill_logs(self, side, order_price, order_quantity, take_profit_ticks, timestamp_since_hedge = 0): 
        return {
            'side': side,
            'price': order_price, 
            'quantity': order_quantity, 
            'target_price': take_profit_ticks, 
            'time_stamp_since_hedge': timestamp_since_hedge
        }

# ----------------------- SIGNAL TO MARKET ----------------------
class Trader: 
    def __init__(self): 
        self.order_logs: QueueLogic = QueueLogic()
        self.books_meta_data: LogBookMetadata = LogBookMetadata()
        self.eq_max: int = 10 


    def cointegration_pair(self, 
                           result: dict[List], 
                           product1: str, 
                           product2: str, 
                           order_depth_pair1: OrderDepth, 
                           order_depth_pair2: OrderDepth, 
                           state: TradingState):
        """
        cointegration code for cointegrated products
        cointegration is tested using one lag ADF autoregresive model 
        """
        # strategy params 
        oq = 1
        standard_spread = np.inf
        entry_threshold = 1.5
        exit_threshold = 0.5
        oq_x2 = 0

        # price series 
        x1_prices = self.books_meta_data.product_price_data[product1]
        x2_prices = self.books_meta_data.product_price_data[product2]

        # books 
        x1_book: BookHelpers = BookHelpers(product1, order_depth_pair1, state)
        x2_book: BookHelpers = BookHelpers(product2, order_depth_pair2, state)

        # book metadata 
        x1_pos_valid = x1_book.quantity_remaining(self.eq_max)>0
        x2_pos_valid = x2_book.quantity_remaining(self.eq_max)>0

        (x1_top_bid, _), (x1_top_ask, _) = x1_book.bid_from_top(), x1_book.ask_from_top()

        (x2_top_bid, _), (x2_top_ask, _) = x2_book.bid_from_top(), x2_book.ask_from_top()

        if (x1_top_bid is None or x1_top_ask is None) or (x2_top_bid is None or x2_top_ask is None): 
            return result

        # signals and initial orders
        cointegration: CoIntegration = CoIntegration(x1_prices, x2_prices, state)

        if len(x1_prices)<cointegration.lookback or len(x2_prices)<cointegration.lookback:
            return result

        resid = cointegration.residuals()
        standard_spread, x2_hedge = cointegration.spread_hedge()
        if cointegration.ADF_test(resid): # test for cointegration over lookback 
            if x1_pos_valid and x2_pos_valid:
                oq_x2 = int(round(x2_hedge*oq))
                if oq_x2 == 0:
                    return result # check hedged order quantity is sufficiently large
                # orders
                if standard_spread > entry_threshold: 
                    # high spread -> short x1; long beta*x2
                    self.order_logs.resting_orders[product1].append(
                        {'side': 'short', 'price': x1_top_ask, 'quantity': self.eq_max}
                    )
                elif standard_spread < -entry_threshold:
                    # low spread -> long x1; short beta*x2
                    self.order_logs.resting_orders[product1].append(
                        {'side': 'long', 'price': x1_top_bid, 'quantity': self.eq_max}
                    )

        # resting order update
        x2_order_data = {
            'x2_short': (product2, x2_top_bid, -oq_x2), 
            'x2_long': (product2, x2_top_ask, oq_x2)
        }
        x2_order_fill_logs = {
            'x2_short': self.order_logs._order_fill_logs('short', x2_top_bid, -oq_x2, 0), 
            'x2_long': self.order_logs._order_fill_logs('long', x2_top_ask, oq_x2, 0)
        }
        self.order_logs.update_pairs_trades(result, 
                                            product1, 
                                            product2,
                                            x1_top_ask, 
                                            x1_top_bid, 
                                            x2_order_data, 
                                            x2_order_fill_logs)

        # mean reversion 
        for pair_id, order_payload in list(self.order_logs.pairs_orders.items()):
            if order_payload:
                x1 = order_payload[0]
                x2 = order_payload[1]

                x1_q = x1['quantity']
                x1_side = x1['side']

                hedged_qty = x2['quantity']

                # state based mean reversion exit 
                if abs(standard_spread) < exit_threshold:
                    if x1_side == 'long':
                        result[product1].append(Order(product1, x1_top_bid, -x1_q))
                        result[product2].append(Order(product2, x2_top_ask, -hedged_qty))
                        del self.order_logs.pairs_orders[pair_id]

                    elif x1_side == 'short':
                        result[product1].append(Order(product1, x1_top_ask, -x1_q))
                        result[product2].append(Order(product2, x2_top_bid, -hedged_qty))
                        del self.order_logs.pairs_orders[pair_id]

        return result

    def run(self, 
            state: TradingState):
        """
        stratgey execution and orders to market function 
        """
        result = defaultdict(list)
        traderData = ''
        conversions = 0

        # log metadata
        self.books_meta_data.log_price(state)

        # cointegrated strategy 
        for pair in pairs:
            prod1, prod2 = pair[0], pair[1]
            # send orders
            od1 = state.order_depths[prod1]
            od2 = state.order_depths[prod2]

            if self.books_meta_data.full_book(od1) and self.books_meta_data.full_book(od2): 
                 self.cointegration_pair(result, 
                                         prod1, 
                                         prod2, 
                                         od1, 
                                         od2, 
                                         state)
                

        return result, conversions, traderData

        