import unittest
import numpy as np

def compute_safety_stock(z_val, std_daily_demand, lead_time):
    return int(round(z_val * std_daily_demand * np.sqrt(lead_time)))

def compute_reorder_point(avg_daily_demand, lead_time, safety_stock):
    return int(round((avg_daily_demand * lead_time) + safety_stock))

def compute_eoq(avg_daily_demand, ordering_cost, holding_cost_day):
    # EOQ = sqrt( 2 * D * S / H )
    # D = avg_daily_demand * 365
    # H = holding_cost_day * 365
    # EOQ = sqrt( 2 * avg_daily_demand * ordering_cost / holding_cost_day )
    annual_demand = avg_daily_demand * 365
    holding_cost_year = holding_cost_day * 365
    return int(round(np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_year)))

class TestInventoryCalculations(unittest.TestCase):
    
    def test_safety_stock(self):
        # Setup parameters
        z_val = 1.64 # 95% service level
        std_demand = 2.0
        lead_time = 7
        
        # Calculate
        ss = compute_safety_stock(z_val, std_demand, lead_time)
        
        # Expected: 1.64 * 2.0 * sqrt(7) = 3.28 * 2.64575 = 8.678 => rounded to 9
        self.assertEqual(ss, 9)

    def test_reorder_point(self):
        avg_demand = 10.0
        lead_time = 7
        safety_stock = 9
        
        rp = compute_reorder_point(avg_demand, lead_time, safety_stock)
        
        # Expected: 10 * 7 + 9 = 79
        self.assertEqual(rp, 79)

    def test_eoq(self):
        avg_demand = 10.0
        ordering_cost = 500.0
        holding_cost_day = 2.50
        
        eoq = compute_eoq(avg_demand, ordering_cost, holding_cost_day)
        
        # Expected: sqrt(2 * 3650 * 500 / (2.5 * 365)) = sqrt(3650000 / 912.5) = sqrt(4000) = 63.24 => rounded to 63
        self.assertEqual(eoq, 63)

if __name__ == "__main__":
    unittest.main()
