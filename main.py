import numpy as np
import matplotlib.pyplot as plt

class FuzzyController:
    def __init__(self, temp, humidity):
        self.temp = temp
        self.humidity = humidity

    def fuzzify_temp(self):
        cold = max(min((25 - self.temp) / 10, 1), 0)  # Cold if temp < 25
        warm = max(min((self.temp - 20) / 10, (30 - self.temp) / 5, 1), 0)  # Warm if 20 < temp < 30
        hot = max(min((self.temp - 28) / 5, 1), 0)  # Hot if temp > 28
        return cold, warm, hot

    def fuzzify_humidity(self):
        low = max(min((50 - self.humidity) / 30, 1), 0)  # Low humidity < 50
        medium = max(min((self.humidity - 40) / 20, (70 - self.humidity) / 20, 1), 0)  # Medium humidity 40-70
        high = max(min((self.humidity - 60) / 20, 1), 0)  # High humidity > 60
        return low, medium, high

    def calculate_pwm(self):
        raise NotImplementedError("This method should be overridden by subclasses")

class MamdaniController(FuzzyController):
    def __init__(self, temp, humidity):
        super().__init__(temp, humidity)

    def inference(self):
        cold, warm, hot = self.fuzzify_temp()
        low, medium, high = self.fuzzify_humidity()
        rule1 = min(cold, low) 
        rule2 = min(warm, medium) 
        rule3 = min(warm, high)  
        rule4 = max(min(hot, low), min(hot, medium)) 
        rule5 = min(hot, high) 
        return rule1, rule2, rule3, rule4, rule5

    def defuzzify(self, rules):
        pwm_range = np.arange(0, 256, 1)
        comfortable_pwm = np.minimum(rules[0], np.maximum(1 - pwm_range / 50, 0))  # 0-50
        slightly_uncomfortable_pwm = np.minimum(rules[1], np.maximum(1 - np.abs(pwm_range - 100) / 50, 0))  # 50-150
        very_uncomfortable_pwm = np.minimum(rules[2], np.maximum(1 - np.abs(pwm_range - 150) / 50, 0))  # 100-200
        dangerous_pwm = np.minimum(rules[3], np.maximum(1 - np.abs(pwm_range - 200) / 50, 0))  # 150-250
        heat_stroke_risk_pwm = np.minimum(rules[4], pwm_range / 255)  # 200-25
        aggregated = np.fmax(comfortable_pwm, 
                     np.fmax(slightly_uncomfortable_pwm, 
                     np.fmax(very_uncomfortable_pwm, 
                     np.fmax(dangerous_pwm, heat_stroke_risk_pwm))))
        centroid = int(np.sum(aggregated * pwm_range) / np.sum(aggregated)) if np.sum(aggregated) != 0 else 0
        return centroid

    def calculate_pwm(self):
        rules = self.inference()
        pwm_value = self.defuzzify(rules)
        return pwm_value
    
    def plot_fuzzy_sets(self, rules):
        temp_range = np.linspace(10, 43, 100)  # Temperature from 10°C to 43°C
        humidity_range = np.linspace(0, 100, 100)  # Humidity from 0% to 100%
        
        pwm_range = np.arange(0, 256, 1)
        
        cold_vals = np.maximum(np.minimum((25 - temp_range) / 10, 1), 0)
        warm_vals = np.maximum(np.minimum((temp_range - 20) / 10, (30 - temp_range) / 5), 0)
        hot_vals = np.maximum(np.minimum((temp_range - 28) / 5, 1), 0)
        
        low_vals = np.maximum(np.minimum((50 - humidity_range) / 30, 1), 0)
        medium_vals = np.maximum(np.minimum((humidity_range - 40) / 20, (70 - humidity_range) / 20), 0)
        high_vals = np.maximum(np.minimum((humidity_range - 60) / 20, 1), 0)

        comfortable_pwm = np.minimum(rules[0], np.maximum(1 - pwm_range / 50, 0))  # 0-50
        slightly_uncomfortable_pwm = np.minimum(rules[1], np.maximum(1 - np.abs(pwm_range - 100) / 50, 0))  # 50-150
        very_uncomfortable_pwm = np.minimum(rules[2], np.maximum(1 - np.abs(pwm_range - 150) / 50, 0))  # 100-200
        dangerous_pwm = np.minimum(rules[3], np.maximum(1 - np.abs(pwm_range - 200) / 50, 0))  # 150-250
        heat_stroke_risk_pwm = np.minimum(rules[4], pwm_range / 255)  # 200-255

        aggregated = np.fmax(comfortable_pwm, 
                     np.fmax(slightly_uncomfortable_pwm, 
                     np.fmax(very_uncomfortable_pwm, 
                     np.fmax(dangerous_pwm, heat_stroke_risk_pwm))))
        
        centroid = int(np.sum(aggregated * pwm_range) / np.sum(aggregated)) if np.sum(aggregated) != 0 else 0

        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 2, 1)
        plt.plot(temp_range, cold_vals, label='Cold', color='blue')
        plt.plot(temp_range, warm_vals, label='Warm', color='orange')
        plt.plot(temp_range, hot_vals, label='Hot', color='red')
        plt.title('Fuzzy Sets for Temperature')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(humidity_range, low_vals, label='Low', color='green')
        plt.plot(humidity_range, medium_vals, label='Medium', color='purple')
        plt.plot(humidity_range, high_vals, label='High', color='red')
        plt.title('Fuzzy Sets for Humidity')
        plt.xlabel('Humidity (%)')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(pwm_range, comfortable_pwm, label='Comfortable', color='blue')
        plt.plot(pwm_range, slightly_uncomfortable_pwm, label='Slightly Uncomfortable', color='orange')
        plt.plot(pwm_range, very_uncomfortable_pwm, label='Very Uncomfortable', color='green')
        plt.plot(pwm_range, dangerous_pwm, label='Dangerous', color='red')
        plt.plot(pwm_range, heat_stroke_risk_pwm, label='Heat Stroke Risk', color='purple')
        plt.title('Rule Outputs for PWM')
        plt.xlabel('PWM Value')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(pwm_range, aggregated, label='Aggregated Output', color='black', linewidth=2, linestyle='--')
        plt.axvline(centroid, color='red', linestyle='--', label=f'Centroid: {centroid}')
        plt.title('Aggregated Fuzzy Output')
        plt.xlabel('PWM Value')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

class TakagiSugenoController(FuzzyController):
    def __init__(self, temp, humidity):
        super().__init__(temp, humidity)

    def inference(self):
        cold, warm, hot = self.fuzzify_temp()
        low, medium, high = self.fuzzify_humidity()

        rule1 = min(cold, low)                        # Comfortable
        rule2 = min(warm, medium)                     # Slightly Uncomfortable
        rule3 = min(warm, high)                       # Very Uncomfortable
        rule4 = max(min(hot, low), min(hot, medium))  # Dangerous
        rule5 = min(hot, high)                        # Heat Stroke Risk

        return rule1, rule2, rule3, rule4, rule5

    def defuzzify(self, rules):
        output_rule1 = 50   # Comfortable
        output_rule2 = 100  # Slightly Uncomfortable
        output_rule3 = 150  # Very Uncomfortable
        output_rule4 = 200  # Dangerous
        output_rule5 = 255  # Heat Stroke Risk
        output_rules = [output_rule1, output_rule2, output_rule3, output_rule4, output_rule5]

        numerator = 0
        
        for i in range(len(rules)) :
            numerator += (rules[i] * output_rules[i])
        
        denominator = sum(rules)
        
        return numerator / denominator if denominator != 0 else 0

    def calculate_pwm(self):
        rules = self.inference()
        pwm_value = int(self.defuzzify(rules))
        return pwm_value

def test_fuzzy_systems(temp, humidity):
    mamdani_controller = MamdaniController(temp, humidity)
    mamdani_pwm = mamdani_controller.calculate_pwm()
    rules = mamdani_controller.inference()
    mamdani_controller.plot_fuzzy_sets(rules)
    print(f"Fan Speed PWM (Mamdani): {mamdani_pwm}")

    ts_controller = TakagiSugenoController(temp, humidity)
    ts_pwm = ts_controller.calculate_pwm()
    print(f"Fan Speed PWM (Takagi-Sugeno): {ts_pwm}")

test_fuzzy_systems(temp=37, humidity=83)
