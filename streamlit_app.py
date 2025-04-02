import streamlit as st
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
from typing import Dict, Optional, Tuple  # For type hints
from datetime import datetime, timedelta  # For date handling
import logging  # For logging program events
import blpapi  # Bloomberg API interface
import asyncio  # For asynchronous operations
import threading  # For multi-threading (loading animation)
import time  # For timing operations
import sys  # For system-specific parameters and functions
import yfinance as yf  # For fetching Yahoo Finance data
from sklearn.ensemble import RandomForestRegressor  # Machine learning model
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.model_selection import train_test_split  # For splitting data into training/testing sets
from sklearn.metrics import mean_squared_error  # For evaluating model performance
import os  # For operating system interactions

# Configure logging to write to a file named 'atlas.log'
logging.basicConfig(
    level=logging.DEBUG,  # Log all levels of messages
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format with timestamp
    filename='atlas.log',  # Output file
    filemode='a'  # Append mode
)
logger = logging.getLogger(__name__)  # Get logger instance

class AtlasStockScanner:
    """Class to scan and analyze stock data using various metrics and ML predictions."""
    
    def __init__(self, ticker: str, strategy: str = "balanced"):
        """Initialize the stock scanner with a ticker and strategy."""
        self.ticker = ticker.upper()  # Convert ticker to uppercase
        self.strategy = strategy.lower()  # Convert strategy to lowercase
        self.weights = self._set_strategy_weights()  # Set scoring weights based on strategy
        self.benchmark = "SPY"  # Set S&P 500 as benchmark
        self.loading = True  # Flag for loading animation
        self.data, self.peers = self._fetch_with_loading()  # Fetch data with loading animation
        self.missing_fields = []  # List to track missing data fields
        self.model, self.scaler = self._train_ml_model()  # Train ML model and scaler
        self.ml_prediction = self._predict_return()  # Get ML-predicted return
        self.bsh_score = self._get_bsh_score()  # Calculate Buy/Sell/Hold score

    def _set_strategy_weights(self) -> Dict[str, float]:
        """Set the weights for different scoring factors based on chosen strategy."""
        # Define default weights for different strategies
        strategies = {
            "growth": {
                'financial_health': 0.25, 
                'market_performance': 0.20, 
                'valuation': 0.15,
                'sector_trends': 0.15, 
                'analyst_recommendations': 0.15,
                'technical': 0.10, 
                'earnings_quality': 0.10
            },
            "value": {
                'financial_health': 0.20, 
                'market_performance': 0.15, 
                'valuation': 0.25,
                'sector_trends': 0.10, 
                'analyst_recommendations': 0.15,
                'technical': 0.10, 
                'earnings_quality': 0.15
            },
            "balanced": {
                'financial_health': 0.25, 
                'market_performance': 0.20, 
                'valuation': 0.15,
                'sector_trends': 0.15, 
                'analyst_recommendations': 0.15,
                'technical': 0.10, 
                'earnings_quality': 0.10
            }
        }
        default_weights = strategies.get(self.strategy, strategies["balanced"])  # Get weights or default to balanced
        
        # Display default weights and offer customization
        print(f"\nDefault weights for '{self.strategy}' strategy:")
        for factor, weight in default_weights.items():
            print(f"{factor.replace('_', ' ').title():<25} {weight:.2f}")
        response = input("Would you like to customize these weights? (y/n): ").strip().lower()
        
        if response == 'y':
            custom_weights = {}
            total = 0.0
            factors = list(default_weights.keys())
            # Prompt for custom weights for each factor
            for factor in factors:
                while True:
                    try:
                        weight = float(input(f"Enter weight for {factor.replace('_', ' ').title()} (0.0-1.0, remaining: {1.0 - total:.2f}): ").strip())
                        if 0 <= weight <= 1 and total + weight <= 1:
                            custom_weights[factor] = weight
                            total += weight
                            break
                        else:
                            print("Invalid weight. Must be between 0 and 1, and total must not exceed 1.")
                    except ValueError:
                        print("Please enter a valid number.")
            # Distribute remaining weight evenly if total < 1
            if total < 1:
                print(f"Total weight is {total:.2f}. Remaining {1 - total:.2f} will be distributed evenly.")
                remaining = (1 - total) / len(factors)
                for factor in factors:
                    custom_weights[factor] += remaining
            return custom_weights
        return default_weights

    def _loading_animation(self):
        """Display a loading animation while data is being fetched."""
        animation = "|/-\\"  # Animation characters
        idx = 0
        while self.loading:  # Continue until loading flag is False
            sys.stdout.write(f"\rFetching data for {self.ticker} {animation[idx % len(animation)]}")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)  # Update every 0.1 seconds
        sys.stdout.write("\rData fetched successfully!          \n")
        sys.stdout.flush()

    def _fetch_with_loading(self) -> Tuple[Dict, list]:
        """Fetch data while displaying a loading animation."""
        self.loading = True
        loading_thread = threading.Thread(target=self._loading_animation)  # Start animation in separate thread
        loading_thread.start()
        
        try:
            result = asyncio.run(self._fetch_data())  # Fetch data asynchronously
            self.loading = False  # Stop animation
            loading_thread.join()  # Wait for animation thread to finish
            return result
        except Exception as e:
            self.loading = False
            loading_thread.join()
            raise e  # Re-raise any exceptions

    def _fetch_bloomberg_data_sync(self, request_type: str, securities: list, fields: list, start_date: str = None, end_date: str = None) -> Dict:
        """Synchronously fetch data from Bloomberg API."""
        session = blpapi.Session()  # Start Bloomberg session
        if not session.start():
            logger.error("Failed to start Bloomberg session")
            return {}
        
        service_name = "//blp/refdata"  # Bloomberg reference data service
        if not session.openService(service_name):
            logger.error(f"Failed to open {service_name} service")
            return {}
        
        service = session.getService(service_name)
        request = service.createRequest(request_type)  # Create request of specified type
        
        # Add securities and fields to request
        for sec in securities:
            request.getElement("securities").appendValue(sec)
        for field in fields:
            request.getElement("fields").appendValue(field)
        if request_type == "HistoricalDataRequest":
            request.set("startDate", start_date)
            request.set("endDate", end_date)
        
        session.sendRequest(request)  # Send the request
        
        data = {}
        while True:
            event = session.nextEvent(500)  # Wait for response with 500ms timeout
            if event.eventType() == blpapi.Event.RESPONSE:
                for msg in event:
                    if request_type == "ReferenceDataRequest":
                        security_data = msg.getElement("securityData")
                        for i in range(security_data.numValues()):
                            sec = security_data.getValueAsElement(i)
                            ticker = sec.getElementAsString("security")
                            field_data = sec.getElement("fieldData")
                            data[ticker] = {}
                            # Process each field in the response
                            for field in field_data.elements():
                                field_name = str(field.name())
                                logger.debug(f"Processing field {field_name} for {ticker}, datatype: {field.datatype()}, numValues: {field.numValues()}")
                                
                                if field.datatype() == blpapi.DataType.STRING:
                                    value = field.getValueAsString() if field.numValues() > 0 else ""
                                    data[ticker][field_name] = value
                                    logger.debug(f"String field {field_name} value: '{value}'")
                                    if not value and field_name in ["INDUSTRY_SECTOR"]:
                                        logger.warning(f"Field {field_name} returned empty for {ticker}")
                                elif field_name == "BLOOMBERG_PEERS":
                                    values = []
                                    if field.numValues() > 0:
                                        for j in range(field.numValues()):
                                            peer_element = field.getValueAsElement(j)
                                            if peer_element.numElements() > 0:
                                                for k in range(peer_element.numElements()):
                                                    sub_field = peer_element.getElement(k)
                                                    if sub_field.datatype() == blpapi.DataType.STRING:
                                                        values.append(sub_field.getValueAsString())
                                                        break
                                            else:
                                                logger.warning(f"BLOOMBERG_PEERS element {j} has no sub-fields for {ticker}")
                                    data[ticker][field_name] = ",".join(values) if values else ""
                                    logger.debug(f"BLOOMBERG_PEERS value: '{data[ticker][field_name]}'")
                                    if not values:
                                        logger.warning(f"Field {field_name} returned no valid peer values for {ticker}")
                                elif field.numValues() > 1 or field.isArray():
                                    values = [field.getValueAsString(i) for i in range(field.numValues())]
                                    data[ticker][field_name] = ",".join(values) if values else ""
                                    logger.debug(f"Array field {field_name} value: '{data[ticker][field_name]}'")
                                    if not values:
                                        logger.warning(f"Field {field_name} returned no values for {ticker}")
                                else:
                                    value = field.getValueAsFloat() if field.numValues() > 0 else None
                                    data[ticker][field_name] = value
                                    logger.debug(f"Numeric field {field_name} value: {value}")
                                    if value is None and field_name in ["CURR_ENTP_VAL"]:
                                        logger.warning(f"Field {field_name} returned no value for {ticker}")
                                    elif value == 0 and field_name in ["CURR_ENTP_VAL"]:
                                        logger.warning(f"Field {field_name} returned 0 for {ticker}")
                    elif request_type == "HistoricalDataRequest":
                        security_data = msg.getElement("securityData")
                        ticker = security_data.getElementAsString("security")
                        field_data = security_data.getElement("fieldData")
                        dates = []
                        values = {field: [] for field in fields}
                        # Process historical data into a DataFrame
                        for entry in field_data.values():
                            date = entry.getElementAsString("date")
                            dates.append(date)
                            for field in fields:
                                values[field].append(entry.getElementAsFloat(field) if entry.hasElement(field) else np.nan)
                        if dates:
                            data[ticker] = pd.DataFrame(values, index=pd.to_datetime(dates))
                        else:
                            logger.warning(f"No historical data returned for {ticker}")
                            data[ticker] = pd.DataFrame()
                break
        
        session.stop()  # Close the Bloomberg session
        return data

    def _fetch_yahoo_finance_data(self, ticker: str) -> Dict:
        """Fetch financial data from Yahoo Finance as a fallback."""
        try:
            stock = yf.Ticker(ticker)  # Create Yahoo Finance ticker object
            info = stock.info  # Get stock info dictionary
            yahoo_data = {
                'EBITDA': info.get('ebitda', 0),
                'CF_FREE_CASH_FLOW': info.get('freeCashflow', 0),
                'BEST_NET_INCOME': info.get('netIncomeToCommon', 0),
                'RETURN_ON_EQY': info.get('returnOnEquity', 0) * 100,  # Convert to percentage
                'TRAIL_12M_NET_REVENUE': info.get('totalRevenue', 0),
                'CURR_ENTP_VAL': info.get('enterpriseValue', 0)
            }
            logger.info(f"Fetched Yahoo Finance data for {ticker}: {yahoo_data}")
            return yahoo_data
        except Exception as e:
            logger.error(f"Failed to fetch Yahoo Finance data for {ticker}: {str(e)}")
            return {}

    def _prompt_for_missing_values(self, current_data: Dict, fields_to_check: list) -> Dict:
        """Prompt user to manually input values for missing or zero fields."""
        for field in fields_to_check:
            if current_data.get(field) is None or (isinstance(current_data.get(field), (int, float)) and current_data.get(field) == 0):
                while True:
                    try:
                        prompt = f"Field {field} not found or 0 for {self.ticker}. Enter value manually "
                        # Customize prompt based on field type
                        if field in ["EBITDA", "CF_FREE_CASH_FLOW", "BEST_NET_INCOME", "CURR_ENTP_VAL", "TRAIL_12M_NET_REVENUE"]:
                            prompt += "(in billions, e.g., 60 for $60B, or press Enter to skip): "
                        elif field == "RETURN_ON_EQY":
                            prompt += "(as percentage, e.g., 15.5, or press Enter to skip): "
                        else:
                            prompt += "(or press Enter to skip): "
                        value = input(prompt).strip()
                        if value == "":
                            print(f"Skipping {field}, keeping as N/A.")
                            break
                        if field in ["EBITDA", "CF_FREE_CASH_FLOW", "BEST_NET_INCOME", "CURR_ENTP_VAL", "TRAIL_12M_NET_REVENUE"]:
                            current_data[field] = float(value) * 1e9  # Convert billions to raw value
                        else:
                            current_data[field] = float(value)
                        logger.info(f"User manually set {field} to {current_data[field]} for {self.ticker}")
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number or press Enter to skip.")
        return current_data

    async def _fetch_data(self) -> Tuple[Dict, list]:
        """Asynchronously fetch historical and reference data from Bloomberg."""
        # Define fields to fetch
        hist_fields = ["LAST_CLOSE_TRR_1YR", "VOLATILITY_260D", "CUR_MKT_CAP"]
        ref_fields = [
            "SALES_GROWTH", "ARDR_NON_GAAP_GROSS_MARGIN_PCT", "TOT_DEBT_TO_TOT_EQY",
            "OPER_MARGIN", "NET_DEBT_TO_EBITDA", "CUR_RATIO", "QUICK_RATIO", "FCF_TO_TOTAL_DEBT",
            "ASSET_TURNOVER", "TOT_DEBT_TO_TOT_CAP", "GROSS_MARGIN", "CAP_EXPEND_TO_SALES", "SALES_5YR_AVG_GR",
            "FCF_YIELD_WITH_CUR_ENTP_VAL", "VOLATILITY_90D", "EQY_BETA", "EQY_ALPHA", "CUR_MKT_CAP",
            "LAST_CLOSE_TRR_1YR", "VOLATILITY_260D", "BETA_ADJUSTED", "PE_RATIO", "PX_TO_SALES_RATIO",
            "BEST_CUR_EV_TO_EBITDA", "PX_TO_FREE_CASH_FLOW", "BEST_PEG_RATIO_CUR_YR", "BOOK_VAL_PER_SH",
            "CURRENT_EV_TO_12M_SALES", "PX_TO_BOOK_RATIO", "EQY_DVD_YLD_IND", "EPS_GROWTH",
            "INDUSTRY_SECTOR", "PRICE_TO_INDUSTRY_SPEC_EBITDA", "SHORT_INT_RATIO",
            "RSI_14D", "MACD", "MOV_AVG_50D", "MOV_AVG_200D", "EQY_BOLLINGER_UPPER",
            "EARNINGS_SURPRISE_PCT", "CFO_TO_SALES", "AVG_ANALYST_RATING", "TOT_ANALYST_REC",
            "BEST_ANALYST_RATING", "BLOOMBERG_PEERS"
        ]
        
        # Set date range for historical data (5 years)
        start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")
        
        loop = asyncio.get_event_loop()
        # Fetch historical and reference data concurrently
        hist_data = await loop.run_in_executor(None, self._fetch_bloomberg_data_sync, 
                                               "HistoricalDataRequest", 
                                               [f"{self.ticker} US Equity"], 
                                               hist_fields, start_date, end_date)
        ref_data = await loop.run_in_executor(None, self._fetch_bloomberg_data_sync, 
                                              "ReferenceDataRequest", 
                                              [f"{self.ticker} US Equity"], 
                                              ref_fields)
        
        # Process peer data
        peer_group = ref_data.get(f"{self.ticker} US Equity", {}).get("BLOOMBERG_PEERS", "")
        peers = [p.strip() for p in peer_group.split(',')] if peer_group and isinstance(peer_group, str) else ["005930 KS Equity", "2498 TT Equity"]
        self.peers = peers
        if peers:
            peer_data = await loop.run_in_executor(None, self._fetch_bloomberg_data_sync, 
                                                   "ReferenceDataRequest", 
                                                   peers, 
                                                   ref_fields)
        else:
            peer_data = {}
            logger.warning(f"No peers found for {self.ticker}, using defaults")
        
        # Process historical data into yearly averages
        historical_df = hist_data.get(f"{self.ticker} US Equity", pd.DataFrame())
        if not historical_df.empty:
            historical_df = historical_df.resample('YE').mean().reset_index()
            historical_df['year'] = historical_df['index'].dt.year
            historical_df['future_return'] = historical_df['LAST_CLOSE_TRR_1YR'].shift(-1)
            logger.info(f"Historical data rows for {self.ticker}: {len(historical_df)}, Columns: {list(historical_df.columns)}")
        else:
            logger.warning(f"No historical data fetched for {self.ticker}")
        
        current_data = ref_data.get(f"{self.ticker} US Equity", {})
        peer_data = {p.split()[0]: peer_data.get(p, {}) for p in peers}
        
        # Supplement with Yahoo Finance data
        yahoo_data = self._fetch_yahoo_finance_data(self.ticker)
        current_data['EBITDA'] = yahoo_data.get('EBITDA', 0)
        current_data['CF_FREE_CASH_FLOW'] = yahoo_data.get('CF_FREE_CASH_FLOW', 0)
        current_data['BEST_NET_INCOME'] = yahoo_data.get('BEST_NET_INCOME', 0)
        current_data['RETURN_ON_EQY'] = yahoo_data.get('RETURN_ON_EQY', 0)
        current_data['TRAIL_12M_NET_REVENUE'] = yahoo_data.get('TRAIL_12M_NET_REVENUE', 0)
        if 'CURR_ENTP_VAL' not in current_data or current_data['CURR_ENTP_VAL'] is None or current_data['CURR_ENTP_VAL'] == 0:
            current_data['CURR_ENTP_VAL'] = yahoo_data.get('CURR_ENTP_VAL', 0)
        
        # Prompt for missing critical fields
        critical_fields = ["EBITDA", "CF_FREE_CASH_FLOW", "BEST_NET_INCOME", "CURR_ENTP_VAL", "RETURN_ON_EQY", "TRAIL_12M_NET_REVENUE"]
        current_data = self._prompt_for_missing_values(current_data, critical_fields)
        
        # Check for missing required fields
        required_fields = ["SALES_GROWTH", "TOT_DEBT_TO_TOT_EQY", "LAST_CLOSE_TRR_1YR", "VOLATILITY_260D", "PE_RATIO"]
        self.missing_fields = [f for f in required_fields if f not in current_data or (current_data[f] is None or (isinstance(current_data[f], (int, float)) and current_data[f] == 0))]
        if self.missing_fields:
            logger.warning(f"Missing or zero fields for {self.ticker}: {self.missing_fields}")
        
        # Adjust debt ratios if they seem to be in percentage form
        if current_data.get('TOT_DEBT_TO_TOT_EQY') is not None and current_data.get('TOT_DEBT_TO_TOT_EQY', 0) > 2.5:
            current_data['TOT_DEBT_TO_TOT_EQY'] /= 100
        if current_data.get('TOT_DEBT_TO_TOT_CAP') is not None and current_data.get('TOT_DEBT_TO_TOT_CAP', 0) > 2.5:
            current_data['TOT_DEBT_TO_TOT_CAP'] /= 100
        
        # Set default values for sector data
        current_data['sector_performance'] = 10.0
        current_data['industry_pe'] = current_data.get('PRICE_TO_INDUSTRY_SPEC_EBITDA', 30.0)
        
        # Set Buy/Sell/Hold data with defaults
        current_data['bsh'] = {
            'consensus': 'Buy' if current_data.get('AVG_ANALYST_RATING', 3.0) > 3.5 else 'Hold' if current_data.get('AVG_ANALYST_RATING', 3.0) > 2.5 else 'Sell',
            'avg_rating': current_data.get('AVG_ANALYST_RATING', 4.2),
            'num_analysts': current_data.get('TOT_ANALYST_REC', 35),
            'buy_count': int(current_data.get('TOT_ANALYST_REC', 35) * 0.7),
            'hold_count': int(current_data.get('TOT_ANALYST_REC', 35) * 0.25),
            'sell_count': int(current_data.get('TOT_ANALYST_REC', 35) * 0.05)
        }
        
        # Structure all data into a dictionary, excluding portfolio data
        ticker_data = {
            'historical': historical_df,
            'current': {
                'financial': {k: current_data.get(k, None) for k in ['SALES_GROWTH', 'ARDR_NON_GAAP_GROSS_MARGIN_PCT', 'TOT_DEBT_TO_TOT_EQY', 'OPER_MARGIN', 'NET_DEBT_TO_EBITDA', 'CUR_RATIO', 'QUICK_RATIO', 'FCF_TO_TOTAL_DEBT', 'EBITDA', 'CF_FREE_CASH_FLOW', 'BEST_NET_INCOME', 'ASSET_TURNOVER', 'TOT_DEBT_TO_TOT_CAP', 'RETURN_ON_EQY', 'GROSS_MARGIN', 'CURR_ENTP_VAL', 'CAP_EXPEND_TO_SALES', 'SALES_5YR_AVG_GR', 'FCF_YIELD_WITH_CUR_ENTP_VAL']},
                'market': {k: current_data.get(k, None) for k in ['LAST_CLOSE_TRR_1YR', 'VOLATILITY_260D', 'TRAIL_12M_NET_REVENUE', 'VOLATILITY_90D', 'EQY_BETA', 'EQY_ALPHA', 'CUR_MKT_CAP', 'BETA_ADJUSTED']},
                'valuation': {k: current_data.get(k, None) for k in ['PE_RATIO', 'PX_TO_SALES_RATIO', 'BEST_CUR_EV_TO_EBITDA', 'PX_TO_FREE_CASH_FLOW', 'BEST_PEG_RATIO_CUR_YR', 'BOOK_VAL_PER_SH', 'CURRENT_EV_TO_12M_SALES', 'PX_TO_BOOK_RATIO', 'EQY_DVD_YLD_IND', 'EPS_GROWTH']},
                'sector': {'sector_performance': current_data['sector_performance'], 'industry_pe': current_data['industry_pe'], 'industry_sector': current_data.get('INDUSTRY_SECTOR', 'Unknown')},
                'technical': {k: current_data.get(k, None) for k in ['RSI_14D', 'MACD', 'MOV_AVG_50D', 'MOV_AVG_200D', 'EQY_BOLLINGER_UPPER']},
                'earnings': {'surprise_pct': current_data.get('EARNINGS_SURPRISE_PCT', 5.0), 'consistency': 0.95, 'cfo_to_sales': current_data.get('CFO_TO_SALES', None)},
                'bsh': current_data['bsh']
            },
            'peers': peer_data
        }
        return ticker_data, peers

    def _get_bsh_score(self) -> float:
        """Calculate the Buy/Sell/Hold score based on analyst recommendations."""
        bsh_data = self.data['current']['bsh']
        total_analysts = bsh_data['num_analysts']
        if total_analysts == 0:
            return 50.0  # Default score if no analysts
        buy_weight = bsh_data['buy_count'] / total_analysts * 100
        hold_weight = bsh_data['hold_count'] / total_analysts * 50
        sell_weight = bsh_data['sell_count'] / total_analysts * 0
        return (buy_weight + hold_weight + sell_weight)

    def _train_ml_model(self) -> Tuple[RandomForestRegressor, StandardScaler]:
        """Train a Random Forest model for return prediction."""
        df = self.data['historical']
        features = ['LAST_CLOSE_TRR_1YR', 'VOLATILITY_260D', 'CUR_MKT_CAP']  # Features for ML model
        scaler = StandardScaler()  # Initialize scaler
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)  # Initialize RF model
        
        # Handle case with no or insufficient historical data
        if df.empty or not all(f in df.columns for f in features) or df['LAST_CLOSE_TRR_1YR'].dropna().empty:
            logger.warning(f"No valid historical data for {self.ticker}; using current data as fallback")
            current = self.data['current']
            X_dummy = pd.DataFrame([[current['market']['LAST_CLOSE_TRR_1YR'] or 0, 
                                    current['market']['VOLATILITY_260D'] or 0, 
                                    current['market']['CUR_MKT_CAP'] or 0]], columns=features)
            y_dummy = [current['market']['LAST_CLOSE_TRR_1YR'] or 0]
            scaler.fit(X_dummy)
            model.fit(scaler.transform(X_dummy), y_dummy)
        else:
            # Prepare features and target for training
            X = df[features].fillna({'LAST_CLOSE_TRR_1YR': self.data['current']['market']['LAST_CLOSE_TRR_1YR'] or 0, 
                                    'VOLATILITY_260D': df['VOLATILITY_260D'].mean(), 
                                    'CUR_MKT_CAP': df['CUR_MKT_CAP'].mean()})
            y = df['future_return'].fillna(self.data['current']['market']['LAST_CLOSE_TRR_1YR'] or 0)
            if len(X) > 1 and len(y) > 1:
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                logger.info(f"ML Model trained with historical data for {self.ticker}. MSE: {mse:.2f}")
            else:
                logger.warning(f"Insufficient historical data points ({len(X)}) for {self.ticker}; using current data")
                X_dummy = pd.DataFrame([[self.data['current']['market']['LAST_CLOSE_TRR_1YR'] or 0, 
                                        self.data['current']['market']['VOLATILITY_260D'] or 0, 
                                        self.data['current']['market']['CUR_MKT_CAP'] or 0]], columns=features)
                y_dummy = [self.data['current']['market']['LAST_CLOSE_TRR_1YR'] or 0]
                scaler.fit(X_dummy)
                model.fit(scaler.transform(X_dummy), y_dummy)
        
        return model, scaler

    def _predict_return(self) -> float:
        """Predict next year's return using the trained ML model."""
        current = self.data['current']
        features = [current['market']['LAST_CLOSE_TRR_1YR'] or 0, 
                    current['market']['VOLATILITY_260D'] or 0, 
                    current['market']['CUR_MKT_CAP'] or 0]
        feature_names = ['LAST_CLOSE_TRR_1YR', 'VOLATILITY_260D', 'CUR_MKT_CAP']
        X_current = pd.DataFrame([features], columns=feature_names)
        return self.model.predict(self.scaler.transform(X_current))[0]

    def _normalize_score(self, value: float, min_val: float, max_val: float, reverse: bool = False) -> float:
        """Normalize a value to a 0-100 scale."""
        try:
            scaled = np.clip((value - min_val) / (max_val - min_val), 0, 1) * 100
            return 100 - scaled if reverse else scaled  # Reverse if higher is worse
        except (TypeError, ZeroDivisionError):
            return 50.0  # Default to 50 if calculation fails

    def _calculate_sharpe(self, returns: float, volatility: float, rf_rate: float = 0.02) -> float:
        """Calculate the Sharpe Ratio."""
        if volatility > 0:
            return (returns - rf_rate) / volatility  # (Return - Risk Free Rate) / Volatility
        return 0

    def calculate_financial_health(self) -> float:
        """Calculate financial health score based on various metrics."""
        metrics = self.data['current']['financial']
        scores = {
            'revenue': self._normalize_score(metrics.get('SALES_GROWTH', 0) or 0, -10, 150),
            'margin': self._normalize_score(metrics.get('ARDR_NON_GAAP_GROSS_MARGIN_PCT', 0) or 0, 0, 80),
            'debt': self._normalize_score(metrics.get('TOT_DEBT_TO_TOT_EQY', 0) or 0, 2.5, 0, reverse=True),
            'oper_margin': self._normalize_score(metrics.get('OPER_MARGIN', 0) or 0, 0, 50),
            'net_debt_ebitda': self._normalize_score(metrics.get('NET_DEBT_TO_EBITDA', 0) or 0, 5, 0, reverse=True),
            'current_ratio': self._normalize_score(metrics.get('CUR_RATIO', 0) or 0, 0.5, 3),
            'quick_ratio': self._normalize_score(metrics.get('QUICK_RATIO', 0) or 0, 0.5, 2),
            'fcf_to_debt': self._normalize_score(metrics.get('FCF_TO_TOTAL_DEBT', 0) or 0, 0, 1),
            'ebitda': self._normalize_score(metrics.get('EBITDA', 0) or 0, 0, 1e11),
            'free_cash_flow': self._normalize_score(metrics.get('CF_FREE_CASH_FLOW', 0) or 0, 0, 1e11),
            'net_income': self._normalize_score(metrics.get('BEST_NET_INCOME', 0) or 0, 0, 1e11),
            'asset_turnover': self._normalize_score(metrics.get('ASSET_TURNOVER', 0) or 0, 0, 2),
            'debt_to_cap': self._normalize_score(metrics.get('TOT_DEBT_TO_TOT_CAP', 0) or 0, 1, 0, reverse=True),
            'roe': self._normalize_score(metrics.get('RETURN_ON_EQY', 0) or 0, -20, 100),
            'gross_margin': self._normalize_score(metrics.get('GROSS_MARGIN', 0) or 0, 0, 100),
            'enterprise_value': self._normalize_score(metrics.get('CURR_ENTP_VAL', 0) or 0, 0, 1e12),
            'capex_to_sales': self._normalize_score(metrics.get('CAP_EXPEND_TO_SALES', 0) or 0, 0.5, 0, reverse=True),
            'sales_growth_5yr': self._normalize_score(metrics.get('SALES_5YR_AVG_GR', 0) or 0, -5, 20),
            'fcf_yield': self._normalize_score(metrics.get('FCF_YIELD_WITH_CUR_ENTP_VAL', 0) or 0, 0, 10)
        }
        return np.mean(list(scores.values()))  # Average of all metric scores

    def calculate_market_performance(self) -> float:
        """Calculate market performance score."""
        metrics = self.data['current']['market']
        sharpe = self._calculate_sharpe((metrics['LAST_CLOSE_TRR_1YR'] or 0) / 100, (metrics['VOLATILITY_260D'] or 0) / 100)
        beta = metrics.get('EQY_BETA', 1.0) or 1.0
        return (self._normalize_score(metrics.get('LAST_CLOSE_TRR_1YR', 0) or 0, -20, 100) * 0.2 +
                self._normalize_score(metrics.get('TRAIL_12M_NET_REVENUE', 0) or 0, 0, 1e11) * 0.15 +
                self._normalize_score(metrics.get('VOLATILITY_90D', 0) or 0, 0.5, 0, reverse=True) * 0.15 +
                self._normalize_score(beta, 2.0, 0.5, reverse=True) * 0.15 +
                self._normalize_score(metrics.get('EQY_ALPHA', 0) or 0, -5, 5) * 0.15 +
                self._normalize_score(metrics.get('CUR_MKT_CAP', 0) or 0, 0, 1e12) * 0.2)

    def calculate_valuation(self) -> float:
        """Calculate valuation score relative to industry peers."""
        metrics = self.data['current']['valuation']
        sector_pe = self.data['current']['sector']['industry_pe']
        pe_relative = self._normalize_score((metrics['PE_RATIO'] or 0) / sector_pe, 2, 0.5, reverse=True)
        return (pe_relative * 0.15 +
                self._normalize_score(metrics.get('PX_TO_SALES_RATIO', 0) or 0, 5, 0.5, reverse=True) * 0.1 +
                self._normalize_score(metrics.get('BEST_CUR_EV_TO_EBITDA', 0) or 0, 20, 5, reverse=True) * 0.1 +
                self._normalize_score(metrics.get('PX_TO_FREE_CASH_FLOW', 0) or 0, 30, 5, reverse=True) * 0.1 +
                self._normalize_score(metrics.get('BEST_PEG_RATIO_CUR_YR', 0) or 0, 3, 0.5, reverse=True) * 0.1 +
                self._normalize_score(metrics.get('BOOK_VAL_PER_SH', 0) or 0, 0, 100) * 0.1 +
                self._normalize_score(metrics.get('CURRENT_EV_TO_12M_SALES', 0) or 0, 5, 0.5, reverse=True) * 0.1 +
                self._normalize_score(metrics['PX_TO_BOOK_RATIO'] or 0, 10, 1, reverse=True) * 0.1 +
                self._normalize_score(metrics['EQY_DVD_YLD_IND'] or 0, 0, 5) * 0.05 +
                self._normalize_score(metrics.get('EPS_GROWTH', 0) or 0, -20, 50) * 0.15)

    def calculate_sector_trends(self) -> float:
        """Calculate sector trends score."""
        metrics = self.data['current']['sector']
        return (self._normalize_score(metrics['sector_performance'], -10, 20) * 0.5 +
                self._normalize_score(metrics.get('PRICE_TO_INDUSTRY_SPEC_EBITDA', 30.0) or 30.0, 20, 5, reverse=True) * 0.5)

    def calculate_analyst_recommendations(self) -> float:
        """Calculate analyst recommendations score based on Buy/Hold/Sell distribution."""
        bsh_data = self.data['current']['bsh']
        total_analysts = bsh_data['num_analysts']
        
        if total_analysts == 0:
            return 50.0  # Default score if no analysts
        
        # Assign points: Buy=100, Hold=50, Sell=0
        buy_score = bsh_data['buy_count'] * 100
        hold_score = bsh_data['hold_count'] * 50
        sell_score = bsh_data['sell_count'] * 0
        
        # Calculate maximum possible score (all buys)
        max_possible = total_analysts * 100
        
        # Return percentage of maximum score
        actual_score = (buy_score + hold_score + sell_score) / max_possible * 100
        
        return round(actual_score, 2)

    def calculate_technical(self) -> float:
        """Calculate technical analysis score."""
        metrics = self.data['current']['technical']
        rsi_score = self._normalize_score(metrics['RSI_14D'] or 0, 30, 70) if 30 <= (metrics['RSI_14D'] or 0) <= 70 else 20.0
        macd_score = 100.0 if (metrics['MACD'] or 0) > 0 else 0.0
        ma_score = 100.0 if (metrics['MOV_AVG_50D'] or 0) > (metrics['MOV_AVG_200D'] or 0) else 60.0
        bollinger_upper = self._normalize_score(metrics.get('EQY_BOLLINGER_UPPER', metrics['MOV_AVG_50D']) or metrics['MOV_AVG_50D'] or 0, (metrics['MOV_AVG_50D'] or 0) * 0.9, (metrics['MOV_AVG_50D'] or 0) * 1.1)
        return (rsi_score * 0.25 + macd_score * 0.25 + ma_score * 0.25 + bollinger_upper * 0.25)

    def calculate_earnings_quality(self) -> float:
        """Calculate earnings quality score."""
        metrics = self.data['current']['earnings']
        return (self._normalize_score(metrics['surprise_pct'] or 0, -10, 15) * 0.4 +
                self._normalize_score(metrics['consistency'] or 0, 0, 1) * 0.3 +
                self._normalize_score(metrics.get('CFO_TO_SALES', 0) or 0, 0, 1) * 0.3)

    def calculate_atlas_score(self) -> Dict[str, float]:
        """Calculate the overall Atlas Score combining all factors."""
        scores = {
            'financial_health': self.calculate_financial_health(),
            'market_performance': self.calculate_market_performance(),
            'valuation': self.calculate_valuation(),
            'sector_trends': self.calculate_sector_trends(),
            'analyst_recommendations': self.calculate_analyst_recommendations(),
            'technical': self.calculate_technical(),
            'earnings_quality': self.calculate_earnings_quality()
        }
        base_score = sum(scores[factor] * weight for factor, weight in self.weights.items()) * 0.80  # 80% from weighted factors
        ml_adjustment = self._normalize_score(self.ml_prediction, -20, 30) * 0.15  # 15% from ML prediction
        bsh_adjustment = self.bsh_score * 0.05  # 5% from BSH score
        atlas_score = base_score + ml_adjustment + bsh_adjustment
        return {'atlas_score': round(atlas_score, 2), 'component_scores': {k: round(v, 2) for k, v in scores.items()}}

    def display_results(self) -> None:
        """Display the main results of the stock analysis."""
        result = self.calculate_atlas_score()
        # Define color codes for terminal output
        GREEN, YELLOW, RED, RESET = '\033[92m', '\033[93m', '\033[91m', '\033[0m'
        score_color = GREEN if result['atlas_score'] >= 70 else YELLOW if result['atlas_score'] >= 40 else RED

        # Attempt to display logo if available
        logo_path = os.path.join(os.path.dirname(__file__), 'atlas_logo.png')
        if os.path.exists(logo_path):
            try:
                from PIL import Image
                import IPython.display as display
                img = Image.open(logo_path)
                display.display(img)
                print(f"\n{'Atlas Score':<30} {score_color}{result['atlas_score']:.1f}/100{RESET}\n")
            except Exception as e:
                logger.error(f"Failed to display logo: {str(e)}")
                print(f"\n{'Atlas Score':<30} {score_color}{result['atlas_score']:.1f}/100{RESET}\n")
        else:
            print(f"\n{'Atlas Score':<30} {score_color}{result['atlas_score']:.1f}/100{RESET}\n")

        # Print report header
        print(f"{'='*60}")
        print(f" Project Atlas Report: {self.ticker} ".center(60, '='))
        print(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(60, '='))
        print(f"{'='*60}\n")
        
        # Print summary section
        print(f"{'Summary':<25} {'Value':>15}")
        print(f"{'-'*25} {'-'*15}")
        print(f"{'BSH Adjustment':<25} {round(self.bsh_score * 0.05, 2):>15.2f}")
        print(f"{'Analyst Consensus':<25} {self.data['current']['bsh']['consensus']:>15}")
        print(f"{'Recommendation':<25} {self._get_bsh_recommendation(result['atlas_score']):>15}")
        print(f"\n{'-'*40}\n")
        
        # Print component scores
        print("Component Scores:")
        print(f"{'Factor':<25} {'Score':>10} {'Weight':>10}")
        print(f"{'-'*25} {'-'*10} {'-'*10}")
        for factor, score in result['component_scores'].items():
            color = GREEN if score >= 70 else YELLOW if score >= 40 else RED
            weight = self.weights[factor] * 100
            print(f"{color}{factor.replace('_', ' ').title():<25}{RESET} {score:>7.1f}/100 {weight:>7.1f}%")
        
        # Warn about missing data if any
        if self.missing_fields:
            print(f"\n{'!'*40}")
            print(f" Warning: Missing data for: {', '.join(self.missing_fields)} ".center(40, '!'))
            print(f"{'!'*40}")
        
        # Print score interpretation guide
        print(f"\n{'-'*40}")
        print("Score Interpretation:")
        print(f"{GREEN}70-100: Strong Buy{RESET}".ljust(40))
        print(f"{YELLOW}40-69: Hold{RESET}".ljust(40))
        print(f"{RED}0-39: Sell{RESET}".ljust(40))
        print(f"{'-'*40}")
        
        # Prompt for detailed analysis
        while True:
            response = input("\nWould you like a detailed analysis? (y/n): ").strip().lower()
            if response == '':
                print("Please enter 'y' or 'n' (or press Enter twice to exit).")
                second_response = input("Would you like a detailed analysis? (y/n): ").strip().lower()
                if second_response == '':
                    print("Exiting scanner.")
                    break
                elif second_response in ['y', 'yes']:
                    self.display_deep_dive()
                    break
                elif second_response in ['n', 'no']:
                    print("Skipping detailed analysis.")
                    break
                else:
                    print("Invalid input. Please enter 'y/yes' or 'n/no'.")
            elif response in ['y', 'yes']:
                self.display_deep_dive()
                break
            elif response in ['n', 'no']:
                print("Skipping detailed analysis.")
                break
            else:
                print("Invalid input. Please enter 'y/yes' or 'n/no'.")

    def display_deep_dive(self) -> None:
        """Display detailed analysis of the stock."""
        current = self.data['current']
        bsh_data = current['bsh']
        sharpe = self._calculate_sharpe((current['market']['LAST_CLOSE_TRR_1YR'] or 0) / 100, (current['market']['VOLATILITY_260D'] or 0) / 100)

        # Print detailed analysis header
        print(f"\n{'='*60}")
        print(f" Detailed Analysis: {self.ticker} ".center(60, '='))
        print(f" Date: {datetime.now().strftime('%Y-%m-%d')} ".center(60, '='))
        print(f"{'='*60}\n")
        
        # Financial Health section
        print("Financial Health:")
        print(f"{'Metric':<25} {'Value':>15}")
        print(f"{'-'*25} {'-'*15}")
        print(f"{'Revenue Growth (YoY)':<25} {current['financial']['SALES_GROWTH'] or 0:>15.2f}%")
        print(f"{'Net Profit Margin':<25} {current['financial']['ARDR_NON_GAAP_GROSS_MARGIN_PCT'] or 0:>15.2f}%")
        print(f"{'Debt-to-Equity Ratio':<25} {current['financial']['TOT_DEBT_TO_TOT_EQY'] or 0:>15.2f}")
        print(f"{'Operating Margin':<25} {current['financial']['OPER_MARGIN'] or 0:>15.2f}%")
        print(f"{'Net Debt to EBITDA':<25} {current['financial']['NET_DEBT_TO_EBITDA'] or 0:>15.2f}")
        print(f"{'Current Ratio':<25} {current['financial']['CUR_RATIO'] or 0:>15.2f}")
        print(f"{'Quick Ratio':<25} {current['financial']['QUICK_RATIO'] or 0:>15.2f}")
        print(f"{'FCF to Total Debt':<25} {current['financial']['FCF_TO_TOTAL_DEBT'] or 0:>15.2f}")
        print(f"{'EBITDA':<25} {'N/A' if current['financial']['EBITDA'] is None else f'${current['financial']['EBITDA'] / 1e9:>11.2f}B':>15}")
        print(f"{'Free Cash Flow':<25} {'N/A' if current['financial']['CF_FREE_CASH_FLOW'] is None else f'${current['financial']['CF_FREE_CASH_FLOW'] / 1e9:>11.2f}B':>15}")
        print(f"{'Net Income':<25} {'N/A' if current['financial']['BEST_NET_INCOME'] is None else f'${current['financial']['BEST_NET_INCOME'] / 1e9:>11.2f}B':>15}")
        print(f"{'Asset Turnover':<25} {current['financial']['ASSET_TURNOVER'] or 0:>15.2f}")
        print(f"{'Debt to Capital':<25} {current['financial']['TOT_DEBT_TO_TOT_CAP'] or 0:>15.2f}")
        print(f"{'Return on Equity':<25} {current['financial']['RETURN_ON_EQY'] or 0:>15.2f}%")
        print(f"{'Gross Margin':<25} {current['financial']['GROSS_MARGIN'] or 0:>15.2f}%")
        print(f"{'Enterprise Value':<25} {'N/A' if current['financial']['CURR_ENTP_VAL'] is None else f'${current['financial']['CURR_ENTP_VAL'] / 1e9:>11.2f}B':>15}")
        print(f"{'CapEx to Sales':<25} {current['financial']['CAP_EXPEND_TO_SALES'] or 0:>15.2f}%")
        print(f"{'5-Year Sales Growth':<25} {current['financial']['SALES_5YR_AVG_GR'] or 0:>15.2f}%")
        print(f"{'FCF Yield':<25} {current['financial']['FCF_YIELD_WITH_CUR_ENTP_VAL'] or 0:>15.2f}%")

        # Market Performance section
        print(f"\nMarket Performance:")
        print(f"{'Metric':<25} {'Value':>15}")
        print(f"{'-'*25} {'-'*15}")
        print(f"{'Annualized Return':<25} {current['market']['LAST_CLOSE_TRR_1YR'] or 0:>15.2f}%")
        print(f"{'Trailing 12M Revenue':<25} {'N/A' if current['market']['TRAIL_12M_NET_REVENUE'] is None else f'${current['market']['TRAIL_12M_NET_REVENUE'] / 1e9:>11.2f}B':>15}")
        print(f"{'Volatility (90D)':<25} {current['market']['VOLATILITY_90D'] or 0:>15.2f}")
        print(f"{'Volatility (260D)':<25} {current['market']['VOLATILITY_260D'] or 0:>15.2f}")
        print(f"{'Beta':<25} {current['market']['EQY_BETA'] or 0:>15.2f}")
        print(f"{'Alpha':<25} {current['market']['EQY_ALPHA'] or 0:>15.2f}")
        print(f"{'Market Cap':<25} {'N/A' if current['market']['CUR_MKT_CAP'] is None else f'${current['market']['CUR_MKT_CAP'] / 1e9:>11.2f}B':>15}")
        print(f"{'Sharpe Ratio':<25} {sharpe:>15.2f}")

        # Valuation section
        print(f"\nValuation:")
        print(f"{'Metric':<25} {'Value':>15}")
        print(f"{'-'*25} {'-'*15}")
        print(f"{'P/E Ratio':<25} {current['valuation']['PE_RATIO'] or 0:>15.2f}")
        print(f"{'P/S Ratio':<25} {current['valuation']['PX_TO_SALES_RATIO'] or 0:>15.2f}")
        print(f"{'EV/EBITDA':<25} {current['valuation']['BEST_CUR_EV_TO_EBITDA'] or 0:>15.2f}")
        print(f"{'P/FCF':<25} {current['valuation']['PX_TO_FREE_CASH_FLOW'] or 0:>15.2f}")
        print(f"{'PEG Ratio':<25} {current['valuation']['BEST_PEG_RATIO_CUR_YR'] or 0:>15.2f}")
        print(f"{'Book Value per Share':<25} {'N/A' if current['valuation']['BOOK_VAL_PER_SH'] is None else f'${current['valuation']['BOOK_VAL_PER_SH']:>11.2f}':>15}")
        print(f"{'EV/Sales':<25} {current['valuation']['CURRENT_EV_TO_12M_SALES'] or 0:>15.2f}")
        print(f"{'P/B Ratio':<25} {current['valuation']['PX_TO_BOOK_RATIO'] or 0:>15.2f}")
        print(f"{'Dividend Yield':<25} {current['valuation']['EQY_DVD_YLD_IND'] or 0:>15.2f}%")
        print(f"{'EPS Growth (1Yr)':<25} {current['valuation']['EPS_GROWTH'] or 0:>15.2f}%")
        print(f"{'Industry Avg P/E':<25} {current['sector']['industry_pe']:>15.2f}")

        # Technical Indicators section
        print(f"\nTechnical Indicators:")
        print(f"{'Metric':<25} {'Value':>15}")
        print(f"{'-'*25} {'-'*15}")
        print(f"{'RSI (14-day)':<25} {current['technical']['RSI_14D'] or 0:>15.2f}")
        print(f"{'MACD Trend':<25} {'Positive' if (current['technical']['MACD'] or 0) > 0 else 'Negative':>15}")
        print(f"{'50-Day MA':<25} {'N/A' if current['technical']['MOV_AVG_50D'] is None else f'${current['technical']['MOV_AVG_50D']:>11.2f}':>15}")
        print(f"{'200-Day MA':<25} {'N/A' if current['technical']['MOV_AVG_200D'] is None else f'${current['technical']['MOV_AVG_200D']:>11.2f}':>15}")
        print(f"{'Bollinger Upper':<25} {'N/A' if current['technical']['EQY_BOLLINGER_UPPER'] is None else f'${current['technical']['EQY_BOLLINGER_UPPER']:>11.2f}':>15}")

        # Earnings Quality section
        print(f"\nEarnings Quality:")
        print(f"{'Metric':<25} {'Value':>15}")
        print(f"{'-'*25} {'-'*15}")
        print(f"{'Consistency':<25} {current['earnings']['consistency']:>15.2f}")
        print(f"{'CFO to Sales':<25} {current['earnings']['cfo_to_sales'] or 0:>15.2f}")

        # Analyst Recommendations section
        print(f"\nAnalyst Recommendations:")
        print(f"{'Metric':<25} {'Value':>15}")
        print(f"{'-'*25} {'-'*15}")
        print(f"{'Consensus Rating':<25} {bsh_data['consensus']:>15}")
        print(f"{'Average Rating':<25} {bsh_data['avg_rating']:>15.2f}")
        print(f"{'Number of Analysts':<25} {bsh_data['num_analysts']:>15}")
        print(f"{'Distribution':<25} {'Buy: ' + str(bsh_data['buy_count']) + ', Hold: ' + str(bsh_data['hold_count']) + ', Sell: ' + str(bsh_data['sell_count']):>15}")

    def _get_bsh_recommendation(self, score: float) -> str:
        """Determine Buy/Hold/Sell recommendation based on Atlas Score."""
        if score >= 70:
            return "Buy"
        elif score >= 40:
            return "Hold"
        else:
            return "Sell"

def main():
    """Main function to run the stock scanner."""
    prev_input = None
    while True:
        ticker = input("Enter the stock ticker symbol (or press Enter twice to exit): ").strip().upper()
        if ticker == '' and prev_input == '':
            print("Exiting scanner.")
            return
        elif ticker == '':
            prev_input = ticker
            print("Please enter a ticker symbol (or press Enter again to exit).")
            continue
        else:
            prev_input = ticker
        
        try:
            scanner = AtlasStockScanner(ticker, strategy='growth')  # Create scanner instance
            scanner.display_results()  # Display analysis results
        except Exception as e:
            print(f"\nError processing {ticker}: {str(e)}")
            logger.error(f"Error processing {ticker}: {str(e)}")
        
        # Prompt to scan another stock
        while True:
            response = input("\nScan another stock? (y/yes or n/no, or press Enter twice to exit): ").strip().lower()
            if response == '' and prev_input == '':
                print("Exiting scanner.")
                return
            elif response == '':
                prev_input = response
                print("Please enter 'y/yes' or 'n/no' (or press Enter again to exit).")
                continue
            elif response in ['y', 'yes']:
                prev_input = response
                break
            elif response in ['n', 'no']:
                print("Exiting scanner.")
                return
            else:
                print("Invalid input. Please enter 'y/yes' or 'n/no'.")
                retry = input("Try again (y/yes or n/no): ").strip().lower()
                if retry in ['y', 'yes']:
                    prev_input = retry
                    break
                elif retry in ['n', 'no']:
                    print("Exiting scanner.")
                    return
                else:
                    print("Invalid input again. Exiting scanner.")
                    return

if __name__ == "__main__":
    main() 
    
