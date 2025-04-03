import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime, timedelta
import logging
import blpapi
import asyncio
import time
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='atlas.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

class AtlasStockScanner:
    def __init__(self, ticker: str, weights: Dict[str, float]):
        self.ticker = ticker.upper()
        self.weights = weights
        self.benchmark = "SPY"
        self.loading = True
        self.data, self.peers = self._fetch_with_loading()
        self.missing_fields = []
        self.features = ['LAST_CLOSE_TRR_1YR', 'VOLATILITY_260D', 'CUR_MKT_CAP', 'PE_RATIO', 'SALES_GROWTH', 
                         'TOT_DEBT_TO_TOT_EQY', 'OPER_MARGIN', 'RETURN_ON_EQY', 'RSI_14D']
        self.model, self.scaler, self.feature_importance = self._train_ml_model()
        self.ml_prediction = self._predict_return()
        self.bsh_score = self._get_bsh_score()

    def _fetch_with_loading(self) -> Tuple[Dict, list]:
        self.loading = True
        with st.spinner(f"Fetching data for {self.ticker}..."):
            try:
                result = asyncio.run(self._fetch_data())
                self.loading = False
                st.success("Data fetched successfully!")
                return result
            except Exception as e:
                self.loading = False
                raise e

    def _fetch_bloomberg_data_sync(self, request_type: str, securities: list, fields: list, start_date: str = None, end_date: str = None) -> Dict:
        session = blpapi.Session()
        if not session.start():
            logger.error("Failed to start Bloomberg session")
            return {}
        
        service_name = "//blp/refdata"
        if not session.openService(service_name):
            logger.error(f"Failed to open {service_name} service")
            session.stop()
            return {}
        
        service = session.getService(service_name)
        request = service.createRequest(request_type)
        
        for sec in securities:
            request.getElement("securities").appendValue(sec)
        for field in fields:
            request.getElement("fields").appendValue(field)
        if request_type == "HistoricalDataRequest":
            request.set("startDate", start_date)
            request.set("endDate", end_date)
        
        session.sendRequest(request)
        
        data = {}
        while True:
            event = session.nextEvent(500)
            if event.eventType() == blpapi.Event.RESPONSE:
                for msg in event:
                    if request_type == "ReferenceDataRequest":
                        security_data = msg.getElement("securityData")
                        for i in range(security_data.numValues()):
                            sec = security_data.getValueAsElement(i)
                            ticker = sec.getElementAsString("security")
                            field_data = sec.getElement("fieldData")
                            data[ticker] = {}
                            for field in field_data.elements():
                                field_name = str(field.name())
                                if field.datatype() == blpapi.DataType.STRING:
                                    data[ticker][field_name] = field.getValueAsString() if field.numValues() > 0 else ""
                                elif field_name == "BLOOMBERG_PEERS":
                                    values = [field.getValueAsElement(j).getElement(0).getValueAsString() for j in range(field.numValues()) if field.getValueAsElement(j).numElements() > 0]
                                    data[ticker][field_name] = ",".join(values) if values else ""
                                elif field.numValues() > 1 or field.isArray():
                                    values = [field.getValueAsString(i) for i in range(field.numValues())]
                                    data[ticker][field_name] = ",".join(values) if values else ""
                                else:
                                    data[ticker][field_name] = field.getValueAsFloat() if field.numValues() > 0 else None
                    elif request_type == "HistoricalDataRequest":
                        security_data = msg.getElement("securityData")
                        ticker = security_data.getElementAsString("security")
                        field_data = security_data.getElement("fieldData")
                        dates = []
                        values = {field: [] for field in fields}
                        for entry in field_data.values():
                            date = entry.getElementAsString("date")
                            dates.append(date)
                            for field in fields:
                                values[field].append(entry.getElementAsFloat(field) if entry.hasElement(field) else np.nan)
                        if dates:
                            data[ticker] = pd.DataFrame(values, index=pd.to_datetime(dates))
                        else:
                            data[ticker] = pd.DataFrame()
                break
        
        session.stop()
        return data

    def _fetch_yahoo_finance_data(self, ticker: str) -> Dict:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            yahoo_data = {
                'EBITDA': info.get('ebitda', 0),
                'CF_FREE_CASH_FLOW': info.get('freeCashflow', 0),
                'BEST_NET_INCOME': info.get('netIncomeToCommon', 0),
                'RETURN_ON_EQY': info.get('returnOnEquity', 0) * 100,
                'TRAIL_12M_NET_REVENUE': info.get('totalRevenue', 0),
                'CURR_ENTP_VAL': info.get('enterpriseValue', 0)
            }
            logger.info(f"Fetched Yahoo Finance data for {ticker}")
            return yahoo_data
        except Exception as e:
            logger.error(f"Failed to fetch Yahoo Finance data for {ticker}: {str(e)}")
            return {}

    def _prompt_for_missing_values(self, current_data: Dict, fields_to_check: list) -> Dict:
        for field in fields_to_check:
            if current_data.get(field) is None or (isinstance(current_data.get(field), (int, float)) and current_data.get(field) == 0):
                st.write(f"Field {field} not found or 0 for {self.ticker}. Enter value manually:")
                if field in ["EBITDA", "CF_FREE_CASH_FLOW", "BEST_NET_INCOME", "CURR_ENTP_VAL", "TRAIL_12M_NET_REVENUE"]:
                    value = st.number_input(f"{field} (in billions, e.g., 60 for $60B, or 0 to skip)", min_value=0.0, value=0.0)
                    if value > 0:
                        current_data[field] = float(value) * 1e9
                elif field == "RETURN_ON_EQY":
                    value = st.number_input(f"{field} (as percentage, e.g., 15.5, or 0 to skip)", min_value=0.0, value=0.0)
                    if value > 0:
                        current_data[field] = float(value)
                else:
                    value = st.number_input(f"{field} (or 0 to skip)", min_value=0.0, value=0.0)
                    if value > 0:
                        current_data[field] = float(value)
                logger.info(f"User manually set {field} to {current_data[field]} for {self.ticker}")
        return current_data

    async def _fetch_data(self) -> Tuple[Dict, list]:
        hist_fields = ["LAST_CLOSE_TRR_1YR", "VOLATILITY_260D", "CUR_MKT_CAP", "PE_RATIO", "SALES_GROWTH"]
        ref_fields = [
            "SALES_GROWTH", "ARDR_NON_GAAP_GROSS_MARGIN_PCT", "TOT_DEBT_TO_TOT_EQY",
            "OPER_MARGIN", "NET_DEBT_TO_EBITDA", "CUR_RATIO", "QUICK_RATIO", "FCF_TO_TOTAL_DEBT",
            "ASSET_TURNOVER", "TOT_DEBT_TO_TOT_CAP", "GROSS_MARGIN", "CAP_EXPEND_TO_SALES", "SALES_5YR_AVG_GR",
            "FCF_YIELD_WITH_CUR_ENTP_VAL", "VOLATILITY_90D", "EQY_BETA", "EQY_ALPHA", "CUR_MKT_CAP",
            "LAST_CLOSE_TRR_1YR", "VOLATILITY_260D", "BETA_ADJUSTED", "PE_RATIO", "PX_TO_SALES_RATIO",
            "BEST_CUR_EV_TO_EBITDA", "PX_TO_FREE_CASH_FLOW", "BEST_PEG_RATIO_CUR_YR", "BOOK_VAL_PER_SH",
            "CURRENT_EV_TO_12M_SALES", "PX_TO_BOOK_RATIO", "EQY_DVD_YLD_IND", "EPS_GROWTH",
            "INDUSTRY_SECTOR", "SHORT_INT_RATIO",
            "RSI_14D", "MACD", "MOV_AVG_50D", "MOV_AVG_200D", "EQY_BOLLINGER_UPPER",
            "EARNINGS_SURPRISE_PCT", "CFO_TO_SALES", "AVG_ANALYST_RATING", "TOT_ANALYST_REC",
            "BEST_ANALYST_RATING", "BLOOMBERG_PEERS"
        ]
        
        start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")
        
        loop = asyncio.get_event_loop()
        hist_data = await loop.run_in_executor(None, self._fetch_bloomberg_data_sync, 
                                               "HistoricalDataRequest", 
                                               [f"{self.ticker} US Equity"], 
                                               hist_fields, start_date, end_date)
        ref_data = await loop.run_in_executor(None, self._fetch_bloomberg_data_sync, 
                                              "ReferenceDataRequest", 
                                              [f"{self.ticker} US Equity"], 
                                              ref_fields)
        
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
        
        historical_df = hist_data.get(f"{self.ticker} US Equity", pd.DataFrame())
        if not historical_df.empty:
            historical_df = historical_df.resample('YE').mean().reset_index()
            historical_df['year'] = historical_df['index'].dt.year
            historical_df['future_return'] = historical_df['LAST_CLOSE_TRR_1YR'].shift(-1)
        else:
            logger.warning(f"No historical data fetched for {self.ticker}")
        
        current_data = ref_data.get(f"{self.ticker} US Equity", {})
        peer_data = {p.split()[0]: peer_data.get(p, {}) for p in peers}
        
        yahoo_data = self._fetch_yahoo_finance_data(self.ticker)
        current_data['EBITDA'] = yahoo_data.get('EBITDA', 0)
        current_data['CF_FREE_CASH_FLOW'] = yahoo_data.get('CF_FREE_CASH_FLOW', 0)
        current_data['BEST_NET_INCOME'] = yahoo_data.get('BEST_NET_INCOME', 0)
        current_data['RETURN_ON_EQY'] = yahoo_data.get('RETURN_ON_EQY', 0)
        current_data['TRAIL_12M_NET_REVENUE'] = yahoo_data.get('TRAIL_12M_NET_REVENUE', 0)
        if 'CURR_ENTP_VAL' not in current_data or current_data['CURR_ENTP_VAL'] is None or current_data['CURR_ENTP_VAL'] == 0:
            current_data['CURR_ENTP_VAL'] = yahoo_data.get('CURR_ENTP_VAL', 0)
        
        critical_fields = ["EBITDA", "CF_FREE_CASH_FLOW", "BEST_NET_INCOME", "CURR_ENTP_VAL", "RETURN_ON_EQY", "TRAIL_12M_NET_REVENUE"]
        current_data = self._prompt_for_missing_values(current_data, critical_fields)
        
        required_fields = ["SALES_GROWTH", "TOT_DEBT_TO_TOT_EQY", "LAST_CLOSE_TRR_1YR", "VOLATILITY_260D", "PE_RATIO"]
        self.missing_fields = [f for f in required_fields if f not in current_data or (current_data[f] is None or (isinstance(current_data[f], (int, float)) and current_data[f] == 0))]
        
        if current_data.get('TOT_DEBT_TO_TOT_EQY') is not None and current_data.get('TOT_DEBT_TO_TOT_EQY', 0) > 2.5:
            current_data['TOT_DEBT_TO_TOT_EQY'] /= 100
        if current_data.get('TOT_DEBT_TO_TOT_CAP') is not None and current_data.get('TOT_DEBT_TO_TOT_CAP', 0) > 2.5:
            current_data['TOT_DEBT_TO_TOT_CAP'] /= 100
        
        current_data['bsh'] = {
            'consensus': 'Buy' if current_data.get('AVG_ANALYST_RATING', 3.0) > 3.5 else 'Hold' if current_data.get('AVG_ANALYST_RATING', 3.0) > 2.5 else 'Sell',
            'num_analysts': current_data.get('TOT_ANALYST_REC', 35),
            'buy_count': int(current_data.get('TOT_ANALYST_REC', 35) * 0.7),
            'hold_count': int(current_data.get('TOT_ANALYST_REC', 35) * 0.25),
            'sell_count': int(current_data.get('TOT_ANALYST_REC', 35) * 0.05)
        }
        
        ticker_data = {
            'historical': historical_df,
            'current': {
                'financial': {k: current_data.get(k, None) for k in ['SALES_GROWTH', 'ARDR_NON_GAAP_GROSS_MARGIN_PCT', 'TOT_DEBT_TO_TOT_EQY', 'OPER_MARGIN', 'NET_DEBT_TO_EBITDA', 'CUR_RATIO', 'QUICK_RATIO', 'FCF_TO_TOTAL_DEBT', 'EBITDA', 'CF_FREE_CASH_FLOW', 'BEST_NET_INCOME', 'ASSET_TURNOVER', 'TOT_DEBT_TO_TOT_CAP', 'RETURN_ON_EQY', 'GROSS_MARGIN', 'CURR_ENTP_VAL', 'CAP_EXPEND_TO_SALES', 'SALES_5YR_AVG_GR', 'FCF_YIELD_WITH_CUR_ENTP_VAL']},
                'market': {k: current_data.get(k, None) for k in ['LAST_CLOSE_TRR_1YR', 'VOLATILITY_260D', 'TRAIL_12M_NET_REVENUE', 'VOLATILITY_90D', 'EQY_BETA', 'EQY_ALPHA', 'CUR_MKT_CAP', 'BETA_ADJUSTED']},
                'valuation': {k: current_data.get(k, None) for k in ['PE_RATIO', 'PX_TO_SALES_RATIO', 'BEST_CUR_EV_TO_EBITDA', 'PX_TO_FREE_CASH_FLOW', 'BEST_PEG_RATIO_CUR_YR', 'BOOK_VAL_PER_SH', 'CURRENT_EV_TO_12M_SALES', 'PX_TO_BOOK_RATIO', 'EQY_DVD_YLD_IND', 'EPS_GROWTH']},
                'sector': {'industry_sector': current_data.get('INDUSTRY_SECTOR', 'Unknown')},
                'technical': {k: current_data.get(k, None) for k in ['RSI_14D', 'MACD', 'MOV_AVG_50D', 'MOV_AVG_200D', 'EQY_BOLLINGER_UPPER']},
                'earnings': {'surprise_pct': current_data.get('EARNINGS_SURPRISE_PCT', 5.0), 'consistency': 0.95, 'cfo_to_sales': current_data.get('CFO_TO_SALES', None)},
                'bsh': current_data['bsh']
            },
            'peers': peer_data
        }
        return ticker_data, peers

    def _get_bsh_score(self) -> float:
        bsh_data = self.data['current']['bsh']
        total_analysts = bsh_data['num_analysts']
        if total_analysts == 0:
            return 50.0
        buy_weight = bsh_data['buy_count'] / total_analysts * 100
        hold_weight = bsh_data['hold_count'] / total_analysts * 50
        sell_weight = bsh_data['sell_count'] / total_analysts * 0
        return (buy_weight + hold_weight + sell_weight)

    def _train_ml_model(self) -> Tuple[RandomForestRegressor, StandardScaler, Dict]:
        df = self.data['historical']
        current = self.data['current']
        
        scaler = StandardScaler()
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        
        if df.empty or not any(f in df.columns for f in self.features) or df['LAST_CLOSE_TRR_1YR'].dropna().empty:
            logger.warning(f"No valid historical data for {self.ticker}; using current data as fallback")
            X_dummy = pd.DataFrame([[current['market'].get(f, current['financial'].get(f, current['valuation'].get(f, current['technical'].get(f, 0)))) or 0 
                                   for f in self.features]], columns=self.features)
            y_dummy = [current['market']['LAST_CLOSE_TRR_1YR'] or 0]
            scaler.fit(X_dummy)
            model.fit(scaler.transform(X_dummy), y_dummy)
            feature_importance = {f: 1/len(self.features) for f in self.features}
        else:
            if not df.empty and len(df) > 1:
                df['momentum'] = df['LAST_CLOSE_TRR_1YR'].pct_change().fillna(0)
                df['vol_adj_return'] = df['LAST_CLOSE_TRR_1YR'] / df['VOLATILITY_260D'].replace(0, np.nan)
                df['vol_adj_return'] = df['vol_adj_return'].fillna(0)
                self.features.extend(['momentum', 'vol_adj_return'])

            available_features = [f for f in self.features if f in df.columns]
            X = df[available_features].fillna({
                f: current['market'].get(f, current['financial'].get(f, current['valuation'].get(f, current['technical'].get(f, df[f].mean() if f in df.columns else 0)))) or 0 
                for f in available_features
            })
            y = df['future_return'].fillna(current['market']['LAST_CLOSE_TRR_1YR'] or 0)
            
            if len(X) > 5:
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
                mean_cv_mse = -np.mean(cv_scores)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                logger.info(f"ML Model trained for {self.ticker}. MSE: {mse:.2f}, R2: {r2:.2f}, CV MSE: {mean_cv_mse:.2f}")
                feature_importance = dict(zip(available_features, model.feature_importances_))
            else:
                logger.warning(f"Insufficient historical data points ({len(X)}) for {self.ticker}; using current data")
                X_dummy = pd.DataFrame([[current['market'].get(f, current['financial'].get(f, current['valuation'].get(f, current['technical'].get(f, 0)))) or 0 
                                       for f in self.features]], columns=self.features)
                y_dummy = [current['market']['LAST_CLOSE_TRR_1YR'] or 0]
                scaler.fit(X_dummy)
                model.fit(scaler.transform(X_dummy), y_dummy)
                feature_importance = {f: 1/len(self.features) for f in self.features}
        
        return model, scaler, feature_importance

    def _predict_return(self) -> float:
        current = self.data['current']
        X_current_dict = {f: current['market'].get(f, current['financial'].get(f, current['valuation'].get(f, current['technical'].get(f, 0)))) or 0 
                          for f in self.features}
        
        if 'momentum' in self.features:
            X_current_dict['momentum'] = 0
        if 'vol_adj_return' in self.features:
            last_return = X_current_dict.get('LAST_CLOSE_TRR_1YR', 0)
            volatility = X_current_dict.get('VOLATILITY_260D', 0)
            X_current_dict['vol_adj_return'] = last_return / volatility if volatility != 0 else 0
        
        X_current = pd.DataFrame([list(X_current_dict.values())], columns=self.features)
        return self.model.predict(self.scaler.transform(X_current))[0]

    def _normalize_score(self, value: float, min_val: float, max_val: float, reverse: bool = False) -> float:
        try:
            scaled = np.clip((value - min_val) / (max_val - min_val), 0, 1) * 100
            return 100 - scaled if reverse else scaled
        except (TypeError, ZeroDivisionError):
            return 50.0

    def _calculate_sharpe(self, returns: float, volatility: float, rf_rate: float = 0.02) -> float:
        if volatility > 0:
            return (returns - rf_rate) / volatility
        return 0

    def calculate_financial_health(self) -> float:
        metrics = self.data['current']['financial']
        scores = {
            'revenue': self._normalize_score(metrics.get('SALES_GROWTH', 0) or 0, -10, 50),
            'margin': self._normalize_score(metrics.get('ARDR_NON_GAAP_GROSS_MARGIN_PCT', 0) or 0, 0, 80),
            'debt': self._normalize_score(metrics.get('TOT_DEBT_TO_TOT_EQY', 0) or 0, 1.0, 0.2, reverse=True),
            'oper_margin': self._normalize_score(metrics.get('OPER_MARGIN', 0) or 0, 0, 30),
            'net_debt_ebitda': self._normalize_score(metrics.get('NET_DEBT_TO_EBITDA', 0) or 0, 5, 0, reverse=True),
            'current_ratio': self._normalize_score(metrics.get('CUR_RATIO', 0) or 0, 0.5, 3),
            'quick_ratio': self._normalize_score(metrics.get('QUICK_RATIO', 0) or 0, 0.5, 2),
            'ebitda': self._normalize_score(metrics.get('EBITDA', 0) or 0, 0, 1e11),
            'free_cash_flow': self._normalize_score(metrics.get('CF_FREE_CASH_FLOW', 0) or 0, 0, 1e11),
            'net_income': self._normalize_score(metrics.get('BEST_NET_INCOME', 0) or 0, 0, 1e11),
            'asset_turnover': self._normalize_score(metrics.get('ASSET_TURNOVER', 0) or 0, 0, 2),
            'debt_to_cap': self._normalize_score(metrics.get('TOT_DEBT_TO_TOT_CAP', 0) or 0, 1, 0, reverse=True),
            'roe': self._normalize_score(metrics.get('RETURN_ON_EQY', 0) or 0, -20, 50),
            'gross_margin': self._normalize_score(metrics.get('GROSS_MARGIN', 0) or 0, 0, 100),
            'enterprise_value': self._normalize_score(metrics.get('CURR_ENTP_VAL', 0) or 0, 0, 1e12),
            'capex_to_sales': self._normalize_score(metrics.get('CAP_EXPEND_TO_SALES', 0) or 0, 0.5, 0, reverse=True),
            'sales_growth_5yr': self._normalize_score(metrics.get('SALES_5YR_AVG_GR', 0) or 0, -5, 20),
            'fcf_yield': self._normalize_score(metrics.get('FCF_YIELD_WITH_CUR_ENTP_VAL', 0) or 0, 0, 10)
        }
        mean_score = np.mean(list(scores.values()))
        return min(100, mean_score * 2.0)

    def calculate_market_performance(self) -> float:
        metrics = self.data['current']['market']
        sharpe = self._calculate_sharpe((metrics['LAST_CLOSE_TRR_1YR'] or 0) / 100, (metrics['VOLATILITY_260D'] or 0) / 100)
        beta = metrics.get('EQY_BETA', 1.0) or 1.0
        score = (self._normalize_score(metrics.get('LAST_CLOSE_TRR_1YR', 0) or 0, 0, 40) * 0.2 +
                 self._normalize_score(metrics.get('TRAIL_12M_NET_REVENUE', 0) or 0, 0, 1e11) * 0.15 +
                 self._normalize_score(metrics.get('VOLATILITY_90D', 0) or 0, 0.15, 0, reverse=True) * 0.15 +
                 self._normalize_score(beta, 2.0, 0.5, reverse=True) * 0.15 +
                 self._normalize_score(metrics.get('EQY_ALPHA', 0) or 0, -5, 5) * 0.15 +
                 self._normalize_score(metrics.get('CUR_MKT_CAP', 0) or 0, 0, 3e11) * 0.2)
        return min(100, score * 2.2)

    def calculate_valuation(self) -> float:
        metrics = self.data['current']['valuation']
        pe_relative = self._normalize_score(metrics.get('PE_RATIO', 0) or 0, 10, 5, reverse=True)
        score = (pe_relative * 0.15 +
                 self._normalize_score(metrics.get('PX_TO_SALES_RATIO', 0) or 0, 2, 0.5, reverse=True) * 0.1 +
                 self._normalize_score(metrics.get('BEST_CUR_EV_TO_EBITDA', 0) or 0, 20, 5, reverse=True) * 0.1 +
                 self._normalize_score(metrics.get('PX_TO_FREE_CASH_FLOW', 0) or 0, 30, 5, reverse=True) * 0.1 +
                 self._normalize_score(metrics.get('BEST_PEG_RATIO_CUR_YR', 0) or 0, 3, 0.5, reverse=True) * 0.1 +
                 self._normalize_score(metrics.get('BOOK_VAL_PER_SH', 0) or 0, 0, 100) * 0.1 +
                 self._normalize_score(metrics.get('CURRENT_EV_TO_12M_SALES', 0) or 0, 5, 0.5, reverse=True) * 0.1 +
                 self._normalize_score(metrics.get('PX_TO_BOOK_RATIO', 0) or 0, 10, 1, reverse=True) * 0.1 +
                 self._normalize_score(metrics.get('EQY_DVD_YLD_IND', 0) or 0, 0, 5) * 0.05 +
                 self._normalize_score(metrics.get('EPS_GROWTH', 0) or 0, -20, 25) * 0.15)
        return min(100, score * 1.8)

    def calculate_analyst_recommendations(self) -> float:
        bsh_data = self.data['current']['bsh']
        total_analysts = bsh_data['num_analysts']
        
        if total_analysts == 0:
            return 50.0
        
        buy_score = bsh_data['buy_count'] * 100
        hold_score = bsh_data['hold_count'] * 50
        sell_score = bsh_data['sell_count'] * 0
        
        max_possible = total_analysts * 100
        actual_score = (buy_score + hold_score + sell_score) / max_possible * 100
        return round(actual_score, 2)

    def calculate_technical(self) -> float:
        metrics = self.data['current']['technical']
        rsi_score = self._normalize_score(metrics.get('RSI_14D', 0) or 0, 30, 70) if 30 <= (metrics.get('RSI_14D', 0) or 0) <= 70 else 20.0
        macd_score = 100.0 if (metrics.get('MACD', 0) or 0) > 0 else 0.0
        ma_score = 100.0 if (metrics.get('MOV_AVG_50D', 0) or 0) > (metrics.get('MOV_AVG_200D', 0) or 0) else 60.0
        bollinger_upper = self._normalize_score(metrics.get('EQY_BOLLINGER_UPPER', metrics.get('MOV_AVG_50D', 0)) or metrics.get('MOV_AVG_50D', 0) or 0, (metrics.get('MOV_AVG_50D', 0) or 0) * 0.9, (metrics.get('MOV_AVG_50D', 0) or 0) * 1.1)
        return (rsi_score * 0.25 + macd_score * 0.25 + ma_score * 0.25 + bollinger_upper * 0.25)

    def calculate_earnings_quality(self) -> float:
        metrics = self.data['current']['earnings']
        return (self._normalize_score(metrics.get('surprise_pct', 0) or 0, -10, 15) * 0.4 +
                self._normalize_score(metrics.get('consistency', 0) or 0, 0, 1) * 0.3 +
                self._normalize_score(metrics.get('cfo_to_sales', 0) or 0, 0, 1) * 0.3)

    def calculate_atlas_score(self) -> Dict[str, float]:
        scores = {
            'financial_health': self.calculate_financial_health(),
            'market_performance': self.calculate_market_performance(),
            'valuation': self.calculate_valuation(),
            'analyst_recommendations': self.calculate_analyst_recommendations(),
            'technical': self.calculate_technical(),
            'earnings_quality': self.calculate_earnings_quality()
        }
        base_score = sum(scores[factor] * weight for factor, weight in self.weights.items()) * 0.80
        ml_adjustment = self._normalize_score(self.ml_prediction, -20, 30) * 0.15
        bsh_adjustment = self.bsh_score * 0.05
        atlas_score = base_score + ml_adjustment + bsh_adjustment
        return {'atlas_score': round(atlas_score, 2), 'component_scores': {k: round(v, 2) for k, v in scores.items()}}

    def _get_bsh_recommendation(self, score: float) -> str:
        if score >= 70:
            return "Buy"
        elif score >= 40:
            return "Hold"
        else:
            return "Sell"

    def display_results(self) -> None:
        result = self.calculate_atlas_score()
        
        st.markdown('<h1 style="color: green; text-align: center;">PROJECT ATLAS</h1>', unsafe_allow_html=True)
        
        st.subheader(f"{self.ticker}: Atlas Score: {result['atlas_score']}/100")
        st.subheader("Component Scores")
        for factor, score in result['component_scores'].items():
            st.write(f"{factor.replace('_', ' ').title()}: {score}/100 (Weight: {self.weights[factor]*100:.1f}%)")
        
        st.write("")  # Add a blank line for spacing
        st.write(f"**Recommendation**: {self._get_bsh_recommendation(result['atlas_score'])}")
        st.write(f"**Predicted Return**: {self.ml_prediction:.2f}%")
        
        st.subheader("Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**BSH Adjustment**: {round(self.bsh_score * 0.05, 2)}")
            st.write(f"**Analyst Consensus**: {self.data['current']['bsh']['consensus']}")
        with col2:
            st.write(f"**Number of Analysts**: {self.data['current']['bsh']['num_analysts']}")
            st.write(f"**Buy/Hold/Sell**: Buy: {self.data['current']['bsh']['buy_count']}, Hold: {self.data['current']['bsh']['hold_count']}, Sell: {self.data['current']['bsh']['sell_count']}")
        
        if self.missing_fields:
            st.warning(f"Missing data for: {', '.join(self.missing_fields)}")
        
        st.subheader("Score Interpretation")
        st.write("70-100: Strong Buy")
        st.write("40-69: Hold")
        st.write("0-39: Sell")
        
        self.display_deep_dive()

        if st.button("Download Relevant Financial Information"):
            self.download_relevant_financial_info()

    def display_deep_dive(self) -> None:
        current = self.data['current']
        bsh_data = current['bsh']
        sharpe = self._calculate_sharpe((current['market']['LAST_CLOSE_TRR_1YR'] or 0) / 100, (current['market']['VOLATILITY_260D'] or 0) / 100)

        st.subheader(f"Detailed Analysis: {self.ticker} (as of {datetime.now().strftime('%Y-%m-%d')})")

        st.write("### Financial Health")
        net_income = current['financial']['BEST_NET_INCOME'] or 0
        revenue = current['market']['TRAIL_12M_NET_REVENUE'] or 0
        net_profit_margin_fallback = (net_income / revenue * 100) if revenue != 0 else 0
        
        financial_data = {
            "Revenue Growth (YoY)": f"{current['financial']['SALES_GROWTH'] or 0:.2f}%",
            "Net Profit Margin": f"{current['financial']['ARDR_NON_GAAP_GROSS_MARGIN_PCT'] or net_profit_margin_fallback:.2f}%",
            "Debt-to-Equity Ratio": f"{current['financial']['TOT_DEBT_TO_TOT_EQY'] or 0:.2f}",
            "Operating Margin": f"{current['financial']['OPER_MARGIN'] or 0:.2f}%",
            "Net Debt to EBITDA": f"{current['financial']['NET_DEBT_TO_EBITDA'] or 0:.2f}",
            "Current Ratio": f"{current['financial']['CUR_RATIO'] or 0:.2f}",
            "Quick Ratio": f"{current['financial']['QUICK_RATIO'] or 0:.2f}",
            "EBITDA": 'N/A' if current['financial']['EBITDA'] is None else f"${current['financial']['EBITDA'] / 1e9:.2f}B",
            "Free Cash Flow": 'N/A' if current['financial']['CF_FREE_CASH_FLOW'] is None else f"${current['financial']['CF_FREE_CASH_FLOW'] / 1e9:.2f}B",
            "Net Income": 'N/A' if current['financial']['BEST_NET_INCOME'] is None else f"${current['financial']['BEST_NET_INCOME'] / 1e9:.2f}B",
            "Asset Turnover": f"{current['financial']['ASSET_TURNOVER'] or 0:.2f}",
            "Debt to Capital": f"{current['financial']['TOT_DEBT_TO_TOT_CAP'] or 0:.2f}",
            "Return on Equity": f"{current['financial']['RETURN_ON_EQY'] or 0:.2f}%",
            "Gross Margin": f"{current['financial']['GROSS_MARGIN'] or 0:.2f}%",
            "Enterprise Value": 'N/A' if current['financial']['CURR_ENTP_VAL'] is None else f"${current['financial']['CURR_ENTP_VAL'] / 1e9:.2f}B",
            "CapEx to Sales": f"{current['financial']['CAP_EXPEND_TO_SALES'] or 0:.2f}%",
            "5-Year Sales Growth": f"{current['financial']['SALES_5YR_AVG_GR'] or 0:.2f}%",
            "FCF Yield": f"{current['financial']['FCF_YIELD_WITH_CUR_ENTP_VAL'] or 0:.2f}%"
        }
        st.table(pd.DataFrame.from_dict(financial_data, orient='index', columns=['Value']))

        st.write("### Market Performance")
        market_data = {
            "Annualized Return": f"{current['market']['LAST_CLOSE_TRR_1YR'] or 0:.2f}%",
            "Trailing 12M Revenue": 'N/A' if current['market']['TRAIL_12M_NET_REVENUE'] is None else f"${current['market']['TRAIL_12M_NET_REVENUE'] / 1e9:.2f}B",
            "Volatility (90D)": f"{current['market']['VOLATILITY_90D'] or 0:.2f}%",
            "Volatility (260D)": f"{current['market']['VOLATILITY_260D'] or 0:.2f}%",
            "Beta": f"{current['market']['EQY_BETA'] or 0:.2f}",
            "Alpha": f"{current['market']['EQY_ALPHA'] or 0:.2f}",
            "Market Cap": 'N/A' if current['market']['CUR_MKT_CAP'] is None else f"${current['market']['CUR_MKT_CAP'] / 1e9:.2f}B",
            "Sharpe Ratio": f"{sharpe:.2f}"
        }
        st.table(pd.DataFrame.from_dict(market_data, orient='index', columns=['Value']))

        st.write("### Valuation")
        valuation_data = {
            "P/E Ratio": f"{current['valuation']['PE_RATIO'] or 0:.2f}",
            "P/S Ratio": f"{current['valuation']['PX_TO_SALES_RATIO'] or 0:.2f}",
            "EV/EBITDA": f"{current['valuation']['BEST_CUR_EV_TO_EBITDA'] or 0:.2f}",
            "P/FCF": f"{current['valuation']['PX_TO_FREE_CASH_FLOW'] or 0:.2f}",
            "PEG Ratio": f"{current['valuation']['BEST_PEG_RATIO_CUR_YR'] or 0:.2f}",
            "Book Value per Share": 'N/A' if current['valuation']['BOOK_VAL_PER_SH'] is None else f"${current['valuation']['BOOK_VAL_PER_SH']:.2f}",
            "EV/Sales": f"{current['valuation']['CURRENT_EV_TO_12M_SALES'] or 0:.2f}",
            "P/B Ratio": f"{current['valuation']['PX_TO_BOOK_RATIO'] or 0:.2f}",
            "Dividend Yield": f"{current['valuation']['EQY_DVD_YLD_IND'] or 0:.2f}%",
            "EPS Growth (1Yr)": f"{current['valuation']['EPS_GROWTH'] or 0:.2f}%"
        }
        st.table(pd.DataFrame.from_dict(valuation_data, orient='index', columns=['Value']))

        st.write("### Analyst Recommendations")
        analyst_data = {
            "Consensus Rating": bsh_data['consensus'],
            "Number of Analysts": f"{bsh_data['num_analysts']}",
            "Distribution": f"Buy: {bsh_data['buy_count']}, Hold: {bsh_data['hold_count']}, Sell: {bsh_data['sell_count']}"
        }
        st.table(pd.DataFrame.from_dict(analyst_data, orient='index', columns=['Value']))

        st.write("### Technical Indicators")
        technical_data = {
            "RSI (14-day)": f"{current['technical']['RSI_14D'] or 0:.2f}",
            "MACD Trend": 'Positive' if (current['technical']['MACD'] or 0) > 0 else 'Negative',
            "50-Day MA": 'N/A' if current['technical']['MOV_AVG_50D'] is None else f"${current['technical']['MOV_AVG_50D']:.2f}",
            "200-Day MA": 'N/A' if current['technical']['MOV_AVG_200D'] is None else f"${current['technical']['MOV_AVG_200D']:.2f}",
            "Bollinger Upper": 'N/A' if current['technical']['EQY_BOLLINGER_UPPER'] is None else f"${current['technical']['EQY_BOLLINGER_UPPER']:.2f}"
        }
        st.table(pd.DataFrame.from_dict(technical_data, orient='index', columns=['Value']))

        st.write("### Earnings Quality")
        earnings_data = {
            "Consistency": f"{current['earnings']['consistency']:.2f}",
            "CFO to Sales": f"{current['earnings']['cfo_to_sales'] or 0:.2f}"
        }
        st.table(pd.DataFrame.from_dict(earnings_data, orient='index', columns=['Value']))

    def download_relevant_financial_info(self) -> None:
        current = self.data['current']
        
        valuation_metrics = {
            'P/E': [current['valuation'].get('PE_RATIO', 28.93), 21.23, 19.57],
            'EV/Sales': [current['valuation'].get('CURRENT_EV_TO_12M_SALES', 6.10), 5.04, 4.80],
            'P/Sales': [current['valuation'].get('PX_TO_SALES_RATIO', 5.28), 4.32, 4.12],
            'P/CF': [current['valuation'].get('PX_TO_FREE_CASH_FLOW', 24.06), 18.00, 16.48],
            'P/Book': [current['valuation'].get('PX_TO_BOOK_RATIO', 5.25), 3.84, 3.41]
        }
        
        financial_metrics = {
            'ROE': f"{current['financial'].get('RETURN_ON_EQY', 15.65)}%",
            'EV/EBITDA': f"{current['valuation'].get('BEST_CUR_EV_TO_EBITDA', 16.22)}x",
            'Current Ratio': f"{current['financial'].get('CUR_RATIO', 2.53)}",
            'Short Interest': f"{current['market'].get('SHORT_INT_RATIO', 5.05)}%",
            'Dividend Yield': f"{current['valuation'].get('EQY_DVD_YLD_IND', 1.48)}%",
            'Debt Rating': current['financial'].get('TOT_DEBT_TO_TOT_EQY', 'BBB'),
            'Next Earnings Date': '05/20/2025',
            'Analyst Coverage': '4/6/0'
        }

        content = "Year\t2024\t2025E\t2026E\n"
        for metric, values in valuation_metrics.items():
            content += f"{metric}\t{values[0]:.2f}\t{values[1]:.2f}\t{values[2]:.2f}\n"
        
        content += "\nRelevant Information\n"
        content += "------------------\n"
        for metric, value in financial_metrics.items():
            content += f"{metric}\t{value}\n"

        if st.button(f"Download Relevant Financial Info for {self.ticker} as Text File"):
            st.download_button(
                label=f"Download {self.ticker}_financial_info.txt",
                data=content,
                file_name=f"{self.ticker}_financial_info.txt",
                mime="text/plain"
            )
            st.success(f"File '{self.ticker}_financial_info.txt' is ready for download!")

def main():
    st.markdown('<h1 style="color: green; text-align: center;">PROJECT ATLAS</h1>', unsafe_allow_html=True)

    st.subheader("Customize Weights (Must Sum to 1.0)")
    default_weights = {
        'financial_health': 0.25,
        'market_performance': 0.20,
        'valuation': 0.20,
        'analyst_recommendations': 0.15,
        'technical': 0.10,
        'earnings_quality': 0.10
    }
    
    strategy_descriptions = {
        "Custom": "Manually adjust weights to your preference.",
        "Growth": "Emphasizes financial health (0.25), market performance (0.20), and technical indicators (0.15) for growth-oriented stocks.",
        "Value": "Prioritizes financial health (0.35) and valuation (0.25) for undervalued stocks with strong fundamentals.",
        "Balanced": "Distributes weights equally (0.167 each) across all factors for a balanced approach."
    }
    
    if 'weights' not in st.session_state:
        st.session_state.weights = default_weights.copy()
    
    weights = st.session_state.weights
    factors = list(default_weights.keys())

    strategy = st.selectbox("Select Weight Strategy", list(strategy_descriptions.keys()), 
                            help="Choose a predefined strategy or customize weights manually.")
    st.write(f"**Description**: {strategy_descriptions[strategy]}")
    
    if strategy == "Growth":
        weights.update({'financial_health': 0.25, 'market_performance': 0.20, 'valuation': 0.20, 
                        'analyst_recommendations': 0.15, 'technical': 0.15, 'earnings_quality': 0.05})
    elif strategy == "Value":
        weights.update({'financial_health': 0.35, 'market_performance': 0.15, 'valuation': 0.25, 
                        'analyst_recommendations': 0.10, 'technical': 0.05, 'earnings_quality': 0.10})
    elif strategy == "Balanced":
        weights.update({f: 1.0/len(factors) for f in factors})
    
    for factor in factors:
        weights[factor] = st.number_input(f"{factor.replace('_', ' ').title()}", 
                                         min_value=0.0, 
                                         max_value=1.0, 
                                         value=weights[factor], 
                                         step=0.05, 
                                         format="%.2f",
                                         key=factor)

    total_weight = sum(weights.values())
    st.write(f"Current Total Weight: {total_weight:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Balance Weights"):
            if total_weight != 1.0:
                if total_weight > 1.0:
                    scale_factor = 1.0 / total_weight
                    for factor in factors:
                        weights[factor] *= scale_factor
                    st.success("Weights scaled down to sum to 1.0!")
                elif total_weight > 0:
                    remaining = 1.0 - total_weight
                    adjustment = remaining / len(factors)
                    for factor in factors:
                        weights[factor] += adjustment
                    st.success("Weights balanced to sum to 1.0!")
                st.session_state.weights = weights.copy()
                st.rerun()
            else:
                st.info("Weights already sum to 1.0. No action needed.")
    
    with col2:
        if st.button("Reset Weights"):
            st.session_state.weights = default_weights.copy()
            st.success("Weights reset to default values!")
            st.rerun()

    if total_weight != 1.0:
        st.error("Total weight must equal 1.0 before scanning. Use 'Balance Weights' or adjust manually.")
        return

    ticker = st.text_input("Enter the stock ticker symbol", "AAPL").strip().upper()
    
    if st.button("Scan Stock"):
        if not ticker:
            st.error("Please enter a stock ticker symbol.")
            return
        
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                scanner = AtlasStockScanner(ticker, weights)
                scanner.display_results()
            except Exception as e:
                st.error(f"Error processing {ticker}: {str(e)}")
                logger.error(f"Error processing {ticker}: {str(e)}")

if __name__ == "__main__":
    main()
