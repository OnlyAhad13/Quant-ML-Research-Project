import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FeatureValidator:
    """
    Comprehensive feature validation using Information Coefficient and statistical tests
    """

    def __init__(self, data: pd.DataFrame, target_col: str = 'target'):
        self.data = data.copy()
        self.target_col = target_col
        self.feature_cols = [col for col in data.columns if col not in 
                            ['open', 'high', 'low', 'close', 'volume', 'return', 
                             'log_return', target_col]]
        self.validation_results = {}
    
    def validate_all(self) -> pd.DataFrame:
        """
        Run complete validation pipeline
        """

        #Remove rows with NaN in target
        clean_data = self.data.dropna(subset=[self.target_col])
        print(f"Data shape after cleaning: {clean_data.shape}")
        print(f"Features to validate: {len(self.feature_cols)}")

        #Information Coefficient 
        ic_results = self._compute_ic(clean_data)
        
        #IC stability over time
        stability_results = self._compute_ic_stability(clean_data)

        #Statistical significance
        significance_results = self._test_significance(clean_data)

        #Feature correlations
        correlation_results = self._compute_feature_correlations(clean_data)

        #Mutual Information
        mi_results = self._compute_mutual_information(clean_data)

        # Combine all results
        summary = pd.DataFrame({
            'feature': self.feature_cols,
            'ic_mean': [ic_results.get(f, {}).get('ic_mean', np.nan) for f in self.feature_cols],
            'ic_std': [ic_results.get(f, {}).get('ic_std', np.nan) for f in self.feature_cols],
            'ic_median': [ic_results.get(f, {}).get('ic_median', np.nan) for f in self.feature_cols],
            'stability_mean': [stability_results.get(f, {}).get('mean_ic', np.nan) for f in self.feature_cols],
            'stability_std': [stability_results.get(f, {}).get('std_ic', np.nan) for f in self.feature_cols],
            'stability_ratio': [stability_results.get(f, {}).get('stability_ratio', np.nan) for f in self.feature_cols],
            'p_value': [significance_results.get(f, {}).get('p_value', np.nan) for f in self.feature_cols],
            'significant': [significance_results.get(f, {}).get('significant', False) for f in self.feature_cols],
            'mutual_info': [mi_results.get(f, np.nan) for f in self.feature_cols],
            'abs_ic_mean': [abs(ic_results.get(f, {}).get('ic_mean', 0)) for f in self.feature_cols]
        })

        # Sort by absolute IC
        summary = summary.sort_values('abs_ic_mean', ascending=False)
        
        self.validation_results['summary'] = summary
        self.validation_results['ic_results'] = ic_results
        self.validation_results['stability_results'] = stability_results
        self.validation_results['correlation_matrix'] = correlation_results
        
        print("\nValidation Complete!")
        print(f"\nTop 10 Features by IC:")
        print(summary[['feature', 'ic_mean', 'stability_ratio', 'significant']].head(10))
    
        return summary
    
    def _compute_ic(self, data: pd.DataFrame) -> Dict:
        """
        Compute Information Coefficient(Spearman rank correlation)
        """

        results = {}
        for feature in self.feature_cols:
            if feature not in data.columns:
                continue
            
            #Remove NaN values
            valid_mask = data[feature].notna() & data[self.target_col].notna()
            feat_vals = data.loc[valid_mask, feature]
            target_vals = data.loc[valid_mask, self.target_col]

            if len(feat_vals) < 30:
                continue
            
            #Compute spearman IC
            ic, p_val = spearmanr(feat_vals, target_vals)
            #Compute pearson as well
            pearson_ic, _ = pearsonr(feat_vals, target_vals)

            results[feature] = {
                'ic_mean': ic, 
                'ic_std': 0, #Will be updated in stability analysis
                'ic_median': ic,
                'pearson_ic': pearson_ic,
                'n_obs': len(feat_vals)
            }
        
        return results
    
    def _compute_ic_stability(self, data: pd.DataFrame, window='90D') -> Dict:
        """
        Compute IC over rolling time windows to assess stability
        """

        results = {}

        #Get date index
        dates = data.index.get_level_values('date').unique().sort_values()

        if len(dates) < 100:
            print("Insufficient data for stability analysis")
            return results

        #Define time windows (quarterly)
        time_windows = pd.date_range(dates.min(), dates.max(), freq='q')

        for feature in self.feature_cols:
            if feature not in data.columns:
                continue
            
            ic_series = []
            for window_end in time_windows:
                window_start = window_end - pd.Timedelta(days=90)
            
                #Filter data for this window
                mask = (
                    (data.index.get_level_values('date') >= window_start) &
                    (data.index.get_level_values('date') <= window_end) &
                    data[feature].notna() &
                    data[self.target_col].notna()
                )

                window_data = data.loc[mask]

                if len(window_data) < 30:
                    continue

                #Compute IC for this window
                ic, _ = spearmanr(window_data[feature], window_data[self.target_col])
                ic_series.append(ic)

            if len(ic_series) > 0:
                ic_array = np.array(ic_series)
                mean_ic = np.nanmean(ic_array)
                std_ic = np.nanstd(ic_array)

                #Stability ratio: higher is better (consistent signal)
                stability_ratio = abs(mean_ic) / std_ic

                results[feature] = {
                    'mean_ic': mean_ic,
                    'std_ic': std_ic,
                    'stability_ratio': stability_ratio,
                    'ic_timeseries': ic_array,
                    'n_windows': len(ic_series)
                }
            
        return results

    def _test_significance(self, data: pd.DataFrame, alpha=0.05) -> Dict:
        """
        Test if IC is significantly different from zero using t-test
        """
        results = {}
        
        for feature in self.feature_cols:
            if feature not in data.columns:
                continue
            
            valid_mask = data[feature].notna() & data[self.target_col].notna()
            feat_vals = data.loc[valid_mask, feature]
            target_vals = data.loc[valid_mask, self.target_col]
            
            if len(feat_vals) < 30:
                continue
            
            # Compute IC
            ic, _ = spearmanr(feat_vals, target_vals)
            
            # T-test: H0: IC = 0
            n = len(feat_vals)
            t_stat = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-8))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            results[feature] = {
                'ic': ic,
                't_stat': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'n_obs': n
            }
        
        return results
    
    def _compute_feature_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pairwise correlations between features to identify redundancy
        """
        # Sample features if too many
        features_to_analyze = self.feature_cols[:100] if len(self.feature_cols) > 100 else self.feature_cols
        
        feature_data = data[features_to_analyze].dropna()
        
        if len(feature_data) == 0:
            return pd.DataFrame()
        
        corr_matrix = feature_data.corr(method='pearson')
        
        return corr_matrix


    def _test_significance(self, data: pd.DataFrame, alpha=0.05) -> Dict:
        """
        Test if IC is significantly different from zero using t-test
        """
        results = {}
        
        for feature in self.feature_cols:
            if feature not in data.columns:
                continue
            
            valid_mask = data[feature].notna() & data[self.target_col].notna()
            feat_vals = data.loc[valid_mask, feature]
            target_vals = data.loc[valid_mask, self.target_col]
            
            if len(feat_vals) < 30:
                continue
            
            # Compute IC
            ic, _ = spearmanr(feat_vals, target_vals)
            
            # T-test: H0: IC = 0
            n = len(feat_vals)
            t_stat = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-8))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            results[feature] = {
                'ic': ic,
                't_stat': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'n_obs': n
            }
        
        return results

    def _compute_feature_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pairwise correlations between features to identify redundancy
        """
        # Sample features if too many
        features_to_analyze = self.feature_cols[:100] if len(self.feature_cols) > 100 else self.feature_cols
        
        feature_data = data[features_to_analyze].dropna()
        
        if len(feature_data) == 0:
            return pd.DataFrame()
        
        corr_matrix = feature_data.corr(method='pearson')
        
        return corr_matrix  
    
    def _compute_mutual_information(self, data: pd.DataFrame, n_samples=50000) -> Dict:
        """
        Compute mutual information between features and target
        """

        results = {}
        #Sample data if too large
        if len(data) > n_samples:
            sample_data = data.sample(n=n_samples, random_state=42)
        else:
            sample_data = data
        
        #prepare feature matrix
        valid_mask = sample_data[self.target_col].notna()
        
        for feature in self.feature_cols:
            if feature not in sample_data.columns:
                continue

            feat_mask = valid_mask & sample_data[feature].notna()

            if feat_mask.sum() < 100:
                continue
            
            X = sample_data.loc[feat_mask, [feature]].values
            y = sample_data.loc[feat_mask, self.target_col].values
            try:
                mi = mutual_info_regression(X, y, random_state=42)[0]
                results[feature] = mi
            except Exception as e:
                results[feature] = np.nan
            
        return results

    def select_top_features(
        self,
        n_features: int = 50,
        method: str = 'ic',
        min_stability: float = 0.0
    ) -> List[str]:
        """
        Select top N features based on validation metrics
        """
        if 'summary' not in self.validation_results:
            raise ValueError("Must run validate_all() first")
        
        summary = self.validation_results['summary'].copy()

        #Filter by stability if specified
        if min_stability > 0:
            summary = summary[summary['stabilty_ratio'] >= min_stability]
            
    def remove_correlated_features(self, features: List[str], threshold: float = 0.85) -> List[str]:
        """Remove highly correlated features to reduce redundancy"""

        if 'correlation_matrix' not in self.validation_results:
            return features
        
        corr_matrix = self.validation_results['correlation_matrix']
        features_to_keep = list(features)

        #Find pairs with correlation above threshold
        for i, feat1 in enumerate(features_to_keep):
            if feat1 not in corr_matrix.columns:
                continue

            for feat2 in features_to_keep[i+1:]:
                    if feat2 not in corr_matrix.columns:
                        continue

                    if(abs(corr_matrix.loc[feat1, feat2]) >= threshold):
                        #Keep the one with higher IC
                        summary = self.validation_results['summary']
                        ic1 = summary[summary['feature'] == feat1]['abs_ic_mean'].values[0]
                        ic2 = summary[summary['feature'] == feat2]['abs_ic_mean'].values[0]

                        if ic1 >= ic2 and feat2 in features_to_keep:
                            features_to_keep.remove(feat2)
                        elif feat1 in features_to_keep:
                            features_to_keep.remove(feat1)
        
        print(f"\nRemoved {len(features) - len(features_to_keep)} highly correlated features")
        print(f"Remaining: {len(features_to_keep)} features")

        return features_to_keep
    
    def feature_importance_ml(
        self, 
        features: List[str],
        n_estimators: int = 100,
        max_samples: int = 50000,
    ) -> pd.DataFrame:
        """
        Compute feature importance using Random Forest
        """

        print("Computing ML-based feature importance.....")
        
        #Prepare data
        data = self.data[features + [self.target_col]].dropna()

        if len(data) > max_samples:
            data = data.sample(n=max_samples, random_state=42)
        
        X = data[features]
        y = data[self.target_col]

        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=8, min_samples_split=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        #Get importance
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"   R² Score: {rf.score(X, y):.4f}")
        print(f"\nTop 10 by ML Importance:")
        print(importance_df.head(10))
        
        return importance_df
    
    def get_validation_summary(self) -> Dict:
        """Get comprehensive validation summary"""

        if 'summary' not in self.validation_results:
            raise ValueError("Must run validate_all() first")
        
        summary = self.validation_results['summary']

        return {
            'total_features': len(summary),
            'significant_features': summary['significant'].sum(),
            'mean_ic': summary['ic_mean'].mean(),
            'median_ic': summary['ic_mean'].median(),
            'mean_stability': summary['stability_ratio'].mean(),
            'top_10_features': summary.head(10)['feature'].tolist(),
            'top_10_ics': summary.head(10)['ic_mean'].tolist()
        }
    
class ICAnalyzer:
    """Advanced IC analysis and visualization support"""

    @staticmethod
    def compute_cumulative_ic(
        data: pd.DataFrame,
        feature: str,
        target: str = 'target'
    ) -> pd.Series:
        """
        Compute cumulative IC over time
        """

        dates = data.index.get_level_values('date').unique().sort_values()
        cumulative_ic = []

        for date in dates:
            date_data = data.loc[data.index.get_level_values('date') <= date]
            valid_mask = date_data[feature].notna() & date_data[target].notna()

            if valid_mask.sum() < 30:
                cumulative_ic.append(np.nan)
                continue

            ic, _ = spearmanr(date_data.loc[valid_mask, feature], date_data.loc[valid_mask, target])
            cumulative_ic.append(ic)

        return pd.Series(cumulative_ic, index=dates)
    
    @staticmethod
    def compute_rolling_ic(data: pd.DataFrame, feature: str, window: int = 252, target: str = 'target') -> pd.Series:
        """Compute rolling IC with specified time"""

        dates = data.index.get_level_values('date').unique().sort_values()
        rolling_ic = []

        for i, date in enumerate(dates):
            if i < window:
                rolling_ic.append(np.nan)
                continue
                
            window_dates = dates[max(0, i-window): i]
            window_data = data.loc[data.index.get_level_values('date').isin(window_dates)]

            valid_mask = window_data[feature].notna() & window_data[target].notna()

            if valid_mask.sum() < 30:
                rolling_ic.append(np.nan)
                continue
            
            ic, _ = spearmanr(window_data.loc[valid_mask, feature], window_data.loc[valid_mask, target])
            rolling_ic.append(ic)

        return pd.Series(rolling_ic, index=dates)

if __name__ == "__main__":
    print("Feature Validation module ready for import")




