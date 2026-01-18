import pandas as pd
import numpy as np
from variables import PROJECT_ROOT_DIRECTORY, COLUMNS_DICT#, REGRESSOR_DICT, CLASSIFIER_DICT
from config import REGRESSOR_PARAM_GRIDS, CLASSIFIER_PARAM_GRIDS

# Import commonly used machine learning models
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge, LogisticRegression
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
#import shap

organic_genome_path = PROJECT_ROOT_DIRECTORY+'02-metadata/06-csv-files/01-organic-genome.csv'
organic_descriptors_path = PROJECT_ROOT_DIRECTORY+'02-metadata/06-csv-files/02-organic-descriptors.csv'
mo_energetics_dataframe_path = PROJECT_ROOT_DIRECTORY+'02-metadata/06-csv-files/04-mo-energetics.csv'
pca_dataframe_path = PROJECT_ROOT_DIRECTORY+'02-metadata/06-csv-files/03-pca.csv'

train_test_split_dataframe_path = PROJECT_ROOT_DIRECTORY+'01-rawdata/10-machine-learning/train-test-split.csv'

def get_pca_dataframe(organic_descriptors_dataframe, jitter_coef=0.2, save=False): #jitter range summing both sides
    standardized_descriptors = StandardScaler().fit_transform(organic_descriptors_dataframe.values)
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(standardized_descriptors)
    pca_dataframe = pd.DataFrame(data=pca_components, columns=['d1','d2'],index=organic_descriptors_dataframe.index)
    pca_dataframe['d1_jitter'] = pca_dataframe['d1'] + jitter_coef * np.random.rand(pca_dataframe.shape[0]) -jitter_coef*0.5
    pca_dataframe['d2_jitter'] = pca_dataframe['d2'] + jitter_coef * np.random.rand(pca_dataframe.shape[0]) -jitter_coef*0.5
    print('pca components: ', pca.components_, 'pca_variance: ', pca.explained_variance_)
    if save:
        pca_dataframe.to_csv(pca_dataframe_path)
    return pca_dataframe

def split_train_test_identifier(dataframe, train_set_query='generation <= 3.0', random_state=4, test_size=0.2, save_csv=False):

    train_test_dataframe = dataframe.query(train_set_query)

    train_test_identifier_list = train_test_dataframe.index.to_list()
    train_identifier_list, test_identifier_list = train_test_split(
        train_test_identifier_list, random_state=random_state, test_size=test_size
        )

    print('data has been split into train set and test set with test size: ', test_size)
    print('train_identifier_list and test_identifier_list created with specified random_state: ', random_state)
    if save_csv:
        train_test_split_dataframe = pd.DataFrame(index=dataframe.index)
        for identifier in train_identifier_list:
            train_test_split_dataframe.at[identifier, 'type'] = 'train'
        for identifier in test_identifier_list:
            train_test_split_dataframe.at[identifier, 'type'] = 'test'

        train_test_split_dataframe.to_csv(train_test_split_dataframe_path)
        print('train test split information saved to csv file.')
    return train_identifier_list, test_identifier_list

# Dictionary to dynamically create regressor models
REGRESSOR_MODELS = {
    "linear_regression": LinearRegression,
    "lasso": Lasso,
    "ridge": Ridge,
    "elastic_net": ElasticNet,
    "svr_linear": lambda: SVR(kernel='linear'),
    "svr_rbf": lambda: SVR(kernel='rbf'),
    "svr_poly": lambda: SVR(kernel='poly'),
    "knn_regressor": KNeighborsRegressor,
    "random_forest_regressor": RandomForestRegressor,
}

# Dictionary to dynamically create classifier models
CLASSIFIER_MODELS = {
    "svc_linear": lambda: SVC(kernel='linear'),
    "linear_svc": LinearSVC,
    "knn_classifier": KNeighborsClassifier,
    "decision_tree_classifier": DecisionTreeClassifier,
    "svc_rbf": lambda: SVC(kernel='rbf'),
    "gaussian_nb": GaussianNB,
    "logistic_regression": LogisticRegression,
}

# Example method to create a model instance and retrieve its parameter grid
def create_model_and_params(model_name, model_type='regressor'):
    if model_type == 'regressor':
        model = REGRESSOR_MODELS[model_name]()
        param_grid = REGRESSOR_PARAM_GRIDS[model_name]
    else:
        model = CLASSIFIER_MODELS[model_name]()
        param_grid = CLASSIFIER_PARAM_GRIDS[model_name]
    return model, param_grid


class MolecularOrbitalPredictor:

    # input_dataframes is preprocessed dataframe contains genome, features, targets
    def __init__(self, input_dataframe, target):
        self.target = target
        self.input_dataframe = input_dataframe
        train_identifier_list, test_identifier_list = split_train_test_identifier(
            dataframe=input_dataframe
        )
        self.X_train_test = input_dataframe.loc[train_identifier_list + test_identifier_list, COLUMNS_DICT['machine_learning_features']].values
        self.X_train = input_dataframe.loc[train_identifier_list, COLUMNS_DICT['machine_learning_features']].values
        self.X_test = input_dataframe.loc[test_identifier_list, COLUMNS_DICT['machine_learning_features']].values
        self.y_train_test = input_dataframe.loc[train_identifier_list + test_identifier_list, target].values
        self.y_train = input_dataframe.loc[train_identifier_list, target].values
        self.y_test = input_dataframe.loc[test_identifier_list, target].values

        print('Follow step: train lasso regression model')
        

    def train_regressor(self, model_name):
        print ('Training model:', model_name)
        model, param_grid = create_model_and_params(model_name, model_type='regressor')
        # This pipeline involves standard scaler and lasso regression model
        pipeline = Pipeline([("scaler", StandardScaler()), (model_name, model)])
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5)        
        grid_search.fit(self.X_train, self.y_train)
        best_pipeline = grid_search.best_estimator_
        best_param = grid_search.best_params_
        print('best parameter: ', best_param)
        print('best score: ', grid_search.best_score_)
        return best_pipeline, best_param



        # this pipeline involves standard scaler and lasso regression model
        #pipeline = Pipeline(
        #    [("scaler", StandardScaler()), 
        #     (model_name, REGRESSOR_DICT[model_name]['model'])]
        #)
        #grid_search = GridSearchCV(
        #    estimator=pipeline,
        #    param_grid=REGRESSOR_DICT[model_name]['param'],
        #    cv=5
        #)
        #grid_search.fit(self.X_train, self.y_train)
        #best_pipeline = grid_search.best_estimator_
        #best_param = grid_search.best_params_
        #print('best parameter: ', best_param)
        #print('best score: ', grid_search.best_score_)
        #return best_pipeline, best_param

    def compare_models(self):
        model_comparison_by_score, model_comparison_by_prediction, model_comparison_by_feature_importance = {}, {}, {}

        for key, value in REGRESSOR_DICT.items():
            best_pipeline, best_param = self.train_regressor(model_name=key)

            y_train_test_prediction = best_pipeline.predict(self.X_train_test)
            model_score = {
                'best_param': best_param,
                'r2_score': r2_score(self.y_train_test,y_train_test_prediction),
                'train_score': best_pipeline.score(self.X_train, self.y_train),
                'test_score': best_pipeline.score(self.X_test, self.y_test),
                'rmse': root_mean_squared_error(self.y_train_test,y_train_test_prediction)
            }
            model_comparison_by_score.update({key: model_score})

            X_apply = self.input_dataframe.loc[:, COLUMNS_DICT['machine_learning_features']].values
            y_apply_prediction = best_pipeline.predict(X_apply)
            model_comparison_by_prediction.update({key: y_apply_prediction})

            if key in ['linear_regression', 'lasso', 'ridge', 'elastic_net', 'svr_linear']:
                feature_importance = best_pipeline[1].coef_
                model_comparison_by_feature_importance.update({key: feature_importance.tolist()})

        model_comparison_by_score_dataframe = pd.DataFrame(data=model_comparison_by_score)
        model_comparison_by_prediction_dataframe = pd.DataFrame(data=model_comparison_by_prediction, index=self.input_dataframe.index)
        model_comparison_by_feature_importance_dataframe = pd.DataFrame(
            data=model_comparison_by_feature_importance, index=COLUMNS_DICT['machine_learning_features']
        )

        return model_comparison_by_score_dataframe, model_comparison_by_prediction_dataframe, model_comparison_by_feature_importance_dataframe

    def train_lasso_regression(self):
        
        self.best_pipeline, best_param = self.train_regressor('lasso')

        self.explainer = shap.Explainer(
            self.best_pipeline[1], 
            self.best_pipeline[0].transform(self.X_train),
            feature_names=COLUMNS_DICT['machine_learning_features'])

        train_score = self.best_pipeline.score(self.X_train, self.y_train) 
        test_score = self.best_pipeline.score(self.X_test, self.y_test) 

        print('train score: ', train_score)
        print('test score: ', test_score)
        y_train_test_prediction = self.best_pipeline.predict(self.X_train_test)
        r_2_score = r2_score(self.y_train_test,y_train_test_prediction)
        rmse = root_mean_squared_error(self.y_train_test,y_train_test_prediction)
        print('R2 score: ', r_2_score)
        print('RMSE: ', rmse)

        self.feature_importance = self.best_pipeline[1].coef_
        return self.feature_importance
    
    def predict_n_explain(self, fingerprint_dataframe, base_identifier=26):
        X_apply = fingerprint_dataframe.loc[:, COLUMNS_DICT['machine_learning_features']].values
        y_apply_prediction = self.best_pipeline.predict(X_apply)
        prediction_dataframe = pd.DataFrame(
            data={self.target+'_predicted': y_apply_prediction}, index=fingerprint_dataframe.index
        )
        shap_values = self.explainer(self.best_pipeline[0].transform(X_apply))

        if base_identifier:
            print('shapley value is calibrated by base identifier ', base_identifier)
            base_shap_value = shap_values.values[base_identifier - 1]
            shap_values.values = shap_values.values - base_shap_value
            shap_values.base_values = np.repeat(y_apply_prediction[base_identifier - 1], len(shap_values.base_values))

        shap_values_dataframe = pd.DataFrame(
            data=shap_values.values, columns=COLUMNS_DICT['machine_learning_features'],
            index=fingerprint_dataframe.index
        )
        return prediction_dataframe, shap_values_dataframe, shap_values

class ExistencePredictor:
    # input_dataframes is preprocessed dataframe contains genome, features, targets
    def __init__(self, input_dataframe, target):
        self.target = target
        self.input_dataframe = input_dataframe
        train_identifier_list, test_identifier_list = split_train_test_identifier(
            dataframe=input_dataframe
        )
        self.X_train_test = input_dataframe.loc[train_identifier_list + test_identifier_list, COLUMNS_DICT['machine_learning_features']].values
        self.X_train = input_dataframe.loc[train_identifier_list, COLUMNS_DICT['machine_learning_features']].values
        self.X_test = input_dataframe.loc[test_identifier_list, COLUMNS_DICT['machine_learning_features']].values
        self.y_train_test = input_dataframe.loc[train_identifier_list + test_identifier_list, target].values
        self.y_train = input_dataframe.loc[train_identifier_list, target].values
        self.y_test = input_dataframe.loc[test_identifier_list, target].values

        print('Follow step: train logistic regression model')

    def train_classifier(self, model_name):
        print ('training model ' + model_name + '...')
        # this pipeline involves standard scaler and lasso regression model
        pipeline = Pipeline(
            [("scaler", StandardScaler()), 
             (model_name, CLASSIFIER_DICT[model_name]['model'])]
        )
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=CLASSIFIER_DICT[model_name]['param'],
            cv=5
        )
        grid_search.fit(self.X_train, self.y_train)
        best_pipeline = grid_search.best_estimator_
        best_param = grid_search.best_params_
        print('best parameter: ', best_param)
        print('best score: ', grid_search.best_score_)
        return best_pipeline, best_param

    def compare_models(self):
        model_comparison_by_score, model_comparison_by_prediction = {}, {}

        for key, value in CLASSIFIER_DICT.items():
            best_pipeline, best_param = self.train_classifier(model_name=key)

            y_train_test_prediction = best_pipeline.predict(self.X_train_test)
            model_score = {
                'best_param': best_param,
                'accuracy_score': accuracy_score(self.y_train_test,y_train_test_prediction),
                'train_score': best_pipeline.score(self.X_train, self.y_train),
                'test_score': best_pipeline.score(self.X_test, self.y_test),
            }
            model_comparison_by_score.update({key: model_score})

            X_apply = self.input_dataframe.loc[:, COLUMNS_DICT['machine_learning_features']].values
            y_apply_prediction = best_pipeline.predict(X_apply)
            model_comparison_by_prediction.update({key: y_apply_prediction})

        model_comparison_by_score_dataframe = pd.DataFrame(data=model_comparison_by_score)
        model_comparison_by_prediction_dataframe = pd.DataFrame(data=model_comparison_by_prediction, index=self.input_dataframe.index)


        return model_comparison_by_score_dataframe, model_comparison_by_prediction_dataframe   

    def train_logistic_regression(self):
        
        self.best_pipeline, best_param = self.train_classifier('logistic_regression')

        self.explainer = shap.Explainer(
            self.best_pipeline[1], 
            self.best_pipeline[0].transform(self.X_train),
            feature_names=COLUMNS_DICT['machine_learning_features'])

        train_score = self.best_pipeline.score(self.X_train, self.y_train) 
        test_score = self.best_pipeline.score(self.X_test, self.y_test) 

        print('train score: ', train_score)
        print('test score: ', test_score)
        y_train_test_prediction = self.best_pipeline.predict(self.X_train_test)
        score = accuracy_score(self.y_train_test,y_train_test_prediction)
        print('accuracy score: ', score)

        self.feature_importance = self.best_pipeline[1].coef_
        return self.feature_importance
    
    def predict_n_explain(self, fingerprint_dataframe, base_identifier=26):
        X_apply = fingerprint_dataframe.loc[:, COLUMNS_DICT['machine_learning_features']].values
        y_apply_prediction = self.best_pipeline.predict(X_apply)
        y_apply_proba_prediction = self.best_pipeline.predict_proba(X_apply)
        y_apply_decision_function = self.best_pipeline.decision_function(X_apply)

        prediction_dataframe = pd.DataFrame(
            data={self.target+'_predicted': y_apply_prediction,
                  self.target+'_predicted_proba': y_apply_proba_prediction[:,1],
                  self.target+'_decision_function': y_apply_decision_function,
                  }, 
            index=fingerprint_dataframe.index
        )
        shap_values = self.explainer(self.best_pipeline[0].transform(X_apply))

        if base_identifier:
            print('shapley value is calibrated by base identifier ', base_identifier)
            base_shap_value = shap_values.values[base_identifier - 1]
            shap_values.values = shap_values.values - base_shap_value
            shap_values.base_values = np.repeat(y_apply_prediction[base_identifier - 1], len(shap_values.base_values))

        shap_values_dataframe = pd.DataFrame(
            data=shap_values.values, columns=COLUMNS_DICT['machine_learning_features'],
            index=fingerprint_dataframe.index
        )
        return prediction_dataframe, shap_values_dataframe, shap_values


def get_shap_coefficient(fingerprint_dataframe, shap_values_dataframe):
    shap_coefficient = {}
    for feature_name in COLUMNS_DICT['machine_learning_features']:
        x = fingerprint_dataframe.loc[:, feature_name].values
        y = shap_values_dataframe.loc[:, feature_name].values
        slope, intercept = np.polyfit(x, y, 1)
        shap_coefficient.update({feature_name: slope})
    return shap_coefficient


def get_possible_shap_value_dataframe(generation_max, shap_coefficient):
    feature_value_six_ring, feature_value_linkage = [1], [0]
    generation_value_six_ring, generation_value_linkage = [0], [0]
    for generation in np.arange(1, generation_max+1, 1):
        feature_value_six_ring_this_gen = np.linspace(0,1,generation+2)
        for feature_value in feature_value_six_ring_this_gen:
            if feature_value not in feature_value_six_ring:
                feature_value_six_ring.append(feature_value)
                generation_value_six_ring.append(generation)
        feature_value_linkage_fusion_this_gen = np.linspace(0, 1, generation+1)
        for feature_value in feature_value_linkage_fusion_this_gen:
            if feature_value not in feature_value_linkage:
                feature_value_linkage.append(feature_value)
                generation_value_linkage.append(generation)

    feature_value = {
        'ringcount': np.arange(1, generation_max+2, 1),
        'six_ring_p': np.array(feature_value_six_ring),
        'linkage_p': np.array(feature_value_linkage),
        'primaryamine': np.arange(1,3,1),
        'linker_length': np.arange(0, generation_max+3, 1),
        'linker_position': np.array(feature_value_linkage),
        'hetero_nitrogen': np.arange(0, generation_max+1, 1),
        'fluorination': np.arange(0, generation_max+1, 1),
        'furan': np.arange(0, generation_max-1, 1),
        'pyrrole': np.arange(0, generation_max-1, 1),
        'sidechain_on_linker': np.arange(0, generation_max-1, 1),
        'sidechain_on_backbone': np.arange(0, generation_max-1, 1),
    }

    generation_value = {
        'ringcount': np.arange(0, generation_max+1, 1),
        'six_ring_p': np.array(generation_value_six_ring),
        'linkage_p': np.array(generation_value_linkage),
        'primaryamine': np.arange(0,2,1),
        'linker_length': abs(np.arange(-2, generation_max+1, 1)),
        'linker_position': np.array(generation_value_linkage),
        'hetero_nitrogen': np.arange(0, generation_max+1, 1),
        'fluorination': np.arange(0, generation_max+1, 1),
        'furan': np.arange(2, generation_max+1, 1),
        'pyrrole': np.arange(2, generation_max+1, 1),
        'sidechain_on_linker': np.arange(0, generation_max-1, 1),
        'sidechain_on_backbone': np.arange(0, generation_max-1, 1),
    }

    base_value = {
        'ringcount': 1,
        'six_ring_p': 1,
        'linkage_p': 0,
        'primaryamine': 2,
        'linker_length': 2,
        'linker_position': 1,
        'hetero_nitrogen': 0,
        'fluorination': 0,
        'furan': 0,
        'pyrrole': 0,
        'sidechain_on_linker': 0,
        'sidechain_on_backbone': 0,
    }

    shap_value_all_possible_dataframe = pd.DataFrame()
    for i in range(len(COLUMNS_DICT['machine_learning_features'])):
        feature_name = COLUMNS_DICT['machine_learning_features'][i]
        dataframe = pd.DataFrame(data=(feature_value[feature_name]-base_value[feature_name])*shap_coefficient[feature_name],columns=['shap_value'])
        dataframe['feature_name'] = str(feature_name)
        dataframe['feature_index'] = i
        dataframe['generation_value'] = generation_value[feature_name]
        shap_value_all_possible_dataframe = pd.concat([shap_value_all_possible_dataframe, dataframe], ignore_index=True)
    return shap_value_all_possible_dataframe