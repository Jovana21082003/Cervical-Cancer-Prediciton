import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import RANSACRegressor
from statsmodels.stats.diagnostic import het_white
from pomocne_funkcije import *
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import HuberRegressor
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt



df= pd.read_csv("/Users/jovanavlaskalic/Projekat_NANS/data.csv")
print(df.shape)
# 1. PRETPROCESIRANJE PODATAKA

#print(df.isna().sum()) #omogucava da se vidi broj kolona koje ne sadrze ovaj podatak

#matrica korelacije
correlation_matrix = df[['Number_of_STD_Diagnosis', 'Time_Since_First_STD_Diagnosis', 
                         'Time_Since_Last_STD_Diagnosis', 'Number_of_Years_with_STDs', 
                         'IUD_in_Years', 'Is_Diagnosis_Cancer','Smoking_in_Years', 'Smoking_in_Packs_per_Year']].corr()
print(correlation_matrix['Is_Diagnosis_Cancer'])


# izbačene kolone sa 85% nedostajućih vrednosti
df = df.drop(columns=['Number_of_STD_Diagnosis', 'Time_Since_First_STD_Diagnosis', 'Time_Since_Last_STD_Diagnosis', 'Number_of_Years_with_STDs', 'IUD_in_Years', 'Smoking_in_Years'])


# Umesto true/false stavljamo brojeve
bool_columns = df.select_dtypes(include='bool').columns

#Ovo je jednostavnije 0/1 -> one hot encoding povecao dimenzionalnost problema
df[bool_columns] = df[bool_columns].astype(int)
#print(df.head()) 

#Vizualizacija 
#total_cancer = df['Is_Diagnosis_Cancer'].value_counts()
#cancer_with_hpv = df[(df['Is_On_Hormonal_Contraceptives'] == 1) & (df['Is_Diagnosis_Cancer'] == 1)].shape[0]
#nocancer_with_hpv = df[(df['Is_On_Hormonal_Contraceptives'] == 1) & (df['Is_Diagnosis_Cancer'] == 0)].shape[0]
# Priprema podataka za prikaz
#data = {
 #   "Grupa": ["Ukupan broj žena sa kancerom", "Broj žena sa kancerom koje su koristile kontraceptivne pilule", 
  #            "Ukupan broj žena bez kancera", "Broj žena bez kancera i koje su koristile kontraceptivne pilule"],
  #  "Broj": [total_cancer[1], cancer_with_hpv, 
   #          total_cancer[0], nocancer_with_hpv]
#}

#viz_data = pd.DataFrame(data)
#
# Vizualizacija podataka
#plt.figure(figsize=(10, 6))
#sns.barplot(x="Grupa", y="Broj", data=viz_data, palette="flare")

# Postavljanje naslova i oznaka
#plt.title("Odnos između raka i upotrebe kontraceptivnih pilula", fontsize=14)
#plt.xlabel("")
#plt.ylabel("Broj pacijentkinja", fontsize=12)
#plt.xticks(rotation=20, fontsize=10)
#plt.tight_layout()
#plt.show()

# Normalizovanje vrednosti 
scaler = MinMaxScaler()
df= pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

#KNN imputer
imputer = KNNImputer(n_neighbors=50)
df= pd.DataFrame(imputer.fit_transform(df),columns = df.columns)
#print(df_sklearn_encoded.iloc[37:45])
#print(df_sklearn_encoded.isna().sum())

#sns.displot(df['Hormonal_Contraceptives_in_Years'])
#sns.boxplot(df['Age_of_Respondents'])
index_list = []
for feature in df.columns:
 index_list.extend(outliers(df,feature))

df_new = remove(df,index_list)
print(df_new.shape)

x = df_new.drop(columns=['Is_Diagnosis_Cancer'])
y = df_new['Is_Diagnosis_Cancer']
#print(df.var())

#plot_correlation_for_col(df, col_name='Is_Diagnosis_Cancer') #najveca pozitivna korelacija sa Is_Diagnosis_HPV, Is_Screening_Biopsy... Negativna Is_STD_Condylomatisis
#Splitovanje na trening, test i validacioni skup
#Prvo train skup dobije 60%, temporary skup dobije 40%
x_train, x_temp, y_train, y_temp = train_test_split(x, y,train_size=0.6, shuffle=True, random_state=42)

# Preostalih 40% se podeli i dodeli se test i validacionom skupu
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, train_size=0.5, shuffle=True, random_state=42)

#Kreiranje OLS modela bez primene PCA
model = get_fitted_model(x_train,y_train)

# L.I.N.E pretpostavke

x_pom=sm.add_constant(x_train)
if linear_assumption(model,x_pom,y_train,p_value_thresh=0.05, plot=False):
    print('Zadovoljena je pretpostavka o linearnosti')
else: print('Nije zadovoljena pretpostavka o linearnosti')

autocorrelation, _ = independence_of_errors_assumption(model,x_pom,y_train, plot=False)
if autocorrelation is None:
    print('Zadovoljena je pretpostavka o nezavisnosti gresaka')
else: print('Nije zadovoljena pretpostavka o neazavisnosti gresaka')

variance, p = equal_variance_assumption(model,x_pom,y_train,p_value_thresh=0.05, plot=False)

if variance == 'equal':
    print('Zadovoljena je pretpostavka o jednakoj varijansi')
else: print('Nije zadovoljena pretpostavka o jednakoj varijansi')

if  perfect_collinearity_assumption(x_pom,plot=False):
    print('Nije zadovoljena pretpostavka o savrsenoj korelaciji')
else: print('Zadovoljena je pretpostavka o savrsenoj korelaciji')

normal, _ = normality_of_errors_assumption(model,x_pom,y_train,p_value_thresh=0.05, plot=False)
if normal == 'normal':
    print('Zadovoljena je pretpostavka o normalnosti gresaka')
else: print('Nije zadovoljena pretpostavka o normalnosti gresaka')

#Ovo sto nije zadovoljena pretpostavka o normalnosti gresaka ne predstavlja problem jer se radi sa velikim skupom podataka

zadovoljene = are_assumptions_satisfied(model, x_train, y_train)

if zadovoljene == True:
    print("Sve pretpostavke su zadovoljene")
else:  print("Nisu sve pretpostavke zadovoljene")


# RMSE
val_rmse = get_rmse(model, x_val, y_val)
print(f"RMSE za validacioni skup OLS modela je {val_rmse:.5f}")
test_rmse = get_rmse(model, x_test, y_test)
print(f"RMSE za test skup OLS modela je {test_rmse:.5f}")



# ADJ R^2
val_rq = get_rsquared_adj(model, x_val, y_val)
print(f"adj r^2 za validacioni skup je {val_rq:.5f}")
test_rq = get_rsquared_adj(model, x_test, y_test)
print(f"adj r^2 za test skup je {test_rq:.5f}")


#Primena PCA

pca_model = PCA(n_components=11, random_state=42)
principal_components_train = pca_model.fit_transform(x_train)
#plot_explained_variance(pca_model)
print(f'UKUPNA VARIJANSA: {sum(pca_model.explained_variance_ratio_) * 100:.1f}%')
#varijansa obuhvaćenih podataka je 95.5%
#plot_correlation_for_col(df_new, col_name='Is_Diagnosis_Cancer')
principal_components_val = pca_model.transform(x_val)
principal_components_test = pca_model.transform(x_test)


modelPCA = get_fitted_model(principal_components_train, y_train)

principal_components_val = pca_model.transform(x_val)
principal_components_test = pca_model.transform(x_test)

# ADJ R^2
val_rq_pca = get_rsquared_adj(modelPCA, principal_components_val, y_val)
print(f"adj r^2 za validacioni skup PCA modela je {val_rq_pca:.5f}")
test_rq_pca = get_rsquared_adj(modelPCA, principal_components_test, y_test)
print(f"adj r^2 za test skup PCA modela je {test_rq_pca:.5f}")

#RMSE 

val_rmse = get_rmse(modelPCA, principal_components_val, y_val)
print(f"RMSE za validacioni skup PCA modela je {val_rmse:.5f}")
test_rmse = get_rmse(modelPCA, principal_components_test, y_test)
print(f"RMSE za test skup PCA modela je {test_rmse:.5f}")

#print(modelPCA.summary())
#plot_pc_loading(pca_model, 10, x.columns) 

#target_col_train = df_new.loc[x_train.index, 'Is_Diagnosis_Cancer']
#visualize_principal_components(principal_components_train, n_principal_components=3, target_col=target_col_train)



#TEST na heteroscedasticity
names = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
test_result = sms.het_breuschpagan(model.resid, sm.add_constant(principal_components_train))
zip=lzip(names, test_result)
print(zip)
test_stat, p_value, _, _ = het_white(model.resid, x_pom)
print(f"White Test p-value: {p_value}")
# vrednosti oba testa su manje od <0.05 -> postoji heteroscedasticity-> WLS model


#Kreiranje RANSAC modela
model_pom = HuberRegressor(epsilon=1.35, max_iter=1000)
ransac = RANSACRegressor(estimator=model_pom, min_samples=0.8, residual_threshold=0.5, max_trials=1000)
ransac_model = ransac.fit(principal_components_train, y_train)
r2_train = ransac.score(principal_components_train, y_train)
print("RANSAC")
y_pred = ransac_model.predict(principal_components_test)
r2_test = r2_score(y_test, y_pred)

y_pred_val = ransac_model.predict(principal_components_val)
r2_val = r2_score(y_val, y_pred_val)

n_train = principal_components_train.shape[0]
n_test = principal_components_test.shape[0]
n_val = principal_components_val.shape[0]
p = principal_components_train.shape[1]

# Adjusted R² za svaki skup
adj_r2_train = 1 - ((1 - r2_train) * (n_train - 1)) / (n_train - p - 1)
adj_r2_test = 1 - ((1 - r2_test) * (n_test - 1)) / (n_test - p - 1)
adj_r2_val = 1 - ((1 - r2_val) * (n_val - 1)) / (n_val - p - 1)

# Prikaz rezultata
print(f"Adjusted R² za RANSAC model za validacioni skup: {adj_r2_val:.5f}")
print(f"Adjusted R² za RANSAC model za test skup: {adj_r2_test:.5f}")

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE za test skup RANSAC modela: {rmse_test:.5f}")
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"RMSE za validacioni skup RANSAC modela: {rmse_val:.5f}")

y_pred_ransac_train = ransac_model.predict(principal_components_train)
y_pred_ransac_test = ransac_model.predict(principal_components_test)
y_pred_ransac_val = ransac_model.predict(principal_components_val)


residuals_train = y_train - y_pred_ransac_train

# Kreiranje težina na osnovu residuala
weights_train = np.var(residuals_train)  

# Fit WLS model sa težinama
wls_model = sm.WLS(y_train, sm.add_constant(principal_components_train), weights=1/weights_train)
wls_results = wls_model.fit()


# Predikcije iz WLS modela
wls_predictions_train = wls_results.predict(sm.add_constant(principal_components_train))
wls_predictions_test = wls_results.predict(sm.add_constant(principal_components_test))
wls_predictions_val = wls_results.predict(sm.add_constant(principal_components_val))

# Izračunavanje performansi
r2_train_wls = r2_score(y_train, wls_predictions_train)
r2_test_wls = r2_score(y_test, wls_predictions_test)
r2_val_wls = r2_score(y_val, wls_predictions_val)

# Adjusted R² za WLS
n_train = principal_components_train.shape[0]
n_test = principal_components_test.shape[0]
n_val = principal_components_val.shape[0]
p = principal_components_train.shape[1]

adj_r2_train_wls = 1 - ((1 - r2_train_wls) * (n_train - 1)) / (n_train - p - 1)
adj_r2_test_wls = 1 - ((1 - r2_test_wls) * (n_test - 1)) / (n_test - p - 1)
adj_r2_val_wls = 1 - ((1 - r2_val_wls) * (n_val - 1)) / (n_val - p - 1)

# RMSE za WLS
rmse_train_wls = np.sqrt(mean_squared_error(y_train, wls_predictions_train))
rmse_test_wls = np.sqrt(mean_squared_error(y_test, wls_predictions_test))
rmse_val_wls = np.sqrt(mean_squared_error(y_val, wls_predictions_val))


print("\nWLS Model:")
print(f"Adjusted R² za test skup WLS modela: {adj_r2_test_wls:.5f}")
print(f"Adjusted R² za validacioni skup WLS modela: {adj_r2_val_wls:.5f}")
print(f"RMSE za validacioni skup{rmse_val_wls:.5f}")
print(f"RMSE za test skup{rmse_test_wls:.5f}")