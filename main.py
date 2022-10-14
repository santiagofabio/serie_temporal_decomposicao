from traceback import print_tb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import statsmodels.tsa.stattools
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sympy import decompose 
rcParams['figure.figsize'] =[15,6]

concentracao = sm.datasets.co2.load_pandas().data


serie_temporal_concetracao =pd.Series(concentracao['co2'].values, index=concentracao.index )

#decomposicao = seasonal_decompose(serie_temporal_concetracao)
print(f'Numero de dados faltantes: {serie_temporal_concetracao.isnull().sum()}')
concentracao.dropna(inplace =True)

concentracao.plot()
plt.xlabel('Anos')
plt.ylabel('Concentracao de CO2')
plt.title('Concentração de CO2 no  tempo')
plt.legend('Cconcentracao')
plt.savefig('serie_temporal.tiff', format='tiff', dpi =200)
plt.show()


decomposicao =seasonal_decompose(concentracao,period =7, model ='additive')
decomposicao.plot()
plt.savefig('decomposicao_aditiva.tiff', format='tiff', dpi =200)
plt.show()

plt.subplot(411)
plt.plot(concentracao, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(decomposicao.trend, label='Tendência')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(decomposicao.seasonal,label='Sazonalidade')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(decomposicao.resid, label='Resíduos')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


decomposicao_mult =seasonal_decompose(concentracao,period =7, model ='multiplicative')
decomposicao_mult.plot()
plt.savefig('decomposicao_multiplicativa.tiff', format='tiff', dpi =200)
plt.show()